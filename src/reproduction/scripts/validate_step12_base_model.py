#!/usr/bin/env python3
"""Strict artifact validation for the pinned Step 12 Base model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path("/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
REVISION_METADATA = REPO_ROOT / "reproduction/env/step12_base_huggingface_revision.json"
PROTOCOL = REPO_ROOT / "reproduction/env/step12_base_comparison_protocol.json"
OUTPUT = Path("/root/autodl-tmp/outputs/step12_base_artifact_validation.json")
REVISION_METADATA_SHA256 = "fb8c62723161017c615f95e48e5cb9a21d3c841c9eac0dd12e5fb29414b42a48"
PROTOCOL_SHA256 = "3a80ee1fe4685cde68335a1ad336a3cf6f8f970a71f9664f6f66e22ee3d651f5"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, maximum: int = 16 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    require(path.is_file() and not path.is_symlink(), f"not a regular JSON file: {path}")
    data = path.read_bytes()
    require(len(data) <= maximum, f"JSON file too large: {path}")
    value = json.loads(data.decode("utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value, data


def record(path: Path, *, include_sha: bool = True) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    result = {
        "path": str(path.resolve(strict=True)),
        "bytes": path.stat().st_size,
    }
    if include_sha:
        result["sha256"] = sha256_file(path)
    return result


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate() -> dict[str, Any]:
    revision, revision_bytes = read_json(REVISION_METADATA)
    protocol, protocol_bytes = read_json(PROTOCOL)
    require(sha256_bytes(revision_bytes) == REVISION_METADATA_SHA256, "revision metadata SHA mismatch")
    require(sha256_bytes(protocol_bytes) == PROTOCOL_SHA256, "Step 12 protocol SHA mismatch")
    require(revision.get("revision") == MODEL_REVISION, "revision metadata commit mismatch")
    require(protocol.get("base_model", {}).get("revision") == MODEL_REVISION, "protocol commit mismatch")
    require(protocol.get("status") == "registered_before_step12_download_and_inference", "protocol registration status mismatch")
    expected_files = revision.get("files")
    require(isinstance(expected_files, list) and len(expected_files) == 16, "expected file metadata mismatch")
    require(MODEL_DIR.is_dir() and not MODEL_DIR.is_symlink(), "Base model directory missing or symlinked")
    incomplete = sorted(str(path) for path in MODEL_DIR.rglob("*.incomplete"))
    require(not incomplete, f"incomplete downloads remain: {len(incomplete)}")
    expected_names = {item["path"] for item in expected_files}
    actual_names = {path.name for path in MODEL_DIR.iterdir() if path.is_file()}
    require(actual_names == expected_names, "Base model top-level file set mismatch")
    require(not any(path.is_symlink() for path in MODEL_DIR.iterdir()), "Base model top-level symlink found")
    file_records = []
    total_bytes = 0
    shards = []
    for item in expected_files:
        require(set(item) == {"path", "bytes", "lfs_sha256"}, f"unexpected metadata fields for {item.get('path')}")
        path = MODEL_DIR / item["path"]
        require(path.is_file() and not path.is_symlink(), f"missing model file: {item['path']}")
        size = path.stat().st_size
        require(size == item["bytes"], f"byte count mismatch: {item['path']}")
        total_bytes += size
        current = record(path)
        if item["lfs_sha256"] is not None:
            require(current["sha256"] == item["lfs_sha256"], f"LFS SHA mismatch: {item['path']}")
            shards.append(current)
        file_records.append(current)
    require(total_bytes == revision["total_bytes"] == 16595981281, "total model bytes mismatch")
    require(len(shards) == 5, "expected five model shards")
    config, _ = read_json(MODEL_DIR / "config.json")
    require(config.get("model_type") == "qwen2_5_vl", "Base model_type mismatch")
    require(config.get("architectures") == ["Qwen2_5_VLForConditionalGeneration"], "Base architecture mismatch")
    require(config.get("torch_dtype") in ("bfloat16", "bf16"), "Base config dtype mismatch")
    index, _ = read_json(MODEL_DIR / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    require(isinstance(weight_map, dict) and weight_map, "weight map missing")
    shard_names = {item["path"] for item in expected_files if item["lfs_sha256"]}
    require(set(weight_map.values()) == shard_names, "weight index shard set mismatch")

    import torch
    import transformers

    require(torch.cuda.is_available(), "CUDA unavailable during Base validation")
    require(torch.cuda.get_device_capability(0) == (12, 0), "unexpected GPU capability")
    return {
        "schema_version": 1,
        "status": "passed",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": revision["repo_id"],
        "revision": MODEL_REVISION,
        "model_dir": str(MODEL_DIR),
        "file_count": len(file_records),
        "total_bytes": total_bytes,
        "file_records": file_records,
        "shard_count": len(shards),
        "all_lfs_shard_sha256_verified": True,
        "exact_top_level_file_set_verified": True,
        "no_incomplete_downloads": True,
        "weight_index_shard_set_verified": True,
        "config": {
            "model_type": config["model_type"],
            "architectures": config["architectures"],
            "torch_dtype": config["torch_dtype"],
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "device_name": torch.cuda.get_device_name(0),
        },
        "revision_metadata": record(REVISION_METADATA),
        "protocol": record(PROTOCOL),
        "credentials_recorded": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        require(sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA self-test failed")
        print(json.dumps({"status": "passed", "pure_self_tests": 1}))
        return
    result = validate()
    atomic_write_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "revision": result["revision"],
        "file_count": result["file_count"],
        "total_bytes": result["total_bytes"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
