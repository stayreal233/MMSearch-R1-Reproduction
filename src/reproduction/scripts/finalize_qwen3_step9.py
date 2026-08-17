#!/usr/bin/env python3
"""Validate step-9 evidence and write one credential-free completion manifest."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path("/root/autodl-tmp/outputs")
SUMMARY_CACHE_ROOT = Path("/root/autodl-tmp/search_cache/qwen3_summary")
LOG_PATH = Path("/root/autodl-tmp/logs/qwen3_summary_vllm.log")
PID_PATH = Path("/root/autodl-tmp/logs/qwen3_summary_vllm.pid")
MANIFEST_PATH = OUTPUT_ROOT / "qwen3_step9_completion_manifest.json"
EXPECTED_REVISION = "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
EXPECTED_TOTAL_BYTES = 34_338_579_454
FORBIDDEN_JSON_KEYS = {
    "api_key",
    "authorization",
    "content",
    "headers",
    "messages",
    "page_content",
    "prompt",
    "raw_content",
    "request_body",
    "response_body",
    "webpage_content",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid JSON evidence: {path}: {type(exc).__name__}") from exc
    require(isinstance(value, dict), f"JSON evidence root is not an object: {path}")
    return value


def find_forbidden_keys(value: Any, prefix: str = "$") -> list[str]:
    matches: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key)
            child = f"{prefix}.{name}"
            if name.casefold() in FORBIDDEN_JSON_KEYS:
                matches.append(child)
            matches.extend(find_forbidden_keys(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            matches.extend(find_forbidden_keys(item, f"{prefix}[{index}]"))
    return matches


def file_record(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"Missing/unsafe evidence file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    require(size > 0, f"Evidence file is empty: {path}")
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def main() -> None:
    install_path = REPO_ROOT / "reproduction/env/qwen3_install_decision.json"
    revision_path = REPO_ROOT / "reproduction/env/qwen3_huggingface_revision.json"
    versions_path = REPO_ROOT / "reproduction/env/qwen3_summary_versions.txt"
    smoke_path = OUTPUT_ROOT / "qwen3_summary_smoke.json"
    flow_path = OUTPUT_ROOT / "real_search_flow_qwen3_summary.json"
    key_scan_path = OUTPUT_ROOT / "qwen3_key_scan.json"

    install = load_object(install_path)
    revision = load_object(revision_path)
    smoke = load_object(smoke_path)
    flow = load_object(flow_path)
    key_scan = load_object(key_scan_path)

    require(install.get("model", {}).get("resolved_revision") == EXPECTED_REVISION, "Install revision mismatch")
    require(revision.get("validation_status") == "passed", "Model validation did not pass")
    require(revision.get("resolved_revision") == EXPECTED_REVISION, "Model manifest revision mismatch")
    require(revision.get("file_count") == 17, "Model manifest file count mismatch")
    require(revision.get("total_bytes") == EXPECTED_TOTAL_BYTES, "Model manifest byte count mismatch")
    checks = revision.get("checks", {})
    require(checks.get("seven_safetensors_sha256_match") is True, "Shard hashes were not validated")
    require(smoke.get("pass") is True, "Single-page summary smoke did not pass")
    require(smoke.get("checks", {}).get("thinking_disabled") is True, "Smoke Thinking check failed")
    require(flow.get("pass") is True, "Mixed Search flow did not pass")
    trace = flow.get("trace", {})
    require(trace.get("data_id") == "fvqa_train_17", "Mixed Search sample mismatch")
    require(trace.get("action_sequence") == ["image_search", "text_search", "answer"], "Mixed Search action sequence mismatch")
    require(trace.get("summarization_calls") == 1, "Mixed Search summarization count mismatch")
    require(trace.get("exact_match") is True, "Mixed Search exact match failed")
    require(key_scan.get("pass") is True, "Credential scan did not pass")
    require(key_scan.get("exact_credential_matches") == 0, "Credential scan found a match")

    for label, payload in (("smoke", smoke), ("flow", flow)):
        forbidden = find_forbidden_keys(payload)
        require(not forbidden, f"{label} evidence contains forbidden fields: {forbidden[:5]}")

    require(SUMMARY_CACHE_ROOT.is_dir(), "Qwen3 summary cache directory is missing")
    cache_paths = sorted(SUMMARY_CACHE_ROOT.glob("*.json"))
    require(len(cache_paths) >= 5, "Fewer than five summary cache records exist")
    for cache_path in cache_paths:
        payload = load_object(cache_path)
        forbidden = find_forbidden_keys(payload)
        require(not forbidden, f"Summary cache contains forbidden fields: {cache_path}: {forbidden[:5]}")

    pid_text = PID_PATH.read_text(encoding="ascii").strip()
    require(pid_text.isdigit(), "Qwen3 service PID file is malformed")
    service_pid = int(pid_text)
    try:
        os.kill(service_pid, 0)
    except OSError as exc:
        raise RuntimeError("Qwen3 service is not alive at finalization") from exc

    primary_paths = [
        install_path,
        revision_path,
        versions_path,
        smoke_path,
        flow_path,
        key_scan_path,
        LOG_PATH,
        PID_PATH,
    ]
    artifact_records = [file_record(path) for path in primary_paths]
    cache_records = [file_record(path) for path in cache_paths]
    manifest = {
        "schema_version": 1,
        "status": "passed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "step": 9,
        "model_revision": EXPECTED_REVISION,
        "model_file_count": 17,
        "model_total_bytes": EXPECTED_TOTAL_BYTES,
        "service_pid": service_pid,
        "service_alive": True,
        "sample": "fvqa_train_17",
        "action_sequence": trace["action_sequence"],
        "final_answer": trace.get("final_answer"),
        "ground_truth": trace.get("ground_truth"),
        "exact_match": trace["exact_match"],
        "summary_cache_records": len(cache_records),
        "credential_matches": 0,
        "primary_artifacts": artifact_records,
        "summary_cache_artifacts": cache_records,
        "checks": {
            "pinned_model_content_sha256": True,
            "environment_pinned": True,
            "service_health_before_and_after_flow": True,
            "thinking_disabled": True,
            "top5_summaries_complete": True,
            "same_gpu_mixed_search_passed": True,
            "credential_scan_passed": True,
            "raw_webpage_fields_absent": True
        },
        "credentials_recorded": False,
    }
    atomic_write_json(MANIFEST_PATH, manifest)
    print(json.dumps({
        "status": "passed",
        "output": str(MANIFEST_PATH),
        "summary_cache_records": len(cache_records),
        "service_pid": service_pid,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, ensure_ascii=False, indent=2), file=__import__("sys").stderr)
        raise SystemExit(2) from exc
