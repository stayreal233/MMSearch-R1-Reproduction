#!/usr/bin/env python3
"""Scan selected step-9 artifacts for the live Serper credential without printing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHUNK_BYTES = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def collect_files(targets: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for raw_target in targets:
        target = raw_target.resolve(strict=True)
        require(not target.is_symlink(), f"Refusing symlink target: {target}")
        if target.is_file():
            files.add(target)
            continue
        require(target.is_dir(), f"Unsupported scan target: {target}")
        for path in target.rglob("*"):
            require(not path.is_symlink(), f"Refusing symlink inside target: {path}")
            if path.is_file():
                files.add(path.resolve(strict=True))
    require(bool(files), "No files were selected for credential scanning")
    return sorted(files, key=str)


def scan_file(path: Path, needle: bytes) -> dict[str, Any]:
    digest = hashlib.sha256()
    matched = False
    overlap = b""
    total = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
            if needle in overlap + chunk:
                matched = True
            keep = max(0, len(needle) - 1)
            overlap = chunk[-keep:] if keep else b""
    return {
        "path": str(path),
        "bytes": total,
        "sha256": digest.hexdigest(),
        "credential_match": matched,
    }


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
    args = parse_args()
    secret = os.environ.get("SERPER_API_KEY")
    require(isinstance(secret, str) and bool(secret), "SERPER_API_KEY is not set")
    require("\n" not in secret and "\r" not in secret, "SERPER_API_KEY is malformed")
    encoded_secret = secret.encode("utf-8")
    records = [scan_file(path, encoded_secret) for path in collect_files(args.target)]
    matches = [record["path"] for record in records if record["credential_match"]]
    payload = {
        "schema_version": 1,
        "scanned_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "selected_qwen3_step9_outputs_caches_logs_and_manifests",
        "secret_source": "SERPER_API_KEY environment variable",
        "secret_value_recorded": False,
        "files_scanned": len(records),
        "total_bytes_scanned": sum(record["bytes"] for record in records),
        "exact_credential_matches": len(matches),
        "matching_files": matches,
        "files": records,
        "pass": not matches,
    }
    atomic_write_json(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "status": "passed" if not matches else "failed",
                "files_scanned": len(records),
                "exact_credential_matches": len(matches),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if matches:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=__import__("sys").stderr,
        )
        raise SystemExit(2) from exc
