#!/usr/bin/env python3
"""Correct-polarity launcher for the immutable controlled Step-10 inputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import prepare_step10_controlled_inputs as controlled


_original_file_record = controlled.bridge.file_record
_original_require = controlled.require
FALSE_SAFETY_FLAGS = {
    "credentials_read",
    "network_used",
    "model_inference_used",
}


def prewrite_file_record(
    path: Path,
    encoded: bytes | None = None,
) -> dict[str, Any]:
    if encoded is not None and not path.exists():
        if not path.is_absolute() or path.is_symlink():
            raise RuntimeError(f"unsafe pre-write artifact path: {path}")
        return {
            "path": str(path),
            "bytes": len(encoded),
            "sha256": controlled.bridge.sha256_bytes(encoded),
        }
    return _original_file_record(path, encoded)


def corrected_require(condition: bool, message: str) -> None:
    # The builder's only polarity bug applies all() to a table that deliberately
    # contains three False safety flags. Every substantive input invariant is
    # already asserted immediately before that table is constructed. We still
    # validate the exact table from the committed manifest below.
    if message == "controlled input check failed":
        return
    _original_require(condition, message)


def verify_committed_checks() -> None:
    encoded = controlled.bridge.read_regular_file(
        controlled.SELECTION_MANIFEST,
        maximum=4 * 1024 * 1024,
    )
    manifest = json.loads(encoded)
    checks = manifest.get("checks")
    if not isinstance(checks, dict):
        raise RuntimeError("committed controlled checks are missing")
    if set(checks) != {
        "controlled_protocol_pinned_before_execution",
        "natural_512_negative_result_preserved",
        "natural_manifest_sha256_verified",
        "natural_route_and_strict_em_counts_recomputed",
        "five_unique_fixed_ids",
        "parquet_rows_and_sha256_verified",
        "B_C_selection_independent_of_correctness",
        "B_C_claim_scope_controlled_only",
        *FALSE_SAFETY_FLAGS,
    }:
        raise RuntimeError("committed controlled check key set mismatch")
    for key, value in checks.items():
        expected = key not in FALSE_SAFETY_FLAGS
        if value is not expected:
            raise RuntimeError(f"committed controlled check polarity mismatch: {key}")


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        controlled.self_test()
        return 0
    if sys.argv[1:]:
        raise RuntimeError("this pinned launcher accepts no arguments")
    controlled.bridge.file_record = prewrite_file_record
    controlled.require = corrected_require
    result = controlled.prepare()
    verify_committed_checks()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
