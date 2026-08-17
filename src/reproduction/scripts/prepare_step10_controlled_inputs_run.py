#!/usr/bin/env python3
"""Run the controlled input builder with pre-write artifact records enabled.

The shared bridge helper resolves paths strictly even when bytes are supplied;
controlled artifacts are intentionally recorded before their atomic write. This
wrapper narrows that behavior only for supplied bytes and non-existing targets.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import prepare_step10_controlled_inputs as controlled


_original_file_record = controlled.bridge.file_record


def prewrite_file_record(
    path: Path,
    encoded: bytes | None = None,
) -> dict[str, Any]:
    if encoded is not None and not path.exists():
        require_absolute = path.is_absolute()
        if not require_absolute or path.is_symlink():
            raise RuntimeError(f"unsafe pre-write artifact path: {path}")
        return {
            "path": str(path),
            "bytes": len(encoded),
            "sha256": controlled.bridge.sha256_bytes(encoded),
        }
    return _original_file_record(path, encoded)


def main() -> int:
    controlled.bridge.file_record = prewrite_file_record
    return controlled.main()


if __name__ == "__main__":
    raise SystemExit(main())
