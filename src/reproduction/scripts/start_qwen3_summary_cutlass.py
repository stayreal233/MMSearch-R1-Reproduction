#!/usr/bin/env python3
"""Launch the pinned Qwen3 service with the audited SM120 CUTLASS fallback."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import start_qwen3_summary as base


LINEAR_BACKEND = "cutlass"
DEEP_GEMM_ENABLED = False
_real_sanitized_environment = base.sanitized_environment
_real_popen = subprocess.Popen
_launch_injected = False


def cutlass_environment() -> dict[str, str]:
    environment = _real_sanitized_environment()
    environment["VLLM_USE_DEEP_GEMM"] = "0"
    return environment


def cutlass_popen(command: Any, *args: Any, **kwargs: Any) -> Any:
    global _launch_injected
    if (
        isinstance(command, (list, tuple))
        and len(command) >= 2
        and str(command[0]) == str(base.VLLM)
        and command[1] == "serve"
    ):
        if "--linear-backend" in command:
            raise RuntimeError("Refusing duplicate --linear-backend arguments")
        environment = kwargs.get("env")
        if not isinstance(environment, dict) or environment.get("VLLM_USE_DEEP_GEMM") != "0":
            raise RuntimeError("CUTLASS launch requires VLLM_USE_DEEP_GEMM=0")
        command = list(command)
        insertion_index = command.index("--default-chat-template-kwargs")
        command[insertion_index:insertion_index] = ["--linear-backend", LINEAR_BACKEND]
        base.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        with base.LOG_PATH.open("ab", buffering=0) as log_handle:
            log_handle.write(
                (
                    f"\n[{timestamp}] audited SM120 recovery contract; "
                    "VLLM_USE_DEEP_GEMM=0 linear_backend=cutlass\n"
                ).encode("utf-8")
            )
        _launch_injected = True
    return _real_popen(command, *args, **kwargs)


def main() -> None:
    base.sanitized_environment = cutlass_environment
    subprocess.Popen = cutlass_popen
    base.main()
    if not _launch_injected:
        raise RuntimeError("CUTLASS launch command was not injected")
    print(
        json.dumps(
            {
                "recovery_contract": {
                    "linear_backend": LINEAR_BACKEND,
                    "deep_gemm": DEEP_GEMM_ENABLED,
                    "environment": {"VLLM_USE_DEEP_GEMM": "0"},
                }
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(
            json.dumps(
                {"status": "failed_validation", "returncode": exc.returncode},
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
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
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
