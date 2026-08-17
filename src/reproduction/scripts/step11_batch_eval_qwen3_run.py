#!/usr/bin/env python3
"""Strict launcher for the staged Step-11 runner and its resume records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import step11_batch_eval_qwen3 as runner


def combined_script_record() -> dict[str, Any]:
    wrapper = Path(__file__).resolve()
    implementation = Path(runner.__file__).resolve()
    return {
        "wrapper": runner.record(wrapper),
        "implementation": runner.record(implementation),
    }


def validate_state_strict(state: dict[str, Any], static: dict[str, Any]) -> None:
    runner.require(state.get("schema_version") == "mmsearch.step11.state.v1", "state schema mismatch")
    runner.require(state.get("protocol") == static["protocol"], "state protocol mismatch")
    runner.require(state.get("input_manifest") == static["input_manifest"], "state input mismatch")
    runner.require(state.get("runner") == static["runner"], "state runner mismatch")
    predictions = state.get("predictions")
    completed = state.get("completed_count")
    runner.require(isinstance(predictions, list) and completed == len(predictions), "state prediction count mismatch")
    for expected_index, item in enumerate(predictions, start=1):
        runner.require(item.get("eval_index") == expected_index, "state eval index mismatch")
        path = Path(item.get("path", ""))
        actual = runner.record(path)
        claimed = {key: item.get(key) for key in ("path", "bytes", "sha256")}
        runner.require(actual == claimed, f"state prediction hash mismatch: {expected_index}")
    stages = state.get("stages")
    runner.require(isinstance(stages, dict), "state stages missing")
    for key, item in stages.items():
        runner.require(int(key) in runner.STAGES, f"unknown committed stage: {key}")
        runner.require(runner.record(Path(item["path"])) == item, f"state stage hash mismatch: {key}")
    runner.require(state.get("credentials_recorded") is False, "state credential flag mismatch")


def self_test() -> None:
    runner.self_test()
    combined = combined_script_record()
    runner.require(set(combined) == {"wrapper", "implementation"}, "combined script record test failed")
    runner.require(combined["wrapper"]["sha256"] != combined["implementation"]["sha256"], "script records unexpectedly collide")
    print(json.dumps({"status": "passed", "launcher_self_tests": 2}, sort_keys=True))


def main() -> int:
    runner.script_record = combined_script_record
    runner.validate_state = validate_state_strict
    if __import__("sys").argv[1:] == ["--self-test"]:
        self_test()
        return 0
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
