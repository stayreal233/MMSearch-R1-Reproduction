#!/usr/bin/env python3
"""Approved v2 launcher: isolate ordinary sample tool failures and continue."""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path
from typing import Any

import step11_batch_eval_qwen3 as runner


PROTOCOL_V2 = Path(
    "/root/autodl-tmp/multimodal-search-r1/reproduction/env/step11_eval_protocol_v2.json"
)
PROTOCOL_V2_SHA256 = "d8d74572a8b369b37f44a66bc314d0e3c2e6bfef081122326a34e287ad92ff17"
PROTOCOL_V1_SHA256 = "f2fc533b824c65d5102fc10dbaebe0c3069242b00f9178c1417f6f0935c6000e"
INPUT_MANIFEST_SHA256 = "dbc28df74f3a1a0b87fd435255fda8ed73455dfe3a3d465dce9539fad37564ab"
V1_PREDICTION = Path(
    "/root/autodl-tmp/outputs/step11_eval_v1/predictions/001_fvqa_train_4724.json"
)
V1_PREDICTION_SHA256 = "534863c9b57e9fd0f9f966bef1cbc80fd9e53551fa8c822773934eea6b1a359e"
OUTPUT_V2 = Path("/root/autodl-tmp/outputs/step11_eval_v2")


# Switch all implementation globals before any runner function is called.
runner.PROTOCOL = PROTOCOL_V2
runner.PROTOCOL_SHA256 = PROTOCOL_V2_SHA256
runner.OUTPUT_ROOT = OUTPUT_V2
runner.STATE_PATH = OUTPUT_V2 / "state.json"
runner.PREDICTION_DIR = OUTPUT_V2 / "predictions"

_strict_require = runner.require
_original_execute_case = runner.base.execute_case


def combined_script_record() -> dict[str, Any]:
    wrapper = Path(__file__).resolve()
    implementation = Path(runner.__file__).resolve()
    return {
        "wrapper_v2": runner.record(wrapper),
        "implementation": runner.record(implementation),
    }


def validate_static_inputs_v2() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    protocol, protocol_bytes = runner.load_json(PROTOCOL_V2, 512 * 1024)
    _strict_require(runner.base.sha256_bytes(protocol_bytes) == PROTOCOL_V2_SHA256, "v2 protocol SHA mismatch")
    _strict_require(protocol.get("schema_version") == 2 and protocol.get("status") == "registered_before_step11_v2_resume", "v2 protocol identity mismatch")
    _strict_require(protocol.get("supersedes", {}).get("protocol_v1", {}).get("sha256") == PROTOCOL_V1_SHA256, "v1 protocol binding mismatch")
    _strict_require(protocol.get("supersedes", {}).get("v1_failure", {}).get("prediction", {}).get("sha256") == V1_PREDICTION_SHA256, "v1 failure binding mismatch")
    _strict_require(protocol.get("resume", {}).get("v1_example_1_is_imported_without_reexecution") is True, "v1 import rule missing")

    manifest, manifest_bytes = runner.load_json(runner.INPUT_MANIFEST, 8 * 1024 * 1024)
    _strict_require(runner.base.sha256_bytes(manifest_bytes) == INPUT_MANIFEST_SHA256, "input manifest SHA mismatch")
    _strict_require(manifest.get("schema_version") == "mmsearch.step11.eval-inputs.v1" and manifest.get("status") == "passed", "input manifest identity mismatch")
    _strict_require(manifest.get("protocol", {}).get("sha256") == PROTOCOL_V1_SHA256, "input manifest v1 selection protocol mismatch")
    _strict_require(manifest.get("credentials_recorded") is False, "input credential flag mismatch")
    examples = manifest.get("examples")
    _strict_require(isinstance(examples, list) and len(examples) == 50, "input example count mismatch")
    _strict_require([item.get("eval_index") for item in examples] == list(range(1, 51)), "input eval order mismatch")
    _strict_require(len({item.get("data_id") for item in examples}) == 50, "input IDs are not unique")
    for stage, expected in runner.EXPECTED_PREFIX_COUNTS.items():
        actual = dict(collections.Counter(item.get("category") for item in examples[:stage]))
        _strict_require(actual == expected, f"stage {stage} category balance mismatch")
    static = {
        "protocol": runner.record(PROTOCOL_V2, protocol_bytes),
        "input_manifest": runner.record(runner.INPUT_MANIFEST, manifest_bytes),
        "runner": combined_script_record(),
    }
    return manifest, examples, static


def validate_state_v2(state: dict[str, Any], static: dict[str, Any]) -> None:
    _strict_require(state.get("schema_version") == "mmsearch.step11.state.v1", "state schema mismatch")
    _strict_require(state.get("protocol") == static["protocol"], "state protocol mismatch")
    _strict_require(state.get("input_manifest") == static["input_manifest"], "state input mismatch")
    _strict_require(state.get("runner") == static["runner"], "state runner mismatch")
    predictions = state.get("predictions")
    completed = state.get("completed_count")
    _strict_require(isinstance(predictions, list) and completed == len(predictions), "state prediction count mismatch")
    for expected_index, item in enumerate(predictions, start=1):
        _strict_require(item.get("eval_index") == expected_index, "state eval index mismatch")
        actual = runner.record(Path(item.get("path", "")))
        claimed = {key: item.get(key) for key in ("path", "bytes", "sha256")}
        _strict_require(actual == claimed, f"state prediction hash mismatch: {expected_index}")
    stages = state.get("stages")
    _strict_require(isinstance(stages, dict), "state stages missing")
    for key, item in stages.items():
        _strict_require(int(key) in runner.STAGES, f"unknown committed stage: {key}")
        _strict_require(runner.record(Path(item["path"])) == item, f"state stage hash mismatch: {key}")
    _strict_require(state.get("credentials_recorded") is False, "state credential flag mismatch")


def import_v1_prediction(static: dict[str, Any]) -> dict[str, Any]:
    encoded = runner.read_bytes(V1_PREDICTION, 32 * 1024 * 1024)
    _strict_require(runner.base.sha256_bytes(encoded) == V1_PREDICTION_SHA256, "v1 prediction SHA mismatch")
    value = json.loads(encoded)
    _strict_require(isinstance(value, dict), "v1 prediction root invalid")
    trace = value.get("trace")
    _strict_require(
        value.get("eval_index") == 1
        and isinstance(trace, dict)
        and trace.get("data_id") == "fvqa_train_4724"
        and trace.get("terminal_status") == "text_search_hard_failure"
        and trace.get("tool_infrastructure_success") is False,
        "v1 prediction provenance mismatch",
    )
    imported = dict(value)
    imported["v2_import"] = {
        "source": runner.record(V1_PREDICTION, encoded),
        "reexecuted": False,
        "reason": "user-approved sample-level failure isolation; no network retry",
    }
    imported["evaluation_semantics"] = "natural_model_policy_sample_tool_failure"
    output = runner.PREDICTION_DIR / "001_fvqa_train_4724.json"
    runner.base.atomic_write_json(output, imported)
    output_record = runner.record(output)
    return {
        "path": output_record["path"],
        "bytes": output_record["bytes"],
        "sha256": output_record["sha256"],
        "eval_index": 1,
    }


def prepare_output_v2(target: int, static: dict[str, Any]) -> dict[str, Any]:
    outputs_root = runner.OUTPUT_ROOT.parent.resolve(strict=True)
    _strict_require(runner.OUTPUT_ROOT.parent == outputs_root, "v2 output parent mismatch")
    _strict_require(not runner.OUTPUT_ROOT.is_symlink(), "v2 output is a symlink")
    if not runner.OUTPUT_ROOT.exists():
        _strict_require(target == 5, "first v2 target must be 5")
        runner.OUTPUT_ROOT.mkdir(mode=0o700, exist_ok=False)
        runner.PREDICTION_DIR.mkdir(mode=0o700, exist_ok=False)
        state = runner.initial_state(static)
        imported_record = import_v1_prediction(static)
        state["predictions"] = [imported_record]
        state["completed_count"] = 1
        state["imported_v1_failure"] = {
            "source": runner.record(V1_PREDICTION),
            "reexecuted": False,
        }
        runner.base.atomic_write_json(runner.STATE_PATH, state)
        return state
    _strict_require(runner.OUTPUT_ROOT.is_dir() and runner.PREDICTION_DIR.is_dir(), "v2 output structure invalid")
    _strict_require(not (runner.OUTPUT_ROOT / "step11_run_failure.json").exists(), "prior v2 global hard failure requires user direction")
    state, _ = runner.load_json(runner.STATE_PATH)
    validate_state_v2(state, static)
    completed = state["completed_count"]
    _strict_require(target > completed, "target must expand completed prefix")
    if target == 20:
        _strict_require("5" in state["stages"], "stage 5 commit missing")
    if target == 50:
        _strict_require("20" in state["stages"], "stage 20 commit missing")
    return state


def tolerant_require(condition: bool, message: str) -> None:
    if condition:
        return
    if re.fullmatch(r"example [1-9][0-9]* did not satisfy healthy answered contract", message):
        return
    _strict_require(False, message)


def execute_case_with_global_guards(**kwargs: Any) -> dict[str, Any]:
    result = _original_execute_case(**kwargs)
    trace = result.get("trace", {})
    for failure in trace.get("hard_failures", []):
        stage = str(failure.get("stage", ""))
        error = str(failure.get("error", ""))
        lowered = error.lower()
        if stage == "mmsearch_generation" or any(
            marker in lowered
            for marker in ("cuda", "out of memory", "device-side", "cublas", "cudnn")
        ):
            raise RuntimeError(f"global MMSearch/CUDA failure: {error}")
    for round_item in trace.get("rounds", []):
        tool = round_item.get("tool", {})
        if tool.get("type") != "serper_dev_plus_jina_reader_plus_qwen3_summary":
            continue
        search = tool.get("status", {}).get("search", {})
        http_status = search.get("http_status")
        error = str(search.get("error", ""))
        if http_status in {401, 403, 429} or re.search(
            r"(?i)(http\s*(?:401|403|429)|authenticat|unauthoriz|rate.?limit)",
            error,
        ):
            raise RuntimeError("global Serper authentication/rate-limit failure")
    return result


def component_failure_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for result in results:
        trace = result["trace"]
        for item in trace.get("hard_failures", []):
            counts[f"hard_stage:{item.get('stage')}"] += 1
        for round_item in trace.get("rounds", []):
            tool = round_item.get("tool", {})
            status = tool.get("status", {})
            if tool.get("type", "").startswith("fvqa"):
                counts["image_item_failures"] += len(status.get("failures", []))
            elif tool.get("type") == "serper_dev_plus_jina_reader_plus_qwen3_summary":
                counts["serper_component_failures"] += int(status.get("search", {}).get("success") is not True)
                counts["jina_page_failures"] += len(status.get("reader", {}).get("failures", []))
                counts["qwen_summary_failures"] += len(status.get("summary", {}).get("failures", []))
    return dict(sorted(counts.items()))


def commit_stage_v2(
    target: int,
    state: dict[str, Any],
    static: dict[str, Any],
    results: list[dict[str, Any]],
    final_health: dict[str, Any],
    final_snapshot: dict[str, Any],
) -> dict[str, Any]:
    metrics, failure_summary = runner.aggregate(results)
    _strict_require(metrics["evaluated"] == target, "v2 stage metric count mismatch")
    healthy = [
        result["trace"] for result in results
        if result["trace"]["terminal_status"] == "answered"
        and result["trace"]["tool_infrastructure_success"] is True
        and result["trace"]["network_and_cache"].get("count_complete") is True
    ]
    healthy_correct = sum(trace["exact_match"] is True for trace in healthy)
    metrics.update({
        "sample_failure_isolation_enabled": True,
        "sample_failure_count": target - len(healthy),
        "sample_failure_rate_percent": round((target - len(healthy)) / target * 100, 4),
        "healthy_answered_count": len(healthy),
        "healthy_answered_correct": healthy_correct,
        "healthy_answered_accuracy_percent": round(healthy_correct / len(healthy) * 100, 4) if healthy else None,
        "end_to_end_failures_counted_in_denominator": True,
        "component_failure_counts": component_failure_counts(results),
    })
    failure_summary.update({
        "sample_failure_isolation_enabled": True,
        "sample_tool_failure_count": metrics["sample_failure_count"],
        "component_failure_counts": metrics["component_failure_counts"],
    })
    predictions_path = runner.OUTPUT_ROOT / f"predictions_{target}.jsonl"
    metrics_path = runner.OUTPUT_ROOT / f"metrics_{target}.json"
    failure_path = runner.OUTPUT_ROOT / f"failure_summary_{target}.json"
    runner.write_jsonl(predictions_path, results)
    runner.base.atomic_write_json(metrics_path, metrics)
    runner.base.atomic_write_json(failure_path, failure_summary)
    stage_files = [predictions_path, metrics_path, failure_path]
    if target == 50:
        runner.write_jsonl(runner.OUTPUT_ROOT / "predictions.jsonl", results)
        runner.base.atomic_write_json(runner.OUTPUT_ROOT / "metrics.json", metrics)
        runner.base.atomic_write_json(runner.OUTPUT_ROOT / "failure_summary.json", failure_summary)
        stage_files.extend((runner.OUTPUT_ROOT / "predictions.jsonl", runner.OUTPUT_ROOT / "metrics.json", runner.OUTPUT_ROOT / "failure_summary.json"))
    prediction_paths = [Path(item["path"]) for item in state["predictions"]]
    scan = runner.base.scan_serialized_evidence(
        prediction_paths + [metrics_path, failure_path],
        [predictions_path],
    )
    _strict_require(scan["pass"] is True, "v2 stage evidence scan failed")
    manifest = {
        "schema_version": "mmsearch.step11.stage.v2",
        "status": "passed_with_sample_failure_isolation",
        "stage_target": target,
        "completed_at_utc": runner.base.utc_now(),
        "protocol": static["protocol"],
        "input_manifest": static["input_manifest"],
        "runner": static["runner"],
        "prediction_count": len(results),
        "prediction_records": state["predictions"],
        "artifacts": [runner.record(path) for path in stage_files],
        "metrics": metrics,
        "failure_summary_digest": runner.record(failure_path),
        "qwen3_health_after_stage": runner.base.sanitize_public(final_health),
        "gpu_snapshot_after_stage": final_snapshot,
        "serialized_evidence_scan": scan,
        "ordinary_sample_failures_allowed": True,
        "global_hard_failures": 0,
        "credentials_recorded": False,
    }
    stage_path = runner.OUTPUT_ROOT / f"stage_{target}_manifest.json"
    runner.base.atomic_write_json(stage_path, manifest)
    state["stages"][str(target)] = runner.record(stage_path)
    state["status"] = "passed_stage_50" if target == 50 else "in_progress"
    runner.base.atomic_write_json(runner.STATE_PATH, state)
    if target == 50:
        completion = {
            "schema_version": "mmsearch.step11.completion.v2",
            "status": "passed_with_sample_failure_isolation",
            "step": 11,
            "completed_at_utc": runner.base.utc_now(),
            "completed_examples": 50,
            "protocol": static["protocol"],
            "input_manifest": static["input_manifest"],
            "runner": static["runner"],
            "stages": state["stages"],
            "final_metrics": metrics,
            "final_failure_summary": runner.record(runner.OUTPUT_ROOT / "failure_summary.json"),
            "final_predictions": runner.record(runner.OUTPUT_ROOT / "predictions.jsonl"),
            "state": runner.record(runner.STATE_PATH),
            "qwen3_health_after": runner.base.sanitize_public(final_health),
            "gpu_snapshot_after": final_snapshot,
            "sample_failure_isolation_enabled": True,
            "natural_policy_only": True,
            "controlled_step10_B_C_excluded": True,
            "credentials_recorded": False,
        }
        runner.base.atomic_write_json(runner.OUTPUT_ROOT / "step11_completion_manifest.json", completion)
    return manifest


def self_test() -> None:
    runner.self_test()
    scripts = combined_script_record()
    _strict_require(set(scripts) == {"wrapper_v2", "implementation"}, "v2 script record self-test failed")
    _strict_require(tolerant_require(False, "example 1 did not satisfy healthy answered contract") is None, "tolerance self-test failed")
    try:
        tolerant_require(False, "Qwen3 PID changed")
    except RuntimeError:
        pass
    else:
        raise RuntimeError("global hard-stop self-test failed")
    print(json.dumps({"status": "passed", "v2_launcher_self_tests": 3}, sort_keys=True))


def main() -> int:
    runner.script_record = combined_script_record
    runner.validate_static_inputs = validate_static_inputs_v2
    runner.validate_state = validate_state_v2
    runner.prepare_output = prepare_output_v2
    runner.commit_stage = commit_stage_v2
    runner.require = tolerant_require
    runner.base.execute_case = execute_case_with_global_guards
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return 0
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
