#!/usr/bin/env python3
"""Independent, read-only consistency audit for the completed Step 11 run."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "reproduction" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import step11_batch_eval_qwen3 as implementation  # noqa: E402


OUTPUT_ROOT = Path("/root/autodl-tmp/outputs/step11_eval_v2")
INPUT_MANIFEST = Path("/root/autodl-tmp/mmsearch_step11_inputs/eval_manifest.json")
COMPLETION = OUTPUT_ROOT / "step11_completion_manifest.json"
AUDIT_OUTPUT = OUTPUT_ROOT / "step11_completion_audit.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_bytes(path: Path, maximum: int = 128 * 1024 * 1024) -> bytes:
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    data = path.read_bytes()
    require(len(data) <= maximum, f"file exceeds audit limit: {path}")
    return data


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(read_bytes(path).decode("utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    data = read_bytes(path)
    return {
        "path": str(path.resolve(strict=True)),
        "bytes": len(data),
        "sha256": digest(data),
    }


def verify_record(record: dict[str, Any], label: str) -> dict[str, Any]:
    require(isinstance(record, dict), f"{label} record is not an object")
    require(set(("path", "bytes", "sha256")).issubset(record), f"{label} record fields missing")
    current = file_record(Path(record["path"]))
    require(current["bytes"] == record["bytes"], f"{label} byte count mismatch")
    require(current["sha256"] == record["sha256"], f"{label} SHA-256 mismatch")
    return current


def strict_em(answer: Any, ground_truth: Any) -> bool:
    return (
        isinstance(answer, str)
        and isinstance(ground_truth, str)
        and answer.strip().lower() == ground_truth.strip().lower()
    )


def rounded_ratio(numerator: int | float, denominator: int | float) -> float:
    return round(numerator / denominator * 100, 4)


def audit() -> dict[str, Any]:
    completion = read_json(COMPLETION)
    require(completion.get("schema_version") == "mmsearch.step11.completion.v2", "completion schema mismatch")
    require(completion.get("status") == "passed_with_sample_failure_isolation", "completion status mismatch")
    require(completion.get("step") == 11 and completion.get("completed_examples") == 50, "completion count mismatch")
    require(completion.get("sample_failure_isolation_enabled") is True, "v2 isolation flag missing")
    require(completion.get("natural_policy_only") is True, "run is not natural-policy only")
    require(completion.get("controlled_step10_B_C_excluded") is True, "controlled Step 10 traces were not excluded")
    require(completion.get("credentials_recorded") is False, "completion claims credentials were recorded")

    verified_records: dict[str, Any] = {}
    for label, record in (
        ("protocol", completion["protocol"]),
        ("input_manifest", completion["input_manifest"]),
        ("runner_wrapper_v2", completion["runner"]["wrapper_v2"]),
        ("runner_implementation", completion["runner"]["implementation"]),
        ("final_failure_summary", completion["final_failure_summary"]),
        ("final_predictions", completion["final_predictions"]),
        ("state", completion["state"]),
    ):
        verified_records[label] = verify_record(record, label)
    for stage in (5, 20, 50):
        verified_records[f"stage_{stage}"] = verify_record(completion["stages"][str(stage)], f"stage {stage}")

    require(file_record(INPUT_MANIFEST)["sha256"] == completion["input_manifest"]["sha256"], "fixed input manifest changed")
    inputs = read_json(INPUT_MANIFEST)
    examples = inputs.get("examples")
    require(isinstance(examples, list) and len(examples) == 50, "fixed input count is not 50")
    require([item["eval_index"] for item in examples] == list(range(1, 51)), "input eval indices are not 1..50")
    require(len({item["data_id"] for item in examples}) == 50, "input data IDs are not unique")
    require(collections.Counter(item["category"] for item in examples) == {"search_free": 25, "search_required": 25}, "input category balance mismatch")

    stage_manifests: dict[str, dict[str, Any]] = {}
    for stage in (5, 20, 50):
        value = read_json(Path(completion["stages"][str(stage)]["path"]))
        require(value.get("schema_version") == "mmsearch.step11.stage.v2", f"stage {stage} schema mismatch")
        require(value.get("status") == "passed_with_sample_failure_isolation", f"stage {stage} status mismatch")
        require(value.get("stage_target") == stage and value.get("prediction_count") == stage, f"stage {stage} count mismatch")
        require(value.get("global_hard_failures") == 0, f"stage {stage} has a global hard failure")
        require(value.get("credentials_recorded") is False, f"stage {stage} credential flag mismatch")
        require(value.get("serialized_evidence_scan", {}).get("pass") is True, f"stage {stage} evidence scan failed")
        require(value.get("qwen3_health_after_stage", {}).get("success") is True, f"stage {stage} Qwen health failed")
        require(value.get("qwen3_health_after_stage", {}).get("thinking_enabled") is False, f"stage {stage} Thinking was enabled")
        stage_manifests[str(stage)] = value

    final_stage = stage_manifests["50"]
    prediction_records = final_stage.get("prediction_records")
    require(isinstance(prediction_records, list) and len(prediction_records) == 50, "stage 50 prediction record count mismatch")
    results: list[dict[str, Any]] = []
    prediction_files: list[Path] = []
    for expected_index, (entry, record) in enumerate(zip(examples, prediction_records, strict=True), start=1):
        require(record.get("eval_index") == expected_index, f"prediction record order mismatch at {expected_index}")
        current = verify_record(record, f"prediction {expected_index}")
        path = Path(current["path"])
        require(path.parent == OUTPUT_ROOT / "predictions", f"prediction {expected_index} path escaped output root")
        require(path.name.startswith(f"{expected_index:03d}_"), f"prediction {expected_index} filename mismatch")
        result = read_json(path)
        trace = result.get("trace")
        require(isinstance(trace, dict), f"prediction {expected_index} trace missing")
        require(result.get("eval_index") == expected_index, f"prediction {expected_index} embedded index mismatch")
        require(trace.get("data_id") == entry["data_id"], f"prediction {expected_index} data ID mismatch")
        require(trace.get("category") == entry["category"], f"prediction {expected_index} category mismatch")
        require(trace.get("source_row_index") == entry["source_row_index"], f"prediction {expected_index} source row mismatch")
        require(result.get("selection", {}).get("selection_rank_sha256") == entry["selection_rank_sha256"], f"prediction {expected_index} rank digest mismatch")
        recomputed_em = strict_em(trace.get("final_answer"), trace.get("ground_truth"))
        require(trace.get("exact_match") is recomputed_em, f"prediction {expected_index} strict EM mismatch")
        actions = trace.get("action_sequence")
        require(isinstance(actions, list) and all(isinstance(item, str) for item in actions), f"prediction {expected_index} action sequence invalid")
        require(trace.get("image_search_calls") == actions.count("image_search"), f"prediction {expected_index} image call mismatch")
        require(trace.get("text_search_calls") == actions.count("text_search"), f"prediction {expected_index} text call mismatch")
        require(trace.get("total_turns") == len(trace.get("rounds", [])), f"prediction {expected_index} turn count mismatch")
        require(trace.get("route_origin") == "natural_checkpoint_policy", f"prediction {expected_index} is not natural policy")
        require(trace.get("controller_intervention") is False, f"prediction {expected_index} has controller intervention")
        results.append(result)
        prediction_files.append(path)

    traces = [item["trace"] for item in results]
    correct = sum(trace["exact_match"] is True for trace in traces)
    image_calls = sum(trace["image_search_calls"] for trace in traces)
    text_calls = sum(trace["text_search_calls"] for trace in traces)
    healthy = [
        trace for trace in traces
        if trace.get("terminal_status") == "answered"
        and trace.get("tool_infrastructure_success") is True
        and trace.get("network_and_cache", {}).get("count_complete") is True
    ]
    recomputed = {
        "evaluated": 50,
        "correct": correct,
        "accuracy_percent": rounded_ratio(correct, 50),
        "image_search_calls": image_calls,
        "text_search_calls": text_calls,
        "total_search_calls": image_calls + text_calls,
        "search_ratio_percent": rounded_ratio(image_calls + text_calls, 100),
        "average_turns": round(sum(trace["total_turns"] for trace in traces) / 50, 4),
        "route_counts": dict(sorted(collections.Counter(" -> ".join(trace["action_sequence"]) if trace["action_sequence"] else "(none)" for trace in traces).items())),
        "sample_failure_count": 50 - len(healthy),
        "sample_failure_rate_percent": rounded_ratio(50 - len(healthy), 50),
        "healthy_answered_count": len(healthy),
        "healthy_answered_correct": sum(trace["exact_match"] is True for trace in healthy),
    }
    recomputed["healthy_answered_accuracy_percent"] = rounded_ratio(recomputed["healthy_answered_correct"], len(healthy)) if healthy else None
    metrics = read_json(OUTPUT_ROOT / "metrics.json")
    for key, value in recomputed.items():
        require(metrics.get(key) == value, f"final metric mismatch: {key}")
    require(metrics == completion["final_metrics"], "completion embedded metrics differ from metrics.json")
    require(metrics.get("strict_exact_match_definition") == "prediction.strip().lower() == ground_truth.strip().lower()", "strict EM definition changed")
    require(metrics.get("natural_policy_only") is True and metrics.get("controlled_step10_B_C_excluded") is True, "metric policy scope mismatch")

    for category in ("search_free", "search_required"):
        selected = [trace for trace in traces if trace["category"] == category]
        category_correct = sum(trace["exact_match"] is True for trace in selected)
        category_searches = sum(trace["image_search_calls"] + trace["text_search_calls"] for trace in selected)
        expected = {
            "count": len(selected),
            "correct": category_correct,
            "accuracy_percent": rounded_ratio(category_correct, len(selected)),
            "total_search_calls": category_searches,
            "search_ratio_percent": rounded_ratio(category_searches, len(selected) * 2),
            "average_turns": round(sum(trace["total_turns"] for trace in selected) / len(selected), 4),
        }
        require(metrics["categories"][category] == expected, f"category metrics mismatch: {category}")

    final_jsonl = read_bytes(OUTPUT_ROOT / "predictions.jsonl")
    expected_jsonl = ("\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for item in results) + "\n").encode("utf-8")
    require(final_jsonl == expected_jsonl, "predictions.jsonl differs from per-sample JSON records")
    require(read_bytes(OUTPUT_ROOT / "predictions_50.jsonl") == final_jsonl, "stage-50 predictions differ from final predictions")
    require(read_bytes(OUTPUT_ROOT / "metrics_50.json") == read_bytes(OUTPUT_ROOT / "metrics.json"), "stage-50 metrics differ from final metrics")
    require(read_bytes(OUTPUT_ROOT / "failure_summary_50.json") == read_bytes(OUTPUT_ROOT / "failure_summary.json"), "stage-50 failure summary differs from final failure summary")

    failure_summary = read_json(OUTPUT_ROOT / "failure_summary.json")
    require(failure_summary.get("evaluated") == 50, "failure summary evaluated count mismatch")
    require(failure_summary.get("failure_count") == 50 - correct, "failure summary failure count mismatch")
    require(sum(failure_summary.get("failure_layer_counts", {}).values()) == 50 - correct, "failure layer totals mismatch")
    require(failure_summary.get("sample_tool_failure_count") == 50 - len(healthy), "sample tool failure total mismatch")
    require(failure_summary.get("credentials_recorded") is False, "failure summary credential flag mismatch")

    state = read_json(OUTPUT_ROOT / "state.json")
    require(state.get("status") == "passed_stage_50", "checkpoint state is not at stage 50")
    require(state.get("completed_count") == 50, "checkpoint completed count mismatch")
    require(len(state.get("predictions", [])) == 50, "checkpoint prediction record count mismatch")

    json_paths = prediction_files + [
        OUTPUT_ROOT / "metrics.json",
        OUTPUT_ROOT / "failure_summary.json",
        OUTPUT_ROOT / "state.json",
        OUTPUT_ROOT / "stage_5_manifest.json",
        OUTPUT_ROOT / "stage_20_manifest.json",
        OUTPUT_ROOT / "stage_50_manifest.json",
        COMPLETION,
    ]
    text_paths = [
        OUTPUT_ROOT / "predictions_5.jsonl",
        OUTPUT_ROOT / "predictions_20.jsonl",
        OUTPUT_ROOT / "predictions_50.jsonl",
        OUTPUT_ROOT / "predictions.jsonl",
    ]
    evidence_scan = implementation.base.scan_serialized_evidence(json_paths, text_paths)
    require(evidence_scan.get("pass") is True, "independent serialized-evidence scan failed")

    v1_imports = sum(item.get("selection", {}).get("execution_mode") == "imported_v1_failure_without_reexecution" for item in results)
    require(v1_imports == 1, "expected exactly one imported v1 failure")

    return {
        "schema_version": "mmsearch.step11.completion-audit.v1",
        "status": "passed",
        "step": 11,
        "audit_scope": "independent structural, hash, metric, policy-scope, and serialized-evidence audit",
        "completion_manifest": file_record(COMPLETION),
        "verified_records": verified_records,
        "prediction_files_verified": 50,
        "fixed_input_order_verified": True,
        "balanced_categories_verified": {"search_free": 25, "search_required": 25},
        "strict_em_recomputed_for_all": True,
        "search_calls_and_routes_recomputed_for_all": True,
        "stage_prefixes_verified": [5, 20, 50],
        "v1_failure_imported_without_reexecution_count": v1_imports,
        "natural_policy_only": True,
        "controlled_step10_B_C_excluded": True,
        "recomputed_metrics": recomputed,
        "failure_layer_total": sum(failure_summary["failure_layer_counts"].values()),
        "sample_tool_failure_count": failure_summary["sample_tool_failure_count"],
        "global_hard_failures": 0,
        "qwen3_health_at_stage_completion": completion["qwen3_health_after"],
        "serialized_evidence_scan": evidence_scan,
        "credentials_recorded": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=AUDIT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        require(strict_em(" Answer ", "answer"), "strict EM positive self-test failed")
        require(not strict_em("two  spaces", "two spaces"), "strict EM whitespace self-test failed")
        print(json.dumps({"status": "passed", "pure_self_tests": 2}))
        return
    require(os.environ.get("SERPER_API_KEY"), "SERPER_API_KEY is required for exact credential-value scan")
    result = audit()
    implementation.base.atomic_write_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "prediction_files_verified": result["prediction_files_verified"],
        "accuracy_percent": result["recomputed_metrics"]["accuracy_percent"],
        "search_ratio_percent": result["recomputed_metrics"]["search_ratio_percent"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
