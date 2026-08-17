#!/usr/bin/env python3
"""Independent completion audit for Step 12 Base comparison artifacts."""

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
SCRIPTS_DIR = REPO_ROOT / "reproduction/scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import step10_case_suite_qwen3 as evidence  # noqa: E402


PROTOCOL = REPO_ROOT / "reproduction/env/step12_base_comparison_protocol.json"
PROTOCOL_SHA256 = "3a80ee1fe4685cde68335a1ad336a3cf6f8f970a71f9664f6f66e22ee3d651f5"
INPUT_MANIFEST = Path("/root/autodl-tmp/mmsearch_step11_inputs/eval_manifest.json")
INPUT_MANIFEST_SHA256 = "dbc28df74f3a1a0b87fd435255fda8ed73455dfe3a3d465dce9539fad37564ab"
BASE_ROOT = Path("/root/autodl-tmp/outputs/step12_base_direct_v1")
BASE_COMPLETION = BASE_ROOT / "step12_base_completion_manifest.json"
COMPARISON_ROOT = Path("/root/autodl-tmp/outputs/step12_comparison_v1")
COMPARISON_COMPLETION = COMPARISON_ROOT / "step12_completion_manifest.json"
OUTPUT = COMPARISON_ROOT / "step12_completion_audit.json"
MM_PREDICTIONS = Path("/root/autodl-tmp/outputs/step11_eval_v2/predictions.jsonl")
MM_PREDICTIONS_SHA256 = "beda0f6a02f750b89a5a53cb9f39dcaac56dbe047e2fb1f4ecc01290d6ac48ff"
ANSWER_REPLACEMENT_DEFINITION = "prediction.strip().lower() == ground_truth.strip().lower()"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path, maximum: int = 32 * 1024 * 1024) -> dict[str, Any]:
    return evidence.load_json_object(path, max_bytes=maximum)[0]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    data = evidence.read_regular_file(path, max_bytes=128 * 1024 * 1024)
    values = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    require(all(isinstance(item, dict) for item in values), f"invalid JSONL: {path}")
    return values


def verify_record(record: dict[str, Any], label: str) -> dict[str, Any]:
    current = evidence.file_record(Path(record["path"]))
    require(current["bytes"] == record["bytes"], f"{label} bytes mismatch")
    require(current["sha256"] == record["sha256"], f"{label} SHA mismatch")
    return current


def strict_em(answer: Any, ground_truth: Any) -> bool:
    return isinstance(answer, str) and isinstance(ground_truth, str) and answer.strip().lower() == ground_truth.strip().lower()


def paired_label(base_correct: bool, mm_correct: bool) -> str:
    if base_correct and mm_correct:
        return "both_correct"
    if base_correct:
        return "base_only_correct"
    if mm_correct:
        return "mmsearch_only_correct"
    return "both_wrong"


def audit() -> dict[str, Any]:
    require(evidence.file_record(PROTOCOL)["sha256"] == PROTOCOL_SHA256, "protocol SHA mismatch")
    require(evidence.file_record(INPUT_MANIFEST)["sha256"] == INPUT_MANIFEST_SHA256, "input manifest SHA mismatch")
    require(evidence.file_record(MM_PREDICTIONS)["sha256"] == MM_PREDICTIONS_SHA256, "MMSearch predictions SHA mismatch")
    inputs = read_json(INPUT_MANIFEST)
    examples = inputs.get("examples")
    require(isinstance(examples, list) and len(examples) == 50, "input count mismatch")
    require(collections.Counter(item["category"] for item in examples) == {"search_free": 25, "search_required": 25}, "input balance mismatch")

    base_completion = read_json(BASE_COMPLETION)
    require(base_completion.get("schema_version") == "mmsearch.step12.base-completion.v1", "Base completion schema mismatch")
    require(base_completion.get("status") == "passed" and base_completion.get("completed_examples") == 50, "Base completion status mismatch")
    require(base_completion.get("unexpected_external_tool_calls") == 0, "Base reported external tool calls")
    require(base_completion.get("base_fully_gpu_resident") is True, "Base residency flag mismatch")
    require(base_completion.get("credentials_recorded") is False, "Base credential flag mismatch")
    for stage in (5, 20, 50):
        record = base_completion.get("stages", {}).get(str(stage))
        verify_record(record, f"Base stage {stage}")
        value = read_json(Path(record["path"]))
        require(value.get("status") == "passed" and value.get("stage_target") == stage, f"Base stage {stage} status mismatch")
        require(value.get("prediction_count") == stage, f"Base stage {stage} count mismatch")
        require(value.get("serialized_evidence_scan", {}).get("pass") is True, f"Base stage {stage} evidence scan mismatch")
        require(value.get("unexpected_external_tool_calls") == 0, f"Base stage {stage} external tool calls")
    verify_record(base_completion["final_predictions"], "Base final predictions")
    verify_record(base_completion["final_failure_summary"], "Base final failure summary")
    verify_record(base_completion["state"], "Base state")
    stage50 = read_json(Path(base_completion["stages"]["50"]["path"]))
    prediction_records = stage50.get("prediction_records")
    require(isinstance(prediction_records, list) and len(prediction_records) == 50, "Base prediction record count mismatch")
    base_values = []
    prediction_paths = []
    for expected_index, (entry, record) in enumerate(zip(examples, prediction_records, strict=True), start=1):
        require(record.get("eval_index") == expected_index, f"Base record order mismatch at {expected_index}")
        current = verify_record(record, f"Base prediction {expected_index}")
        value = read_json(Path(current["path"]))
        require(value.get("eval_index") == expected_index, f"Base embedded index mismatch at {expected_index}")
        require(value.get("data_id") == entry["data_id"], f"Base data ID mismatch at {expected_index}")
        require(value.get("category") == entry["category"], f"Base category mismatch at {expected_index}")
        require(value.get("source_row_index") == entry["source_row_index"], f"Base source row mismatch at {expected_index}")
        require(value.get("selection_rank_sha256") == entry["selection_rank_sha256"], f"Base rank mismatch at {expected_index}")
        execution = value.get("execution", {})
        require(execution.get("mode") == "direct_answer_no_tools", f"Base mode mismatch at {expected_index}")
        require(execution.get("external_tool_calls") == execution.get("image_search_calls") == execution.get("text_search_calls") == 0, f"Base tool call found at {expected_index}")
        require(execution.get("candidate_answers_provided_to_model") is False and execution.get("ground_truth_provided_to_model") is False, f"Base leakage flag mismatch at {expected_index}")
        result = value.get("result", {})
        recomputed = strict_em(result.get("answer"), result.get("ground_truth"))
        require(result.get("strict_exact_match") is recomputed, f"Base strict EM mismatch at {expected_index}")
        require(result.get("strict_exact_match_definition") == ANSWER_REPLACEMENT_DEFINITION, f"Base EM definition mismatch at {expected_index}")
        base_values.append(value)
        prediction_paths.append(Path(current["path"]))

    base_jsonl = read_jsonl(BASE_ROOT / "predictions.jsonl")
    require(base_jsonl == base_values, "Base JSONL differs from prediction files")
    base_metrics = read_json(BASE_ROOT / "metrics.json")
    base_correct = sum(item["result"]["strict_exact_match"] is True for item in base_values)
    require(base_metrics.get("evaluated") == 50 and base_metrics.get("correct") == base_correct, "Base metrics count mismatch")
    require(base_metrics.get("accuracy_percent") == round(base_correct / 50 * 100, 4), "Base accuracy mismatch")
    require(base_metrics.get("total_search_calls") == 0 and base_metrics.get("search_ratio_percent") == 0.0, "Base Search Ratio mismatch")
    for category in ("search_free", "search_required"):
        selected = [item for item in base_values if item["category"] == category]
        correct = sum(item["result"]["strict_exact_match"] is True for item in selected)
        require(base_metrics["categories"][category]["count"] == 25, f"Base category count mismatch: {category}")
        require(base_metrics["categories"][category]["correct"] == correct, f"Base category correct mismatch: {category}")
        require(base_metrics["categories"][category]["accuracy_percent"] == round(correct / 25 * 100, 4), f"Base category accuracy mismatch: {category}")

    comparison_completion = read_json(COMPARISON_COMPLETION)
    require(comparison_completion.get("schema_version") == "mmsearch.step12.completion.v1", "comparison completion schema mismatch")
    require(comparison_completion.get("status") == "passed" and comparison_completion.get("step") == 12, "comparison completion status mismatch")
    require(comparison_completion.get("credentials_recorded") is False, "comparison credential flag mismatch")
    verify_record(comparison_completion["protocol"], "comparison protocol")
    verify_record(comparison_completion["base_completion"], "comparison Base completion")
    verify_record(comparison_completion["mmsearch_completion"], "comparison MMSearch completion")
    verify_record(comparison_completion["base_predictions"], "comparison Base predictions")
    verify_record(comparison_completion["mmsearch_predictions"], "comparison MMSearch predictions")
    for index, record in enumerate(comparison_completion.get("artifacts", []), start=1):
        verify_record(record, f"comparison artifact {index}")
    require(comparison_completion.get("serialized_evidence_scan", {}).get("pass") is True, "comparison producer scan mismatch")

    mm_values = read_jsonl(MM_PREDICTIONS)
    require(len(mm_values) == 50, "MMSearch paired count mismatch")
    paired = read_jsonl(COMPARISON_ROOT / "paired_outcomes.jsonl")
    require(len(paired) == 50, "paired outcome count mismatch")
    labels = collections.Counter()
    for index, (base_item, mm_item, pair) in enumerate(zip(base_values, mm_values, paired, strict=True), start=1):
        trace = mm_item["trace"]
        require(pair.get("eval_index") == index, f"paired index mismatch at {index}")
        require(pair.get("data_id") == base_item["data_id"] == trace["data_id"], f"paired ID mismatch at {index}")
        require(pair.get("ground_truth") == base_item["result"]["ground_truth"] == trace["ground_truth"], f"paired ground truth mismatch at {index}")
        expected_label = paired_label(base_item["result"]["strict_exact_match"] is True, trace["exact_match"] is True)
        require(pair.get("outcome") == expected_label, f"paired label mismatch at {index}")
        labels[expected_label] += 1
    comparison_metrics = read_json(COMPARISON_ROOT / "comparison_metrics.json")
    require(comparison_metrics == comparison_completion["comparison_metrics"], "embedded comparison metrics mismatch")
    require(comparison_metrics["base"]["correct"] == base_correct, "comparison Base correct mismatch")
    mm_correct = sum(item["trace"]["exact_match"] is True for item in mm_values)
    require(comparison_metrics["mmsearch"]["correct"] == mm_correct, "comparison MMSearch correct mismatch")
    require(comparison_metrics["paired_outcomes"] == {
        "both_correct": labels["both_correct"],
        "base_only_correct": labels["base_only_correct"],
        "mmsearch_only_correct": labels["mmsearch_only_correct"],
        "both_wrong": labels["both_wrong"],
    }, "paired outcome totals mismatch")
    require(comparison_metrics["mmsearch_minus_base_accuracy_points"] == round((mm_correct - base_correct) / 50 * 100, 4), "accuracy delta mismatch")

    comparison_json = [
        COMPARISON_ROOT / "comparison_metrics.json",
        COMPARISON_ROOT / "success_failure_examples.json",
        COMPARISON_ROOT / "model_and_dataset_revisions.json",
        COMPARISON_COMPLETION,
    ]
    comparison_text = [
        COMPARISON_ROOT / "paired_outcomes.jsonl",
        COMPARISON_ROOT / "final_report.md",
        COMPARISON_ROOT / "pip_freeze.txt",
        COMPARISON_ROOT / "gpu_info.txt",
        COMPARISON_ROOT / "git_commit.txt",
    ]
    base_json = prediction_paths + [
        BASE_ROOT / "metrics.json",
        BASE_ROOT / "failure_summary.json",
        BASE_ROOT / "stage_5_manifest.json",
        BASE_ROOT / "stage_20_manifest.json",
        BASE_ROOT / "stage_50_manifest.json",
        BASE_COMPLETION,
    ]
    base_text = [BASE_ROOT / "predictions_5.jsonl", BASE_ROOT / "predictions_20.jsonl", BASE_ROOT / "predictions_50.jsonl", BASE_ROOT / "predictions.jsonl"]
    scan = evidence.scan_serialized_evidence(base_json + comparison_json, base_text + comparison_text)
    require(scan.get("pass") is True, "independent Step 12 serialized evidence scan failed")
    return {
        "schema_version": "mmsearch.step12.completion-audit.v1",
        "status": "passed",
        "step": 12,
        "prediction_files_verified": 50,
        "fixed_input_order_verified": True,
        "strict_em_recomputed_for_base": True,
        "paired_outcomes_recomputed": True,
        "base_external_tool_calls": 0,
        "base_correct": base_correct,
        "mmsearch_correct": mm_correct,
        "paired_outcomes": dict(sorted(labels.items())),
        "accuracy_delta_points": round((mm_correct - base_correct) / 50 * 100, 4),
        "base_completion": evidence.file_record(BASE_COMPLETION),
        "comparison_completion": evidence.file_record(COMPARISON_COMPLETION),
        "serialized_evidence_scan": scan,
        "credentials_recorded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        require(strict_em(" Answer ", "answer"), "EM positive self-test failed")
        require(not strict_em("two  spaces", "two spaces"), "EM whitespace self-test failed")
        require(paired_label(True, False) == "base_only_correct", "paired label self-test failed")
        print(json.dumps({"status": "passed", "pure_self_tests": 3}))
        return
    require(os.environ.get("SERPER_API_KEY"), "SERPER_API_KEY required for exact credential-value scan")
    result = audit()
    evidence.atomic_write_json(OUTPUT, result)
    print(json.dumps({
        "status": result["status"],
        "base_correct": result["base_correct"],
        "mmsearch_correct": result["mmsearch_correct"],
        "accuracy_delta_points": result["accuracy_delta_points"],
        "output": str(OUTPUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
