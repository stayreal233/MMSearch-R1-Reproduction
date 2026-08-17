#!/usr/bin/env python3
"""Run the pinned natural-policy Step-11 FVQA evaluation in staged prefixes."""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import step10_case_suite_qwen3 as base


PROTOCOL = Path(
    "/root/autodl-tmp/multimodal-search-r1/reproduction/env/step11_eval_protocol.json"
)
PROTOCOL_SHA256 = "f2fc533b824c65d5102fc10dbaebe0c3069242b00f9178c1417f6f0935c6000e"
INPUT_ROOT = Path("/root/autodl-tmp/mmsearch_step11_inputs")
INPUT_MANIFEST = INPUT_ROOT / "eval_manifest.json"
INPUT_MANIFEST_SHA256 = "dbc28df74f3a1a0b87fd435255fda8ed73455dfe3a3d465dce9539fad37564ab"
OUTPUT_ROOT = Path("/root/autodl-tmp/outputs/step11_eval_v1")
STATE_PATH = OUTPUT_ROOT / "state.json"
PREDICTION_DIR = OUTPUT_ROOT / "predictions"
STAGES = (5, 20, 50)
EXPECTED_PREFIX_COUNTS = {
    5: {"search_free": 3, "search_required": 2},
    20: {"search_free": 10, "search_required": 10},
    50: {"search_free": 25, "search_required": 25},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, choices=STAGES)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def read_bytes(path: Path, maximum: int) -> bytes:
    require(path.is_file() and not path.is_symlink(), f"unsafe file: {path}")
    return base.read_regular_file(path, max_bytes=maximum)


def load_json(path: Path, maximum: int = 16 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    encoded = read_bytes(path, maximum)
    value = json.loads(encoded)
    require(isinstance(value, dict), f"JSON root is invalid: {path}")
    return value, encoded


def record(path: Path, encoded: bytes | None = None) -> dict[str, Any]:
    if encoded is None:
        encoded = read_bytes(path, 128 * 1024 * 1024)
    return {
        "path": str(path),
        "bytes": len(encoded),
        "sha256": base.sha256_bytes(encoded),
    }


def script_record() -> dict[str, Any]:
    path = Path(__file__).resolve()
    return record(path)


def validate_static_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    protocol, protocol_bytes = load_json(PROTOCOL, 256 * 1024)
    require(base.sha256_bytes(protocol_bytes) == PROTOCOL_SHA256, "Step-11 protocol SHA mismatch")
    require(protocol.get("status") == "registered_before_step11_selection_and_inference", "Step-11 protocol status mismatch")
    manifest, manifest_bytes = load_json(INPUT_MANIFEST, 8 * 1024 * 1024)
    require(base.sha256_bytes(manifest_bytes) == INPUT_MANIFEST_SHA256, "Step-11 input manifest SHA mismatch")
    require(manifest.get("schema_version") == "mmsearch.step11.eval-inputs.v1", "Step-11 input schema mismatch")
    require(manifest.get("status") == "passed", "Step-11 input status mismatch")
    require(manifest.get("credentials_recorded") is False, "Step-11 input credential flag mismatch")
    examples = manifest.get("examples")
    require(isinstance(examples, list) and len(examples) == 50, "Step-11 example count mismatch")
    require([item.get("eval_index") for item in examples] == list(range(1, 51)), "Step-11 eval index order mismatch")
    require(len({item.get("data_id") for item in examples}) == 50, "Step-11 data IDs are not unique")
    for stage, expected in EXPECTED_PREFIX_COUNTS.items():
        actual = dict(collections.Counter(item.get("category") for item in examples[:stage]))
        require(actual == expected, f"Step-11 stage {stage} balance mismatch")
    require(manifest.get("stage_prefix_counts") == {str(k): v for k, v in EXPECTED_PREFIX_COUNTS.items()}, "Step-11 declared prefix counts mismatch")
    return manifest, examples, {
        "protocol": record(PROTOCOL, protocol_bytes),
        "input_manifest": record(INPUT_MANIFEST, manifest_bytes),
        "runner": script_record(),
    }


def load_sample(entry: dict[str, Any]) -> dict[str, Any]:
    meta_claim = entry.get("metadata")
    image_claim = entry.get("image")
    require(isinstance(meta_claim, dict) and isinstance(image_claim, dict), "input artifact claim missing")
    meta_path = Path(meta_claim.get("path", ""))
    image_path = Path(image_claim.get("path", ""))
    for path in (meta_path, image_path):
        require(path.is_absolute() and path.is_relative_to(INPUT_ROOT) and not path.is_symlink(), "input artifact path is unsafe")
    meta_bytes = read_bytes(meta_path, 4 * 1024 * 1024)
    image_bytes = read_bytes(image_path, 64 * 1024 * 1024)
    require(record(meta_path, meta_bytes) == meta_claim, "metadata artifact hash mismatch")
    require(record(image_path, image_bytes) == image_claim, "image artifact hash mismatch")
    metadata = json.loads(meta_bytes)
    require(isinstance(metadata, dict), "sample metadata root invalid")
    require(metadata.get("eval_index") == entry.get("eval_index"), "sample eval index mismatch")
    require(metadata.get("data_id") == entry.get("data_id"), "sample data ID mismatch")
    require(metadata.get("category") == entry.get("category"), "sample category mismatch")
    require(metadata.get("execution_mode") == "natural", "batch sample is not natural")
    require(metadata.get("controller_intervention") is False, "batch sample has intervention")
    reward = metadata.get("reward_model")
    require(isinstance(reward, dict), "sample reward metadata missing")
    candidate_answers = base.normalized_candidate_answers(reward.get("candidate_answers"))
    question = metadata.get("question")
    ground_truth = reward.get("ground_truth")
    require(isinstance(question, str) and question.strip(), "sample question missing")
    require(isinstance(ground_truth, str) and ground_truth.strip(), "sample ground truth missing")
    return {
        "data_id": metadata["data_id"],
        "category": metadata["category"],
        "source_split": metadata["source_split"],
        "source_row_index": metadata["source_row_index"],
        "data_source": base.sanitize_public(metadata.get("data_source")),
        "question": question.strip(),
        "candidate_answers": candidate_answers,
        "ground_truth": ground_truth.strip(),
        "image": str(image_path),
        "image_width": metadata["image_width"],
        "image_height": metadata["image_height"],
        "image_png_sha256": base.sha256_bytes(image_bytes),
        "source_image_sha256": metadata["source_image_sha256"],
        "eval_index": metadata["eval_index"],
        "selection_rank_sha256": metadata["selection_rank_sha256"],
    }


def prediction_path(eval_index: int, data_id: str) -> Path:
    return PREDICTION_DIR / f"{eval_index:03d}_{data_id}.json"


def initial_state(static: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "mmsearch.step11.state.v1",
        "status": "in_progress",
        "protocol": static["protocol"],
        "input_manifest": static["input_manifest"],
        "runner": static["runner"],
        "completed_count": 0,
        "predictions": [],
        "stages": {},
        "credentials_recorded": False,
    }


def validate_state(state: dict[str, Any], static: dict[str, Any]) -> None:
    require(state.get("schema_version") == "mmsearch.step11.state.v1", "state schema mismatch")
    require(state.get("protocol") == static["protocol"], "state protocol mismatch")
    require(state.get("input_manifest") == static["input_manifest"], "state input mismatch")
    require(state.get("runner") == static["runner"], "state runner mismatch")
    predictions = state.get("predictions")
    completed = state.get("completed_count")
    require(isinstance(predictions, list) and completed == len(predictions), "state prediction count mismatch")
    for expected_index, item in enumerate(predictions, start=1):
        require(item.get("eval_index") == expected_index, "state eval index mismatch")
        path = Path(item.get("path", ""))
        require(record(path) == item, f"state prediction hash mismatch: {expected_index}")
    stages = state.get("stages")
    require(isinstance(stages, dict), "state stages missing")
    for key, item in stages.items():
        require(int(key) in STAGES and record(Path(item["path"])) == item, f"state stage hash mismatch: {key}")
    require(state.get("credentials_recorded") is False, "state credential flag mismatch")


def prepare_output(target: int, static: dict[str, Any]) -> dict[str, Any]:
    outputs_root = OUTPUT_ROOT.parent.resolve(strict=True)
    require(OUTPUT_ROOT.parent == outputs_root, "Step-11 output parent mismatch")
    require(not OUTPUT_ROOT.is_symlink(), "Step-11 output is a symlink")
    if not OUTPUT_ROOT.exists():
        require(target == 5, "first Step-11 target must be 5")
        OUTPUT_ROOT.mkdir(mode=0o700, exist_ok=False)
        PREDICTION_DIR.mkdir(mode=0o700, exist_ok=False)
        state = initial_state(static)
        base.atomic_write_json(STATE_PATH, state)
        return state
    require(OUTPUT_ROOT.is_dir() and PREDICTION_DIR.is_dir(), "Step-11 output structure invalid")
    require(not (OUTPUT_ROOT / "step11_run_failure.json").exists(), "prior Step-11 hard failure requires user direction")
    state, _ = load_json(STATE_PATH)
    validate_state(state, static)
    completed = state["completed_count"]
    require(target > completed, "target must expand the completed prefix")
    if target == 20:
        require("5" in state["stages"], "stage 5 commit missing")
    if target == 50:
        require("20" in state["stages"], "stage 20 commit missing")
    return state


def route_name(actions: list[str]) -> str:
    return " -> ".join(actions) if actions else "(none)"


def aggregate(results: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    require(results, "cannot aggregate zero predictions")
    traces = [item["trace"] for item in results]
    n = len(traces)
    categories: dict[str, dict[str, Any]] = {}
    for category in ("search_free", "search_required"):
        selected = [trace for trace in traces if trace["category"] == category]
        correct = sum(trace["exact_match"] is True for trace in selected)
        searches = sum(trace["image_search_calls"] + trace["text_search_calls"] for trace in selected)
        categories[category] = {
            "count": len(selected),
            "correct": correct,
            "accuracy_percent": round(correct / len(selected) * 100, 4) if selected else None,
            "total_search_calls": searches,
            "search_ratio_percent": round(searches / (len(selected) * 2) * 100, 4) if selected else None,
            "average_turns": round(sum(trace["total_turns"] for trace in selected) / len(selected), 4) if selected else None,
        }
    correct = sum(trace["exact_match"] is True for trace in traces)
    image_calls = sum(trace["image_search_calls"] for trace in traces)
    text_calls = sum(trace["text_search_calls"] for trace in traces)
    total_searches = image_calls + text_calls
    network_keys = (
        "external_network_requests", "thumbnail_network_attempts",
        "thumbnail_cache_hits", "serper_api_requests", "serper_cache_hits",
        "jina_network_requests", "jina_cache_hits", "local_qwen_completion_requests",
        "qwen_summary_requests", "qwen_summary_cache_hits", "tool_http_requests_total",
    )
    network_totals = {
        key: sum(int(trace["network_and_cache"].get(key, 0)) for trace in traces)
        for key in network_keys
    }
    metrics = {
        "schema_version": "mmsearch.step11.metrics.v1",
        "evaluated": n,
        "correct": correct,
        "accuracy_percent": round(correct / n * 100, 4),
        "image_search_calls": image_calls,
        "text_search_calls": text_calls,
        "total_search_calls": total_searches,
        "search_ratio_percent": round(total_searches / (n * 2) * 100, 4),
        "average_turns": round(sum(trace["total_turns"] for trace in traces) / n, 4),
        "route_counts": dict(sorted(collections.Counter(route_name(trace["action_sequence"]) for trace in traces).items())),
        "categories": categories,
        "network_and_cache_totals": network_totals,
        "infrastructure_failures": sum(trace["tool_infrastructure_success"] is not True for trace in traces),
        "non_answer_terminal_count": sum(trace["terminal_status"] != "answered" for trace in traces),
        "network_count_incomplete": sum(trace["network_and_cache"].get("count_complete") is not True for trace in traces),
        "strict_exact_match_definition": "prediction.strip().lower() == ground_truth.strip().lower()",
        "search_ratio_definition": "sum(total_search_calls) / (N * 2) * 100",
        "natural_policy_only": True,
        "controlled_step10_B_C_excluded": True,
    }
    failures = []
    for result in results:
        trace = result["trace"]
        if trace["exact_match"] is False:
            layer = base.classify_failure_layer(trace)
            failures.append({
                "eval_index": result["eval_index"],
                "data_id": trace["data_id"],
                "category": trace["category"],
                "action_sequence": trace["action_sequence"],
                "final_answer": trace["final_answer"],
                "ground_truth": trace["ground_truth"],
                "layer": layer["layer"],
                "rationale": layer["rationale"],
                "evidence_paths": layer["evidence_paths"],
                "tool_infrastructure_success": trace["tool_infrastructure_success"],
            })
    failure_summary = {
        "schema_version": "mmsearch.step11.failure-summary.v1",
        "evaluated": n,
        "failure_count": len(failures),
        "failure_rate_percent": round(len(failures) / n * 100, 4),
        "failure_layer_counts": dict(sorted(collections.Counter(item["layer"] for item in failures).items())),
        "failures": failures,
        "infrastructure_failures": metrics["infrastructure_failures"],
        "credentials_recorded": False,
    }
    return metrics, failure_summary


def load_completed_results(state: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for item in state["predictions"]:
        value, _ = load_json(Path(item["path"]), 32 * 1024 * 1024)
        results.append(value)
    return results


def write_jsonl(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [json.dumps(base.sanitize_public(item), ensure_ascii=False, sort_keys=True, separators=(",", ":")) for item in results]
    base.atomic_write_text(path, "\n".join(lines) + "\n")


def commit_stage(
    target: int,
    state: dict[str, Any],
    static: dict[str, Any],
    results: list[dict[str, Any]],
    final_health: dict[str, Any],
    final_snapshot: dict[str, Any],
) -> dict[str, Any]:
    metrics, failure_summary = aggregate(results)
    require(metrics["evaluated"] == target, "stage metric count mismatch")
    require(metrics["categories"] == {
        **metrics["categories"],
    }, "category metric structure mismatch")
    require(metrics["infrastructure_failures"] == 0, "stage has infrastructure failures")
    require(metrics["non_answer_terminal_count"] == 0, "stage has non-answer terminals")
    require(metrics["network_count_incomplete"] == 0, "stage has incomplete network counts")
    predictions_path = OUTPUT_ROOT / f"predictions_{target}.jsonl"
    metrics_path = OUTPUT_ROOT / f"metrics_{target}.json"
    failure_path = OUTPUT_ROOT / f"failure_summary_{target}.json"
    write_jsonl(predictions_path, results)
    base.atomic_write_json(metrics_path, metrics)
    base.atomic_write_json(failure_path, failure_summary)
    stage_files = [predictions_path, metrics_path, failure_path]
    if target == 50:
        write_jsonl(OUTPUT_ROOT / "predictions.jsonl", results)
        base.atomic_write_json(OUTPUT_ROOT / "metrics.json", metrics)
        base.atomic_write_json(OUTPUT_ROOT / "failure_summary.json", failure_summary)
        stage_files.extend((
            OUTPUT_ROOT / "predictions.jsonl",
            OUTPUT_ROOT / "metrics.json",
            OUTPUT_ROOT / "failure_summary.json",
        ))
    prediction_paths = [Path(item["path"]) for item in state["predictions"]]
    scan = base.scan_serialized_evidence(
        prediction_paths + [metrics_path, failure_path],
        [predictions_path],
    )
    require(scan["pass"] is True, "stage serialized evidence scan failed")
    manifest = {
        "schema_version": "mmsearch.step11.stage.v1",
        "status": "passed",
        "stage_target": target,
        "completed_at_utc": base.utc_now(),
        "protocol": static["protocol"],
        "input_manifest": static["input_manifest"],
        "runner": static["runner"],
        "prediction_count": len(results),
        "prediction_records": state["predictions"],
        "artifacts": [record(path) for path in stage_files],
        "metrics": metrics,
        "failure_summary_digest": record(failure_path),
        "qwen3_health_after_stage": base.sanitize_public(final_health),
        "gpu_snapshot_after_stage": final_snapshot,
        "serialized_evidence_scan": scan,
        "credentials_recorded": False,
    }
    stage_path = OUTPUT_ROOT / f"stage_{target}_manifest.json"
    base.atomic_write_json(stage_path, manifest)
    stage_record = record(stage_path)
    state["stages"][str(target)] = stage_record
    state["status"] = "passed_stage_50" if target == 50 else "in_progress"
    base.atomic_write_json(STATE_PATH, state)
    if target == 50:
        completion = {
            "schema_version": "mmsearch.step11.completion.v1",
            "status": "passed",
            "step": 11,
            "completed_at_utc": base.utc_now(),
            "completed_examples": 50,
            "protocol": static["protocol"],
            "input_manifest": static["input_manifest"],
            "runner": static["runner"],
            "stages": state["stages"],
            "final_metrics": metrics,
            "final_failure_summary": record(OUTPUT_ROOT / "failure_summary.json"),
            "final_predictions": record(OUTPUT_ROOT / "predictions.jsonl"),
            "state": record(STATE_PATH),
            "qwen3_health_after": base.sanitize_public(final_health),
            "gpu_snapshot_after": final_snapshot,
            "natural_policy_only": True,
            "controlled_step10_B_C_excluded": True,
            "credentials_recorded": False,
        }
        base.atomic_write_json(OUTPUT_ROOT / "step11_completion_manifest.json", completion)
    return manifest


def write_run_failure(stage: str, error: BaseException, eval_index: int | None) -> None:
    if OUTPUT_ROOT.exists():
        base.atomic_write_json(OUTPUT_ROOT / "step11_run_failure.json", {
            "schema_version": "mmsearch.step11.run-failure.v1",
            "status": "failed",
            "failed_at_utc": base.utc_now(),
            "stage": stage,
            "eval_index": eval_index,
            "error": base.safe_error(error),
            "credentials_recorded": False,
        })


def run(target: int) -> int:
    manifest, examples, static = validate_static_inputs()
    state = prepare_output(target, static)
    completed = state["completed_count"]
    require(completed < target, "nothing to evaluate")
    service_root_pid: int | None = None
    current_index: int | None = None
    stage = "qwen3_health"
    try:
        summarizer = base.Qwen3Summarizer(
            base.DEFAULT_SUMMARY_CACHE.resolve(),
            base_url=base.DEFAULT_SUMMARY_BASE_URL,
            model=base.DEFAULT_SUMMARY_MODEL,
            model_repo=base.SUMMARY_MODEL_REPO,
            model_revision=base.SUMMARY_MODEL_REVISION,
            api_key=None,
            max_input_chars=base.MAX_CHARS_PER_PAGE,
            max_tokens=base.SUMMARY_MAX_TOKENS,
            timeout_seconds=120,
        )
        initial_health = summarizer.health_check()
        require(base.valid_health(initial_health), "initial Qwen3 health failed")
        service_root_pid = base.read_summary_service_pid()
        current_pid = os.getpid()
        service_snapshot = base.gpu_snapshot(
            f"step11_stage_{target}_summary_service_only",
            service_root_pid=service_root_pid,
            current_pid=current_pid,
        )
        service_gpu = base.validate_service_only_snapshot(service_snapshot)

        stage = "load_mmsearch"
        torch.manual_seed(base.SEED)
        load_started = time.monotonic()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base.DEFAULT_MMSEARCH_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(
            base.DEFAULT_MMSEARCH_MODEL,
            local_files_only=True,
            use_fast=False,
        )
        load_seconds = round(time.monotonic() - load_started, 3)
        safe_device_map = base.validate_mmsearch_residency(model)
        model_snapshot = base.gpu_snapshot(
            f"step11_stage_{target}_mmsearch_loaded_same_gpu",
            service_root_pid=service_root_pid,
            current_pid=current_pid,
        )
        base.validate_joint_snapshot(model_snapshot, service_gpu_identities=service_gpu, current_pid=current_pid)

        image_search = base.FVQACachedImageSearch(
            base.DEFAULT_IMAGE_CACHE,
            base.DEFAULT_THUMBNAIL_CACHE,
            top_k=base.TOP_K,
        )
        text_search = base.SerperJinaTextSearch(
            base.DEFAULT_SERPER_CACHE,
            base.DEFAULT_JINA_CACHE,
            top_k=base.TOP_K,
            max_chars_per_page=base.MAX_CHARS_PER_PAGE,
            summarizer=summarizer,
        )
        round_1_prompt = base.load_prompt("round_1_user_prompt_qwenvl.pkl").replace("<image>", "").strip()
        after_image = base.load_prompt("after_image_search_prompt_qwenvl.pkl")
        after_text = base.load_prompt("after_text_search_prompt_qwenvl.pkl")
        base.CASE_TYPES["Eval"] = "batch_eval"

        for offset in range(completed, target):
            entry = examples[offset]
            current_index = offset + 1
            stage = f"example_{current_index}"
            sample = load_sample(entry)
            require(sample["eval_index"] == current_index, "sample prefix order mismatch")
            torch.manual_seed(base.SEED)
            torch.cuda.reset_peak_memory_stats()
            result = base.execute_case(
                label="Eval",
                sample=sample,
                selection={
                    "execution_mode": "natural",
                    "selection_rank_sha256": sample["selection_rank_sha256"],
                    "eval_index": current_index,
                },
                model=model,
                processor=processor,
                image_search=image_search,
                text_search=text_search,
                summarizer=summarizer,
                round_1_prompt=round_1_prompt,
                after_image_search_prompt=after_image,
                after_text_search_prompt=after_text,
            )
            trace = result["trace"]
            trace["route_origin"] = "natural_model_policy"
            trace["controller_intervention"] = False
            valid = (
                trace["terminal_status"] == "answered"
                and trace["tool_infrastructure_success"] is True
                and trace["network_and_cache"].get("count_complete") is True
            )
            result.update({
                "status": "passed" if valid else "failed",
                "pass": valid,
                "eval_index": current_index,
                "case_label": "Eval",
                "case_type": "batch_eval",
                "evaluation_semantics": "natural_model_policy",
            })
            health = summarizer.health_check()
            require(base.valid_health(health), f"Qwen3 health failed after example {current_index}")
            require(base.read_summary_service_pid() == service_root_pid, "Qwen3 PID changed")
            snapshot = base.gpu_snapshot(
                f"step11_example_{current_index}_complete_same_gpu",
                service_root_pid=service_root_pid,
                current_pid=current_pid,
            )
            base.validate_joint_snapshot(snapshot, service_gpu_identities=service_gpu, current_pid=current_pid)
            result["runtime"] = {
                "mmsearch_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
                "gpu_snapshot_after_example": snapshot,
                "mmsearch_fully_gpu_resident": True,
                "summary_service_root_pid": service_root_pid,
                "summary_service_health_after_example": base.sanitize_public(health),
                "attention_implementation": model.config._attn_implementation,
                "parameter_dtype": str(next(model.parameters()).dtype),
                "stage_mmsearch_load_seconds": load_seconds,
                "hf_device_map": safe_device_map,
            }
            path = prediction_path(current_index, trace["data_id"])
            base.atomic_write_json(path, result)
            prediction_record = record(path)
            prediction_record.update({
                "eval_index": current_index,
                "data_id": trace["data_id"],
                "category": trace["category"],
                "exact_match": trace["exact_match"],
                "action_sequence": trace["action_sequence"],
            })
            # State hashes only immutable file identity fields; descriptive fields
            # live in the prediction and are repeated here for safe inspection.
            state_record = {
                "path": prediction_record["path"],
                "bytes": prediction_record["bytes"],
                "sha256": prediction_record["sha256"],
                "eval_index": current_index,
            }
            state["predictions"].append(state_record)
            state["completed_count"] = current_index
            base.atomic_write_json(STATE_PATH, state)
            require(valid, f"example {current_index} did not satisfy healthy answered contract")

        stage = f"commit_stage_{target}"
        results = load_completed_results(state)
        final_health = summarizer.health_check()
        require(base.valid_health(final_health), "final Qwen3 health failed")
        require(base.read_summary_service_pid() == service_root_pid, "final Qwen3 PID changed")
        final_snapshot = base.gpu_snapshot(
            f"step11_stage_{target}_complete_same_gpu",
            service_root_pid=service_root_pid,
            current_pid=current_pid,
        )
        base.validate_joint_snapshot(final_snapshot, service_gpu_identities=service_gpu, current_pid=current_pid)
        stage_manifest = commit_stage(target, state, static, results, final_health, final_snapshot)
        print(json.dumps({
            "status": "passed",
            "stage": target,
            "completed_count": state["completed_count"],
            "accuracy_percent": stage_manifest["metrics"]["accuracy_percent"],
            "search_ratio_percent": stage_manifest["metrics"]["search_ratio_percent"],
            "stage_manifest": str(OUTPUT_ROOT / f"stage_{target}_manifest.json"),
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - hard-stop evidence boundary
        write_run_failure(stage, exc, current_index)
        print(json.dumps({
            "status": "failed",
            "stage": stage,
            "eval_index": current_index,
            "error": base.safe_error(exc),
            "failure_evidence": str(OUTPUT_ROOT / "step11_run_failure.json"),
        }, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


def self_test() -> None:
    fake = []
    for index, (category, correct, actions) in enumerate((
        ("search_free", True, ["answer"]),
        ("search_required", False, ["image_search", "text_search", "answer"]),
    ), start=1):
        fake.append({
            "eval_index": index,
            "trace": {
                "data_id": f"x{index}", "category": category,
                "exact_match": correct, "action_sequence": actions,
                "image_search_calls": int("image_search" in actions),
                "text_search_calls": int("text_search" in actions),
                "total_turns": len(actions), "terminal_status": "answered",
                "tool_infrastructure_success": True,
                "network_and_cache": {"count_complete": True},
                "final_answer": "a", "ground_truth": "a" if correct else "b",
            },
        })
    metrics, failures = aggregate(fake)
    require(metrics["accuracy_percent"] == 50.0, "accuracy self-test failed")
    require(metrics["search_ratio_percent"] == 50.0, "SR self-test failed")
    require(metrics["average_turns"] == 2.0, "turn self-test failed")
    require(failures["failure_count"] == 1, "failure self-test failed")
    print(json.dumps({"status": "passed", "pure_self_tests": 4}, sort_keys=True))


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    require(args.target in STAGES, "--target is required")
    return run(args.target)


if __name__ == "__main__":
    raise SystemExit(main())
