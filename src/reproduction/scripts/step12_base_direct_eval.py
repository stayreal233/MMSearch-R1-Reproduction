#!/usr/bin/env python3
"""Hash-bound, resumable Step 12 Base Direct Answer evaluation."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "reproduction" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import torch  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402

import step10_case_suite_qwen3 as evidence  # noqa: E402
from placeholder_control_flow import generate_response, image_to_data_uri  # noqa: E402
from real_text_search_flow_qwen3 import (  # noqa: E402
    gpu_snapshot,
    read_summary_service_pid,
    validate_joint_snapshot,
    validate_mmsearch_residency,
    validate_service_only_snapshot,
)


PROTOCOL = REPO_ROOT / "reproduction/env/step12_base_comparison_protocol.json"
REVISION_METADATA = REPO_ROOT / "reproduction/env/step12_base_huggingface_revision.json"
ARTIFACT_VALIDATION = Path("/root/autodl-tmp/outputs/step12_base_artifact_validation.json")
INPUT_MANIFEST = Path("/root/autodl-tmp/mmsearch_step11_inputs/eval_manifest.json")
MMSEARCH_COMPLETION = Path("/root/autodl-tmp/outputs/step11_eval_v2/step11_completion_manifest.json")
MMSEARCH_PREDICTIONS = Path("/root/autodl-tmp/outputs/step11_eval_v2/predictions.jsonl")
MMSEARCH_METRICS = Path("/root/autodl-tmp/outputs/step11_eval_v2/metrics.json")
MODEL_DIR = Path("/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
OUTPUT_ROOT = Path("/root/autodl-tmp/outputs/step12_base_direct_v1")
STATE_PATH = OUTPUT_ROOT / "state.json"
RUNNER_PATH = Path(__file__).resolve()

PROTOCOL_SHA256 = "3a80ee1fe4685cde68335a1ad336a3cf6f8f970a71f9664f6f66e22ee3d651f5"
REVISION_METADATA_SHA256 = "fb8c62723161017c615f95e48e5cb9a21d3c841c9eac0dd12e5fb29414b42a48"
INPUT_MANIFEST_SHA256 = "dbc28df74f3a1a0b87fd435255fda8ed73455dfe3a3d465dce9539fad37564ab"
MMSEARCH_COMPLETION_SHA256 = "f2747c945a022d578e3d053e7112a05937b0267894dc03b6f74adeabefe2ad87"
MMSEARCH_PREDICTIONS_SHA256 = "beda0f6a02f750b89a5a53cb9f39dcaac56dbe047e2fb1f4ecc01290d6ac48ff"
MMSEARCH_METRICS_SHA256 = "ec7c289417defc57ddd5a731e6bd1a2cb0a498013d281df863ad0aaccc5f0445"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MODEL_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
STAGES = (5, 20, 50)
MAX_NEW_TOKENS = 512
MAX_PIXELS = 672 * 672
DIRECT_INSTRUCTION = (
    "Answer the question directly using the image and your internal knowledge. "
    "Do not request or call any external tools. Return only one final answer "
    "enclosed in <answer> and </answer> tags, with no explanation."
)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_record(record: dict[str, Any], label: str) -> dict[str, Any]:
    current = evidence.file_record(Path(record["path"]))
    require(current["bytes"] == record["bytes"], f"{label} byte count mismatch")
    require(current["sha256"] == record["sha256"], f"{label} SHA mismatch")
    return current


def load_json(path: Path, maximum: int = 16 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    return evidence.load_json_object(path, max_bytes=maximum)


def parse_answer(response: str) -> tuple[str | None, str]:
    match = ANSWER_RE.search(response)
    if match is not None and match.group(1).strip():
        return match.group(1).strip(), "answer_tags"
    stripped = response.strip()
    if stripped:
        return stripped, "plain_text_fallback"
    return None, "empty_response"


def strict_em(answer: Any, ground_truth: Any) -> bool:
    return (
        isinstance(answer, str)
        and isinstance(ground_truth, str)
        and answer.strip().lower() == ground_truth.strip().lower()
    )


def validate_static_inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol, protocol_bytes = load_json(PROTOCOL)
    revision, revision_bytes = load_json(REVISION_METADATA)
    artifact, _ = load_json(ARTIFACT_VALIDATION, 32 * 1024 * 1024)
    inputs, input_bytes = load_json(INPUT_MANIFEST, 8 * 1024 * 1024)
    require(sha256_bytes(protocol_bytes) == PROTOCOL_SHA256, "protocol SHA mismatch")
    require(sha256_bytes(revision_bytes) == REVISION_METADATA_SHA256, "revision metadata SHA mismatch")
    require(sha256_bytes(input_bytes) == INPUT_MANIFEST_SHA256, "input manifest SHA mismatch")
    require(protocol.get("status") == "registered_before_step12_download_and_inference", "protocol status mismatch")
    require(protocol.get("base_model", {}).get("revision") == MODEL_REVISION, "protocol Base revision mismatch")
    require(protocol.get("scope", {}).get("sample_count") == 50, "protocol sample count mismatch")
    require(protocol.get("scope", {}).get("tool_enabled_base_secondary_experiment") is False, "tool-enabled Base is outside protocol")
    require(revision.get("revision") == MODEL_REVISION, "revision commit mismatch")
    require(artifact.get("status") == "passed", "artifact validation did not pass")
    require(artifact.get("revision") == MODEL_REVISION, "artifact validation revision mismatch")
    require(artifact.get("file_count") == 16 and artifact.get("shard_count") == 5, "artifact validation counts mismatch")
    require(artifact.get("all_lfs_shard_sha256_verified") is True, "artifact shard verification missing")
    for path, expected, label in (
        (MMSEARCH_COMPLETION, MMSEARCH_COMPLETION_SHA256, "MMSearch completion"),
        (MMSEARCH_PREDICTIONS, MMSEARCH_PREDICTIONS_SHA256, "MMSearch predictions"),
        (MMSEARCH_METRICS, MMSEARCH_METRICS_SHA256, "MMSearch metrics"),
    ):
        require(evidence.file_record(path)["sha256"] == expected, f"{label} SHA mismatch")
    examples = inputs.get("examples")
    require(isinstance(examples, list) and len(examples) == 50, "input examples must contain 50 entries")
    require([item["eval_index"] for item in examples] == list(range(1, 51)), "input indices are not 1..50")
    require(len({item["data_id"] for item in examples}) == 50, "input data IDs are not unique")
    require(collections.Counter(item["category"] for item in examples) == {"search_free": 25, "search_required": 25}, "category balance mismatch")
    static = {
        "protocol": evidence.file_record(PROTOCOL),
        "revision_metadata": evidence.file_record(REVISION_METADATA),
        "artifact_validation": evidence.file_record(ARTIFACT_VALIDATION),
        "input_manifest": evidence.file_record(INPUT_MANIFEST),
        "mmsearch_reference": {
            "completion_manifest": evidence.file_record(MMSEARCH_COMPLETION),
            "predictions": evidence.file_record(MMSEARCH_PREDICTIONS),
            "metrics": evidence.file_record(MMSEARCH_METRICS),
        },
        "runner": evidence.file_record(RUNNER_PATH),
    }
    return static, examples


def load_sample(entry: dict[str, Any]) -> dict[str, Any]:
    image_record = evidence.file_record(Path(entry["image"]["path"]))
    metadata_record = evidence.file_record(Path(entry["metadata"]["path"]))
    require(image_record["bytes"] == entry["image"]["bytes"] and image_record["sha256"] == entry["image"]["sha256"], "sample image binding mismatch")
    require(metadata_record["bytes"] == entry["metadata"]["bytes"] and metadata_record["sha256"] == entry["metadata"]["sha256"], "sample metadata binding mismatch")
    metadata, _ = load_json(Path(metadata_record["path"]))
    require(metadata.get("eval_index") == entry["eval_index"], "metadata eval index mismatch")
    require(metadata.get("data_id") == entry["data_id"], "metadata data ID mismatch")
    require(metadata.get("category") == entry["category"], "metadata category mismatch")
    require(metadata.get("source_row_index") == entry["source_row_index"], "metadata source row mismatch")
    require(metadata.get("selection_rank_sha256") == entry["selection_rank_sha256"], "selection rank mismatch")
    require(metadata.get("execution_mode") == "natural", "source metadata execution mode mismatch")
    require(metadata.get("controller_intervention") is False, "source metadata controller intervention mismatch")
    reward = metadata.get("reward_model")
    require(isinstance(reward, dict) and isinstance(reward.get("ground_truth"), str), "ground truth missing")
    require(isinstance(metadata.get("question"), str) and metadata["question"].strip(), "question missing")
    require(metadata.get("image") == image_record["path"], "metadata image path mismatch")
    return {
        "eval_index": entry["eval_index"],
        "data_id": entry["data_id"],
        "category": entry["category"],
        "source_row_index": entry["source_row_index"],
        "selection_rank_sha256": entry["selection_rank_sha256"],
        "question": metadata["question"],
        "ground_truth": reward["ground_truth"],
        "candidate_answers": reward.get("candidate_answers", []),
        "image": image_record,
        "metadata": metadata_record,
        "image_width": metadata.get("image_width"),
        "image_height": metadata.get("image_height"),
    }


def initial_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"{DIRECT_INSTRUCTION}\nQuestion: {sample['question']}\nImage: ",
            },
            {
                "type": "image",
                "image": image_to_data_uri(Path(sample["image"]["path"])),
                "max_pixels": MAX_PIXELS,
            },
        ],
    }]


def execute_sample(
    sample: dict[str, Any],
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats()
    case_started = time.monotonic()
    response, input_tokens, output_tokens, generation_seconds = generate_response(
        model,
        processor,
        initial_messages(sample),
        MAX_NEW_TOKENS,
    )
    answer, parser_mode = parse_answer(response)
    exact = strict_em(answer, sample["ground_truth"])
    return {
        "schema_version": "mmsearch.step12.base-prediction.v1",
        "status": "answered" if answer is not None else "empty_response",
        "eval_index": sample["eval_index"],
        "data_id": sample["data_id"],
        "category": sample["category"],
        "source_row_index": sample["source_row_index"],
        "selection_rank_sha256": sample["selection_rank_sha256"],
        "model": {
            "repo_id": MODEL_REPO,
            "revision": MODEL_REVISION,
            "local_dir": str(MODEL_DIR),
        },
        "execution": {
            "mode": "direct_answer_no_tools",
            "external_tool_calls": 0,
            "image_search_calls": 0,
            "text_search_calls": 0,
            "controller_intervention": False,
            "candidate_answers_provided_to_model": False,
            "ground_truth_provided_to_model": False,
        },
        "decoding": {
            "seed": 0,
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "torch_dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "max_pixels": MAX_PIXELS,
        },
        "input": {
            "image": sample["image"],
            "metadata": sample["metadata"],
            "image_width": sample["image_width"],
            "image_height": sample["image_height"],
            "question": sample["question"],
        },
        "result": {
            "response_text": response,
            "answer": answer,
            "parser_mode": parser_mode,
            "ground_truth": sample["ground_truth"],
            "strict_exact_match": exact,
            "strict_exact_match_definition": "prediction.strip().lower() == ground_truth.strip().lower()",
        },
        "runtime": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generation_seconds": round(generation_seconds, 6),
            "case_seconds": round(time.monotonic() - case_started, 6),
            "cuda_memory_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_memory_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "credentials_recorded": False,
    }


def prediction_path(sample: dict[str, Any]) -> Path:
    return OUTPUT_ROOT / "predictions" / f"{sample['eval_index']:03d}_{sample['data_id']}.json"


def initial_state(static: dict[str, Any], service_root_pid: int) -> dict[str, Any]:
    return {
        "schema_version": "mmsearch.step12.base-state.v1",
        "status": "in_progress",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "completed_count": 0,
        "predictions": [],
        "stages": {},
        "static": static,
        "summary_service_root_pid": service_root_pid,
    }


def prepare_state(target: int, static: dict[str, Any], service_root_pid: int) -> dict[str, Any]:
    if not OUTPUT_ROOT.exists():
        require(target == 5, "fresh Step 12 Base run must begin at target 5")
        OUTPUT_ROOT.mkdir(mode=0o700, parents=False, exist_ok=False)
        (OUTPUT_ROOT / "predictions").mkdir(mode=0o700)
        state = initial_state(static, service_root_pid)
        evidence.atomic_write_json(STATE_PATH, state)
        return state
    require(OUTPUT_ROOT.is_dir() and not OUTPUT_ROOT.is_symlink(), "Base output root invalid")
    state, _ = load_json(STATE_PATH)
    require(state.get("schema_version") == "mmsearch.step12.base-state.v1", "state schema mismatch")
    require(state.get("static") == static, "state static input/runner binding mismatch")
    require(state.get("summary_service_root_pid") == service_root_pid, "Qwen3 service root PID changed")
    completed = state.get("completed_count")
    require(isinstance(completed, int) and 0 <= completed < target, "target does not extend checkpoint")
    require(len(state.get("predictions", [])) == completed, "state prediction count mismatch")
    for index, record in enumerate(state["predictions"], start=1):
        require(record.get("eval_index") == index, f"state prediction order mismatch at {index}")
        verify_record(record, f"prediction {index}")
    if target == 20:
        require("5" in state.get("stages", {}), "stage 5 checkpoint missing")
    if target == 50:
        require("20" in state.get("stages", {}), "stage 20 checkpoint missing")
    require("50" not in state.get("stages", {}), "stage 50 already completed")
    return state


def load_results(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [load_json(Path(record["path"]), 32 * 1024 * 1024)[0] for record in state["predictions"]]


def rounded_percent(numerator: int | float, denominator: int | float) -> float:
    return round(numerator / denominator * 100, 4)


def aggregate(results: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    require(results, "cannot aggregate zero Base predictions")
    n = len(results)
    correct = sum(item["result"]["strict_exact_match"] is True for item in results)
    category_metrics: dict[str, dict[str, Any]] = {}
    for category in ("search_free", "search_required"):
        selected = [item for item in results if item["category"] == category]
        category_correct = sum(item["result"]["strict_exact_match"] is True for item in selected)
        category_metrics[category] = {
            "count": len(selected),
            "correct": category_correct,
            "accuracy_percent": rounded_percent(category_correct, len(selected)),
            "search_calls": 0,
            "search_ratio_percent": 0.0,
            "average_generation_seconds": round(sum(item["runtime"]["generation_seconds"] for item in selected) / len(selected), 6),
        }
    generation_times = [item["runtime"]["generation_seconds"] for item in results]
    metrics = {
        "schema_version": "mmsearch.step12.base-metrics.v1",
        "evaluated": n,
        "correct": correct,
        "accuracy_percent": rounded_percent(correct, n),
        "categories": category_metrics,
        "image_search_calls": 0,
        "text_search_calls": 0,
        "total_search_calls": 0,
        "search_ratio_percent": 0.0,
        "average_generation_seconds": round(sum(generation_times) / n, 6),
        "median_generation_seconds": round(statistics.median(generation_times), 6),
        "total_generation_seconds": round(sum(generation_times), 6),
        "total_input_tokens": sum(item["runtime"]["input_tokens"] for item in results),
        "total_output_tokens": sum(item["runtime"]["output_tokens"] for item in results),
        "parser_mode_counts": dict(sorted(collections.Counter(item["result"]["parser_mode"] for item in results).items())),
        "empty_response_count": sum(item["status"] != "answered" for item in results),
        "strict_exact_match_definition": "prediction.strip().lower() == ground_truth.strip().lower()",
        "execution_mode": "direct_answer_no_tools",
        "external_tool_calls": 0,
    }
    failures = [
        {
            "eval_index": item["eval_index"],
            "data_id": item["data_id"],
            "category": item["category"],
            "answer": item["result"]["answer"],
            "ground_truth": item["result"]["ground_truth"],
            "parser_mode": item["result"]["parser_mode"],
            "layer": "base_final_answer" if item["status"] == "answered" else "base_empty_response",
            "rationale": "Base emitted a direct answer but strict Exact Match is false; no upstream tool layer exists." if item["status"] == "answered" else "Base emitted no non-empty decoded answer.",
        }
        for item in results
        if item["result"]["strict_exact_match"] is False
    ]
    failure_summary = {
        "schema_version": "mmsearch.step12.base-failure-summary.v1",
        "evaluated": n,
        "failure_count": len(failures),
        "failure_rate_percent": rounded_percent(len(failures), n),
        "failure_layer_counts": dict(sorted(collections.Counter(item["layer"] for item in failures).items())),
        "failures": failures,
        "credentials_recorded": False,
    }
    return metrics, failure_summary


def write_jsonl(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [json.dumps(evidence.sanitize_public(item), ensure_ascii=False, sort_keys=True, separators=(",", ":")) for item in results]
    evidence.atomic_write_text(path, "\n".join(lines) + "\n")


def commit_stage(
    target: int,
    state: dict[str, Any],
    static: dict[str, Any],
    results: list[dict[str, Any]],
    service_root_pid: int,
    service_gpu_identities: set[str],
    model_residency: dict[str, str],
    load_seconds: float,
) -> dict[str, Any]:
    require(len(results) == target, "stage result count mismatch")
    metrics, failure_summary = aggregate(results)
    predictions_path = OUTPUT_ROOT / f"predictions_{target}.jsonl"
    metrics_path = OUTPUT_ROOT / f"metrics_{target}.json"
    failure_path = OUTPUT_ROOT / f"failure_summary_{target}.json"
    write_jsonl(predictions_path, results)
    evidence.atomic_write_json(metrics_path, metrics)
    evidence.atomic_write_json(failure_path, failure_summary)
    stage_files = [predictions_path, metrics_path, failure_path]
    if target == 50:
        write_jsonl(OUTPUT_ROOT / "predictions.jsonl", results)
        evidence.atomic_write_json(OUTPUT_ROOT / "metrics.json", metrics)
        evidence.atomic_write_json(OUTPUT_ROOT / "failure_summary.json", failure_summary)
        stage_files.extend((OUTPUT_ROOT / "predictions.jsonl", OUTPUT_ROOT / "metrics.json", OUTPUT_ROOT / "failure_summary.json"))
    prediction_paths = [Path(record["path"]) for record in state["predictions"]]
    scan = evidence.scan_serialized_evidence(prediction_paths + [metrics_path, failure_path], [predictions_path])
    require(scan.get("pass") is True, "Base stage serialized evidence scan failed")
    snapshot = gpu_snapshot(
        f"step12_base_stage_{target}_complete_joint_gpu",
        service_root_pid=service_root_pid,
        current_pid=os.getpid(),
    )
    validate_joint_snapshot(snapshot, service_gpu_identities=service_gpu_identities, current_pid=os.getpid())
    manifest = {
        "schema_version": "mmsearch.step12.base-stage.v1",
        "status": "passed",
        "stage_target": target,
        "completed_at_utc": utc_now(),
        "static": static,
        "prediction_count": target,
        "prediction_records": state["predictions"],
        "artifacts": [evidence.file_record(path) for path in stage_files],
        "metrics": metrics,
        "failure_summary": evidence.file_record(failure_path),
        "model_load_seconds": round(load_seconds, 6),
        "base_fully_gpu_resident": True,
        "base_hf_device_map": model_residency,
        "gpu_snapshot": snapshot,
        "unexpected_external_tool_calls": 0,
        "offline_environment": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
        "serialized_evidence_scan": scan,
        "credentials_recorded": False,
    }
    stage_path = OUTPUT_ROOT / f"stage_{target}_manifest.json"
    evidence.atomic_write_json(stage_path, manifest)
    state["stages"][str(target)] = evidence.file_record(stage_path)
    state["status"] = "passed_stage_50" if target == 50 else "in_progress"
    state["updated_at_utc"] = utc_now()
    evidence.atomic_write_json(STATE_PATH, state)
    if target == 50:
        completion = {
            "schema_version": "mmsearch.step12.base-completion.v1",
            "status": "passed",
            "step": 12,
            "scope": "Base Direct Answer sub-experiment",
            "completed_at_utc": utc_now(),
            "completed_examples": 50,
            "static": static,
            "stages": state["stages"],
            "final_predictions": evidence.file_record(OUTPUT_ROOT / "predictions.jsonl"),
            "final_metrics": metrics,
            "final_failure_summary": evidence.file_record(OUTPUT_ROOT / "failure_summary.json"),
            "state": evidence.file_record(STATE_PATH),
            "base_fully_gpu_resident": True,
            "qwen3_service_root_pid_unchanged": service_root_pid,
            "unexpected_external_tool_calls": 0,
            "credentials_recorded": False,
        }
        evidence.atomic_write_json(OUTPUT_ROOT / "step12_base_completion_manifest.json", completion)
    return manifest


def run(target: int) -> dict[str, Any]:
    require(target in STAGES, "target must be one of 5, 20, 50")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "CUDA_VISIBLE_DEVICES must be pinned to 0")
    require(os.environ.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE must be 1")
    require(os.environ.get("TRANSFORMERS_OFFLINE") == "1", "TRANSFORMERS_OFFLINE must be 1")
    torch.manual_seed(0)
    static, examples = validate_static_inputs()
    service_root_pid = read_summary_service_pid()
    service_snapshot = gpu_snapshot(
        "step12_qwen3_service_only_before_base_load",
        service_root_pid=service_root_pid,
        current_pid=os.getpid(),
    )
    service_gpu_identities = validate_service_only_snapshot(service_snapshot)
    state = prepare_state(target, static, service_root_pid)
    load_started = time.monotonic()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model.eval()
    load_seconds = time.monotonic() - load_started
    model_residency = validate_mmsearch_residency(model)
    after_load = gpu_snapshot(
        "step12_qwen3_plus_base_loaded",
        service_root_pid=service_root_pid,
        current_pid=os.getpid(),
    )
    validate_joint_snapshot(after_load, service_gpu_identities=service_gpu_identities, current_pid=os.getpid())
    results = load_results(state)
    try:
        for index in range(state["completed_count"] + 1, target + 1):
            sample = load_sample(examples[index - 1])
            result = execute_sample(sample, model, processor)
            path = prediction_path(sample)
            evidence.atomic_write_json(path, result)
            record = evidence.file_record(path)
            record["eval_index"] = index
            state["predictions"].append(record)
            state["completed_count"] = index
            state["updated_at_utc"] = utc_now()
            evidence.atomic_write_json(STATE_PATH, state)
            results.append(result)
    except BaseException as exc:
        failure = {
            "schema_version": "mmsearch.step12.base-run-failure.v1",
            "status": "failed_global_hard_stop",
            "failed_at_utc": utc_now(),
            "target": target,
            "completed_count": state.get("completed_count"),
            "error": evidence.safe_error(exc),
            "static": static,
            "credentials_recorded": False,
        }
        evidence.atomic_write_json(OUTPUT_ROOT / "step12_base_run_failure.json", failure)
        raise
    require(read_summary_service_pid() == service_root_pid, "Qwen3 service PID changed during Base stage")
    return commit_stage(
        target,
        state,
        static,
        results,
        service_root_pid,
        service_gpu_identities,
        model_residency,
        load_seconds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, choices=STAGES)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    require(parse_answer("<answer> Cheshire </answer>") == ("Cheshire", "answer_tags"), "tag parser self-test failed")
    require(parse_answer("plain") == ("plain", "plain_text_fallback"), "fallback parser self-test failed")
    require(parse_answer("  ") == (None, "empty_response"), "empty parser self-test failed")
    require(strict_em(" Answer ", "answer"), "strict EM positive self-test failed")
    require(not strict_em("two  spaces", "two spaces"), "strict EM whitespace self-test failed")
    dummy = []
    for index, (category, exact) in enumerate((("search_free", True), ("search_required", False)), start=1):
        dummy.append({
            "status": "answered",
            "eval_index": index,
            "data_id": f"id_{index}",
            "category": category,
            "result": {"strict_exact_match": exact, "answer": "a", "ground_truth": "a" if exact else "b", "parser_mode": "answer_tags"},
            "runtime": {"generation_seconds": float(index), "input_tokens": 10, "output_tokens": 2},
        })
    metrics, failures = aggregate(dummy)
    require(metrics["accuracy_percent"] == 50.0 and metrics["search_ratio_percent"] == 0.0, "metric self-test failed")
    require(failures["failure_count"] == 1, "failure summary self-test failed")
    print(json.dumps({"status": "passed", "pure_self_tests": 7}))


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    require(args.target is not None, "--target is required outside self-test")
    manifest = run(args.target)
    print(json.dumps({
        "status": manifest["status"],
        "stage": manifest["stage_target"],
        "completed_count": manifest["prediction_count"],
        "accuracy_percent": manifest["metrics"]["accuracy_percent"],
        "search_ratio_percent": manifest["metrics"]["search_ratio_percent"],
        "stage_manifest": str(OUTPUT_ROOT / f"stage_{args.target}_manifest.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
