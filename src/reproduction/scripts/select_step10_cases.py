#!/usr/bin/env python3
"""Deterministically select Step-10 B/C/failure cases with offline placeholders.

This scanner deliberately does not select the pre-fixed Case A (fvqa_train_0)
or Case D (fvqa_train_17). It scans only official FVQA train rows whose
``category`` is ``search_required``, in parquet order, and persists every
evaluated record in one atomic selection manifest.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

# Make every Hugging Face access fail closed before importing Transformers.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pyarrow.parquet as pq
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import placeholder_control_flow as placeholder


SCHEMA_VERSION = "mmsearch.step10.case-selection.v1"
EXPECTED_PARQUET = Path("/root/autodl-tmp/datasets/FVQA/fvqa_train.parquet")
EXPECTED_MODEL = Path("/root/autodl-tmp/models/MMSearch-R1-7B")
DATASET_REPOSITORY = "lmms-lab/FVQA"
DATASET_REVISION = "bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5"
MODEL_REPOSITORY = "lmms-lab/MMSearch-R1-7B"
MODEL_REVISION = "3cdec93e6db79a409aff4a4b2eadc77a5a8a1e46"

SOURCE_SPLIT = "train"
REQUIRED_CATEGORY = "search_required"
MAX_SEARCH_REQUIRED_CANDIDATES = 256
SEED = 0
DO_SAMPLE = False
MAX_NEW_TOKENS = 512
MAX_ROUNDS = 3
IMAGE_SEARCH_LIMIT = 1
TEXT_SEARCH_LIMIT = 1

PRE_FIXED_CASE_A = "fvqa_train_0"
PRE_FIXED_CASE_D = "fvqa_train_17"
PRE_FIXED_IDS = {PRE_FIXED_CASE_A, PRE_FIXED_CASE_D}

CASE_B_SEQUENCE = ["image_search", "answer"]
CASE_C_SEQUENCE = ["text_search", "answer"]

# These hashes pin the already-reviewed, offline placeholder implementation and
# the exact prompts used for selection. A mismatch aborts before model loading.
PINNED_SOURCE_SHA256 = {
    "reproduction/env/huggingface_revisions.json": (
        "cb06add420730746cca627729fb39e8d14ba9e7d3cf5dae9c4cec2a177c2635b"
    ),
    "reproduction/scripts/placeholder_control_flow.py": (
        "318a58a837aaab0e161044161de595fe23dab5468c234ff7c940b367c9e204d5"
    ),
    "mmsearch_r1/utils/tools/image_search.py": (
        "91d4f843b638711e9dc4fc9dc8d5d2e329d354fb4fcf0542f1dc1ca81ba8f339"
    ),
    "mmsearch_r1/utils/tools/text_search.py": (
        "4587ae40d48c5c126f4e8ad2b8d0d7da9662b6c6d0e44bea72813fe7ea361635"
    ),
    "mmsearch_r1/prompts/round_1_user_prompt_qwenvl.pkl": (
        "c1296c0d44e18f8367e4702d0bee66124db5e579108d5b35257595431113e41d"
    ),
    "mmsearch_r1/prompts/after_image_search_prompt_qwenvl.pkl": (
        "60b1c910cd2f80b2961bf15230a4f819f2c79b367080e8f5f6ab95a979281399"
    ),
    "mmsearch_r1/prompts/after_text_search_prompt_qwenvl.pkl": (
        "0a0490c2bab021a8d457934c27f90186de45fadfe530dc48c17444e4da137b04"
    ),
}

ARTIFACT_BASENAMES = {
    "case_b": "case_b_image_search",
    "case_c": "case_c_text_search",
    "failure": "failure_case",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select deterministic Step-10 B/C/failure cases from the pinned "
            "FVQA train parquet with offline official placeholder tools."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def json_clean(value: Any) -> Any:
    """Return a JSON-compatible copy without retaining custom objects."""

    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace one file after fsyncing its contents and directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        os.chmod(temporary_path, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, value: Any) -> tuple[int, str]:
    payload = json_bytes(value)
    atomic_write_bytes(path, payload)
    return len(payload), sha256_bytes(payload)


def require_absolute(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {path}")


def validate_paths(args: argparse.Namespace) -> dict[str, Path]:
    for label, path in (
        ("--parquet", args.parquet),
        ("--model-path", args.model_path),
        ("--output-dir", args.output_dir),
        ("--manifest", args.manifest),
    ):
        require_absolute(path, label)

    parquet = args.parquet.resolve(strict=True)
    model_path = args.model_path.resolve(strict=True)
    output_dir = args.output_dir.resolve(strict=False)
    manifest = args.manifest.resolve(strict=False)

    if parquet != EXPECTED_PARQUET.resolve(strict=True):
        raise ValueError(f"--parquet is not the pinned FVQA train parquet: {parquet}")
    if model_path != EXPECTED_MODEL.resolve(strict=True):
        raise ValueError(f"--model-path is not the pinned MMSearch model: {model_path}")
    if not parquet.is_file():
        raise ValueError(f"Pinned parquet is not a file: {parquet}")
    if not model_path.is_dir():
        raise ValueError(f"Pinned model is not a directory: {model_path}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"--output-dir exists but is not a directory: {output_dir}")
    if manifest.exists() and manifest.is_dir():
        raise ValueError(f"--manifest is a directory: {manifest}")
    if manifest.suffix.lower() != ".json":
        raise ValueError("--manifest must end in .json")
    if output_dir == model_path or model_path in output_dir.parents:
        raise ValueError("--output-dir must not be inside the pinned model directory")
    if output_dir == parquet.parent or parquet.parent in output_dir.parents:
        raise ValueError("--output-dir must not be inside the pinned dataset directory")

    artifact_paths = {
        output_dir / f"{basename}.{suffix}"
        for basename in ARTIFACT_BASENAMES.values()
        for suffix in ("png", "json")
    }
    if manifest in artifact_paths:
        raise ValueError("--manifest collides with a selected sample artifact")

    return {
        "parquet": parquet,
        "model_path": model_path,
        "output_dir": output_dir,
        "manifest": manifest,
    }


def verify_source_pins(model_path: Path) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for relative_path, expected_sha256 in PINNED_SOURCE_SHA256.items():
        path = REPO_ROOT / relative_path
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Pinned source changed: {relative_path}; "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        evidence[relative_path] = {
            "bytes": path.stat().st_size,
            "sha256": actual_sha256,
        }

    revision_manifest_path = REPO_ROOT / "reproduction/env/huggingface_revisions.json"
    revision_manifest = json.loads(revision_manifest_path.read_text(encoding="utf-8"))
    expected_model = revision_manifest.get("model", {})
    expected_dataset = revision_manifest.get("dataset", {})
    if expected_model.get("repo_id") != MODEL_REPOSITORY:
        raise RuntimeError("Pinned model repository does not match revision manifest")
    if expected_model.get("revision") != MODEL_REVISION:
        raise RuntimeError("Pinned model revision does not match revision manifest")
    if Path(expected_model.get("local_dir", "")).resolve() != model_path:
        raise RuntimeError("Pinned model path does not match revision manifest")
    if expected_dataset.get("repo_id") != DATASET_REPOSITORY:
        raise RuntimeError("Pinned dataset repository does not match revision manifest")
    if expected_dataset.get("revision") != DATASET_REVISION:
        raise RuntimeError("Pinned dataset revision does not match revision manifest")

    model_config = model_path / "config.json"
    evidence["model/config.json"] = {
        "path": str(model_config),
        "bytes": model_config.stat().st_size,
        "sha256": sha256_file(model_config),
    }
    return evidence


def question_from_prompt(prompt: list[dict[str, Any]]) -> str:
    for message in prompt:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    raise ValueError("FVQA row has no string user question")


def parse_candidate_answers(reward_model: dict[str, Any]) -> list[str]:
    value = reward_model.get("candidate_answers", [])
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise TypeError("reward_model.candidate_answers must be a JSON list or list")
    if not all(isinstance(item, str) for item in value):
        raise TypeError("Every candidate answer must be a string")
    return list(value)


def png_bytes(raw_image: bytes) -> tuple[bytes, int, int]:
    with Image.open(BytesIO(raw_image)) as image:
        image.load()
        image = image.convert("RGB")
        width, height = image.size
        buffer = BytesIO()
        image.save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue(), width, height


def initial_messages(
    round_1_prompt: str,
    question: str,
    normalized_png: bytes,
) -> list[dict[str, Any]]:
    encoded = base64.b64encode(normalized_png).decode("ascii")
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{round_1_prompt}\nQuestion: {question}\nImage: ",
                },
                {
                    "type": "image",
                    "image": f"data:image/png;base64,{encoded}",
                    "max_pixels": placeholder.MAX_PIXELS,
                },
            ],
        }
    ]


def search_required_rows(
    parquet_path: Path,
) -> Iterator[tuple[int, int, dict[str, Any]]]:
    columns = [
        "prompt",
        "images",
        "reward_model",
        "data_source",
        "data_id",
        "category",
    ]
    source_row_index = 0
    search_required_number = 0
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(batch_size=64, columns=columns):
        for row in batch.to_pylist():
            current_row_index = source_row_index
            source_row_index += 1
            if row.get("category") != REQUIRED_CATEGORY:
                continue
            search_required_number += 1
            yield search_required_number, current_row_index, row


def normalize_answer(value: str) -> str:
    # The experiment documents define strict EM as strip + lowercase only;
    # internal whitespace must remain significant.
    return value.strip().lower()


def run_placeholder_flow(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    *,
    candidate_number: int,
    source_row_index: int,
    row: dict[str, Any],
    normalized_png: bytes,
    raw_image_sha256: str,
    normalized_png_sha256: str,
    raw_image_bytes: int,
    normalized_png_bytes: int,
    width: int,
    height: int,
    round_1_prompt: str,
    after_image_search_prompt: str,
    after_text_search_prompt: str,
) -> dict[str, Any]:
    data_id = row.get("data_id")
    if not isinstance(data_id, str) or not data_id:
        raise ValueError(f"Invalid data_id at parquet row {source_row_index}")
    if row.get("category") != REQUIRED_CATEGORY:
        raise ValueError(f"Unexpected category for {data_id}: {row.get('category')}")

    reward_model = row.get("reward_model")
    if not isinstance(reward_model, dict):
        raise TypeError(f"{data_id} has invalid reward_model")
    ground_truth = reward_model.get("ground_truth")
    if not isinstance(ground_truth, str):
        raise TypeError(f"{data_id} has non-string ground truth")

    question = question_from_prompt(row.get("prompt") or [])
    messages = initial_messages(round_1_prompt, question, normalized_png)
    rounds: list[dict[str, Any]] = []
    action_sequence: list[str] = []
    image_search_calls = 0
    text_search_calls = 0
    final_answer: str | None = None
    terminal_status: str | None = None

    for round_number in range(1, MAX_ROUNDS + 1):
        response, input_tokens, output_tokens, generation_seconds = (
            placeholder.generate_response(
                model,
                processor,
                messages,
                MAX_NEW_TOKENS,
            )
        )
        action, payload = placeholder.classify_response(response)
        action_sequence.append(action)
        round_record: dict[str, Any] = {
            "round": round_number,
            "raw_response": response,
            "has_reason": bool(
                re.search(r"<reason>.*?</reason>", response, flags=re.DOTALL)
            ),
            "action": action,
            "action_payload": payload,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generation_seconds": round(generation_seconds, 6),
        }
        rounds.append(round_record)

        if action == "answer":
            final_answer = payload
            terminal_status = "answered"
            break
        if action == "warning":
            terminal_status = "warning"
            break
        if action == "invalid":
            terminal_status = "invalid_response"
            break

        # Match the corrected official placeholder conversation structure:
        # assistant tool action first, then the synthetic user tool result.
        messages.append(placeholder.assistant_message(response))

        if action == "image_search":
            if image_search_calls >= IMAGE_SEARCH_LIMIT:
                terminal_status = "image_search_limit"
                break
            _, returned_images, tool_stat = placeholder.call_image_search(
                image_url=f"fvqa://train/{data_id}"
            )
            image_search_calls += 1
            titles = tool_stat.get(
                "titles",
                [
                    f"Webpage Title {index + 1}"
                    for index in range(len(returned_images))
                ],
            )
            round_record["tool"] = {
                "type": "official_placeholder_image_search",
                "offline_synthetic": True,
                "status": json_clean(tool_stat),
                "titles": list(titles),
                "returned_images": len(returned_images),
                "returned_body_persisted": False,
            }
            messages.append(
                placeholder.image_search_result_message(
                    returned_images,
                    list(titles),
                    question,
                    after_image_search_prompt,
                )
            )
            continue

        if action == "text_search":
            if text_search_calls >= TEXT_SEARCH_LIMIT:
                terminal_status = "text_search_limit"
                break
            query = payload or ""
            returned_text, tool_stat = placeholder.call_text_search(query)
            text_search_calls += 1
            round_record["query"] = query
            round_record["tool"] = {
                "type": "official_placeholder_text_search",
                "offline_synthetic": True,
                "status": json_clean(tool_stat),
                "returned_body_persisted": False,
            }
            messages.append(
                placeholder.text_search_result_message(
                    returned_text,
                    question,
                    after_text_search_prompt,
                )
            )
            continue

        terminal_status = "invalid_response"
        break
    else:
        terminal_status = "max_rounds"

    exact_match = (
        terminal_status == "answered"
        and final_answer is not None
        and normalize_answer(final_answer) == normalize_answer(ground_truth)
    )
    return {
        "candidate_number": candidate_number,
        "source_row_index": source_row_index,
        "source_split": SOURCE_SPLIT,
        "data_id": data_id,
        "category": REQUIRED_CATEGORY,
        "data_source": row.get("data_source"),
        "question": question,
        "ground_truth": ground_truth,
        "candidate_answers": parse_candidate_answers(reward_model),
        "reward_style": reward_model.get("style"),
        "image": {
            "source_image_sha256": raw_image_sha256,
            "normalized_png_sha256": normalized_png_sha256,
            "source_bytes": raw_image_bytes,
            "normalized_png_bytes": normalized_png_bytes,
            "width": width,
            "height": height,
        },
        "rounds": rounds,
        "action_sequence": action_sequence,
        "image_search_calls": image_search_calls,
        "text_search_calls": text_search_calls,
        "total_turns": len(rounds),
        "terminal_status": terminal_status,
        "final_answer": final_answer,
        "exact_match": exact_match,
        "selection_evaluation": {},
        "not_selected_reasons": [],
    }


def snapshot(record: dict[str, Any], normalized_png: bytes) -> dict[str, Any]:
    return {
        "record": record,
        "normalized_png": normalized_png,
    }


def choose_independent_failure(
    failure_pool: list[dict[str, Any]],
    selected_b: dict[str, Any] | None,
    selected_c: dict[str, Any] | None,
) -> dict[str, Any] | None:
    excluded = set(PRE_FIXED_IDS)
    if selected_b is not None:
        excluded.add(selected_b["record"]["data_id"])
    if selected_c is not None:
        excluded.add(selected_c["record"]["data_id"])
    for candidate in failure_pool:
        if candidate["record"]["data_id"] not in excluded:
            return candidate
    return None


def annotate_selection_evaluations(
    records: list[dict[str, Any]],
    selected_b: dict[str, Any] | None,
    selected_c: dict[str, Any] | None,
    selected_failure: dict[str, Any] | None,
) -> None:
    selected_b_id = selected_b["record"]["data_id"] if selected_b else None
    selected_c_id = selected_c["record"]["data_id"] if selected_c else None
    selected_failure_id = (
        selected_failure["record"]["data_id"] if selected_failure else None
    )

    for record in records:
        data_id = record["data_id"]
        action_sequence = record["action_sequence"]
        pre_fixed = data_id in PRE_FIXED_IDS

        if pre_fixed:
            case_b_reason = "pre_fixed_case_excluded_from_scanner_selection"
        elif action_sequence != CASE_B_SEQUENCE:
            case_b_reason = "action_sequence_not_exact_image_search_answer"
        elif data_id == selected_b_id:
            case_b_reason = "selected_first_exact_image_search_answer"
        elif selected_b_id is not None:
            case_b_reason = "earlier_case_b_already_selected"
        else:
            case_b_reason = "eligible_but_case_b_not_selected"

        if pre_fixed:
            case_c_reason = "pre_fixed_case_excluded_from_scanner_selection"
        elif action_sequence != CASE_C_SEQUENCE:
            case_c_reason = "action_sequence_not_exact_text_search_answer"
        elif data_id == selected_c_id:
            case_c_reason = "selected_first_exact_text_search_answer"
        elif selected_c_id is not None:
            case_c_reason = "earlier_case_c_already_selected"
        else:
            case_c_reason = "eligible_but_case_c_not_selected"

        if pre_fixed:
            failure_reason = "pre_fixed_case_a_or_d_excluded"
        elif record["terminal_status"] != "answered":
            failure_reason = "terminal_status_not_answered"
        elif record["exact_match"]:
            failure_reason = "exact_match_true"
        elif data_id in {selected_b_id, selected_c_id}:
            failure_reason = "excluded_overlap_with_selected_case_b_or_c"
        elif data_id == selected_failure_id:
            failure_reason = "selected_first_independent_answered_em_false"
        elif selected_failure_id is not None:
            failure_reason = "earlier_independent_failure_already_selected"
        else:
            failure_reason = "eligible_but_independent_failure_not_selected"

        evaluation = {
            "case_b": {
                "selected": data_id == selected_b_id,
                "reason": case_b_reason,
            },
            "case_c": {
                "selected": data_id == selected_c_id,
                "reason": case_c_reason,
            },
            "independent_failure": {
                "selected": data_id == selected_failure_id,
                "reason": failure_reason,
            },
        }
        record["selection_evaluation"] = evaluation
        record["not_selected_reasons"] = [
            f"{role}:{details['reason']}"
            for role, details in evaluation.items()
            if not details["selected"]
        ]


def selection_summary(selected: dict[str, Any] | None, reason: str) -> Any:
    if selected is None:
        return None
    record = selected["record"]
    return {
        "data_id": record["data_id"],
        "candidate_number": record["candidate_number"],
        "source_row_index": record["source_row_index"],
        "action_sequence": record["action_sequence"],
        "terminal_status": record["terminal_status"],
        "exact_match": record["exact_match"],
        "selection_reason": reason,
    }


def build_manifest(
    *,
    status: str,
    started_at_utc: str,
    paths: dict[str, Path],
    source_pins: dict[str, dict[str, Any]],
    model_info: dict[str, Any],
    records: list[dict[str, Any]],
    selected_b: dict[str, Any] | None,
    selected_c: dict[str, Any] | None,
    selected_failure: dict[str, Any] | None,
    artifacts: dict[str, Any],
    error: dict[str, Any] | None,
) -> dict[str, Any]:
    missing = [
        role
        for role, selected in (
            ("case_b", selected_b),
            ("case_c", selected_c),
            ("independent_failure", selected_failure),
        )
        if selected is None
    ]
    selected_ids = [
        selected["record"]["data_id"]
        for selected in (selected_b, selected_c, selected_failure)
        if selected is not None
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "started_at_utc": started_at_utc,
        "completed_at_utc": utc_now() if status != "running" else None,
        "purpose": "Step-10 deterministic Case B/Case C/independent Failure selection",
        "inputs": {
            "parquet": str(paths["parquet"]),
            "dataset_repository": DATASET_REPOSITORY,
            "dataset_revision": DATASET_REVISION,
            "model_path": str(paths["model_path"]),
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "output_dir": str(paths["output_dir"]),
            "manifest": str(paths["manifest"]),
        },
        "protocol": {
            "source_split": SOURCE_SPLIT,
            "required_official_category": REQUIRED_CATEGORY,
            "order": "original parquet row order",
            "maximum_search_required_candidates": MAX_SEARCH_REQUIRED_CANDIDATES,
            "seed": SEED,
            "do_sample": DO_SAMPLE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_rounds": MAX_ROUNDS,
            "image_search_limit": IMAGE_SEARCH_LIMIT,
            "text_search_limit": TEXT_SEARCH_LIMIT,
            "case_b_exact_action_sequence": CASE_B_SEQUENCE,
            "case_c_exact_action_sequence": CASE_C_SEQUENCE,
            "independent_failure": (
                "first terminal_status=answered and exact_match=false, excluding "
                "pre-fixed A/D and selected B/C data IDs"
            ),
            "pre_fixed_cases_not_selected": {
                "case_a": PRE_FIXED_CASE_A,
                "case_d": PRE_FIXED_CASE_D,
            },
            "tools": "pinned official offline placeholder implementations",
            "network_allowed": False,
            "credential_files_read": False,
            "raw_webpage_content_present": False,
            "exact_match_definition": (
                "final_answer.strip().lower() == "
                "reward_model.ground_truth.strip().lower()"
            ),
            "stop_condition": (
                "stop immediately after Case B, Case C, and independent Failure "
                "are all found; otherwise fail after 256 search_required rows"
            ),
        },
        "source_pins": source_pins,
        "model_runtime": model_info,
        "scanned_search_required_count": len(records),
        "selections": {
            "case_a": {
                "data_id": PRE_FIXED_CASE_A,
                "selection": "pre_fixed_outside_this_scanner",
            },
            "case_b": selection_summary(
                selected_b, "first exact [image_search, answer]"
            ),
            "case_c": selection_summary(
                selected_c, "first exact [text_search, answer]"
            ),
            "case_d": {
                "data_id": PRE_FIXED_CASE_D,
                "selection": "pre_fixed_outside_this_scanner",
            },
            "independent_failure": selection_summary(
                selected_failure,
                "first independent answered record with strict EM=false",
            ),
        },
        "missing_selections": missing,
        "selected_scan_ids_are_distinct": len(selected_ids) == len(set(selected_ids)),
        "artifacts": artifacts,
        "scan_records": records,
        "checks": {
            "only_train_search_required_scanned": all(
                record["source_split"] == SOURCE_SPLIT
                and record["category"] == REQUIRED_CATEGORY
                for record in records
            ),
            "parquet_order_preserved": all(
                left["source_row_index"] < right["source_row_index"]
                for left, right in zip(records, records[1:])
            ),
            "scan_limit_respected": len(records) <= MAX_SEARCH_REQUIRED_CANDIDATES,
            "case_b_exact": (
                selected_b is not None
                and selected_b["record"]["action_sequence"] == CASE_B_SEQUENCE
            ),
            "case_c_exact": (
                selected_c is not None
                and selected_c["record"]["action_sequence"] == CASE_C_SEQUENCE
            ),
            "failure_answered_em_false": (
                selected_failure is not None
                and selected_failure["record"]["terminal_status"] == "answered"
                and not selected_failure["record"]["exact_match"]
            ),
            "failure_independent": (
                selected_failure is not None
                and selected_failure["record"]["data_id"]
                not in {
                    PRE_FIXED_CASE_A,
                    PRE_FIXED_CASE_D,
                    selected_b["record"]["data_id"] if selected_b else None,
                    selected_c["record"]["data_id"] if selected_c else None,
                }
            ),
            "all_required_selections_found": not missing,
            "placeholder_bodies_not_persisted": all(
                all(
                    round_record.get("tool", {}).get("returned_body_persisted")
                    is not True
                    for round_record in record["rounds"]
                )
                for record in records
            ),
        },
        "error": error,
        "credentials_recorded": False,
    }


def write_selected_artifacts(
    output_dir: Path,
    selections: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    artifact_manifest: dict[str, Any] = {}
    for role, selected in selections.items():
        basename = ARTIFACT_BASENAMES[role]
        image_path = output_dir / f"{basename}.png"
        meta_path = output_dir / f"{basename}.json"
        png_payload = selected["normalized_png"]
        record = selected["record"]
        reason = {
            "case_b": "first exact [image_search, answer]",
            "case_c": "first exact [text_search, answer]",
            "failure": "first independent answered record with strict EM=false",
        }[role]
        metadata = {
            "schema_version": "mmsearch.step10.selected-sample.v1",
            "role": role,
            "selection_reason": reason,
            "image": str(image_path),
            "record": copy.deepcopy(record),
            "credentials_recorded": False,
            "raw_webpage_content_present": False,
        }
        metadata_payload = json_bytes(metadata)

        # Each artifact is independently atomic; the manifest is written last
        # and acts as the commit marker for the six selected files.
        atomic_write_bytes(image_path, png_payload)
        atomic_write_bytes(meta_path, metadata_payload)
        artifact_manifest[role] = {
            "data_id": record["data_id"],
            "image": {
                "path": str(image_path),
                "bytes": len(png_payload),
                "sha256": sha256_bytes(png_payload),
            },
            "sample_meta": {
                "path": str(meta_path),
                "bytes": len(metadata_payload),
                "sha256": sha256_bytes(metadata_payload),
            },
        }
    return artifact_manifest


def main() -> None:
    args = parse_args()
    paths = validate_paths(args)
    os.chdir(REPO_ROOT)
    started_at_utc = utc_now()
    source_pins: dict[str, dict[str, Any]] = {}
    model_info: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    selected_b: dict[str, Any] | None = None
    selected_c: dict[str, Any] | None = None
    selected_failure: dict[str, Any] | None = None
    failure_pool: list[dict[str, Any]] = []

    try:
        source_pins = verify_source_pins(paths["model_path"])
        if placeholder.IMAGE_SEARCH_LIMIT != IMAGE_SEARCH_LIMIT:
            raise RuntimeError("Reused placeholder image-search limit is no longer 1")
        if placeholder.TEXT_SEARCH_LIMIT != TEXT_SEARCH_LIMIT:
            raise RuntimeError("Reused placeholder text-search limit is no longer 1")

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the pinned MMSearch scanner")
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()

        round_1_prompt = placeholder.load_prompt(
            "round_1_user_prompt_qwenvl.pkl"
        ).replace("<image>", "").strip()
        after_image_search_prompt = placeholder.load_prompt(
            "after_image_search_prompt_qwenvl.pkl"
        )
        after_text_search_prompt = placeholder.load_prompt(
            "after_text_search_prompt_qwenvl.pkl"
        )

        load_started = time.monotonic()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            paths["model_path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(
            paths["model_path"],
            local_files_only=True,
            use_fast=False,
        )
        load_seconds = time.monotonic() - load_started
        parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
        model_info = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "attention_implementation": model.config._attn_implementation,
            "parameter_dtype": str(next(model.parameters()).dtype),
            "parameter_devices": parameter_devices,
            "hf_device_map": json_clean(getattr(model, "hf_device_map", None)),
            "load_seconds": round(load_seconds, 6),
        }

        seen_data_ids: set[str] = set()
        for candidate_number, source_row_index, row in search_required_rows(
            paths["parquet"]
        ):
            if candidate_number > MAX_SEARCH_REQUIRED_CANDIDATES:
                break
            images = row.get("images")
            if not images or not isinstance(images[0], dict):
                raise ValueError(f"Parquet row {source_row_index} has no embedded image")
            raw_image = images[0].get("bytes")
            if not isinstance(raw_image, bytes) or not raw_image:
                raise ValueError(f"Parquet row {source_row_index} has no image bytes")
            normalized_png, width, height = png_bytes(raw_image)
            record = run_placeholder_flow(
                model,
                processor,
                candidate_number=candidate_number,
                source_row_index=source_row_index,
                row=row,
                normalized_png=normalized_png,
                raw_image_sha256=sha256_bytes(raw_image),
                normalized_png_sha256=sha256_bytes(normalized_png),
                raw_image_bytes=len(raw_image),
                normalized_png_bytes=len(normalized_png),
                width=width,
                height=height,
                round_1_prompt=round_1_prompt,
                after_image_search_prompt=after_image_search_prompt,
                after_text_search_prompt=after_text_search_prompt,
            )
            if record["data_id"] in seen_data_ids:
                raise RuntimeError(f"Duplicate scanned data_id: {record['data_id']}")
            seen_data_ids.add(record["data_id"])
            records.append(record)
            current_snapshot = snapshot(record, normalized_png)

            if record["data_id"] not in PRE_FIXED_IDS:
                if selected_b is None and record["action_sequence"] == CASE_B_SEQUENCE:
                    selected_b = current_snapshot
                if selected_c is None and record["action_sequence"] == CASE_C_SEQUENCE:
                    selected_c = current_snapshot
                if record["terminal_status"] == "answered" and not record["exact_match"]:
                    failure_pool.append(current_snapshot)

            # Re-evaluate the earliest failure after each B/C selection so a
            # failure observed first can never overlap the final B or C ID.
            selected_failure = choose_independent_failure(
                failure_pool,
                selected_b,
                selected_c,
            )
            print(
                f"[scan {candidate_number}/{MAX_SEARCH_REQUIRED_CANDIDATES}] "
                f"{record['data_id']} actions={record['action_sequence']} "
                f"terminal={record['terminal_status']} em={record['exact_match']}"
            )
            if all(
                selected is not None
                for selected in (selected_b, selected_c, selected_failure)
            ):
                break

        annotate_selection_evaluations(
            records,
            selected_b,
            selected_c,
            selected_failure,
        )
        model_info["peak_gpu_memory_mib"] = round(
            torch.cuda.max_memory_allocated() / 1024**2,
            2,
        )

        missing = [
            role
            for role, selected in (
                ("case_b", selected_b),
                ("case_c", selected_c),
                ("independent_failure", selected_failure),
            )
            if selected is None
        ]
        if missing:
            manifest = build_manifest(
                status="failed_selection_not_found_within_limit",
                started_at_utc=started_at_utc,
                paths=paths,
                source_pins=source_pins,
                model_info=model_info,
                records=records,
                selected_b=selected_b,
                selected_c=selected_c,
                selected_failure=selected_failure,
                artifacts={},
                error={
                    "type": "SelectionIncomplete",
                    "message": (
                        "Required selections not found within the first "
                        f"{MAX_SEARCH_REQUIRED_CANDIDATES} search_required rows: {missing}"
                    ),
                },
            )
            atomic_write_json(paths["manifest"], manifest)
            print(
                f"[FAIL] missing selections {missing}; manifest={paths['manifest']}",
                file=sys.stderr,
            )
            raise SystemExit(3)

        assert selected_b is not None
        assert selected_c is not None
        assert selected_failure is not None
        artifacts = write_selected_artifacts(
            paths["output_dir"],
            {
                "case_b": selected_b,
                "case_c": selected_c,
                "failure": selected_failure,
            },
        )
        manifest = build_manifest(
            status="passed",
            started_at_utc=started_at_utc,
            paths=paths,
            source_pins=source_pins,
            model_info=model_info,
            records=records,
            selected_b=selected_b,
            selected_c=selected_c,
            selected_failure=selected_failure,
            artifacts=artifacts,
            error=None,
        )
        atomic_write_json(paths["manifest"], manifest)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "scanned": len(records),
                    "case_b": selected_b["record"]["data_id"],
                    "case_c": selected_c["record"]["data_id"],
                    "failure": selected_failure["record"]["data_id"],
                    "manifest": str(paths["manifest"]),
                },
                ensure_ascii=False,
            )
        )
    except SystemExit:
        raise
    except Exception as exc:
        annotate_selection_evaluations(
            records,
            selected_b,
            selected_c,
            selected_failure,
        )
        failure_manifest = build_manifest(
            status="failed_runtime_error",
            started_at_utc=started_at_utc,
            paths=paths,
            source_pins=source_pins,
            model_info=model_info,
            records=records,
            selected_b=selected_b,
            selected_c=selected_c,
            selected_failure=selected_failure,
            artifacts={},
            error={
                "type": type(exc).__name__,
                "message": str(exc)[:2000],
            },
        )
        atomic_write_json(paths["manifest"], failure_manifest)
        raise


if __name__ == "__main__":
    main()
