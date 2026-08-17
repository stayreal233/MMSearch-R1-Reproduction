#!/usr/bin/env python3
"""Strictly bridge the pinned Step-10 selector output into suite inputs.

This program performs no selection, model inference, network access, or
credential lookup.  It verifies the pre-registered protocol, the complete
selector commit manifest and artifacts, and the pinned FVQA parquet before it
writes direct sample objects accepted by ``step10_case_suite_qwen3.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SCHEMA = 1
SELECTOR_SCHEMA = "mmsearch.step10.case-selection.v1"
SELECTED_SAMPLE_SCHEMA = "mmsearch.step10.selected-sample.v1"
SUITE_SCHEMA = 1
BRIDGE_SCHEMA = "mmsearch.step10.suite-input-bridge.v1"

DATASET_ID = "lmms-lab/FVQA"
DATASET_REVISION = "bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5"
DATASET_SPLIT = "train"
MODEL_ID = "lmms-lab/MMSearch-R1-7B"
MODEL_REVISION = "3cdec93e6db79a409aff4a4b2eadc77a5a8a1e46"
PINNED_PARQUET = Path("/root/autodl-tmp/datasets/FVQA/fvqa_train.parquet")
PINNED_MODEL = Path("/root/autodl-tmp/models/MMSearch-R1-7B")
PINNED_PROTOCOL_SHA256 = "63a3ab7a753fb88ac1b923693f8e013052b14ae3e2b960dccf7f6a82095727bc"
PINNED_PARQUET_SHA256 = "d23be97f4493846381f71c6953a29777fe1522aaf37942a26393605ffd78171f"
FIXED_A = "fvqa_train_0"
FIXED_D = "fvqa_train_17"
FIXED_IDS = {FIXED_A, FIXED_D}
MAX_SCAN = 256
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

ROLE_CONFIG = {
    "B": {
        "selector_role": "case_b",
        "selector_basename": "case_b_image_search",
        "sequence": ["image_search", "answer"],
        "reason": "first exact [image_search, answer]",
    },
    "C": {
        "selector_role": "case_c",
        "selector_basename": "case_c_text_search",
        "sequence": ["text_search", "answer"],
        "reason": "first exact [text_search, answer]",
    },
    "Failure": {
        "selector_role": "failure",
        "selector_basename": "failure_case",
        "sequence": None,
        "reason": "first independent answered record with strict EM=false",
    },
}

OUTPUT_BASENAMES = {
    "A": "case_a",
    "B": "case_b",
    "C": "case_c",
    "D": "case_d",
    "Failure": "failure",
}

PINNED_SOURCE_SHA256 = {
    "reproduction/env/huggingface_revisions.json": "cb06add420730746cca627729fb39e8d14ba9e7d3cf5dae9c4cec2a177c2635b",
    "reproduction/scripts/placeholder_control_flow.py": "318a58a837aaab0e161044161de595fe23dab5468c234ff7c940b367c9e204d5",
    "mmsearch_r1/utils/tools/image_search.py": "91d4f843b638711e9dc4fc9dc8d5d2e329d354fb4fcf0542f1dc1ca81ba8f339",
    "mmsearch_r1/utils/tools/text_search.py": "4587ae40d48c5c126f4e8ad2b8d0d7da9662b6c6d0e44bea72813fe7ea361635",
    "mmsearch_r1/prompts/round_1_user_prompt_qwenvl.pkl": "c1296c0d44e18f8367e4702d0bee66124db5e579108d5b35257595431113e41d",
    "mmsearch_r1/prompts/after_image_search_prompt_qwenvl.pkl": "60b1c910cd2f80b2961bf15230a4f819f2c79b367080e8f5f6ab95a979281399",
    "mmsearch_r1/prompts/after_text_search_prompt_qwenvl.pkl": "0a0490c2bab021a8d457934c27f90186de45fadfe530dc48c17444e4da137b04",
}

EXPECTED_SELECTOR_PROTOCOL = {
    "source_split": "train",
    "required_official_category": "search_required",
    "order": "original parquet row order",
    "maximum_search_required_candidates": 256,
    "seed": 0,
    "do_sample": False,
    "max_new_tokens": 512,
    "max_rounds": 3,
    "image_search_limit": 1,
    "text_search_limit": 1,
    "case_b_exact_action_sequence": ["image_search", "answer"],
    "case_c_exact_action_sequence": ["text_search", "answer"],
    "independent_failure": (
        "first terminal_status=answered and exact_match=false, excluding "
        "pre-fixed A/D and selected B/C data IDs"
    ),
    "pre_fixed_cases_not_selected": {"case_a": FIXED_A, "case_d": FIXED_D},
    "tools": "pinned official offline placeholder implementations",
    "network_allowed": False,
    "credential_files_read": False,
    "raw_webpage_content_present": False,
    "exact_match_definition": (
        "final_answer.strip().lower() == "
        "reward_model.ground_truth.strip().lower()"
    ),
    "stop_condition": (
        "stop immediately after Case B, Case C, and independent Failure are all "
        "found; otherwise fail after 256 search_required rows"
    ),
}

EXPECTED_PROTOCOL_BODY = {
    "schema_version": 1,
    "status": "registered_before_step10_selection",
    "amended_before_selection_at_utc": "2026-08-16T14:12:12Z",
    "amendment": (
        "Clarified the documented strict EM formula before any candidate inference."
    ),
    "dataset": {
        "repo": DATASET_ID,
        "revision": DATASET_REVISION,
        "split": DATASET_SPLIT,
        "parquet": str(PINNED_PARQUET),
        "order": "physical parquet row order ascending",
        "candidate_category": "search_required",
        "maximum_search_required_candidates": 256,
    },
    "model": {
        "repo": MODEL_ID,
        "revision": MODEL_REVISION,
        "path": str(PINNED_MODEL),
        "seed": 0,
        "do_sample": False,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "max_new_tokens": 512,
        "max_rounds": 3,
        "image_search_limit": 1,
        "text_search_limit": 1,
    },
    "structural_scan": {
        "tools": "official repository placeholder image/text tools",
        "external_network_allowed": False,
        "record_every_scanned_candidate": True,
        "record_raw_model_responses": True,
        "stop_only_when_all_dynamic_slots_selected": True,
    },
    "case_rules": {
        "A_search_free": {
            "data_id": FIXED_A,
            "selection": "pre-fixed first official search_free train row",
            "required_final_action_sequence": ["answer"],
        },
        "B_image_search": {
            "selection": "first scanned candidate with exact placeholder action sequence, independent of exact match",
            "required_placeholder_action_sequence": ["image_search", "answer"],
            "required_final_action_sequence": ["image_search", "answer"],
        },
        "C_text_search": {
            "selection": "first scanned candidate with exact placeholder action sequence, independent of exact match",
            "required_placeholder_action_sequence": ["text_search", "answer"],
            "required_final_action_sequence": ["text_search", "answer"],
        },
        "D_mixed_search": {
            "data_id": FIXED_D,
            "selection": "pre-fixed by the prior recorded parquet-order scan and Step-9 Qwen3 trace",
            "required_final_action_sequence": ["image_search", "text_search", "answer"],
        },
        "failure": {
            "selection": "first scanned candidate in source order that terminates with answer and exact_match=false, excluding A, D, and the finally selected B/C IDs",
            "must_use_independent_data_id": True,
            "required_final_exact_match": False,
            "final_action_sequence": "recorded as observed; not selected by route",
        },
    },
    "selection_is_independent_of_correctness": True,
    "representative_case_exact_match_required": False,
    "exact_match_definition": (
        "prediction.strip().lower() == ground_truth.strip().lower()"
    ),
    "formal_run": {
        "top_k": 5,
        "jina_max_characters_per_page": 12000,
        "qwen3_max_tokens": 512,
        "qwen3_temperature": 0,
        "qwen3_seed": 0,
        "qwen3_thinking_enabled": False,
        "representative_route_mismatch_action": "stop_and_request_user_direction",
        "selected_failure_becomes_exact_match_action": "stop_and_request_user_direction",
        "tool_or_infrastructure_failure_action": "stop_and_request_user_direction",
    },
    "hard_stop_conditions": [
        "B, C, or an independent answered exact-match failure is absent after 256 candidates",
        "the fixed ordering, range, seed, or selection rules would need to change",
        "a formal representative route differs from its pre-registered required route",
        "the selected formal failure no longer has exact_match=false",
        "authentication, rate limit, tool, webpage, summarizer, CUDA, OOM, or service-health failure",
        "credential leakage or raw Jina webpage body in a formal artifact",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pinned-protocol", type=Path)
    parser.add_argument("--selector-manifest", type=Path)
    parser.add_argument("--selector-output-dir", type=Path)
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--suite-input-dir", type=Path)
    parser.add_argument("--suite-manifest", type=Path)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def read_regular_file(path: Path, *, maximum: int | None = None) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"cannot open regular file: {path}") from exc
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), f"not a regular file: {path}")
        require(info.st_size > 0, f"empty required file: {path}")
        if maximum is not None:
            require(info.st_size <= maximum, f"required file is too large: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_json(path: Path, *, maximum: int = 128 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    encoded = read_regular_file(path, maximum=maximum)
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON: {path}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value, encoded


def hash_file(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    total = 0
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), f"not a regular file: {path}")
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    finally:
        os.close(descriptor)
    return total, digest.hexdigest()


def file_record(path: Path, encoded: bytes | None = None) -> dict[str, Any]:
    if encoded is None:
        size, digest = hash_file(path)
    else:
        size, digest = len(encoded), sha256_bytes(encoded)
    return {"path": str(path.resolve(strict=True)), "bytes": size, "sha256": digest}


def strict_file(path: Path, label: str) -> Path:
    require(path.is_absolute(), f"{label} must be absolute")
    require(not path.is_symlink(), f"{label} must not be a symlink")
    resolved = path.resolve(strict=True)
    require(resolved.is_file(), f"{label} is not a file")
    return resolved


def strict_dir(path: Path, label: str) -> Path:
    require(path.is_absolute(), f"{label} must be absolute")
    require(not path.is_symlink(), f"{label} must not be a symlink")
    resolved = path.resolve(strict=True)
    require(resolved.is_dir(), f"{label} is not a directory")
    return resolved


def claimed_path(value: Any, label: str, *, strict: bool = True) -> Path:
    require(isinstance(value, str) and value, f"{label} path is missing")
    path = Path(value)
    require(path.is_absolute(), f"{label} path is not absolute")
    return path.resolve(strict=strict)


def parse_time(value: Any, label: str) -> datetime:
    require(isinstance(value, str) and value, f"{label} timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"{label} timestamp is invalid") from exc
    require(parsed.tzinfo is not None, f"{label} timestamp lacks timezone")
    return parsed


def atomic_write(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def normalized_png(raw_image: bytes) -> tuple[bytes, int, int]:
    with Image.open(BytesIO(raw_image)) as image:
        image.load()
        rgb = image.convert("RGB")
        width, height = rgb.size
        output = BytesIO()
        rgb.save(output, format="PNG", compress_level=6)
    return output.getvalue(), width, height


def question_from_prompt(prompt: Any) -> str:
    require(isinstance(prompt, list), "FVQA prompt is not a list")
    for message in prompt:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content:
                return content
    raise RuntimeError("FVQA row has no user question")


def candidate_answers(reward: dict[str, Any]) -> list[str]:
    value = reward.get("candidate_answers", [])
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError("candidate_answers is invalid JSON") from exc
    require(isinstance(value, list), "candidate_answers is not a list")
    require(all(isinstance(item, str) for item in value), "candidate answer is not a string")
    return list(value)  # The official Case A value is deliberately allowed to be [].


def row_image(row: dict[str, Any]) -> bytes:
    images = row.get("images")
    require(isinstance(images, list) and images and isinstance(images[0], dict), "row image is missing")
    value = images[0].get("bytes")
    require(isinstance(value, bytes) and value, "row image bytes are missing")
    return value


def validate_protocol(path: Path, encoded: bytes) -> tuple[dict[str, Any], datetime]:
    protocol = json.loads(encoded.decode("utf-8"))
    require(isinstance(protocol, dict), "pinned protocol root is invalid")
    require(sha256_bytes(encoded) == PINNED_PROTOCOL_SHA256, "pinned protocol SHA-256 mismatch")
    require(set(protocol) == set(EXPECTED_PROTOCOL_BODY) | {"registered_at_utc"}, "pinned protocol keys mismatch")
    body = {key: value for key, value in protocol.items() if key != "registered_at_utc"}
    require(body == EXPECTED_PROTOCOL_BODY, "pinned protocol content mismatch")
    registered = parse_time(protocol["registered_at_utc"], "protocol registration")
    amended = parse_time(
        protocol["amended_before_selection_at_utc"], "protocol amendment"
    )
    require(registered <= amended, "protocol amendment predates registration")
    return protocol, amended


def validate_source_pins(source_pins: Any) -> None:
    require(isinstance(source_pins, dict), "selector source_pins is missing")
    require(set(source_pins) == set(PINNED_SOURCE_SHA256) | {"model/config.json"}, "selector source pin keys mismatch")
    for relative, expected_digest in PINNED_SOURCE_SHA256.items():
        record = source_pins.get(relative)
        require(isinstance(record, dict), f"source pin is invalid: {relative}")
        source = (REPO_ROOT / relative).resolve(strict=True)
        size, digest = hash_file(source)
        require(digest == expected_digest, f"local pinned source changed: {relative}")
        require(record == {"bytes": size, "sha256": digest}, f"selector source pin mismatch: {relative}")
    model_record = source_pins["model/config.json"]
    require(isinstance(model_record, dict), "model config source pin is invalid")
    model_config = (PINNED_MODEL / "config.json").resolve(strict=True)
    size, digest = hash_file(model_config)
    require(model_record == {"path": str(model_config), "bytes": size, "sha256": digest}, "model config source pin mismatch")


def validate_scan_records(manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    records = manifest.get("scan_records")
    require(isinstance(records, list) and 0 < len(records) <= MAX_SCAN, "selector scan_records count is invalid")
    require(manifest.get("scanned_search_required_count") == len(records), "selector scanned count mismatch")
    seen: set[str] = set()
    previous_row = -1
    by_id: dict[str, dict[str, Any]] = {}
    for expected_number, record in enumerate(records, 1):
        require(isinstance(record, dict), "selector scan record is invalid")
        require(record.get("candidate_number") == expected_number, "candidate numbers are not contiguous")
        row_index = record.get("source_row_index")
        require(isinstance(row_index, int) and not isinstance(row_index, bool) and row_index > previous_row, "scan row order is invalid")
        previous_row = row_index
        require(record.get("source_split") == DATASET_SPLIT, "scan split mismatch")
        require(record.get("category") == "search_required", "scan category mismatch")
        final_answer = record.get("final_answer")
        ground_truth = record.get("ground_truth")
        strict_exact_match = (
            record.get("terminal_status") == "answered"
            and isinstance(final_answer, str)
            and isinstance(ground_truth, str)
            and final_answer.strip().lower() == ground_truth.strip().lower()
        )
        require(
            record.get("exact_match") is strict_exact_match,
            "scan exact_match does not match the pre-registered strict formula",
        )
        data_id = record.get("data_id")
        require(isinstance(data_id, str) and data_id and data_id not in seen, "duplicate/invalid scan data_id")
        seen.add(data_id)
        by_id[data_id] = record

    dynamic = [record for record in records if record["data_id"] not in FIXED_IDS]
    first_b = next((record for record in dynamic if record.get("action_sequence") == ROLE_CONFIG["B"]["sequence"]), None)
    first_c = next((record for record in dynamic if record.get("action_sequence") == ROLE_CONFIG["C"]["sequence"]), None)
    require(first_b is not None and first_c is not None, "cannot derive selector B/C from scan")
    excluded = FIXED_IDS | {first_b["data_id"], first_c["data_id"]}
    first_failure = next(
        (
            record for record in records
            if record["data_id"] not in excluded
            and record.get("terminal_status") == "answered"
            and record.get("exact_match") is False
        ),
        None,
    )
    require(first_failure is not None, "cannot derive independent selector failure from scan")
    chosen = {"B": first_b, "C": first_c, "Failure": first_failure}
    chosen_ids = {label: record["data_id"] for label, record in chosen.items()}
    require(len(set(chosen_ids.values()) | FIXED_IDS) == 5, "five selected IDs are not independent")
    require(records[-1]["candidate_number"] == max(record["candidate_number"] for record in chosen.values()), "selector did not stop when all dynamic slots were fixed")

    role_keys = {"B": "case_b", "C": "case_c", "Failure": "independent_failure"}
    for record in records:
        evaluation = record.get("selection_evaluation")
        require(isinstance(evaluation, dict), "selection evaluation is missing")
        for label, key in role_keys.items():
            require(
                evaluation.get(key, {}).get("selected")
                is (record is chosen[label]),
                f"selector {label} selection flag mismatch",
            )
    return chosen, chosen_ids


def expected_selection_summary(label: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "data_id": record["data_id"],
        "candidate_number": record["candidate_number"],
        "source_row_index": record["source_row_index"],
        "action_sequence": record["action_sequence"],
        "terminal_status": record["terminal_status"],
        "exact_match": record["exact_match"],
        "selection_reason": ROLE_CONFIG[label]["reason"],
    }


def validate_selector_manifest(
    manifest_path: Path,
    manifest: dict[str, Any],
    selector_output: Path,
    parquet: Path,
    protocol_registered: datetime,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    require(manifest.get("schema_version") == SELECTOR_SCHEMA, "selector schema mismatch")
    require(manifest.get("status") == "passed", "selector status is not passed")
    require(manifest.get("error") is None, "selector has an error")
    require(manifest.get("credentials_recorded") is False, "selector credential flag mismatch")
    require(manifest.get("missing_selections") == [], "selector has missing selections")
    require(manifest.get("selected_scan_ids_are_distinct") is True, "selector distinct-ID check failed")
    require(manifest.get("protocol") == EXPECTED_SELECTOR_PROTOCOL, "selector protocol mismatch")
    require(protocol_registered <= parse_time(manifest.get("started_at_utc"), "selector start"), "protocol was not registered before selection")
    require(parse_time(manifest.get("completed_at_utc"), "selector completion") >= parse_time(manifest.get("started_at_utc"), "selector start"), "selector timestamps are reversed")

    inputs = manifest.get("inputs")
    require(isinstance(inputs, dict), "selector inputs are missing")
    require(claimed_path(inputs.get("parquet"), "selector parquet") == parquet, "selector parquet path mismatch")
    require(inputs.get("dataset_repository") == DATASET_ID and inputs.get("dataset_revision") == DATASET_REVISION, "selector dataset pin mismatch")
    require(claimed_path(inputs.get("model_path"), "selector model") == PINNED_MODEL.resolve(strict=True), "selector model path mismatch")
    require(inputs.get("model_repository") == MODEL_ID and inputs.get("model_revision") == MODEL_REVISION, "selector model pin mismatch")
    require(claimed_path(inputs.get("output_dir"), "selector output") == selector_output, "selector output path mismatch")
    require(claimed_path(inputs.get("manifest"), "selector manifest") == manifest_path, "selector manifest self-path mismatch")
    validate_source_pins(manifest.get("source_pins"))

    runtime = manifest.get("model_runtime")
    require(isinstance(runtime, dict), "selector model runtime is missing")
    require(runtime.get("attention_implementation") == "sdpa", "selector attention runtime mismatch")
    require(runtime.get("parameter_dtype") == "torch.bfloat16", "selector dtype runtime mismatch")

    expected_checks = {
        "only_train_search_required_scanned",
        "parquet_order_preserved",
        "scan_limit_respected",
        "case_b_exact",
        "case_c_exact",
        "failure_answered_em_false",
        "failure_independent",
        "all_required_selections_found",
        "placeholder_bodies_not_persisted",
    }
    checks = manifest.get("checks")
    require(isinstance(checks, dict) and set(checks) == expected_checks, "selector check keys mismatch")
    require(all(value is True for value in checks.values()), "a selector check is not true")

    chosen, chosen_ids = validate_scan_records(manifest)
    selections = manifest.get("selections")
    require(isinstance(selections, dict), "selector selections are missing")
    require(selections.get("case_a") == {"data_id": FIXED_A, "selection": "pre_fixed_outside_this_scanner"}, "selector Case A pin mismatch")
    require(selections.get("case_d") == {"data_id": FIXED_D, "selection": "pre_fixed_outside_this_scanner"}, "selector Case D pin mismatch")
    for label, key in (("B", "case_b"), ("C", "case_c"), ("Failure", "independent_failure")):
        require(selections.get(key) == expected_selection_summary(label, chosen[label]), f"selector {label} summary mismatch")
    require(chosen["B"].get("action_sequence") == ["image_search", "answer"], "Case B action mismatch")
    require(chosen["C"].get("action_sequence") == ["text_search", "answer"], "Case C action mismatch")
    require(chosen["B"].get("terminal_status") == "answered", "Case B must terminate answered")
    require(chosen["C"].get("terminal_status") == "answered", "Case C must terminate answered")
    require(chosen["Failure"].get("terminal_status") == "answered" and chosen["Failure"].get("exact_match") is False, "Failure must be answered EM=false")
    return chosen, chosen_ids


def validate_selector_artifacts(
    manifest: dict[str, Any], selector_output: Path, chosen: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, dict) and set(artifacts) == {"case_b", "case_c", "failure"}, "selector artifact roles mismatch")
    result: dict[str, dict[str, Any]] = {}
    for label in ("B", "C", "Failure"):
        config = ROLE_CONFIG[label]
        role = config["selector_role"]
        artifact = artifacts.get(role)
        require(isinstance(artifact, dict), f"selector {label} artifact is invalid")
        record = chosen[label]
        require(artifact.get("data_id") == record["data_id"], f"selector {label} artifact ID mismatch")
        unresolved_image = selector_output / f"{config['selector_basename']}.png"
        unresolved_meta = selector_output / f"{config['selector_basename']}.json"
        require(not unresolved_image.is_symlink() and not unresolved_meta.is_symlink(), f"selector {label} artifact is a symlink")
        expected_image = unresolved_image.resolve(strict=True)
        expected_meta = unresolved_meta.resolve(strict=True)
        require(expected_image.is_relative_to(selector_output) and expected_meta.is_relative_to(selector_output), f"selector {label} artifact escapes output directory")
        image_record = artifact.get("image")
        meta_record = artifact.get("sample_meta")
        require(isinstance(image_record, dict) and isinstance(meta_record, dict), f"selector {label} artifact records are missing")
        require(claimed_path(image_record.get("path"), f"selector {label} image") == expected_image, f"selector {label} image path mismatch")
        require(claimed_path(meta_record.get("path"), f"selector {label} metadata") == expected_meta, f"selector {label} metadata path mismatch")
        image_bytes = read_regular_file(expected_image, maximum=64 * 1024 * 1024)
        meta, meta_bytes = load_json(expected_meta)
        require(image_record == {"path": str(expected_image), "bytes": len(image_bytes), "sha256": sha256_bytes(image_bytes)}, f"selector {label} image hash record mismatch")
        require(meta_record == {"path": str(expected_meta), "bytes": len(meta_bytes), "sha256": sha256_bytes(meta_bytes)}, f"selector {label} metadata hash record mismatch")
        require(meta.get("schema_version") == SELECTED_SAMPLE_SCHEMA and meta.get("role") == role, f"selector {label} selected metadata identity mismatch")
        require(meta.get("selection_reason") == config["reason"], f"selector {label} selected reason mismatch")
        require(meta.get("credentials_recorded") is False and meta.get("raw_webpage_content_present") is False, f"selector {label} safety flags mismatch")
        require(claimed_path(meta.get("image"), f"selector {label} embedded image") == expected_image, f"selector {label} embedded image path mismatch")
        require(meta.get("record") == record, f"selector {label} metadata is not the full selected scan record")
        image_meta = record.get("image")
        require(isinstance(image_meta, dict), f"selector {label} record image metadata missing")
        require(image_meta.get("normalized_png_sha256") == sha256_bytes(image_bytes), f"selector {label} normalized PNG digest mismatch")
        require(image_meta.get("normalized_png_bytes") == len(image_bytes), f"selector {label} normalized PNG byte count mismatch")
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            require(image.format == "PNG" and image.mode == "RGB", f"selector {label} image is not normalized RGB PNG")
            require([image.width, image.height] == [image_meta.get("width"), image_meta.get("height")], f"selector {label} image dimensions mismatch")
        result[label] = {
            "record": record,
            "image_bytes": image_bytes,
            "candidate_scan_sha256": canonical_json_sha256(record),
        }
    return result


def locate_and_load_rows(parquet: Path, selected: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    target_ids = {FIXED_A, FIXED_D} | {value["record"]["data_id"] for value in selected.values()}
    locations: dict[str, list[int]] = {data_id: [] for data_id in target_ids}
    first_search_free: tuple[int, str] | None = None
    parquet_file = pq.ParquetFile(parquet)
    offset = 0
    for batch in parquet_file.iter_batches(batch_size=4096, columns=["data_id", "category"]):
        for local_index, row in enumerate(batch.to_pylist()):
            index = offset + local_index
            data_id = row.get("data_id")
            if row.get("category") == "search_free" and first_search_free is None:
                first_search_free = (index, data_id)
            if data_id in locations:
                locations[data_id].append(index)
        offset += batch.num_rows
    require(all(len(indices) == 1 for indices in locations.values()), "a selected data_id is absent or duplicated in parquet")
    location = {data_id: indices[0] for data_id, indices in locations.items()}
    require(first_search_free == (location[FIXED_A], FIXED_A), "Case A is not the first official search_free row")
    for value in selected.values():
        record = value["record"]
        require(location[record["data_id"]] == record["source_row_index"], "selector source_row_index does not match parquet")

    wanted = set(location.values())
    columns = ["prompt", "images", "reward_model", "data_source", "data_id", "category"]
    rows: dict[int, dict[str, Any]] = {}
    offset = 0
    for batch in parquet_file.iter_batches(batch_size=64, columns=columns):
        for local_index, row in enumerate(batch.to_pylist()):
            index = offset + local_index
            if index in wanted:
                rows[index] = row
        offset += batch.num_rows
        if len(rows) == len(wanted):
            break
    require(set(rows) == wanted, "failed to load all selected parquet rows")
    return {data_id: rows[index] for data_id, index in location.items()}, location


def validate_record_against_row(record: dict[str, Any], row: dict[str, Any], selector_png: bytes) -> None:
    require(row.get("data_id") == record.get("data_id"), "selector/parquet data_id mismatch")
    require(row.get("category") == record.get("category") == "search_required", "selector/parquet category mismatch")
    require(row.get("data_source") == record.get("data_source"), "selector/parquet data_source mismatch")
    require(question_from_prompt(row.get("prompt")) == record.get("question"), "selector/parquet question mismatch")
    reward = row.get("reward_model")
    require(isinstance(reward, dict), "parquet reward_model is invalid")
    require(reward.get("ground_truth") == record.get("ground_truth"), "selector/parquet ground truth mismatch")
    require(candidate_answers(reward) == record.get("candidate_answers"), "selector/parquet candidate answers mismatch")
    require(reward.get("style") == record.get("reward_style"), "selector/parquet reward style mismatch")
    raw = row_image(row)
    image_meta = record["image"]
    require(sha256_bytes(raw) == image_meta.get("source_image_sha256") and len(raw) == image_meta.get("source_bytes"), "selector/parquet source image mismatch")
    rebuilt, width, height = normalized_png(raw)
    require(rebuilt == selector_png, "selector normalized PNG cannot be reproduced from parquet")
    require([width, height] == [image_meta.get("width"), image_meta.get("height")], "selector/parquet dimensions mismatch")


def suite_sample(
    *, row: dict[str, Any], row_index: int, image_path: Path, image_bytes: bytes, source_sha256: str
) -> dict[str, Any]:
    reward = row.get("reward_model")
    require(isinstance(reward, dict), "FVQA reward_model is invalid")
    ground_truth = reward.get("ground_truth")
    require(isinstance(ground_truth, str) and ground_truth.strip(), "FVQA ground truth is invalid")
    with Image.open(BytesIO(image_bytes)) as image:
        image.load()
        width, height = image.size
    return {
        "schema_version": BRIDGE_SCHEMA,
        "data_id": row["data_id"],
        "source_split": DATASET_SPLIT,
        "source_row_index": row_index,
        "category": row["category"],
        "data_source": row.get("data_source"),
        "question": question_from_prompt(row.get("prompt")),
        "reward_model": {
            "ground_truth": ground_truth,
            "candidate_answers": candidate_answers(reward),
            "style": reward.get("style"),
        },
        "image": str(image_path),
        "image_width": width,
        "image_height": height,
        "source_image_sha256": source_sha256,
        "normalized_png_sha256": sha256_bytes(image_bytes),
    }


def validate_output_paths(args: argparse.Namespace, selector_output: Path) -> tuple[Path, Path, dict[str, tuple[Path, Path]]]:
    suite_dir_arg = args.suite_input_dir
    manifest_arg = args.suite_manifest
    require(suite_dir_arg.is_absolute() and manifest_arg.is_absolute(), "suite output paths must be absolute")
    require(not suite_dir_arg.is_symlink() and not manifest_arg.is_symlink(), "suite output paths must not be symlinks")
    suite_dir = suite_dir_arg.resolve(strict=False)
    suite_manifest = manifest_arg.resolve(strict=False)
    require(suite_dir != selector_output and not suite_dir.is_relative_to(selector_output) and not selector_output.is_relative_to(suite_dir), "suite and selector output directories overlap")
    if suite_dir.exists():
        require(suite_dir.is_dir(), "suite-input-dir is not a directory")
    require(not suite_manifest.exists(), "suite manifest already exists; refusing to overwrite commit marker")
    require(suite_manifest.suffix == ".json", "suite manifest must end in .json")
    targets: dict[str, tuple[Path, Path]] = {}
    for label, basename in OUTPUT_BASENAMES.items():
        image_path = suite_dir / f"{basename}.png"
        meta_path = suite_dir / f"{basename}.json"
        require(not image_path.exists() and not meta_path.exists(), f"suite target already exists for {label}")
        require(not image_path.is_symlink() and not meta_path.is_symlink(), f"suite target is a symlink for {label}")
        require(suite_manifest not in {image_path, meta_path}, "suite manifest collides with sample artifact")
        targets[label] = (image_path, meta_path)
    return suite_dir, suite_manifest, targets


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    required = ("pinned_protocol", "selector_manifest", "selector_output_dir", "parquet", "suite_input_dir", "suite_manifest")
    for name in required:
        require(getattr(args, name) is not None, f"--{name.replace('_', '-')} is required")
    protocol_path = strict_file(args.pinned_protocol, "pinned protocol")
    selector_manifest_path = strict_file(args.selector_manifest, "selector manifest")
    selector_output = strict_dir(args.selector_output_dir, "selector output directory")
    parquet = strict_file(args.parquet, "FVQA parquet")
    require(parquet == PINNED_PARQUET.resolve(strict=True), "parquet path is not pinned")
    suite_dir, suite_manifest, targets = validate_output_paths(args, selector_output)

    protocol, protocol_bytes = load_json(protocol_path)
    protocol, protocol_registered = validate_protocol(protocol_path, protocol_bytes)
    parquet_size, parquet_digest = hash_file(parquet)
    require(parquet_digest == PINNED_PARQUET_SHA256, "pinned FVQA parquet SHA-256 mismatch")
    selector_manifest, selector_manifest_bytes = load_json(selector_manifest_path)
    chosen, chosen_ids = validate_selector_manifest(
        selector_manifest_path, selector_manifest, selector_output, parquet, protocol_registered
    )
    selected = validate_selector_artifacts(selector_manifest, selector_output, chosen)
    rows, locations = locate_and_load_rows(parquet, selected)

    image_payloads: dict[str, bytes] = {}
    for label in ("B", "C", "Failure"):
        record = selected[label]["record"]
        validate_record_against_row(record, rows[record["data_id"]], selected[label]["image_bytes"])
        image_payloads[label] = selected[label]["image_bytes"]
    for label, data_id in (("A", FIXED_A), ("D", FIXED_D)):
        row = rows[data_id]
        expected_category = "search_free" if label == "A" else "search_required"
        require(row.get("category") == expected_category, f"Case {label} category mismatch")
        image_payloads[label], _, _ = normalized_png(row_image(row))

    require(len({FIXED_A, FIXED_D, *chosen_ids.values()}) == 5, "suite IDs are not unique")
    metadata_payloads: dict[str, bytes] = {}
    output_records: dict[str, Any] = {}
    for label in ("A", "B", "C", "D", "Failure"):
        data_id = FIXED_A if label == "A" else FIXED_D if label == "D" else chosen_ids[label]
        row = rows[data_id]
        raw = row_image(row)
        image_path, meta_path = targets[label]
        sample = suite_sample(
            row=row,
            row_index=locations[data_id],
            image_path=image_path,
            image_bytes=image_payloads[label],
            source_sha256=sha256_bytes(raw),
        )
        metadata_payloads[label] = pretty_json_bytes(sample)
        output_records[label] = {
            "data_id": data_id,
            "image": {"path": str(image_path), "bytes": len(image_payloads[label]), "sha256": sha256_bytes(image_payloads[label])},
            "metadata": {"path": str(meta_path), "bytes": len(metadata_payloads[label]), "sha256": sha256_bytes(metadata_payloads[label])},
        }

    cases: dict[str, Any] = {}
    protocol_rules = protocol["case_rules"]
    rule_keys = {"A": "A_search_free", "B": "B_image_search", "C": "C_text_search", "D": "D_mixed_search", "Failure": "failure"}
    for label in ("A", "B", "C", "D", "Failure"):
        entry: dict[str, Any] = {
            "meta_file": targets[label][1].name,
            "selection_rule": protocol_rules[rule_keys[label]]["selection"],
        }
        if label in selected:
            entry["candidate_number"] = selected[label]["record"]["candidate_number"]
            entry["candidate_scan_sha256"] = selected[label]["candidate_scan_sha256"]
        cases[label] = entry

    suite_payload = {
        "schema_version": SUITE_SCHEMA,
        "selection_rule": "pre-registered fixed protocol plus verified selector commit; no case was reselected by this bridge",
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
        },
        "seed": 0,
        "cases": cases,
        "bridge": {
            "schema_version": BRIDGE_SCHEMA,
            "pinned_protocol": file_record(protocol_path, protocol_bytes),
            "pinned_parquet": {"path": str(parquet), "bytes": parquet_size, "sha256": parquet_digest},
            "selector_manifest": file_record(selector_manifest_path, selector_manifest_bytes),
            "selector_output_dir": str(selector_output),
            "candidate_scan_sha256_definition": "sha256(UTF-8 JSON of the full selector scan record with ensure_ascii=false, sort_keys=true, separators=(',', ':'))",
            "selected_ids": {"A": FIXED_A, "B": chosen_ids["B"], "C": chosen_ids["C"], "D": FIXED_D, "Failure": chosen_ids["Failure"]},
            "output_artifacts": output_records,
            "checks": {
                "protocol_sha256_pinned": True,
                "parquet_sha256_pinned": True,
                "selector_status_protocol_pins_checks_verified": True,
                "selector_selection_rederived_without_model": True,
                "selector_artifact_paths_and_hashes_verified": True,
                "five_unique_parquet_ids_verified": True,
                "fixed_a_and_d_rebuilt_from_parquet": True,
                "dynamic_images_copied_without_reselection": True,
                "credentials_read": False,
                "network_used": False,
                "model_inference_used": False,
            },
        },
        "credentials_recorded": False,
    }
    suite_manifest_bytes = pretty_json_bytes(suite_payload)

    suite_dir.mkdir(parents=True, exist_ok=True)
    for label in ("A", "B", "C", "D", "Failure"):
        image_path, meta_path = targets[label]
        atomic_write(image_path, image_payloads[label])
        atomic_write(meta_path, metadata_payloads[label])
        require(read_regular_file(image_path, maximum=64 * 1024 * 1024) == image_payloads[label], f"post-write image verification failed for {label}")
        require(read_regular_file(meta_path, maximum=4 * 1024 * 1024) == metadata_payloads[label], f"post-write metadata verification failed for {label}")
    atomic_write(suite_manifest, suite_manifest_bytes)  # Commit marker is last.
    require(read_regular_file(suite_manifest, maximum=4 * 1024 * 1024) == suite_manifest_bytes, "post-write suite manifest verification failed")
    return {
        "status": "passed",
        "suite_input_dir": str(suite_dir),
        "suite_manifest": str(suite_manifest),
        "selected_ids": suite_payload["bridge"]["selected_ids"],
        "suite_manifest_sha256": sha256_bytes(suite_manifest_bytes),
    }


def run_self_tests() -> None:
    require(candidate_answers({"candidate_answers": []}) == [], "empty candidate answer test failed")
    record = {"b": [2, 1], "a": {"z": False}}
    require(canonical_json_sha256(record) == canonical_json_sha256({"a": {"z": False}, "b": [2, 1]}), "canonical JSON digest test failed")
    image = Image.new("RGBA", (2, 3), (1, 2, 3, 4))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    normalized, width, height = normalized_png(buffer.getvalue())
    require((width, height) == (2, 3) and normalized.startswith(b"\x89PNG\r\n\x1a\n"), "PNG normalization test failed")
    dummy = {
        "scan_records": [
            {"candidate_number": 1, "source_row_index": 1, "source_split": "train", "data_id": "failure", "category": "search_required", "action_sequence": ["answer"], "terminal_status": "answered", "final_answer": "wrong", "ground_truth": "right", "exact_match": False, "selection_evaluation": {"case_b": {"selected": False}, "case_c": {"selected": False}, "independent_failure": {"selected": True}}},
            {"candidate_number": 2, "source_row_index": 2, "source_split": "train", "data_id": "b", "category": "search_required", "action_sequence": ["image_search", "answer"], "terminal_status": "answered", "final_answer": "RIGHT", "ground_truth": "right", "exact_match": True, "selection_evaluation": {"case_b": {"selected": True}, "case_c": {"selected": False}, "independent_failure": {"selected": False}}},
            {"candidate_number": 3, "source_row_index": 3, "source_split": "train", "data_id": "c", "category": "search_required", "action_sequence": ["text_search", "answer"], "terminal_status": "answered", "final_answer": " right ", "ground_truth": "RIGHT", "exact_match": True, "selection_evaluation": {"case_b": {"selected": False}, "case_c": {"selected": True}, "independent_failure": {"selected": False}}},
        ],
        "scanned_search_required_count": 3,
    }
    chosen, ids = validate_scan_records(dummy)
    require(ids == {"B": "b", "C": "c", "Failure": "failure"} and chosen["Failure"]["exact_match"] is False, "selection derivation test failed")
    print(json.dumps({"status": "passed", "pure_self_tests": 4}, indent=2))


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_tests()
        return 0
    result = prepare(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
