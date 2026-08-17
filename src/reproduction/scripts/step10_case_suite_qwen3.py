#!/usr/bin/env python3
"""Run and validate the five fixed MMSearch-R1 Step-10 case traces.

The input selection manifest is deliberately separate from execution so case
selection cannot depend on answer correctness.  Its minimal schema is::

    {
      "schema_version": 1,
      "selection_rule": "fixed before suite execution ...",
      "dataset": {
        "id": "lmms-lab/FVQA",
        "revision": "bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5",
        "split": "train"
      },
      "cases": {
        "A": {"meta_file": "case_a.json"},
        "B": {"meta_file": "case_b.json"},
        "C": {"meta_file": "case_c.json"},
        "D": {"meta_file": "case_d.json"},
        "Failure": {"meta_file": "failure.json"}
      }
    }

``meta_key`` may accompany ``meta_file`` when the JSON file is a mapping such
as the existing ``mmsearch_demo/meta.json``.  Optional selection provenance
fields are ``candidate_number``, ``selection_rule`` and
``candidate_scan_sha256``.  This runner never reads an env/key file.  A real
Serper credential, when needed for a cache miss, must already be present in the
process environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Match the selector and make direct script execution independent of the
# caller's PYTHONPATH while keeping all imports pinned to this checkout.
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from placeholder_control_flow import (
    assistant_message,
    classify_response,
    generate_response,
    image_search_result_message,
    image_to_data_uri,
    load_prompt,
    text_search_result_message,
)
from real_text_search_flow_qwen3 import (
    DEFAULT_SUMMARY_BASE_URL,
    DEFAULT_SUMMARY_MODEL,
    MAX_CHARS_PER_PAGE,
    SUMMARY_MAX_TOKENS,
    SUMMARY_MODEL_REPO,
    SUMMARY_MODEL_REVISION,
    SUMMARY_SERVICE_LAUNCH_CONTRACT,
    SUMMARY_SERVICE_PID_PATH,
    TOP_K,
    gpu_snapshot,
    has_reason,
    public_summarizer_config,
    read_summary_service_pid,
    validate_joint_snapshot,
    validate_mmsearch_residency,
    validate_service_only_snapshot,
    valid_health,
    valid_text_status,
)
from reproduction.mmsearch_tools.cached_image_search import FVQACachedImageSearch
from reproduction.mmsearch_tools.qwen3_summarizer import Qwen3Summarizer
from reproduction.mmsearch_tools.real_text_search import SerperJinaTextSearch


SCHEMA_VERSION = 1
SEED = 0
MAX_ROUNDS = 3
MAX_NEW_TOKENS = 512
IMAGE_SEARCH_LIMIT = 1
TEXT_SEARCH_LIMIT = 1
DATASET_ID = "lmms-lab/FVQA"
DATASET_REVISION = "bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5"
DATASET_SPLIT = "train"
MMSEARCH_REPO = "lmms-lab/MMSearch-R1-7B"
MMSEARCH_REVISION = "3cdec93e6db79a409aff4a4b2eadc77a5a8a1e46"
DEFAULT_MMSEARCH_MODEL = Path("/root/autodl-tmp/models/MMSearch-R1-7B")
DEFAULT_IMAGE_CACHE = Path(
    "/root/autodl-tmp/datasets/FVQA/fvqa_train_image_search_results_cache.pkl"
)
DEFAULT_THUMBNAIL_CACHE = Path("/root/autodl-tmp/search_cache/fvqa_thumbnails")
DEFAULT_SERPER_CACHE = Path("/root/autodl-tmp/search_cache/serper/json")
DEFAULT_JINA_CACHE = Path("/root/autodl-tmp/search_cache/jina")
DEFAULT_SUMMARY_CACHE = Path("/root/autodl-tmp/search_cache/qwen3_summary")
OUTPUT_ROOT = Path("/root/autodl-tmp/outputs")
HEX_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

CASE_ORDER = ("A", "B", "C", "D", "Failure")
CASE_TYPES = {
    "A": "search_free",
    "B": "image_search",
    "C": "text_search",
    "D": "mixed_search",
    "Failure": "failure",
}
EXPECTED_ACTION_SEQUENCES: dict[str, list[str]] = {
    "A": ["answer"],
    "B": ["image_search", "answer"],
    "C": ["text_search", "answer"],
    "D": ["image_search", "text_search", "answer"],
}
CASE_FILENAMES = {
    "A": "case_A_search_free.json",
    "B": "case_B_image_search.json",
    "C": "case_C_text_search.json",
    "D": "case_D_mixed_search.json",
    "Failure": "failure_case.json",
}

_FORBIDDEN_NORMALIZED_KEYS = {
    "apikey",
    "authorization",
    "base64image",
    "body",
    "content",
    "headers",
    "imagedata",
    "inputtext",
    "messages",
    "pagecontent",
    "pagetext",
    "prompt",
    "rawcontent",
    "requestbody",
    "responsebody",
    "secret",
    "token",
    "webpagecontent",
}
_RAW_JINA_MARKERS = (
    b"[webpage contents]",
    b"webpage content:",
)
_DATA_IMAGE_BASE64_RE = re.compile(
    rb"data:image/[^;\s]{1,32};base64,",
    flags=re.IGNORECASE,
)
_LONG_BASE64_RE = re.compile(rb"(?:[A-Za-z0-9+/]{512,}={0,2})")
_AUTH_HEADER_RE = re.compile(rb"(?i)(?:\bbearer\s+|x-api-key)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--selected-meta-dir", type=Path)
    parser.add_argument("--cache-pickle", type=Path)
    parser.add_argument("--thumbnail-cache-dir", type=Path)
    parser.add_argument("--serper-cache-dir", type=Path)
    parser.add_argument("--jina-cache-dir", type=Path)
    parser.add_argument("--summary-cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--summary-base-url", "--base-url", dest="summary_base_url",
        default=DEFAULT_SUMMARY_BASE_URL,
    )
    parser.add_argument("--summary-model", default=DEFAULT_SUMMARY_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def strict_exact_normalize(value: str) -> str:
    return value.strip().lower()


def _normalized_key(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _known_secrets() -> list[str]:
    values: list[str] = []
    for name in ("SERPER_API_KEY", "SUMMARIZER_API_KEY"):
        value = os.environ.get(name)
        if value and value != "EMPTY":
            values.append(value)
    return sorted(set(values), key=len, reverse=True)


def redact_text(value: str) -> str:
    result = value
    for secret in _known_secrets():
        result = result.replace(secret, "[REDACTED]")
    result = re.sub(r"(?i)bearer\s+\S+", "[REDACTED_AUTH]", result)
    result = re.sub(
        r"(?i)(?:api[_-]?key|token|authorization)=([^&\s]+)",
        lambda match: match.group(0).split("=", 1)[0] + "=[REDACTED]",
        result,
    )
    return result


def safe_error(exc: BaseException) -> str:
    return redact_text(f"{type(exc).__name__}: {exc}")[:500]


def sanitize_public(value: Any) -> Any:
    """Remove forbidden fields and redact known secret values recursively."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if _normalized_key(name) in _FORBIDDEN_NORMALIZED_KEYS:
                continue
            result[name] = sanitize_public(item)
        return result
    if isinstance(value, (list, tuple)):
        return [sanitize_public(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_text(str(value))


def forbidden_paths(value: Any, prefix: str = "$") -> list[str]:
    matches: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key)
            child = f"{prefix}.{name}"
            if _normalized_key(name) in _FORBIDDEN_NORMALIZED_KEYS:
                matches.append(child)
            matches.extend(forbidden_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            matches.extend(forbidden_paths(item, f"{prefix}[{index}]"))
    return matches


def _assert_secret_free(encoded: bytes) -> None:
    for secret in _known_secrets():
        require(secret.encode("utf-8") not in encoded, "Serialized evidence contains a credential")


def _atomic_replace(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        os.replace(temporary_name, path)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    public = sanitize_public(payload)
    require(isinstance(public, dict), "Public JSON root is not an object")
    require(not forbidden_paths(public), "Public JSON contains a forbidden field")
    encoded = (
        json.dumps(public, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    )
    _assert_secret_free(encoded)
    _atomic_replace(path, encoded)


def atomic_write_text(path: Path, value: str) -> None:
    safe = redact_text(value)
    encoded = safe.encode("utf-8")
    _assert_secret_free(encoded)
    _atomic_replace(path, encoded)


def read_regular_file(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"Cannot open required file: {path}") from exc
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), f"Required path is not regular: {path}")
        require(0 < info.st_size <= max_bytes, f"Required file has unsafe size: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(8 * 1024 * 1024, max_bytes - total + 1))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            require(total <= max_bytes, f"Required file exceeds safety limit: {path}")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def load_json_object(path: Path, *, max_bytes: int = 4 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    encoded = read_regular_file(path, max_bytes=max_bytes)
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid JSON object: {path}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value, encoded


def file_record(path: Path) -> dict[str, Any]:
    encoded = read_regular_file(path, max_bytes=128 * 1024 * 1024)
    return {
        "path": str(path.resolve(strict=True)),
        "bytes": len(encoded),
        "sha256": sha256_bytes(encoded),
    }


def scan_serialized_evidence(
    json_paths: list[Path],
    text_paths: list[Path],
) -> dict[str, Any]:
    forbidden_json_key_paths: list[str] = []
    data_image_base64_matches = 0
    long_base64_matches = 0
    raw_jina_marker_matches = 0
    authorization_marker_matches = 0
    known_credential_value_matches = 0
    files_scanned = 0
    for path in [*json_paths, *text_paths]:
        encoded = read_regular_file(path, max_bytes=128 * 1024 * 1024)
        files_scanned += 1
        lowered = encoded.lower()
        data_image_base64_matches += len(_DATA_IMAGE_BASE64_RE.findall(encoded))
        long_base64_matches += len(_LONG_BASE64_RE.findall(encoded))
        raw_jina_marker_matches += sum(
            lowered.count(marker) for marker in _RAW_JINA_MARKERS
        )
        authorization_marker_matches += len(_AUTH_HEADER_RE.findall(encoded))
        known_credential_value_matches += sum(
            encoded.count(secret.encode("utf-8"))
            for secret in _known_secrets()
        )
        if path in json_paths:
            try:
                payload = json.loads(encoded.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Serialized case is not valid JSON: {path.name}"
                ) from exc
            forbidden_json_key_paths.extend(
                f"{path.name}:{item}" for item in forbidden_paths(payload)
            )
    result = {
        "files_scanned": files_scanned,
        "json_files_scanned": len(json_paths),
        "text_files_scanned": len(text_paths),
        "forbidden_json_key_paths": forbidden_json_key_paths,
        "data_image_base64_matches": data_image_base64_matches,
        "long_base64_matches": long_base64_matches,
        "raw_jina_marker_matches": raw_jina_marker_matches,
        "authorization_marker_matches": authorization_marker_matches,
        "known_credential_value_matches": known_credential_value_matches,
    }
    result["pass"] = (
        files_scanned == len(json_paths) + len(text_paths)
        and not forbidden_json_key_paths
        and data_image_base64_matches == 0
        and long_base64_matches == 0
        and raw_jina_marker_matches == 0
        and authorization_marker_matches == 0
        and known_credential_value_matches == 0
    )
    return result


def normalized_candidate_answers(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError("candidate_answers is not a JSON list") from exc
    else:
        decoded = value
    require(isinstance(decoded, list), "candidate_answers must be a list")
    require(all(isinstance(item, str) and item.strip() for item in decoded), "candidate answer is invalid")
    return [item.strip() for item in decoded]


def _safe_child(root: Path, relative_name: str, *, suffix: str | None = None) -> Path:
    require(isinstance(relative_name, str) and relative_name, "Relative path is missing")
    unresolved = Path(relative_name)
    require(not unresolved.is_absolute() and ".." not in unresolved.parts, "Path escapes selected-meta-dir")
    candidate = root / unresolved
    require(not candidate.is_symlink(), "Selected metadata path is a symlink")
    resolved = candidate.resolve(strict=True)
    require(resolved.is_relative_to(root), "Selected metadata path escapes root")
    if suffix is not None:
        require(resolved.suffix == suffix, f"Selected metadata path must end in {suffix}")
    return resolved


def _manifest_case_entry(cases: dict[str, Any], label: str) -> dict[str, Any]:
    aliases = {
        "A": ("A", "search_free"),
        "B": ("B", "image_search"),
        "C": ("C", "text_search"),
        "D": ("D", "mixed_search"),
        "Failure": ("Failure", "failure"),
    }[label]
    present = [key for key in aliases if key in cases]
    require(len(present) == 1, f"Selection manifest must define one entry for case {label}")
    raw = cases[present[0]]
    if isinstance(raw, str):
        return {"meta_file": raw}
    require(isinstance(raw, dict), f"Case {label} selection entry is invalid")
    return raw


def load_selected_samples(
    selection_manifest: Path,
    selected_meta_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    manifest, manifest_bytes = load_json_object(selection_manifest)
    require(manifest.get("schema_version") == SCHEMA_VERSION, "Selection manifest schema mismatch")
    selection_rule = manifest.get("selection_rule")
    require(isinstance(selection_rule, str) and selection_rule.strip(), "Selection rule is missing")
    dataset = manifest.get("dataset")
    require(isinstance(dataset, dict), "Selection dataset provenance is missing")
    require(dataset.get("id") == DATASET_ID, "Selection dataset ID mismatch")
    require(dataset.get("revision") == DATASET_REVISION, "Selection dataset revision mismatch")
    require(dataset.get("split") == DATASET_SPLIT, "Selection dataset split mismatch")
    require(manifest.get("seed", SEED) == SEED, "Selection seed must be zero")
    cases = manifest.get("cases")
    require(isinstance(cases, dict), "Selection cases mapping is missing")

    samples: dict[str, dict[str, Any]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for label in CASE_ORDER:
        entry = _manifest_case_entry(cases, label)
        meta_path = _safe_child(selected_meta_dir, entry.get("meta_file"), suffix=".json")
        payload, meta_bytes = load_json_object(meta_path)
        meta_key = entry.get("meta_key")
        if meta_key is not None:
            require(isinstance(meta_key, str) and meta_key in payload, f"Case {label} meta_key is missing")
            payload = payload[meta_key]
            require(isinstance(payload, dict), f"Case {label} selected metadata is not an object")

        data_id = payload.get("data_id")
        require(isinstance(data_id, str) and data_id, f"Case {label} data_id is invalid")
        require(data_id not in seen_ids, "All five Step-10 cases must have independent data IDs")
        seen_ids.add(data_id)
        require(payload.get("source_split") == DATASET_SPLIT, f"Case {label} split mismatch")
        category = payload.get("category")
        require(category in {"search_free", "search_required"}, f"Case {label} category is invalid")
        if label == "A":
            require(category == "search_free", "Case A must use an official search_free row")
        elif label in {"B", "C", "D", "Failure"}:
            require(category == "search_required", f"Case {label} must use search_required")

        image_value = payload.get("image")
        require(isinstance(image_value, str), f"Case {label} image path is missing")
        image_path = Path(image_value)
        require(image_path.is_absolute() and not image_path.is_symlink(), "Image path must be absolute/non-symlink")
        image_path = image_path.resolve(strict=True)
        require(image_path.is_relative_to(selected_meta_dir), "Selected image escapes selected-meta-dir")
        image_bytes = read_regular_file(image_path, max_bytes=64 * 1024 * 1024)
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
        require(payload.get("image_width") == width, f"Case {label} image width mismatch")
        require(payload.get("image_height") == height, f"Case {label} image height mismatch")
        source_digest = payload.get("source_image_sha256")
        require(
            isinstance(source_digest, str) and HEX_SHA256_RE.fullmatch(source_digest) is not None,
            f"Case {label} source image digest is invalid",
        )
        question = payload.get("question")
        reward = payload.get("reward_model")
        require(isinstance(question, str) and question.strip(), f"Case {label} question is missing")
        require(isinstance(reward, dict), f"Case {label} reward metadata is missing")
        ground_truth = reward.get("ground_truth")
        require(isinstance(ground_truth, str) and ground_truth.strip(), f"Case {label} ground truth is missing")
        candidate_answers = normalized_candidate_answers(reward.get("candidate_answers"))
        row_index = payload.get("source_row_index")
        require(isinstance(row_index, int) and not isinstance(row_index, bool) and row_index >= 0, "source_row_index is invalid")

        samples[label] = {
            "data_id": data_id,
            "category": category,
            "source_split": DATASET_SPLIT,
            "source_row_index": row_index,
            "data_source": sanitize_public(payload.get("data_source")),
            "question": question.strip(),
            "candidate_answers": candidate_answers,
            "ground_truth": ground_truth.strip(),
            "image": str(image_path),
            "image_width": width,
            "image_height": height,
            "image_png_sha256": sha256_bytes(image_bytes),
            "source_image_sha256": source_digest,
        }
        candidate_scan_sha = entry.get("candidate_scan_sha256")
        if candidate_scan_sha is not None:
            require(
                isinstance(candidate_scan_sha, str)
                and HEX_SHA256_RE.fullmatch(candidate_scan_sha) is not None,
                f"Case {label} candidate scan digest is invalid",
            )
        candidate_number = entry.get("candidate_number")
        if candidate_number is not None:
            require(
                isinstance(candidate_number, int)
                and not isinstance(candidate_number, bool)
                and candidate_number > 0,
                f"Case {label} candidate number is invalid",
            )
        provenance[label] = {
            "selection_rule": str(entry.get("selection_rule") or selection_rule),
            "candidate_number": candidate_number,
            "candidate_scan_sha256": candidate_scan_sha,
            "selected_metadata": {
                "path": str(meta_path),
                "bytes": len(meta_bytes),
                "sha256": sha256_bytes(meta_bytes),
                "meta_key": meta_key,
            },
        }

    manifest_record = {
        "path": str(selection_manifest),
        "bytes": len(manifest_bytes),
        "sha256": sha256_bytes(manifest_bytes),
    }
    return samples, provenance, sanitize_public(dataset), manifest_record


def initial_messages(sample: dict[str, Any], round_1_prompt: str) -> list[dict[str, Any]]:
    return [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"{round_1_prompt}\nQuestion: {sample['question']}\nImage: ",
            },
            {
                "type": "image",
                "image": image_to_data_uri(Path(sample["image"])),
                "max_pixels": 672 * 672,
            },
        ],
    }]


def derive_network_metrics(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "thumbnail_network_attempts": 0,
        "serper_api_requests": 0,
        "jina_network_requests": 0,
        "qwen_summary_requests": 0,
        "thumbnail_cache_hits": 0,
        "serper_cache_hits": 0,
        "jina_cache_hits": 0,
        "qwen_summary_cache_hits": 0,
        "count_complete": True,
        "count_incomplete_reasons": [],
    }
    for round_trace in rounds:
        tool = round_trace.get("tool")
        if not isinstance(tool, dict):
            continue
        status = tool.get("status", {})
        if not isinstance(status, dict):
            metrics["count_complete"] = False
            metrics["count_incomplete_reasons"].append(
                f"round_{round_trace.get('round')}_tool_status_missing"
            )
            continue
        if tool.get("type") == "fvqa_official_image_search_cache":
            raw_requested = status.get("requested")
            raw_cache_hits = status.get("cache_hits")
            image_counts_valid = (
                isinstance(raw_requested, int)
                and not isinstance(raw_requested, bool)
                and isinstance(raw_cache_hits, int)
                and not isinstance(raw_cache_hits, bool)
                and raw_requested >= raw_cache_hits >= 0
            )
            if not image_counts_valid:
                metrics["count_complete"] = False
                metrics["count_incomplete_reasons"].append(
                    f"round_{round_trace.get('round')}_image_counts_incomplete"
                )
            else:
                metrics["thumbnail_cache_hits"] += raw_cache_hits
                metrics["thumbnail_network_attempts"] += (
                    raw_requested - raw_cache_hits
                )
        elif tool.get("type") == "serper_dev_plus_jina_reader_plus_qwen3_summary":
            search = status.get("search", {})
            reader = status.get("reader", {})
            summaries = status.get("summaries", [])
            if isinstance(search, dict):
                search_hit_value = search.get("response_cache_hit")
                if not isinstance(search_hit_value, bool):
                    metrics["count_complete"] = False
                    metrics["count_incomplete_reasons"].append(
                        f"round_{round_trace.get('round')}_serper_count_incomplete"
                    )
                else:
                    metrics["serper_cache_hits"] += int(search_hit_value)
                    metrics["serper_api_requests"] += int(not search_hit_value)
            else:
                metrics["count_complete"] = False
                metrics["count_incomplete_reasons"].append(
                    f"round_{round_trace.get('round')}_serper_status_missing"
                )
            if isinstance(reader, dict):
                raw_requested = reader.get("requested")
                raw_cache_hits = reader.get("cache_hits")
                reader_counts_valid = (
                    isinstance(raw_requested, int)
                    and not isinstance(raw_requested, bool)
                    and isinstance(raw_cache_hits, int)
                    and not isinstance(raw_cache_hits, bool)
                    and raw_requested >= raw_cache_hits >= 0
                )
                if not reader_counts_valid:
                    metrics["count_complete"] = False
                    metrics["count_incomplete_reasons"].append(
                        f"round_{round_trace.get('round')}_jina_counts_incomplete"
                    )
                else:
                    metrics["jina_cache_hits"] += raw_cache_hits
                    metrics["jina_network_requests"] += (
                        raw_requested - raw_cache_hits
                    )
            else:
                metrics["count_complete"] = False
                metrics["count_incomplete_reasons"].append(
                    f"round_{round_trace.get('round')}_jina_status_missing"
                )
            if isinstance(summaries, list) and len(summaries) == TOP_K:
                for summary in summaries:
                    if not isinstance(summary, dict):
                        metrics["count_complete"] = False
                        metrics["count_incomplete_reasons"].append(
                            f"round_{round_trace.get('round')}_qwen_trace_invalid"
                        )
                        continue
                    if (
                        not isinstance(summary.get("cache_hit"), bool)
                        or not isinstance(summary.get("api_called"), bool)
                    ):
                        metrics["count_complete"] = False
                        metrics["count_incomplete_reasons"].append(
                            f"round_{round_trace.get('round')}_qwen_count_incomplete"
                        )
                    cache_hit = summary.get("cache_hit") is True
                    api_called = summary.get("api_called") is True
                    metrics["qwen_summary_cache_hits"] += int(cache_hit)
                    metrics["qwen_summary_requests"] += int(api_called)
            else:
                metrics["count_complete"] = False
                metrics["count_incomplete_reasons"].append(
                    f"round_{round_trace.get('round')}_qwen_status_missing"
                )
        else:
            metrics["count_complete"] = False
            metrics["count_incomplete_reasons"].append(
                f"round_{round_trace.get('round')}_unknown_tool_type"
            )
    metrics["external_network_requests"] = sum(
        metrics[name]
        for name in (
            "thumbnail_network_attempts",
            "serper_api_requests",
            "jina_network_requests",
        )
    )
    metrics["local_qwen_completion_requests"] = metrics["qwen_summary_requests"]
    metrics["tool_http_requests_total"] = (
        metrics["external_network_requests"]
        + metrics["local_qwen_completion_requests"]
    )
    return metrics


def validate_image_status(status: Any) -> bool:
    if not isinstance(status, dict):
        return False
    images = status.get("num_images")
    titles = status.get("titles")
    urls = status.get("thumbnail_urls")
    paths = status.get("local_thumbnail_paths")
    failures = status.get("failures")
    cache_hits = status.get("cache_hits")
    return (
        status.get("success") is True
        and status.get("requested") == TOP_K
        and isinstance(images, int)
        and not isinstance(images, bool)
        and 4 <= images <= TOP_K
        and isinstance(titles, list)
        and len(titles) == images
        and isinstance(urls, list)
        and len(urls) == images
        and isinstance(paths, list)
        and len(paths) == images
        and isinstance(failures, list)
        and images + len(failures) == TOP_K
        and isinstance(cache_hits, int)
        and not isinstance(cache_hits, bool)
        and 0 <= cache_hits <= images
    )


def validate_text_status(status: Any) -> bool:
    return (
        isinstance(status, dict)
        and valid_text_status(status)
        and isinstance(status.get("documents"), list)
        and len(status["documents"]) == TOP_K
        and isinstance(status.get("summaries"), list)
        and len(status["summaries"]) == TOP_K
    )


def classify_failure_layer(trace: dict[str, Any]) -> dict[str, Any]:
    """Classify only the narrowest failure boundary supported by trace evidence."""
    rounds = trace.get("rounds", [])
    if not isinstance(rounds, list):
        rounds = []
    for index, round_trace in enumerate(rounds):
        if not isinstance(round_trace, dict):
            continue
        tool = round_trace.get("tool")
        if not isinstance(tool, dict):
            continue
        status = tool.get("status")
        status_path = f"$.rounds[{index}].tool.status"
        if tool.get("type") == "fvqa_official_image_search_cache":
            if isinstance(status, dict) and not validate_image_status(status):
                return {
                    "layer": "image_search_results",
                    "rationale": "The persisted official image-search status fails its pinned five-result/4-of-5 contract.",
                    "evidence_paths": [status_path],
                }
        elif (
            tool.get("type")
            == "serper_dev_plus_jina_reader_plus_qwen3_summary"
            and isinstance(status, dict)
        ):
            search = status.get("search")
            reader = status.get("reader")
            summary = status.get("summary")
            if isinstance(search, dict) and (
                search.get("success") is not True
                or search.get("num_results") != TOP_K
            ):
                return {
                    "layer": "serper_search_results",
                    "rationale": "The persisted Serper component status explicitly fails the pinned top-5 contract.",
                    "evidence_paths": [f"{status_path}.search"],
                }
            if isinstance(reader, dict) and (
                reader.get("success") is not True
                or reader.get("num_documents") != TOP_K
                or reader.get("failures") != []
            ):
                return {
                    "layer": "jina_reader",
                    "rationale": "The persisted Jina component status explicitly fails the pinned top-5/no-failure contract.",
                    "evidence_paths": [f"{status_path}.reader"],
                }
            if isinstance(summary, dict) and (
                summary.get("success") is not True
                or summary.get("num_summaries") != TOP_K
                or summary.get("failures") != []
            ):
                return {
                    "layer": "qwen_summarization",
                    "rationale": "The persisted Qwen component status explicitly fails the pinned top-5/no-failure contract.",
                    "evidence_paths": [f"{status_path}.summary"],
                }
            health = tool.get("summary_service_health_after_tool")
            if isinstance(health, dict) and not valid_health(health):
                return {
                    "layer": "qwen_summarization",
                    "rationale": "The persisted post-tool Qwen health status explicitly fails its service contract.",
                    "evidence_paths": [
                        f"$.rounds[{index}].tool.summary_service_health_after_tool"
                    ],
                }
    if (
        isinstance(trace.get("final_answer"), str)
        and trace.get("terminal_status") == "answered"
        and trace.get("exact_match") is False
    ):
        return {
            "layer": "final_answer",
            "rationale": "A final answer was emitted and strict normalized Exact Match is false; upstream causality is not inferred.",
            "evidence_paths": [
                "$.final_answer",
                "$.ground_truth",
                "$.exact_match",
                "$.tool_infrastructure_success",
            ],
        }
    return {
        "layer": "control_flow",
        "rationale": "No narrower tool-component failure is supported by persisted status; the trace did not finish with an EM-false answered result.",
        "evidence_paths": [
            "$.terminal_status",
            "$.action_sequence",
            "$.hard_failures",
        ],
    }


def valid_failure_capture(trace: dict[str, Any]) -> bool:
    return (
        trace.get("terminal_status") == "answered"
        and isinstance(trace.get("final_answer"), str)
        and trace.get("exact_match") is False
        and trace.get("tool_infrastructure_success") is True
        and isinstance(trace.get("network_and_cache"), dict)
        and trace["network_and_cache"].get("count_complete") is True
    )


def execute_case(
    *,
    label: str,
    sample: dict[str, Any],
    selection: dict[str, Any],
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    image_search: FVQACachedImageSearch,
    text_search: SerperJinaTextSearch,
    summarizer: Qwen3Summarizer,
    round_1_prompt: str,
    after_image_search_prompt: str,
    after_text_search_prompt: str,
) -> dict[str, Any]:
    messages = initial_messages(sample, round_1_prompt)
    started = time.monotonic()
    trace: dict[str, Any] = {
        "data_id": sample["data_id"],
        "source_split": sample["source_split"],
        "source_row_index": sample["source_row_index"],
        "category": sample["category"],
        "data_source": sample["data_source"],
        "question": sample["question"],
        "candidate_answers": sample["candidate_answers"],
        "ground_truth": sample["ground_truth"],
        "image": {
            "path": sample["image"],
            "width": sample["image_width"],
            "height": sample["image_height"],
            "png_sha256": sample["image_png_sha256"],
            "source_sha256": sample["source_image_sha256"],
        },
        "rounds": [],
        "image_search_calls": 0,
        "text_search_calls": 0,
        "summarization_calls": 0,
        "final_answer": None,
        "terminal_status": None,
        "hard_failures": [],
    }

    for round_number in range(1, MAX_ROUNDS + 1):
        try:
            response, input_tokens, output_tokens, elapsed = generate_response(
                model, processor, messages, MAX_NEW_TOKENS
            )
        except Exception as exc:  # noqa: BLE001 - evidence boundary
            trace["hard_failures"].append({
                "stage": "mmsearch_generation",
                "round": round_number,
                "error": safe_error(exc),
            })
            trace["terminal_status"] = "mmsearch_generation_error"
            break

        action, payload = classify_response(response)
        public_response = redact_text(response)
        public_payload = redact_text(payload) if isinstance(payload, str) else None
        round_trace: dict[str, Any] = {
            "round": round_number,
            "response": public_response,
            "response_sha256": sha256_text(public_response),
            "has_reason": has_reason(response),
            "action": action,
            "parsed_payload": public_payload,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generation_seconds": round(elapsed, 3),
        }
        trace["rounds"].append(round_trace)

        if action == "answer":
            trace["final_answer"] = public_payload
            trace["terminal_status"] = "answered"
            break
        if action in {"warning", "invalid"}:
            trace["terminal_status"] = (
                "warning" if action == "warning" else "invalid_response"
            )
            break

        messages.append(assistant_message(response))
        if action == "image_search":
            if trace["image_search_calls"] >= IMAGE_SEARCH_LIMIT:
                trace["terminal_status"] = "image_search_limit"
                break
            try:
                returned_text, returned_images, status = image_search(sample["data_id"])
                trace["image_search_calls"] += 1
                public_status = sanitize_public(status)
                round_trace["tool"] = {
                    "type": "fvqa_official_image_search_cache",
                    "status": public_status,
                    "returned_text_characters": len(returned_text),
                    "returned_text_sha256": sha256_text(returned_text),
                }
                if not validate_image_status(status):
                    raise RuntimeError("Image Search returned an invalid/empty public status")
                messages.append(image_search_result_message(
                    returned_images,
                    status["titles"],
                    sample["question"],
                    after_image_search_prompt,
                ))
            except Exception as exc:  # noqa: BLE001 - evidence boundary
                trace["hard_failures"].append({
                    "stage": "image_search",
                    "round": round_number,
                    "error": safe_error(exc),
                })
                round_trace.setdefault("tool", {
                    "type": "fvqa_official_image_search_cache",
                    "status": {"success": False},
                })
                trace["terminal_status"] = "image_search_hard_failure"
                break
            continue

        if action == "text_search":
            if trace["text_search_calls"] >= TEXT_SEARCH_LIMIT:
                trace["terminal_status"] = "text_search_limit"
                break
            query = payload or ""
            round_trace["query"] = redact_text(query)
            try:
                returned_text, status = text_search(query)
                trace["text_search_calls"] += 1
                trace["summarization_calls"] += 1
                public_status = sanitize_public(status)
                post_tool_health = summarizer.health_check()
                round_trace["tool"] = {
                    "type": "serper_dev_plus_jina_reader_plus_qwen3_summary",
                    "status": public_status,
                    "summary_service_health_after_tool": sanitize_public(post_tool_health),
                    "returned_text_characters": len(returned_text),
                    "returned_text_sha256": sha256_text(returned_text),
                }
                if not validate_text_status(status):
                    raise RuntimeError("Text/Search/Reader/Qwen top-5 contract failed")
                if not valid_health(post_tool_health):
                    raise RuntimeError("Qwen3 health failed after text tool")
                messages.append(text_search_result_message(
                    returned_text,
                    sample["question"],
                    after_text_search_prompt,
                ))
            except Exception as exc:  # noqa: BLE001 - evidence boundary
                trace["hard_failures"].append({
                    "stage": "text_search_jina_qwen",
                    "round": round_number,
                    "error": safe_error(exc),
                })
                round_trace.setdefault("tool", {
                    "type": "serper_dev_plus_jina_reader_plus_qwen3_summary",
                    "status": {"success": False},
                })
                trace["terminal_status"] = "text_search_hard_failure"
                break
            continue

        trace["terminal_status"] = "unsupported_action"
        break
    else:
        trace["terminal_status"] = "max_rounds"

    if trace["terminal_status"] is None:
        trace["terminal_status"] = "max_rounds_after_tool"
    trace["total_turns"] = len(trace["rounds"])
    trace["action_sequence"] = [item["action"] for item in trace["rounds"]]
    trace["exact_match"] = (
        strict_exact_normalize(trace["final_answer"])
        == strict_exact_normalize(trace["ground_truth"])
        if isinstance(trace["final_answer"], str)
        else False
    )
    trace["tool_infrastructure_success"] = not trace["hard_failures"]
    trace["network_and_cache"] = derive_network_metrics(trace["rounds"])
    trace["mmsearch_input_tokens"] = sum(item["input_tokens"] for item in trace["rounds"])
    trace["mmsearch_output_tokens"] = sum(item["output_tokens"] for item in trace["rounds"])
    trace["mmsearch_generation_seconds"] = round(
        sum(item["generation_seconds"] for item in trace["rounds"]), 3
    )
    trace["case_seconds"] = round(time.monotonic() - started, 3)

    expected = EXPECTED_ACTION_SEQUENCES.get(label)
    path_pass = trace["action_sequence"] == expected if expected is not None else None
    representative_pass = (
        label != "Failure"
        and path_pass is True
        and trace["terminal_status"] == "answered"
        and trace["tool_infrastructure_success"] is True
        and trace["network_and_cache"].get("count_complete") is True
    )
    failure_capture_pass = (
        label == "Failure"
        and valid_failure_capture(trace)
    )
    trace["expected_action_sequence"] = expected
    trace["path_pass"] = path_pass
    trace["representative_pass"] = representative_pass
    trace["failure_capture_pass"] = failure_capture_pass
    if label == "Failure":
        layer = classify_failure_layer(trace)
        trace["failure_analysis"] = {
            **layer,
            "infrastructure_failure": trace["tool_infrastructure_success"] is not True,
            "terminal_status": trace["terminal_status"],
            "actual_action_sequence": trace["action_sequence"],
            "exact_match": trace["exact_match"],
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "expected_failure_captured"
            if failure_capture_pass
            else "passed"
            if representative_pass
            else "failed"
        ),
        "case_label": label,
        "case_type": CASE_TYPES[label],
        "selection": selection,
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
        },
        "models": {
            "mmsearch": {
                "repository": MMSEARCH_REPO,
                "revision": MMSEARCH_REVISION,
                "local_path": str(DEFAULT_MMSEARCH_MODEL),
            },
            "summarizer": {
                "repository": SUMMARY_MODEL_REPO,
                "revision": SUMMARY_MODEL_REVISION,
                "local_path": DEFAULT_SUMMARY_MODEL,
            },
        },
        "decoding": {
            "seed": SEED,
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_rounds": MAX_ROUNDS,
            "image_search_limit": IMAGE_SEARCH_LIMIT,
            "text_search_limit": TEXT_SEARCH_LIMIT,
        },
        "tool_contract": {
            "image_search": "official_fvqa_cache_top5",
            "text_search": "serper_dev_top5",
            "reader": "jina_top5_max_12000_chars_per_page",
            "summarizer": "qwen3_top5_temperature0_seed0_max512_thinking_false",
        },
        "trace": trace,
        "pass": representative_pass or failure_capture_pass,
    }


def validate_runtime_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    required = (
        "model_path",
        "selection_manifest",
        "selected_meta_dir",
        "cache_pickle",
        "thumbnail_cache_dir",
        "serper_cache_dir",
        "jina_cache_dir",
        "summary_cache_dir",
        "output_dir",
    )
    for name in required:
        require(getattr(args, name) is not None, f"--{name.replace('_', '-')} is required")
    require(args.max_new_tokens == MAX_NEW_TOKENS, "max_new_tokens is pinned to 512")
    require(args.max_rounds == MAX_ROUNDS, "max_rounds is pinned to 3")
    require(args.summary_model == DEFAULT_SUMMARY_MODEL, "summary model path mismatch")
    require(args.summary_base_url == DEFAULT_SUMMARY_BASE_URL, "summary base URL mismatch")

    model_path = args.model_path.resolve(strict=True)
    selection_manifest = args.selection_manifest.resolve(strict=True)
    selected_meta_dir = args.selected_meta_dir.resolve(strict=True)
    require(model_path == DEFAULT_MMSEARCH_MODEL.resolve(strict=True), "MMSearch model path mismatch")
    require(args.cache_pickle.resolve(strict=True) == DEFAULT_IMAGE_CACHE.resolve(strict=True), "Image cache path mismatch")
    require(selected_meta_dir.is_dir() and not args.selected_meta_dir.is_symlink(), "selected-meta-dir is unsafe")
    require(not args.selection_manifest.is_symlink(), "selection manifest is a symlink")

    fixed_dirs = (
        (args.thumbnail_cache_dir, DEFAULT_THUMBNAIL_CACHE, "thumbnail cache"),
        (args.serper_cache_dir, DEFAULT_SERPER_CACHE, "Serper cache"),
        (args.jina_cache_dir, DEFAULT_JINA_CACHE, "Jina cache"),
        (args.summary_cache_dir, DEFAULT_SUMMARY_CACHE, "summary cache"),
    )
    for actual, expected, label in fixed_dirs:
        require(actual.resolve() == expected.resolve(), f"{label} path mismatch")

    output_root = OUTPUT_ROOT.resolve(strict=True)
    require(not args.output_dir.is_symlink(), "output-dir is a symlink")
    output_dir = args.output_dir.resolve()
    require(
        output_dir.parent == output_root,
        "output-dir must be one new immediate child of outputs",
    )
    require(not output_dir.exists(), "output-dir already exists; stale evidence is forbidden")
    os.mkdir(output_dir, mode=0o700)
    os.chmod(output_dir, 0o700)
    directory_fd = os.open(
        output_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return selection_manifest, selected_meta_dir, output_dir


def case_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# MMSearch-R1 Step 10 cases",
        "",
        "Selection was fixed before execution. Representative path acceptance is independent of Exact Match.",
        "",
        "| Case | data_id | actions | path pass | EM | terminal |",
        "|---|---|---|---:|---:|---|",
    ]
    for result in results:
        trace = result["trace"]
        actions = " → ".join(trace["action_sequence"]) or "(none)"
        path_pass = trace.get("path_pass")
        path_display = "n/a" if path_pass is None else str(path_pass).lower()
        lines.append(
            f"| {result['case_label']} ({result['case_type']}) | {trace['data_id']} | "
            f"{actions} | {path_display} | {str(trace['exact_match']).lower()} | "
            f"{trace['terminal_status']} |"
        )
    failure = results[-1]["trace"].get("failure_analysis", {})
    lines.extend([
        "",
        "## Failure case",
        "",
        f"- data_id: `{results[-1]['trace']['data_id']}`",
        f"- classified layer: `{failure.get('layer')}`",
        f"- rationale: {failure.get('rationale')}",
        f"- evidence paths: `{', '.join(failure.get('evidence_paths', []))}`",
        f"- infrastructure failure: `{str(failure.get('infrastructure_failure')).lower()}`",
        f"- terminal status: `{results[-1]['trace']['terminal_status']}`",
        "",
    ])
    return "\n".join(lines)


def write_failure_evidence(
    output_dir: Path,
    *,
    stage: str,
    error: str,
    completed_records: list[dict[str, Any]],
) -> None:
    atomic_write_json(
        output_dir / "step10_suite_failure.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "failed_at_utc": utc_now(),
            "stage": stage,
            "error": redact_text(error),
            "completed_case_artifacts": completed_records,
            "credentials_recorded": False,
        },
    )


def run_suite(args: argparse.Namespace) -> int:
    selection_manifest, selected_meta_dir, output_dir = validate_runtime_args(args)
    completed_records: list[dict[str, Any]] = []
    stage = "load_selection"
    try:
        samples, selections, dataset, selection_record = load_selected_samples(
            selection_manifest, selected_meta_dir
        )

        stage = "qwen3_health_and_service_snapshot"
        summarizer = Qwen3Summarizer(
            args.summary_cache_dir.resolve(),
            base_url=args.summary_base_url,
            model=args.summary_model,
            model_repo=SUMMARY_MODEL_REPO,
            model_revision=SUMMARY_MODEL_REVISION,
            api_key=None,
            max_input_chars=MAX_CHARS_PER_PAGE,
            max_tokens=SUMMARY_MAX_TOKENS,
            timeout_seconds=120,
        )
        initial_health = summarizer.health_check()
        require(valid_health(initial_health), "Initial Qwen3 health contract failed")
        service_root_pid = read_summary_service_pid()
        current_pid = os.getpid()
        service_snapshot = gpu_snapshot(
            "step10_summary_service_only",
            service_root_pid=service_root_pid,
            current_pid=current_pid,
        )
        service_gpu_identities = validate_service_only_snapshot(service_snapshot)

        stage = "load_mmsearch"
        torch.manual_seed(SEED)
        load_started = time.monotonic()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            local_files_only=True,
            use_fast=False,
        )
        load_seconds = round(time.monotonic() - load_started, 3)
        safe_device_map = validate_mmsearch_residency(model)
        model_snapshot = gpu_snapshot(
            "step10_mmsearch_loaded_same_gpu",
            service_root_pid=service_root_pid,
            current_pid=current_pid,
        )
        validate_joint_snapshot(
            model_snapshot,
            service_gpu_identities=service_gpu_identities,
            current_pid=current_pid,
        )

        image_search = FVQACachedImageSearch(
            args.cache_pickle,
            args.thumbnail_cache_dir,
            top_k=TOP_K,
        )
        text_search = SerperJinaTextSearch(
            args.serper_cache_dir,
            args.jina_cache_dir,
            top_k=TOP_K,
            max_chars_per_page=MAX_CHARS_PER_PAGE,
            summarizer=summarizer,
        )
        round_1_prompt = load_prompt("round_1_user_prompt_qwenvl.pkl").replace(
            "<image>", ""
        ).strip()
        after_image_search_prompt = load_prompt("after_image_search_prompt_qwenvl.pkl")
        after_text_search_prompt = load_prompt("after_text_search_prompt_qwenvl.pkl")

        results: list[dict[str, Any]] = []
        for label in CASE_ORDER:
            stage = f"case_{label}"
            torch.manual_seed(SEED)
            torch.cuda.reset_peak_memory_stats()
            result = execute_case(
                label=label,
                sample=samples[label],
                selection=selections[label],
                model=model,
                processor=processor,
                image_search=image_search,
                text_search=text_search,
                summarizer=summarizer,
                round_1_prompt=round_1_prompt,
                after_image_search_prompt=after_image_search_prompt,
                after_text_search_prompt=after_text_search_prompt,
            )
            case_health = summarizer.health_check()
            require(valid_health(case_health), f"Case {label} Qwen3 health contract failed")
            require(
                read_summary_service_pid() == service_root_pid,
                f"Case {label} Qwen3 service PID changed",
            )
            snapshot = gpu_snapshot(
                f"step10_case_{label}_complete_same_gpu",
                service_root_pid=service_root_pid,
                current_pid=current_pid,
            )
            validate_joint_snapshot(
                snapshot,
                service_gpu_identities=service_gpu_identities,
                current_pid=current_pid,
            )
            result["runtime"] = {
                "mmsearch_peak_allocated_mib": round(
                    torch.cuda.max_memory_allocated() / 1024**2, 2
                ),
                "gpu_snapshot_after_case": snapshot,
                "mmsearch_fully_gpu_resident": True,
                "summary_service_root_pid": service_root_pid,
                "summary_service_health_after_case": sanitize_public(case_health),
                "attention_implementation": model.config._attn_implementation,
                "parameter_dtype": str(next(model.parameters()).dtype),
            }
            result["summarizer"] = {
                "service_launch_contract": dict(SUMMARY_SERVICE_LAUNCH_CONTRACT),
                "config": public_summarizer_config(
                    summarizer, args.summary_cache_dir.resolve()
                ),
            }
            output_path = output_dir / CASE_FILENAMES[label]
            atomic_write_json(output_path, result)
            record = file_record(output_path)
            record.update({
                "case_label": label,
                "case_type": CASE_TYPES[label],
                "data_id": result["trace"]["data_id"],
                "action_sequence": result["trace"]["action_sequence"],
                "path_pass": result["trace"]["path_pass"],
                "exact_match": result["trace"]["exact_match"],
                "tool_infrastructure_success": result["trace"]["tool_infrastructure_success"],
            })
            completed_records.append(record)
            results.append(result)

            if label != "Failure" and result["trace"]["representative_pass"] is not True:
                raise RuntimeError(f"Representative case {label} path/infrastructure validation failed")
            if label == "Failure" and result["trace"]["failure_capture_pass"] is not True:
                raise RuntimeError("Failure case is not an EM-false infrastructure-success trace")

        stage = "final_health_and_manifest"
        final_health = summarizer.health_check()
        require(valid_health(final_health), "Final Qwen3 health contract failed")
        require(read_summary_service_pid() == service_root_pid, "Qwen3 service PID changed")
        final_snapshot = gpu_snapshot(
            "step10_suite_complete_same_gpu",
            service_root_pid=service_root_pid,
            current_pid=current_pid,
        )
        validate_joint_snapshot(
            final_snapshot,
            service_gpu_identities=service_gpu_identities,
            current_pid=current_pid,
        )

        cases_md_path = output_dir / "cases.md"
        atomic_write_text(cases_md_path, case_markdown(results))
        cases_md_record = file_record(cases_md_path)
        stage = "serialized_producer_scan"
        serialized_scan = scan_serialized_evidence(
            [Path(record["path"]) for record in completed_records],
            [cases_md_path],
        )
        require(
            serialized_scan["pass"] is True,
            "Serialized case/cases.md producer scan failed",
        )
        representative_pass_count = sum(
            result["trace"]["representative_pass"] is True
            for result in results[:4]
        )
        failure_capture_valid = valid_failure_capture(results[-1]["trace"])
        require(representative_pass_count == 4, "Representative producer count mismatch")
        require(failure_capture_valid, "Failure producer contract mismatch")
        stage = "write_suite_manifest"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "step": 10,
            "completed_at_utc": utc_now(),
            "selection_manifest": selection_record,
            "dataset": dataset,
            "models": {
                "mmsearch": {
                    "repository": MMSEARCH_REPO,
                    "revision": MMSEARCH_REVISION,
                    "local_path": str(DEFAULT_MMSEARCH_MODEL),
                },
                "summarizer": {
                    "repository": SUMMARY_MODEL_REPO,
                    "revision": SUMMARY_MODEL_REVISION,
                    "local_path": DEFAULT_SUMMARY_MODEL,
                },
            },
            "shared_runtime": {
                "mmsearch_load_seconds": load_seconds,
                "attention_implementation": model.config._attn_implementation,
                "parameter_dtype": str(next(model.parameters()).dtype),
                "mmsearch_gpu_residency": {
                    "hf_device_map": safe_device_map,
                    "parameter_devices": ["cuda:0"],
                    "fully_gpu_resident": True,
                },
                "summary_health_before": sanitize_public(initial_health),
                "summary_health_after": sanitize_public(final_health),
                "summary_service_root_pid": service_root_pid,
                "gpu_snapshots": [service_snapshot, model_snapshot, final_snapshot],
            },
            "case_artifacts": completed_records,
            "cases_markdown": cases_md_record,
            "representative_paths": {
                label: {
                    "expected": EXPECTED_ACTION_SEQUENCES[label],
                    "actual": results[index]["trace"]["action_sequence"],
                    "path_pass": results[index]["trace"]["path_pass"],
                    "exact_match": results[index]["trace"]["exact_match"],
                }
                for index, label in enumerate(("A", "B", "C", "D"))
            },
            "failure_case": {
                "data_id": results[-1]["trace"]["data_id"],
                "actual_action_sequence": results[-1]["trace"]["action_sequence"],
                "terminal_status": results[-1]["trace"]["terminal_status"],
                "exact_match": results[-1]["trace"]["exact_match"],
                "tool_infrastructure_success": results[-1]["trace"]["tool_infrastructure_success"],
                "analysis": results[-1]["trace"]["failure_analysis"],
                "capture_pass": results[-1]["trace"]["failure_capture_pass"],
            },
            "producer_checks": {
                "case_artifact_count": len(completed_records),
                "representative_path_pass_count": representative_pass_count,
                "required_representative_path_count": len(
                    EXPECTED_ACTION_SEQUENCES
                ),
                "failure_capture_valid": failure_capture_valid,
                "serialized_evidence_scan": serialized_scan,
            },
            "credentials_recorded": False,
        }
        manifest_path = output_dir / "step10_completion_manifest.json"
        atomic_write_json(manifest_path, manifest)
        print(json.dumps({
            "status": "passed",
            "output_dir": str(output_dir),
            "case_files": len(completed_records),
            "manifest": str(manifest_path),
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - suite evidence boundary
        write_failure_evidence(
            output_dir,
            stage=stage,
            error=safe_error(exc),
            completed_records=completed_records,
        )
        print(json.dumps({
            "status": "failed",
            "stage": stage,
            "error": safe_error(exc),
            "failure_evidence": str(output_dir / "step10_suite_failure.json"),
        }, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


def run_self_tests() -> None:
    require(
        normalized_candidate_answers('["A", "B"]') == ["A", "B"],
        "candidate answer normalization self-test failed",
    )
    require(
        normalized_candidate_answers("[]") == [],
        "empty candidate answer list self-test failed",
    )
    sanitized = sanitize_public({
        "content": "must disappear",
        "input_tokens": 3,
        "nested": {"messages": ["must disappear"], "ok": True},
    })
    require(sanitized == {"input_tokens": 3, "nested": {"ok": True}}, "sanitizer self-test failed")
    metrics = derive_network_metrics([
        {
            "tool": {
                "type": "fvqa_official_image_search_cache",
                "status": {"requested": 5, "cache_hits": 4},
            }
        },
        {
            "tool": {
                "type": "serper_dev_plus_jina_reader_plus_qwen3_summary",
                "status": {
                    "search": {"response_cache_hit": True},
                    "reader": {"requested": 5, "cache_hits": 3},
                    "summaries": [
                        {"cache_hit": True, "api_called": False},
                        {"cache_hit": False, "api_called": True},
                        {"cache_hit": True, "api_called": False},
                        {"cache_hit": True, "api_called": False},
                        {"cache_hit": True, "api_called": False},
                    ],
                },
            }
        },
    ])
    require(
        metrics["count_complete"] is True
        and metrics["external_network_requests"] == 3
        and metrics["local_qwen_completion_requests"] == 1
        and metrics["tool_http_requests_total"] == 4,
        "network metrics self-test failed",
    )
    failure_trace = {
        "final_answer": "wrong",
        "ground_truth": "right",
        "exact_match": False,
        "terminal_status": "answered",
        "tool_infrastructure_success": True,
        "network_and_cache": {"count_complete": True},
        "rounds": [{"action": "answer"}],
        "hard_failures": [],
    }
    require(
        classify_failure_layer(failure_trace)["layer"] == "final_answer",
        "failure layer self-test failed",
    )
    require(valid_failure_capture(failure_trace), "failure capture self-test failed")
    require(
        strict_exact_normalize("  A  B  ") != strict_exact_normalize("a b"),
        "strict EM whitespace self-test failed",
    )
    require(
        validate_image_status({
            "success": True,
            "requested": 5,
            "num_images": 4,
            "cache_hits": 4,
            "titles": ["a", "b", "c", "d"],
            "thumbnail_urls": ["a", "b", "c", "d"],
            "local_thumbnail_paths": ["a", "b", "c", "d"],
            "failures": [{}],
        }),
        "4-of-5 image contract self-test failed",
    )
    require(
        EXPECTED_ACTION_SEQUENCES["D"]
        == ["image_search", "text_search", "answer"],
        "representative path self-test failed",
    )
    print(json.dumps({"status": "passed", "pure_function_tests": 9}, indent=2))


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_tests()
        return 0
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
