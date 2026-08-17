#!/usr/bin/env python3
"""Freeze the deterministic 50-example Step-11 FVQA evaluation set."""

from __future__ import annotations

import collections
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

import prepare_step10_suite_inputs as bridge


PROTOCOL = Path(
    "/root/autodl-tmp/multimodal-search-r1/reproduction/env/step11_eval_protocol.json"
)
PROTOCOL_SHA256 = "f2fc533b824c65d5102fc10dbaebe0c3069242b00f9178c1417f6f0935c6000e"
PARQUET = Path("/root/autodl-tmp/datasets/FVQA/fvqa_train.parquet")
PARQUET_SHA256 = "d23be97f4493846381f71c6953a29777fe1522aaf37942a26393605ffd78171f"
IMAGE_CACHE = Path(
    "/root/autodl-tmp/datasets/FVQA/fvqa_train_image_search_results_cache.pkl"
)
OUTPUT_DIR = Path("/root/autodl-tmp/mmsearch_step11_inputs")
MANIFEST = OUTPUT_DIR / "eval_manifest.json"
EXCLUDED = {
    "fvqa_train_0",
    "fvqa_train_6",
    "fvqa_train_9",
    "fvqa_train_17",
    "fvqa_train_32",
}
CATEGORIES = ("search_free", "search_required")
TARGET_PER_CATEGORY = 25
RANK_PREFIX = "mmsearch-step11-seed0|"
STAGES = (5, 20, 50)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_record(path: Path, encoded: bytes) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": len(encoded),
        "sha256": bridge.sha256_bytes(encoded),
    }


def rank_digest(data_id: str) -> str:
    return hashlib.sha256(f"{RANK_PREFIX}{data_id}".encode("utf-8")).hexdigest()


def eligible_cache_entry(cache: dict[str, Any], data_id: str) -> bool:
    entry = cache.get(data_id)
    return (
        isinstance(entry, dict)
        and len(entry.get("tool_returned_web_title_list", [])) >= 5
        and len(entry.get("tool_returned_images_urls", [])) >= 5
    )


def interleave(
    search_free: list[dict[str, Any]],
    search_required: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    require(len(search_free) == len(search_required), "balanced lists are required")
    result: list[dict[str, Any]] = []
    for free, required in zip(search_free, search_required, strict=True):
        result.extend((free, required))
    return result


def load_protocol() -> tuple[dict[str, Any], bytes]:
    require(PROTOCOL.is_file() and not PROTOCOL.is_symlink(), "unsafe Step-11 protocol")
    encoded = bridge.read_regular_file(PROTOCOL, maximum=256 * 1024)
    require(bridge.sha256_bytes(encoded) == PROTOCOL_SHA256, "Step-11 protocol SHA mismatch")
    value = json.loads(encoded)
    require(isinstance(value, dict), "Step-11 protocol root is invalid")
    require(value.get("status") == "registered_before_step11_selection_and_inference", "Step-11 protocol status mismatch")
    selection = value.get("selection")
    require(isinstance(selection, dict), "Step-11 selection contract missing")
    require(selection.get("target_per_category") == TARGET_PER_CATEGORY, "category target mismatch")
    require(selection.get("excluded_data_ids") == sorted(EXCLUDED, key=lambda item: int(item.rsplit("_", 1)[1])), "exclusion contract mismatch")
    require(value.get("document_contract", {}).get("stages") == list(STAGES), "stage contract mismatch")
    return value, encoded


def load_cache() -> tuple[dict[str, Any], bytes]:
    require(IMAGE_CACHE.is_file() and not IMAGE_CACHE.is_symlink(), "unsafe FVQA image cache")
    encoded = bridge.read_regular_file(IMAGE_CACHE, maximum=2 * 1024 * 1024 * 1024)
    cache = pickle.loads(encoded)
    require(isinstance(cache, dict), "FVQA image cache root is invalid")
    require(len(cache) == 4849, "FVQA train image cache entry count mismatch")
    return cache, encoded


def select_rows(cache: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    columns = ["prompt", "images", "reward_model", "data_source", "data_id", "category"]
    pools: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORIES}
    row_index = 0
    seen: set[str] = set()
    parquet = pq.ParquetFile(PARQUET)
    for batch in parquet.iter_batches(batch_size=64, columns=columns):
        for row in batch.to_pylist():
            current = row_index
            row_index += 1
            data_id = row.get("data_id")
            category = row.get("category")
            if category not in pools or not isinstance(data_id, str) or not data_id:
                continue
            require(data_id not in seen, f"duplicate FVQA data_id: {data_id}")
            seen.add(data_id)
            if data_id in EXCLUDED or not eligible_cache_entry(cache, data_id):
                continue
            reward = row.get("reward_model")
            require(isinstance(reward, dict), f"invalid reward_model: {data_id}")
            require(isinstance(reward.get("ground_truth"), str) and reward["ground_truth"].strip(), f"invalid ground truth: {data_id}")
            question = bridge.question_from_prompt(row.get("prompt"))
            raw_image = bridge.row_image(row)
            require(raw_image, f"empty image: {data_id}")
            pools[category].append({
                "data_id": data_id,
                "category": category,
                "source_row_index": current,
                "rank_sha256": rank_digest(data_id),
                "row": row,
                "question": question,
                "raw_image": raw_image,
            })
    require(row_index == 4856, "FVQA train row count mismatch")
    selected_by_category: dict[str, list[dict[str, Any]]] = {}
    eligible_counts: dict[str, int] = {}
    for category in CATEGORIES:
        ranked = sorted(
            pools[category],
            key=lambda item: (item["rank_sha256"], item["source_row_index"]),
        )
        eligible_counts[category] = len(ranked)
        require(len(ranked) >= TARGET_PER_CATEGORY, f"insufficient eligible {category} rows")
        selected_by_category[category] = ranked[:TARGET_PER_CATEGORY]
    return interleave(
        selected_by_category["search_free"],
        selected_by_category["search_required"],
    ), eligible_counts


def prepare() -> dict[str, Any]:
    require(not OUTPUT_DIR.exists() and not OUTPUT_DIR.is_symlink(), "Step-11 input directory already exists")
    require(PARQUET.is_file() and not PARQUET.is_symlink(), "unsafe FVQA parquet")
    protocol, protocol_bytes = load_protocol()
    parquet_size, parquet_sha = bridge.hash_file(PARQUET)
    require(parquet_sha == PARQUET_SHA256, "FVQA parquet SHA mismatch")
    cache, cache_bytes = load_cache()
    selected, eligible_counts = select_rows(cache)
    require(len(selected) == 50, "Step-11 selected count mismatch")
    require(len({item["data_id"] for item in selected}) == 50, "Step-11 selected IDs are not unique")
    prefix_counts = {
        str(stage): dict(collections.Counter(item["category"] for item in selected[:stage]))
        for stage in STAGES
    }
    require(prefix_counts == {
        "5": {"search_free": 3, "search_required": 2},
        "20": {"search_free": 10, "search_required": 10},
        "50": {"search_free": 25, "search_required": 25},
    }, "Step-11 stage prefix balance mismatch")

    examples: list[dict[str, Any]] = []
    payloads: list[tuple[Path, bytes]] = []
    for eval_index, item in enumerate(selected, start=1):
        data_id = item["data_id"]
        normalized, width, height = bridge.normalized_png(item["raw_image"])
        image_path = OUTPUT_DIR / "images" / f"{eval_index:03d}_{data_id}.png"
        meta_path = OUTPUT_DIR / "metadata" / f"{eval_index:03d}_{data_id}.json"
        sample = bridge.suite_sample(
            row=item["row"],
            row_index=item["source_row_index"],
            image_path=image_path,
            image_bytes=normalized,
            source_sha256=bridge.sha256_bytes(item["raw_image"]),
        )
        sample.update({
            "eval_index": eval_index,
            "selection_rank_sha256": item["rank_sha256"],
            "execution_mode": "natural",
            "controller_intervention": False,
        })
        meta_bytes = bridge.pretty_json_bytes(sample)
        payloads.extend(((image_path, normalized), (meta_path, meta_bytes)))
        examples.append({
            "eval_index": eval_index,
            "data_id": data_id,
            "category": item["category"],
            "source_row_index": item["source_row_index"],
            "selection_rank_sha256": item["rank_sha256"],
            "image": file_record(image_path, normalized),
            "metadata": file_record(meta_path, meta_bytes),
            "image_cache_top5_eligible": True,
        })

    manifest = {
        "schema_version": "mmsearch.step11.eval-inputs.v1",
        "status": "passed",
        "selection_rule": protocol["selection"],
        "dataset": protocol["dataset"],
        "protocol": file_record(PROTOCOL, protocol_bytes),
        "sources": {
            "parquet": {"path": str(PARQUET), "bytes": parquet_size, "sha256": parquet_sha},
            "image_cache": file_record(IMAGE_CACHE, cache_bytes),
        },
        "eligible_counts_after_exclusions_and_cache_filter": eligible_counts,
        "excluded_data_ids": sorted(EXCLUDED, key=lambda item: int(item.rsplit("_", 1)[1])),
        "examples": examples,
        "stage_prefix_counts": prefix_counts,
        "checks": {
            "selection_before_inference": True,
            "selection_independent_of_outputs_and_correctness": True,
            "balanced_25_plus_25": True,
            "all_ids_unique": True,
            "all_images_and_metadata_hashed": True,
            "all_official_image_cache_top5_eligible": True,
            "controller_interventions": False,
            "credentials_read": False,
            "network_used": False,
            "model_inference_used": False,
        },
        "credentials_recorded": False,
    }
    manifest_bytes = bridge.pretty_json_bytes(manifest)
    OUTPUT_DIR.mkdir(mode=0o700, parents=False, exist_ok=False)
    (OUTPUT_DIR / "images").mkdir(mode=0o700)
    (OUTPUT_DIR / "metadata").mkdir(mode=0o700)
    for path, encoded in payloads:
        bridge.atomic_write(path, encoded)
    bridge.atomic_write(MANIFEST, manifest_bytes)
    require(bridge.read_regular_file(MANIFEST, maximum=4 * 1024 * 1024) == manifest_bytes, "Step-11 manifest commit verification failed")
    return {
        "status": "passed",
        "manifest": str(MANIFEST),
        "manifest_sha256": bridge.sha256_bytes(manifest_bytes),
        "examples": len(examples),
        "stage_prefix_counts": prefix_counts,
    }


def self_test() -> None:
    a = [{"data_id": "a1"}, {"data_id": "a2"}]
    b = [{"data_id": "b1"}, {"data_id": "b2"}]
    require([item["data_id"] for item in interleave(a, b)] == ["a1", "b1", "a2", "b2"], "interleave test failed")
    require(rank_digest("x") == rank_digest("x") and rank_digest("x") != rank_digest("y"), "rank test failed")
    require(not (EXCLUDED & {"fvqa_train_1"}), "exclusion test failed")
    print(json.dumps({"status": "passed", "pure_self_tests": 3}, sort_keys=True))


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
