#!/usr/bin/env python3
"""Extract deterministic search-free and search-required FVQA examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


TARGET_CATEGORIES = ("search_free", "search_required")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/FVQA/fvqa_train.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/autodl-tmp/mmsearch_demo"),
    )
    parser.add_argument(
        "--dataset-revision",
        default="bb4a4ff4c9c3fd0382d11f5d7fccd66d0b8428b5",
    )
    return parser.parse_args()


def user_question(prompt: list[dict[str, str]]) -> str:
    for message in prompt:
        if message.get("role") == "user":
            return message["content"]
    if prompt:
        return prompt[0]["content"]
    raise ValueError("Sample has no prompt messages")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "prompt",
        "images",
        "reward_model",
        "data_source",
        "data_id",
        "category",
    ]
    selected: dict[str, tuple[int, dict]] = {}
    row_index = 0

    parquet = pq.ParquetFile(args.parquet)
    for batch in parquet.iter_batches(batch_size=64, columns=columns):
        for row in batch.to_pylist():
            category = row["category"]
            if category in TARGET_CATEGORIES and category not in selected:
                selected[category] = (row_index, row)
            row_index += 1
            if all(category in selected for category in TARGET_CATEGORIES):
                break
        if all(category in selected for category in TARGET_CATEGORIES):
            break

    missing = set(TARGET_CATEGORIES) - selected.keys()
    if missing:
        raise RuntimeError(f"Missing FVQA categories: {sorted(missing)}")

    metadata: dict[str, dict] = {}
    for category in TARGET_CATEGORIES:
        source_row_index, row = selected[category]
        images = row["images"]
        if not images or not images[0] or not images[0].get("bytes"):
            raise ValueError(f"{row['data_id']} has no embedded image bytes")

        raw_image = images[0]["bytes"]
        raw_sha256 = hashlib.sha256(raw_image).hexdigest()
        with Image.open(BytesIO(raw_image)) as image:
            image.load()
            image = image.convert("RGB")
            width, height = image.size
            image_path = args.output_dir / f"{category}.png"
            image.save(image_path, format="PNG", compress_level=6)

        metadata[category] = {
            "data_id": row["data_id"],
            "category": category,
            "source_split": "train",
            "source_row_index": source_row_index,
            "data_source": row["data_source"],
            "image": str(image_path),
            "image_width": width,
            "image_height": height,
            "source_image_sha256": raw_sha256,
            "question": user_question(row["prompt"]),
            "reward_model": row["reward_model"],
        }

    meta_path = args.output_dir / "meta.json"
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "dataset_id": "lmms-lab/FVQA",
        "dataset_revision": args.dataset_revision,
        "source_parquet": str(args.parquet),
        "selection": "first row in parquet order for each target category",
        "categories": list(TARGET_CATEGORIES),
        "metadata": str(meta_path),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
