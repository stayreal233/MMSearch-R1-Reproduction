#!/usr/bin/env python3
"""Fetch and validate one official FVQA cached image-search entry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from reproduction.mmsearch_tools.cached_image_search import FVQACachedImageSearch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-pickle", type=Path, required=True)
    parser.add_argument("--thumbnail-cache-dir", type=Path, required=True)
    parser.add_argument("--data-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search = FVQACachedImageSearch(
        args.cache_pickle,
        args.thumbnail_cache_dir,
        top_k=5,
    )
    returned_text, images, tool_stat = search(args.data_id)
    image_info = []
    for image, local_path in zip(images, tool_stat["local_thumbnail_paths"]):
        path = Path(local_path) if local_path else None
        image_info.append(
            {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "local_path": local_path,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path else None,
            }
        )
    result = {
        "data_id": args.data_id,
        "returned_text": returned_text,
        "tool_stat": tool_stat,
        "images": image_info,
        "pass": (
            tool_stat["success"]
            and len(images) > 0
            and len(images) == len(tool_stat["titles"])
            and all(not title.startswith("Webpage Title ") for title in tool_stat["titles"])
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
