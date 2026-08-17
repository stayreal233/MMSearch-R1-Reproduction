#!/usr/bin/env python3
"""Validate a SerpApi account and run one cached Google Lens request."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from reproduction.mmsearch_tools.serpapi_lens import SerpAPIGoogleLensSearch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-url", required=True)
    parser.add_argument("--search-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    key_present = bool(os.environ.get("SERPAPI_API_KEY"))
    base_result = {
        "api_key_present": key_present,
        "api_key_value_logged": False,
        "image_url": args.image_url,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not key_present:
        result = {
            **base_result,
            "status": "blocked_missing_SERPAPI_API_KEY",
            "pass": False,
        }
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(4)

    search = SerpAPIGoogleLensSearch(
        args.search_cache_dir,
        top_k=args.top_k,
    )
    account = search.account_summary()
    if args.check_only:
        result = {
            **base_result,
            "status": "account_check_passed",
            "account": account,
            "pass": True,
        }
    else:
        returned_text, images, tool_stat = search(args.image_url)
        image_info = []
        for image, local_path in zip(images, tool_stat["local_thumbnail_paths"]):
            path = Path(local_path)
            image_info.append(
                {
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "local_path": local_path,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        result = {
            **base_result,
            "status": "search_completed",
            "account": account,
            "returned_text": returned_text,
            "tool_stat": tool_stat,
            "images": image_info,
            "pass": tool_stat["success"] and len(images) > 0,
        }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
