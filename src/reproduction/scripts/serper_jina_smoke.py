#!/usr/bin/env python3
"""Run one Serper.dev text search and fetch its result pages with Jina Reader."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from reproduction.mmsearch_tools.real_text_search import SerperJinaTextSearch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--serper-cache-dir", type=Path, required=True)
    parser.add_argument("--jina-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-chars-per-page", type=int, default=12_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    key_present = bool(os.environ.get("SERPER_API_KEY"))
    if not key_present:
        result = {
            "mode": "serper_dev_plus_jina_reader_smoke",
            "api_key_present": False,
            "api_key_value_logged": False,
            "status": "blocked_missing_SERPER_API_KEY",
            "pass": False,
        }
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(4)

    tool = SerperJinaTextSearch(
        args.serper_cache_dir,
        args.jina_cache_dir,
        top_k=args.top_k,
        max_chars_per_page=args.max_chars_per_page,
    )
    try:
        returned_text, tool_stat = tool(args.query)
    except Exception as exc:
        result = {
            "mode": "serper_dev_plus_jina_reader_smoke",
            "api_key_present": True,
            "api_key_value_logged": False,
            "query": args.query,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "pass": False,
        }
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(2) from exc

    result = {
        "mode": "serper_dev_plus_jina_reader_smoke",
        "api_key_present": True,
        "api_key_value_logged": False,
        "query": args.query,
        "returned_text": returned_text,
        "tool_stat": tool_stat,
        "status": "completed",
        "pass": (
            tool_stat["success"]
            and tool_stat["search"]["num_results"] == args.top_k
            and tool_stat["reader"]["num_documents"] >= 1
        ),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": result["status"],
        "serper_results": tool_stat["search"]["num_results"],
        "serper_cache_hit": tool_stat["search"]["response_cache_hit"],
        "jina_documents": tool_stat["reader"]["num_documents"],
        "jina_failures": len(tool_stat["reader"]["failures"]),
        "jina_cache_hits": tool_stat["reader"]["cache_hits"],
        "returned_characters": tool_stat["reader"]["returned_characters"],
        "output": str(args.output),
        "pass": result["pass"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
