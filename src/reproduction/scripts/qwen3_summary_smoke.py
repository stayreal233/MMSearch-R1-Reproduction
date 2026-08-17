#!/usr/bin/env python3
"""Summarize rank 1 from the existing raw-Jina trace without web access."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from reproduction.mmsearch_tools.qwen3_summarizer import Qwen3Summarizer


MODEL_REPO = "Qwen/Qwen3-32B-FP8"
MODEL_REVISION = "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
MODEL_PATH = "/root/autodl-tmp/models/Qwen3-32B-FP8"
JINA_CACHE_ROOT = Path("/root/autodl-tmp/search_cache/jina")
FORBIDDEN_METADATA_FIELDS = {
    "api_key",
    "authorization",
    "body",
    "content",
    "headers",
    "input_text",
    "messages",
    "page_content",
    "page_text",
    "prompt",
    "raw_content",
    "request_body",
    "response_body",
    "webpage_content",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local Qwen3 smoke test using only the cached rank-1 Jina page."
    )
    parser.add_argument(
        "--raw-trace",
        type=Path,
        default=Path("/root/autodl-tmp/outputs/real_search_flow_raw_jina.json"),
    )
    parser.add_argument(
        "--summary-cache-dir",
        type=Path,
        default=Path("/root/autodl-tmp/search_cache/qwen3_summary"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/root/autodl-tmp/outputs/qwen3_summary_smoke.json"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--max-input-chars", type=int, default=12_000)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def public_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): public_metadata(item)
            for key, item in value.items()
            if str(key).casefold() not in FORBIDDEN_METADATA_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [public_metadata(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def load_rank_one(raw_trace_path: Path, max_input_chars: int) -> tuple[str, dict[str, Any]]:
    try:
        result = json.loads(raw_trace_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read raw-Jina trace: {type(exc).__name__}") from exc
    require(result.get("pass") is True, "The source raw-Jina trace is not a passing trace")
    require(
        result.get("mode") == "cached_image_search_plus_serper_dev_plus_raw_jina",
        "Unexpected source trace mode",
    )
    trace = result.get("trace")
    require(isinstance(trace, dict), "Source trace payload is missing")
    require(trace.get("data_id") == "fvqa_train_17", "Smoke source must be fvqa_train_17")
    rounds = trace.get("rounds")
    require(isinstance(rounds, list), "Source trace rounds are missing")
    text_rounds = [item for item in rounds if item.get("action") == "text_search"]
    require(len(text_rounds) == 1, "Source trace must contain exactly one text-search round")
    text_round = text_rounds[0]
    query = text_round.get("query")
    require(isinstance(query, str) and query.strip(), "Source trace query is missing")
    tool = text_round.get("tool")
    require(isinstance(tool, dict), "Source text-search tool record is missing")
    require(tool.get("type") == "serper_dev_plus_jina_reader", "Unexpected source tool")
    status = tool.get("status")
    require(isinstance(status, dict) and status.get("success") is True, "Source tool failed")
    require(
        status.get("search", {}).get("response_cache_hit") is True,
        "Source Serper result was not a cache hit",
    )
    documents = status.get("documents")
    require(isinstance(documents, list), "Source Jina document metadata is missing")
    rank_one = [item for item in documents if item.get("rank") == 1]
    require(len(rank_one) == 1, "Source trace must contain exactly one rank-1 page")
    metadata = rank_one[0]

    cache_path_value = metadata.get("cache_path")
    require(isinstance(cache_path_value, str), "Rank-1 Jina cache path is missing")
    cache_path = Path(cache_path_value).resolve(strict=True)
    cache_root = JINA_CACHE_ROOT.resolve(strict=True)
    try:
        cache_path.relative_to(cache_root)
    except ValueError as exc:
        raise RuntimeError("Rank-1 cache path escapes the pinned Jina cache root") from exc
    require(cache_path.suffix == ".md", "Rank-1 Jina cache is not a Markdown cache file")

    cache_metadata_path = cache_path.with_suffix(".json")
    cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    require(cache_metadata.get("target_url") == metadata.get("url"), "Jina URL mismatch")
    cached_page = cache_path.read_text(encoding="utf-8")
    expected_full_chars = cache_metadata.get("characters")
    require(isinstance(expected_full_chars, int), "Jina cache character metadata is missing")
    if len(cached_page) == expected_full_chars + 1 and cached_page.endswith("\n"):
        cached_page = cached_page[:-1]
    require(len(cached_page) == expected_full_chars, "Jina cache character count mismatch")
    require(metadata.get("full_characters") == len(cached_page), "Trace/cache length mismatch")
    returned_chars = metadata.get("returned_characters")
    require(
        isinstance(returned_chars, int) and 0 < returned_chars <= max_input_chars,
        "Invalid raw-trace returned character count",
    )
    bounded_page = cached_page[:returned_chars]
    require(len(bounded_page) == returned_chars, "Rank-1 cached page is truncated unexpectedly")
    document = {
        "rank": 1,
        "title": metadata.get("title"),
        "url": metadata.get("url"),
        "snippet": metadata.get("snippet", ""),
        "content": bounded_page,
        "full_characters": len(cached_page),
        "returned_characters": len(bounded_page),
        "truncated": len(bounded_page) < len(cached_page),
        "cache_hit": True,
        "cache_path": str(cache_path),
        "http_status": metadata.get("http_status"),
    }
    return query.strip(), document


def thinking_is_disabled(stat: dict[str, Any]) -> bool:
    for name in ("enable_thinking", "thinking_enabled"):
        if name in stat:
            return stat[name] is False
    settings = stat.get("settings")
    if isinstance(settings, dict):
        for name in ("enable_thinking", "thinking_enabled"):
            if name in settings:
                return settings[name] is False
    return False


def main() -> None:
    args = parse_args()
    require(args.max_input_chars == 12_000, "Smoke max_input_chars is pinned to 12000")
    require(args.max_tokens == 512, "Smoke max_tokens is pinned to 512")
    parsed_base_url = urlparse(args.base_url)
    require(parsed_base_url.scheme == "http", "Summary service must use local HTTP")
    require(
        parsed_base_url.hostname in {"127.0.0.1", "localhost"}
        and parsed_base_url.port == 8001,
        "Summary service must be the pinned loopback port 8001",
    )
    query, document = load_rank_one(args.raw_trace.resolve(), args.max_input_chars)
    summarizer = Qwen3Summarizer(
        args.summary_cache_dir.resolve(),
        base_url=args.base_url,
        model=MODEL_PATH,
        model_repo=MODEL_REPO,
        model_revision=MODEL_REVISION,
        api_key=None,
        max_input_chars=args.max_input_chars,
        max_tokens=args.max_tokens,
        timeout_seconds=120,
    )
    health = summarizer.health_check()
    require(
        health.get("success") is True and health.get("model_available") is True,
        "Pinned Qwen3 model is unavailable from /v1/models",
    )
    require(
        health.get("thinking_enabled") is False,
        "Health check did not prove Thinking is disabled",
    )
    summary, stat = summarizer.summarize_document(query, document)
    require(isinstance(summary, str) and summary.strip(), "Qwen3 returned an empty summary")
    require(isinstance(stat, dict), "Qwen3 summarizer did not return public statistics")
    require(thinking_is_disabled(stat), "Qwen3 summary did not prove Thinking is disabled")
    require("<think>" not in summary.casefold(), "Qwen3 summary contains a thinking block")
    require("</think>" not in summary.casefold(), "Qwen3 summary contains a thinking block")

    payload = {
        "schema_version": 1,
        "mode": "qwen3_summary_cached_raw_jina_rank1_smoke",
        "network_scope": "loopback_qwen3_only",
        "source_trace": str(args.raw_trace.resolve()),
        "source": {
            "data_id": "fvqa_train_17",
            "rank": document["rank"],
            "title": document["title"],
            "url": document["url"],
            "jina_cache_path": document["cache_path"],
            "jina_full_characters": document["full_characters"],
            "summarizer_input_characters": document["returned_characters"],
        },
        "service_health": public_metadata(health),
        "summary": summary.strip(),
        "summary_stat": public_metadata(stat),
        "checks": {
            "source_trace_passed": True,
            "serper_network_called": False,
            "jina_network_called": False,
            "jina_cache_read_only": True,
            "thinking_disabled": True,
            "raw_webpage_body_omitted": True,
            "credential_omitted": True,
        },
        "pass": True,
    }
    encoded = json.dumps(payload, ensure_ascii=False)
    require(document["content"] not in encoded, "Raw webpage body leaked into smoke output")
    atomic_write_json(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "status": "passed",
                "rank": 1,
                "summary_characters": len(summary.strip()),
                "summary_cache_hit": stat.get("cache_hit"),
                "thinking_disabled": True,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
