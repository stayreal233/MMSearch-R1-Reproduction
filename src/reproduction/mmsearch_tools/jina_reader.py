"""Jina Reader adapter with per-URL caching and bounded webpage text."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urlparse

import requests


class JinaReader:
    endpoint = "https://r.jina.ai"

    def __init__(
        self,
        cache_dir: Path,
        *,
        max_chars_per_page: int = 12_000,
        timeout_seconds: float = 60.0,
        max_download_bytes: int = 5 * 1024 * 1024,
        max_workers: int = 5,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_chars_per_page = max_chars_per_page
        self.timeout_seconds = timeout_seconds
        self.max_download_bytes = max_download_bytes
        self.max_workers = max_workers
        if self.max_chars_per_page < 1:
            raise ValueError("max_chars_per_page must be positive")

    @staticmethod
    def _normalize_url(url: str) -> str:
        normalized, _ = urldefrag(url.strip())
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Jina Reader target must be a public HTTP(S) URL")
        return normalized

    def _cache_paths(self, url: str) -> tuple[Path, Path]:
        signature = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return (
            self.cache_dir / f"{signature}.md",
            self.cache_dir / f"{signature}.json",
        )

    def _load_or_fetch(self, url: str) -> tuple[str, Path, bool, int]:
        normalized_url = self._normalize_url(url)
        content_path, metadata_path = self._cache_paths(normalized_url)
        if content_path.exists():
            text = content_path.read_text(encoding="utf-8")
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                expected_characters = metadata.get("characters")
                if isinstance(expected_characters, int):
                    if len(text) == expected_characters + 1 and text.endswith("\n"):
                        # Compatibility with caches written before the exact-length fix.
                        text = text[:expected_characters]
                    elif len(text) != expected_characters:
                        raise RuntimeError(
                            "Cached Jina content length does not match its metadata"
                        )
            if text.strip():
                return text, content_path, True, 200

        reader_url = f"{self.endpoint}/{normalized_url}"
        try:
            response = requests.get(
                reader_url,
                headers={
                    "Accept": "text/plain",
                    "User-Agent": "MMSearch-R1-reproduction/1.0",
                },
                timeout=(5.0, self.timeout_seconds),
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Jina Reader transport failed: {type(exc).__name__}"
            ) from exc
        if response.status_code != 200:
            raise RuntimeError(f"Jina Reader failed (HTTP {response.status_code})")
        content = response.content
        if len(content) > self.max_download_bytes:
            raise RuntimeError(
                f"Jina Reader response exceeds {self.max_download_bytes} bytes"
            )
        text = response.text.replace("\x00", "").strip()
        if not text:
            raise RuntimeError("Jina Reader returned empty content")

        content_path.write_text(text, encoding="utf-8")
        metadata_path.write_text(
            json.dumps(
                {
                    "target_url": normalized_url,
                    "reader_endpoint": self.endpoint,
                    "http_status": response.status_code,
                    "characters": len(text),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return text, content_path, False, response.status_code

    def _read_one(self, result: dict[str, Any]) -> dict[str, Any]:
        full_text, cache_path, cache_hit, http_status = self._load_or_fetch(result["url"])
        bounded_text = full_text[: self.max_chars_per_page]
        return {
            "rank": result["rank"],
            "title": result["title"],
            "url": result["url"],
            "snippet": result.get("snippet", ""),
            "content": bounded_text,
            "full_characters": len(full_text),
            "returned_characters": len(bounded_text),
            "truncated": len(bounded_text) < len(full_text),
            "cache_hit": cache_hit,
            "cache_path": str(cache_path),
            "http_status": http_status,
        }

    def read_many(
        self,
        search_results: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        if not search_results:
            raise ValueError("search_results must not be empty")
        documents: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        worker_count = min(self.max_workers, len(search_results))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(self._read_one, result): result
                for result in search_results
            }
            for future in as_completed(futures):
                result = futures[future]
                try:
                    documents.append(future.result())
                except Exception as exc:
                    failures.append(
                        {
                            "rank": result.get("rank"),
                            "title": result.get("title"),
                            "url": result.get("url"),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        documents.sort(key=lambda item: item["rank"])
        failures.sort(key=lambda item: item.get("rank") or 0)
        if not documents:
            raise RuntimeError("Jina Reader failed for every Serper result")

        lines = [
            "[Webpage Contents] Jina Reader output, ranked by Serper relevance. "
            f"Each page is limited to {self.max_chars_per_page} characters."
        ]
        for document in documents:
            lines.extend(
                [
                    "",
                    f"=== Result {document['rank']} ===",
                    f"Title: {document['title']}",
                    f"URL: {document['url']}",
                    f"Search snippet: {document['snippet']}",
                    "Webpage content:",
                    document["content"],
                ]
            )
        returned_text = "\n".join(lines).strip() + "\n"
        tool_stat = {
            "success": True,
            "source": "jina_reader",
            "endpoint": self.endpoint,
            "requested": len(search_results),
            "num_documents": len(documents),
            "cache_hits": sum(document["cache_hit"] for document in documents),
            "max_chars_per_page": self.max_chars_per_page,
            "returned_characters": sum(
                document["returned_characters"] for document in documents
            ),
            "failures": failures,
        }
        return returned_text, documents, tool_stat
