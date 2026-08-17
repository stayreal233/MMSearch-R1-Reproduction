"""Serper.dev text-search adapter with secret-free local response caching."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


def _is_sensitive_field(name: str) -> bool:
    normalized = "".join(character for character in name.lower() if character.isalnum())
    return normalized in {
        "apikey",
        "authorization",
        "accountemail",
        "accountid",
        "token",
    }


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize(item)
            for key, item in value.items()
            if not _is_sensitive_field(str(key))
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


class SerperTextSearch:
    """Query Serper's Google Search endpoint and return ranked organic results."""

    endpoint = "https://google.serper.dev/search"

    def __init__(
        self,
        cache_dir: Path,
        *,
        api_key: str | None = None,
        top_k: int = 5,
        timeout_seconds: float = 30.0,
        country: str = "us",
        language: str = "en",
    ) -> None:
        self.api_key = api_key or os.environ.get("SERPER_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.timeout_seconds = timeout_seconds
        self.country = country
        self.language = language
        if self.top_k < 1:
            raise ValueError("top_k must be positive")

    def _public_request(self, query: str) -> tuple[dict[str, Any], str]:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("Serper query must not be empty")
        request = {
            "q": normalized_query,
            "gl": self.country,
            "hl": self.language,
            "num": self.top_k,
        }
        serialized = json.dumps(request, sort_keys=True, separators=(",", ":"))
        signature = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return request, signature

    def _load_or_query(
        self,
        query: str,
    ) -> tuple[dict[str, Any], Path, bool, dict[str, Any]]:
        public_request, signature = self._public_request(query)
        cache_path = self.cache_dir / f"{signature}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("Cached Serper response is not a JSON object")
            return payload, cache_path, True, public_request
        if not self.api_key:
            raise RuntimeError(
                "SERPER_API_KEY is not set and no matching response cache exists"
            )

        try:
            response = requests.post(
                self.endpoint,
                headers={
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json",
                },
                json=public_request,
                timeout=(5.0, self.timeout_seconds),
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Serper request transport failed: {type(exc).__name__}"
            ) from exc
        if response.status_code != 200:
            raise RuntimeError(f"Serper text search failed (HTTP {response.status_code})")
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("Serper returned HTTP 200 with non-JSON content") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Serper returned a non-object JSON response")
        if payload.get("error"):
            raise RuntimeError("Serper returned an API error")

        sanitized = _sanitize(payload)
        cache_path.write_text(
            json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return sanitized, cache_path, False, public_request

    @staticmethod
    def _valid_public_url(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def __call__(
        self,
        query: str,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        payload, cache_path, cache_hit, request = self._load_or_query(query)
        organic = payload.get("organic", [])
        if not isinstance(organic, list):
            raise RuntimeError("Serper response field 'organic' is not a list")

        results: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for response_index, item in enumerate(organic):
            if len(results) >= self.top_k:
                break
            if not isinstance(item, dict) or not self._valid_public_url(item.get("link")):
                skipped.append(
                    {
                        "response_index": response_index,
                        "reason": "organic result has no valid HTTP(S) link",
                    }
                )
                continue
            results.append(
                {
                    "rank": len(results) + 1,
                    "position": item.get("position"),
                    "title": str(item.get("title") or "Untitled"),
                    "url": item["link"],
                    "snippet": str(item.get("snippet") or ""),
                }
            )
        if not results:
            raise RuntimeError("Serper response contains no usable organic results")

        lines = ["[Serper Text Search Results] Organic results ranked by relevance:"]
        for result in results:
            lines.extend(
                [
                    f"{result['rank']}. {result['title']}",
                    f"URL: {result['url']}",
                    f"Snippet: {result['snippet']}",
                ]
            )
        returned_text = "\n".join(lines) + "\n"
        tool_stat = {
            "success": True,
            "source": "serper_dev_google_search",
            "endpoint": self.endpoint,
            "request": request,
            "response_cache_hit": cache_hit,
            "raw_response_cache": str(cache_path),
            "requested": self.top_k,
            "num_results": len(results),
            "skipped": skipped,
        }
        return returned_text, results, tool_stat
