"""SerpApi Google Lens adapter with local JSON and thumbnail caching."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _safe_json(item)
            for key, item in value.items()
            if key.lower() not in {"api_key", "account_email", "account_id"}
        }
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    return value


class SerpAPIGoogleLensSearch:
    endpoint = "https://serpapi.com/search.json"
    account_endpoint = "https://serpapi.com/account.json"

    def __init__(
        self,
        search_cache_dir: Path,
        *,
        api_key: str | None = None,
        top_k: int = 5,
        timeout_seconds: float = 60.0,
        language: str = "en",
        country: str = "us",
    ) -> None:
        self.api_key = api_key or os.environ.get("SERPAPI_API_KEY")
        self.search_cache_dir = Path(search_cache_dir)
        self.json_cache_dir = self.search_cache_dir / "json"
        self.thumbnail_cache_dir = self.search_cache_dir / "thumbnails"
        self.json_cache_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnail_cache_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.timeout_seconds = timeout_seconds
        self.language = language
        self.country = country

    @staticmethod
    def _validate_image_url(image_url: str) -> None:
        parsed = urlparse(image_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("SerpApi Google Lens requires a public HTTP(S) image URL")

    def _request_signature(self, image_url: str) -> tuple[dict[str, str], str]:
        public_parameters = {
            "engine": "google_lens",
            "type": "visual_matches",
            "url": image_url,
            "hl": self.language,
            "country": self.country,
            "no_cache": "false",
            "output": "json",
        }
        serialized = json.dumps(public_parameters, sort_keys=True, separators=(",", ":"))
        signature = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return public_parameters, signature

    def account_summary(self) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("SERPAPI_API_KEY is not set")
        response = requests.get(
            self.account_endpoint,
            params={"api_key": self.api_key},
            timeout=(5.0, 30.0),
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"SerpApi account endpoint returned HTTP {response.status_code} with non-JSON content") from exc
        if response.status_code != 200 or payload.get("error"):
            raise RuntimeError(
                f"SerpApi account check failed (HTTP {response.status_code}): {payload.get('error', 'unknown error')}"
            )
        allowed = {
            "account_status",
            "plan_id",
            "plan_name",
            "searches_per_month",
            "plan_searches_left",
            "extra_credits",
            "total_searches_left",
            "this_month_usage",
            "this_hour_searches",
            "last_hour_searches",
            "account_rate_limit_per_hour",
            "plan_renewal_date",
        }
        return {key: payload[key] for key in allowed if key in payload}

    def _load_or_query(self, image_url: str) -> tuple[dict[str, Any], Path, bool]:
        public_parameters, signature = self._request_signature(image_url)
        cache_path = self.json_cache_dir / f"{signature}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8")), cache_path, True
        if not self.api_key:
            raise RuntimeError("SERPAPI_API_KEY is not set and no matching local response cache exists")

        private_parameters = dict(public_parameters)
        private_parameters["api_key"] = self.api_key
        response = requests.get(
            self.endpoint,
            params=private_parameters,
            timeout=(5.0, self.timeout_seconds),
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"SerpApi returned HTTP {response.status_code} with non-JSON content") from exc
        if response.status_code != 200 or payload.get("error"):
            raise RuntimeError(
                f"SerpApi Google Lens failed (HTTP {response.status_code}): {payload.get('error', 'unknown error')}"
            )
        sanitized = _safe_json(payload)
        cache_path.write_text(
            json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return sanitized, cache_path, False

    def _thumbnail_path(self, signature: str, position: int, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return self.thumbnail_cache_dir / f"{signature[:16]}_{position}_{digest}.png"

    def _fetch_thumbnail(
        self,
        signature: str,
        position: int,
        url: str,
    ) -> tuple[Image.Image, Path, bool]:
        path = self._thumbnail_path(signature, position, url)
        if path.exists():
            try:
                with Image.open(path) as image:
                    image.load()
                    return image.convert("RGB"), path, True
            except Exception:
                path.unlink(missing_ok=True)

        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "Chrome/124.0 Safari/537.36"
                )
            },
            timeout=(5.0, 20.0),
        )
        if response.status_code != 200:
            raise RuntimeError(f"thumbnail HTTP {response.status_code}")
        if len(response.content) > 10 * 1024 * 1024:
            raise ValueError("thumbnail exceeds 10 MiB")
        with Image.open(BytesIO(response.content)) as image:
            image.load()
            image = image.convert("RGB")
            image.save(path, format="PNG", compress_level=6)
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB"), path, False

    def __call__(self, image_url: str):
        self._validate_image_url(image_url)
        payload, raw_cache_path, response_cache_hit = self._load_or_query(image_url)
        _, signature = self._request_signature(image_url)
        matches = list(payload.get("visual_matches", []))[: self.top_k]
        if not matches:
            raise RuntimeError("SerpApi response contains no visual_matches")

        failures: list[dict[str, Any]] = []
        resolved: list[tuple[int, dict[str, Any], Image.Image, Path, bool]] = []
        with ThreadPoolExecutor(max_workers=len(matches)) as executor:
            futures = {}
            for index, match in enumerate(matches):
                thumbnail = match.get("thumbnail") or match.get("image")
                if not thumbnail:
                    failures.append(
                        {
                            "index": index,
                            "title": match.get("title"),
                            "error": "visual match has no thumbnail or image URL",
                        }
                    )
                    continue
                future = executor.submit(
                    self._fetch_thumbnail,
                    signature,
                    index + 1,
                    thumbnail,
                )
                futures[future] = (index, match)
            for future in as_completed(futures):
                index, match = futures[future]
                try:
                    image, local_path, cache_hit = future.result()
                except Exception as exc:
                    failures.append(
                        {
                            "index": index,
                            "title": match.get("title"),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                else:
                    resolved.append((index, match, image, local_path, cache_hit))

        resolved.sort(key=lambda item: item[0])
        images = [item[2] for item in resolved]
        titles = [item[1].get("title") or item[1].get("source") or "Untitled" for item in resolved]
        links = [item[1].get("link") for item in resolved]
        thumbnail_urls = [item[1].get("thumbnail") or item[1].get("image") for item in resolved]
        local_paths = [str(item[3]) for item in resolved]
        thumbnail_cache_hits = sum(bool(item[4]) for item in resolved)

        lines = ["[Image Search Results] SerpApi Google Lens visual matches, ranked by relevance:"]
        for rank, (title, link) in enumerate(zip(titles, links), start=1):
            lines.append(
                f"{rank}. image: <|vision_start|><|image_pad|><|vision_end|>\n"
                f"title: {title}\nlink: {link or ''}"
            )
        returned_text = "\n".join(lines) + "\n"
        metadata = payload.get("search_metadata", {})
        tool_stat = {
            "success": bool(images),
            "source": "serpapi_google_lens",
            "search_id": metadata.get("id"),
            "search_status": metadata.get("status"),
            "response_cache_hit": response_cache_hit,
            "raw_response_cache": str(raw_cache_path),
            "requested": min(len(matches), self.top_k),
            "num_images": len(images),
            "thumbnail_cache_hits": thumbnail_cache_hits,
            "titles": titles,
            "links": links,
            "thumbnail_urls": thumbnail_urls,
            "local_thumbnail_paths": local_paths,
            "failures": sorted(failures, key=lambda item: item["index"]),
        }
        return returned_text, images, tool_stat
