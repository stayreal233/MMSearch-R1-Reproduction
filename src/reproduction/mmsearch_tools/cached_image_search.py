"""Adapter for the official FVQA cached image-search results."""

from __future__ import annotations

import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image


class FVQACachedImageSearch:
    def __init__(
        self,
        cache_pickle: Path,
        thumbnail_cache_dir: Path,
        *,
        top_k: int = 5,
        timeout_seconds: float = 20.0,
        max_download_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        self.cache_pickle = Path(cache_pickle)
        self.thumbnail_cache_dir = Path(thumbnail_cache_dir)
        self.top_k = top_k
        self.timeout_seconds = timeout_seconds
        self.max_download_bytes = max_download_bytes
        self.thumbnail_cache_dir.mkdir(parents=True, exist_ok=True)
        with self.cache_pickle.open("rb") as handle:
            cache = pickle.load(handle)
        if not isinstance(cache, dict):
            raise TypeError(f"Expected dict in {self.cache_pickle}, got {type(cache).__name__}")
        self.cache: dict[str, dict[str, Any]] = cache

    def _thumbnail_path(self, data_id: str, index: int, source: str) -> Path:
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]
        return self.thumbnail_cache_dir / f"{data_id}_{index + 1}_{digest}.png"

    @staticmethod
    def _read_cached_image(path: Path) -> Image.Image:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")

    def _fetch_url(
        self,
        data_id: str,
        index: int,
        url: str,
    ) -> tuple[Image.Image, Path, bool]:
        destination = self._thumbnail_path(data_id, index, url)
        if destination.exists():
            try:
                return self._read_cached_image(destination), destination, True
            except Exception:
                destination.unlink(missing_ok=True)

        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "Chrome/124.0 Safari/537.36"
                )
            },
            timeout=(5.0, self.timeout_seconds),
        )
        response.raise_for_status()
        content = response.content
        if len(content) > self.max_download_bytes:
            raise ValueError(f"thumbnail exceeds {self.max_download_bytes} bytes")
        with Image.open(BytesIO(content)) as image:
            image.load()
            image = image.convert("RGB")
            image.save(destination, format="PNG", compress_level=6)
        return self._read_cached_image(destination), destination, False

    def _resolve_item(
        self,
        data_id: str,
        index: int,
        source: Any,
    ) -> tuple[Image.Image, Path | None, bool]:
        if isinstance(source, Image.Image):
            return source.convert("RGB"), None, True
        if isinstance(source, str):
            return self._fetch_url(data_id, index, source)
        raise TypeError(f"unsupported cached thumbnail type: {type(source).__name__}")

    def __call__(self, data_id: str):
        if data_id not in self.cache:
            raise KeyError(f"No official FVQA image-search cache entry for {data_id}")
        entry = self.cache[data_id]
        titles = list(entry.get("tool_returned_web_title_list", []))[: self.top_k]
        sources = list(entry.get("tool_returned_images_urls", []))[: self.top_k]
        if not titles or not sources:
            raise ValueError(f"Official cache entry for {data_id} has no titles or thumbnails")

        count = min(len(titles), len(sources), self.top_k)
        failures: list[dict[str, Any]] = []
        resolved: list[tuple[int, Image.Image, Path | None, bool]] = []
        with ThreadPoolExecutor(max_workers=count) as executor:
            futures = {
                executor.submit(self._resolve_item, data_id, index, sources[index]): index
                for index in range(count)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    image, local_path, cache_hit = future.result()
                except Exception as exc:
                    failures.append(
                        {
                            "index": index,
                            "title": titles[index],
                            "source": str(sources[index]),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                else:
                    resolved.append((index, image, local_path, cache_hit))

        resolved.sort(key=lambda item: item[0])
        returned_images = [item[1] for item in resolved]
        returned_titles = [titles[item[0]] for item in resolved]
        returned_sources = [str(sources[item[0]]) for item in resolved]
        local_paths = [str(item[2]) if item[2] is not None else None for item in resolved]
        cache_hits = sum(bool(item[3]) for item in resolved)

        lines = [
            "[Image Search Results] Official FVQA cached Google Lens results, ranked by relevance:"
        ]
        for rank, (title, source) in enumerate(
            zip(returned_titles, returned_sources),
            start=1,
        ):
            lines.append(
                f"{rank}. image: <|vision_start|><|image_pad|><|vision_end|>\n"
                f"title: {title}\nthumbnail_url: {source}"
            )
        returned_text = "\n".join(lines) + "\n"
        tool_stat = {
            "success": bool(returned_images),
            "source": "fvqa_official_image_search_cache",
            "data_id": data_id,
            "requested": count,
            "num_images": len(returned_images),
            "cache_hits": cache_hits,
            "titles": returned_titles,
            "thumbnail_urls": returned_sources,
            "local_thumbnail_paths": local_paths,
            "failures": sorted(failures, key=lambda item: item["index"]),
        }
        return returned_text, returned_images, tool_stat
