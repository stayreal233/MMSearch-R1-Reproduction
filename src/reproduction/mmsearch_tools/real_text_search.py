"""Composite Serper.dev search and Jina Reader tool for MMSearch-R1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from reproduction.mmsearch_tools.jina_reader import JinaReader
from reproduction.mmsearch_tools.serper_text_search import SerperTextSearch


class DocumentSummarizer(Protocol):
    """Narrow interface used by the composite text-search tool."""

    def summarize_many(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]: ...


class SerperJinaTextSearch:
    def __init__(
        self,
        serper_cache_dir: Path,
        jina_cache_dir: Path,
        *,
        top_k: int = 5,
        max_chars_per_page: int = 12_000,
        summarizer: DocumentSummarizer | None = None,
    ) -> None:
        self.search = SerperTextSearch(serper_cache_dir, top_k=top_k)
        self.reader = JinaReader(
            jina_cache_dir,
            max_chars_per_page=max_chars_per_page,
        )
        self.summarizer = summarizer

    def __call__(self, query: str) -> tuple[str, dict[str, Any]]:
        search_text, results, search_stat = self.search(query)
        webpage_text, documents, reader_stat = self.reader.read_many(results)
        public_documents = [
            {key: value for key, value in document.items() if key != "content"}
            for document in documents
        ]

        if self.summarizer is None:
            # Preserve the step-8 raw-Jina behavior unless explicitly enabled.
            returned_text = f"{search_text}\n{webpage_text}"
            tool_stat = {
                "success": search_stat["success"] and reader_stat["success"],
                "source": "serper_dev_plus_jina_reader",
                "search": search_stat,
                "reader": reader_stat,
                "results": results,
                "documents": public_documents,
            }
            return returned_text, tool_stat

        summary_text, summaries, summary_stat = self.summarizer.summarize_many(
            query,
            documents,
        )
        returned_text = f"{search_text}\n{summary_text}"
        tool_stat = {
            "success": (
                search_stat["success"]
                and reader_stat["success"]
                and summary_stat.get("success") is True
            ),
            "source": "serper_dev_plus_jina_reader_plus_qwen3_summary",
            "search": search_stat,
            "reader": reader_stat,
            "summary": summary_stat,
            "results": results,
            "documents": public_documents,
            "summaries": summaries,
        }
        return returned_text, tool_stat
