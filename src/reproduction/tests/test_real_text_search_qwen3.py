import copy
import unittest
from unittest.mock import Mock, patch

from reproduction.mmsearch_tools.real_text_search import SerperJinaTextSearch


class SerperJinaTextSearchQwen3Test(unittest.TestCase):
    def _make_tool(self, summarizer=None):
        with (
            patch(
                "reproduction.mmsearch_tools.real_text_search.SerperTextSearch"
            ),
            patch("reproduction.mmsearch_tools.real_text_search.JinaReader"),
        ):
            tool = SerperJinaTextSearch(
                "/unused/serper-cache",
                "/unused/jina-cache",
                summarizer=summarizer,
            )
        tool.search = Mock()
        tool.reader = Mock()
        return tool

    def test_without_summarizer_preserves_legacy_contract_and_inputs(self):
        query = "legacy query"
        search_text = "search results"
        webpage_text = "retrieved pages"
        results = [{"title": "Result", "link": "https://example.test/result"}]
        documents = [
            {
                "title": "Page",
                "url": "https://example.test/page",
                "content": "full page content",
            }
        ]
        original_documents = copy.deepcopy(documents)
        search_stat = {"success": True, "cached": False}
        reader_stat = {"success": True, "pages": 1}
        tool = self._make_tool(summarizer=None)
        tool.search.return_value = search_text, results, search_stat
        tool.reader.read_many.return_value = webpage_text, documents, reader_stat

        returned_text, tool_stat = tool(query)

        self.assertEqual(returned_text, "search results\nretrieved pages")
        self.assertEqual(
            tool_stat,
            {
                "success": True,
                "source": "serper_dev_plus_jina_reader",
                "search": search_stat,
                "reader": reader_stat,
                "results": results,
                "documents": [
                    {
                        "title": "Page",
                        "url": "https://example.test/page",
                    }
                ],
            },
        )
        self.assertEqual(documents, original_documents)
        self.assertNotIn("content", tool_stat["documents"][0])

    def test_summary_mode_uses_full_documents_and_returns_public_metadata(self):
        query = "summary query"
        search_text = "search results"
        results = [{"title": "Result", "link": "https://example.test/result"}]
        documents = [
            {
                "title": "Page one",
                "url": "https://example.test/one",
                "content": "content used for summarization",
            },
            {
                "title": "Page two",
                "url": "https://example.test/two",
                "content": "more content used for summarization",
                "status": 200,
            },
        ]
        original_documents = copy.deepcopy(documents)
        search_stat = {"success": True, "cached": True}
        reader_stat = {"success": True, "pages": 2}
        summary_text = "condensed answer"
        summaries = [
            {"url": "https://example.test/one", "summary": "first"},
            {"url": "https://example.test/two", "summary": "second"},
        ]
        summary_stat = {"success": True, "documents": 2}
        summarizer = Mock()
        summarizer.summarize_many.return_value = (
            summary_text,
            summaries,
            summary_stat,
        )
        tool = self._make_tool(summarizer=summarizer)
        tool.search.return_value = search_text, results, search_stat
        tool.reader.read_many.return_value = "unused reader text", documents, reader_stat

        returned_text, tool_stat = tool(query)

        self.assertEqual(returned_text, "search results\ncondensed answer")
        summarizer.summarize_many.assert_called_once_with(query, documents)
        self.assertEqual(tool_stat["source"], "serper_dev_plus_jina_reader_plus_qwen3_summary")
        self.assertIs(tool_stat["summary"], summary_stat)
        self.assertIs(tool_stat["summaries"], summaries)
        self.assertEqual(
            tool_stat["documents"],
            [
                {"title": "Page one", "url": "https://example.test/one"},
                {
                    "title": "Page two",
                    "url": "https://example.test/two",
                    "status": 200,
                },
            ],
        )
        self.assertTrue(tool_stat["success"])
        self.assertEqual(documents, original_documents)
        self.assertTrue(all("content" not in doc for doc in tool_stat["documents"]))

    def test_summary_failure_makes_overall_call_unsuccessful(self):
        summarizer = Mock()
        summarizer.summarize_many.return_value = (
            "summary unavailable",
            [],
            {"success": False, "error": "model failure"},
        )
        tool = self._make_tool(summarizer=summarizer)
        tool.search.return_value = "search results", [], {"success": True}
        tool.reader.read_many.return_value = (
            "reader text",
            [{"url": "https://example.test", "content": "page"}],
            {"success": True},
        )

        _returned_text, tool_stat = tool("failing summary query")

        self.assertFalse(tool_stat["success"])
        self.assertEqual(tool_stat["summary"]["error"], "model failure")
        summarizer.summarize_many.assert_called_once()


if __name__ == "__main__":
    unittest.main()
