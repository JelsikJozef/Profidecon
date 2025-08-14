"""
Unit tests for the LLM Enricher module.

Tests cover LLM-based enrichment functionality including:
- Summary generation
- Tag extraction
- Error handling and fallbacks
- Mock testing to avoid actual LLM calls
"""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

from Main_programme.preprocessor.processors.enricher_llm import (
    LLMEnricher,
    run
)


class TestLLMEnricher(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.sample_text = """This is a legal document about residence permits.
        It contains information about required documents and procedures.
        The process involves submitting forms and paying fees."""

    @patch('Main_programme.preprocessor.processors.enricher_llm.LLMPicker')
    def test_llm_enricher_initialization(self, mock_llm_picker):
        """Test LLM enricher initialization."""
        enricher = LLMEnricher(llm_backend="test_backend")

        mock_llm_picker.assert_called_once_with(backend="test_backend")
        self.assertEqual(enricher.backend, "test_backend")

    @patch('Main_programme.preprocessor.processors.enricher_llm.LLMPicker')
    def test_generate_summary_and_tags_success(self, mock_llm_picker):
        """Test successful summary and tags generation."""
        # Mock LLM response
        mock_instance = mock_llm_picker.return_value
        mock_instance.generate_summary_and_tags.return_value = (
            "Test summary about residence permits",
            ["residence", "permits", "documents"]
        )

        enricher = LLMEnricher()
        summary, tags = enricher.generate_summary_and_tags(self.sample_text)

        self.assertEqual(summary, "Test summary about residence permits")
        self.assertEqual(tags, ["residence", "permits", "documents"])
        mock_instance.generate_summary_and_tags.assert_called_once_with(self.sample_text)

    @patch('Main_programme.preprocessor.processors.enricher_llm.LLMPicker')
    def test_generate_summary_and_tags_empty_text(self, mock_llm_picker):
        """Test handling of empty text."""
        enricher = LLMEnricher()
        summary, tags = enricher.generate_summary_and_tags("")

        self.assertEqual(summary, "")
        self.assertEqual(tags, [])
        # Should not call LLM for empty text
        mock_llm_picker.return_value.generate_summary_and_tags.assert_not_called()

    @patch('Main_programme.preprocessor.processors.enricher_llm.LLMPicker')
    def test_generate_summary_and_tags_exception(self, mock_llm_picker):
        """Test exception handling in LLM calls."""
        # Mock LLM to raise exception
        mock_instance = mock_llm_picker.return_value
        mock_instance.generate_summary_and_tags.side_effect = Exception("LLM failed")

        enricher = LLMEnricher()
        summary, tags = enricher.generate_summary_and_tags(self.sample_text)

        self.assertEqual(summary, "")
        self.assertEqual(tags, [])

    @patch('Main_programme.preprocessor.processors.enricher_llm.LLMPicker')
    def test_llm_enricher_run(self, mock_llm_picker):
        """Test the main LLMEnricher.run method."""
        # Mock LLM response
        mock_instance = mock_llm_picker.return_value
        mock_instance.generate_summary_and_tags.return_value = (
            "Generated summary",
            ["tag1", "tag2", "tag3"]
        )

        enricher = LLMEnricher()

        document = {
            "text": self.sample_text,
            "metadata": {
                "source": "/path/to/document.txt"
            }
        }

        result = enricher.run(document)

        # Check that original document data is preserved
        self.assertEqual(result["text"], self.sample_text)
        self.assertEqual(result["metadata"]["source"], "/path/to/document.txt")

        # Check that LLM-generated content is added
        self.assertEqual(result["summary"], "Generated summary")
        self.assertEqual(result["tags"], ["tag1", "tag2", "tag3"])

    @patch('Main_programme.preprocessor.processors.enricher_llm.LLMPicker')
    def test_llm_enricher_run_no_text(self, mock_llm_picker):
        """Test run method with document containing no text."""
        enricher = LLMEnricher()

        document = {
            "metadata": {"source": "/path/to/document.txt"}
        }

        result = enricher.run(document)

        # Should handle missing text gracefully
        self.assertEqual(result["summary"], "")
        self.assertEqual(result["tags"], [])
        # Should not call LLM for empty text
        mock_llm_picker.return_value.generate_summary_and_tags.assert_not_called()

    @patch('Main_programme.preprocessor.processors.enricher_llm.LLMPicker')
    def test_llm_enricher_run_preserves_existing_data(self, mock_llm_picker):
        """Test that run method preserves existing document data."""
        # Mock LLM response
        mock_instance = mock_llm_picker.return_value
        mock_instance.generate_summary_and_tags.return_value = (
            "New summary",
            ["new_tag1", "new_tag2"]
        )

        enricher = LLMEnricher()

        document = {
            "text": self.sample_text,
            "metadata": {
                "source": "/path/to/document.txt",
                "existing_field": "existing_value"
            },
            "existing_summary": "old summary",
            "existing_tags": ["old_tag"],
            "custom_field": "custom_value"
        }

        result = enricher.run(document)

        # Check that existing fields are preserved
        self.assertEqual(result["metadata"]["existing_field"], "existing_value")
        self.assertEqual(result["existing_summary"], "old summary")
        self.assertEqual(result["existing_tags"], ["old_tag"])
        self.assertEqual(result["custom_field"], "custom_value")

        # Check that new LLM content is added
        self.assertEqual(result["summary"], "New summary")
        self.assertEqual(result["tags"], ["new_tag1", "new_tag2"])

    @patch('Main_programme.preprocessor.processors.enricher_llm.LLMPicker')
    def test_stateless_run_function(self, mock_llm_picker):
        """Test the stateless run function interface."""
        # Mock LLM response
        mock_instance = mock_llm_picker.return_value
        mock_instance.generate_summary_and_tags.return_value = (
            "Function summary",
            ["func_tag1", "func_tag2"]
        )

        document = {
            "text": self.sample_text,
            "metadata": {"source": "/path/to/document.txt"}
        }

        result = run(document, llm_backend="test_backend")

        # Verify LLMPicker was initialized with correct backend
        mock_llm_picker.assert_called_with(backend="test_backend")

        # Check results
        self.assertEqual(result["summary"], "Function summary")
        self.assertEqual(result["tags"], ["func_tag1", "func_tag2"])

    @patch('Main_programme.preprocessor.processors.enricher_llm.LLMPicker')
    def test_different_llm_backends(self, mock_llm_picker):
        """Test initialization with different LLM backends."""
        backends = ["openai", "ollama", "huggingface"]

        for backend in backends:
            enricher = LLMEnricher(llm_backend=backend)
            mock_llm_picker.assert_called_with(backend=backend)
            self.assertEqual(enricher.backend, backend)


if __name__ == '__main__':
    unittest.main()
