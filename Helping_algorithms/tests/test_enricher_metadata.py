"""
Unit tests for the Metadata Enricher module.

Tests cover all non-LLM enrichment functionality including:
- Language detection
- Text statistics (characters, words, tokens, sentences, paragraphs)
- File statistics
- PII risk scoring
- Content hashing
- Category extraction
"""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

from Main_programme.preprocessor.processors.enricher_metadata import (
    MetadataEnricher,
    estimate_tokens,
    estimate_characters,
    estimate_words,
    estimate_sentences,
    estimate_paragraphs,
    pii_risk_score,
    detect_language,
    calculate_content_hash,
    extract_category_from_path,
    get_file_stats,
    run
)


class TestMetadataEnricher(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.root_path = Path(self.temp_dir)
        self.enricher = MetadataEnricher(self.root_path)

        # Sample text for testing
        self.sample_text = """This is a test document.
        It contains multiple sentences! Does it work properly?
        
        This is a second paragraph.
        It has more content for testing purposes.
        
        Contact us at test@example.com or call +421900123456."""

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_estimate_tokens(self):
        """Test token estimation."""
        text = "This is a test with five tokens"
        self.assertEqual(estimate_tokens(text), 7)
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("single"), 1)

    def test_estimate_characters(self):
        """Test character count."""
        text = "Hello World"
        self.assertEqual(estimate_characters(text), 11)
        self.assertEqual(estimate_characters(""), 0)

    def test_estimate_words(self):
        """Test word count using regex."""
        text = "Hello, world! How are you?"
        self.assertEqual(estimate_words(text), 5)
        self.assertEqual(estimate_words(""), 0)

    def test_estimate_sentences(self):
        """Test sentence count."""
        text = "First sentence. Second sentence! Third sentence?"
        self.assertEqual(estimate_sentences(text), 3)
        text_no_sentences = "Just some text without proper endings"
        self.assertEqual(estimate_sentences(text_no_sentences), 0)

    def test_estimate_paragraphs(self):
        """Test paragraph count."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        self.assertEqual(estimate_paragraphs(text), 3)
        single_para = "Just one paragraph"
        self.assertEqual(estimate_paragraphs(single_para), 1)

    def test_pii_risk_score(self):
        """Test PII risk scoring."""
        # Text with email and phone
        pii_text = "Contact me at john@example.com or call +421900123456"
        score = pii_risk_score(pii_text)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Clean text
        clean_text = "This is a normal document without any personal information"
        clean_score = pii_risk_score(clean_text)
        self.assertEqual(clean_score, 0.0)

        # Empty text
        self.assertEqual(pii_risk_score(""), 0.0)

    @patch('Main_programme.preprocessor.processors.enricher_metadata.detect')
    def test_detect_language(self, mock_detect):
        """Test language detection with mocking."""
        mock_detect.return_value = 'en'
        result = detect_language("This is English text")
        self.assertEqual(result, 'en')

        # Test fallback for short text
        result = detect_language("Hi")
        self.assertEqual(result, 'unknown')

        # Test exception handling
        mock_detect.side_effect = Exception("Detection failed")
        result = detect_language("This should fail")
        self.assertEqual(result, 'unknown')

    def test_calculate_content_hash(self):
        """Test content hashing."""
        text1 = "Same content"
        text2 = "Same content"
        text3 = "Different content"

        hash1 = calculate_content_hash(text1)
        hash2 = calculate_content_hash(text2)
        hash3 = calculate_content_hash(text3)

        self.assertEqual(hash1, hash2)  # Same content = same hash
        self.assertNotEqual(hash1, hash3)  # Different content = different hash
        self.assertEqual(len(hash1), 40)  # SHA-1 hash length

    def test_extract_category_from_path(self):
        """Test category extraction from file paths."""
        # Create test directory structure
        category_dir = self.root_path / "test_category" / "subcategory"
        category_dir.mkdir(parents=True)
        test_file = category_dir / "test.txt"
        test_file.touch()

        category = extract_category_from_path(test_file, self.root_path)
        self.assertEqual(category, "test_category")

        # Test root level file
        root_file = self.root_path / "root_file.txt"
        root_file.touch()
        root_category = extract_category_from_path(root_file, self.root_path)
        self.assertEqual(root_category, "root")

    def test_get_file_stats(self):
        """Test file statistics retrieval."""
        # Create a test file
        test_file = self.root_path / "test.txt"
        test_content = "Test content for file stats"
        test_file.write_text(test_content)

        stats = get_file_stats(test_file)

        self.assertIn("size_bytes", stats)
        self.assertIn("created_ts", stats)
        self.assertIn("modified_ts", stats)
        self.assertEqual(stats["size_bytes"], len(test_content))
        self.assertGreater(stats["created_ts"], 0)
        self.assertGreater(stats["modified_ts"], 0)

        # Test non-existent file
        non_existent = self.root_path / "non_existent.txt"
        stats = get_file_stats(non_existent)
        self.assertEqual(stats["size_bytes"], 0)

    def test_metadata_enricher_run(self):
        """Test the main MetadataEnricher.run method."""
        # Create a test file
        test_file = self.root_path / "documents" / "test.txt"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(self.sample_text)

        document = {
            "text": self.sample_text,
            "metadata": {
                "source": str(test_file)
            }
        }

        with patch('Main_programme.preprocessor.processors.enricher_metadata.detect_language') as mock_lang:
            mock_lang.return_value = 'en'

            result = self.enricher.run(document)

            self.assertIn("metadata", result)
            metadata = result["metadata"]

            # Check all expected metadata fields
            self.assertIn("char_count", metadata)
            self.assertIn("word_count", metadata)
            self.assertIn("token_estimate", metadata)
            self.assertIn("sentence_count", metadata)
            self.assertIn("paragraph_count", metadata)
            self.assertIn("language", metadata)
            self.assertIn("hash_content", metadata)
            self.assertIn("hash_sha1", metadata)  # Backward compatibility
            self.assertIn("pii_risk", metadata)
            self.assertIn("category", metadata)
            self.assertIn("size_bytes", metadata)
            self.assertIn("created_ts", metadata)
            self.assertIn("modified_ts", metadata)

            # Check specific values
            self.assertEqual(metadata["language"], 'en')
            self.assertEqual(metadata["category"], "documents")
            self.assertGreater(metadata["char_count"], 0)
            self.assertGreater(metadata["word_count"], 0)
            self.assertGreater(metadata["sentence_count"], 0)
            self.assertGreater(metadata["pii_risk"], 0)  # Contains email and phone

    def test_stateless_run_function(self):
        """Test the stateless run function interface."""
        document = {
            "text": "Test document",
            "metadata": {"source": str(self.root_path / "test.txt")}
        }

        result = run(document, self.root_path)
        self.assertIn("metadata", result)
        self.assertIn("token_estimate", result["metadata"])


if __name__ == '__main__':
    unittest.main()
