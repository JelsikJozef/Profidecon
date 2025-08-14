"""
Unit tests for PII Analyzer module.

Tests cover:
- Regex and Presidio backends
- Multi-language detection (English, Slovak, German)
- Entity type filtering and confidence thresholds
- Overlap resolution
- Configuration handling
- Performance requirements
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import time
from typing import List

from Main_programme.preprocessor.processors.pii_analyzer import (
    PiiAnalyzer,
    PiiEntity,
    PiiConfig,
    RegexPiiDetector,
    PresidioPiiDetector,
    create_analyzer
)


class TestPiiConfig(unittest.TestCase):
    """Test PII configuration handling."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PiiConfig()
        self.assertEqual(config.backend, "regex")
        self.assertEqual(config.langs, "auto")
        self.assertTrue(config.return_values)
        self.assertEqual(config.min_confidence, 0.60)
        self.assertIsNone(config.types_include)
        self.assertIsNone(config.types_exclude)
        self.assertEqual(config.max_entities_per_doc, 1000)

    @patch.dict(os.environ, {
        "PII_BACKEND": "presidio",
        "PII_LANGS": "en,sk,de",
        "PII_RETURN_VALUES": "false",
        "PII_MIN_CONFIDENCE": "0.80",
        "PII_TYPES_INCLUDE": "EMAIL,PHONE,PERSON_NAME",
        "PII_TYPES_EXCLUDE": "URL,DATE_OF_BIRTH",
        "PII_MAX_ENTITIES_PER_DOC": "500"
    })
    def test_config_from_env(self):
        """Test configuration loading from environment variables."""
        config = PiiConfig.from_env()
        self.assertEqual(config.backend, "presidio")
        self.assertEqual(config.langs, "en,sk,de")
        self.assertFalse(config.return_values)
        self.assertEqual(config.min_confidence, 0.80)
        self.assertEqual(config.types_include, {"EMAIL", "PHONE", "PERSON_NAME"})
        self.assertEqual(config.types_exclude, {"URL", "DATE_OF_BIRTH"})
        self.assertEqual(config.max_entities_per_doc, 500)


class TestRegexPiiDetector(unittest.TestCase):
    """Test regex-based PII detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = PiiConfig()
        self.detector = RegexPiiDetector(self.config)

    def test_email_detection(self):
        """Test email address detection."""
        text = "Contact us at john.doe@example.com or support@company.sk"
        entities = self.detector.detect(text)

        email_entities = [e for e in entities if e["type"] == "EMAIL"]
        self.assertEqual(len(email_entities), 2)

        # Check first email - fix expected end position
        self.assertEqual(email_entities[0]["value"], "john.doe@example.com")
        self.assertEqual(email_entities[0]["start"], 14)
        self.assertEqual(email_entities[0]["end"], 34)  # Fixed: correct end position
        self.assertGreater(email_entities[0]["confidence"], 0.9)

        # Check second email
        self.assertEqual(email_entities[1]["value"], "support@company.sk")
        self.assertGreater(email_entities[1]["confidence"], 0.9)

    def test_phone_detection(self):
        """Test phone number detection."""
        text = "Call me at +421 900 123 456 or (02) 1234-5678"
        entities = self.detector.detect(text)

        phone_entities = [e for e in entities if e["type"] == "PHONE"]
        self.assertGreater(len(phone_entities), 0)

        # Check international format
        intl_phone = next((e for e in phone_entities if "+421" in e["value"]), None)
        self.assertIsNotNone(intl_phone)
        self.assertGreater(intl_phone["confidence"], 0.7)

    def test_slovak_names_with_diacritics(self):
        """Test Slovak names with diacritics detection."""
        text = "Stretol som Jána Nováka a Mária Svobodová bola tiež prítomná."
        entities = self.detector.detect(text)

        name_entities = [e for e in entities if e["type"] == "PERSON_NAME"]
        self.assertGreater(len(name_entities), 0)

        # Should detect names with Slovak diacritics
        names = [e["value"] for e in name_entities if e["value"]]
        self.assertTrue(any("Ján" in name for name in names) or
                       any("Novák" in name for name in names))

    def test_iban_detection(self):
        """Test IBAN detection."""
        text = "My bank account is SK8975000000000012345671"
        entities = self.detector.detect(text)

        iban_entities = [e for e in entities if e["type"] == "IBAN"]
        self.assertEqual(len(iban_entities), 1)
        self.assertEqual(iban_entities[0]["value"], "SK8975000000000012345671")
        self.assertGreater(iban_entities[0]["confidence"], 0.8)

    def test_business_id_exclusion(self):
        """Test that business IDs get lower confidence."""
        text = "Invoice number 123456789 and order reference 987654321"
        entities = self.detector.detect(text)

        id_entities = [e for e in entities if e["type"] == "ID_NUMBER"]
        # Should have low confidence or be filtered out
        for entity in id_entities:
            self.assertLess(entity["confidence"], 0.6)

    def test_confidence_threshold(self):
        """Test confidence threshold filtering."""
        high_confidence_config = PiiConfig()
        high_confidence_config.min_confidence = 0.90
        detector = RegexPiiDetector(high_confidence_config)

        text = "Email: test@example.com and phone: +1234567890"
        entities = detector.detect(text)

        # Only high-confidence entities should remain
        for entity in entities:
            self.assertGreaterEqual(entity["confidence"], 0.90)

    def test_return_values_false(self):
        """Test that values are hidden when configured."""
        no_values_config = PiiConfig()
        no_values_config.return_values = False
        detector = RegexPiiDetector(no_values_config)

        text = "Contact: john@example.com"
        entities = detector.detect(text)

        for entity in entities:
            self.assertIsNone(entity["value"])


class TestPresidioPiiDetector(unittest.TestCase):
    """Test Presidio-based PII detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = PiiConfig()

    def test_presidio_initialization(self):
        """Test Presidio analyzer initialization."""
        # Test with mocking the presidio import directly in the PresidioPiiDetector._init_presidio method
        with patch.object(PresidioPiiDetector, '_init_presidio') as mock_init:
            mock_detector = MagicMock()
            mock_init.return_value = None

            detector = PresidioPiiDetector(self.config)
            detector.analyzer = mock_detector
            self.assertIsNotNone(detector.analyzer)

    def test_presidio_detection(self):
        """Test Presidio entity detection and type mapping."""
        # Create a detector with mocked presidio functionality
        detector = PresidioPiiDetector(self.config)

        # Mock the analyzer directly
        mock_result = MagicMock()
        mock_result.entity_type = "PERSON"
        mock_result.start = 0
        mock_result.end = 10
        mock_result.score = 0.95

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [mock_result]
        detector.analyzer = mock_analyzer

        entities = detector.detect("John Smith called")

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["type"], "PERSON_NAME")  # Mapped from "PERSON"
        self.assertEqual(entities[0]["confidence"], 0.95)
        self.assertEqual(entities[0]["pattern"], "presidio_PERSON")

    def test_presidio_fallback(self):
        """Test fallback to regex when Presidio is not available."""
        detector = PresidioPiiDetector(self.config)
        detector.analyzer = None  # Simulate Presidio not available

        # Should fall back to regex detection
        text = "Email: test@example.com"
        entities = detector.detect(text)

        # Should still detect entities using regex fallback
        email_entities = [e for e in entities if e["type"] == "EMAIL"]
        self.assertGreater(len(email_entities), 0)


class TestPiiAnalyzer(unittest.TestCase):
    """Test main PII analyzer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = PiiAnalyzer()

    def test_multilingual_detection(self):
        """Test detection across multiple languages."""
        # Mixed language text with PII
        text = """
        English: Contact John Smith at john@company.com or +1-555-123-4567
        Slovak: Kontaktujte Jána Nováka na jana.novak@firma.sk alebo +421 900 123 456  
        German: Kontaktieren Sie Hans Müller unter hans@unternehmen.de oder +49 30 12345678
        """

        entities = self.analyzer.detect(text)

        # Should detect emails from all languages
        email_entities = [e for e in entities if e["type"] == "EMAIL"]
        self.assertGreaterEqual(len(email_entities), 3)

        # Should detect phones from all languages
        phone_entities = [e for e in entities if e["type"] == "PHONE"]
        self.assertGreaterEqual(len(phone_entities), 3)

        # Should detect names with diacritics
        name_entities = [e for e in entities if e["type"] == "PERSON_NAME"]
        self.assertGreater(len(name_entities), 0)

    def test_overlap_resolution(self):
        """Test handling of overlapping entities."""
        # Create a scenario where EMAIL might overlap with a generic pattern
        text = "Contact user@domain.com which is also ID 123456"
        entities = self.analyzer.detect(text)

        # Verify no overlapping spans
        sorted_entities = sorted(entities, key=lambda e: e["start"])
        for i in range(len(sorted_entities) - 1):
            current = sorted_entities[i]
            next_entity = sorted_entities[i + 1]
            self.assertLessEqual(current["end"], next_entity["start"])

    def test_type_filtering_include(self):
        """Test entity type inclusion filtering."""
        analyzer = PiiAnalyzer()
        analyzer.config.types_include = {"EMAIL", "PHONE"}

        text = "Email: test@example.com, Phone: +1234567890, Name: John Smith"
        entities = analyzer.detect(text)

        # Should only include EMAIL and PHONE types
        entity_types = {e["type"] for e in entities}
        allowed_types = {"EMAIL", "PHONE"}
        self.assertTrue(entity_types.issubset(allowed_types))

    def test_type_filtering_exclude(self):
        """Test entity type exclusion filtering."""
        analyzer = PiiAnalyzer()
        analyzer.config.types_exclude = {"URL", "DATE_OF_BIRTH"}

        text = "Visit https://example.com or born on 01/01/1990, email: test@example.com"
        entities = analyzer.detect(text)

        # Should not include excluded types
        entity_types = {e["type"] for e in entities}
        excluded_types = {"URL", "DATE_OF_BIRTH"}
        self.assertTrue(entity_types.isdisjoint(excluded_types))

    def test_deterministic_output(self):
        """Test that same input produces identical results."""
        text = "Contact John at john@example.com or +1-555-123-4567"

        entities1 = self.analyzer.detect(text)
        entities2 = self.analyzer.detect(text)

        # Results should be identical
        self.assertEqual(len(entities1), len(entities2))

        for e1, e2 in zip(entities1, entities2):
            self.assertEqual(e1["type"], e2["type"])
            self.assertEqual(e1["start"], e2["start"])
            self.assertEqual(e1["end"], e2["end"])
            self.assertEqual(e1["confidence"], e2["confidence"])

    def test_sorted_output(self):
        """Test that entities are returned sorted by start position."""
        text = "Phone: +1234567890. Email at end: test@example.com"
        entities = self.analyzer.detect(text)

        # Verify sorted by start position
        for i in range(len(entities) - 1):
            self.assertLessEqual(entities[i]["start"], entities[i + 1]["start"])

    def test_safety_limits(self):
        """Test entity count safety limits."""
        analyzer = PiiAnalyzer()
        analyzer.config.max_entities_per_doc = 5

        # Create text with many potential entities
        text = " ".join([f"test{i}@example.com" for i in range(20)])
        entities = analyzer.detect(text)

        # Should be limited to max count
        self.assertLessEqual(len(entities), 5)

    def test_performance_large_text(self):
        """Test performance on large text (≤ 200ms for 50KB)."""
        # Generate ~50KB of text with some PII
        large_text = "This is a test document. " * 2000  # ~50KB
        large_text += "Contact us at performance@test.com or +1-555-999-8888"

        start_time = time.time()
        entities = self.analyzer.detect(large_text)
        end_time = time.time()

        processing_time = (end_time - start_time) * 1000  # Convert to ms

        # Should complete within performance requirement
        self.assertLess(processing_time, 200,
                       f"Processing took {processing_time:.1f}ms, should be ≤200ms")

        # Should still detect entities
        self.assertGreater(len(entities), 0)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty text
        entities = self.analyzer.detect("")
        self.assertEqual(len(entities), 0)

        # Very short text
        entities = self.analyzer.detect("Hi")
        self.assertEqual(len(entities), 0)

        # Text with only whitespace
        entities = self.analyzer.detect("   \n\t   ")
        self.assertEqual(len(entities), 0)

        # Text with special characters
        entities = self.analyzer.detect("Special chars: !@#$%^&*()")
        # Should not crash, may or may not detect entities


class TestFactoryFunction(unittest.TestCase):
    """Test factory function."""

    def test_create_analyzer_regex(self):
        """Test creating analyzer with regex backend."""
        analyzer = create_analyzer("regex")
        self.assertIsInstance(analyzer, PiiAnalyzer)
        self.assertEqual(analyzer.config.backend, "regex")

    def test_create_analyzer_with_config(self):
        """Test creating analyzer with custom configuration."""
        analyzer = create_analyzer("regex", min_confidence=0.80, return_values=False)
        self.assertEqual(analyzer.config.min_confidence, 0.80)
        self.assertFalse(analyzer.config.return_values)


class TestBackendParity(unittest.TestCase):
    """Test parity between backends when both are available."""

    def test_backend_parity(self):
        """Test that both backends produce reasonable results on same input."""
        text = "Contact us at test@example.com for support"

        # Test regex backend
        regex_analyzer = PiiAnalyzer("regex")
        regex_entities = regex_analyzer.detect(text)

        # Test Presidio backend (with fallback to regex if not available)
        presidio_analyzer = PiiAnalyzer("presidio")
        presidio_entities = presidio_analyzer.detect(text)

        # Both should detect at least one entity
        self.assertGreater(len(regex_entities), 0)
        self.assertGreater(len(presidio_entities), 0)

        # Both should detect EMAIL type
        regex_types = {e["type"] for e in regex_entities}
        presidio_types = {e["type"] for e in presidio_entities}

        self.assertIn("EMAIL", regex_types)
        self.assertIn("EMAIL", presidio_types)


if __name__ == '__main__':
    unittest.main()
