"""
PII Analyzer Module

Provides robust, configurable PII detection with pluggable backends.
Returns typed entities with character spans and confidence scores without
altering the input text.

Supported backends:
- regex: Baseline regex patterns (no external dependencies)
- presidio: Microsoft Presidio (if available, falls back to regex)

Environment configuration:
- PII_BACKEND: regex|presidio (default: regex)
- PII_LANGS: auto|en,sk,de,... (language hints)
- PII_RETURN_VALUES: true|false (return raw values or None)
- PII_MIN_CONFIDENCE: 0.60 (minimum confidence threshold)
- PII_TYPES_INCLUDE: comma-separated allowed types
- PII_TYPES_EXCLUDE: comma-separated excluded types
- PII_MAX_ENTITIES_PER_DOC: safety limit (default: 1000)
"""

import os
import re
import logging
from typing import TypedDict, Protocol, List, Optional, Dict, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PiiEntity(TypedDict):
    """Represents a detected PII entity with metadata."""
    type: str            # e.g. "PERSON_NAME", "EMAIL", "PHONE", "ADDRESS", "ID_SSN", "PASSPORT", "ORG"
    start: int           # char offset (inclusive)
    end: int             # char offset (exclusive)
    value: Optional[str] # may be None if backend avoids returning raw value
    confidence: float    # 0..1
    pattern: Optional[str]  # optional rule name/pattern id
    locale: Optional[str]   # detector locale hint if any


class PiiDetector(Protocol):
    """Protocol for PII detection backends."""

    def detect(self, text: str, *, locale: Optional[str] = None) -> List[PiiEntity]:
        """Detect PII entities in text."""
        ...


@dataclass
class PiiConfig:
    """Configuration for PII analyzer."""
    backend: str = "regex"
    langs: str = "auto"
    return_values: bool = True
    min_confidence: float = 0.60
    types_include: Optional[Set[str]] = None
    types_exclude: Optional[Set[str]] = None
    max_entities_per_doc: int = 1000

    @classmethod
    def from_env(cls) -> 'PiiConfig':
        """Create configuration from environment variables."""
        config = cls()
        config.backend = os.getenv("PII_BACKEND", "regex")
        config.langs = os.getenv("PII_LANGS", "auto")
        config.return_values = os.getenv("PII_RETURN_VALUES", "true").lower() == "true"
        config.min_confidence = float(os.getenv("PII_MIN_CONFIDENCE", "0.60"))
        config.max_entities_per_doc = int(os.getenv("PII_MAX_ENTITIES_PER_DOC", "1000"))

        # Parse include/exclude types
        include_str = os.getenv("PII_TYPES_INCLUDE")
        if include_str:
            config.types_include = set(t.strip().upper() for t in include_str.split(","))

        exclude_str = os.getenv("PII_TYPES_EXCLUDE")
        if exclude_str:
            config.types_exclude = set(t.strip().upper() for t in exclude_str.split(","))

        return config


class RegexPiiDetector:
    """Regex-based PII detector with multilingual support."""

    # Common PII patterns with confidence scoring
    PATTERNS = {
        "EMAIL": {
            "pattern": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "confidence": 0.95,
            "flags": re.IGNORECASE
        },
        "PHONE": {
            # International phone patterns
            "pattern": r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            "confidence": 0.80,
            "flags": 0
        },
        "IBAN": {
            "pattern": r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b',
            "confidence": 0.90,
            "flags": 0
        },
        "PASSPORT": {
            # Generic passport patterns (letters + numbers)
            "pattern": r'\b[A-Z]{1,2}\d{6,9}\b',
            "confidence": 0.75,
            "flags": 0
        },
        "ID_NUMBER": {
            # Generic ID patterns (6+ digits with optional separators)
            "pattern": r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{3,4}\b',
            "confidence": 0.70,
            "flags": 0
        },
        "DATE_OF_BIRTH": {
            # Common date formats that might be DOB
            "pattern": r'\b\d{1,2}[./\-]\d{1,2}[./\-]\d{4}\b',
            "confidence": 0.60,
            "flags": 0
        },
        "PERSON_NAME": {
            # Names with diacritics (simplified heuristic)
            "pattern": r'\b[A-ZÁČĎÉĚÍĽĹŇÓÔŔŠŤÚŮÝŽ][a-záčďéěíľĺňóôŕšťúůýž]+(?:\s+[A-ZÁČĎÉĚÍĽĹŇÓÔŔŠŤÚŮÝŽ][a-záčďéěíľĺňóôŕšťúůýž]+)+\b',
            "confidence": 0.65,
            "flags": 0
        },
        "ADDRESS": {
            # Address-like patterns (number + street-like words)
            "pattern": r'\d+\s+[A-Za-záčďéěíľĺňóôŕšťúůýž\s]+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|ulica|námestie)\b',
            "confidence": 0.70,
            "flags": re.IGNORECASE
        },
        "URL": {
            "pattern": r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
            "confidence": 0.85,
            "flags": re.IGNORECASE
        },
        "CREDIT_CARD": {
            # Credit card patterns
            "pattern": r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',
            "confidence": 0.75,
            "flags": 0
        }
    }

    def __init__(self, config: PiiConfig):
        self.config = config
        self._compiled_patterns = {}
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        for entity_type, pattern_info in self.PATTERNS.items():
            pattern = pattern_info["pattern"]
            flags = pattern_info.get("flags", 0)
            self._compiled_patterns[entity_type] = re.compile(pattern, flags)

    def _calculate_confidence(self, entity_type: str, match: re.Match, text: str) -> float:
        """Calculate confidence score for a match."""
        base_confidence = self.PATTERNS[entity_type]["confidence"]
        matched_text = match.group()

        # Length bonus for longer matches (more specific)
        length_bonus = min(0.1, len(matched_text) / 100)

        # Context bonus: check surrounding characters
        start, end = match.span()
        context_bonus = 0.0

        # Check for word boundaries
        if start > 0 and text[start-1].isspace():
            context_bonus += 0.02
        if end < len(text) and text[end].isspace():
            context_bonus += 0.02

        # Penalty for numbers that look like invoice/order numbers
        if entity_type in ["ID_NUMBER", "PASSPORT"] and self._looks_like_business_id(matched_text, text, start):
            context_bonus -= 0.3

        final_confidence = min(1.0, base_confidence + length_bonus + context_bonus)
        return max(0.0, final_confidence)

    def _looks_like_business_id(self, matched_text: str, full_text: str, position: int) -> bool:
        """Check if an ID-like pattern is likely a business identifier."""
        # Look for business context keywords around the match
        context_window = 50
        start_context = max(0, position - context_window)
        end_context = min(len(full_text), position + len(matched_text) + context_window)
        context = full_text[start_context:end_context].lower()

        business_keywords = [
            "invoice", "order", "reference", "ref", "id", "number", "no",
            "faktúra", "objednávka", "referencia", "číslo", "dokument"
        ]

        return any(keyword in context for keyword in business_keywords)

    def detect(self, text: str, *, locale: Optional[str] = None) -> List[PiiEntity]:
        """Detect PII entities using regex patterns."""
        entities = []

        for entity_type, compiled_pattern in self._compiled_patterns.items():
            for match in compiled_pattern.finditer(text):
                confidence = self._calculate_confidence(entity_type, match, text)

                # Skip low confidence matches
                if confidence < self.config.min_confidence:
                    continue

                entity = PiiEntity(
                    type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    value=match.group() if self.config.return_values else None,
                    confidence=confidence,
                    pattern=entity_type,
                    locale=locale
                )
                entities.append(entity)

        return entities


class PresidioPiiDetector:
    """Presidio-based PII detector (if available)."""

    # Mapping from Presidio entity types to our canonical types
    TYPE_MAPPING = {
        "PERSON": "PERSON_NAME",
        "EMAIL_ADDRESS": "EMAIL",
        "PHONE_NUMBER": "PHONE",
        "IBAN_CODE": "IBAN",
        "CREDIT_CARD": "CREDIT_CARD",
        "US_SSN": "ID_SSN",
        "US_PASSPORT": "PASSPORT",
        "URL": "URL",
        "DATE_TIME": "DATE_OF_BIRTH",
        "LOCATION": "ADDRESS",
        "ORG": "ORG"
    }

    def __init__(self, config: PiiConfig):
        self.config = config
        self.analyzer = None
        self._init_presidio()

    def _init_presidio(self):
        """Initialize Presidio analyzer if available."""
        try:
            from presidio_analyzer import AnalyzerEngine
            self.analyzer = AnalyzerEngine()
            logger.info("Presidio analyzer initialized successfully")
        except ImportError:
            logger.warning("Presidio not available, will fall back to regex backend")
            self.analyzer = None

    def detect(self, text: str, *, locale: Optional[str] = None) -> List[PiiEntity]:
        """Detect PII entities using Presidio."""
        if not self.analyzer:
            # Fall back to regex if Presidio not available
            regex_detector = RegexPiiDetector(self.config)
            return regex_detector.detect(text, locale=locale)

        try:
            # Use Presidio for detection
            language = self._get_presidio_language(locale)
            results = self.analyzer.analyze(text=text, language=language)

            entities = []
            for result in results:
                # Map Presidio types to our canonical types
                canonical_type = self.TYPE_MAPPING.get(result.entity_type, result.entity_type)

                entity = PiiEntity(
                    type=canonical_type,
                    start=result.start,
                    end=result.end,
                    value=text[result.start:result.end] if self.config.return_values else None,
                    confidence=result.score,
                    pattern=f"presidio_{result.entity_type}",
                    locale=locale
                )

                # Apply confidence threshold
                if entity["confidence"] >= self.config.min_confidence:
                    entities.append(entity)

            return entities

        except Exception as e:
            logger.error(f"Presidio detection failed: {e}")
            # Fall back to regex on error
            regex_detector = RegexPiiDetector(self.config)
            return regex_detector.detect(text, locale=locale)

    def _get_presidio_language(self, locale: Optional[str]) -> str:
        """Map locale to Presidio language code."""
        if not locale:
            return "en"

        # Simple mapping - can be extended
        locale_mapping = {
            "sk": "en",  # Presidio doesn't support Slovak, use English
            "de": "de",
            "es": "es",
            "fr": "fr",
            "it": "it"
        }

        return locale_mapping.get(locale, "en")


class PiiAnalyzer:
    """Main PII analyzer with configurable backends."""

    def __init__(self, backend: str = "regex", **kwargs):
        """
        Initialize PII analyzer.

        Args:
            backend: Backend type ("regex" or "presidio")
            **kwargs: Additional configuration options
        """
        self.config = PiiConfig.from_env()

        # Override config with provided arguments
        if backend != "regex":
            self.config.backend = backend

        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self.detector = self._create_detector()

    def _create_detector(self) -> PiiDetector:
        """Create the appropriate detector based on configuration."""
        if self.config.backend == "presidio":
            return PresidioPiiDetector(self.config)
        else:
            return RegexPiiDetector(self.config)

    def detect(self, text: str, *, locale: Optional[str] = None) -> List[PiiEntity]:
        """
        Detect PII entities in text.

        Args:
            text: Input text to analyze
            locale: Optional locale hint

        Returns:
            List of PII entities sorted by start position
        """
        # Get raw detections from backend
        entities = self.detector.detect(text, locale=locale)

        # Apply type filtering
        entities = self._filter_entity_types(entities)

        # Handle overlapping entities
        entities = self._resolve_overlaps(entities)

        # Apply safety limits
        if len(entities) > self.config.max_entities_per_doc:
            logger.warning(f"Found {len(entities)} entities, limiting to {self.config.max_entities_per_doc}")
            entities = entities[:self.config.max_entities_per_doc]

        # Sort by start position for deterministic output
        entities.sort(key=lambda e: e["start"])

        return entities

    def _filter_entity_types(self, entities: List[PiiEntity]) -> List[PiiEntity]:
        """Filter entities based on include/exclude configuration."""
        if not self.config.types_include and not self.config.types_exclude:
            return entities

        filtered = []
        for entity in entities:
            entity_type = entity["type"].upper()

            # Check exclude list first
            if self.config.types_exclude and entity_type in self.config.types_exclude:
                continue

            # Check include list if specified
            if self.config.types_include and entity_type not in self.config.types_include:
                continue

            filtered.append(entity)

        return filtered

    def _resolve_overlaps(self, entities: List[PiiEntity]) -> List[PiiEntity]:
        """
        Resolve overlapping entities by preferring higher confidence and more specific types.
        """
        if len(entities) <= 1:
            return entities

        # Sort by start position, then by confidence (desc)
        sorted_entities = sorted(entities, key=lambda e: (e["start"], -e["confidence"]))

        # Type specificity order (higher number = more specific)
        type_specificity = {
            "EMAIL": 10,
            "IBAN": 9,
            "CREDIT_CARD": 8,
            "PASSPORT": 7,
            "PHONE": 6,
            "PERSON_NAME": 5,
            "ADDRESS": 4,
            "ID_NUMBER": 3,
            "DATE_OF_BIRTH": 2,
            "URL": 1
        }

        resolved = []
        for entity in sorted_entities:
            # Check for overlaps with already resolved entities
            overlaps = False
            for existing in resolved:
                if self._entities_overlap(entity, existing):
                    # Choose the better entity
                    if self._is_better_entity(entity, existing, type_specificity):
                        # Remove the existing entity and add the new one
                        resolved.remove(existing)
                        resolved.append(entity)
                    overlaps = True
                    break

            if not overlaps:
                resolved.append(entity)

        return resolved

    def _entities_overlap(self, entity1: PiiEntity, entity2: PiiEntity) -> bool:
        """Check if two entities overlap."""
        return not (entity1["end"] <= entity2["start"] or entity2["end"] <= entity1["start"])

    def _is_better_entity(self, new_entity: PiiEntity, existing_entity: PiiEntity,
                         type_specificity: Dict[str, int]) -> bool:
        """Determine if the new entity is better than the existing one."""
        new_confidence = new_entity["confidence"]
        existing_confidence = existing_entity["confidence"]

        # Prefer higher confidence
        if abs(new_confidence - existing_confidence) > 0.05:
            return new_confidence > existing_confidence

        # If confidence is similar, prefer more specific type
        new_specificity = type_specificity.get(new_entity["type"], 0)
        existing_specificity = type_specificity.get(existing_entity["type"], 0)

        return new_specificity > existing_specificity


def create_analyzer(backend: str = "regex", **kwargs) -> PiiAnalyzer:
    """Factory function to create PII analyzer."""
    return PiiAnalyzer(backend=backend, **kwargs)
