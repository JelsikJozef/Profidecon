"""
Metadata Enricher Module

This module performs only non-LLM enrichment tasks including:
- Language detection
- Character/word/token counts
- Basic structural metadata (paragraph count, sentence count, etc.)
- File statistics useful for downstream processors
- PII risk scoring
- Content hashing
- Category extraction from file paths

This module must not call any LLM or network API to maintain separation of concerns
and ensure fast, reliable metadata enrichment.
"""

import hashlib
import re
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Language detection with fallback
try:
    from langdetect import detect, LangDetectError
except ImportError:
    def detect(_text: str) -> str:
        return "unknown"

    class LangDetectError(Exception):
        pass


def estimate_tokens(text: str) -> int:
    """Estimate token count using simple whitespace splitting."""
    return len(text.split())


def estimate_characters(text: str) -> int:
    """Count total characters including whitespace."""
    return len(text)


def estimate_words(text: str) -> int:
    """Count words using regex to handle punctuation better."""
    words = re.findall(r'\b\w+\b', text)
    return len(words)


def estimate_sentences(text: str) -> int:
    """Estimate sentence count by counting sentence terminators."""
    sentence_endings = re.findall(r'[.!?]+', text)
    return len(sentence_endings)


def estimate_paragraphs(text: str) -> int:
    """Count paragraphs by splitting on double newlines."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return len(paragraphs)


def pii_risk_score(text: str) -> float:
    """
    Calculate PII risk score based on pattern detection.

    Looks for:
    - Email addresses
    - Phone numbers
    - Potential ID numbers

    Returns: Risk score between 0.0 and 1.0
    """
    total_words = len(text.split())
    if total_words == 0:
        return 0.0

    # Email patterns (fix the original regex)
    emails = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))

    # Phone patterns (fix the original regex)
    phones = len(re.findall(r'\+?\d{6,}', text))

    total_pii_indicators = emails + phones
    risk_score = min(total_pii_indicators / total_words, 1.0)

    return round(risk_score, 4)


def detect_language(text: str) -> str:
    """
    Detect document language with fallback handling.

    Returns: Language code (e.g., 'en', 'sk', 'de') or 'unknown'
    """
    if len(text.strip()) < 20:  # Too short for reliable detection
        return "unknown"

    try:
        return detect(text)
    except (LangDetectError, Exception):
        logger.warning("Language detection failed, returning 'unknown'")
        return "unknown"


def calculate_content_hash(text: str) -> str:
    """Calculate SHA-1 hash of document content for deduplication."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def extract_category_from_path(file_path: Path, root_path: Path) -> str:
    """
    Extract document category from file path relative to root.

    Args:
        file_path: Full path to the document
        root_path: Root directory path

    Returns: Category name (first directory in relative path) or 'root'
    """
    try:
        relative_path = file_path.relative_to(root_path)
        return relative_path.parent.parts[0] if len(relative_path.parent.parts) > 0 else "root"
    except ValueError:
        # File is not under root_path
        logger.warning(f"File {file_path} is not under root path {root_path}")
        return "external"


def get_file_stats(file_path: Path) -> Dict[str, Any]:
    """
    Get file system statistics.

    Returns: Dictionary with size_bytes, created_ts, modified_ts
    """
    try:
        stat = file_path.stat()
        return {
            "size_bytes": stat.st_size,
            "created_ts": stat.st_ctime,
            "modified_ts": stat.st_mtime,
        }
    except (OSError, FileNotFoundError) as e:
        logger.warning(f"Could not get file stats for {file_path}: {e}")
        return {
            "size_bytes": 0,
            "created_ts": 0,
            "modified_ts": 0,
        }


class MetadataEnricher:
    """
    Handles all non-LLM metadata enrichment tasks.

    This enricher focuses on extracting factual, deterministic metadata
    that doesn't require AI models or network calls.
    """

    def __init__(self, root_path: Path):
        """
        Initialize the metadata enricher.

        Args:
            root_path: Root directory for category extraction
        """
        self.root_path = root_path

    def run(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich document with metadata.

        Args:
            document: Document dictionary with 'text' and 'metadata' keys

        Returns: Document dictionary with enriched metadata
        """
        text = document.get("text", "")
        metadata = document.get("metadata", {})
        source_path = Path(metadata.get("source", ""))

        logger.debug(f"Enriching metadata for {source_path.name}")

        # Text analysis
        char_count = estimate_characters(text)
        word_count = estimate_words(text)
        token_count = estimate_tokens(text)
        sentence_count = estimate_sentences(text)
        paragraph_count = estimate_paragraphs(text)

        # Language detection
        language = detect_language(text)

        # Content hash for deduplication
        content_hash = calculate_content_hash(text)

        # PII risk assessment
        pii_risk = pii_risk_score(text)

        # File statistics
        file_stats = get_file_stats(source_path)

        # Category extraction
        category = extract_category_from_path(source_path, self.root_path)

        # Merge with existing metadata
        enriched_metadata = {
            **metadata,
            # Text statistics
            "char_count": char_count,
            "word_count": word_count,
            "token_estimate": token_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            # Content analysis
            "language": language,
            "hash_content": content_hash,
            "hash_sha1": content_hash,  # Keep backward compatibility
            "pii_risk": pii_risk,
            "category": category,
            # File statistics
            **file_stats,
        }

        return {
            **document,
            "metadata": enriched_metadata
        }


def run(document: Dict[str, Any], root_path: Path) -> Dict[str, Any]:
    """
    Stateless function interface for metadata enrichment.

    Args:
        document: Document dictionary to enrich
        root_path: Root path for category extraction

    Returns: Enriched document dictionary
    """
    enricher = MetadataEnricher(root_path)
    return enricher.run(document)
