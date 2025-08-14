"""
Unified Enricher Module

This module provides backward compatibility by orchestrating both
metadata and LLM enrichment through the refactored modules.

The original Enricher class is maintained for existing code compatibility,
but internally delegates to the new specialized enrichers.
"""

import logging
from pathlib import Path

from Main_programme.preprocessor.parsers.base import ParsedDocument
from .normalizer import normalize
from .enricher_metadata import MetadataEnricher
from .enricher_llm import LLMEnricher

logger = logging.getLogger(__name__)


class Enricher:
    """
    Unified enricher that maintains backward compatibility.

    This class orchestrates both metadata and LLM enrichment
    while preserving the original API.
    """

    def __init__(self, root_path: Path, llm_backend: str = "openai"):
        """
        Initialize the unified enricher.

        Args:
            root_path: Root directory for category extraction
            llm_backend: Backend to use for LLM operations
        """
        self.root_path = root_path
        self.metadata_enricher = MetadataEnricher(root_path)
        self.llm_enricher = LLMEnricher(llm_backend)

    def enrich(self, doc: ParsedDocument) -> ParsedDocument:
        """
        Enrich document with both metadata and LLM-generated content.

        Args:
            doc: ParsedDocument to enrich

        Returns: Enriched ParsedDocument
        """
        # Normalize text first
        normalized_doc = normalize(doc)
        text = normalized_doc.text

        # Convert to dict format for the new enrichers
        document_dict = {
            "text": text,
            "metadata": doc.metadata,
            "summary": doc.summary,
            "tags": doc.tags
        }

        # Apply metadata enrichment
        enriched_dict = self.metadata_enricher.run(document_dict)

        # Apply LLM enrichment
        final_dict = self.llm_enricher.run(enriched_dict)

        # Convert back to ParsedDocument
        return ParsedDocument(
            text=final_dict.get("text", ""),
            summary=final_dict.get("summary", ""),
            tags=final_dict.get("tags", []),
            metadata=final_dict.get("metadata", {})
        )
