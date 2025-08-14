"""
LLM Enricher Module

This module performs LLM-based enrichment tasks including:
- Generating document summaries
- Extracting semantic tags
- Any other semantic enrichment that requires an LLM

This module encapsulates all AI/LLM dependencies and network calls,
maintaining clear separation from the metadata enricher.
"""

import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path

from .llm_picker import LLMPicker

logger = logging.getLogger(__name__)


class LLMEnricher:
    """
    Handles all LLM-based enrichment tasks.

    This enricher focuses on semantic analysis that requires
    AI models or external API calls.
    """

    def __init__(self, llm_backend: str = "huggingface"):
        """
        Initialize the LLM enricher.

        Args:
            llm_backend: Backend to use ('openai', 'ollama', 'huggingface')
        """
        self.llm_picker = LLMPicker(backend=llm_backend)
        self.backend = llm_backend

    def generate_summary_and_tags(self, text: str) -> Tuple[str, List[str]]:
        """
        Generate summary and extract tags using LLM.

        Args:
            text: Document text to analyze

        Returns: Tuple of (summary, tags_list)
        """
        if not text.strip():
            return "", []

        try:
            summary, tags = self.llm_picker.generate_summary_and_tags(text)
            return summary, tags
        except Exception as e:
            logger.error(f"LLM enrichment failed: {e}")
            return "", []

    def run(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich document with LLM-generated content.

        Args:
            document: Document dictionary with 'text' key

        Returns: Document dictionary with LLM-generated summary and tags
        """
        text = document.get("text", "")
        metadata = document.get("metadata", {})
        source_path = Path(metadata.get("source", ""))

        logger.info(f"Generating LLM enrichment for {source_path.name}")

        # Generate summary and tags
        summary, tags = self.generate_summary_and_tags(text)

        return {
            **document,
            "summary": summary,
            "tags": tags,
        }


def run(document: Dict[str, Any], llm_backend: str = "huggingface") -> Dict[str, Any]:
    """
    Stateless function interface for LLM enrichment.

    Args:
        document: Document dictionary to enrich
        llm_backend: LLM backend to use

    Returns: Document dictionary with LLM-generated content
    """
    enricher = LLMEnricher(llm_backend)
    return enricher.run(document)
