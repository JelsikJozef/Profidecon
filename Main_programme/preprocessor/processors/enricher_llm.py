"""
LLM Enricher Module (Phase-3)

This module performs LLM-based enrichment tasks on pseudonymized text only:
- Generating document summaries
- Extracting semantic tags

Design:
- Model-agnostic via injected api_client adapter
- No plaintext PII must be logged or sent here; caller must pass pseudonymized text
- Tracks latency and returns it in the result
"""

from __future__ import annotations

import logging
import re
import time
from typing import Dict, Any, List, TypedDict

from ..observability.metrics import llm_calls_total, llm_latency_ms
from ..observability.tracing import span

logger = logging.getLogger(__name__)
security_logger = logging.getLogger("security.events")


class LlmEnrichmentResult(TypedDict):
    summary: str
    tags: List[str]
    model: str
    latency_ms: int


class LlmEnricher:
    """
    LLM Enricher that takes pseudonymized text and returns a concise summary
    and normalized tags. The class is model-agnostic through an injected
    api_client that exposes an `enrich(text, model, temperature, max_tokens)` method.
    """

    PROMPT_TEMPLATE = (
        "You are a document enrichment system.\n"
        "Given the following pseudonymized text, produce:\n"
        "1. A concise summary (3–5 sentences)\n"
        "2. A list of 5–10 relevant tags (lowercase, underscore_separated)\n\n"
        "Text:\n{input}\n"
    )

    def __init__(
        self,
        api_client: Any,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        *,
        max_tokens_input: int = 4096,
    ):
        self.client = api_client
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        # For simplicity we interpret max_tokens_input as a character budget.
        # This avoids adding a tokenizer dependency and is sufficient for tests.
        self.max_tokens_input = int(max_tokens_input)

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if limit <= 0 or not text:
            return ""
        if len(text) <= limit:
            return text
        return text[:limit]

    @staticmethod
    def _normalize_tags(raw_tags: Any) -> List[str]:
        """
        Normalize tags to lowercase underscore_separated and drop empties.
        Accepts either a list of strings or a single string with separators.
        """
        tags: List[str] = []
        if isinstance(raw_tags, list):
            tags = [str(t) for t in raw_tags]
        elif isinstance(raw_tags, str):
            # split on commas/semicolons/newlines
            parts = re.split(r"[\n,;]+", raw_tags)
            tags = [p.strip() for p in parts]
        else:
            return []

        normed: List[str] = []
        for t in tags:
            t = t.strip().lower().replace(" ", "_")
            t = re.sub(r"[^a-z0-9_]+", "", t)
            if t:
                normed.append(t)
        # de-duplicate preserving order
        seen = set()
        out: List[str] = []
        for t in normed:
            if t not in seen:
                out.append(t)
                seen.add(t)
        return out[:10]

    def run(self, text: str, metadata: Dict[str, Any]) -> LlmEnrichmentResult:
        """
        Send pseudonymized text to the LLM and return enrichment results.

        The caller is responsible for ensuring `text` is pseudonymized. This
        function will never attempt to de-tokenize or access plaintext PII.
        """
        safe_text = str(text or "")
        if not safe_text.strip():
            return LlmEnrichmentResult(summary="", tags=[], model=self.model, latency_ms=0)

        # Truncate input to protect latency and cost
        input_chunk = self._truncate(safe_text, self.max_tokens_input)

        # Only log the pseudonymized content length and a tiny safe preview
        preview = input_chunk[:80].replace("\n", " ")
        logger.info("LLM enrichment on pseudonymized text (len=%d, preview='%s...')", len(input_chunk), preview)

        # Build prompt
        prompt = self.PROMPT_TEMPLATE.format(input=input_chunk)

        t0 = time.monotonic()
        try:
            llm_calls_total.inc()
            with span("llm_enrich", model=self.model):
                resp = self.client.enrich(
                    text=prompt,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
        except Exception as e:
            logger.error("LLM API call failed: %s", e)
            return LlmEnrichmentResult(summary="", tags=[], model=self.model, latency_ms=int((time.monotonic() - t0) * 1000))
        dt_ms = int((time.monotonic() - t0) * 1000)
        llm_latency_ms.observe(dt_ms)

        # Extract fields
        summary = ""
        tags: List[str] = []
        if isinstance(resp, dict):
            summary = str(resp.get("summary", "") or "")
            tags = self._normalize_tags(resp.get("tags", []))
        elif isinstance(resp, str):
            summary = resp
            token_types = re.findall(r"\[([A-Z_]+):[0-9A-Z]+\]", input_chunk)
            inferred = [t.lower() for t in token_types]
            tags = self._normalize_tags(inferred)
        else:
            summary = ""
            tags = []

        # Prompt injection filtering: strip token patterns from outputs
        token_like = re.compile(r"\[[A-Z_]+:[0-9A-Z]+\]")
        if token_like.search(summary):
            security_logger.warning("llm_output_token_like_detected")
            summary = token_like.sub("", summary).strip()
        cleaned_tags: List[str] = []
        for t in tags:
            if token_like.search(t or ""):
                security_logger.warning("llm_output_token_like_detected_tag")
                continue
            cleaned_tags.append(t)
        tags = cleaned_tags

        return LlmEnrichmentResult(summary=summary, tags=tags, model=self.model, latency_ms=dt_ms)


# --- Backward-compatible wrapper for existing tests and modules ---
try:
    # Local import to avoid heavy dependencies unless used
    from .llm_picker import LLMPicker  # type: ignore
except Exception:  # pragma: no cover
    LLMPicker = None  # type: ignore


class LLMEnricher:
    """
    Backward-compatible wrapper that mirrors the old interface used in tests.

    It uses LLMPicker internally and exposes:
    - generate_summary_and_tags(text) -> (summary, tags)
    - run(document_dict) -> document_dict with 'summary' and 'tags'
    """

    def __init__(self, llm_backend: str = "huggingface"):
        self.backend = llm_backend
        if LLMPicker is None:
            raise ImportError("LLMPicker is not available")
        self.llm_picker = LLMPicker(backend=llm_backend)

    def generate_summary_and_tags(self, text: str) -> tuple[str, List[str]]:
        if not text or not text.strip():
            return "", []
        try:
            return self.llm_picker.generate_summary_and_tags(text)
        except Exception as e:  # Keep silent behavior expected by tests
            logger.error("LLM enrichment failed: %s", e)
            return "", []

    def run(self, document: Dict[str, Any]) -> Dict[str, Any]:
        text = str(document.get("text", "") or "")
        meta = document.get("metadata") or {}
        logger.info("Generating LLM enrichment for %s", meta.get("source") or "<unknown>")
        summary, tags = self.generate_summary_and_tags(text)
        out = dict(document)
        out["summary"] = summary
        out["tags"] = tags
        return out


def run(document: Dict[str, Any], llm_backend: str = "huggingface") -> Dict[str, Any]:
    enricher = LLMEnricher(llm_backend)
    return enricher.run(document)
