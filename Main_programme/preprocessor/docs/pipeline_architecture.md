# Profidecon Pipeline Architecture

This document outlines the end-to-end pipeline used in Profidecon for secure document processing.

## Phases

1. Phase-1: Metadata & PII Detection (processors/enricher_metadata.py, processors/pii_analyzer.py)
   - No LLM calls. Extracts language, counts, category, computes hash, detects PII spans.
2. Phase-2: Pseudonymization (processors/pseudonymizer.py)
   - Deterministic, reversible tokens via Token Vault. Emits spans and counts; never stores plaintext.
3. Phase-3: LLM Enrichment (processors/enricher_llm.py)
   - Operates on pseudonymized text (text_pseudo) only; returns summary/tags; strips token-like output.
4. Phase-4: On-Device De-anonymization (middleware/response_deanonymizer.py)
   - Restores plaintext only at the presentation layer for authorized actors on trusted devices.

## Data Flow

source docs → parse/normalize → Phase-1 JSONL → Phase-2 JSONL (text_pseudo, token_spans) → Phase-3 JSONL (llm_enrichment) → UI (Phase-4 masked/controlled de-anonymization)

## Module Boundaries

- processors/: parsing, normalization, metadata, pseudonymizer, LLM enricher
- token_vault/: storage, crypto, RBAC, service factory
- middleware/: query pseudonymizer (ingress), response de-anonymizer (egress)

## Observability

- observability/metrics.py: Prometheus counters/histograms
- observability/tracing.py: OpenTelemetry spans
- observability/logging_config.py: JSON structured logs with redaction


