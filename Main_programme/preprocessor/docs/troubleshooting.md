# Troubleshooting

Common issues and fixes across the pipeline phases.

## Phase mismatch (phase guards)
- Symptom: CLI exits with "Phase guard failed".
- Cause: Input directory contains files from the wrong phase.
- Fix: Point to the correct directory; use `--force` only for dev verification.

## LLM API errors
- Symptom: HTTP timeouts or quota errors.
- Fix: Set OPENAI_API_KEY (or use the heuristic client). Reduce `--max-tokens`. Retry with exponential backoff.

## Token Vault connectivity
- Symptom: Database connection refused.
- Fix: Verify DATABASE_URL; for SQLite use `sqlite:///:memory:`. Ensure schema initialized. Check logs for `vault.audit`.

## Device trust failures
- Symptom: De-anonymization returns `[TYPE:REDACTED]` even for authorized users.
- Fix: Ensure `device_trust_level` equals `DEANON_DEVICE_REQUIRED` (default `edge_trusted`). Verify roles.

## Plaintext in outputs
- Symptom: Plaintext PII in intermediate files.
- Fix: Ensure Phase-2 outputs contain `text_pseudo`, not `text`. LLM phase must read `text_pseudo` only. Check security logs for injection filtering.

