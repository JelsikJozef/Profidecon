# Security Model

This document describes the security model for the pseudonymization → enrichment → de-anonymization pipeline.

## Principles

- Minimum exposure: Only the UI/device sees plaintext. All servers operate on pseudonymized data.
- Deterministic tokens: Token Vault returns stable, reversible tokens per scope/tenant.
- Tenant isolation: Tokens are namespaced by tenant. Resolve requires matching tenant and RBAC.
- Separation of duties: Phase-1/2/3 do not require decryption. Phase-4 resolves on device edge.
- No plaintext in logs: Structured JSON logs redact common PII patterns and never include resolved values.

## RBAC & Device Trust

- Default policy allows case_handler to resolve common PII (EMAIL, PHONE, PERSON_NAME, ADDRESS, ID_SSN) only on edge_trusted devices.
- Admin is allowed for all types.
- Viewer cannot resolve.
- Device trust enforced via DEANON_DEVICE_REQUIRED (default edge_trusted).

## Token Vault

- AES-GCM envelope encryption with KEK (PBKDF2-derived if needed).
- HMAC-based token IDs (Crockford base32) from normalized inputs and scope/tenant.
- Audit logs for create/resolve with: op, decision, token_type, token_id, tenant_id (no plaintext).
- SQLite/Postgres storage; schema supports tenant_id scoping.

## LLM Safety

- Phase-3 LLM receives pseudonymized text only.
- Output is filtered to strip token-like patterns and suspicious injection content.
- Security events are logged to security.events logger if token-like output is detected.

## Phase Guards

- CLI refuses out-of-order phases unless --force is set.
- Phase-3 requires text_pseudo and pseudonymized metadata.
- Phase-4 requires device trust; dev modes return masked values only.


