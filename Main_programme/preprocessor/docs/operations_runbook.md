# Operations Runbook

This guide explains how to run each pipeline phase, what to expect, and how to recover from failures.

## Phase-1: Preprocess (metadata + PII detection)

Run the preprocessing pipeline to produce Phase-1 JSONL files.

- Input: Directory of source documents.
- Output: JSONL files with text, metadata, and detected PII.

Command:

```
profidecon preprocess --input <docs_dir> --output <phase1_dir>
```

## Phase-2: Pseudonymize

Convert plaintext PII to deterministic tokens using the Token Vault.

- Input: Phase-1 JSONL directory.
- Output: Phase-2 JSONL with `text_pseudo`, `metadata.token_spans`, `metadata.pseudonymized`.

Command:

```
profidecon pseudonymize --input <phase1_dir> --output <phase2_dir> [--scope tenant|global] [--tenant-id <id>] [--force]
```

Notes:
- Use `--force` to override phase guards (dev only) if needed.
- No plaintext PII is written to output.

## Phase-3: LLM Enrichment (on pseudonymized text)

Generate summary and tags from pseudonymized text.

- Input: Phase-2 JSONL.
- Output: Phase-3 JSONL with `metadata.llm_enrichment` and `metadata.phase=3`.

Command:

```
profidecon enrich-llm --input <phase2_dir> --output <phase3_dir> --model <model_id> [--skip-if-present true|false]
```

Notes:
- Only `text_pseudo` is used. Never sends plaintext to LLM.
- Output is filtered for token-like patterns.

## Phase-4: On-Device De-anonymization

Preview masked de-anonymized output for dev on an authorized device.

Command (dev only):

```
ALLOW_DEV_DEANON=true profidecon dev-deanonymize --input <phase3_doc.jsonl> --tenant-id <id> --role case_handler --device edge_trusted --scope tenant
```

Notes:
- Output is masked even in dev mode.
- No persistence is performed; UI is responsible for display.

## Recovery Procedures

- Phase mismatch: Double-check you’re pointing to the correct phase directory. Use `--force` only for troubleshooting.
- Partial batch failures: Re-run the command; files with completed outputs are skipped unless `--force` is used.
- Token Vault unavailable: Retry once database/connectivity is restored. No tokens or plaintext are lost.

