from __future__ import annotations

import logging
from typing import Optional

from ..processors.pii_analyzer import PiiAnalyzer, PiiEntity
from ..token_vault.service import get_vault_from_env

logger = logging.getLogger(__name__)


class QueryPseudonymizer:
    """
    Ingress query pseudonymizer. Detects PII in user queries and replaces them
    with deterministic tokens using the Token Vault so queries match tokenized corpus.

    Never logs plaintext values.
    """

    def __init__(self):
        self.vault = get_vault_from_env(actor_ctx=None)
        self.analyzer = PiiAnalyzer()

    def run(self, query: str, *, tenant_id: Optional[str], scope: str = "tenant") -> str:
        if not query:
            return ""
        # Detect entities
        entities = self.analyzer.detect(query)
        if not entities:
            return query
        # Replace right to left by spans
        parts = []
        curr = 0
        for ent in sorted(entities, key=lambda e: int(e["start"])):
            s = int(ent["start"]); e = int(ent["end"])
            if s < curr or s >= e:
                continue
            if s > curr:
                parts.append(query[curr:s])
            value = query[s:e]
            pii_type = str(ent["type"]).upper()
            try:
                tok = self.vault.get_or_create(value=value, type=pii_type, scope=scope, tenant_id=tenant_id)
                parts.append(tok["display"])  # safe display token
            except Exception as ex:  # On failure, keep plaintext to not break query (still avoid logging value)
                logger.error("Query pseudonymization error for type=%s: %s", pii_type, type(ex).__name__)
                parts.append(value)
            curr = e
        if curr < len(query):
            parts.append(query[curr:])
        return ''.join(parts)

