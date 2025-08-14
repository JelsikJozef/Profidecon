from __future__ import annotations

import os
import logging
from typing import TypedDict, List, Dict, Optional, Tuple

from Main_programme.preprocessor.processors.pii_analyzer import PiiEntity
from Main_programme.preprocessor.token_vault.service import get_vault_from_env

logger = logging.getLogger(__name__)


class TokenSpan(TypedDict):
    token: str
    token_id: str
    type: str
    orig_start: int
    orig_end: int
    token_start: int
    token_end: int


class PseudonymizationResult(TypedDict):
    text: str
    spans: List[TokenSpan]
    stats: Dict[str, int]


class Pseudonymizer:
    """
    Deterministic, reversible pseudonymizer using Token Vault.

    - Replaces PII spans with typed display tokens, e.g., "[EMAIL:K7V2WQ3M]".
    - Never logs plaintext values; only metadata and counts.
    - Handles overlapping spans (prefers longer), replaces right-to-left.
    """

    def __init__(self, scope: str = "tenant", tenant_id: Optional[str] = None):
        # Read defaults from env when not explicitly provided
        env_scope = os.getenv("PSEUDO_SCOPE", scope)
        self.scope = env_scope if env_scope in ("tenant", "global") else "tenant"
        self.tenant_id = tenant_id or os.getenv("PSEUDO_TENANT_ID") or None
        self.require_annotations = os.getenv("PSEUDONYMIZER_REQUIRE_ANNOTATIONS", "true").lower() == "true"
        self.vault = get_vault_from_env()

    @staticmethod
    def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return not (a[1] <= b[0] or b[1] <= a[0])

    def _resolve_overlaps(self, entities: List[PiiEntity]) -> List[PiiEntity]:
        """
        Resolve overlapping spans by preferring longer spans (more specific).
        Sort by (start asc, length desc) and greedily select non-overlapping ones.
        """
        if not entities:
            return []
        # Sort by start asc, length desc
        ents = sorted(entities, key=lambda e: (int(e["start"]), -(int(e["end"]) - int(e["start"])) ))
        selected: List[PiiEntity] = []
        for ent in ents:
            s, e = int(ent["start"]), int(ent["end"])
            if s >= e:
                continue
            if any(self._overlap((s, e), (int(se["start"]), int(se["end"])) ) for se in selected):
                continue
            selected.append(ent)
        selected.sort(key=lambda e: (int(e["start"]), int(e["end"])) )
        return selected

    def _filter_types(self, entities: List[PiiEntity], include: Optional[set[str]], exclude: Optional[set[str]]) -> List[PiiEntity]:
        out: List[PiiEntity] = []
        for e in entities:
            t = str(e["type"]).upper()
            if exclude and t in exclude:
                continue
            if include and t not in include:
                continue
            out.append(e)
        return out

    def run(
        self,
        text: str,
        entities: List[PiiEntity],
        *,
        types_include: Optional[List[str]] = None,
        types_exclude: Optional[List[str]] = None,
        max_entities: Optional[int] = None,
    ) -> PseudonymizationResult:
        # Prepare filters (fallback to env if not provided)
        if types_include is None:
            env_inc = os.getenv("PII_TYPES_INCLUDE")
            types_include = [t.strip() for t in env_inc.split(",")] if env_inc else None
        if types_exclude is None:
            env_exc = os.getenv("PII_TYPES_EXCLUDE")
            types_exclude = [t.strip() for t in env_exc.split(",")] if env_exc else None
        if max_entities is None:
            env_max = os.getenv("PII_MAX_ENTITIES_PER_DOC")
            max_entities = int(env_max) if env_max else None

        include_set = {t.strip().upper() for t in types_include} if types_include else None
        exclude_set = {t.strip().upper() for t in types_exclude} if types_exclude else None

        # Filter, resolve overlaps
        ents = self._filter_types(entities, include_set, exclude_set)
        ents = self._resolve_overlaps(ents)

        # Cache token lookups per (value,type) within this document
        cache: Dict[Tuple[str, str], Tuple[str, str]] = {}
        unique_keys_count = 0

        # We'll build the final output left-to-right to compute token indices accurately
        out_parts: List[str] = []
        spans: List[TokenSpan] = []
        counts: Dict[str, int] = {}
        curr = 0  # current position in original text
        out_len = 0  # length of output built so far

        for e in ents:
            start = int(e["start"])
            end = int(e["end"])
            if start < 0 or end > len(text) or start >= end or start < curr:
                logger.warning("Skipping invalid span indices for entity type %s", str(e.get("type")).upper())
                continue
            pii_type = str(e["type"]).upper()
            value = text[start:end]
            key = (value, pii_type)

            # Append the non-PII region before this entity
            if start > curr:
                seg = text[curr:start]
                if seg:
                    out_parts.append(seg)
                    out_len += len(seg)

            # Get/create token for this key, respecting unique limit
            if key in cache:
                token_display, token_id = cache[key]
            else:
                if max_entities is not None and unique_keys_count >= max_entities:
                    # Reached limit of unique values; do not replace, keep plaintext
                    out_parts.append(value)
                    out_len += len(value)
                    curr = end
                    continue
                token = self.vault.get_or_create(value=value, type=pii_type, scope=self.scope, tenant_id=self.tenant_id)
                token_id = token["id"]
                token_display = token["display"]
                cache[key] = (token_display, token_id)
                unique_keys_count += 1

            # Insert token and record span
            token_start = out_len
            out_parts.append(token_display)
            out_len += len(token_display)
            token_end = out_len
            spans.append(TokenSpan(
                token=token_display,
                token_id=token_id,
                type=pii_type,
                orig_start=start,
                orig_end=end,
                token_start=token_start,
                token_end=token_end,
            ))
            counts[pii_type] = counts.get(pii_type, 0) + 1
            curr = end

        # Append the remaining tail
        if curr < len(text):
            tail = text[curr:]
            if tail:
                out_parts.append(tail)
                out_len += len(tail)

        pseudonymized = ''.join(out_parts)
        # Spans are naturally in ascending order by token_start

        return PseudonymizationResult(text=pseudonymized, spans=spans, stats=counts)
