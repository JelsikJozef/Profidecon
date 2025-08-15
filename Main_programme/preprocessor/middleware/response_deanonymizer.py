from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Protocol, Optional, Iterable

from ..observability.metrics import deanon_allow_total, deanon_deny_total, phase_duration
from ..token_vault.service import get_vault_from_env
from ..token_vault.errors import NotFoundError, PermissionError, VaultError, IntegrityError

logger = logging.getLogger(__name__)

# Metrics


class ActorContext(Protocol):
    """Represents the authenticated user/session.
    Must provide: tenant_id, roles, device_trust_level, request_id."""
    tenant_id: Optional[str]
    roles: Iterable[str]
    device_trust_level: str  # e.g., "edge_trusted", "web", "unknown"
    request_id: str


class DeanonymizationPolicy(Protocol):
    def may_deanonymize(self, *, actor: ActorContext, scope: str, token_type: str) -> bool: ...
    def redact_on_deny(self, token_display: str) -> str: ...


class DefaultDeanonymizationPolicy:
    ALLOWED_TYPES_BY_ROLE = {
        "case_handler": {"PERSON_NAME", "EMAIL", "PHONE", "ADDRESS", "ID_SSN"},
        "viewer": set(),
        "admin": {"*"},
    }

    def __init__(self):
        # Optional override via env
        raw = os.getenv("DEANON_ALLOWED_TYPES_BY_ROLE")
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    self.ALLOWED_TYPES_BY_ROLE = {str(k): set(map(str, v)) for k, v in data.items()}
            except Exception:
                logger.warning("Invalid DEANON_ALLOWED_TYPES_BY_ROLE; using defaults")

    def may_deanonymize(self, *, actor: ActorContext, scope: str, token_type: str) -> bool:
        required = os.getenv("DEANON_DEVICE_REQUIRED", "edge_trusted")
        if actor.device_trust_level != required:
            return False
        if any(r == "admin" for r in actor.roles):
            return True
        allowed: set[str] = set()
        for r in actor.roles:
            allowed |= self.ALLOWED_TYPES_BY_ROLE.get(r, set())
        return ("*" in allowed) or (token_type in allowed)

    def redact_on_deny(self, token_display: str) -> str:
        ttype = token_display.split(":", 1)[0].lstrip("[")
        return f"[{ttype}:REDACTED]"


def _mask_partial(value: str, token_type: str) -> str:
    """Partial masking for dev/preview. Avoid leaking full plaintext."""
    if token_type == "EMAIL":
        # j***@domain
        parts = value.split("@", 1)
        if len(parts) == 2 and parts[0]:
            name, dom = parts
            head = name[:1]
            tail = name[-1:] if len(name) > 1 else ""
            return f"{head}***{tail}@{dom}"
    # Generic: keep first/last 2
    if len(value) <= 4:
        return "*" * len(value)
    return value[:2] + ("*" * (len(value) - 4)) + value[-2:]


class ResponseDeanonymizer:
    TOKEN_RE = re.compile(r"\[([A-Z_]+):([0-9A-Z]+)\]")

    def __init__(self, policy: DeanonymizationPolicy):
        # resolve uses RBAC hooks internally (bound at construction). For dev, default is permissive.
        self.vault = get_vault_from_env(actor_ctx=None)
        self.policy = policy

    def run(self, text_with_tokens: str, *, actor: ActorContext, scope: str = "tenant") -> str:
        """
        Replace display tokens like `[EMAIL:K7V2WQ3M]` with plaintext values
        for which the actor is authorized. Never logs plaintext.
        """
        if not text_with_tokens:
            return text_with_tokens or ""

        # Idempotent: if no tokens present, return as-is
        def _sub(match: re.Match) -> str:
            token_type = match.group(1)
            token_id = match.group(2)
            token_display = match.group(0)

            allowed = self.policy.may_deanonymize(actor=actor, scope=scope, token_type=token_type)
            audit_base = {
                "request_id": actor.request_id,
                "tenant_id": actor.tenant_id,
                "roles": list(actor.roles),
                "token_type": token_type,
                "token_id": token_id,
            }

            if not allowed:
                deanon_deny_total.inc()
                logging.warning("deanon_audit decision=deny %s", audit_base)
                return self.policy.redact_on_deny(token_display)

            # Resolve with tracing and latency
            t0 = time.perf_counter()
            try:
                plaintext = self.vault.resolve(token_id=token_id, scope=scope, tenant_id=actor.tenant_id)
            except (NotFoundError, PermissionError, IntegrityError, VaultError) as e:
                deanon_deny_total.inc()
                logging.warning("deanon_audit decision=deny error=%s %s", type(e).__name__, audit_base)
                return self.policy.redact_on_deny(token_display)
            finally:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                # record as phase 4 latency via phase_duration
                phase_duration.labels(phase="4").observe(dt_ms / 1000.0)

            deanon_allow_total.inc()
            logging.warning("deanon_audit decision=allow %s", audit_base)

            # Optional dev safety: never return full plaintext if server persistence is enabled
            if os.getenv("DEANON_PERSIST_SERVER", "false").lower() == "true":
                logging.warning("DEANON_PERSIST_SERVER=true (DEV ONLY). Returning masked values.")
                return _mask_partial(plaintext, token_type)

            return plaintext

        # Replace tokens safely
        return self.TOKEN_RE.sub(_sub, text_with_tokens)
