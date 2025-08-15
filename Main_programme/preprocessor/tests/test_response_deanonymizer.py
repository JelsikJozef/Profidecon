import os
import re
import json
import time
from dataclasses import dataclass

import pytest

from Main_programme.preprocessor.middleware.response_deanonymizer import (
    ResponseDeanonymizer,
    DefaultDeanonymizationPolicy,
)
from Main_programme.preprocessor.token_vault.service import get_vault_from_env


@dataclass
class FakeActor:
    tenant_id: str | None
    roles: list[str]
    device_trust_level: str
    request_id: str


@pytest.fixture(autouse=True)
def vault_env(monkeypatch):
    env = {
        "TOKEN_VAULT_BACKEND": "postgres",
        "DATABASE_URL": "sqlite:///:memory:",
        "HMAC_KEY_MATERIAL": "base64:dGVzdF9obWFjX2tleV8zMl9ieXRlc19sb25nX2Vub3VnaF9mb3JfdGVzdGluZw==",
        "KEK_MATERIAL": "base64:dGVzdF9rZWtfMzJfYnl0ZXNfbG9uZ19lbm91Z2hfZm9yX3Rlc3Rpbmc=",
        "SALT_V1": "base64:dGVzdF9zYWx0XzE2X2J5dGVzX2hlcmU=",
        "TOKEN_ID_BYTES": "10",
        "PSEUDO_SCOPE": "tenant",
        "DEANON_DEVICE_REQUIRED": "edge_trusted",
        "DEANON_PERSIST_SERVER": "false",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    yield


def _make_token(value: str, type_: str, tenant: str):
    vault = get_vault_from_env(actor_ctx=None)
    tok = vault.get_or_create(value=value, type=type_, scope="tenant", tenant_id=tenant)
    return tok


def test_allow_vs_deny(caplog):
    tok = _make_token("alice@example.com", "EMAIL", "t1")
    text = f"Hello {tok['display']} world"
    de = ResponseDeanonymizer(DefaultDeanonymizationPolicy())

    allow_actor = FakeActor(tenant_id="t1", roles=["case_handler"], device_trust_level="edge_trusted", request_id="r1")
    out_allow = de.run(text, actor=allow_actor)
    assert "alice@example.com" in out_allow

    deny_actor = FakeActor(tenant_id="t1", roles=["viewer"], device_trust_level="edge_trusted", request_id="r2")
    out_deny = de.run(text, actor=deny_actor)
    assert "[EMAIL:REDACTED]" in out_deny

    # No plaintext PII in logs
    for rec in caplog.records:
        assert "alice@example.com" not in rec.getMessage()


def test_unicode_and_idempotency():
    tok = _make_token("žáno@example.sk", "EMAIL", "t1")
    s = f"Stretol som Jána na ulici. {tok['display']} Straße العربية עברית."
    de = ResponseDeanonymizer(DefaultDeanonymizationPolicy())
    actor = FakeActor(tenant_id="t1", roles=["case_handler"], device_trust_level="edge_trusted", request_id="r3")

    once = de.run(s, actor=actor)
    assert "Jána" in once and "Straße" in once and "العربية" in once and "עברית" in once
    assert "@example.sk" in once

    twice = de.run(once, actor=actor)
    assert twice == once


def test_audit_and_tenant_isolation(caplog):
    tok = _make_token("bob@example.com", "EMAIL", "t1")
    text = f"{tok['display']}"
    de = ResponseDeanonymizer(DefaultDeanonymizationPolicy())

    wrong_tenant_actor = FakeActor(tenant_id="t2", roles=["case_handler"], device_trust_level="edge_trusted", request_id="r4")
    out = de.run(text, actor=wrong_tenant_actor)
    assert out == "[EMAIL:REDACTED]"

    # Audit events present; no plaintext
    logs = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "deanon_audit" in logs
    assert "decision=deny" in logs
    assert "bob@example.com" not in logs


def test_dev_persist_masks(monkeypatch):
    # When DEANON_PERSIST_SERVER=true, middleware returns masked values (dev only)
    monkeypatch.setenv("DEANON_PERSIST_SERVER", "true")
    tok = _make_token("john.doe@example.org", "EMAIL", "t1")
    text = f"{tok['display']}"
    de = ResponseDeanonymizer(DefaultDeanonymizationPolicy())
    actor = FakeActor(tenant_id="t1", roles=["case_handler"], device_trust_level="edge_trusted", request_id="r5")
    out = de.run(text, actor=actor)
    assert out.endswith("@example.org")
    assert "john.doe@" not in out

