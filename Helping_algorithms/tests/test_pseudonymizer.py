import os
import time
import json
import re
import pytest

from Main_programme.preprocessor.processors.pseudonymizer import Pseudonymizer
from Main_programme.preprocessor.processors.pii_analyzer import PiiAnalyzer, PiiEntity
from Main_programme.preprocessor.token_vault.service import get_vault_from_env
from Main_programme.preprocessor.token_vault.rbac import ActorContext


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
        "PSEUDONYMIZER_REQUIRE_ANNOTATIONS": "true",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    yield


def test_basic_roundtrip_and_determinism():
    analyzer = PiiAnalyzer()
    text = "Contact jane.doe@example.com or jane.doe@example.com and +421 900 123 456."
    entities = analyzer.detect(text)
    # Only EMAIL and PHONE
    entities = [e for e in entities if e["type"] in ("EMAIL", "PHONE")]

    pseudo = Pseudonymizer(scope="tenant", tenant_id="t1")
    r1 = pseudo.run(text, entities)
    r2 = pseudo.run(text, entities)

    assert r1["text"] == r2["text"]
    # Replace both emails with identical token string
    email_tokens = [s for s in r1["spans"] if s["type"] == "EMAIL"]
    assert len(email_tokens) == 2
    assert email_tokens[0]["token"] == email_tokens[1]["token"]
    # Phone token present
    assert any(s["type"] == "PHONE" for s in r1["spans"])
    # Token display format
    for s in r1["spans"]:
        assert re.match(r"^\[[A-Z_]+:[0-9A-Z]+\]$", s["token"]) is not None


def test_unicode_spans_correct():
    text = "Stretol som Jána Nováka na ulici. Straße, العربية, עברית."
    # Manually craft person name span for "Jána Nováka"
    start = text.index("Jána")
    end = start + len("Jána Nováka")
    ent: PiiEntity = {"type": "PERSON_NAME", "start": start, "end": end, "value": None, "confidence": 0.99, "pattern": None, "locale": None}
    pseudo = Pseudonymizer(scope="tenant", tenant_id="tenantX")
    res = pseudo.run(text, [ent])

    # Ensure token inserted and diacritics outside are intact
    assert res["text"].count("Jána") == 0
    assert "Straße" in res["text"]
    assert "العربية" in res["text"]
    assert "עברית" in res["text"]
    # Check span mapping token_start/token_end length correctness
    ts = res["spans"][0]
    assert res["text"][ts["token_start"]:ts["token_end"]] == ts["token"]


def test_overlap_longer_wins():
    text = "Ref 123-4567 is code"
    # Short inside long
    long_start = text.index("123-4567")
    long_end = long_start + len("123-4567")
    short_start = text.index("4567")
    short_end = short_start + 4
    ents = [
        {"type": "ID_NUMBER", "start": short_start, "end": short_end, "value": None, "confidence": 0.7, "pattern": None, "locale": None},
        {"type": "ID_NUMBER", "start": long_start, "end": long_end, "value": None, "confidence": 0.9, "pattern": None, "locale": None},
    ]
    pseudo = Pseudonymizer(scope="tenant", tenant_id="t2")
    res = pseudo.run(text, ents)
    assert sum(res["stats"].values()) == 1
    assert len(res["spans"]) == 1


def test_idempotent_runs_same_output():
    analyzer = PiiAnalyzer()
    text = "Email: john@example.com."
    ents = analyzer.detect(text)
    pseudo = Pseudonymizer(scope="tenant", tenant_id="t3")
    r1 = pseudo.run(text, ents)
    r2 = pseudo.run(text, ents)
    assert r1["text"] == r2["text"]


def test_include_exclude_filters():
    analyzer = PiiAnalyzer()
    text = "John Doe john@example.com +1 202 555 0100"
    ents = analyzer.detect(text)
    pseudo = Pseudonymizer(scope="tenant", tenant_id="t4")

    # Include only EMAIL
    r_inc = pseudo.run(text, ents, types_include=["EMAIL"])
    assert any(s["type"] == "EMAIL" for s in r_inc["spans"])
    assert not any(s["type"] == "PHONE" for s in r_inc["spans"])

    # Exclude EMAIL
    r_exc = pseudo.run(text, ents, types_exclude=["EMAIL"])
    assert not any(s["type"] == "EMAIL" for s in r_exc["spans"])


def test_large_text_performance():
    email = "user@example.com"
    # Build a ~60KB text
    chunk = ("prefix " + email + " suffix ") * 1000
    text = chunk
    analyzer = PiiAnalyzer()
    ents = analyzer.detect(text)
    pseudo = Pseudonymizer(scope="tenant", tenant_id="perf")

    t0 = time.time()
    res = pseudo.run(text, ents, max_entities=500)
    dt_ms = (time.time() - t0) * 1000
    assert dt_ms < 300.0
    # At least some replacements happened
    assert res["text"].count("example.com") == 0


def test_no_plaintext_in_logs(caplog):
    analyzer = PiiAnalyzer()
    text = "Contact alice@example.com"
    ents = analyzer.detect(text)
    pseudo = Pseudonymizer(scope="tenant", tenant_id="t5")
    with caplog.at_level("INFO"):
        res = pseudo.run(text, ents)
    # Ensure logs captured don't contain plaintext email
    for rec in caplog.records:
        assert "alice@example.com" not in rec.getMessage()

