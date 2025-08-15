import os

import pytest

from Main_programme.preprocessor.middleware.query_pseudonymizer import QueryPseudonymizer
from Main_programme.preprocessor.token_vault.service import get_vault_from_env


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
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    yield


def test_deterministic_token_matches_corpus():
    qp = QueryPseudonymizer()
    tenant = "tenantX"
    plaintext = "Contact john.doe@example.com please"
    tokenized_query = qp.run(plaintext, tenant_id=tenant, scope="tenant")

    # Build a corpus-like token by creating the same token directly
    vault = get_vault_from_env(actor_ctx=None)
    t = vault.get_or_create(value="john.doe@example.com", type="EMAIL", scope="tenant", tenant_id=tenant)
    assert t["display"] in tokenized_query


def test_retrieval_parity_pseudonymized_same_as_plaintext():
    qp = QueryPseudonymizer()
    tenant = "t1"
    plain = "Email john@example.com"
    tokenized = qp.run(plain, tenant_id=tenant, scope="tenant")

    # If user typed token directly, the query should be equal to pseudonymized form
    vault = get_vault_from_env(actor_ctx=None)
    t = vault.get_or_create(value="john@example.com", type="EMAIL", scope="tenant", tenant_id=tenant)
    direct = f"Email {t['display']}"
    assert tokenized == direct

