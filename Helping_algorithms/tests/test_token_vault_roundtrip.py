"""
Token Vault Round-trip Tests

Tests the basic functionality of token creation and resolution,
ensuring that values can be tokenized and then recovered correctly.
"""

import unittest
import tempfile
import os
from pathlib import Path

from Main_programme.preprocessor.token_vault import (
    get_vault_from_env, Token, VaultError, IntegrityError, NotFoundError
)
from Main_programme.preprocessor.token_vault.rbac import ActorContext
from Main_programme.preprocessor.token_vault.crypto import get_crypto_manager


class TestTokenVaultRoundtrip(unittest.TestCase):
    """Test round-trip token operations."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment with crypto keys."""
        # Set up minimal environment for testing
        cls.test_env = {
            "TOKEN_VAULT_BACKEND": "postgres",
            "DATABASE_URL": "sqlite:///:memory:",  # Would use PostgreSQL in real tests
            "HMAC_KEY_MATERIAL": "base64:dGVzdF9obWFjX2tleV8zMl9ieXRlc19sb25nX2Vub3VnaF9mb3JfdGVzdGluZw==",
            "KEK_MATERIAL": "base64:dGVzdF9rZWtfMzJfYnl0ZXNfbG9uZ19lbm91Z2hfZm9yX3Rlc3Rpbmc=",
            "SALT_V1": "base64:dGVzdF9zYWx0XzE2X2J5dGVzX2hlcmU=",
            "TOKEN_ID_BYTES": "10",
            "PSEUDO_SCOPE": "tenant"
        }

        # Apply test environment
        for key, value in cls.test_env.items():
            os.environ[key] = value

    def setUp(self):
        """Set up test fixtures."""
        self.actor_ctx = ActorContext(
            user_id="test_user",
            tenant_id="test_tenant",
            roles=["vault_admin"],
            permissions=["vault:create", "vault:resolve"]
        )

    def test_basic_round_trip(self):
        """Test basic token creation and resolution."""
        vault = get_vault_from_env(self.actor_ctx)

        # Test data for different PII types
        test_cases = [
            ("john.doe@example.com", "EMAIL"),
            ("+1-555-123-4567", "PHONE"),
            ("John Doe", "PERSON_NAME"),
            ("SK8975000000000012345671", "IBAN"),
            ("A1234567", "PASSPORT")
        ]

        for value, pii_type in test_cases:
            with self.subTest(value=value, type=pii_type):
                # Create token
                token = vault.get_or_create(
                    value=value,
                    type=pii_type,
                    scope="tenant",
                    tenant_id="test_tenant"
                )

                # Verify token structure
                self.assertIsInstance(token, dict)
                self.assertIn("id", token)
                self.assertIn("type", token)
                self.assertIn("display", token)
                self.assertEqual(token["type"], pii_type)
                self.assertTrue(token["display"].startswith(f"[{pii_type}:"))
                self.assertTrue(token["display"].endswith("]"))

                # Resolve token back to original value
                resolved_value = vault.resolve(
                    token_id=token["id"],
                    scope="tenant",
                    tenant_id="test_tenant"
                )

                # Verify round-trip preserves normalized value
                crypto = get_crypto_manager()
                expected_normalized = crypto.normalize(value, pii_type)
                self.assertEqual(resolved_value, expected_normalized)

    def test_deterministic_tokens(self):
        """Test that same input produces same token."""
        vault = get_vault_from_env(self.actor_ctx)

        # Create token twice with same input
        token1 = vault.get_or_create(
            value="test@example.com",
            type="EMAIL",
            scope="tenant",
            tenant_id="test_tenant"
        )

        token2 = vault.get_or_create(
            value="test@example.com",
            type="EMAIL",
            scope="tenant",
            tenant_id="test_tenant"
        )

        # Should be identical
        self.assertEqual(token1["id"], token2["id"])
        self.assertEqual(token1["display"], token2["display"])
        self.assertEqual(token1["type"], token2["type"])

    def test_normalization_consistency(self):
        """Test that normalization produces consistent tokens."""
        vault = get_vault_from_env(self.actor_ctx)

        # Test email normalization (case insensitive)
        token1 = vault.get_or_create(
            value="Test@Example.Com",
            type="EMAIL",
            scope="tenant",
            tenant_id="test_tenant"
        )

        token2 = vault.get_or_create(
            value="test@example.com",
            type="EMAIL",
            scope="tenant",
            tenant_id="test_tenant"
        )

        self.assertEqual(token1["id"], token2["id"])

        # Resolve should return normalized form
        resolved = vault.resolve(
            token_id=token1["id"],
            scope="tenant",
            tenant_id="test_tenant"
        )
        self.assertEqual(resolved, "test@example.com")

    def test_scope_isolation(self):
        """Test that different scopes produce different tokens."""
        vault = get_vault_from_env(self.actor_ctx)

        # Same value in different scopes should produce different tokens
        tenant_token = vault.get_or_create(
            value="test@example.com",
            type="EMAIL",
            scope="tenant",
            tenant_id="test_tenant"
        )

        global_token = vault.get_or_create(
            value="test@example.com",
            type="EMAIL",
            scope="global"
        )

        self.assertNotEqual(tenant_token["id"], global_token["id"])

    def test_tenant_isolation(self):
        """Test that different tenants produce different tokens."""
        vault = get_vault_from_env(self.actor_ctx)

        token1 = vault.get_or_create(
            value="test@example.com",
            type="EMAIL",
            scope="tenant",
            tenant_id="tenant1"
        )

        token2 = vault.get_or_create(
            value="test@example.com",
            type="EMAIL",
            scope="tenant",
            tenant_id="tenant2"
        )

        self.assertNotEqual(token1["id"], token2["id"])

    def test_type_isolation(self):
        """Test that different types produce different tokens."""
        vault = get_vault_from_env(self.actor_ctx)

        # Same value with different types should produce different tokens
        email_token = vault.get_or_create(
            value="12345",
            type="EMAIL",  # Not a real email, but testing type isolation
            scope="tenant",
            tenant_id="test_tenant"
        )

        id_token = vault.get_or_create(
            value="12345",
            type="ID_NUMBER",
            scope="tenant",
            tenant_id="test_tenant"
        )

        self.assertNotEqual(email_token["id"], id_token["id"])

    def test_token_not_found(self):
        """Test resolution of non-existent token."""
        vault = get_vault_from_env(self.actor_ctx)

        with self.assertRaises(NotFoundError):
            vault.resolve(
                token_id="NONEXISTENT123",
                scope="tenant",
                tenant_id="test_tenant"
            )

    def test_token_exists(self):
        """Test token existence checking."""
        vault = get_vault_from_env(self.actor_ctx)

        # Create a token
        token = vault.get_or_create(
            value="exists@example.com",
            type="EMAIL",
            scope="tenant",
            tenant_id="test_tenant"
        )

        # Should exist
        self.assertTrue(vault.exists(token_id=token["id"], scope="tenant"))

        # Non-existent token should not exist
        self.assertFalse(vault.exists(token_id="NONEXISTENT", scope="tenant"))


if __name__ == '__main__':
    unittest.main()
