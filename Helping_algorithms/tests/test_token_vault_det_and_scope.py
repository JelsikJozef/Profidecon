"""
Token Vault Determinism and Scope Tests

Tests deterministic token generation, scope isolation,
and collision resistance across large datasets.
"""

import unittest
import os
import random
import string
from collections import defaultdict

from Main_programme.preprocessor.token_vault import get_vault_from_env
from Main_programme.preprocessor.token_vault.rbac import ActorContext
from Main_programme.preprocessor.token_vault.crypto import get_crypto_manager


class TestTokenVaultDeterminismAndScope(unittest.TestCase):
    """Test deterministic behavior and scope isolation."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_env = {
            "TOKEN_VAULT_BACKEND": "postgres",
            "DATABASE_URL": "sqlite:///:memory:",
            "HMAC_KEY_MATERIAL": "base64:dGVzdF9obWFjX2tleV8zMl9ieXRlc19sb25nX2Vub3VnaF9mb3JfdGVzdGluZw==",
            "KEK_MATERIAL": "base64:dGVzdF9rZWtfMzJfYnl0ZXNfbG9uZ19lbm91Z2hfZm9yX3Rlc3Rpbmc=",
            "SALT_V1": "base64:dGVzdF9zYWx0XzE2X2J5dGVzX2hlcmU=",
            "TOKEN_ID_BYTES": "10",
            "PSEUDO_SCOPE": "tenant"
        }

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

    def test_deterministic_generation(self):
        """Test that token generation is deterministic."""
        vault = get_vault_from_env(self.actor_ctx)

        # Test multiple calls with same parameters
        test_cases = [
            ("john@example.com", "EMAIL", "tenant", "tenant1"),
            ("Jane Doe", "PERSON_NAME", "global", None),
            ("+1-555-987-6543", "PHONE", "tenant", "tenant2"),
        ]

        for value, pii_type, scope, tenant_id in test_cases:
            with self.subTest(value=value, type=pii_type, scope=scope):
                tokens = []

                # Generate same token multiple times
                for _ in range(5):
                    token = vault.get_or_create(
                        value=value,
                        type=pii_type,
                        scope=scope,
                        tenant_id=tenant_id
                    )
                    tokens.append(token)

                # All should be identical
                first_token = tokens[0]
                for token in tokens[1:]:
                    self.assertEqual(token["id"], first_token["id"])
                    self.assertEqual(token["type"], first_token["type"])
                    self.assertEqual(token["display"], first_token["display"])

    def test_scope_isolation_comprehensive(self):
        """Test comprehensive scope isolation."""
        vault = get_vault_from_env(self.actor_ctx)

        test_values = [
            "test@example.com",
            "John Smith",
            "+1-555-123-4567",
            "SK8975000000000012345671"
        ]

        types = ["EMAIL", "PERSON_NAME", "PHONE", "IBAN"]

        # Generate tokens in different scopes
        tenant_tokens = []
        global_tokens = []

        for i, (value, pii_type) in enumerate(zip(test_values, types)):
            tenant_token = vault.get_or_create(
                value=value,
                type=pii_type,
                scope="tenant",
                tenant_id=f"tenant_{i}"
            )

            global_token = vault.get_or_create(
                value=value,
                type=pii_type,
                scope="global"
            )

            tenant_tokens.append(tenant_token)
            global_tokens.append(global_token)

        # Verify no collisions between scopes
        tenant_ids = {token["id"] for token in tenant_tokens}
        global_ids = {token["id"] for token in global_tokens}

        self.assertEqual(len(tenant_ids.intersection(global_ids)), 0,
                        "Token IDs should not collide across scopes")

    def test_tenant_isolation_comprehensive(self):
        """Test comprehensive tenant isolation."""
        vault = get_vault_from_env(self.actor_ctx)

        value = "sensitive@company.com"
        pii_type = "EMAIL"
        tenants = [f"tenant_{i}" for i in range(10)]

        tokens = {}
        for tenant in tenants:
            token = vault.get_or_create(
                value=value,
                type=pii_type,
                scope="tenant",
                tenant_id=tenant
            )
            tokens[tenant] = token

        # All tokens should be different
        token_ids = [token["id"] for token in tokens.values()]
        unique_ids = set(token_ids)

        self.assertEqual(len(token_ids), len(unique_ids),
                        "All tenant tokens should have unique IDs")

    def test_type_isolation_comprehensive(self):
        """Test comprehensive type isolation."""
        vault = get_vault_from_env(self.actor_ctx)

        value = "12345678"  # Could be interpreted as different PII types
        types = ["EMAIL", "PHONE", "ID_NUMBER", "PASSPORT", "CREDIT_CARD"]

        tokens = {}
        for pii_type in types:
            token = vault.get_or_create(
                value=value,
                type=pii_type,
                scope="tenant",
                tenant_id="test_tenant"
            )
            tokens[pii_type] = token

        # All tokens should be different
        token_ids = [token["id"] for token in tokens.values()]
        unique_ids = set(token_ids)

        self.assertEqual(len(token_ids), len(unique_ids),
                        "Same value with different types should produce unique tokens")

    def test_collision_resistance(self):
        """Test collision resistance with large dataset."""
        vault = get_vault_from_env(self.actor_ctx)

        # Generate 1000 random values (reduced from 50k for test performance)
        num_values = 1000
        token_ids = set()

        def generate_random_email():
            """Generate random email address."""
            username = ''.join(random.choices(string.ascii_lowercase, k=8))
            domain = ''.join(random.choices(string.ascii_lowercase, k=6))
            return f"{username}@{domain}.com"

        for _ in range(num_values):
            value = generate_random_email()
            token = vault.get_or_create(
                value=value,
                type="EMAIL",
                scope="tenant",
                tenant_id="collision_test"
            )

            # Check for collision
            if token["id"] in token_ids:
                self.fail(f"Token ID collision detected: {token['id']}")

            token_ids.add(token["id"])

        # Verify we generated expected number of unique tokens
        self.assertEqual(len(token_ids), num_values,
                        "Should generate unique token for each input")

    def test_normalization_determinism(self):
        """Test that normalization produces deterministic results."""
        crypto = get_crypto_manager()

        # Test cases with expected normalization
        test_cases = [
            # (input, type, expected_normalized)
            ("Test@Example.COM", "EMAIL", "test@example.com"),
            ("  John   Doe  ", "PERSON_NAME", "John Doe"),
            ("+1-555-123-4567", "PHONE", "+15551234567"),
            ("sk 8975 0000 0000 0012 3456 71", "IBAN", "SK8975000000000123456771"),
            ("https://Example.COM/path", "URL", "https://example.com/path"),
        ]

        for input_value, pii_type, expected in test_cases:
            with self.subTest(input=input_value, type=pii_type):
                # Normalize multiple times - should be deterministic
                normalized1 = crypto.normalize(input_value, pii_type)
                normalized2 = crypto.normalize(input_value, pii_type)

                self.assertEqual(normalized1, normalized2,
                                "Normalization should be deterministic")
                self.assertEqual(normalized1, expected,
                                f"Normalization failed for {pii_type}")

    def test_token_id_length_consistency(self):
        """Test that token IDs have consistent length."""
        vault = get_vault_from_env(self.actor_ctx)

        # Generate tokens of different types and values
        test_cases = [
            ("short@x.co", "EMAIL"),
            ("very.long.email.address@extremely.long.domain.name.example.com", "EMAIL"),
            ("A", "PERSON_NAME"),
            ("Very Long Person Name With Many Words", "PERSON_NAME"),
        ]

        token_ids = []
        for value, pii_type in test_cases:
            token = vault.get_or_create(
                value=value,
                type=pii_type,
                scope="tenant",
                tenant_id="length_test"
            )
            token_ids.append(token["id"])

        # All token IDs should have same length (base32 encoding of fixed-length HMAC)
        first_length = len(token_ids[0])
        for token_id in token_ids[1:]:
            self.assertEqual(len(token_id), first_length,
                            "All token IDs should have consistent length")

        # Length should be reasonable (10 bytes -> 16 base32 chars)
        expected_length = 16  # base32 encoding of 10 bytes
        self.assertEqual(first_length, expected_length,
                        f"Token ID length should be {expected_length}")


if __name__ == '__main__':
    unittest.main()
