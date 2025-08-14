"""
Token Vault Security Tests

Tests security aspects including encryption integrity, RBAC enforcement,
logging security, and environment-based configuration.
"""

import unittest
import os
import logging
from unittest.mock import patch, MagicMock
from io import StringIO

from Main_programme.preprocessor.token_vault import (
    get_vault_from_env, PermissionError, IntegrityError
)
from Main_programme.preprocessor.token_vault.rbac import ActorContext, set_rbac_hook, StrictRBACHook
from Main_programme.preprocessor.token_vault.crypto import get_crypto_manager
from Main_programme.preprocessor.token_vault.errors import ConfigurationError


class TestTokenVaultSecurity(unittest.TestCase):
    """Test security aspects of token vault."""

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
        self.authorized_actor = ActorContext(
            user_id="authorized_user",
            tenant_id="test_tenant",
            roles=["vault_admin"],
            permissions=["vault:create", "vault:resolve"]
        )

        self.unauthorized_actor = ActorContext(
            user_id="unauthorized_user",
            tenant_id="test_tenant",
            roles=["readonly"],
            permissions=["vault:read"]
        )

    def test_encryption_integrity(self):
        """Test that tampering with encrypted data is detected."""
        vault = get_vault_from_env(self.authorized_actor)
        crypto = get_crypto_manager()

        # Create a token
        token = vault.get_or_create(
            value="sensitive@example.com",
            type="EMAIL",
            scope="tenant",
            tenant_id="test_tenant"
        )

        # Test direct crypto tampering
        plaintext = "original_value"
        encrypted_key, cipher_value, nonce, tag = crypto.aesgcm_encrypt(plaintext)

        # Tamper with ciphertext
        tampered_cipher = bytearray(cipher_value)
        tampered_cipher[0] ^= 1  # Flip one bit

        # Should raise IntegrityError
        with self.assertRaises(IntegrityError):
            crypto.aesgcm_decrypt(encrypted_key, bytes(tampered_cipher), nonce, tag)

        # Tamper with tag
        tampered_tag = bytearray(tag)
        tampered_tag[0] ^= 1

        with self.assertRaises(IntegrityError):
            crypto.aesgcm_decrypt(encrypted_key, cipher_value, nonce, bytes(tampered_tag))

    def test_rbac_enforcement_resolve(self):
        """Test RBAC enforcement on token resolution."""
        # Set strict RBAC
        set_rbac_hook(StrictRBACHook())

        try:
            # Create token with authorized actor
            authorized_vault = get_vault_from_env(self.authorized_actor)
            token = authorized_vault.get_or_create(
                value="protected@example.com",
                type="EMAIL",
                scope="tenant",
                tenant_id="test_tenant"
            )

            # Try to resolve with unauthorized actor
            unauthorized_vault = get_vault_from_env(self.unauthorized_actor)

            with self.assertRaises(PermissionError):
                unauthorized_vault.resolve(
                    token_id=token["id"],
                    scope="tenant",
                    tenant_id="test_tenant"
                )

            # Authorized actor should succeed
            resolved = authorized_vault.resolve(
                token_id=token["id"],
                scope="tenant",
                tenant_id="test_tenant"
            )
            self.assertIsNotNone(resolved)

        finally:
            # Reset to permissive for other tests
            from Main_programme.preprocessor.token_vault.rbac import DefaultRBACHook
            set_rbac_hook(DefaultRBACHook(permissive=True))

    def test_rbac_enforcement_create(self):
        """Test RBAC enforcement on token creation."""
        set_rbac_hook(StrictRBACHook())

        try:
            unauthorized_vault = get_vault_from_env(self.unauthorized_actor)

            # Unauthorized actor should not be able to create tokens
            with self.assertRaises(PermissionError):
                unauthorized_vault.get_or_create(
                    value="blocked@example.com",
                    type="EMAIL",
                    scope="tenant",
                    tenant_id="test_tenant"
                )

        finally:
            from Main_programme.preprocessor.token_vault.rbac import DefaultRBACHook
            set_rbac_hook(DefaultRBACHook(permissive=True))

    def test_tenant_isolation_rbac(self):
        """Test that RBAC enforces tenant isolation."""
        set_rbac_hook(StrictRBACHook())

        try:
            # Create token in tenant1
            tenant1_actor = ActorContext(
                user_id="user1",
                tenant_id="tenant1",
                roles=["vault_admin"],
                permissions=["vault:create", "vault:resolve"]
            )

            vault1 = get_vault_from_env(tenant1_actor)
            token = vault1.get_or_create(
                value="isolated@example.com",
                type="EMAIL",
                scope="tenant",
                tenant_id="tenant1"
            )

            # Try to resolve from different tenant
            tenant2_actor = ActorContext(
                user_id="user2",
                tenant_id="tenant2",
                roles=["vault_admin"],
                permissions=["vault:create", "vault:resolve"]
            )

            vault2 = get_vault_from_env(tenant2_actor)

            # Should fail due to tenant mismatch
            with self.assertRaises(PermissionError):
                vault2.resolve(
                    token_id=token["id"],
                    scope="tenant",
                    tenant_id="tenant1"  # Different tenant
                )

        finally:
            from Main_programme.preprocessor.token_vault.rbac import DefaultRBACHook
            set_rbac_hook(DefaultRBACHook(permissive=True))

    def test_no_sensitive_data_in_logs(self):
        """Test that sensitive data is not logged."""
        # Capture logs
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)

        vault_logger = logging.getLogger('Main_programme.preprocessor.token_vault')
        vault_logger.addHandler(handler)
        vault_logger.setLevel(logging.DEBUG)

        try:
            vault = get_vault_from_env(self.authorized_actor)

            sensitive_value = "very_sensitive_email@secret.com"
            token = vault.get_or_create(
                value=sensitive_value,
                type="EMAIL",
                scope="tenant",
                tenant_id="test_tenant"
            )

            # Resolve the token
            resolved = vault.resolve(
                token_id=token["id"],
                scope="tenant",
                tenant_id="test_tenant"
            )

            # Check logs don't contain sensitive data
            log_content = log_stream.getvalue()

            # Should not contain the original sensitive value
            self.assertNotIn(sensitive_value, log_content)
            self.assertNotIn("secret.com", log_content)

            # Should not contain resolved sensitive data
            self.assertNotIn(resolved, log_content)

            # Should contain token ID (which is safe)
            self.assertIn(token["id"], log_content)

        finally:
            vault_logger.removeHandler(handler)

    def test_kek_configuration_warning(self):
        """Test warning when KEK is not configured."""
        # Temporarily remove KEK
        original_kek = os.environ.get("KEK_MATERIAL")
        if "KEK_MATERIAL" in os.environ:
            del os.environ["KEK_MATERIAL"]

        try:
            with patch('Main_programme.preprocessor.token_vault.crypto.logger') as mock_logger:
                crypto = get_crypto_manager()

                # Should warn about insecure configuration
                mock_logger.warning.assert_called()
                warning_calls = [call for call in mock_logger.warning.call_args_list
                               if "KEK" in str(call)]
                self.assertGreater(len(warning_calls), 0, "Should warn about missing KEK")

        finally:
            # Restore KEK
            if original_kek:
                os.environ["KEK_MATERIAL"] = original_kek

    def test_missing_required_config(self):
        """Test behavior with missing required configuration."""
        required_vars = ["HMAC_KEY_MATERIAL", "SALT_V1"]

        for var_name in required_vars:
            original_value = os.environ.get(var_name)
            del os.environ[var_name]

            try:
                # Should raise configuration error
                with self.assertRaises(ConfigurationError):
                    get_crypto_manager()

            finally:
                # Restore original value
                if original_value:
                    os.environ[var_name] = original_value

    def test_envelope_encryption_flow(self):
        """Test envelope encryption with KEK."""
        crypto = get_crypto_manager()

        plaintext = "envelope_test_data"

        # Encrypt
        encrypted_key, cipher_value, nonce, tag = crypto.aesgcm_encrypt(plaintext)

        # Verify envelope encryption was used (encrypted key should be longer than raw key)
        self.assertGreater(len(encrypted_key), 32, "Encrypted key should include envelope data")

        # Decrypt
        decrypted = crypto.aesgcm_decrypt(encrypted_key, cipher_value, nonce, tag)
        self.assertEqual(decrypted, plaintext)

    def test_key_derivation_deterministic(self):
        """Test that key derivation is deterministic."""
        crypto = get_crypto_manager()

        test_data = "deterministic_test"

        # Generate HMAC multiple times
        hmac1 = crypto.hmac_id(test_data.encode())
        hmac2 = crypto.hmac_id(test_data.encode())

        self.assertEqual(hmac1, hmac2, "HMAC generation should be deterministic")

        # Generate salted hash multiple times
        hash1 = crypto.salted_hash(test_data)
        hash2 = crypto.salted_hash(test_data)

        self.assertEqual(hash1, hash2, "Salted hash should be deterministic")


if __name__ == '__main__':
    unittest.main()
