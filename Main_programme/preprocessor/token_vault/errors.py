"""
Token Vault Errors

Custom exception classes for token vault operations.
"""

class VaultError(Exception):
    """Base exception for all vault operations."""
    pass

class PermissionError(VaultError):
    """Raised when access is denied due to insufficient permissions."""
    pass

class IntegrityError(VaultError):
    """Raised when data integrity checks fail (e.g., tampering detected)."""
    pass

class NotFoundError(VaultError):
    """Raised when a token is not found in the vault."""
    pass

class ConfigurationError(VaultError):
    """Raised when vault configuration is invalid or incomplete."""
    pass

class EncryptionError(VaultError):
    """Raised when encryption/decryption operations fail."""
    pass
