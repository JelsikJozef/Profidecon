"""
Token Vault Package

Provides reversible, deterministic, and secure tokenization service for PII values.
Supports typed display tokens (e.g., [PERSON:ABCD1234]) with secure storage
and last-mile de-anonymization capabilities.
"""

from .service import TokenVault, get_vault_from_env
from .models import Token, VaultRecord, Scope
from .errors import VaultError, PermissionError, IntegrityError, NotFoundError

__all__ = [
    "TokenVault",
    "get_vault_from_env",
    "Token",
    "VaultRecord",
    "Scope",
    "VaultError",
    "PermissionError",
    "IntegrityError",
    "NotFoundError"
]
