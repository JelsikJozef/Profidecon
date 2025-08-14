"""
Token Vault Service Interface

Provides the main public interface for token vault operations including
deterministic token generation, secure storage, and resolution.
"""

import os
import logging
from typing import Protocol, Optional, Dict, Any
from .models import Token, Scope, TokenStats
from .rbac import ActorContext
from .errors import VaultError

logger = logging.getLogger(__name__)

class TokenVault(Protocol):
    """Protocol defining the token vault interface."""

    def get_or_create(self, *, value: str, type: str, scope: Scope,
                      tenant_id: Optional[str] = None) -> Token:
        """
        Deterministically map a value/type to a token within scope.
        Must return same token for same normalized(value)/type/scope/tenant.

        Args:
            value: PII value to tokenize
            type: PII type (EMAIL, PHONE, etc.)
            scope: Token scope (tenant or global)
            tenant_id: Optional tenant identifier

        Returns:
            Token with stable ID and display format
        """
        ...

    def resolve(self, *, token_id: str, scope: Scope,
                tenant_id: Optional[str] = None) -> str:
        """
        Return plaintext value (subject to RBAC).

        Args:
            token_id: Token identifier to resolve
            scope: Token scope
            tenant_id: Optional tenant identifier

        Returns:
            Original plaintext value

        Raises:
            NotFoundError: Token not found
            PermissionError: Access denied
        """
        ...

    def exists(self, *, token_id: str, scope: Scope) -> bool:
        """
        Check if token exists in vault.

        Args:
            token_id: Token identifier
            scope: Token scope

        Returns:
            True if token exists
        """
        ...

    def stats(self) -> TokenStats:
        """
        Get vault statistics.

        Returns:
            Dictionary with vault statistics
        """
        ...

def get_vault_from_env(actor_ctx: Optional[ActorContext] = None) -> TokenVault:
    """
    Factory function to create TokenVault instance from environment configuration.

    Args:
        actor_ctx: Actor context for RBAC (optional)

    Returns:
        Configured TokenVault instance

    Raises:
        ConfigurationError: Invalid or missing configuration
    """
    backend = os.getenv("TOKEN_VAULT_BACKEND", "postgres")

    if backend == "postgres":
        from .store_postgres import PostgresTokenVault
        return PostgresTokenVault(actor_ctx=actor_ctx)
    else:
        raise VaultError(f"Unsupported token vault backend: {backend}")

# Convenience functions for common operations
def create_token(value: str, type: str, scope: Scope = "tenant",
                tenant_id: Optional[str] = None,
                actor_ctx: Optional[ActorContext] = None) -> Token:
    """Create a token using default vault configuration."""
    vault = get_vault_from_env(actor_ctx)
    return vault.get_or_create(value=value, type=type, scope=scope, tenant_id=tenant_id)

def resolve_token(token_id: str, scope: Scope = "tenant",
                 tenant_id: Optional[str] = None,
                 actor_ctx: Optional[ActorContext] = None) -> str:
    """Resolve a token using default vault configuration."""
    vault = get_vault_from_env(actor_ctx)
    return vault.resolve(token_id=token_id, scope=scope, tenant_id=tenant_id)
