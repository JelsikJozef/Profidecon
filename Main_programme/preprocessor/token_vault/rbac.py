"""
Token Vault Role-Based Access Control

Provides pluggable authorization hooks for token vault operations.
Currently implements stub functionality that can be extended with
real authorization logic.
"""

import logging
from typing import Optional, Any, Protocol

logger = logging.getLogger(__name__)

class ActorContext:
    """Context object representing an authenticated actor."""

    def __init__(self, user_id: Optional[str] = None, tenant_id: Optional[str] = None,
                 roles: Optional[list[str]] = None, permissions: Optional[list[str]] = None):
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.roles = roles or []
        self.permissions = permissions or []

    def has_role(self, role: str) -> bool:
        """Check if actor has a specific role."""
        return role in self.roles

    def has_permission(self, permission: str) -> bool:
        """Check if actor has a specific permission."""
        return permission in self.permissions

class RBACHook(Protocol):
    """Protocol for RBAC authorization hooks."""

    def check_resolve_permission(self, token_id: str, tenant_id: Optional[str],
                                actor_ctx: Optional[ActorContext]) -> bool:
        """Check if actor can resolve a token to its plaintext value."""
        ...

    def check_create_permission(self, value: str, type: str, tenant_id: Optional[str],
                               actor_ctx: Optional[ActorContext]) -> bool:
        """Check if actor can create tokens."""
        ...

class DefaultRBACHook:
    """Default RBAC implementation with configurable behavior."""

    def __init__(self, permissive: bool = True):
        """
        Initialize RBAC hook.

        Args:
            permissive: If True, allows all operations (dev mode)
        """
        self.permissive = permissive

    def check_resolve_permission(self, token_id: str, tenant_id: Optional[str],
                                actor_ctx: Optional[ActorContext]) -> bool:
        """Check token resolution permissions."""
        if self.permissive:
            logger.debug(f"Permissive mode: allowing token resolution for {token_id}")
            return True

        if not actor_ctx:
            logger.warning("No actor context provided, denying access")
            return False

        # Tenant isolation check
        if tenant_id and actor_ctx.tenant_id != tenant_id:
            logger.warning(f"Tenant mismatch: actor {actor_ctx.tenant_id} vs token {tenant_id}")
            return False

        # Check for required permissions
        required_permissions = ["vault:resolve", "pii:read"]
        if any(actor_ctx.has_permission(perm) for perm in required_permissions):
            return True

        # Check for admin role
        if actor_ctx.has_role("admin") or actor_ctx.has_role("vault_admin"):
            return True

        logger.warning(f"User {actor_ctx.user_id} lacks permission to resolve token {token_id}")
        return False

    def check_create_permission(self, value: str, type: str, tenant_id: Optional[str],
                               actor_ctx: Optional[ActorContext]) -> bool:
        """Check token creation permissions."""
        if self.permissive:
            return True

        if not actor_ctx:
            logger.warning("No actor context provided, denying token creation")
            return False

        # Tenant isolation check
        if tenant_id and actor_ctx.tenant_id != tenant_id:
            logger.warning(f"Tenant mismatch for token creation: {actor_ctx.tenant_id} vs {tenant_id}")
            return False

        # Check for required permissions
        required_permissions = ["vault:create", "pii:tokenize"]
        if any(actor_ctx.has_permission(perm) for perm in required_permissions):
            return True

        # Check for admin role
        if actor_ctx.has_role("admin") or actor_ctx.has_role("vault_admin"):
            return True

        logger.warning(f"User {actor_ctx.user_id} lacks permission to create {type} tokens")
        return False

class StrictRBACHook(DefaultRBACHook):
    """Strict RBAC implementation that denies by default."""

    def __init__(self):
        super().__init__(permissive=False)

# Global RBAC hook instance
_rbac_hook: Optional[RBACHook] = None

def set_rbac_hook(hook: RBACHook) -> None:
    """Set the global RBAC hook."""
    global _rbac_hook
    _rbac_hook = hook

def get_rbac_hook() -> RBACHook:
    """Get the current RBAC hook."""
    global _rbac_hook
    if _rbac_hook is None:
        # Default to permissive mode for development
        _rbac_hook = DefaultRBACHook(permissive=True)
    return _rbac_hook

def check_resolve_permission(token_id: str, tenant_id: Optional[str],
                           actor_ctx: Optional[ActorContext]) -> bool:
    """Check if actor can resolve a token."""
    return get_rbac_hook().check_resolve_permission(token_id, tenant_id, actor_ctx)

def check_create_permission(value: str, type: str, tenant_id: Optional[str],
                           actor_ctx: Optional[ActorContext]) -> bool:
    """Check if actor can create tokens."""
    return get_rbac_hook().check_create_permission(value, type, tenant_id, actor_ctx)
