"""
PostgreSQL Token Vault Implementation

Provides PostgreSQL-backed storage for tokens with AES-GCM encryption,
deterministic token generation, and multi-tenant support.

Note: For testing, this can also work with SQLite by adapting the SQL syntax.
"""

import os
import time
import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse

from .models import Token, VaultRecord, Scope, TokenStats
from .crypto import get_crypto_manager
from .rbac import ActorContext, check_resolve_permission, check_create_permission
from .errors import (
    VaultError, NotFoundError, PermissionError, IntegrityError,
    ConfigurationError, EncryptionError
)

logger = logging.getLogger(__name__)

# Global cache for SQLite connections keyed by db_url
_SQLITE_CONNS: dict[str, Any] = {}

class PostgresTokenVault:
    """PostgreSQL implementation of TokenVault."""

    def __init__(self, actor_ctx: Optional[ActorContext] = None):
        """
        Initialize PostgreSQL token vault.

        Args:
            actor_ctx: Actor context for RBAC
        """
        self.actor_ctx = actor_ctx
        self.crypto = get_crypto_manager()
        self.db_url = os.getenv("DATABASE_URL", "sqlite:///:memory:")

        # Parse default scope
        self.default_scope = os.getenv("PSEUDO_SCOPE", "tenant")
        if self.default_scope not in ["tenant", "global"]:
            raise ConfigurationError(f"Invalid PSEUDO_SCOPE: {self.default_scope}")

        self._connection = None
        self._ensure_schema()

    def _get_connection(self):
        """Get database connection."""
        try:
            # For testing with SQLite
            if self.db_url.startswith("sqlite://"):
                import sqlite3
                # Normalize key; support in-memory shared db
                key = self.db_url
                conn = _SQLITE_CONNS.get(key)
                if conn is None:
                    # Use a shared in-memory database if :memory: requested
                    if self.db_url.endswith(":memory:"):
                        uri = "file:token_vault_shared?mode=memory&cache=shared"
                        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
                    else:
                        # sqlite:///path or other forms
                        path = self.db_url.replace("sqlite://", "", 1)
                        if not path or path == "/":
                            # Fallback to shared memory
                            uri = "file:token_vault_shared?mode=memory&cache=shared"
                            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
                        else:
                            # Strip leading slash for absolute paths if needed
                            db_path = path
                            conn = sqlite3.connect(db_path, check_same_thread=False)
                    conn.row_factory = sqlite3.Row
                    _SQLITE_CONNS[key] = conn
                return conn
            else:
                # PostgreSQL
                import psycopg2
                from psycopg2.extras import RealDictCursor
                return psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)
        except Exception as e:
            raise VaultError(f"Failed to connect to database: {e}")

    def _ensure_schema(self):
        """Ensure database schema exists."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Create vault_records table (SQLite compatible)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS vault_records (
                    token_id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    tenant_id TEXT,
                    encrypted_data_key BLOB NOT NULL,
                    cipher_value BLOB NOT NULL,
                    nonce BLOB NOT NULL,
                    tag BLOB NOT NULL,
                    salted_hash BLOB NOT NULL,
                    created_at REAL NOT NULL
                )
            """)

            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_vault_records_salted_hash 
                ON vault_records(salted_hash)
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_vault_records_tenant_scope 
                ON vault_records(tenant_id, scope)
            """)

            conn.commit()
            logger.debug("Database schema initialized successfully")

        except Exception as e:
            raise VaultError(f"Failed to initialize database schema: {e}")

    def get_or_create(self, *, value: str, type: str, scope: Scope,
                      tenant_id: Optional[str] = None) -> Token:
        """
        Deterministically map a value/type to a token within scope.
        """
        # Check create permissions
        if not check_create_permission(value, type, tenant_id, self.actor_ctx):
            raise PermissionError(f"Access denied for creating {type} token")

        # Normalize value for deterministic processing
        normalized_value = self.crypto.normalize(value, type)

        # Generate deterministic token ID
        token_id = self.crypto.generate_token_id(normalized_value, type, scope, tenant_id)

        # Create display token
        display = f"[{type}:{token_id}]"

        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Try to find existing token first
            cur.execute("""
                SELECT token_id, type FROM vault_records 
                WHERE token_id = ? AND scope = ? AND (tenant_id = ? OR (tenant_id IS NULL AND ? IS NULL))
            """, (token_id, scope, tenant_id, tenant_id))

            existing = cur.fetchone()
            if existing:
                # Token already exists
                logger.debug(f"Returning existing token {token_id}")
                logging.getLogger("vault.audit").warning("vault_audit op=create decision=allow token_type=%s token_id=%s tenant_id=%s", type, token_id, tenant_id)
                return Token(id=token_id, type=type, display=display)

            # Create new token
            encrypted_data_key, cipher_value, nonce, tag = self.crypto.aesgcm_encrypt(normalized_value)
            salted_hash = self.crypto.salted_hash(normalized_value)

            # Insert new token
            cur.execute("""
                INSERT OR IGNORE INTO vault_records 
                (token_id, type, scope, tenant_id, encrypted_data_key, cipher_value, nonce, tag, salted_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (token_id, type, scope, tenant_id, encrypted_data_key, cipher_value, nonce, tag, salted_hash, time.time()))

            conn.commit()

            if cur.rowcount > 0:
                logger.info(f"Created new token {token_id} for type {type}")
            else:
                logger.debug(f"Token {token_id} already existed (race condition)")

            logging.getLogger("vault.audit").warning("vault_audit op=create decision=allow token_type=%s token_id=%s tenant_id=%s", type, token_id, tenant_id)
            return Token(id=token_id, type=type, display=display)

        except Exception as e:
            logger.error(f"Failed to create/retrieve token: {e}")
            logging.getLogger("vault.audit").warning("vault_audit op=create decision=deny error=%s token_type=%s tenant_id=%s", type(e).__name__, type, tenant_id)
            raise VaultError(f"Token operation failed: {e}")

    def resolve(self, *, token_id: str, scope: Scope,
                tenant_id: Optional[str] = None) -> str:
        """
        Return plaintext value (subject to RBAC).
        """
        # Check resolve permissions
        if not check_resolve_permission(token_id, tenant_id, self.actor_ctx):
            logging.getLogger("vault.audit").warning("vault_audit op=resolve decision=deny token_id=%s tenant_id=%s", token_id, tenant_id)
            raise PermissionError(f"Access denied for resolving token {token_id}")

        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                SELECT encrypted_data_key, cipher_value, nonce, tag, type 
                FROM vault_records 
                WHERE token_id = ? AND scope = ? AND (tenant_id = ? OR (tenant_id IS NULL AND ? IS NULL))
            """, (token_id, scope, tenant_id, tenant_id))

            record = cur.fetchone()
            if not record:
                logging.getLogger("vault.audit").warning("vault_audit op=resolve decision=deny error=NotFound token_id=%s tenant_id=%s", token_id, tenant_id)
                raise NotFoundError(f"Token {token_id} not found in scope {scope}")

            # Decrypt the value
            try:
                plaintext = self.crypto.aesgcm_decrypt(
                    record['encrypted_data_key'],
                    record['cipher_value'],
                    record['nonce'],
                    record['tag']
                )

                logger.debug(f"Successfully resolved token {token_id}")
                logging.getLogger("vault.audit").warning("vault_audit op=resolve decision=allow token_type=%s token_id=%s tenant_id=%s", record['type'], token_id, tenant_id)
                return plaintext

            except Exception as e:
                logging.getLogger("vault.audit").warning("vault_audit op=resolve decision=deny error=Integrity token_id=%s tenant_id=%s", token_id, tenant_id)
                raise IntegrityError(f"Failed to decrypt token {token_id}: {e}")

        except (NotFoundError, PermissionError, IntegrityError):
            raise
        except Exception as e:
            logger.error(f"Failed to resolve token {token_id}: {e}")
            logging.getLogger("vault.audit").warning("vault_audit op=resolve decision=deny error=Vault token_id=%s tenant_id=%s", token_id, tenant_id)
            raise VaultError(f"Token resolution failed: {e}")

    def exists(self, *, token_id: str, scope: Scope) -> bool:
        """
        Check if token exists in vault.
        """
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                SELECT 1 FROM vault_records 
                WHERE token_id = ? AND scope = ?
            """, (token_id, scope))

            return cur.fetchone() is not None

        except Exception as e:
            logger.error(f"Failed to check token existence: {e}")
            return False

    def stats(self) -> TokenStats:
        """
        Get vault statistics.
        """
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Total tokens
            cur.execute("SELECT COUNT(*) as total FROM vault_records")
            total = cur.fetchone()['total']

            # Tokens by type
            cur.execute("""
                SELECT type, COUNT(*) as count 
                FROM vault_records 
                GROUP BY type
            """)
            tokens_by_type = {row['type']: row['count'] for row in cur.fetchall()}

            # Tokens by scope
            cur.execute("""
                SELECT scope, COUNT(*) as count 
                FROM vault_records 
                GROUP BY scope
            """)
            tokens_by_scope = {row['scope']: row['count'] for row in cur.fetchall()}

            return TokenStats(
                total_tokens=total,
                tokens_by_type=tokens_by_type,
                tokens_by_scope=tokens_by_scope,
                creation_rate_24h=0.0,  # Simplified for testing
                resolution_rate_24h=0.0
            )

        except Exception as e:
            logger.error(f"Failed to get vault statistics: {e}")
            return TokenStats(
                total_tokens=0,
                tokens_by_type={},
                tokens_by_scope={},
                creation_rate_24h=0.0,
                resolution_rate_24h=0.0
            )
