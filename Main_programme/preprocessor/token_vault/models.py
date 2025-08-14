"""
Token Vault Models

Defines the core data structures for tokenization including tokens,
vault records, scopes, and type definitions.
"""

from typing import TypedDict, Literal, Optional
from enum import Enum

# Type aliases
Scope = Literal["tenant", "global"]

class TokenType(Enum):
    """Supported PII token types."""
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    PERSON_NAME = "PERSON_NAME"
    IBAN = "IBAN"
    PASSPORT = "PASSPORT"
    ID_NUMBER = "ID_NUMBER"
    ADDRESS = "ADDRESS"
    CREDIT_CARD = "CREDIT_CARD"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    URL = "URL"
    ORG = "ORG"

class Token(TypedDict):
    """Public token representation returned to clients."""
    id: str          # stable ID, e.g. base32 of truncated HMAC
    type: str        # "EMAIL", "PHONE", "PERSON_NAME", ...
    display: str     # e.g. "[EMAIL:K7V2WQ3M]"

class VaultRecord(TypedDict):
    """Internal vault record structure for database storage."""
    token_id: str
    type: str
    scope: Scope
    tenant_id: Optional[str]
    # No plaintext PII at rest:
    cipher_value: bytes     # AES-GCM(envelope(key), plaintext)
    nonce: bytes
    tag: bytes
    salted_hash: bytes      # fast lookup; SHA-256(salt || normalized_value)
    created_at: float

class TokenStats(TypedDict):
    """Token vault statistics."""
    total_tokens: int
    tokens_by_type: dict[str, int]
    tokens_by_scope: dict[str, int]
    creation_rate_24h: float
    resolution_rate_24h: float
