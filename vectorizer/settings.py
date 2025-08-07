"""
Configuration settings for the vectorizer module.
Uses Pydantic for validation and environment variable loading.
"""
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import os


class Settings(BaseSettings):
    """
    Configuration settings for the vectorizer package.
    Automatically loads from environment variables with VECTORIZER_ prefix.
    """

    # ──────────────────────────────────────────────
    # Embedding model configuration
    # ──────────────────────────────────────────────
    embed_model: str = Field(
        default="intfloat/multilingual-e5-base",
        description="HuggingFace model name or local path for embeddings"
    )
    device: str = Field(
        default="cpu",
        description="Device for embedding model: 'cpu', 'cuda', or 'mps'"
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation",
        ge=1,
        le=512
    )
    max_seq_length: int = Field(
        default=512,
        description="Maximum sequence length for the embedding model"
    )

    # ──────────────────────────────────────────────
    # Text chunking configuration
    # ──────────────────────────────────────────────
    chunk_size: int = Field(
        default=450,
        description="Maximum number of words per chunk",
        ge=50,
        le=2000
    )
    chunk_overlap: int = Field(
        default=100,
        description="Word overlap between consecutive chunks",
        ge=0
    )

    # ──────────────────────────────────────────────
    # Qdrant configuration
    # ──────────────────────────────────────────────
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL"
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API key for authentication"
    )
    qdrant_collection: str = Field(
        default="profidecon_docs",
        description="Qdrant collection name"
    )
    qdrant_timeout: float = Field(
        default=30.0,
        description="Qdrant client timeout in seconds"
    )

    # ──────────────────────────────────────────────
    # Processing configuration
    # ──────────────────────────────────────────────
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed operations",
        ge=0,
        le=10
    )
    retry_delay: float = Field(
        default=1.0,
        description="Delay between retries in seconds",
        ge=0.1
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for taxonomy and embedding"
    )

    # ──────────────────────────────────────────────
    # Retrieval and search configuration
    # ──────────────────────────────────────────────
    tag_boost: float = Field(
        default=0.20,
        description="Tag boost factor for hybrid search (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    search_limit: int = Field(
        default=10,
        description="Default number of search results to return",
        ge=1,
        le=100
    )
    use_summary_vector: bool = Field(
        default=False,
        description="Whether to search summary vectors by default"
    )

    @validator('device')
    def validate_device(cls, v):
        allowed_devices = ['cpu', 'cuda', 'mps']
        if v not in allowed_devices:
            raise ValueError(f"Device must be one of {allowed_devices}")
        return v

    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, values):
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v

    @validator('log_level')
    def validate_log_level(cls, v):
        allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of {allowed_levels}")
        return v.upper()

    class Config:
        env_prefix = "VECTORIZER_"
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

# Legacy constants for backward compatibility
EMBED_MODEL = settings.embed_model
DEVICE = settings.device
BATCH_SIZE = settings.batch_size
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
QDRANT_HOST = settings.qdrant_url.split("://")[1].split(":")[0] if "://" in settings.qdrant_url else settings.qdrant_url
QDRANT_PORT = int(settings.qdrant_url.split(":")[-1]) if ":" in settings.qdrant_url else 6333
QDRANT_COLLECTION = settings.qdrant_collection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
