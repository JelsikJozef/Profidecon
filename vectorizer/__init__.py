"""
Vectorizer package for Profidecon.

Provides text embedding and vector database functionality for RAG systems.
"""
from .settings import Settings, settings
from .embedder import Embedder
from .loader import QdrantLoader, ProcessingStats, load_folder

__version__ = "0.1.0"
__all__ = [
    "Settings",
    "settings",
    "Embedder",
    "QdrantLoader",
    "ProcessingStats",
    "load_folder"
]
