"""
Embedding engine for the vectorizer module.
Handles text embedding generation using sentence-transformers.
"""
import logging
import time
from typing import List, Iterable, Optional, Union
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

from .settings import Settings


logger = logging.getLogger(__name__)


class Embedder:
    """
    Text embedding engine using sentence-transformers.

    Supports local and HuggingFace models with proper batching,
    error handling, and device management.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the embedder with configuration settings.

        Args:
            settings: Configuration settings for the embedder
        """
        self.settings = settings
        self.model: Optional[SentenceTransformer] = None
        self.device = self._get_device()

        logger.info(f"Initializing Embedder with model: {settings.embed_model}")
        logger.info(f"Using device: {self.device}")

        self._load_model()

    def _get_device(self) -> str:
        """
        Determine the best available device for the embedding model.

        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        requested_device = self.settings.device.lower()

        if requested_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif requested_device == "mps" and torch.backends.mps.is_available():
            return "mps"
        else:
            if requested_device != "cpu":
                logger.warning(f"Requested device '{requested_device}' not available, falling back to CPU")
            return "cpu"

    def _load_model(self) -> None:
        """Load the sentence-transformer model."""
        try:
            start_time = time.time()

            # Check if it's a local path
            model_path = Path(self.settings.embed_model)
            if model_path.exists() and model_path.is_dir():
                logger.info(f"Loading local model from: {model_path}")
                self.model = SentenceTransformer(str(model_path), device=self.device)
            else:
                logger.info(f"Loading HuggingFace model: {self.settings.embed_model}")
                self.model = SentenceTransformer(
                    self.settings.embed_model,
                    device=self.device
                )

            # Set model parameters
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.settings.max_seq_length

            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            logger.info(f"Model dimension: {self.get_embedding_dimension()}")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Could not initialize embedding model: {e}") from e

    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension of the loaded model.

        Returns:
            Embedding dimension size
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Get dimension from model configuration
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(
        self,
        texts: Iterable[str],
        show_progress: bool = True,
        normalize_embeddings: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: Iterable of text strings to embed
            show_progress: Whether to show progress bar
            normalize_embeddings: Whether to normalize embeddings to unit length

        Returns:
            List of embedding vectors (each as a list of floats)

        Raises:
            RuntimeError: If model is not loaded or embedding fails
        """
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")

        # Convert to list if needed
        text_list = list(texts) if not isinstance(texts, list) else texts

        if not text_list:
            logger.warning("No texts provided for embedding")
            return []

        logger.info(f"Generating embeddings for {len(text_list)} texts")

        try:
            embeddings = []
            batch_size = self.settings.batch_size

            # Process in batches with progress bar
            batches = [
                text_list[i:i + batch_size]
                for i in range(0, len(text_list), batch_size)
            ]

            if show_progress:
                batches = tqdm(batches, desc="Embedding batches")

            for batch in batches:
                try:
                    # Generate embeddings for the batch
                    batch_embeddings = self.model.encode(
                        batch,
                        convert_to_tensor=False,
                        normalize_embeddings=normalize_embeddings,
                        show_progress_bar=False
                    )

                    # Convert to list of lists
                    if isinstance(batch_embeddings, np.ndarray):
                        batch_embeddings = batch_embeddings.tolist()

                    embeddings.extend(batch_embeddings)

                except Exception as e:
                    logger.error(f"Failed to embed batch: {e}")
                    # Add zero vectors for failed batch
                    dim = self.get_embedding_dimension()
                    zero_embeddings = [[0.0] * dim] * len(batch)
                    embeddings.extend(zero_embeddings)

            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e

    def embed_single_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector as a list of floats
        """
        embeddings = self.embed_texts([text], show_progress=False)
        return embeddings[0] if embeddings else []

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score between -1 and 1
        """
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")

        embeddings = self.embed_texts([text1, text2], show_progress=False)
        if len(embeddings) < 2:
            return 0.0

        # Calculate cosine similarity
        from sentence_transformers.util import cos_sim
        similarity_score = cos_sim(embeddings[0], embeddings[1])

        return float(similarity_score)

    def __repr__(self) -> str:
        """String representation of the embedder."""
        model_name = self.settings.embed_model
        device = self.device
        dimension = self.get_embedding_dimension() if self.model else "unknown"
        return f"Embedder(model='{model_name}', device='{device}', dim={dimension})"
