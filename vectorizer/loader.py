"""
Vector loader for uploading embeddings to Qdrant.
Handles JSONL file processing and vector database operations.
"""
import json
import logging
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
from tqdm import tqdm

from .settings import Settings
from .embedder import Embedder


logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for processing operations."""
    total_files: int = 0
    processed_files: int = 0
    total_documents: int = 0
    successful_uploads: int = 0
    failed_uploads: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        """Processing duration in seconds."""
        return self.end_time - self.start_time if self.end_time > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.successful_uploads / self.total_documents) * 100


class QdrantLoader:
    """
    Vector loader for uploading document embeddings to Qdrant.

    Handles JSONL file processing, embedding generation, and vector database operations
    with proper error handling, retries, and statistics tracking.
    """

    def __init__(self, settings: Settings, embedder: Embedder):
        """
        Initialize the Qdrant loader.

        Args:
            settings: Configuration settings
            embedder: Text embedder instance
        """
        self.settings = settings
        self.embedder = embedder
        self.client: Optional[QdrantClient] = None
        self.stats = ProcessingStats()

        logger.info(f"Initializing QdrantLoader for collection: {settings.qdrant_collection}")
        self._connect_to_qdrant()

    def _connect_to_qdrant(self) -> None:
        """Establish connection to Qdrant server."""
        try:
            self.client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                timeout=self.settings.qdrant_timeout
            )

            # Test connection
            collections = self.client.get_collections()
            logger.info(f"Connected to Qdrant at {self.settings.qdrant_url}")
            logger.info(f"Available collections: {[c.name for c in collections.collections]}")

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise RuntimeError(f"Could not connect to Qdrant at {self.settings.qdrant_url}: {e}") from e

    def _ensure_collection_exists(self) -> None:
        """Create collection if it doesn't exist with dual vector support (body + summary)."""
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        try:
            # Check if collection exists
            collections = self.client.get_collections()
            existing_collections = [c.name for c in collections.collections]

            if self.settings.qdrant_collection in existing_collections:
                logger.info(f"Collection '{self.settings.qdrant_collection}' already exists")
                return

            # Get embedding dimension
            embedding_dim = self.embedder.get_embedding_dimension()

            # Create collection with dual vector support
            self.client.create_collection(
                collection_name=self.settings.qdrant_collection,
                vectors_config={
                    "body": qdrant_models.VectorParams(
                        size=embedding_dim,
                        distance=qdrant_models.Distance.COSINE
                    ),
                    "summary": qdrant_models.VectorParams(
                        size=embedding_dim,
                        distance=qdrant_models.Distance.COSINE
                    )
                }
            )

            logger.info(f"Created collection '{self.settings.qdrant_collection}' with dual vectors (body + summary), dimension {embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise RuntimeError(f"Could not create collection '{self.settings.qdrant_collection}': {e}") from e

    def delete_collection(self) -> bool:
        """
        Delete the target collection.

        Returns:
            True if collection was deleted, False if it didn't exist
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        try:
            collections = self.client.get_collections()
            existing_collections = [c.name for c in collections.collections]

            if self.settings.qdrant_collection not in existing_collections:
                logger.info(f"Collection '{self.settings.qdrant_collection}' doesn't exist")
                return False

            self.client.delete_collection(collection_name=self.settings.qdrant_collection)
            logger.info(f"Deleted collection '{self.settings.qdrant_collection}'")
            return True

        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise RuntimeError(f"Could not delete collection '{self.settings.qdrant_collection}': {e}") from e

    def _load_jsonl_files(self, folder: Path, pattern: str) -> Iterator[Tuple[Path, Dict[str, Any]]]:
        """
        Load JSONL files from directory.

        Args:
            folder: Directory to search
            pattern: Glob pattern for files

        Yields:
            Tuple of (file_path, document_data)
        """
        jsonl_files = list(folder.glob(pattern))
        self.stats.total_files = len(jsonl_files)

        logger.info(f"Found {len(jsonl_files)} JSONL files matching pattern '{pattern}'")

        for file_path in tqdm(jsonl_files, desc="Processing files"):
            try:
                with file_path.open('r', encoding='utf-8') as f:
                    document_data = json.load(f)
                    yield file_path, document_data
                    self.stats.processed_files += 1

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in file {file_path}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                continue

    def _prepare_document_for_upload(self, file_path: Path, document_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Prepare a document for upload to Qdrant with dual vectors (body + summary).

        Args:
            file_path: Path to the source file
            document_data: Document data from JSONL

        Returns:
            Prepared document data or None if invalid
        """
        # Validate required fields
        if 'text' not in document_data:
            logger.warning(f"Document in {file_path} missing 'text' field")
            return None

        text = document_data['text']
        if not text or not text.strip():
            logger.warning(f"Document in {file_path} has empty text")
            return None

        # Get summary for dual vector embedding
        summary = document_data.get('summary', '')
        if not summary or not summary.strip():
            # Fallback to first 200 characters of text if no summary
            summary = text[:200] + "..." if len(text) > 200 else text

        # Prepare metadata payload
        payload = {
            'text': text,
            'source': str(file_path),
            'hash_content': document_data.get('hash_content', ''),
            'category': document_data.get('category', ''),
            'language': document_data.get('language', ''),
            'summary': summary,
            'tags': document_data.get('tags', []),
            'created_ts': document_data.get('created_ts', 0.0),
            'modified_ts': document_data.get('modified_ts', 0.0),
            'size_bytes': document_data.get('size_bytes', 0),
            'token_estimate': document_data.get('token_estimate', 0),
            'pii_risk': document_data.get('pii_risk', 0.0),
            'is_duplicate': document_data.get('is_duplicate', False),
        }

        # Generate UUID from hash_content or file path
        hash_content = document_data.get('hash_content')
        if hash_content:
            # Convert SHA-1 hash to UUID using uuid5
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, hash_content))
        else:
            # Fallback to file path-based UUID
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(file_path)))

        return {
            'id': doc_id,
            'text': text,
            'summary': summary,
            'payload': payload
        }

    def _upload_batch_with_retry(self, points_batch: List[qdrant_models.PointStruct]) -> int:
        """
        Upload a batch of points with retry logic.

        Args:
            points_batch: Batch of points to upload

        Returns:
            Number of successfully uploaded points
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        for attempt in range(self.settings.max_retries + 1):
            try:
                operation_info = self.client.upsert(
                    collection_name=self.settings.qdrant_collection,
                    points=points_batch
                )

                if operation_info.status == qdrant_models.UpdateStatus.COMPLETED:
                    return len(points_batch)
                else:
                    logger.warning(f"Upload attempt {attempt + 1} failed with status: {operation_info.status}")

            except Exception as e:
                logger.warning(f"Upload attempt {attempt + 1} failed: {e}")
                if attempt < self.settings.max_retries:
                    time.sleep(self.settings.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Failed to upload batch after {self.settings.max_retries} retries")

        return 0

    def load_directory(self, folder: Path, pattern: str = "**/*.jsonl") -> ProcessingStats:
        """
        Load JSONL files from directory and upload to Qdrant with dual vectors (body + summary).

        Args:
            folder: Directory containing JSONL files
            pattern: Glob pattern for files to process

        Returns:
            Processing statistics
        """
        logger.info(f"Starting vector loading from: {folder}")
        logger.info(f"File pattern: {pattern}")

        # Initialize stats
        self.stats = ProcessingStats()
        self.stats.start_time = time.time()

        try:
            # Ensure collection exists
            self._ensure_collection_exists()

            # Collect documents for batch processing
            documents = []
            texts_for_embedding = []
            summaries_for_embedding = []

            # Load and prepare documents
            for file_path, document_data in self._load_jsonl_files(folder, pattern):
                prepared_doc = self._prepare_document_for_upload(file_path, document_data)
                if prepared_doc:
                    documents.append(prepared_doc)
                    texts_for_embedding.append(prepared_doc['text'])
                    summaries_for_embedding.append(prepared_doc['summary'])
                    self.stats.total_documents += 1

            if not documents:
                logger.warning("No valid documents found to process")
                return self.stats

            logger.info(f"Generating dual embeddings for {len(documents)} documents...")

            # Generate embeddings for both text and summary
            logger.info("Generating body embeddings...")
            body_embeddings = self.embedder.embed_texts(texts_for_embedding, show_progress=True)

            logger.info("Generating summary embeddings...")
            summary_embeddings = self.embedder.embed_texts(summaries_for_embedding, show_progress=True)

            if len(body_embeddings) != len(documents) or len(summary_embeddings) != len(documents):
                logger.error(f"Embedding count mismatch: body={len(body_embeddings)}, summary={len(summary_embeddings)}, docs={len(documents)}")
                return self.stats

            # Upload to Qdrant in batches
            batch_size = self.settings.batch_size
            total_batches = (len(documents) + batch_size - 1) // batch_size

            logger.info(f"Uploading {len(documents)} documents with dual vectors in {total_batches} batches...")

            for i in tqdm(range(0, len(documents), batch_size), desc="Uploading batches"):
                batch_docs = documents[i:i + batch_size]
                batch_body_embeddings = body_embeddings[i:i + batch_size]
                batch_summary_embeddings = summary_embeddings[i:i + batch_size]

                # Create points for this batch with dual vectors
                points_batch = []
                for doc, body_emb, summary_emb in zip(batch_docs, batch_body_embeddings, batch_summary_embeddings):
                    point = qdrant_models.PointStruct(
                        id=doc['id'],
                        vector={
                            "body": body_emb,
                            "summary": summary_emb
                        },
                        payload=doc['payload']
                    )
                    points_batch.append(point)

                # Upload batch with retries
                uploaded_count = self._upload_batch_with_retry(points_batch)
                self.stats.successful_uploads += uploaded_count
                self.stats.failed_uploads += len(points_batch) - uploaded_count

            self.stats.end_time = time.time()

            # Log final statistics
            logger.info("=== Dual Vector Loading Complete ===")
            logger.info(f"Total files processed: {self.stats.processed_files}/{self.stats.total_files}")
            logger.info(f"Total documents: {self.stats.total_documents}")
            logger.info(f"Successful uploads: {self.stats.successful_uploads}")
            logger.info(f"Failed uploads: {self.stats.failed_uploads}")
            logger.info(f"Success rate: {self.stats.success_rate:.1f}%")
            logger.info(f"Processing time: {self.stats.duration:.2f} seconds")

            return self.stats

        except Exception as e:
            self.stats.end_time = time.time()
            logger.error(f"Vector loading failed: {e}")
            raise RuntimeError(f"Failed to load vectors: {e}") from e

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Collection information dictionary
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        try:
            collection_info = self.client.get_collection(self.settings.qdrant_collection)
            return {
                'name': collection_info.config.params.vectors.size,
                'dimension': collection_info.config.params.vectors.distance,
                'distance_metric': collection_info.config.params.vectors.distance,
                'points_count': collection_info.points_count,
                'status': collection_info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}


# Convenience function for the main CLI
def load_folder(folder: Path, glob_pattern: str = "*.jsonl") -> ProcessingStats:
    """
    Convenience function to load vectors from a folder.

    Args:
        folder: Directory containing JSONL files
        glob_pattern: Glob pattern for files

    Returns:
        Processing statistics
    """
    from .settings import settings

    # Initialize components
    embedder = Embedder(settings)
    loader = QdrantLoader(settings, embedder)

    # Load vectors
    return loader.load_directory(folder, glob_pattern)
