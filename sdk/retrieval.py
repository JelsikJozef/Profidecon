"""
Retrieval engine for Profidecon with hybrid search capabilities.
Supports dual vector search (body + summary) and tag-based score boosting.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint, SearchParams
from sentence_transformers import SentenceTransformer

from vectorizer.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    Enhanced search result with metadata and boosting information.

    Attributes:
        id: Document ID
        score: Final similarity score (potentially boosted)
        original_score: Original vector similarity score
        text: Document text content
        summary: Document summary
        tags: Document tags
        category: Document category
        source: Document source
        was_boosted: Whether tag boosting was applied
        boost_factor: The boost factor applied (1.0 = no boost)
        matched_tags: Tags that matched user query tags
    """
    id: str
    score: float
    original_score: float
    text: str
    summary: str
    tags: List[str]
    category: str
    source: str
    was_boosted: bool = False
    boost_factor: float = 1.0
    matched_tags: List[str] = None

    def __post_init__(self):
        if self.matched_tags is None:
            self.matched_tags = []


class RetrievalEngine:
    """
    Main retrieval engine with hybrid search capabilities.

    Supports:
    - Dual vector search (body and summary embeddings)
    - Tag-based score boosting
    - Configurable search parameters
    """

    def __init__(self, settings: Settings = None):
        """
        Initialize the retrieval engine.

        Args:
            settings: Vectorizer settings (uses global settings if None)
        """
        from vectorizer.settings import settings as default_settings
        self.settings = settings or default_settings
        self.client: Optional[QdrantClient] = None
        self.embedding_model: Optional[SentenceTransformer] = None

        self._connect_to_qdrant()
        self._load_embedding_model()

    def _connect_to_qdrant(self) -> None:
        """Connect to Qdrant instance."""
        try:
            self.client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                timeout=self.settings.qdrant_timeout
            )

            # Test connection
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.settings.qdrant_collection not in collection_names:
                raise RuntimeError(f"Collection '{self.settings.qdrant_collection}' not found")

            logger.info(f"Connected to Qdrant at {self.settings.qdrant_url}")
            logger.info(f"Using collection: {self.settings.qdrant_collection}")

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise RuntimeError(f"Could not connect to Qdrant: {e}") from e

    def _load_embedding_model(self) -> None:
        """Load the embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.settings.embed_model}")
            self.embedding_model = SentenceTransformer(self.settings.embed_model)
            logger.info("Embedding model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Could not load embedding model: {e}") from e

    def _embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.

        Args:
            query: Query text

        Returns:
            Query embedding vector
        """
        if not self.embedding_model:
            raise RuntimeError("Embedding model not loaded")

        try:
            embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise RuntimeError(f"Could not embed query: {e}") from e

    def _vector_search(
        self,
        query_vector: List[float],
        limit: int = None,
        use_summary_vector: bool = None,
        search_params: SearchParams = None
    ) -> List[ScoredPoint]:
        """
        Perform vector similarity search.

        Args:
            query_vector: Query embedding vector
            limit: Number of results to return (uses settings default if None)
            use_summary_vector: Whether to use summary vector (uses settings default if None)
            search_params: Qdrant search parameters

        Returns:
            List of scored points from Qdrant
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        try:
            # Use provided values or defaults from settings
            limit = limit or self.settings.search_limit
            use_summary = use_summary_vector if use_summary_vector is not None else self.settings.use_summary_vector
            search_params = search_params or SearchParams(hnsw_ef=128)

            # Choose vector name based on configuration
            vector_name = "summary" if use_summary else "body"

            # Perform search with named vector
            hits = self.client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=(vector_name, query_vector),  # Named vector format
                limit=limit,
                search_params=search_params,
                with_payload=True
            )

            logger.debug(f"Vector search found {len(hits)} results using '{vector_name}' vector")
            return hits

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise RuntimeError(f"Vector search failed: {e}") from e

    def _apply_tag_boosting(
        self,
        hits: List[ScoredPoint],
        user_tags: List[str],
        tag_boost: float = None
    ) -> List[SearchResult]:
        """
        Apply tag-based score boosting and convert to SearchResult objects.

        Args:
            hits: Raw search results from Qdrant
            user_tags: User-provided tags for boosting
            tag_boost: Boost factor (uses settings default if None)

        Returns:
            List of enhanced search results with potential boosting applied
        """
        tag_boost = tag_boost if tag_boost is not None else self.settings.tag_boost
        results = []

        for hit in hits:
            # Extract metadata from payload
            payload = hit.payload or {}
            doc_tags = payload.get('tags', [])

            # Find matching tags
            matched_tags = []
            if user_tags and doc_tags:
                matched_tags = [tag for tag in user_tags if tag in doc_tags]

            # Calculate boost
            was_boosted = False
            boost_factor = 1.0
            final_score = hit.score

            if matched_tags and tag_boost > 0:
                boost_factor = 1.0 + tag_boost
                final_score = hit.score * boost_factor
                was_boosted = True

            # Create enhanced result
            result = SearchResult(
                id=str(hit.id),
                score=final_score,
                original_score=hit.score,
                text=payload.get('text', ''),
                summary=payload.get('summary', ''),
                tags=doc_tags,
                category=payload.get('category', ''),
                source=payload.get('source', ''),
                was_boosted=was_boosted,
                boost_factor=boost_factor,
                matched_tags=matched_tags
            )
            results.append(result)

        # Re-sort by boosted scores if boosting was applied
        if any(r.was_boosted for r in results):
            results.sort(key=lambda x: x.score, reverse=True)
            logger.debug(f"Applied tag boosting to {sum(1 for r in results if r.was_boosted)} documents")

        return results

    def search(
        self,
        query: str,
        user_tags: List[str] = None,
        limit: int = None,
        tag_boost: float = None,
        use_summary_vector: bool = None,
        search_params: SearchParams = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search with optional tag boosting.

        Args:
            query: Search query text
            user_tags: List of tags for boosting (optional)
            limit: Number of results to return (uses settings default if None)
            tag_boost: Boost factor (uses settings default if None)
            use_summary_vector: Whether to use summary vector (uses settings default if None)
            search_params: Qdrant search parameters

        Returns:
            List of search results, potentially boosted and re-ranked
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        user_tags = user_tags or []

        logger.info(f"Searching for: '{query}' with tags: {user_tags}")

        try:
            # 1. Generate query embedding
            query_vector = self._embed_query(query)

            # 2. Perform vector search
            hits = self._vector_search(query_vector, limit, use_summary_vector, search_params)

            if not hits:
                logger.info("No results found")
                return []

            # 3. Apply tag boosting and create enhanced results
            results = self._apply_tag_boosting(hits, user_tags, tag_boost)

            # Log search summary
            boosted_count = sum(1 for r in results if r.was_boosted)
            logger.info(f"Search complete: {len(results)} results, {boosted_count} boosted")

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Search failed: {e}") from e

    def search_summary_vector(
        self,
        query: str,
        user_tags: List[str] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Convenience method to search using summary vectors.

        Args:
            query: Search query text
            user_tags: List of tags for boosting (optional)
            **kwargs: Additional search parameters

        Returns:
            List of search results using summary vector search
        """
        return self.search(query, user_tags, use_summary_vector=True, **kwargs)

    def get_document_by_id(self, doc_id: str) -> Optional[SearchResult]:
        """
        Retrieve a specific document by ID.

        Args:
            doc_id: Document ID

        Returns:
            SearchResult object or None if not found
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        try:
            points = self.client.retrieve(
                collection_name=self.settings.qdrant_collection,
                ids=[doc_id],
                with_payload=True
            )

            if not points:
                return None

            point = points[0]
            payload = point.payload or {}

            return SearchResult(
                id=str(point.id),
                score=1.0,  # No similarity score for direct retrieval
                original_score=1.0,
                text=payload.get('text', ''),
                summary=payload.get('summary', ''),
                tags=payload.get('tags', []),
                category=payload.get('category', ''),
                source=payload.get('source', ''),
                was_boosted=False,
                boost_factor=1.0,
                matched_tags=[]
            )

        except Exception as e:
            logger.error(f"Failed to retrieve document {doc_id}: {e}")
            return None

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        try:
            collection_info = self.client.get_collection(self.settings.qdrant_collection)

            return {
                'collection_name': self.settings.qdrant_collection,
                'points_count': collection_info.points_count,
                'vectors_config': collection_info.config.params.vectors,
                'status': collection_info.status
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
