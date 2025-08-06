from qdrant_client import QdrantClient, models
import settings, hashlib

client = QdrantClient(url=settings.QDRANT_URL)

def ensure_collection(dim: int):
    """ Ensures that the specified collection exists in Qdrant.
    :param dim: Dimension of the vectors to be stored in the collection."""
    if settings.COLLECTION not in client.get_collections().collections:
        client.create_collection(
            collection_name=settings.COLLECTION,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE
            )
        )
def upsert(chunk_id: str, vector: list[float], payload: dict):
    """ Upserts a point into the Qdrant collection.
    :param chunk_id: Unique identifier for the chunk.
    :param vector: Dense vector representation of the chunk.
    :param payload: Additional metadata to store with the chunk."""
    client.upsert(
        collection_name=settings.COLLECTION,
        points=[models.PointStruct(
            id=chunk_id,
            vector=vector,
            payload=payload
        )]
    )
