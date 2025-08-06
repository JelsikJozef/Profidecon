#!/usr/bin/env python3
"""
RAG Pipeline Evaluation Script

This script evaluates the retrieval quality of a RAG pipeline using Qdrant and SentenceTransformers.
It loads ground truth queries, retrieves documents, and computes retrieval metrics.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings

# Suppress transformer warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import SearchParams
except ImportError as e:
    print(f"[ERROR] Missing required package: {e}")
    print("Please install: pip install sentence-transformers qdrant-client pandas numpy")
    sys.exit(1)


class RAGEvaluator:
    """Evaluates RAG retrieval quality against ground truth data."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        """
        Initialize the RAG evaluator.

        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.client = None
        self.collection_name = "profidecon_docs"

    def setup_environment(self) -> None:
        """Initialize embedding model and Qdrant client."""
        try:
            print("🔧 Setting up environment...")

            # Initialize embedding model
            print(f"📊 Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Model loaded successfully (dimension: {self.model.get_sentence_embedding_dimension()})")

            # Initialize Qdrant client
            print("🔍 Connecting to Qdrant...")
            self.client = QdrantClient(host="localhost", port=6333)

            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                print(f"[ERROR] Collection '{self.collection_name}' not found in Qdrant")
                print(f"Available collections: {collection_names}")
                sys.exit(1)

            print(f"✅ Connected to Qdrant, collection '{self.collection_name}' found")

        except Exception as e:
            print(f"[ERROR] Failed to setup environment: {e}")
            sys.exit(1)

    def load_ground_truth(self, file_path: str = "tests/ground_truth_corrected.csv") -> pd.DataFrame:
        """
        Load ground truth data from CSV file.

        Args:
            file_path: Path to the ground truth CSV file

        Returns:
            DataFrame with ground truth queries and expected document IDs
        """
        try:
            print(f"📂 Loading ground truth from: {file_path}")

            if not Path(file_path).exists():
                print(f"[ERROR] Ground truth file not found: {file_path}")
                print("Expected format: CSV with columns 'query' and 'expected_doc_id'")
                sys.exit(1)

            df = pd.read_csv(file_path)

            # Validate required columns
            required_columns = ['query', 'expected_doc_id']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"[ERROR] Missing required columns in ground truth file: {missing_columns}")
                print(f"Found columns: {list(df.columns)}")
                print("Expected columns: query, expected_doc_id")
                sys.exit(1)

            # Remove rows with missing values
            initial_count = len(df)
            df = df.dropna(subset=required_columns)
            final_count = len(df)

            if final_count < initial_count:
                print(f"⚠️  Removed {initial_count - final_count} rows with missing values")

            if final_count == 0:
                print("[ERROR] No valid queries found in ground truth file")
                sys.exit(1)

            print(f"✅ Loaded {final_count} ground truth queries")
            return df

        except Exception as e:
            print(f"[ERROR] Failed to load ground truth data: {e}")
            sys.exit(1)

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Generate embeddings for queries.

        Args:
            queries: List of query strings

        Returns:
            Array of query embeddings
        """
        try:
            print(f"🔮 Generating embeddings for {len(queries)} queries...")

            embeddings = self.model.encode(
                queries,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_tensor=False
            )

            print(f"✅ Generated {len(embeddings)} query embeddings")
            return embeddings

        except Exception as e:
            print(f"[ERROR] Failed to generate embeddings: {e}")
            sys.exit(1)

    def retrieve_documents(
        self,
        query_embedding: List[float],
        limit: int = 5
    ) -> List[models.ScoredPoint]:
        """
        Retrieve documents for a single query embedding.

        Args:
            query_embedding: Query embedding vector
            limit: Number of documents to retrieve

        Returns:
            List of retrieved documents with scores
        """
        try:
            # Use the newer query_points method instead of deprecated search
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=limit,
                with_payload=True
            )
            return hits.points if hasattr(hits, 'points') else hits

        except Exception as e:
            # Fallback to basic search without params if query_points fails
            try:
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=limit
                )
                return hits
            except Exception as e2:
                print(f"[ERROR] Failed to retrieve documents: {e2}")
                return []

    def evaluate_retrieval(self, ground_truth: pd.DataFrame) -> Dict:
        """
        Evaluate retrieval quality against ground truth.

        Args:
            ground_truth: DataFrame with queries and expected document IDs

        Returns:
            Dictionary with evaluation results
        """
        try:
            print("🎯 Evaluating retrieval quality...")

            queries = ground_truth['query'].tolist()
            expected_ids = ground_truth['expected_doc_id'].tolist()

            # Generate embeddings for all queries
            query_embeddings = self.embed_queries(queries)

            results = []
            found_count = 0

            print("\n🔍 Retrieving documents for each query...")

            for i, (query, expected_id, embedding) in enumerate(zip(queries, expected_ids, query_embeddings)):
                print(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")

                # Retrieve documents
                hits = self.retrieve_documents(embedding.tolist(), limit=5)

                # Extract retrieved document IDs
                retrieved_ids = [hit.id for hit in hits]

                # Check if expected document is in top-5
                found = False
                rank = "n/a"

                if str(expected_id) in [str(rid) for rid in retrieved_ids]:
                    found = True
                    found_count += 1
                    # Find rank (1-indexed)
                    rank = next(
                        (idx + 1 for idx, rid in enumerate(retrieved_ids)
                         if str(rid) == str(expected_id)),
                        "n/a"
                    )

                results.append({
                    'query': query,
                    'expected_doc_id': expected_id,
                    'found': 'yes' if found else 'no',
                    'rank': rank,
                    'retrieved_ids': retrieved_ids,
                    'scores': [hit.score for hit in hits] if hits else []
                })

            # Calculate metrics
            total_queries = len(queries)
            recall_at_5 = found_count / total_queries if total_queries > 0 else 0.0

            evaluation_results = {
                'results': results,
                'total_queries': total_queries,
                'found_count': found_count,
                'recall_at_5': recall_at_5,
                'not_found_queries': [r['query'] for r in results if r['found'] == 'no']
            }

            print(f"✅ Evaluation complete: {found_count}/{total_queries} queries found")
            return evaluation_results

        except Exception as e:
            print(f"[ERROR] Failed to evaluate retrieval: {e}")
            sys.exit(1)

    def print_results(self, evaluation_results: Dict) -> None:
        """
        Print evaluation results in a formatted table.

        Args:
            evaluation_results: Dictionary with evaluation results
        """
        print("\n" + "="*80)
        print("📊 RAG RETRIEVAL EVALUATION RESULTS")
        print("="*80)

        # Print markdown table
        print("\n| Query | Expected Doc ID | Found | Rank |")
        print("|-------|----------------|-------|------|")

        for result in evaluation_results['results']:
            query_short = result['query'][:40] + "..." if len(result['query']) > 40 else result['query']
            expected_id_short = str(result['expected_doc_id'])[:20]
            print(f"| {query_short} | {expected_id_short} | {result['found']} | {result['rank']} |")

        print(f"\n📈 **Recall@5: {evaluation_results['recall_at_5']:.3f}**")
        print(f"📊 Found: {evaluation_results['found_count']}/{evaluation_results['total_queries']} queries")

        # List queries not found
        if evaluation_results['not_found_queries']:
            print(f"\n❌ **Queries not found in top-5:**")
            for i, query in enumerate(evaluation_results['not_found_queries'], 1):
                query_short = query[:60] + "..." if len(query) > 60 else query
                print(f"{i}. {query_short}")
        else:
            print("\n✅ **All queries found in top-5 results!**")

        print("\n" + "="*80)


def main():
    """Main function to run the RAG evaluation."""
    print("🚀 Starting RAG Pipeline Evaluation")
    print("="*50)

    # Initialize evaluator
    evaluator = RAGEvaluator()

    # Setup environment
    evaluator.setup_environment()

    # Load ground truth
    ground_truth = evaluator.load_ground_truth()

    # Evaluate retrieval
    results = evaluator.evaluate_retrieval(ground_truth)

    # Print results
    evaluator.print_results(results)

    print("\n🏁 Evaluation completed successfully!")


if __name__ == "__main__":
    main()
