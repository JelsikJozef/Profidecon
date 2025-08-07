#!/usr/bin/env python3
"""
Debug script to check Qdrant document IDs and help fix the evaluation
"""

import uuid
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import pandas as pd
from pathlib import Path

def check_qdrant_collection():
    """Check what's actually in the Qdrant collection"""
    print("🔍 Checking Qdrant collection...")

    try:
        client = QdrantClient(host="localhost", port=6333)
        collection_name = "profidecon_docs"

        # Get collection info
        collection_info = client.get_collection(collection_name)
        print(f"📊 Collection '{collection_name}' has {collection_info.points_count} points")

        # Get a few sample points to see the ID format
        points = client.scroll(
            collection_name=collection_name,
            limit=5,
            with_payload=True
        )

        print("\n📋 Sample documents in Qdrant:")
        print("-" * 80)

        for point in points[0]:
            hash_content = point.payload.get('hash_content', 'N/A')
            source = point.payload.get('source', 'N/A')
            text_preview = point.payload.get('text', '')[:100] + "..." if point.payload.get('text') else 'N/A'

            print(f"Qdrant ID: {point.id}")
            print(f"Hash Content: {hash_content}")
            print(f"Source: {source}")
            print(f"Text: {text_preview}")
            print("-" * 40)

            # Show conversion
            if hash_content != 'N/A':
                converted_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, hash_content))
                print(f"Hash → UUID: {converted_uuid}")
                print(f"Match: {'✅' if str(point.id) == converted_uuid else '❌'}")
            print("-" * 80)

        return True

    except Exception as e:
        print(f"❌ Error checking Qdrant: {e}")
        return False

def check_ground_truth():
    """Check the ground truth file format"""
    print("\n🎯 Checking ground truth file...")

    gt_file = "ground_truth.csv"
    if not Path(gt_file).exists():
        print(f"❌ Ground truth file not found: {gt_file}")
        return False

    try:
        df = pd.read_csv(gt_file)
        print(f"📊 Ground truth has {len(df)} entries")

        print("\n📋 Sample ground truth entries:")
        print("-" * 80)

        for i, row in df.head(5).iterrows():
            query = row['query']
            expected_id = row['expected_doc_id']

            print(f"Query: {query[:60]}...")
            print(f"Expected ID: {expected_id}")

            # Try to convert to UUID if it looks like a hash
            if len(str(expected_id)) == 40:  # SHA-1 hash length
                converted_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(expected_id)))
                print(f"Hash → UUID: {converted_uuid}")

            print("-" * 40)

        return True

    except Exception as e:
        print(f"❌ Error reading ground truth: {e}")
        return False

def test_search():
    """Test a simple search to see what gets returned"""
    print("\n🔍 Testing search functionality...")

    try:
        # Initialize components
        model = SentenceTransformer("intfloat/multilingual-e5-base")
        client = QdrantClient(host="localhost", port=6333)

        # Test query
        test_query = "vnútropodnikové vyslanie ICT"
        print(f"Test query: {test_query}")

        # Generate embedding
        embedding = model.encode([test_query], normalize_embeddings=True)[0]

        # Search
        hits = client.search(
            collection_name="profidecon_docs",
            query_vector=embedding.tolist(),
            limit=3,
            with_payload=True
        )

        print(f"\n📋 Search results for '{test_query}':")
        print("-" * 80)

        for i, hit in enumerate(hits, 1):
            hash_content = hit.payload.get('hash_content', 'N/A')
            text_preview = hit.payload.get('text', '')[:100] + "..." if hit.payload.get('text') else 'N/A'

            print(f"{i}. ID: {hit.id}")
            print(f"   Hash: {hash_content}")
            print(f"   Score: {hit.score:.4f}")
            print(f"   Text: {text_preview}")
            print("-" * 40)

        return True

    except Exception as e:
        print(f"❌ Error testing search: {e}")
        return False

def fix_ground_truth():
    """Create a corrected ground truth file with proper UUIDs"""
    print("\n🔧 Attempting to fix ground truth file...")

    gt_file = "ground_truth.csv"
    if not Path(gt_file).exists():
        print(f"❌ Ground truth file not found: {gt_file}")
        return False

    try:
        df = pd.read_csv(gt_file)

        # Convert hash_content to UUIDs
        df['original_expected_doc_id'] = df['expected_doc_id']
        df['expected_doc_id'] = df['expected_doc_id'].apply(
            lambda x: str(uuid.uuid5(uuid.NAMESPACE_DNS, str(x))) if len(str(x)) == 40 else x
        )

        # Save corrected version
        corrected_file = "ground_truth_corrected.csv"
        df[['query', 'expected_doc_id']].to_csv(corrected_file, index=False)

        print(f"✅ Created corrected ground truth file: {corrected_file}")
        print("📋 Sample conversions:")

        for i, row in df.head(3).iterrows():
            print(f"Original: {row['original_expected_doc_id']}")
            print(f"UUID:     {row['expected_doc_id']}")
            print("-" * 40)

        return True

    except Exception as e:
        print(f"❌ Error fixing ground truth: {e}")
        return False

def main():
    """Main debug function"""
    print("🐛 RAG Evaluation Debug Tool")
    print("=" * 50)

    # Check Qdrant collection
    qdrant_ok = check_qdrant_collection()

    # Check ground truth
    gt_ok = check_ground_truth()

    # Test search
    search_ok = test_search()

    # Try to fix ground truth
    if gt_ok:
        fix_ground_truth()

    print("\n📋 Summary:")
    print(f"Qdrant connection: {'✅' if qdrant_ok else '❌'}")
    print(f"Ground truth file: {'✅' if gt_ok else '❌'}")
    print(f"Search functionality: {'✅' if search_ok else '❌'}")

    if qdrant_ok and gt_ok and search_ok:
        print("\n💡 Recommendation:")
        print("Use the corrected ground truth file: tests/ground_truth_corrected.csv")
        print("Run evaluation with: python evaluate_rag.py")

if __name__ == "__main__":
    main()
