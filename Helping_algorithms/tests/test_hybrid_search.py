#!/usr/bin/env python3
"""
Test script for the new hybrid retrieval system with dual vectors and tag boosting.
"""

import logging
from Main_programme.sdk.retrieval import RetrievalEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hybrid_search():
    """Test the hybrid search functionality."""
    print("🚀 Testing Hybrid Search with Dual Vectors & Tag Boosting")
    print("=" * 60)

    # Initialize retrieval engine
    print("🔧 Initializing retrieval engine...")
    engine = RetrievalEngine()

    # Get collection stats
    stats = engine.get_collection_stats()
    print(f"📊 Collection: {stats.get('collection_name')}")
    print(f"📈 Documents: {stats.get('points_count')}")
    print(f"🎯 Vectors: {stats.get('vectors_config')}")
    print()

    # Test queries with different scenarios
    test_cases = [
        {
            "query": "poplatok na ambasáde",
            "tags": ["poplatok", "prechodný pobyt"],
            "description": "Základný dotaz s matching tagmi"
        },
        {
            "query": "vnútropodnikové vyslanie ICT",
            "tags": ["vnútropodnikové vyslanie", "ICT"],
            "description": "Špecifický dotaz s presnými tagmi"
        },
        {
            "query": "žiadosť o trvalý pobyt",
            "tags": ["žiadosť", "trvalý pobyt"],
            "description": "Legislatívny dotaz"
        },
        {
            "query": "zdravotné poistenie",
            "tags": ["poistenie", "zdravie"],
            "description": "Dotaz bez matching tagov"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"🔍 Test {i}: {test_case['description']}")
        print(f"   Query: '{test_case['query']}'")
        print(f"   Tags: {test_case['tags']}")

        # Search with tag boosting
        results = engine.search(
            query=test_case['query'],
            user_tags=test_case['tags'],
            limit=5,
            tag_boost=0.25  # 25% boost
        )

        print(f"   📋 Results: {len(results)} found")

        # Display top 3 results
        for j, result in enumerate(results[:3], 1):
            boost_indicator = "📈" if result.was_boosted else "  "
            print(f"   {j}. {boost_indicator} Score: {result.score:.3f} (orig: {result.original_score:.3f})")
            print(f"      Category: {result.category}")
            if result.was_boosted:
                print(f"      🎯 Matched tags: {result.matched_tags}")
            print(f"      Text: {result.text[:80]}...")
        print()

    # Test summary vector search
    print("📝 Testing Summary Vector Search")
    print("-" * 40)

    summary_results = engine.search_summary_vector(
        query="dokumenty potrebné pre pobyt",
        user_tags=["dokumenty", "pobyt"],
        limit=3
    )

    print(f"📋 Summary vector results: {len(summary_results)}")
    for i, result in enumerate(summary_results, 1):
        boost_indicator = "📈" if result.was_boosted else "  "
        print(f"{i}. {boost_indicator} Score: {result.score:.3f}")
        print(f"   Summary: {result.summary[:100]}...")
        if result.matched_tags:
            print(f"   🎯 Matched tags: {result.matched_tags}")
    print()

    # Test document retrieval by ID
    print("🔍 Testing Document Retrieval by ID")
    print("-" * 40)

    if results:
        first_result_id = results[0].id
        doc = engine.get_document_by_id(first_result_id)
        if doc:
            print(f"📄 Retrieved document: {doc.id}")
            print(f"   Category: {doc.category}")
            print(f"   Tags: {doc.tags}")
            print(f"   Source: {doc.source}")
        else:
            print("❌ Document not found")

    print("✅ Hybrid search testing completed!")

if __name__ == "__main__":
    test_hybrid_search()
