#!/usr/bin/env python3
"""
Test hybrid search system against human.csv ground truth.
Compares results from the new dual vector + tag boosting system.
"""

import csv
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from sdk.retrieval import RetrievalEngine

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test query."""
    query: str
    expected_id: str
    found: bool
    rank: Optional[int]
    top_results: List[Dict]
    boost_applied: bool


def hash_to_uuid(hash_content: str) -> str:
    """Convert hash_content to UUID format (same as vectorizer)."""
    if len(hash_content) == 40:  # SHA-1 hash
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, hash_content))
    else:
        return hash_content


def load_ground_truth(file_path: str = "tests/human.csv") -> List[Tuple[str, str]]:
    """Load ground truth queries and expected results."""
    ground_truth = []

    if not Path(file_path).exists():
        logger.error(f"Ground truth file not found: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row.get('query', '').strip()
            expected_id = row.get('expected_doc_id', '').strip()

            if query and expected_id:
                ground_truth.append((query, expected_id))

    logger.info(f"Loaded {len(ground_truth)} ground truth queries")
    return ground_truth


def test_hybrid_search_with_ground_truth():
    """Test hybrid search system against ground truth data."""
    print("🧪 TESTING HYBRID SEARCH WITH GROUND TRUTH")
    print("=" * 60)

    # Initialize retrieval engine
    engine = RetrievalEngine()

    # Load ground truth
    ground_truth = load_ground_truth()
    if not ground_truth:
        print("❌ No ground truth data found")
        return

    # Test results
    test_results = []
    found_count = 0
    boost_count = 0

    print(f"🔍 Testing {len(ground_truth)} queries...\n")

    for i, (query, expected_hash) in enumerate(ground_truth, 1):
        print(f"Query {i}/{len(ground_truth)}: {query}")

        # Convert expected hash to UUID
        expected_uuid = hash_to_uuid(expected_hash)

        # Extract potential tags from query for boosting
        user_tags = []
        if 'poplatok' in query.lower():
            user_tags.append('poplatok')
        if 'pobyt' in query.lower():
            user_tags.extend(['pobyt', 'prechodný pobyt', 'trvalý pobyt'])
        if 'žiadosť' in query.lower():
            user_tags.append('žiadosť')
        if 'poistenie' in query.lower():
            user_tags.append('poistenie')
        if 'dokumenty' in query.lower() or 'doklady' in query.lower():
            user_tags.append('dokumenty')
        if 'víza' in query.lower() or 'visa' in query.lower():
            user_tags.append('víza')

        # Search with hybrid system
        results = engine.search(
            query=query,
            user_tags=user_tags,
            limit=10,
            tag_boost=0.25  # 25% boost
        )

        # Check if expected document is found
        found = False
        rank = None
        boost_applied = any(r.was_boosted for r in results)

        for idx, result in enumerate(results):
            if result.id == expected_uuid:
                found = True
                rank = idx + 1
                found_count += 1
                break

        if boost_applied:
            boost_count += 1

        # Store test result
        top_results = []
        for j, result in enumerate(results[:5]):
            top_results.append({
                'rank': j + 1,
                'id': result.id,
                'score': result.score,
                'original_score': result.original_score,
                'category': result.category,
                'was_boosted': result.was_boosted,
                'matched_tags': result.matched_tags,
                'text_preview': result.text[:60] + "..." if len(result.text) > 60 else result.text
            })

        test_results.append(TestResult(
            query=query,
            expected_id=expected_hash,
            found=found,
            rank=rank,
            top_results=top_results,
            boost_applied=boost_applied
        ))

        # Display result
        status = "✅ FOUND" if found else "❌ NOT FOUND"
        rank_text = f" (rank {rank})" if rank else ""
        boost_text = " 📈 BOOST" if boost_applied else ""
        print(f"   {status}{rank_text}{boost_text}")

        if user_tags:
            print(f"   🏷️  Used tags: {user_tags}")

        if found and results:
            best_result = results[0]
            print(f"   🎯 Best match: {best_result.category} - Score: {best_result.score:.3f}")
        elif results:
            print(f"   📄 Top result: {results[0].category} - Score: {results[0].score:.3f}")

        print()

    # Calculate metrics
    recall_at_5 = found_count / len(ground_truth)
    recall_at_10 = found_count / len(ground_truth)  # We searched with limit=10

    # Print summary
    print("📊 SUMMARY RESULTS")
    print("-" * 40)
    print(f"Total queries: {len(ground_truth)}")
    print(f"Found in top-10: {found_count}")
    print(f"Recall@10: {recall_at_10:.3f} ({recall_at_10*100:.1f}%)")
    print(f"Queries with boost: {boost_count}")
    print(f"Boost rate: {boost_count/len(ground_truth)*100:.1f}%")

    # Show some detailed results
    print(f"\n📋 DETAILED RESULTS ")
    print("-" * 50)

    for i, result in enumerate(test_results):
        status = "✅" if result.found else "❌"
        rank_text = f"rank {result.rank}" if result.rank else "not found"
        boost_text = "📈" if result.boost_applied else "  "

        print(f"{i+1:2d}. {status} {boost_text} {result.query[:30]:<30} | {rank_text}")

        if result.top_results:
            best = result.top_results[0]
            boost_indicator = "📈" if best['was_boosted'] else "  "
            print(f"     {boost_indicator} Top: {best['score']:.3f} - {best['category']}")

    # Find queries that benefited from boosting
    boosted_and_found = [r for r in test_results if r.found and r.boost_applied]
    if boosted_and_found:
        print(f"\n🚀 QUERIES THAT BENEFITED FROM TAG BOOSTING ({len(boosted_and_found)})")
        print("-" * 55)
        for result in boosted_and_found:
            print(f"• {result.query}")
            if result.top_results:
                best = result.top_results[0]
                if best['was_boosted']:
                    print(f"  Score: {best['score']:.3f} (boosted from {best['original_score']:.3f})")
                    print(f"  Tags: {best['matched_tags']}")

    print(f"\n✅ Hybrid search testing completed!")
    return test_results


if __name__ == "__main__":
    test_hybrid_search_with_ground_truth()
