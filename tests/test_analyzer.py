# tests/test_analyzer.py
import json
from pathlib import Path
import pytest
from preprocessor.taxonomy.analyzer import (
    load_raw_metadata,
    build_tag_corpus,
    cluster_tags,
    propose_taxonomy,
)


# Fixture: create a temporary Preprocessed directory with JSONL files
@pytest.fixture
def sample_metadata(tmp_path: Path) -> Path:
    records = [
        {"source": "/tmp/doc1.pdf", "tags": ["a", "b"], "summary": "text1"},
        {"source": "/tmp/doc2.pdf", "tags": ["b", "c"], "summary": "text2"},
    ]
    pre_dir = tmp_path / "Preprocessed"
    pre_dir.mkdir()
    for idx, rec in enumerate(records):
        path = pre_dir / f"file{idx}.jsonl"
        with path.open('w', encoding='utf-8') as f:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")
    return pre_dir


def test_load_raw_metadata(sample_metadata: Path):
    records = load_raw_metadata(sample_metadata)
    assert isinstance(records, list)
    assert len(records) == 2
    summaries = {rec["summary"] for rec in records}
    assert "text1" in summaries and "text2" in summaries


def test_build_tag_corpus_and_cluster(sample_metadata: Path):
    records = load_raw_metadata(sample_metadata)
    X, features = build_tag_corpus(records)
    # TF-IDF matrix should match shape (2 docs, number of unique tags)
    unique_tags = sorted({t for rec in records for t in rec.get("tags", [])})
    assert X.shape == (2, len(unique_tags))
    labels = cluster_tags(X, n_clusters=2)
    assert list(labels) in ([0, 1], [1, 0])  # two distinct clusters


def test_propose_taxonomy(monkeypatch):
    # stub OpenAI response
    fake_tree = {"Root": {"Child": ["a", "b"]}}

    class FakeResp:
        choices = [type('c', (object,), {'message': type('m', (object,), {'content': json.dumps(fake_tree)})()})()]

    def fake_create(**kwargs):
        return FakeResp()

    monkeypatch.setattr('openai.ChatCompletion.create', fake_create)
    tags = ["a", "b"]
    tree = propose_taxonomy(tags)
    assert isinstance(tree, dict)
    assert tree == fake_tree
