# tests/test_analyzer.py
import json
import os
from pathlib import Path
import pytest
from preprocessor.taxonomy.analyzer import (
    load_raw_metadata,
    build_tag_corpus,
    cluster_tags,
    propose_taxonomy,
)


# Fixture: create a temporary metadata_raw.jsonl
@pytest.fixture
def sample_metadata(tmp_path):
    records = [
        {"cesta_k_suboru": "/tmp/doc1.pdf", "nazov_suboru": "doc1.pdf", "tagy": ["a", "b"], "sumar": "text1"},
        {"cesta_k_suboru": "/tmp/doc2.pdf", "nazov_suboru": "doc2.pdf", "tagy": ["b", "c"], "sumar": "text2"},
    ]
    path = tmp_path / "metadata_raw.jsonl"
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")
    return path


def test_load_raw_metadata(sample_metadata):
    records = load_raw_metadata(sample_metadata)
    assert isinstance(records, list)
    assert len(records) == 2
    assert records[0]["nazov_suboru"] == "doc1.pdf"


def test_build_tag_corpus_and_cluster(sample_metadata):
    records = load_raw_metadata(sample_metadata)
    X, features = build_tag_corpus(records)
    # TF-IDF matrix should match shape (2 docs, number of unique tags)
    unique_tags = sorted({t for rec in records for t in rec.get("tagy", [])})
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
