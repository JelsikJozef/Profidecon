import pytest
from pathlib import Path
from preprocessor.parsers.base import ParsedDocument
from preprocessor.processors.enricher import Enricher

@pytest.fixture
def sample_txt(tmp_path):
    f = tmp_path / "CategoryA" / "file.txt"
    f.parent.mkdir()
    content = "Hello world. Contact: test@example.com, +421900123456."
    f.write_text(content, encoding="utf-8")
    return f

def test_enrichment(sample_txt):
    # priprava ParsedDocument so surovou metadata
    raw = ParsedDocument(text=sample_txt.read_text(), metadata={"source": str(sample_txt)})
    enr = Enricher(root_path=sample_txt.parent.parent)
    out = enr.enrich(raw)

    # overenie metadát
    assert out.metadata["size_bytes"] == sample_txt.stat().st_size
    assert out.metadata["language"] == "en"
    assert out.metadata["token_estimate"] >= 5
    assert "hash_sha1" in out.metadata
    assert 0 <= out.metadata["pii_risk"] <= 1
    assert out.metadata["category"] == "CategoryA"
