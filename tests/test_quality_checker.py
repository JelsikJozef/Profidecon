import pytest
from preprocessor.processors.quality_checker import QualityChecker
from preprocessor.parsers.base import ParsedDocument

@pytest.fixture
def small_doc():
    text = "Line\n" * 10
    return ParsedDocument(text=text, metadata={})

@pytest.fixture
def big_doc():
    # vytvoríme reťazec >1 000 riadkov
    text = "L\n" * 1001
    return ParsedDocument(text=text, metadata={})

def test_small_quality(small_doc):
    qc = QualityChecker(max_bytes=1000, max_lines=100)
    out = qc.process(small_doc)
    assert out.metadata["requires_split"] is False

def test_big_quality(big_doc):
    qc = QualityChecker(max_bytes=1000, max_lines=1000)
    out = qc.process(big_doc)
    assert out.metadata["requires_split"] is True
