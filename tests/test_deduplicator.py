from preprocessor.processors.deduplicator import Deduplicator
from preprocessor.parsers.base import ParsedDocument

def test_deduplication():
    dedup = Deduplicator()
    doc1 = ParsedDocument(text="Hello", metadata={})
    out1 = dedup.process(doc1)
    assert out1.metadata["is_duplicate"] is False

    # rovnaký obsah = duplicate
    doc2 = ParsedDocument(text="Hello", metadata={})
    out2 = dedup.process(doc2)
    assert out2.metadata["is_duplicate"] is True

    # iný obsah = non-duplicate
    doc3 = ParsedDocument(text="World", metadata={})
    out3 = dedup.process(doc3)
    assert out3.metadata["is_duplicate"] is False


def test_deduplicator_preserves_summary_and_tags():
    dedup = Deduplicator()
    doc = ParsedDocument(text="Hello", metadata={}, summary="sum", tags=["tag"])
    out = dedup.process(doc)
    assert out.summary == "sum"
    assert out.tags == ["tag"]
