from pathlib import Path
import json
import pytest
from preprocessor.processors.serializer import JsonlSerializer
from preprocessor.parsers.base import ParsedDocument

def test_serialization(tmp_path):
    meta = {"hash_sha1": "abc123"}
    doc = ParsedDocument(text="Test", metadata=meta)
    ser = JsonlSerializer(output_dir=tmp_path)
    out_file = ser.serialize(doc)

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["text"] == "Test"
    assert data["metadata"] == meta
    assert out_file.name == "abc123.jsonl"
