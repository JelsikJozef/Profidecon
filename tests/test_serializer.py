from pathlib import Path
import json
import pytest
from preprocessor.processors.serializer import JsonlSerializer
from preprocessor.parsers.base import ParsedDocument

def test_serialization_with_optional_fields(tmp_path):
    meta = {"hash_sha1": "abc123"}
    doc = ParsedDocument(
        text="Test",
        metadata=meta,
        summary="súhrn",
        tags=["a", "b"],
    )
    ser = JsonlSerializer(output_dir=tmp_path)
    out_file = ser.serialize(doc)

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["text"] == "Test"
    assert data["source"] == "src.txt"
    assert data["size_bytes"] == 100
    assert "metadata" not in data
    assert "hash_sha1" not in data
    assert "summary" in data and data["summary"] == ""
    assert "tags" in data and data["tags"] == []
    assert out_file.name == "abc123.jsonl"


def test_serialization_without_optional_fields(tmp_path):
    meta = {"hash_sha1": "def456"}
    doc = ParsedDocument(text="Test2", metadata=meta)
    ser = JsonlSerializer(output_dir=tmp_path)
    out_file = ser.serialize(doc)

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "summary" not in data
    assert "tags" not in data
    assert data["text"] == "Test2"
    assert data["metadata"] == meta
    assert out_file.name == "def456.jsonl"
