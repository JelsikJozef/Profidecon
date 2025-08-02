from preprocessor.parsers.registry import ParserRegistry
from pathlib import Path

def test_get_parser_docx():
    reg = ParserRegistry()
    parser = reg.get_parser(".docx")
    assert parser is not None
    assert ".docx" in parser.suffixes

def test_get_parser_unknown():
    reg = ParserRegistry()
    try:
        reg.get_parser(".xyz")
    except ValueError as e:
        assert "No parser found" in str(e)