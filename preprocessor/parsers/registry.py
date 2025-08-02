from .base import BaseParser
from .docx_parser import DocxParser
from .pdf_parser import PdfParser
from .msg_parser import MsgParser

class ParserRegistry:
    def __init__(self):
        self._parsers: list[BaseParser] = [
            DocxParser(),
            PdfParser(),
            MsgParser(),
        ]

    def get_parser(self, suffix: str) -> BaseParser:
        suffix = suffix.lower()
        for parser in self._parsers:
            if suffix in parser.suffixes:
                return parser
        raise ValueError(f"No parser found for suffix '{suffix}'")
