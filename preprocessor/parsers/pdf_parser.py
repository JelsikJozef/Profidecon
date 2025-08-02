from .base import BaseParser, ParsedDocument
from pathlib import Path
import pdfminer.high_level

class PdfParser(BaseParser):
    suffixes = (".pdf",)

    def parse(self, path: Path) -> ParsedDocument:
        text = pdfminer.high_level.extract_text(str(path))
        return ParsedDocument(text=text, metadata={"source": str(path)})
