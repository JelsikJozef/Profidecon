from preprocessor.parsers.base import ParsedDocument

class QualityChecker:
    """
    Flaguje dokumenty, ktoré sú príliš veľké alebo majú priveľa riadkov.
    """
    def __init__(self, max_bytes: int = 8_000_000, max_lines: int = 1000):
        self.max_bytes = max_bytes
        self.max_lines = max_lines

    def process(self, doc: ParsedDocument) -> ParsedDocument:
        text = doc.text
        size_bytes = len(text.encode("utf-8"))
        lines = text.count("\n") + 1

        requires_split = (size_bytes > self.max_bytes) or (lines > self.max_lines)

        new_meta = {
            **doc.metadata,
            "size_bytes_text": size_bytes,
            "line_count": lines,
            "requires_split": requires_split,
        }
        return ParsedDocument(text=text, metadata=new_meta)
