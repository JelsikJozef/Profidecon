from .base import BaseParser, ParsedDocument
from extract_msg import Message
from pathlib import Path

class MsgParser(BaseParser):
    suffixes = (".msg",)

    def parse(self, path: Path) -> ParsedDocument:
        msg = Message(str(path))
        return ParsedDocument(
            text=msg.body or "",
            metadata={"source": str(path),
                      "subject":msg.subject
            }
        )