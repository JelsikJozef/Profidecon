from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel
from typing import Any

class ParsedDocument(BaseModel):
    text: str
    metadata: dict[str, Any]

class BaseParser(ABC):
    suffixes: tuple[str, ...]

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        pass
