from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any,List, Optional

class ParsedDocument(BaseModel):
    text: str
    metadata: dict[str, Any]
    summary: str = ""
    tags: list[str] = Field(default_factory=list)

class BaseParser(ABC):
    suffixes: tuple[str, ...]

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        pass
