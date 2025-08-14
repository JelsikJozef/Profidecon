from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any,List, Optional

class ParsedDocument(BaseModel):
    text: str
    metadata: dict[str, Any]
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    pseudonymized: bool = False
    mapping_id: Optional[str] = None
    pii_detected: List[str] = []

class BaseParser(ABC):
    suffixes: tuple[str, ...]

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        pass
