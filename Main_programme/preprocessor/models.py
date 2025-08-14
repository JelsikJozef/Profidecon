from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime

class RawDocument(BaseModel):
    path: Path
    size_bytes: int
    modified_ts: datetime
