import asyncio
from pathlib import Path
from typing import AsyncIterator
from watchfiles import awatch

from .models import RawDocument
import os
from datetime import datetime

async def ingest_batch(root: Path,
                       include: tuple[str, ...]=(".pdf", ".docx", ".msg")) -> list[RawDocument]:
    """Prehľadá FS rekurzívne + vráti zoznam dokumentov vhodných na parsing."""
    docs: list[RawDocument] = []
    for ext in include:
        for p in root.rglob(f"*{ext}"):
            st = p.stat()
            docs.append(RawDocument(
                path=p,
                size_bytes=st.st_size,
                modified_ts=datetime.fromtimestamp(st.st_mtime)
            ))
    return docs

async def ingest_watch(root: Path) -> AsyncIterator[RawDocument]:
    """Nepretržité sledovanie priečinka – yielduje nové/zmienené súbory."""
    async for changes in awatch(root):
        for typ, pstr in changes:
            p = Path(pstr)
            if p.suffix.lower() in {".pdf", ".docx", ".msg"}:
                st = p.stat()
                yield RawDocument(
                    path=p,
                    size_bytes=st.st_size,
                    modified_ts=datetime.fromtimestamp(st.st_mtime)
                )
