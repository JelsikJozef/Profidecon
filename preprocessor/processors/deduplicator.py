import hashlib
from pathlib import Path
from preprocessor.parsers.base import ParsedDocument

class Deduplicator:
    """
    Jednoduchý dedup proof-of-concept.
    V produkcii odporúčame použiť Redis/Mongo pre persistenciu.
    """
    def __init__(self):
        # tu by bol napr. Redis connection a množina hashov
        self._seen_hashes: set[str] = set()

    def process(self, doc: ParsedDocument) -> ParsedDocument:
        # vypočítame hash textu (alebo ak chceme kombinovať metadata, path aj text)
        content = doc.text.encode("utf-8")
        doc_hash = hashlib.sha1(content).hexdigest()

        is_dup = doc_hash in self._seen_hashes
        if not is_dup:
            self._seen_hashes.add(doc_hash)

        # pridáme flagy do metadata
        new_meta = {
            **doc.metadata,
            "hash_content": doc_hash,
            "is_duplicate": is_dup,
        }
        return ParsedDocument(text=doc.text, metadata=new_meta)
