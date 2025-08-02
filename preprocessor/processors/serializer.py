import json
from pathlib import Path
from preprocessor.parsers.base import ParsedDocument

class JsonlSerializer:
    """
    Serializuje ParsedDocument do JSONL formátu,
    jeden dokument = jeden riadok.
    """
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def serialize(self, doc: ParsedDocument) -> Path:
        # použiť doc.metadata['hash_sha1'] alebo generovať nové ID
        doc_id = doc.metadata.get("hash_sha1") or ""
        filename = f"{doc_id}.jsonl"
        out_path = self.output_dir / filename

        with open(out_path, "w", encoding="utf-8") as f:
            # zachováme text aj všetky metadata
            json.dump({
                "text": doc.text,
                "metadata": doc.metadata
            }, f, ensure_ascii=False)
            f.write("\n")
        return out_path
