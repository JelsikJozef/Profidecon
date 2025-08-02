import hashlib
from pathlib import Path
from langdetect import detect
from preprocessor.parsers.base import ParsedDocument
from .normalizer import normalize

# jednoduchý odhad tokenov podľa splitu na whitespace
def estimate_tokens(text: str) -> int:
    return len(text.split())

# stub pre PII risk (napr. hľadanie číselných sekvencií ako telefóny)
def pii_risk_score(text: str) -> float:
    # jednoduchý koncept: koľko e-mailov / číselných vzorov
    import re
    emails = len(re.findall(r"[\\w\\.-]+@[\\w\\.-]+", text))
    phones = len(re.findall(r"\\+?\\d{6,}", text))
    total = len(text.split())
    return round((emails + phones) / max(total, 1), 4)

class Enricher:
    def __init__(self, root_path: Path):
        self.root = root_path

    def enrich(self, doc: ParsedDocument) -> ParsedDocument:
        path = Path(doc.metadata.get("source", ""))
        # 1) Normalize už spracovaný text (ak potrebujeme)
        text = normalize(doc).text

        # 2) Základné súborové vlastnosti
        size = path.stat().st_size
        mtime = path.stat().st_mtime
        ctime = path.stat().st_ctime

        # 3) Language detection
        try:
            lang = detect(text)
        except:
            lang = "unknown"

        # 4) Token estimate
        tokens = estimate_tokens(text)

        # 5) SHA-1 hash
        sha1 = hashlib.sha1(text.encode("utf-8")).hexdigest()

        # 6) PII risk
        pii = pii_risk_score(text)

        # 7) Kategória podľa priečinkov (relatívne)
        rel = path.relative_to(self.root)
        category = rel.parent.parts[0] if len(rel.parent.parts) > 0 else "root"

        # 8) Zlúčime existujúcu metadata s novými
        new_meta = {
            **doc.metadata,
            "size_bytes": size,
            "created_ts": ctime,
            "modified_ts": mtime,
            "language": lang,
            "token_estimate": tokens,
            "hash_sha1": sha1,
            "pii_risk": pii,
            "category": category,
        }

        return ParsedDocument(text=text, metadata=new_meta)
