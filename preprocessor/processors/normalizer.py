import re
import unicodedata
from preprocessor.parsers.base import ParsedDocument

def normalize(doc: ParsedDocument) -> ParsedDocument:
    text = doc.text

    # Odstráň časté patterny hlavičiek/pätiek (napr. page numbers)
    text = re.sub(r"Page \\d+ of \\d+", "", text)

    # Odstráň zbytočné whitespace, zalomenia
    text = re.sub(r"\\s+", " ", text)

    # Normalizuj Unicode (NFC)
    text = unicodedata.normalize("NFC", text)

    # Odstráň časté e-mailové úvodzovky: "From: … Sent: … To: …"
    text = re.sub(r"(?i)(from|sent|to|subject):.+?\\n", "", text)

    return ParsedDocument(
        text=text.strip(),
        metadata=doc.metadata
    )
