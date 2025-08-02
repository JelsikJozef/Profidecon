from preprocessor.parsers.base import ParsedDocument
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path

def needs_ocr(doc: ParsedDocument) -> bool:
    return len(doc.text.strip()) < 50  # málo textu = možno len obrázok

def apply_ocr(pdf_path: Path) -> ParsedDocument:
    images = convert_from_path(str(pdf_path))
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang='eng+slk') + "\n"
    return ParsedDocument(text=text.strip(), metadata={"ocr": True, "source": str(pdf_path)})
