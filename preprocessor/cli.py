import argparse
from pathlib import Path
import logging
import asyncio

from preprocessor.ingestion import ingest_batch
from preprocessor.parsers.registry import ParserRegistry
from preprocessor.processors.normalizer import normalize
from preprocessor.processors.ocr import needs_ocr, apply_ocr
from preprocessor.processors.enricher import Enricher
from preprocessor.processors.deduplicator import Deduplicator
from preprocessor.processors.quality_checker import QualityChecker
from preprocessor.processors.serializer import JsonlSerializer

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run document preprocessor pipeline")
    parser.add_argument("--input",  "-i", type=Path, required=True,
                        help="Cesta k priečinku s dokumentmi")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Cesta, kam uložiť JSONL výstupy")
    args = parser.parse_args()

    root = args.input
    output = args.output

    logger.info(f"📂 Ingesting from {root}")
    #docs = ingest_batch(root)  # ak chceš asynchrónne, uprav podľa potreby
    docs = asyncio.run(ingest_batch(root))

    registry    = ParserRegistry()
    enricher    = Enricher(root_path=root)
    deduplicator= Deduplicator()
    qc          = QualityChecker()
    serializer  = JsonlSerializer(output_dir=output)

    for raw in docs:
        try:
            # 1) Parse
            parser = registry.get_parser(raw.path.suffix)
            parsed = parser.parse(raw.path)

            # 2) Normalize
            normed = normalize(parsed)

            # 3) OCR fallback
            if raw.path.suffix.lower() == ".pdf" and needs_ocr(normed):
                logger.info(f"OCR for {raw.path.name}")
                parsed = apply_ocr(raw.path)
            else:
                parsed = normed

            # 4) Enrich
            enriched = enricher.enrich(parsed)

            # 5) Dedup
            deduped = deduplicator.process(enriched)

            # 6) Quality check
            checked = qc.process(deduped)

            # 7) Serialize
            out_path = serializer.serialize(checked)
            logger.info(f"→ Wrote {out_path.name}")

        except Exception as e:
            logger.error(f"Chyba pri spracovaní {raw.path}: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    main()
