# src/preprocessor/cli.py
"""
Main CLI for Document Preprocessor including Taxonomy modules
"""
import argparse
import logging
import asyncio
import time
from pathlib import Path

from preprocessor.ingestion import ingest_batch
from preprocessor.parsers.registry import ParserRegistry
from preprocessor.processors.normalizer import normalize
from preprocessor.processors.ocr import needs_ocr, apply_ocr
from preprocessor.processors.enricher import Enricher
from preprocessor.processors.deduplicator import Deduplicator
from preprocessor.processors.quality_checker import QualityChecker
from preprocessor.processors.serializer import JsonlSerializer
from preprocessor.taxonomy.extractor import TaxonomyExtractor
from preprocessor.taxonomy.analyzer import main as analyze_taxonomy

logger = logging.getLogger(__name__)


def run_pipeline(input_dir: Path, output_dir: Path):
    """Run document preprocessing pipeline"""
    logger.info(f"📂 Ingesting from {input_dir}")
    docs = asyncio.run(ingest_batch(input_dir))

    registry    = ParserRegistry()
    enricher    = Enricher(root_path=input_dir)
    deduplicator= Deduplicator()
    qc          = QualityChecker()
    serializer  = JsonlSerializer(output_dir=output_dir)

    for raw in docs:
        try:
            # 1) Parse
            logger.info(f"→ Parsing {raw.path.name}")
            parser = registry.get_parser(raw.path.suffix)
            parsed = parser.parse(raw.path)

            # 2) Normalize
            logger.info("→ Normalizing content")
            normed = normalize(parsed)

            # 3) OCR fallback only for PDF
            logger.info("→ Checking for OCR needs")
            if raw.path.suffix.lower() == ".pdf" and needs_ocr(normed):
                logger.info(f"OCR for {raw.path.name}")
                parsed = apply_ocr(raw.path)
            else:
                parsed = normed

            # 4) Enrich
            logger.info("→ Enriching metadata")
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


def main():
    parser = argparse.ArgumentParser(description="Document Preprocessor CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # preprocess command
    p_pre = subparsers.add_parser("preprocess", help="Run preprocessing pipeline")
    p_pre.add_argument("--input", "-i", type=Path, required=True, help="Input folder with documents")
    p_pre.add_argument("--output", "-o", type=Path, required=True, help="Output folder for JSONL files")

    # taxonomy extract
    p_ext = subparsers.add_parser("taxonomy-extract", help="Extract metadata for taxonomy using OpenAI")
    p_ext.add_argument("root", type=Path, help="Input root folder for documents")
    p_ext.add_argument("--out", type=Path, default=Path("metadata_raw.jsonl"), help="Output metadata JSONL file")

    # taxonomy analyze
    p_ana = subparsers.add_parser("taxonomy-analyze", help="Generate taxonomy from raw metadata")
    p_ana.add_argument("root", type=Path, help="Root folder path")
    p_ana.add_argument("--preprocessed", type=Path, default=Path("Preprocessed"),
                       help="Path to directory with preprocessed JSONL files")
    p_ana.add_argument("--out", type=Path, default=Path("taxonomy.json"), help="Output taxonomy JSON file")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "preprocess":
        run_pipeline(args.input, args.output)
    elif args.command == "taxonomy-extract":
        ext = TaxonomyExtractor(out_path=args.out)
        files = (p for p in args.root.rglob("*.*") if p.is_file())
        ext.process_files(files)
    elif args.command == "taxonomy-analyze":
        analyze_taxonomy(root=args.root, preprocessed_dir=args.preprocessed, out_tax=args.out)
if __name__ == "__main__":
    main()
