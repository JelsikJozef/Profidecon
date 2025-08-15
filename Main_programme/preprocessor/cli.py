# src/preprocessor/cli.py
"""
Main CLI for Document Preprocessor including Taxonomy modules
"""
import argparse
import logging
import asyncio
from pathlib import Path

from Main_programme.preprocessor.ingestion import ingest_batch
from Main_programme.preprocessor.parsers.registry import ParserRegistry
from Main_programme.preprocessor.processors.normalizer import normalize
from Main_programme.preprocessor.processors.ocr import needs_ocr, apply_ocr
from Main_programme.preprocessor.processors.enricher import Enricher
from Main_programme.preprocessor.processors.pii_analyzer import PiiAnalyzer
from Main_programme.preprocessor.processors.deduplicator import Deduplicator
from Main_programme.preprocessor.processors.quality_checker import QualityChecker
from Main_programme.preprocessor.processors.serializer import JsonlSerializer
from Main_programme.preprocessor.taxonomy.extractor import TaxonomyExtractor
from Main_programme.preprocessor.taxonomy.analyzer import main as analyze_taxonomy
from Main_programme.preprocessor.processors.pseudonymizer import Pseudonymizer
from Main_programme.preprocessor.processors.enricher_llm import LlmEnricher
from Main_programme.preprocessor.observability.logging_config import setup_json_logging
from Main_programme.preprocessor.observability.metrics import phase_duration, pii_tokens_total, errors_total
from Main_programme.preprocessor.observability.tracing import span

import json
import os
import sys
import time

logger = logging.getLogger(__name__)


def run_pipeline(input_dir: Path, output_dir: Path):
    counter = 0
    """Run document preprocessing pipeline"""
    logger.info(f"📂 Ingesting from {input_dir}")
    docs = asyncio.run(ingest_batch(input_dir))

    registry    = ParserRegistry()
    enricher    = Enricher(root_path=input_dir, llm_backend="openai")
    pii_analyzer = PiiAnalyzer()  # Initialize PII analyzer
    deduplicator= Deduplicator()
    qc          = QualityChecker()
    serializer  = JsonlSerializer(output_dir=output_dir)

    for raw in docs:
        counter += 1
        try:
            # 1) Parse
            logger.info(f"→ {counter} Parsing {raw.path.name}")
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

            # 5) PII Analysis (Phase-1: detection only)
            logger.info("→ Analyzing PII entities")
            pii_entities = pii_analyzer.detect(enriched.text)

            # Store PII entities in metadata for later use by pseudonymizer
            enriched.metadata["pii_entities"] = [dict(entity) for entity in pii_entities]
            enriched.metadata["pii_count"] = len(pii_entities)

            if pii_entities:
                entity_types = set(e["type"] for e in pii_entities)
                logger.info(f"   Found {len(pii_entities)} PII entities: {', '.join(entity_types)}")

            # 6) Dedup
            deduped = deduplicator.process(enriched)

            # 7) Quality check
            checked = qc.process(deduped)

            # 8) Serialize
            out_path = serializer.serialize(checked)
            logger.info(f"→ Wrote {out_path.name}")

        except Exception as e:
            logger.error(f"Chyba pri spracovaní {raw.path}: {e}")


def run_pseudonymization(input_dir: Path, output_dir: Path, *, scope: str, tenant_id: str | None,
                         types_include: list[str] | None, types_exclude: list[str] | None,
                         max_entities: int | None, require_annotations: bool, force: bool):
    logger.info(f"🔒 Pseudonymizing Phase-1 docs: {input_dir} → {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pseudo = Pseudonymizer(scope=scope, tenant_id=tenant_id)
    analyzer = None if require_annotations else PiiAnalyzer()

    def _sanitize_entities(ents: list[dict]) -> list[dict]:
        out: list[dict] = []
        for e in ents or []:
            ed = dict(e)
            if 'value' in ed:
                ed['value'] = None
            out.append(ed)
        return out

    processed = skipped = errors = 0
    t0 = time.time()

    for path in sorted(input_dir.glob('*.jsonl')):
        try:
            data = json.loads(path.read_text(encoding='utf-8').splitlines()[0])
        except Exception as e:
            logger.error(f"Failed to read {path.name}: {e}")
            errors += 1
            continue

        meta = data.get('metadata', {}) if isinstance(data.get('metadata'), dict) else {}
        already = bool(meta.get('pseudonymized') or data.get('pseudonymized'))
        if already and not force:
            skipped += 1
            continue

        text = data.get('text') or ''
        if not text:
            skipped += 1
            continue

        ents = data.get('pii_entities') or meta.get('pii_entities')
        if not ents:
            if require_annotations:
                logger.error(f"Missing pii_entities for {path.name} and require-annotations is True")
                errors += 1
                continue
            else:
                ents = analyzer.detect(text)

        try:
            res = pseudo.run(
                text=text,
                entities=ents,
                types_include=types_include,
                types_exclude=types_exclude,
                max_entities=max_entities,
            )
        except Exception as e:
            logger.error(f"Pseudonymization failed for {path.name}: {e}")
            errors += 1
            continue

        out_doc = {}
        for k, v in data.items():
            if k in ('text', 'pii_entities'):
                continue
            out_doc[k] = v

        out_meta = out_doc.get('metadata') if isinstance(out_doc.get('metadata'), dict) else {}
        out_meta['pseudonymized'] = True
        out_meta['pseudonymization'] = {
            'scope': scope,
            'tenant_id': tenant_id,
            'counts': dict(res['stats'])
        }
        out_meta['token_spans'] = list(res['spans'])
        out_meta['pii_entities'] = _sanitize_entities(ents)
        out_doc['metadata'] = out_meta
        out_doc['text_pseudo'] = res['text']

        out_path = output_dir / path.name
        try:
            with out_path.open('w', encoding='utf-8') as f:
                json.dump(out_doc, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to write {out_path.name}: {e}")
            errors += 1
            continue

        processed += 1
        logger.info(f"→ Wrote pseudonymized {out_path.name}")

    dt_ms = int((time.time() - t0) * 1000)
    logger.info(f"Pseudonymization done: processed={processed}, skipped={skipped}, errors={errors}, time_ms={dt_ms}")
    if errors:
        sys.exit(1)


def run_enrich_llm(
    input_dir: Path,
    output_dir: Path,
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    skip_if_present: bool = True,
    max_tokens_input: int = 4096,
):
    """
    Phase-3: Read Phase-2 pseudonymized docs and enrich via LLM using text_pseudo only.
    """
    logger.info(f"🤖 LLM enrichment Phase-3: {input_dir} → {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    class _OpenAIChatClient:
        def __init__(self):
            from openai import OpenAI  # type: ignore
            self._client = OpenAI()
        def enrich(self, *, text: str, model: str, temperature: float, max_tokens: int):
            # Expect complete prompt in text
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": text}],
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
                msg = resp.choices[0].message.content or ""
                # Best-effort JSON parse
                try:
                    import json as _json
                    cleaned = msg.replace("```json", "").replace("```", "").strip()
                    data = _json.loads(cleaned)
                    return {"summary": data.get("summary", ""), "tags": data.get("tags", [])}
                except Exception:
                    return {"summary": msg, "tags": []}
            except Exception as e:  # pragma: no cover
                logger.error("OpenAI enrich failed: %s", e)
                raise

    class _HeuristicClient:
        def enrich(self, *, text: str, model: str, temperature: float, max_tokens: int):
            # Simple fallback: take last part after 'Text:' as the body
            body = text.split("Text:\n", 1)[-1] if "Text:\n" in text else text
            body = body.strip()
            # Summary: first 3 sentences or 400 chars
            import re as _re
            sents = _re.split(r"(?<=[.!?])\s+", body)
            summary = " ".join(sents[:3])
            summary = summary[:400]
            # Tags: collect up to 10 token types like [EMAIL:XXXX] → email
            token_types = _re.findall(r"\[([A-Z_]+):[0-9A-Z]+\]", body)
            tags = []
            seen = set()
            for t in token_types:
                n = t.lower()
                if n not in seen:
                    seen.add(n)
                    tags.append(n)
                if len(tags) >= 10:
                    break
            return {"summary": summary, "tags": tags}

    # Choose client: use OpenAI only if API key is present; otherwise fallback
    api_key = os.getenv("OPENAI_API_KEY")
    client: Any
    if api_key:
        try:
            client = _OpenAIChatClient()
            logger.info("Using OpenAI client for LLM enrichment")
        except Exception as e:  # pragma: no cover
            logger.warning("Falling back to heuristic client: %s", e)
            client = _HeuristicClient()
    else:
        client = _HeuristicClient()
        logger.info("Using heuristic client (no OPENAI_API_KEY)")

    processed = skipped = errors = 0
    t0 = time.time()

    for path in sorted(input_dir.glob('*.jsonl')):
        try:
            raw_line = path.read_text(encoding='utf-8').splitlines()[0]
            data = json.loads(raw_line)
        except Exception as e:
            logger.error(f"Failed to read {path.name}: {e}")
            errors += 1
            continue

        meta = data.get('metadata') if isinstance(data.get('metadata'), dict) else {}
        if skip_if_present and isinstance(meta.get('llm_enrichment'), dict):
            skipped += 1
            continue

        pseudo = data.get('text_pseudo') or ''
        if not pseudo.strip():
            logger.warning("Missing text_pseudo in %s; skipping", path.name)
            skipped += 1
            continue

        enricher = LlmEnricher(api_client=client, model=model, temperature=temperature, max_tokens=max_tokens, max_tokens_input=max_tokens_input)
        try:
            res = enricher.run(pseudo, meta)
        except Exception as e:
            logger.error("LLM enrichment failed for %s: %s", path.name, e)
            errors += 1
            continue

        # Build output doc: preserve all, add llm_enrichment in metadata, set phase=3
        out_doc = dict(data)
        out_meta = out_doc.get('metadata') if isinstance(out_doc.get('metadata'), dict) else {}
        out_meta['llm_enrichment'] = {
            'summary': res['summary'],
            'tags': list(res['tags']),
            'model': res['model'],
            'latency_ms': int(res['latency_ms']),
        }
        out_meta['phase'] = 3
        out_doc['metadata'] = out_meta

        out_path = output_dir / path.name
        try:
            with out_path.open('w', encoding='utf-8') as f:
                json.dump(out_doc, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            logger.error("Failed to write %s: %s", out_path.name, e)
            errors += 1
            continue

        processed += 1
        logger.info("→ Wrote LLM-enriched %s (latency_ms=%s)", out_path.name, res['latency_ms'])

    dt_ms = int((time.time() - t0) * 1000)
    logger.info(f"LLM enrichment done: processed={processed}, skipped={skipped}, errors={errors}, time_ms={dt_ms}")
    if errors:
        sys.exit(1)


def _assert_phase1_dir(input_dir: Path) -> bool:
    ok = True
    for p in sorted(input_dir.glob('*.jsonl')):
        try:
            data = json.loads(p.read_text(encoding='utf-8').splitlines()[0])
        except Exception:
            continue
        meta = data.get('metadata') if isinstance(data.get('metadata'), dict) else {}
        if data.get('text_pseudo') or meta.get('pseudonymized') or meta.get('phase', 1) > 1:
            ok = False
            break
    return ok


def _assert_phase2_dir(input_dir: Path) -> bool:
    for p in sorted(input_dir.glob('*.jsonl')):
        try:
            data = json.loads(p.read_text(encoding='utf-8').splitlines()[0])
        except Exception:
            continue
        meta = data.get('metadata') if isinstance(data.get('metadata'), dict) else {}
        if not data.get('text_pseudo'):
            return False
        if not (meta.get('pseudonymized') or meta.get('phase', 2) >= 2):
            return False
    return True


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

    # pseudonymize command
    p_pseudo = subparsers.add_parser("pseudonymize", help="Run Phase-2 pseudonymization on Phase-1 JSONL")
    p_pseudo.add_argument("--input", "-i", type=Path, required=True, help="Input Phase-1 JSONL dir")
    p_pseudo.add_argument("--output", "-o", type=Path, required=True, help="Output Phase-2 dir")
    p_pseudo.add_argument("--scope", choices=["tenant", "global"], default=os.getenv("PSEUDO_SCOPE", "tenant"))
    p_pseudo.add_argument("--tenant-id", default=os.getenv("PSEUDO_TENANT_ID"))
    p_pseudo.add_argument("--types-include", default=os.getenv("PII_TYPES_INCLUDE"))
    p_pseudo.add_argument("--types-exclude", default=os.getenv("PII_TYPES_EXCLUDE"))
    p_pseudo.add_argument("--max-entities", type=int, default=int(os.getenv("PII_MAX_ENTITIES_PER_DOC", "0")))
    p_pseudo.add_argument("--require-annotations", choices=["true", "false"], default=os.getenv("PSEUDONYMIZER_REQUIRE_ANNOTATIONS", "true"))
    p_pseudo.add_argument("--force", action="store_true")

    # enrich-llm command
    p_llm = subparsers.add_parser("enrich-llm", help="Run Phase-3 LLM enrichment on Phase-2 JSONL (uses text_pseudo)")
    p_llm.add_argument("--input", "-i", type=Path, required=True, help="Input Phase-2 JSONL dir")
    p_llm.add_argument("--output", "-o", type=Path, required=True, help="Output Phase-3 dir")
    p_llm.add_argument("--model", required=True, help="LLM model identifier")
    p_llm.add_argument("--temperature", type=float, default=0.0)
    p_llm.add_argument("--max-tokens", dest="max_tokens", type=int, default=512)
    p_llm.add_argument("--max-tokens-input", dest="max_tokens_input", type=int, default=4096)
    p_llm.add_argument("--skip-if-present", choices=["true", "false"], default="true")
    p_llm.add_argument("--force", action="store_true", help="Force processing even if phase guard fails")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    setup_json_logging()

    if args.command == "preprocess":
        with span("phase1_preprocess"):
            with phase_duration.labels(phase="1").time():
                run_pipeline(args.input, args.output)
    elif args.command == "taxonomy-extract":
        ext = TaxonomyExtractor(out_path=args.out)
        files = (p for p in args.root.rglob("*.*") if p.is_file())
        ext.process_files(files)
    elif args.command == "taxonomy-analyze":
        analyze_taxonomy(preprocessed_dir=str(args.preprocessed), output_dir=str(args.out))
    elif args.command == "pseudonymize":
        if not _assert_phase1_dir(args.input) and not args.force:
            logger.error("Phase guard failed: input does not look like Phase-1 output. Use --force to override.")
            sys.exit(1)
        inc = [t.strip() for t in args.types_include.split(',')] if args.types_include else None
        exc = [t.strip() for t in args.types_exclude.split(',')] if args.types_exclude else None
        max_e = args.max_entities if args.max_entities and args.max_entities > 0 else None
        req = str(args.require_annotations).lower() == "true"
        with span("phase2_pseudonymize"):
            with phase_duration.labels(phase="2").time():
                run_pseudonymization(
                    args.input, args.output,
                    scope=args.scope,
                    tenant_id=args.tenant_id,
                    types_include=inc,
                    types_exclude=exc,
                    max_entities=max_e,
                    require_annotations=req,
                    force=args.force,
                )
    elif args.command == "enrich-llm":
        skip = str(args.skip_if_present).lower() == "true"
        if not _assert_phase2_dir(args.input) and not getattr(args, 'force', False):
            logger.error("Phase guard failed: input does not look like Phase-2 output. Use --force to override.")
            sys.exit(1)
        with span("phase3_enrich_llm", model=args.model):
            with phase_duration.labels(phase="3").time():
                run_enrich_llm(
                    args.input,
                    args.output,
                    model=args.model,
                    temperature=float(args.temperature),
                    max_tokens=int(args.max_tokens),
                    skip_if_present=skip,
                    max_tokens_input=int(args.max_tokens_input),
                )
if __name__ == "__main__":
    main()
