#!/usr/bin/env python3

"""
Profidecon CLI - Command Line Interface for Document Processing and Analysis
This CLI provides commands for preprocessing documents, extracting and analyzing taxonomies,
and loading vector embeddings into Qdrant for semantic search.
It serves as a unified interface for various document processing tasks, including:
- Preprocessing documents into JSONL format
- Extracting metadata and taxonomies from documents
- Analyzing taxonomies to create hierarchical structures
- Loading vector embeddings into Qdrant for semantic search capabilities
"""

import click
import logging
import sys
import os
from pathlib import Path
from typing import Optional

# Import existing CLI functions
from Main_programme.preprocessor.cli import run_pipeline
from Main_programme.preprocessor.taxonomy.extractor import TaxonomyExtractor
from Main_programme.preprocessor.taxonomy.analyzer import main as analyze_taxonomy
from Main_programme.vectorizer import load_folder
from Main_programme.preprocessor.processors.pseudonymizer import Pseudonymizer
from Main_programme.preprocessor.processors.pii_analyzer import PiiAnalyzer
from Main_programme.preprocessor.cli import run_enrich_llm, run_pseudonymization
from Main_programme.preprocessor.middleware.response_deanonymizer import ResponseDeanonymizer, DefaultDeanonymizationPolicy
import json
import time


# Global options that apply to all commands
@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def profidecon(ctx, verbose: bool, config: Optional[str]):
    """
    Profidecon CLI - Unified command line interface for document processing and analysis.
    This tool provides commands for preprocessing documents, extracting and analyzing taxonomies,
    and loading vector embeddings into Qdrant for semantic search.
    :param ctx: Click context object to pass options between commands.
    :param verbose: Enable verbose logging for debugging.
    :param config: Path to configuration file for additional settings.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Set up logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Store global options in context
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config

    logger = logging.getLogger(__name__)
    if verbose:
        logger.debug("Verbose logging enabled")
    if config:
        logger.info(f"Using config file: {config}")


@profidecon.command()
@click.option('--input', '-i', type=click.Path(exists=True, file_okay=False, path_type=Path),
              required=True, help='Input folder with documents')
@click.option('--output', '-o', type=click.Path(file_okay=False, path_type=Path),
              required=True, help='Output folder for JSONL files')
@click.pass_context
def preprocess(ctx, input: Path, output: Path):
    """
    Preprocess documents in the specified input directory.
    :param input: Path to the input directory containing documents.
    :param output: Path to the output directory where preprocessed JSONL files will be saved
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting preprocessing pipeline: {input} → {output}")

    try:
        # Create output directory if it doesn't exist
        output.mkdir(parents=True, exist_ok=True)

        # Run the existing preprocessing pipeline
        run_pipeline(input, output)

        logger.info("✅ Preprocessing pipeline completed successfully")

    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        if ctx.obj.get('verbose'):
            logger.exception("Full traceback:")
        sys.exit(1)


@profidecon.command('taxonomy-extract')
@click.argument('root', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--out', type=click.Path(path_type=Path), default=Path("metadata_raw.jsonl"),
              help='Output metadata JSONL file')
@click.pass_context
def taxonomy_extract(ctx, root: Path, out: Path):
    """
    Extract taxonomy from documents in the specified directory.
    This command scans the directory for files, extracts metadata, and saves it in a JSONL format.
    :param root: Path to the directory containing documents.
    :param out: Path to the output JSONL file where extracted metadata will be saved.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting taxonomy extraction from: {root}")

    try:
        extractor = TaxonomyExtractor(out_path=out)
        files = (p for p in root.rglob("*.*") if p.is_file())
        extractor.process_files(files)

        logger.info(f"✅ Taxonomy extraction completed: {out}")

    except Exception as e:
        logger.error(f"❌ Taxonomy extraction failed: {e}")
        if ctx.obj.get('verbose'):
            logger.exception("Full traceback:")
        sys.exit(1)


@profidecon.command('taxonomy-analyze')
@click.argument('root', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--preprocessed', type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=Path("../../Preprocessed"), help='Path to directory with preprocessed JSONL files')
@click.option('--out', type=click.Path(path_type=Path), default=Path("taxonomy.json"),
              help='Output taxonomy JSON file')
@click.pass_context
def taxonomy_analyze(ctx, root: Path, preprocessed: Path, out: Path):
    """
    Generate taxonomy from preprocessed JSONL files.

    Analyzes preprocessed documents to create a hierarchical taxonomy
    structure suitable for RAG systems, including country categorization.
    :param root: Path to the root directory containing documents.
    :param preprocessed: Path to the directory with preprocessed JSONL files.
    :param out: Path to the output JSON file where the taxonomy will be saved.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting taxonomy analysis from: {preprocessed}")

    try:
        analyze_taxonomy(preprocessed_dir=str(preprocessed), output_dir=str(out))

        logger.info(f"✅ Taxonomy analysis completed: {out}")

    except Exception as e:
        logger.error(f"❌ Taxonomy analysis failed: {e}")
        if ctx.obj.get('verbose'):
            logger.exception("Full traceback:")
        sys.exit(1)


@profidecon.command('vector-load')
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--glob', default="*.jsonl", help='Glob pattern for files to process')
@click.pass_context
def vector_load(ctx, input_dir: Path, glob: str):
    """
    Load vector embeddings from preprocessed JSONL files into Qdrant.
    This command reads JSONL files containing document chunks, generates embeddings,
    and uploads them to a Qdrant collection for semantic search.
    :param input_dir: Path to the directory containing preprocessed JSONL files.
    :param glob: Glob pattern to match files (default: *.jsonl).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting vector loading from: {input_dir} (pattern: {glob})")

    try:
        load_folder(input_dir, glob)

        logger.info("✅ Vector loading completed successfully")

    except Exception as e:
        logger.error(f"❌ Vector loading failed: {e}")
        if ctx.obj.get('verbose'):
            logger.exception("Full traceback:")
        sys.exit(1)


@profidecon.command('pseudonymize')
@click.option('--input', '-i', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help='Input folder with Phase-1 JSONL files')
@click.option('--output', '-o', type=click.Path(file_okay=False, path_type=Path), required=True,
              help='Output folder for Phase-2 pseudonymized JSONL files')
@click.option('--scope', type=click.Choice(['tenant', 'global']), default=lambda: os.getenv('PSEUDO_SCOPE', 'tenant'),
              show_default=True, help='Token scope')
@click.option('--tenant-id', type=str, default=lambda: os.getenv('PSEUDO_TENANT_ID', None), help='Tenant identifier')
@click.option('--types-include', type=str, default=os.getenv('PII_TYPES_INCLUDE', None),
              help='Comma-separated PII types to include')
@click.option('--types-exclude', type=str, default=os.getenv('PII_TYPES_EXCLUDE', None),
              help='Comma-separated PII types to exclude')
@click.option('--max-entities', type=int, default=lambda: int(os.getenv('PII_MAX_ENTITIES_PER_DOC', '0')),
              help='Maximum entities per document (0 means no limit)')
@click.option('--require-annotations', type=bool, default=lambda: os.getenv('PSEUDONYMIZER_REQUIRE_ANNOTATIONS', 'true').lower() == 'true',
              help='Require Phase-1 PII annotations; otherwise detect on the fly')
@click.option('--force', is_flag=True, help='Re-run even if already pseudonymized')
@click.pass_context
def pseudonymize(ctx, input: Path, output: Path, scope: str, tenant_id: Optional[str], types_include: Optional[str],
                 types_exclude: Optional[str], max_entities: int, require_annotations: bool, force: bool):
    """Phase-2: Pseudonymize Phase-1 docs using Token Vault without storing plaintext."""
    logger = logging.getLogger(__name__)
    start_ts = time.time()
    output.mkdir(parents=True, exist_ok=True)

    # Prepare filters
    include_list = [t.strip() for t in types_include.split(',')] if types_include else None
    exclude_list = [t.strip() for t in types_exclude.split(',')] if types_exclude else None
    max_entities_val = max_entities if max_entities and max_entities > 0 else None

    # Init processors
    pseudo = Pseudonymizer(scope=scope, tenant_id=tenant_id)
    analyzer = None if require_annotations else PiiAnalyzer()

    errors = 0
    processed = 0
    skipped = 0

    def _sanitize_entities(ents: list[dict]) -> list[dict]:
        safe: list[dict] = []
        for e in ents or []:
            ee = dict(e)
            if 'value' in ee:
                ee['value'] = None
            safe.append({k: ee[k] for k in ee.keys()})
        return safe

    for path in sorted(input.glob('*.jsonl')):
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
            logger.info(f"Skipping empty text in {path.name}")
            skipped += 1
            continue

        # Load entities from Phase-1 if present
        ents = data.get('pii_entities') or meta.get('pii_entities')
        if not ents:
            if require_annotations:
                logger.error(f"Missing pii_entities for {path.name} and require-annotations is True")
                errors += 1
                continue
            else:
                # Detect on the fly
                ents = analyzer.detect(text)

        try:
            result = pseudo.run(
                text=text,
                entities=ents,
                types_include=include_list,
                types_exclude=exclude_list,
                max_entities=max_entities_val,
            )
        except Exception as e:
            logger.error(f"Pseudonymization failed for {path.name}: {e}")
            errors += 1
            continue

        # Build output document without plaintext PII
        out_doc = {}
        # Keep non-sensitive top-level fields except original text
        for k, v in data.items():
            if k == 'text':
                continue
            if k == 'pii_entities':
                continue
            out_doc[k] = v

        # Ensure metadata exists
        out_meta = out_doc.get('metadata') if isinstance(out_doc.get('metadata'), dict) else {}
        out_meta['pseudonymized'] = True
        out_meta['pseudonymization'] = {
            'scope': scope,
            'tenant_id': tenant_id,
            'counts': dict(result['stats'])
        }
        out_meta['token_spans'] = list(result['spans'])
        out_meta['pii_entities'] = _sanitize_entities(ents)
        out_doc['metadata'] = out_meta

        # Pseudonymized text payload
        out_doc['text_pseudo'] = result['text']

        # Persist with same filename
        out_path = output / path.name
        try:
            with out_path.open('w', encoding='utf-8') as f:
                json.dump(out_doc, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to write {out_path.name}: {e}")
            errors += 1
            continue

        processed += 1
        logger.info(f"Pseudonymized {path.name} (entities: {sum(result['stats'].values())})")

    dt = (time.time() - start_ts) * 1000.0
    logger.info(f"Phase-2 pseudonymization completed: processed={processed}, skipped={skipped}, errors={errors}, time_ms={int(dt)}")

    if errors > 0:
        sys.exit(1)


@profidecon.command('enrich-llm')
@click.option('--input', '-i', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help='Input Phase-2 directory with JSONL files')
@click.option('--output', '-o', type=click.Path(file_okay=False, path_type=Path), required=True,
              help='Output Phase-3 directory')
@click.option('--model', required=True, type=str, help='LLM model ID')
@click.option('--temperature', type=float, default=0.0, show_default=True)
@click.option('--max-tokens', 'max_tokens', type=int, default=512, show_default=True)
@click.option('--max-tokens-input', 'max_tokens_input', type=int, default=4096, show_default=True)
@click.option('--skip-if-present', type=bool, default=True, show_default=True, help='Skip docs already enriched')
@click.pass_context
def enrich_llm(ctx, input: Path, output: Path, model: str, temperature: float, max_tokens: int, max_tokens_input: int, skip_if_present: bool):
    """Phase-3: Enrich pseudonymized documents with LLM (uses text_pseudo only)."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Phase-3 LLM enrichment: {input} → {output}")
    try:
        run_enrich_llm(
            input,
            output,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            skip_if_present=skip_if_present,
            max_tokens_input=max_tokens_input,
        )
        logger.info("✅ LLM enrichment completed successfully")
    except SystemExit as e:  # propagate non-zero exit on errors
        raise
    except Exception as e:
        logger.error(f"❌ LLM enrichment failed: {e}")
        if ctx.obj.get('verbose'):
            logger.exception("Full traceback:")
        sys.exit(1)


@profidecon.command()
@click.pass_context
def version(ctx):
    """Show version information."""
    click.echo("Profidecon v0.1.0")
    click.echo("Document processing pipeline for RAG systems")


# Add a convenience command to run the full pipeline
@profidecon.command('full-pipeline')
@click.option('--input', '-i', type=click.Path(exists=True, file_okay=False, path_type=Path),
              required=True, help='Input folder with documents')
@click.option('--preprocessed', type=click.Path(file_okay=False, path_type=Path),
              default=Path("../../Preprocessed"), help='Preprocessed output folder')
@click.option('--taxonomy-out', type=click.Path(path_type=Path),
              default=Path("taxonomy.json"), help='Taxonomy output file')
@click.option('--skip-vectors', is_flag=True, help='Skip vector loading step')
@click.pass_context
def full_pipeline(ctx, input: Path, preprocessed: Path, taxonomy_out: Path, skip_vectors: bool):
    """
    Run the complete processing pipeline.

    Executes preprocessing, taxonomy analysis, and vector loading in sequence.
    Use --skip-vectors to avoid loading vectors to Qdrant.
    """
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting full pipeline execution")

    try:
        # Step 1: Preprocess documents
        logger.info("📋 Step 1/3: Preprocessing documents...")
        ctx.invoke(preprocess, input=input, output=preprocessed)

        # Step 2: Generate taxonomy
        logger.info("🏷️  Step 2/3: Generating taxonomy...")
        ctx.invoke(taxonomy_analyze, root=input, preprocessed=preprocessed, out=taxonomy_out)

        # Step 3: Load vectors (optional)
        if not skip_vectors:
            logger.info("🔍 Step 3/3: Loading vectors...")
            ctx.invoke(vector_load, input_dir=preprocessed, glob="*.jsonl")
        else:
            logger.info("⏭️  Step 3/3: Skipping vector loading")

        logger.info("🎉 Full pipeline completed successfully!")

    except Exception as e:
        logger.error(f"❌ Full pipeline failed: {e}")
        if ctx.obj.get('verbose'):
            logger.exception("Full traceback:")
        sys.exit(1)


@profidecon.command('dev-deanonymize')
@click.option('--input', '-i', type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help='Input Phase-3 JSONL file (must contain text_pseudo)')
@click.option('--tenant-id', required=True, type=str, help='Tenant ID for resolution scope')
@click.option('--role', 'roles', multiple=True, type=str, help='Actor role(s); can be passed multiple times')
@click.option('--device', type=str, default='edge_trusted', show_default=True, help='Device trust level')
@click.option('--scope', type=click.Choice(['tenant','global']), default='tenant', show_default=True)
@click.pass_context
def dev_deanonymize(ctx, input: Path, tenant_id: str, roles: tuple[str, ...], device: str, scope: str):
    """DEV-ONLY: Preview de-anonymized text on screen with masking. No persistence."""
    logger = logging.getLogger(__name__)
    if os.getenv('ALLOW_DEV_DEANON', 'false').lower() != 'true':
        click.echo('\n' + '='*80)
        click.echo('DEV-ONLY DE-ANONYMIZATION IS DISABLED. Set ALLOW_DEV_DEANON=true to enable.')
        click.echo('='*80 + '\n')
        sys.exit(1)

    # Force masked return even if code is misused on server
    os.environ['DEANON_PERSIST_SERVER'] = 'true'

    try:
        raw = input.read_text(encoding='utf-8').splitlines()[0]
        data = json.loads(raw)
    except Exception as e:
        logger.error(f"Failed to read {input.name}: {e}")
        sys.exit(1)

    text_pseudo = data.get('text_pseudo') or ''
    if not text_pseudo:
        click.echo('Input file does not contain text_pseudo; nothing to de-anonymize.')
        sys.exit(1)

    class _Actor:
        def __init__(self, tenant_id: str, roles: list[str], device: str):
            self.tenant_id = tenant_id
            self.roles = roles
            self.device_trust_level = device
            self.request_id = f"dev-{int(time.time()*1000)}"

    actor = _Actor(tenant_id=tenant_id, roles=list(roles) if roles else ['case_handler'], device=device)
    de = ResponseDeanonymizer(DefaultDeanonymizationPolicy())
    preview = de.run(text_pseudo, actor=actor, scope=scope)

    banner = ('\n' + '!'*80 + '\n' +
              'DEV-ONLY, DO NOT USE IN PROD — DE-ANONYMIZATION PREVIEW (MASKED)\n' +
              '!'*80 + '\n')
    click.echo(banner)
    click.echo(preview)
    click.echo(banner)


@profidecon.command('easy-run')
@click.option('--input', '-i', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help='Input root with documents (e.g., Knowledge/Nemecke_DPH)')
@click.option('--run-dir', '-r', type=click.Path(file_okay=False, path_type=Path),
              help='Run directory to store phase outputs (default: Runs/<input_name>)')
@click.option('--tenant-id', type=str, default=None, help='Tenant ID used for pseudonymization')
@click.option('--scope', type=click.Choice(['tenant','global']), default='tenant', show_default=True)
@click.option('--model', type=str, default='gpt-4o-mini', help='LLM model for Phase-3 (ignored if phase 3 skipped)')
@click.option('--phases', type=str, default='1,2,3', help='Comma-separated phases to run, e.g., 1,2 or 2,3')
@click.pass_context
def easy_run(ctx, input: Path, run_dir: Path | None, tenant_id: str | None, scope: str, model: str, phases: str):
    """Run phases 1–3 with a single command and standard folder layout."""
    logger = logging.getLogger(__name__)
    base = run_dir or Path('Runs') / input.name
    p1 = base / 'phase1'
    p2 = base / 'phase2'
    p3 = base / 'phase3'
    for d in (base, p1, p2, p3):
        d.mkdir(parents=True, exist_ok=True)

    selected = {p.strip() for p in (phases.split(',') if phases else [])}
    if not selected:
        selected = {'1','2','3'}

    logger.info(f"Easy-run starting: input={input} run_dir={base} phases={sorted(selected)}")

    # Phase-1: preprocess
    if '1' in selected:
        logger.info(f"[Phase-1] Writing to {p1}")
        try:
            run_pipeline(input, p1)
        except SystemExit:
            raise
        except Exception as e:
            logger.error(f"Phase-1 failed: {e}")
            sys.exit(1)

    # Phase-2: pseudonymize
    if '2' in selected:
        logger.info(f"[Phase-2] Input={p1} → Output={p2}")
        try:
            run_pseudonymization(
                input_dir=p1,
                output_dir=p2,
                scope=scope,
                tenant_id=tenant_id,
                types_include=None,
                types_exclude=None,
                max_entities=None,
                require_annotations=False,
                force=True,
            )
        except SystemExit:
            raise
        except Exception as e:
            logger.error(f"Phase-2 failed: {e}")
            sys.exit(1)

    # Phase-3: enrich-llm
    if '3' in selected:
        logger.info(f"[Phase-3] Input={p2} → Output={p3} model={model}")
        try:
            run_enrich_llm(
                input_dir=p2,
                output_dir=p3,
                model=model,
                temperature=0.0,
                max_tokens=512,
                skip_if_present=True,
                max_tokens_input=4096,
            )
        except SystemExit:
            raise
        except Exception as e:
            logger.error(f"Phase-3 failed: {e}")
            sys.exit(1)

    logger.info("Easy-run complete")

