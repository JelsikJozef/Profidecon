# Profidecon Preprocessor

## Overview

The `preprocessor` module provides a robust, extensible pipeline for document ingestion, parsing, normalization, enrichment, deduplication, quality control, serialization, and taxonomy extraction/analysis. It is designed to prepare documents for downstream semantic search and retrieval-augmented generation (RAG) workflows.

---

## Features

- **Recursive Ingestion:** Scans directories for supported document types (`.pdf`, `.docx`, `.msg`, `.jpg`, `.jpeg`, `.png`).
- **Parsing:** Uses a registry of parsers for different file formats.
- **Image OCR:** Multi-language OCR with automatic rotation correction for images.
- **Normalization:** Cleans and standardizes parsed content.
- **OCR:** Applies OCR to PDFs if required.
- **Metadata Enrichment:** Adds additional metadata fields.
- **Deduplication:** Removes duplicate documents.
- **Quality Checking:** Ensures data quality and consistency.
- **Serialization:** Outputs processed documents as JSONL files.
- **Taxonomy Extraction:** Uses LLMs (OpenAI) to extract structured metadata for taxonomy.
- **Taxonomy Analysis:** Aggregates metadata into a hierarchical taxonomy.

---

## Image OCR Settings

The Image Parser (`ImageParser`) provides advanced OCR capabilities for image files with automatic rotation correction and multi-language support.

### Supported Formats
- `.jpg`, `.jpeg`, `.png`

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_LANGS` | `eng+slk+deu+ces+pol+spa+ita+fra+por+rus` | Languages for OCR (Tesseract format) |
| `OCR_MAX_DIMENSION` | `2000` | Maximum dimension for image resizing before OCR |

### Language Configuration

#### Static Language Set
```bash
export OCR_LANGS="eng+slk+deu"  # English, Slovak, German
```

#### Automatic Language Detection
```bash
export OCR_LANGS="auto"  # Enables two-step OCR with language detection
```

When `OCR_LANGS="auto"` is set:
1. **First pass**: OCR with conservative language set (`eng+slk`)
2. **Language detection**: Analyze extracted text to identify language
3. **Second pass**: OCR with extended language set based on detected language

### Automatic Rotation Correction

The Image Parser automatically corrects image orientation using:

1. **EXIF metadata**: If available, applies EXIF orientation correction first
2. **OSD (Orientation and Script Detection)**: If EXIF is unavailable, uses Tesseract OSD to detect rotation
3. **Caching**: OSD results are cached during processing session for performance

### Image Preprocessing

Before OCR, images are automatically preprocessed to improve accuracy:
- **Resizing**: Large images are scaled down (configurable via `OCR_MAX_DIMENSION`)
- **Grayscale conversion**: Converts to grayscale for better OCR
- **Contrast enhancement**: Increases contrast by 20%
- **Binarization**: Converts to black and white using threshold

### Metadata Fields

The Image Parser adds the following metadata to processed documents:

```json
{
  "source": "path/to/image.jpg",
  "ocr": true,
  "rotation_applied": 90,
  "rotation_source": "osd",
  "ocr_langs": ["eng+slk", "eng+slk+deu"],
  "image_size": "1920x1080",
  "image_mode": "RGB",
  "ocr_empty": false
}
```

### Tesseract Installation

The Image Parser requires Tesseract OCR with language packs:

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-slk tesseract-ocr-deu tesseract-ocr-ces tesseract-ocr-pol
```

#### macOS (Homebrew)
```bash
brew install tesseract
brew install tesseract-lang
```

#### Windows
1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install with additional language packs
3. Add Tesseract to PATH

#### Available Language Codes
- `eng` - English
- `slk` - Slovak
- `deu` - German
- `ces` - Czech
- `pol` - Polish
- `spa` - Spanish
- `ita` - Italian
- `fra` - French
- `por` - Portuguese
- `rus` - Russian
- `hun` - Hungarian
- `ron` - Romanian
- `ukr` - Ukrainian

### Performance Optimization

- **Image resizing**: Large images are automatically resized to improve processing speed
- **OSD caching**: Rotation detection results are cached during processing session
- **Fallback handling**: Multiple fallback levels ensure processing continues even if advanced features fail

### Example Usage

```bash
# Process images with default settings
python -m preprocessor.cli preprocess --input ./images --output ./processed

# Process with custom language set
export OCR_LANGS="eng+slk+deu"
python -m preprocessor.cli preprocess --input ./images --output ./processed

# Process with automatic language detection
export OCR_LANGS="auto"
python -m preprocessor.cli preprocess --input ./images --output ./processed

# Process with custom image size limit
export OCR_MAX_DIMENSION="1500"
python -m preprocessor.cli preprocess --input ./images --output ./processed
```

### Troubleshooting

**Empty OCR results**: Check that required Tesseract language packs are installed
**Rotation issues**: Verify image EXIF data or try different preprocessing settings
**Performance issues**: Reduce `OCR_MAX_DIMENSION` or limit language set
**Language detection errors**: Use static language configuration instead of "auto" mode

---

## CLI Usage

Run the CLI from the project root or the `preprocessor` directory:

```
python -m preprocessor.cli <command> [options]
```

### Commands

- `preprocess`  
  Run the full preprocessing pipeline.
  - `--input/-i <input_dir>`: Input folder with documents
  - `--output/-o <output_dir>`: Output folder for JSONL files

- `taxonomy-extract`  
  Extract metadata for taxonomy using OpenAI.
  - `<root>`: Input root folder for documents
  - `--out <file>`: Output metadata JSONL file (default: `metadata_raw.jsonl`)

- `taxonomy-analyze`  
  Generate taxonomy from preprocessed JSONL files.
  - `<root>`: Root folder path
  - `--preprocessed <dir>`: Directory with preprocessed JSONL files (default: `Preprocessed`)
  - `--out <file>`: Output taxonomy JSON file (default: `taxonomy.json`)

---

## Pipeline Stages

1. **Ingestion:**
   - Recursively finds documents in the input directory.
   - Supported types: PDF, DOCX, MSG.
2. **Parsing:**
   - Uses a parser registry to select the appropriate parser for each file type.
3. **Normalization:**
   - Cleans and standardizes the parsed content.
4. **OCR (if needed):**
   - Applies OCR to PDFs that require it.
5. **Enrichment:**
   - Adds metadata (e.g., file info, custom fields).
6. **Deduplication:**
   - Removes duplicate documents based on content or metadata.
7. **Quality Checking:**
   - Ensures the document meets quality standards.
8. **Serialization:**
   - Writes the processed document to a JSONL file in the output directory.
9. **Taxonomy Extraction:**
   - Uses OpenAI LLM to extract structured metadata (type, tags, summary) from file snippets.
10. **Taxonomy Analysis:**
    - Aggregates metadata from all documents to build a hierarchical taxonomy.

---

## Extending the Pipeline

- **Add a new parser:**
  - Implement a new parser class and register it in `parsers/registry.py`.
- **Add a new processor:**
  - Implement the processor and add it to the pipeline in `cli.py`.
- **Customize enrichment or quality checks:**
  - Edit `enricher.py` or `quality_checker.py` in `processors/`.

---

## Developer Notes

- The pipeline is modular; each stage can be extended or replaced.
- Logging is enabled for all major steps.
- Taxonomy extraction requires a valid OpenAI API key in your environment.

---

## Example

```
python -m preprocessor.cli preprocess --input ./Knowledge --output ./Preprocessed
python -m preprocessor.cli taxonomy-extract ./Knowledge --out metadata_raw.jsonl
python -m preprocessor.cli taxonomy-analyze ./Knowledge --preprocessed ./Preprocessed --out taxonomy.json
```

---

## File Structure

- `cli.py` – Main CLI entry point
- `ingestion.py` – File system ingestion logic
- `models.py` – Data models
- `parsers/` – File parsers for different formats
- `processors/` – Pipeline processors (normalizer, OCR, enricher, deduplicator, quality checker, serializer)
- `taxonomy/` – Taxonomy extractor and analyzer

---

## License

See project root for license information.

# Enrichment s LLM (OpenAI alebo Ollama)

Modul `enricher.py` využíva triedu `LLMPicker` (`llm_picker.py`) na generovanie summary a tagov buď cez OpenAI API (default), alebo cez lokálny Ollama server s modelom gpt-oss-20b.

### Výber backendu

- **OpenAI** (default): nastavte `OPENAI_API_KEY` v prostredí.
- **Ollama**: nastavte `llm_backend="ollama"` pri vytváraní `Enricher` a spustite Ollama server s požadovaným modelom.

#### Príklad použitia:

```python
from Main_programme.preprocessor.processors.enricher import Enricher

# OpenAI backend (default)
enricher = Enricher(root_path, llm_backend="openai")

# Ollama backend (gpt-oss-20b)
enricher = Enricher(root_path, llm_backend="ollama")
```

#### Priame použitie LLMPicker

```python
from Main_programme.preprocessor.processors.llm_picker import LLMPicker

llm = LLMPicker(backend="ollama")  # alebo "openai"
summary, tags = llm.generate_summary_and_tags("Váš text...")
```

#### Konfigurácia pre Ollama

- Spustite Ollama server:  
  `ollama run gpt-oss-20b`
- Voliteľne nastavte premenné prostredia:
  - `OLLAMA_URL` (default: `http://localhost:11434`)
  - `OLLAMA_MODEL` (default: `gpt-oss-20b`)

#### Poznámka

Ak LLM nie je dostupné alebo dôjde k chybe, enrichment vráti prázdny summary a tagy.

---

## Enrichment Architecture

The enrichment stage has been refactored into two specialized modules for better separation of concerns and maintainability:

### Metadata Enricher (`enricher_metadata.py`)

Handles **non-LLM enrichment tasks** that are fast, deterministic, and don't require network calls:

- **Text Statistics**: Character count, word count, token estimation, sentence count, paragraph count
- **Language Detection**: Automatic language identification using `langdetect`
- **Content Analysis**: SHA-1 hashing for deduplication, PII risk scoring
- **File Statistics**: File size, creation time, modification time
- **Category Extraction**: Document categorization based on directory structure

**Key Features:**
- No LLM dependencies or network calls
- Fast execution suitable for high-volume processing
- Deterministic results for consistent metadata
- Comprehensive text analysis metrics

**Usage:**
```python
from Main_programme.preprocessor.processors.enricher_metadata import MetadataEnricher

enricher = MetadataEnricher(root_path=Path("/path/to/documents"))
enriched_doc = enricher.run(document_dict)

# Or use stateless function
from Main_programme.preprocessor.processors.enricher_metadata import run
enriched_doc = run(document_dict, root_path=Path("/path/to/documents"))
```

### LLM Enricher (`enricher_llm.py`)

Handles **AI-based enrichment tasks** that require language models:

- **Summary Generation**: Creates concise document summaries
- **Tag Extraction**: Identifies semantic tags and keywords
- **Semantic Analysis**: Any other AI-powered content analysis

**Key Features:**
- Supports multiple LLM backends (OpenAI, Ollama, HuggingFace)
- Robust error handling with graceful fallbacks
- Configurable backend selection
- Mock-friendly design for testing

**Usage:**
```python
from Main_programme.preprocessor.processors.enricher_llm import LLMEnricher

enricher = LLMEnricher(llm_backend="huggingface")
enriched_doc = enricher.run(document_dict)

# Or use stateless function
from Main_programme.preprocessor.processors.enricher_llm import run
enriched_doc = run(document_dict, llm_backend="openai")
```

### Unified Enricher (Backward Compatibility)

The original `Enricher` class in `enricher.py` remains unchanged for backward compatibility. It now orchestrates both metadata and LLM enrichment internally:

```python
from Main_programme.preprocessor.processors.enricher import Enricher

# Works exactly as before
enricher = Enricher(root_path=input_dir, llm_backend="ollama")
enriched_doc = enricher.enrich(parsed_document)
```

**Benefits of the Split:**
- **Performance**: Metadata enrichment can run independently without LLM overhead
- **Reliability**: Non-LLM tasks won't fail due to network issues or model unavailability
- **Testing**: Each module can be tested in isolation with appropriate mocking
- **Scalability**: Metadata processing can be parallelized more effectively
- **Maintainability**: Clear separation of concerns makes code easier to understand and extend

---

## PII Analyzer

The PII Analyzer provides robust, configurable detection of personally identifiable information (PII) with pluggable backends. It returns typed entities with character spans and confidence scores without altering the input text.

### Supported Entity Types

- **EMAIL**: Email addresses
- **PHONE**: Phone numbers (international and local formats)
- **PERSON_NAME**: Names with diacritics support (multilingual)
- **IBAN**: International Bank Account Numbers
- **PASSPORT**: Passport-like identifiers
- **ID_NUMBER**: Generic ID patterns
- **ADDRESS**: Address-like patterns
- **CREDIT_CARD**: Credit card numbers
- **DATE_OF_BIRTH**: Date patterns that might be birth dates
- **URL**: Web URLs
- **ORG**: Organizations (Presidio backend only)

### Detection Backends

#### Regex Backend (Default)
- **No external dependencies**: Pure regex patterns
- **Multilingual support**: Unicode-aware patterns with diacritics
- **Performance optimized**: Pre-compiled patterns
- **Context-aware scoring**: Business ID detection to reduce false positives
- **Configurable confidence**: Pattern strength + length + context hints

#### Presidio Backend (Optional)
- **Microsoft Presidio integration**: Advanced NLP-based detection
- **Graceful fallback**: Falls back to regex if Presidio unavailable
- **Entity type mapping**: Maps Presidio types to canonical types
- **Multi-language support**: Where supported by Presidio

### Configuration

Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PII_BACKEND` | `regex` | Backend type (`regex` or `presidio`) |
| `PII_LANGS` | `auto` | Language hints (`auto`, `en,sk,de,...`) |
| `PII_RETURN_VALUES` | `true` | Include raw PII values in results |
| `PII_MIN_CONFIDENCE` | `0.60` | Minimum confidence threshold |
| `PII_TYPES_INCLUDE` | - | Comma-separated allowed types |
| `PII_TYPES_EXCLUDE` | - | Comma-separated excluded types |
| `PII_MAX_ENTITIES_PER_DOC` | `1000` | Safety limit per document |

### Usage Examples

#### Basic Detection
```python
from Main_programme.preprocessor.processors.pii_analyzer import PiiAnalyzer

analyzer = PiiAnalyzer()
entities = analyzer.detect("Contact John at john@example.com or +1-555-123-4567")

for entity in entities:
    print(f"{entity['type']}: {entity['value']} (confidence: {entity['confidence']:.2f})")
```

#### Custom Configuration
```python
from Main_programme.preprocessor.processors.pii_analyzer import PiiAnalyzer

# High-security configuration
analyzer = PiiAnalyzer(
    backend="presidio",
    min_confidence=0.80,
    return_values=False,  # Don't include raw values
    types_exclude={"DATE_OF_BIRTH", "URL"}
)

entities = analyzer.detect(text, locale="sk")
```

#### Environment Configuration
```bash
export PII_BACKEND=presidio
export PII_MIN_CONFIDENCE=0.80
export PII_TYPES_INCLUDE=EMAIL,PHONE,PERSON_NAME
export PII_RETURN_VALUES=false

# Then use with default configuration
python -m preprocessor.cli preprocess --input ./docs --output ./processed
```

### Entity Structure

Each detected entity includes:

```python
{
    "type": "EMAIL",                    # Entity type
    "start": 14,                       # Character start position
    "end": 33,                         # Character end position
    "value": "john@example.com",       # Raw value (optional)
    "confidence": 0.95,               # Confidence score (0-1)
    "pattern": "EMAIL",               # Pattern/rule identifier
    "locale": "en"                    # Locale hint (optional)
}
```

### Integration with Pipeline

The PII analyzer is automatically integrated into the preprocessing pipeline:

1. **Detection Phase**: Runs after metadata enrichment, before deduplication
2. **Storage**: PII entities are stored in document metadata as `pii_entities`
3. **Logging**: Entity counts and types are logged for monitoring
4. **No Text Modification**: Original text remains unchanged (detection only)

### Performance

- **Regex Backend**: ≤200ms for 50KB documents
- **Memory Efficient**: Streaming detection for large documents
- **Deterministic**: Same input produces identical results
- **Safe Limits**: Configurable entity count limits prevent memory issues

### Overlap Resolution

When multiple patterns match overlapping text regions:

1. **Prefer Higher Confidence**: More confident matches take precedence
2. **Prefer Specificity**: EMAIL > PHONE > PERSON_NAME > ID_NUMBER > URL
3. **Deterministic Ordering**: Consistent results across runs

### Multilingual Support

- **Slovak**: Full diacritics support (á, č, ď, é, ě, í, ľ, ĺ, ň, ó, ô, ŕ, š, ť, ú, ů, ý, ž)
- **German**: Umlauts and ß character support
- **Czech**: Complete diacritics coverage
- **English**: Standard ASCII and extended characters
- **Context Aware**: Reduces false positives in business documents

### Installation Requirements

#### Regex Backend (Default)
No additional dependencies required.

#### Presidio Backend (Optional)
```bash
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_sm
```

### Testing

Comprehensive test coverage includes:
- Multi-language detection accuracy
- Performance benchmarks
- Edge case handling
- Configuration validation
- Backend parity testing

```bash
python -m pytest Helping_algorithms/tests/test_pii_analyzer.py -v
```

---

## Phase-2: Pseudonymization

Deterministic, reversible pseudonymization replaces PII spans with typed display tokens (e.g., [EMAIL:K7V2WQ3M]) using the Token Vault. Plaintext PII is never written to Phase‑2 outputs.

Key properties:
- Deterministic for identical (value, type, scope, tenant_id)
- Overlap-safe (prefers longer spans; applied with stable indices)
- Unicode-safe (uses Python codepoint indices)
- No plaintext leakage in logs or outputs (only token IDs and counts)

Environment:
- PSEUDO_SCOPE=tenant|global (default: tenant)
- PSEUDONYMIZER_REQUIRE_ANNOTATIONS=true|false (default: true)
- PII_TYPES_INCLUDE, PII_TYPES_EXCLUDE (comma-separated)
- PII_MAX_ENTITIES_PER_DOC (caps unique values per doc)
- Token Vault env from Prompt #4 (DATABASE_URL, HMAC/KEK/SALT, TOKEN_ID_BYTES)

CLI usage (Click, recommended):
- profidecon pseudonymize \
  --input <phase1_dir> \
  --output <phase2_dir> \
  --scope tenant|global \
  --tenant-id <id> \
  [--types-include EMAIL,PHONE] \
  [--types-exclude URL] \
  [--max-entities 500] \
  [--require-annotations true|false] \
  [--force]

Legacy CLI (argparse):
- python -m Main_programme.preprocessor.cli pseudonymize \
  --input <phase1_dir> --output <phase2_dir> [same flags]

Phase‑2 output fields:
- text_pseudo: pseudonymized text
- metadata.pseudonymized: true
- metadata.pseudonymization: { scope, tenant_id, counts }
- metadata.token_spans: list of TokenSpan mappings
- metadata.pii_entities: original spans with value removed (None)

Notes:
- Idempotent: already pseudonymized docs are skipped unless --force
- On missing annotations and --require-annotations=false, PII detection is run on the fly
- Fail-fast on vault errors with non-zero exit code
