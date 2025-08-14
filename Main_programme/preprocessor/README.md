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
