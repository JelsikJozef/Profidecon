# src/preprocessor/taxonomy/extractor.py
"""LLM-powered extractor using OpenAI gtp40-mini

• Prečíta súbor (path) a jeho text snippet.
• Volá OpenAI ChatCompletion API s modelom gtp-4o-mini.
• Získa JSON {typ_suboru, povaha_suboru, tagy, sumar}.
• Výstup zapisuje do metadata_raw.jsonl.
"""

import json
import logging
import os
from json.decoder import JSONDecodeError
import dotenv
from pathlib import Path
from typing import Iterator, Dict

import openai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Model in OpenAI: gtp-4o--mini
MODEL_NAME = "gpt-4o-mini"

# Ensure that OPENAI_API_KEY is set in the environment
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.warning("OPENAI_API_KEY not set. The extractor will not work without it.")

SYSTEM_PROMPT = (
    "You are a Slovak domain expert. "
    "Extract structured JSON metadata from the given file snippet. "
    "Return only a JSON object with keys: typ_suboru, tagy, sumar. "
    "Use Slovak language when describing the file's nature and tags. "
)


USER_PROMPT_TMPL = (
    "Názov súboru: {file_name}\n"
    "Ukážka textu (prvých 800 znakov):\n"
    """"\n{snippet}\n"""
)
class TaxonomyExtractor:
    """Iteruje vstupné súbory a vytvára surové LLM metadata pomocou OpenAI."""
    def __init__(self, model: str = MODEL_NAME, out_path: Path | None = None):
        self.model = model
        self.out_path = out_path or Path("metadata_raw.jsonl")
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def _call_llm(self, file_name: str, snippet: str) -> Dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TMPL.format(file_name=file_name, snippet=snippet)},
        ]
        try:
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.0,
            )
            if not resp or not resp.choices:
                logger.error("Empty or invalid response for %s: %s", file_name, resp)
                return {}

            content = resp.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            content = content.replace('```json', '').replace('```', '').strip()
            try:
                return json.loads(content)
            except JSONDecodeError:
                logger.error("Failed to parse JSON for %s: %s", file_name, content)
                return {}
        except Exception as e:
            logger.error("LLM call failed for %s: %s", file_name, e)
            return {}

    def process_files(self, paths: Iterator[Path]):
        with self.out_path.open("a", encoding="utf-8") as fout:
            for p in paths:
                try:
                    snippet = self._safe_read_snippet(p)
                    meta = self._call_llm(p.name, snippet)
                    record = {
                        "cesta_k_suboru": str(p),
                        "nazov_suboru": p.name,
                        **meta,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    logger.info("Extracted metadata for %s", p.name)
                except Exception:
                    logger.exception("Extractor failed for %s", p)

    @staticmethod
    def _safe_read_snippet(path: Path, n_chars: int = 800) -> str:
        try:
            raw = path.read_bytes()[:50000]
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            return ""
        return text[:n_chars]


# Example CLI usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Extract metadata for all files using OpenAI gtp40-mini")
    parser.add_argument("root", type=Path, help="Input root folder")
    parser.add_argument("--out", type=Path, default=Path("metadata_raw.jsonl"))
    args = parser.parse_args()

    extractor = TaxonomyExtractor(out_path=args.out)
    files = (p for p in args.root.rglob("*.*") if p.is_file())
    extractor.process_files(files)
