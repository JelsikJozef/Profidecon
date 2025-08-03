# src/preprocessor/taxonomy/analyzer.py
"""
Analyzer for taxonomy proposals

• Načíta metadata_raw.jsonl
• Extrahuje všetky tagy a sumar
• Použije sklearn na hierarchické clustering tagov
• Vytvorí predbežný strom kategórií
• Pošle strom LLM pre vyladenie (OpenAI gtp-4o-mini)
• Uloží finálnu taxonómiu do taxonomy.json
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import openai


load_dotenv()

logger = logging.getLogger(__name__)

MODEL_NAME = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are a Slovak domain expert. I will provide a list of document tags. "
    "Propose a hierarchical taxonomy in JSON with nested categories and sample tag mappings in Slovak language. "
    "Return only the JSON definition in Slovak language."
)


def load_raw_metadata(path: Path) -> List[Dict[str, Any]]:
    """Load raw metadata from a JSONL file.
    :param path: Path to the raw metadata file.
    :return: List of records, each a dictionary with metadata."""
    records = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning('Invalid JSON line, skipping')
    return records


def build_tag_corpus(records: List[Dict[str, Any]]):
    """Build a TF-IDF corpus from the tags in the records.
    :param records: List of records, each a dictionary with 'tagy' key.
    :return: Tuple of TF-IDF matrix and feature names."""
    # flatten all tags
    docs = [' '.join(rec.get('tagy', [])) for rec in records]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)
    return X, vectorizer.get_feature_names_out()


def cluster_tags(X, n_clusters: int = 5):
    """Cluster the tags using Agglomerative Clustering
    :param X: TF-IDF matrix of tags."""
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X.toarray())
    return labels


def propose_taxonomy(tags: List[str]) -> Dict[str, Any]:
    """Propose a taxonomy based on the unique tags using OpenAI LLM."""
    logger.info("Starting taxonomy proposal with %d tags", len(tags))
    prompt = (
            SYSTEM_PROMPT + "\nTags:\n" + json.dumps(tags, ensure_ascii=False)
    )
    logger.info("Using prompt: %s", prompt[:200] + "..." if len(prompt) > 200 else prompt)

    try:
        client = openai.OpenAI()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        logger.info("Sending request to OpenAI API with model: %s", MODEL_NAME)

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1000,  # Changed back to max_tokens for older models
            temperature=0.0,
        )

        if not resp or not resp.choices:
            logger.error("Empty response from OpenAI API")
            return {}

        content = resp.choices[0].message.content.strip()
        if not content:
            logger.error("Empty content in OpenAI response")
            return {}

        logger.info("Received response content (first 200 chars): %s",
                    content[:200] + "..." if len(content) > 200 else content)

        try:
            import re

            # Remove Markdown code block markers if present
            if content.startswith("```"):
                content = content.split('\n', 1)[1]
                content = content.rsplit('```', 1)[0]
                content = content.strip()

            # Replace single quotes with double quotes if needed
            content = re.sub(r"'", r'"', content)
            # Add missing double quotes around unquoted keys
            content = re.sub(r'([{,]\s*)([a-zA-Z_]\w*)(\s*:\s*)', r'\1"\2"\3', content)
            # Remove trailing commas after objects or arrays
            content = re.sub(r",(\s*[}\]])", r"\1", content)
            # Remove newline characters that might break strings (caution: ensure this does not remove valid escapes)
            content = re.sub(r'\\\n', '', content)

            result = json.loads(content)
            return result

        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON: %s", e)
            logger.error("Raw response content: %s", content)
            return {}

    except Exception as e:
        logger.error('OpenAI API call failed: %s', str(e))
        return {}


def save_taxonomy(tree: Dict[str, Any], out_path: Path):
    """Save the proposed taxonomy to a JSON file.
    :param tree: Proposed taxonomy as a dictionary.
    :param out_path: Path to save the taxonomy JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)


def main(root: Path, raw_meta: Path, out_tax: Path):
    """Main function to generate taxonomy from raw metadata.
    :param root: Root directory (not used here, but can be useful for logging).
    :param raw_meta: Path to the raw metadata JSONL file.
    :param out_tax: Path to save the proposed taxonomy JSON file."""
    # load
    records = load_raw_metadata(raw_meta)
    # build corpus
    X, feature_names = build_tag_corpus(records)
    # cluster documents into groups (by tags corpus)
    labels = cluster_tags(X, n_clusters=5)
    # collect unique tags
    unique_tags = sorted({tag for rec in records for tag in rec.get('tagy', [])})
    # propose taxonomy via LLM
    taxonomy = propose_taxonomy(unique_tags)
    save_taxonomy(taxonomy, out_tax)
    logger.info('Taxonomy saved to %s', out_tax)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Generate taxonomy from raw metadata')
    parser.add_argument('--raw', type=Path, default=Path('metadata_raw.jsonl'))
    parser.add_argument('--out', type=Path, default=Path('taxonomy.json'))
    args = parser.parse_args()
    main(Path('.'), args.raw, args.out)
