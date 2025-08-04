# src/preprocessor/taxonomy/analyzer.py
"""
Analyzer for taxonomy proposals

• Načíta všetky JSONL súbory z adresára Preprocessed
• Extrahuje všetky tagy a sumar
• Použije sklearn na hierarchické clustering tagov
• Vytvorí predbežný strom kategórií
• Pošle strom LLM pre vyladenie (OpenAI gpt-4o)
• Uloží finálnu taxonómiu do taxonomy.json
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Iterator
from dotenv import load_dotenv
import glob

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import openai


load_dotenv()

logger = logging.getLogger(__name__)

# Aktualizovaný model na gpt-4o pre lepšiu kvalitu taxonómie
MODEL_NAME = "gpt-4o"
SYSTEM_PROMPT = (
    "You are a Slovak domain expert specializing in document classification and information architecture. "
    "I will provide a list of document tags extracted from legal and administrative documents. "
    "Propose a hierarchical taxonomy in JSON with nested categories and sample tag mappings in Slovak language. "
    "The taxonomy should be optimized for RAG (Retrieval Augmented Generation) systems. "
    "Focus on creating logical groupings with no more than 3 levels of hierarchy. "
    "Return only the JSON definition in Slovak language with this structure:"
    """
    {
      "kategorie": [
        {
          "nazov": "Hlavná kategória 1",
          "popis": "Krátky popis kategórie",
          "podkategorie": [
            {
              "nazov": "Podkategória 1.1",
              "popis": "Krátky popis podkategórie",
              "tagy": ["tag1", "tag2", "tag3"]
            }
          ]
        }
      ]
    }
    """
)


def load_preprocessed_jsonl(preprocessed_dir: Path) -> List[Dict[str, Any]]:
    """Load all preprocessed JSONL files from a directory.
    :param preprocessed_dir: Path to the directory with preprocessed JSONL files.
    :return: List of records, each a dictionary with metadata."""
    records = []
    jsonl_files = preprocessed_dir.glob("*.jsonl")
    total_files = sum(1 for _ in preprocessed_dir.glob("*.jsonl"))
    processed_files = 0

    logger.info(f"Found {total_files} JSONL files to process")

    for jsonl_path in jsonl_files:
        try:
            with jsonl_path.open('r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    records.append(data)
                    processed_files += 1
                    if processed_files % 100 == 0:
                        logger.info(f"Processed {processed_files}/{total_files} files")
                except json.JSONDecodeError:
                    logger.warning(f'Invalid JSON in file {jsonl_path}, skipping')
        except Exception as e:
            logger.error(f"Error processing file {jsonl_path}: {e}")

    logger.info(f"Successfully processed {processed_files} out of {total_files} files")
    return records


def collect_tags_and_categories(records: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Collect unique tags and categories from the records.
    :param records: List of records, each a dictionary with 'tags' and 'category' keys.
    :return: Dictionary with 'tags' and 'categories' keys, each containing a set of unique values."""
    unique_tags = set()
    unique_categories = set()

    for rec in records:
        if 'tags' in rec and isinstance(rec['tags'], list):
            unique_tags.update(rec['tags'])
        # Fallback to 'tagy' if 'tags' is not present
        elif 'tagy' in rec and isinstance(rec['tagy'], list):
            unique_tags.update(rec['tagy'])

        if 'category' in rec and rec['category']:
            unique_categories.add(rec['category'])

    return {
        'tags': unique_tags,
        'categories': unique_categories
    }


def build_tag_corpus(records: List[Dict[str, Any]]):
    """Build a TF-IDF corpus from the tags in the records.
    :param records: List of records, each a dictionary with 'tags' key.
    :return: Tuple of TF-IDF matrix and feature names."""
    # flatten all tags, using 'tagy' as fallback if 'tags' is not present
    docs = []
    for rec in records:
        if 'tags' in rec and isinstance(rec['tags'], list):
            docs.append(' '.join(rec['tags']))
        elif 'tagy' in rec and isinstance(rec['tagy'], list):
            docs.append(' '.join(rec['tagy']))
        else:
            docs.append('')

    # allow single-character tags by adjusting token_pattern
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(docs)
    return X, vectorizer.get_feature_names_out()


def cluster_tags(X, n_clusters: int = 8):
    """Cluster the tags using Agglomerative Clustering
    :param X: TF-IDF matrix of tags.
    :param n_clusters: Number of clusters to form.
    :return: Cluster labels for each document."""
    if X.shape[0] < n_clusters:
        logger.warning(f"Not enough documents ({X.shape[0]}) for {n_clusters} clusters. Adjusting to {max(2, X.shape[0]-1)}")
        n_clusters = max(2, X.shape[0]-1)

    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X.toarray())
    return labels


def propose_taxonomy(tags: Set[str], categories: Set[str]) -> Dict[str, Any]:
    """Propose a taxonomy based on the unique tags and categories using OpenAI LLM.
    :param tags: Set of unique tags.
    :param categories: Set of unique categories.
    :return: Dictionary with proposed taxonomy."""
    logger.info(f"Starting taxonomy proposal with {len(tags)} tags and {len(categories)} categories")

    tags_list = list(tags)
    categories_list = list(categories)

    user_prompt = (
        f"Navrhnite hierarchickú taxonómiu pre systém RAG na základe týchto tagov a kategórií:\n\n"
        f"TAGY: {json.dumps(tags_list, ensure_ascii=False)}\n\n"
        f"KATEGÓRIE: {json.dumps(categories_list, ensure_ascii=False)}\n\n"
        "Vytvorte logickú hierarchickú štruktúru, ktorá tieto tagy a kategórie efektívne organizuje. "
        "Štruktúra by nemala mať viac ako 3 úrovne zanorenia. "
        "Zabezpečte, aby každý tag bol zaradený aspoň do jednej kategórie."
    )

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        logger.info(f"Sending request to OpenAI API with model: {MODEL_NAME}")

        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=4000,
            temperature=0.2,
        )

        if not resp or not resp.choices:
            logger.error("Empty response from OpenAI API")
            return {}

        content = resp.choices[0].message.content.strip()
        if not content:
            logger.error("Empty content in OpenAI response")
            return {}

        logger.info(f"Received response content (first 200 chars): {content[:200] + '...' if len(content) > 200 else content}")

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

            result = json.loads(content)
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Raw response content: {content}")
            return {}

    except Exception as e:
        logger.error(f'OpenAI API call failed: {str(e)}')
        return {}


def save_taxonomy(tree: Dict[str, Any], out_path: Path):
    """Save the proposed taxonomy to a JSON file.
    :param tree: Proposed taxonomy as a dictionary.
    :param out_path: Path to save the taxonomy JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)


def main(root: Path, preprocessed_dir: Path, out_tax: Path):
    """Main function to generate taxonomy from preprocessed JSONL files.
    :param root: Root directory (not used here, but can be useful for logging).
    :param preprocessed_dir: Path to the directory with preprocessed JSONL files.
    :param out_tax: Path to save the proposed taxonomy JSON file."""
    logger.info(f"Starting taxonomy analysis from {preprocessed_dir}")

    # Load all preprocessed JSONL files
    records = load_preprocessed_jsonl(preprocessed_dir)
    logger.info(f"Loaded {len(records)} records from preprocessed files")

    if not records:
        logger.error("No records found in preprocessed files. Exiting.")
        return

    # Collect unique tags and categories
    unique_data = collect_tags_and_categories(records)
    logger.info(f"Found {len(unique_data['tags'])} unique tags and {len(unique_data['categories'])} categories")

    # Build corpus from tags
    X, feature_names = build_tag_corpus(records)

    # Cluster documents into groups (by tags corpus)
    labels = cluster_tags(X, n_clusters=8)
    logger.info(f"Clustered documents into {len(set(labels))} groups")

    # Propose taxonomy via LLM
    taxonomy = propose_taxonomy(unique_data['tags'], unique_data['categories'])

    # Save the taxonomy
    save_taxonomy(taxonomy, out_tax)
    logger.info(f'Taxonomy saved to {out_tax}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Generate taxonomy from preprocessed JSONL files')
    parser.add_argument('root', type=Path, help='Root directory path')
    parser.add_argument('--preprocessed', type=Path, default=Path('Preprocessed'),
                       help='Path to directory with preprocessed JSONL files')
    parser.add_argument('--out', type=Path, default=Path('taxonomy.json'),
                       help='Output taxonomy JSON file')
    args = parser.parse_args()
    main(args.root, args.preprocessed, args.out)
