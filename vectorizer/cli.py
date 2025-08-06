# vectorizer/cli.py
import click
import json
import settings
from vectorizer.qdrant_client import ensure_collection
import embedder
import splitter
import loader

from pathlib import Path


@click.command()
@click.argument("input_dir", type=Path)
@click.option("--glob", default="*.jsonl")
def vector_load(input_dir, glob):
    files = sorted(input_dir.rglob(glob))
    ensure_collection(dim=1536)
    for f in files:
        for line in f.open():
            doc = json.loads(line)
            chunks = splitter.split(doc["text"])
            vectors = embedder.embed([c["text"] for c in chunks])
            for chunk, vec in zip(chunks, vectors):
                pid = f"{doc['metadata']['hash_content']}_{chunk['idx']}"
                payload = {**doc["metadata"],
                           **chunk["metadata"],    # start/end tokens
                           "tags": doc["tagy"],
                           "summary": doc["sumar"]}
                qdrant_client.upsert(pid, vec, payload)
