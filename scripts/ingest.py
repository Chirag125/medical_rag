"""
scripts/ingest.py — the real ingestion entrypoint.

Why this exists: embedder.py's __main__ block is a test harness — it only
processes one hardcoded PDF, only 50 children, and re-running it APPENDS
duplicate chunks (new random UUIDs each time) instead of replacing them.
That's why your last retriever.py run showed 3 identical top results.

This script:
  1. Loads every PDF in data/raw/ (via load_corpus)
  2. Chunks all of it (via split_into_parent_child)
  3. WIPES the existing child_chunks/parent_chunks collections first, then
     stores everything fresh — so running this twice gives you the same
     result, not double the chunks.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --data-dir data/raw --chroma-path data/processed/chroma
"""

import argparse
import os
import sys

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
sys.path.append(".")

from src.ingestion.document_loader import load_corpus
from src.ingestion.chunker import split_into_parent_child
from src.ingestion.embedder import TextEmbedder


def reset_collections(embedder):
    """Delete and recreate both collections so re-ingesting is idempotent."""
    for name in ["child_chunks", "parent_chunks"]:
        try:
            embedder.client.delete_collection(name)
            print(f"Cleared existing collection: {name}")
        except Exception:
            pass  # didn't exist yet, fine

    embedder.child_collection = embedder.client.get_or_create_collection(
        name="child_chunks", metadata={"hnsw:space": "cosine"}
    )
    embedder.parent_collection = embedder.client.get_or_create_collection(
        name="parent_chunks", metadata={"hnsw:space": "cosine"}
    )


def main():
    parser = argparse.ArgumentParser(description="Full-corpus ingestion for medical_rag")
    parser.add_argument("--data-dir", default="data/raw", help="Directory of PDFs to ingest")
    parser.add_argument("--chroma-path", default="data/processed/chroma", help="ChromaDB persist directory")
    parser.add_argument("--max-docs", type=int, default=50, help="Max PDFs to load")
    args = parser.parse_args()

    print(f"Loading all PDFs from {args.data_dir} ...")
    pages = load_corpus(args.data_dir, max_docs=args.max_docs)
    if not pages:
        print("No pages loaded — check that data-dir has PDFs in it.")
        return

    print("Chunking into parent/child splits ...")
    parents, children = split_into_parent_child(pages)
    print(f"  {len(parents)} parent chunks, {len(children)} child chunks")

    embedder = TextEmbedder(chroma_path=args.chroma_path)

    print("\nResetting collections (so this script is safe to re-run) ...")
    reset_collections(embedder)

    embedder.store_parents(parents)
    embedder.store_children(children)

    embedder.get_collection_stats()
    print("\nIngestion complete.")


if __name__ == "__main__":
    main()