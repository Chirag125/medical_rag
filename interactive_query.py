"""
interactive_query.py — loads Retriever + Generator ONCE, then loops taking
questions from you until you quit. This is what makes querying actually
interactive instead of paying model-load cost on every single question.

Also surfaces image_ids from retrieved chunks so you can see, per query,
whether any images were pulled in alongside the text — this is your check
for whether image retrieval is actually doing anything yet.

Usage:
    python interactive_query.py
    (then type questions, one per line; type 'quit' or 'exit' to stop)
"""

import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import sys
sys.path.append(".")

from src.retrieval.retriever import Retriever
from src.generation.generation import Generator  # rename to generator.py if you've done that cleanup

IMAGE_DIR = "data/processed/images"


def collect_image_refs(results):
    """Gather every non-empty image_id across the retrieved results, deduped."""
    all_ids = []
    for r in results:
        ids = [i for i in r.get("image_ids", "").split(",") if i]
        all_ids.extend(ids)
    return list(dict.fromkeys(all_ids))  # dedupe, preserve order


def main():
    print("Loading Retriever (BGE + cross-encoder + ChromaDB) ...")
    retriever = Retriever()

    print("Loading Generator (Phi-3-mini) ...")
    generator = Generator()

    print("\nReady. Type a question (or 'quit' to exit).\n")

    while True:
        query = input("Question: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("Bye.")
            break

        results = retriever.retrieve(query)
        result = generator.generate(query, results)

        print("\n" + "=" * 50)
        print("ANSWER:")
        print(result["answer"])

        print("\nSOURCES:")
        for s in result["sources"]:
            print(f"  {s['doc']} page {s['page']} (relevance: {s['relevance']:.2f})")

        image_ids = collect_image_refs(results)
        if image_ids:
            print(f"\nIMAGES REFERENCED ({len(image_ids)}):")
            for image_id in image_ids:
                image_path = os.path.join(IMAGE_DIR, f"{image_id}.png")
                exists = "found" if os.path.exists(image_path) else "MISSING FILE"
                print(f"  {image_id} -> {image_path} [{exists}]")
        else:
            print("\nIMAGES REFERENCED: none (no image_ids on any retrieved chunk)")

        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()