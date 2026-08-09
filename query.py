"""
query.py — ask an arbitrary question against your ingested corpus.

This is what was missing: retriever.py and generation.py only had a hardcoded
test query in their __main__ blocks. This script takes your question as a
command-line argument (or interactively if you omit it), runs the same
retrieve -> generate pipeline, and prints the answer + sources.

Usage:
    python query.py "What are the safety challenges of GPT-4?"
    python query.py            # prompts you for a question interactively
"""

import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import sys
sys.path.append(".")

from src.retrieval.retriever import Retriever
from src.generation.generation import Generator  # rename to generator.py if you do that cleanup


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Question: ").strip()

    if not query:
        print("No question given.")
        return

    print(f"\nQuery: '{query}'\n")

    retriever = Retriever()
    results = retriever.retrieve(query)

    generator = Generator()
    result = generator.generate(query, results)

    print("=" * 50)
    print("ANSWER:")
    print(result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  Source {s['source_num']}: {s['doc']} page {s['page']} (relevance: {s['relevance']:.2f})")
    print(f"\nContext chunks used: {result['context_used']}")
    print("=" * 50)


if __name__ == "__main__":
    main()