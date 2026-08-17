"""
scripts/ingest_ablation.py — builds one ChromaDB per ablation strategy so you
can run the exact same eval_set.json against each and get a clean four-row
comparison table.

Strategies:
    naive          -> split_naive() from chunker.py (add that function first —
                       see chunker_naive_addition.py)
    context_header -> current split_into_parent_child(), context headers only
                       conceptually the same as parent_child below since your
                       chunker.py already couples headers + parent-child; see
                       note below if you want them fully separated
    parent_child   -> current split_into_parent_child() (your existing pipeline)
    hyde           -> parent_child chunks + hypothetical questions appended
                       before embedding, using generate_hypothetical_questions()

IMPORTANT — context_header vs parent_child:
Your chunker.py currently always does BOTH context headers AND parent-child
together — they're not separable without a small edit. For a true 3-stage
comparison (naive -> +headers -> +parent-child), you'd need a second function
that does flat chunking WITH the header but WITHOUT the parent-child split.
That's a straightforward variant of split_naive() — prepend context_header
before chunking, same flat output shape. If you want that exact function,
just ask and I'll add it; for now this script treats "context_header" and
"parent_child" as the same strategy (your current pipeline) since your
chunker.py doesn't yet support running them independently.

IMPORTANT — HyDE performance:
generate_hypothetical_questions() calls an LLM per chunk. On CPU with
Phi-3-mini, this is slow — expect several seconds PER CHUNK, so hundreds of
chunks could take hours. For the ablation table, this script defaults to
running HyDE on a random SAMPLE of chunks (--hyde-sample-size, default 30)
rather than the full corpus, which is enough to get a directional recall@5
comparison without an overnight run. Increase it if you have time.

Usage:
    python scripts/ingest_ablation.py --strategy naive
    python scripts/ingest_ablation.py --strategy parent_child
    python scripts/ingest_ablation.py --strategy hyde --hyde-sample-size 50

Then point eval.py at each result:
    python -m src.evaluation.eval --data src/evaluation/eval_set.json \\
        --persist-dir data/processed/chroma_naive
    python -m src.evaluation.eval --data src/evaluation/eval_set.json \\
        --persist-dir data/processed/chroma_parent_child
    python -m src.evaluation.eval --data src/evaluation/eval_set.json \\
        --persist-dir data/processed/chroma_hyde
"""

import argparse
import os
import random
import sys

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
sys.path.append(".")

from src.ingestion.document_loader import load_corpus
from src.ingestion.chunker import split_into_parent_child, count_tokens
from src.ingestion.embedder import TextEmbedder

try:
    from src.ingestion.chunker import split_naive
except ImportError:
    split_naive = None  # add split_naive() from chunker_naive_addition.py first


def reset_collections(embedder):
    for name in ["child_chunks", "parent_chunks"]:
        try:
            embedder.client.delete_collection(name)
        except Exception:
            pass
    embedder.child_collection = embedder.client.get_or_create_collection(
        name="child_chunks", metadata={"hnsw:space": "cosine"}
    )
    embedder.parent_collection = embedder.client.get_or_create_collection(
        name="parent_chunks", metadata={"hnsw:space": "cosine"}
    )


def apply_hyde(children, llm_fn, sample_size):
    """
    Append hypothetical questions to a random sample of children before
    embedding. Untouched children keep their original text unchanged, so
    this stays a fair "mostly parent_child, plus HyDE on a sample" test.
    """
    from src.ingestion.chunker import generate_hypothetical_questions

    sample_idx = set(random.sample(range(len(children)), min(sample_size, len(children))))
    for i, child in enumerate(children):
        if i not in sample_idx:
            continue
        questions = generate_hypothetical_questions(child["raw_text"], llm_fn, n_questions=2)
        if questions:
            hyde_text = child["text"] + "\n\n" + "\n".join(questions)
            child["text"] = hyde_text
            child["token_count"] = count_tokens(hyde_text)
    return children


def main():
    parser = argparse.ArgumentParser(description="Build one ChromaDB per ablation strategy")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--strategy", required=True, choices=["naive", "context_header", "parent_child", "hyde"])
    parser.add_argument("--chroma-path", default=None, help="Defaults to data/processed/chroma_<strategy>")
    parser.add_argument("--hyde-sample-size", type=int, default=30)
    parser.add_argument("--max-docs", type=int, default=50)
    args = parser.parse_args()

    chroma_path = args.chroma_path or f"data/processed/chroma_{args.strategy}"

    print(f"Loading PDFs from {args.data_dir} ...")
    pages = load_corpus(args.data_dir, max_docs=args.max_docs)
    if not pages:
        print("No pages loaded.")
        return

    if args.strategy == "naive":
        if split_naive is None:
            print("split_naive() not found in chunker.py — add it from chunker_naive_addition.py first.")
            return
        parents, children = split_naive(pages)
    else:
        # context_header and parent_child both use your current pipeline
        # (see the note above about why they aren't separated yet)
        parents, children = split_into_parent_child(pages)

    if args.strategy == "hyde":
        print(f"Loading generator for HyDE (sampling {args.hyde_sample_size} chunks) ...")
        from src.generation.generation import Generator
        generator = Generator()

        def llm_fn(prompt):
            output = generator.pipe(
                [{"role": "user", "content": prompt}],
                max_new_tokens=80,
                do_sample=False,
                return_full_text=False,
            )
            return output[0]["generated_text"]

        children = apply_hyde(children, llm_fn, args.hyde_sample_size)

    embedder = TextEmbedder(chroma_path=chroma_path)
    print(f"Resetting collections at {chroma_path} ...")
    reset_collections(embedder)
    embedder.store_parents(parents)
    embedder.store_children(children)
    embedder.get_collection_stats()
    print(f"\nDone. Point eval.py at --persist-dir {chroma_path} to test this strategy.")


if __name__ == "__main__":
    main()