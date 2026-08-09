"""
eval.py — Recall@5 evaluation for Chirag125/medical_rag, wired directly to
src/retrieval/retriever.py's Retriever class (the real thing your app runs).

Config values below match your config.py:
    TEXT_EMBED_MODEL  = "BAAI/bge-large-en-v1.5"
    RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CHROMA_PATH       = "data/processed/chroma"
    TOP_K_RETRIEVAL   = 20
    TOP_K_RERANK      = 5
    RECALL_K          = 5

IMPORTANT — metadata schema reality check:
Your Retriever.retrieve_candidates() currently exposes only these metadata
fields per chunk: doc_name, page_num, parent_id (see retriever.py — it never
reads an `image_id`). That's because ingestion so far has only run on a
single text PDF, not VQA-RAD.

So this script supports two relevance definitions — pick with --relevance-key:

  1. --relevance-key doc_name  (default, WORKS TODAY)
     Checks whether the retrieved chunk's doc_name matches the question's
     ground-truth source document. Use this to get a recall@5 number on your
     current PDF pipeline right now, with zero changes to ingestion.

  2. --relevance-key image_id  (VQA-RAD, requires an ingestion change)
     For a "real" VQA-RAD recall@5 (what your SOP wants), image_id needs to
     exist in chunk metadata. That means extending src/ingestion/chunker.py
     or embedder.py to attach an image_id when you ingest the VQA-RAD corpus.
     Once you've done that, this script works unchanged — just point
     --relevance-key at whatever field name you used.

Usage (today, on your current single-PDF pipeline):
    python eval.py --data eval_set.json --relevance-key doc_name

Usage (once VQA-RAD is ingested with image_id in metadata):
    python eval.py --data vqa_rad_test.json --relevance-key image_id

Add --no-rerank to also get bi-encoder-only recall (for the ablation table):
    python eval.py --data eval_set.json --no-rerank

Expected input JSON format, one object per QA pair:
[
  {
    "qid": "0001",
    "question": "What are the safety challenges of GPT-4?",
    "doc_name": "gpt4_safety_paper.pdf"   // or "image_id": "..." for VQA-RAD
  },
  ...
]
"""

import argparse
import json
import os
import sys
from statistics import mean

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# repo root must be on the path so `from src.retrieval.retriever import Retriever` works
sys.path.insert(0, os.getcwd())

try:
    import config
except ImportError:
    config = None

CHROMA_PATH = getattr(config, "CHROMA_PATH", "data/processed/chroma")
TOP_K_RETRIEVAL = getattr(config, "TOP_K_RETRIEVAL", 20)
TOP_K_RERANK = getattr(config, "TOP_K_RERANK", 5)
RECALL_K = getattr(config, "RECALL_K", 5)


def load_dataset(path, relevance_key):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("Expected a non-empty JSON list of QA objects.")
    required = {"qid", "question", relevance_key}
    missing = required - set(data[0].keys())
    if missing:
        raise ValueError(
            f"Dataset items are missing required fields: {missing}. "
            f"(Looking for relevance key '{relevance_key}' — pass --relevance-key to change it.)"
        )
    return data


def is_hit(top_k_results, ground_truth_value, relevance_key):
    """
    top_k_results: list of dicts with a 'metadata' key (as produced below).

    doc_name: exact match (one chunk = one doc).
    image_ids: membership check — metadata stores a comma-joined string of every
               image_id on that chunk's page (see chunker.py), since a page can
               hold multiple images and ChromaDB metadata values must be scalars.
    """
    if relevance_key == "image_ids" or relevance_key == "image_id":
        for r in top_k_results:
            ids_on_chunk = [i for i in r["metadata"].get("image_ids", "").split(",") if i]
            if ground_truth_value in ids_on_chunk:
                return True
        return False
    if relevance_key == "page_num":
        # metadata page_num is an int; make sure the dataset's ground truth compares as int too
        return any(int(r["metadata"].get("page_num", -1)) == int(ground_truth_value) for r in top_k_results)
    return any(r["metadata"].get(relevance_key) == ground_truth_value for r in top_k_results)


def run_eval(retriever, dataset, k, use_rerank, relevance_key):
    from src.retrieval.retriever import Retriever  # noqa: F401 (imported for clarity/type hints)

    hits = []
    per_query = []

    for item in dataset:
        question = item["question"]

        candidates = retriever.retrieve_candidates(question, top_k=TOP_K_RETRIEVAL)

        if use_rerank:
            top_k_raw = retriever.rerank(question, candidates)  # already capped at top_k_rerank
            top_k_raw = top_k_raw[:k]
        else:
            top_k_raw = candidates[:k]

        # normalize shape: retrieve_candidates()/rerank() both return dicts with "metadata"
        top_k_results = [{"metadata": c["metadata"]} for c in top_k_raw]

        hit = is_hit(top_k_results, item[relevance_key], relevance_key)
        hits.append(hit)
        per_query.append({"qid": item["qid"], "question": question, "hit": hit})

    recall_at_k = mean(1.0 if h else 0.0 for h in hits)
    return recall_at_k, per_query


def main():
    parser = argparse.ArgumentParser(description="Recall@k eval for medical_rag, using the real Retriever class")
    parser.add_argument("--data", required=True, help="Path to eval set JSON")
    parser.add_argument("--persist-dir", default=CHROMA_PATH, help=f"ChromaDB persist dir (default: {CHROMA_PATH})")
    parser.add_argument("--k", type=int, default=RECALL_K, help=f"k for recall@k (default: {RECALL_K})")
    parser.add_argument("--no-rerank", action="store_true", help="Skip cross-encoder reranking (bi-encoder-only recall, for the ablation table)")
    parser.add_argument("--relevance-key", default="page_num", choices=["doc_name", "page_num", "image_ids"],
                         help="Metadata field to check hits against. 'page_num' is meaningful even with "
                              "just one ingested PDF (checks if the right page was retrieved); 'doc_name' "
                              "only becomes meaningful once you've ingested multiple documents; "
                              "'image_ids' checks membership in the comma-joined image_ids on the chunk's page.")
    parser.add_argument("--out", default="eval_results.json", help="Where to write per-query results")
    args = parser.parse_args()

    print(f"Loading dataset from {args.data} (relevance key: {args.relevance_key}) ...")
    dataset = load_dataset(args.data, args.relevance_key)
    print(f"  {len(dataset)} questions loaded")

    # Import here (after sys.path is set) so this script can live at the repo root
    from src.retrieval.retriever import Retriever

    print("Loading Retriever (BGE + cross-encoder + ChromaDB)...")
    retriever = Retriever(
        chroma_path=args.persist_dir,
        top_k_retrieve=TOP_K_RETRIEVAL,
        top_k_rerank=args.k,
    )

    use_rerank = not args.no_rerank
    print(f"\nRunning recall@{args.k} eval (rerank={use_rerank}) ...")
    recall_at_k, per_query = run_eval(retriever, dataset, args.k, use_rerank, args.relevance_key)

    label = f"recall@{args.k}" + (" (reranked)" if use_rerank else " (bi-encoder only)")
    print(f"\n{label} = {recall_at_k:.4f}  ({sum(p['hit'] for p in per_query)}/{len(per_query)} hits)")

    with open(args.out, "w") as f:
        json.dump({
            "recall_at_k": recall_at_k,
            "k": args.k,
            "rerank": use_rerank,
            "relevance_key": args.relevance_key,
            "per_query": per_query,
        }, f, indent=2)
    print(f"Per-query results written to {args.out}")


if __name__ == "__main__":
    main()