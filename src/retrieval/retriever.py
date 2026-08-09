# src/retrieval/retriever.py
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import sys
sys.path.append(".")

import torch
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.ingestion.embedder import TextEmbedder, BGE_INSTRUCTION
import numpy as np


class Retriever:
    """
    Two-stage retrieval pipeline:
    Stage 1 — BGE dual encoder: fast approximate retrieval (top-20)
    Stage 2 — Cross encoder reranker: precise reranking (top-5)

    Why two stages:
    - Dual encoder is fast but imprecise (encodes query + doc separately)
    - Cross encoder is slow but precise (reads query + doc together)
    - Run cross encoder only on top-20 candidates, not all chunks
    """

    def __init__(self, chroma_path="data/processed/chroma",
                 top_k_retrieve=20, top_k_rerank=5):

        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank

        # Load BGE for query embedding
        print("Loading BGE for retrieval...")
        self.bi_encoder = SentenceTransformer("BAAI/bge-large-en-v1.5")

        # Load cross-encoder for reranking
        print("Loading cross-encoder reranker...")
        self.cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512
        )

        # Connect to existing ChromaDB
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.child_collection = self.client.get_collection("child_chunks")
        self.parent_collection = self.client.get_collection("parent_chunks")

        print(f"Retriever ready — "
              f"{self.child_collection.count()} chunks indexed")

    def embed_query(self, query):
        """Embed query with BGE instruction prefix."""
        prefixed = BGE_INSTRUCTION + query
        embedding = self.bi_encoder.encode(
            prefixed,
            normalize_embeddings=True
        )
        return embedding

    def retrieve_candidates(self, query, top_k=None):
        """
        Stage 1: fast retrieval from ChromaDB.
        Returns top_k child chunks by cosine similarity.
        """
        top_k = top_k or self.top_k_retrieve
        query_embedding = self.embed_query(query)

        results = self.child_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        candidates = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            candidates.append({
                "text": doc,
                "metadata": meta,
                "bi_encoder_score": 1 - dist,  # convert distance to similarity
                "parent_id": meta["parent_id"]
            })

        return candidates

    def rerank(self, query, candidates):
        """
        Stage 2: cross-encoder reranking.

        Cross encoder reads [query, document] together —
        much more accurate than independent embeddings.
        Slower, so only run on top-20 candidates from stage 1.
        """
        if not candidates:
            return []

        # Build (query, document) pairs for cross encoder
        pairs = [[query, c["text"]] for c in candidates]

        # Score all pairs
        scores = self.cross_encoder.predict(pairs)

        # Attach cross-encoder scores
        for candidate, score in zip(candidates, scores):
            candidate["cross_encoder_score"] = float(score)

        # Sort by cross-encoder score (higher = more relevant)
        reranked = sorted(candidates,
                          key=lambda x: x["cross_encoder_score"],
                          reverse=True)

        return reranked[:self.top_k_rerank]

    def fetch_parents(self, top_candidates):
        """
        Fetch parent chunks for top reranked results.
        Parents have more context — sent to LLM for generation.
        """
        parent_ids = list({c["parent_id"] for c in top_candidates})

        results = self.parent_collection.get(
            ids=parent_ids,
            include=["documents", "metadatas"]
        )

        # Build parent lookup
        parent_map = {}
        for pid, doc, meta in zip(
            results["ids"],
            results["documents"],
            results["metadatas"]
        ):
            parent_map[pid] = {
                "text": doc,
                "metadata": meta
            }

        return parent_map

    def retrieve(self, query, return_parents=True):
        """
        Full retrieval pipeline:
        query → BGE top-20 → cross-encoder rerank → top-5 parents

        Returns list of dicts with text, scores, metadata, and image_ids.
        """
        # Stage 1: candidate retrieval
        candidates = self.retrieve_candidates(query)
        print(f"Stage 1: retrieved {len(candidates)} candidates")

        # Stage 2: reranking
        reranked = self.rerank(query, candidates)
        print(f"Stage 2: reranked to top {len(reranked)}")

        if not return_parents:
            return reranked

        # Fetch parent chunks for context
        parent_map = self.fetch_parents(reranked)

        # Attach parent text to results
        results = []
        for candidate in reranked:
            parent = parent_map.get(candidate["parent_id"], {})
            results.append({
                "child_text": candidate["text"],
                "parent_text": parent.get("text", candidate["text"]),
                "bi_encoder_score": candidate["bi_encoder_score"],
                "cross_encoder_score": candidate["cross_encoder_score"],
                "doc_name": candidate["metadata"]["doc_name"],
                "page_num": candidate["metadata"]["page_num"],
                # NEW: comma-joined image_ids for this chunk's page, "" if none.
                # Lets the generator (or a future image-answer path) know which
                # images are available for this result without re-querying.
                "image_ids": candidate["metadata"].get("image_ids", "")
            })

        return results


if __name__ == "__main__":
    print("=== Testing retriever ===\n")

    retriever = Retriever()

    # Test query
    query = "What are the safety challenges and limitations of GPT-4?"

    print(f"Query: '{query}'\n")
    results = retriever.retrieve(query)

    print(f"\nTop {len(results)} results after reranking:\n")
    for i, r in enumerate(results):
        print(f"Result {i+1}")
        print(f"  BGE score:          {r['bi_encoder_score']:.4f}")
        print(f"  Cross-encoder score: {r['cross_encoder_score']:.4f}")
        print(f"  Page: {r['page_num']} | Doc: {r['doc_name']} | Images: '{r['image_ids']}'")
        print(f"  Child: {r['child_text'][:120]}...")
        print(f"  Parent: {r['parent_text'][:200]}...")
        print()

    print("retriever.py works correctly")