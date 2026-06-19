# src/ingestion/embedder.py
import torch
import chromadb

from sentence_transformers import SentenceTransformer

import numpy as np

from tqdm import tqdm
import os

# BGE requires a specific instruction prefix for retrieval tasks
# This is what makes BGE significantly better than generic embedders
BGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "

class TextEmbedder:
    """
    Embeds text chunks using BGE-large-en-v1.5 and stores in ChromaDB.
    
    BGE (Beijing Academy of AI) consistently outperforms OpenAI ada-002
    on retrieval benchmarks (MTEB) and runs fully locally — no API cost.
    """

    def __init__(self, chroma_path="data/processed/chroma"):
        print("Loading BGE-large-en-v1.5...")
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        print(f"BGE loaded on {self.device}")

        # ChromaDB persistent client — data survives restarts
        os.makedirs(chroma_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=chroma_path)

        # Two separate collections:
        # children → retrieved at query time (small, precise)
        # parents  → fetched after retrieval for LLM context (large)
        self.child_collection = self.client.get_or_create_collection(
            name="child_chunks",
            metadata={"hnsw:space": "cosine"}  # cosine similarity
        )
        self.parent_collection = self.client.get_or_create_collection(
            name="parent_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"ChromaDB ready at {chroma_path}")

    def embed_texts(self, texts, batch_size=32, add_instruction=True):
        """
        Embed a list of texts in batches.
        
        add_instruction: True for children (retrieval queries)
                        False for parents (not embedded through BGE)
        """
        if add_instruction:
            texts = [BGE_INSTRUCTION + t for t in texts]

        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), 
                      desc="Embedding batches"):
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                embeddings = self.model.encode(
                    batch,
                    normalize_embeddings=True,  # normalise for cosine sim
                    show_progress_bar=False
                )
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def store_children(self, children, batch_size=100):
        """
        Embed child chunks and store in ChromaDB.
        Children are what gets searched at query time.
        """
        print(f"Embedding and storing {len(children)} child chunks...")

        # Process in batches to avoid memory issues
        for i in tqdm(range(0, len(children), batch_size),
                      desc="Storing children"):
            batch = children[i:i + batch_size]

            texts = [c["text"] for c in batch]
            ids = [c["id"] for c in batch]
            metadatas = [{
                "parent_id": c["parent_id"],
                "doc_name": c["doc_name"],
                "page_num": c["page_num"],
                "token_count": c["token_count"],
                "chunk_type": "child"
            } for c in batch]

            embeddings = self.embed_texts(texts, 
                                          batch_size=batch_size,
                                          add_instruction=True)

            self.child_collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas
            )

        print(f"Stored {len(children)} children in ChromaDB")

    def store_parents(self, parents):
        """
        Store parent chunks in ChromaDB WITHOUT embedding.
        Parents are fetched by ID after child retrieval —
        they don't need embeddings, just storage.
        
        We store them in ChromaDB for consistency but could
        also use SQLite here.
        """
        print(f"Storing {len(parents)} parent chunks (no embedding)...")

        # ChromaDB requires embeddings — use zero vectors as placeholder
        # Parents are never searched, only fetched by ID
        dummy_embedding = [0.0] * 1024  # BGE-large dimension is 1024

        batch_size = 100
        for i in tqdm(range(0, len(parents), batch_size),
                      desc="Storing parents"):
            batch = parents[i:i + batch_size]

            self.parent_collection.add(
                ids=[p["id"] for p in batch],
                embeddings=[dummy_embedding] * len(batch),
                documents=[p["text"] for p in batch],
                metadatas=[{
                    "doc_name": p["doc_name"],
                    "page_num": p["page_num"],
                    "token_count": p["token_count"],
                    "chunk_type": "parent"
                } for p in batch]
            )

        print(f"Stored {len(parents)} parents in ChromaDB")

    def get_collection_stats(self):
        """Quick sanity check — how many chunks are stored?"""
        n_children = self.child_collection.count()
        n_parents = self.parent_collection.count()
        print(f"ChromaDB stats:")
        print(f"  Child chunks: {n_children}")
        print(f"  Parent chunks: {n_parents}")
        return n_children, n_parents


if __name__ == "__main__":
    import sys
    import os
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    sys.path.append(".")

    from src.ingestion.document_loader import load_pdf
    from src.ingestion.chunker import split_into_parent_child

    print("=== Testing full pipeline ===\n")

    # Load and chunk
    pages = load_pdf("data/raw/test_paper.pdf")
    parents, children = split_into_parent_child(pages)

    # Use first 50 children for speed
    test_children = children[:50]
    test_parents = [p for p in parents
                    if p["id"] in {c["parent_id"] for c in test_children}]

    print(f"Test set: {len(test_parents)} parents, {len(test_children)} children\n")

    # Initialise embedder
    embedder = TextEmbedder()

    # Store
    embedder.store_parents(test_parents)
    embedder.store_children(test_children)

    # Stats
    embedder.get_collection_stats()

    # YOUR FIRST REAL SEMANTIC QUERY
    print("\n=== First semantic query ===")
    query = "What are the capabilities and limitations of GPT-4?"

    query_embedding = embedder.embed_texts([query], add_instruction=True)[0]

    results = embedder.child_collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    print(f"Query: '{query}'\n")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        similarity = 1 - dist
        print(f"Result {i+1} | Score: {similarity:.4f} | "
              f"Page: {meta['page_num']}")
        print(f"  {doc[:200]}\n")

    # Fetch parent for top result
    top_child_meta = results["metadatas"][0][0]
    parent_result = embedder.parent_collection.get(
        ids=[top_child_meta["parent_id"]],
        include=["documents"]
    )
    print("Parent chunk for top result (what LLM will see):")
    print(f"  {parent_result['documents'][0][:300]}")
    print("\nFull pipeline works correctly")