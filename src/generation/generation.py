# src/generation/generator.py
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import sys
sys.path.append(".")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class Generator:
    """
    Answer generation using Phi-3-mini-4k-instruct.
    
    Takes retrieved context chunks and a user query,
    builds a structured prompt, and generates a grounded answer.
    
    Why Phi-3-mini:
    - 3.8B parameters — runs on CPU (slow but works)
    - Instruction-tuned — follows system prompts reliably
    - 4k context window — fits multiple retrieved chunks
    - Free, local, no API cost
    """

    # def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
    #     print(f"Loading {model_name}...")
    #     print("This will take 2-3 minutes on first run (downloading ~2.3GB)...")

    #     self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    #     self.model = AutoModelForCausalLM.from_pretrained(
    #         model_name,
    #         torch_dtype=torch.float32,  # float32 for CPU
    #         device_map="cpu",
    #         trust_remote_code=True
    #     )

    #     self.pipe = pipeline(
    #         "text-generation",
    #         model=self.model,
    #         tokenizer=self.tokenizer,
    #     )
    #     print("Generator ready")

    # In generator.py, replace the model loading section:

    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
        print(f"Loading {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,   # half precision — halves memory usage
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True       # loads weights more efficiently
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        print("Generator ready")

    def build_prompt(self, query, retrieved_results):
        """
        Build a RAG prompt from query + retrieved context.
        
        Structure:
        - System message: instructs model to answer from context only
        - Context: retrieved chunks (use parent text for more context)
        - Question: user query
        - Answer instruction: be specific and cite sources
        """
        # Build context block from retrieved results
        context_parts = []
        for i, result in enumerate(retrieved_results):
            # Use parent text — more context than child
            text = result.get("parent_text", result.get("child_text", ""))
            doc = result.get("doc_name", "unknown")
            page = result.get("page_num", "?")
            score = result.get("cross_encoder_score", 0)

            context_parts.append(
                f"[Source {i+1} | {doc} | Page {page} | "
                f"Relevance: {score:.2f}]\n{text}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # Phi-3 chat template
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a medical research assistant. "
                    "Answer questions using ONLY the provided context. "
                    "If the context does not contain enough information "
                    "to answer the question, say so explicitly. "
                    "Always cite which source (Source 1, 2, etc.) "
                    "your answer comes from."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {query}\n\n"
                    "Answer based on the context above:"
                )
            }
        ]

        return messages

    def generate(self, query, retrieved_results, max_new_tokens=300):
        """
        Generate an answer given a query and retrieved results.
        
        Returns dict with:
        - answer: generated text
        - sources: list of source citations
        - context_used: number of chunks used
        """
        if not retrieved_results:
            return {
                "answer": "No relevant context found to answer this question.",
                "sources": [],
                "context_used": 0
            }

        messages = self.build_prompt(query, retrieved_results)

        # Generate
        output = self.pipe(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # deterministic — better for factual QA
            temperature=1.0,        # ignored when do_sample=False
            return_full_text=False  # only return generated part
        )

        answer = output[0]["generated_text"].strip()

        # Extract source citations
        sources = []
        for i, result in enumerate(retrieved_results):
            if result.get("cross_encoder_score", 0) > 0:
                sources.append({
                    "source_num": i + 1,
                    "doc": result.get("doc_name"),
                    "page": result.get("page_num"),
                    "relevance": result.get("cross_encoder_score")
                })

        return {
            "answer": answer,
            "sources": sources,
            "context_used": len(retrieved_results)
        }


if __name__ == "__main__":
    import sys
    sys.path.append(".")

    from src.retrieval.retriever import Retriever

    print("=== Testing full RAG pipeline ===\n")

    # Step 1: retrieve
    retriever = Retriever()
    query = "What are the safety challenges of GPT-4?"

    print(f"Query: '{query}'\n")
    retrieved = retriever.retrieve(query)
    print(f"Retrieved {len(retrieved)} results\n")

    # Step 2: generate
    generator = Generator()
    result = generator.generate(query, retrieved)

    print("=" * 50)
    print("ANSWER:")
    print(result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  Source {s['source_num']}: {s['doc']} "
              f"page {s['page']} (relevance: {s['relevance']:.2f})")
    print(f"\nContext chunks used: {result['context_used']}")
    print("=" * 50)
    print("\nFull RAG pipeline works correctly")