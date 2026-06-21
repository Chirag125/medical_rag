# Medical Multimodal RAG — Retrieval-Augmented Generation Pipeline

A retrieval-augmented generation system for question answering over clinical
and scientific literature, combining dense retrieval, cross-encoder reranking,
and grounded answer generation.

**Status: text pipeline complete and functional. Multimodal (image) retrieval
and full medical corpus integration in progress.**

---

## The Problem

Clinical and scientific documents combine narrative text, tables, and figures
in ways that text-only retrieval systems handle poorly. A retrieval system
that can jointly index and search across text and images — and ground its
answers with cited sources — is a more realistic fit for how clinicians and
researchers actually need to query literature.

This project builds that system incrementally: text-only retrieval first,
multimodal retrieval next, evaluated against open medical QA benchmarks.

---

## Architecture

```
Document (PDF)
      │
      ▼
┌─────────────────┐
│ document_loader │  pdfplumber — extracts text + images per page
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    chunker      │  Parent-child chunking:
│                 │  - Children (256 tok) → precise retrieval
│                 │  - Parents (512 tok)  → context for generation
│                 │  Context headers prepended (doc name + page)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    embedder     │  BGE-large-en-v1.5 (1024-dim)
│                 │  Stores in ChromaDB (separate parent/child collections)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   retriever     │  Stage 1: BGE bi-encoder → top-20 candidates
│                 │  Stage 2: cross-encoder (ms-marco-MiniLM) → top-5
│                 │  Fetches parent chunks for retrieved children
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   generator     │  Phi-3-mini-4k-instruct
│                 │  Grounded generation with source citations
└─────────────────┘
```

---

## Why Two-Stage Retrieval

A single dense retriever (bi-encoder) embeds the query and each document
independently — fast, but imprecise, because it never lets the query and
document "see" each other. A cross-encoder reads the query and document
together, producing far more accurate relevance scores — but it's too slow
to run against an entire corpus.

The fix: bi-encoder retrieves a wide net of candidates (top-20), cross-encoder
reranks only those candidates down to the final top-5. This is the same
pattern used in production search systems at scale.

## Why Parent-Child Chunking

Small chunks retrieve precisely (specific enough to match narrow queries) but
generate poorly (not enough context for the LLM to produce a complete answer).
Large chunks generate well but retrieve imprecisely (too much irrelevant text
dilutes the embedding).

The fix: index small child chunks for retrieval, but once a child is
retrieved, fetch and pass its larger parent chunk to the generator. Precision
where it matters (search), context where it matters (generation).

---

## Tech Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| PDF parsing | pdfplumber | Text + image extraction |
| Text splitting | LangChain RecursiveCharacterTextSplitter | Parent-child chunking |
| Text embeddings | BGE-large-en-v1.5 | Dense retrieval (1024-dim) |
| Vector store | ChromaDB | Persistent local vector storage |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Precision reranking |
| Generation | Phi-3-mini-4k-instruct | Grounded answer generation |
| (Planned) Image embeddings | CLIP ViT-L/14 | Multimodal retrieval |
| (Planned) Eval | RAGAS | Faithfulness + answer relevance |

---

## Project Structure

```
medical_rag/
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py    # PDF → pages (text + images)
│   │   ├── chunker.py            # parent-child chunking + context headers
│   │   └── embedder.py           # BGE embeddings → ChromaDB
│   ├── retrieval/
│   │   └── retriever.py          # two-stage retrieval (bi-encoder + reranker)
│   ├── generation/
│   │   └── generator.py          # Phi-3 grounded generation
│   └── evaluation/                # (planned) recall@k + RAGAS metrics
├── data/
│   ├── raw/                       # source PDFs (gitignored)
│   └── processed/chroma/          # vector store (gitignored)
└── requirements.txt
```

---

## Quickstart

```bash
pip install -r requirements.txt
```

### 1. Load and chunk a document

```bash
python -m src.ingestion.chunker
```

### 2. Embed and store in ChromaDB

```bash
python -m src.ingestion.embedder
```

### 3. Test retrieval (bi-encoder + reranking)

```bash
python -m src.retrieval.retriever
```

### 4. Run the full pipeline (retrieval + generation)

```bash
python -m src.generation.generation
```

---

## Example Output

**Query:** "What are the safety challenges of GPT-4?"

**Retrieval:** 20 candidates retrieved via BGE → reranked to top 5 via
cross-encoder. Reranking changed result order — a chunk ranked #2 by raw
embedding similarity dropped to #4 after reranking confirmed it was less
relevant to the specific question asked.

**Generated answer:**
> The safety challenges of GPT-4, as outlined in the provided context,
> include significant and novel safety challenges that necessitate careful
> study due to their potential societal impact... [structured into reliability
> issues, context window limitations, and bias/disinformation concerns]

**Sources cited:** 5 sources with relevance scores ranging 0.38–4.07,
all correctly traced to source document and page number.

---

## Design Decisions

**BGE over OpenAI embeddings.** BGE-large-en-v1.5 outperforms ada-002 on
the MTEB retrieval benchmark and runs fully locally with no API cost or
data privacy concerns — relevant for a medical document use case.

**Context header prepending.** Every chunk is prepended with its source
document name and page number before embedding. Without this, a chunk
saying "the model achieved 94% accuracy" loses all document context once
split — the header recovers it.

**Deterministic generation (`do_sample=False`).** For factual QA grounded
in retrieved context, deterministic generation is more appropriate than
sampling-based generation, which introduces unnecessary variance for a
task where there is a definite correct answer to verify against.

**Token-based truncation for children, character-based for parents.**
Children are embedded through BGE, which has a hard 512-token limit —
truncation must be token-aware. Parents are passed directly to the LLM
as raw text and don't need token-precise truncation, so character-based
truncation avoids subword decoding artifacts.

---

## Known Limitations (current state)

- Tested only on a single PDF (academic paper with imperfect text
  extraction — fused words from PDF generation, not corpus quality).
  Will resolve when running on properly formatted medical PDFs.
- Image retrieval (CLIP) not yet implemented — text-only currently.
- No formal evaluation yet — recall@k and RAGAS metrics planned next.
- Generation on CPU is slow (1-5 minutes per query with Phi-3-mini).
  GPU inference or 4-bit quantization would resolve this for production use.
- Low-confidence reranked results (negative cross-encoder scores) are not
  yet filtered out before being passed to the generator.

---

## Planned Next Steps

- [ ] Filter reranked results below a cross-encoder score threshold
- [ ] Add CLIP-based image embedding for multimodal retrieval
- [ ] Add BLIP-2 captioning for figures to bridge image/text embedding spaces
- [ ] Implement hypothetical question generation (HyDE) for improved recall
- [ ] Build evaluation harness: recall@5 on a held-out QA set, RAGAS
      faithfulness and answer relevance scoring
- [ ] Scale corpus from single test document to PMC-OA medical subset
- [ ] Add query rewriting feedback loop for low-confidence retrievals
- [ ] Deploy demo on HuggingFace Spaces

---

## References

- [BGE Embeddings (BAAI)](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [Cross-Encoder Reranking](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219)
- [HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)
- [RAGAS: Automated Evaluation of RAG Systems](https://github.com/explodinggradients/ragas)