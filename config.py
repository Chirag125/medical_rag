# config.py
import os

# Paths
DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"
CHROMA_PATH = "data/processed/chroma"
OUTPUTS = "outputs"

# Models
TEXT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
IMAGE_EMBED_MODEL = "openai/clip-vit-large-patch14"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATOR_MODEL = "microsoft/Phi-3-mini-4k-instruct"

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
PARENT_CHUNK_SIZE = 1024
TOP_K_RETRIEVAL = 20
TOP_K_RERANK = 5

# Eval
RECALL_K = 5