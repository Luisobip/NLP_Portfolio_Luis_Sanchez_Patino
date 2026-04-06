"""Configuration settings for RAG system."""

import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "documents"
EMBEDDINGS_CACHE_DIR = DATA_DIR / "embeddings_cache"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)
EMBEDDINGS_CACHE_DIR.mkdir(exist_ok=True)

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# LLM Parameters
LLM_CONFIG = {
    "model": OLLAMA_MODEL,
    "temperature": 0.3,  # Lower temperature for more factual responses
    "top_p": 0.9,
    "num_ctx": 4096,  # Context window size
    "num_predict": 500,  # Max generation tokens
}

# Embedding Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Document Processing
CHUNK_SIZE = 512  # characters
CHUNK_OVERLAP = 50  # characters for context preservation

# Retrieval Configuration
TOP_K_RETRIEVAL = 3  # Number of document chunks to retrieve
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity for chunk relevance

# Prompt Templates
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided documents. 

Instructions:
- Answer questions using ONLY the provided context
- If the answer is not in the context, say "I don't have information about this in the provided documents"
- Always cite the source (document name) when providing information
- Be concise but informative
- Format lists and complex information clearly"""

RETRIEVAL_PROMPT_TEMPLATE = """Based on the following documents, answer the question:

Documents:
{context}

Question: {question}

Answer:"""

# Gradio Configuration
GRADIO_SHARE = False
GRADIO_DEBUG = False
