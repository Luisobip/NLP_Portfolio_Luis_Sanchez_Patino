"""Embedding and semantic search module."""

import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .document_loader import DocumentChunk


class EmbeddingEngine:
    """Handles document embeddings and semantic search."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding engine.

        Args:
            model_name: Name of the sentence-transformer model
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Storage for embeddings
        self.chunks: List[DocumentChunk] = []
        self.embeddings: np.ndarray = None  # Will be (n_chunks, embedding_dim)

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks and compute their embeddings.

        Args:
            chunks: List of DocumentChunk objects
        """
        # Extract text from chunks, filter out empty/invalid ones
        texts = []
        valid_chunks = []
        
        for chunk in chunks:
            if chunk.text and isinstance(chunk.text, str) and len(chunk.text.strip()) > 0:
                texts.append(chunk.text.strip())
                valid_chunks.append(chunk)
        
        if not texts:
            return

        # Compute embeddings
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        except Exception as e:
            print(f"Error encoding chunks: {e}")
            raise

        # Store chunks and embeddings
        self.chunks.extend(valid_chunks)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        print(f"✓ Total chunks in index: {len(self.chunks)}")
        print(f"✓ Embedding shape: {self.embeddings.shape}")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for relevant chunks using semantic similarity.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of (DocumentChunk, similarity_score) tuples
        """
        if len(self.chunks) == 0:
            return []

        # Embed query
        query_embedding = self.model.encode(query, convert_to_numpy=True)

        # Compute similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return chunks with scores
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            score = float(similarities[idx])
            results.append((chunk, score))

        return results

    def search_with_threshold(
        self, query: str, threshold: float = 0.3, top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search with minimum similarity threshold.

        Args:
            query: Query text
            threshold: Minimum similarity score (0-1)
            top_k: Maximum number of results to return

        Returns:
            List of (DocumentChunk, similarity_score) tuples above threshold
        """
        results = self.search(query, top_k=top_k * 2)  # Get more to filter
        filtered = [(chunk, score) for chunk, score in results if score >= threshold]
        return filtered[:top_k]

    def get_stats(self) -> Dict:
        """Get statistics about the embedding index."""
        if len(self.chunks) == 0:
            return {"num_chunks": 0, "num_documents": 0}

        doc_names = set(chunk.doc_name for chunk in self.chunks)
        return {
            "num_chunks": len(self.chunks),
            "num_documents": len(doc_names),
            "embedding_dim": self.embedding_dim,
            "model": self.model_name,
            "documents": list(doc_names),
        }

    def clear(self) -> None:
        """Clear all stored embeddings."""
        self.chunks = []
        self.embeddings = None
