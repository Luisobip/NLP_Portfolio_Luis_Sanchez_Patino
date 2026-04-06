"""RAG pipeline orchestration."""

from typing import List, Tuple, Dict, Generator, Optional
from .document_loader import DocumentLoader, DocumentChunk
from .embedding_engine import EmbeddingEngine
from .ollama_llm import OllamaLLM
from .config import (
    EMBEDDING_MODEL,
    TOP_K_RETRIEVAL,
    SIMILARITY_THRESHOLD,
    SYSTEM_PROMPT,
)


class RAGPipeline:
    """Complete RAG (Retrieval-Augmented Generation) pipeline."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = EMBEDDING_MODEL,
        top_k: int = TOP_K_RETRIEVAL,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):
        """
        Initialize RAG pipeline.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: Embedding model name
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity score
        """
        self.document_loader = DocumentLoader(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.embedding_engine = EmbeddingEngine(model_name=embedding_model)
        self.llm = OllamaLLM()

        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        # Conversation history for context
        self.conversation_history = []

    def add_documents(self, documents: List[str]) -> Dict:
        """
        Add documents to the pipeline.

        Args:
            documents: List of file paths or document texts

        Returns:
            Statistics about loaded documents
        """
        all_chunks = []

        for doc in documents:
            try:
                chunks = self.document_loader.load_and_chunk_file(doc)
                all_chunks.extend(chunks)
                print(f"✓ Processed {doc}: {len(chunks)} chunks")
            except Exception as e:
                print(f"✗ Error processing {doc}: {e}")

        if all_chunks:
            self.embedding_engine.add_chunks(all_chunks)

        return self.embedding_engine.get_stats()

    def add_documents_from_bytes(
        self, file_contents: Dict[str, bytes]
    ) -> Dict:
        """
        Add documents from bytes (for web uploads).

        Args:
            file_contents: Dict of {filename: file_bytes}

        Returns:
            Statistics about loaded documents
        """
        all_chunks = []

        for filename, content in file_contents.items():
            try:
                chunks = self.document_loader.load_from_bytes(content, filename)
                all_chunks.extend(chunks)
                print(f"✓ Loaded {filename}: {len(chunks)} chunks")
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")

        if all_chunks:
            self.embedding_engine.add_chunks(all_chunks)

        return self.embedding_engine.get_stats()

    def retrieve(self, question: str) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve relevant document chunks for a question.

        Args:
            question: User question

        Returns:
            List of (DocumentChunk, similarity_score) tuples
        """
        results = self.embedding_engine.search_with_threshold(
            question, threshold=self.similarity_threshold, top_k=self.top_k
        )
        return results

    def format_context(
        self, retrieved_chunks: List[Tuple[DocumentChunk, float]]
    ) -> Tuple[str, List[Dict]]:
        """
        Format retrieved chunks into context string.

        Args:
            retrieved_chunks: Retrieved (chunk, score) tuples

        Returns:
            Tuple of (formatted_context_string, chunk_metadata_list)
        """
        context_parts = []
        metadata = []

        for i, (chunk, score) in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"[Document {i}] {chunk.doc_name}\n{chunk.text}\n"
            )
            metadata.append(
                {
                    "document": chunk.doc_name,
                    "score": float(score),
                    "chunk_id": chunk.chunk_id,
                    "text_preview": chunk.text[:100] + "...",
                }
            )

        context = "\n---\n".join(context_parts)
        return context, metadata

    def answer_question(
        self,
        question: str,
        stream: bool = False,
    ) -> Dict:
        """
        Answer a question using RAG.

        Args:
            question: User question
            stream: Whether to stream the response

        Returns:
            Dict with answer, retrieved_chunks, and metadata
        """
        # Check LLM connection
        if not self.llm.check_connection():
            return {
                "answer": "Error: Ollama is not running. Please start Ollama before using the system.",
                "retrieved_chunks": [],
                "metadata": {"error": "Ollama connection failed"},
            }

        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(question)

        if not retrieved_chunks:
            return {
                "answer": "I couldn't find any relevant information in the provided documents.",
                "retrieved_chunks": [],
                "metadata": {"message": "No chunks retrieved"},
            }

        # Format context
        context, metadata = self.format_context(retrieved_chunks)

        # Generate answer
        if stream:
            answer_generator = self.llm.generate_with_context(
                question=question,
                context=context,
                system_prompt=SYSTEM_PROMPT,
                stream=True,
            )
            return {
                "answer": answer_generator,
                "retrieved_chunks": retrieved_chunks,
                "metadata": metadata,
                "context": context,
            }
        else:
            answer = self.llm.generate_with_context(
                question=question,
                context=context,
                system_prompt=SYSTEM_PROMPT,
                stream=False,
            )
            return {
                "answer": answer,
                "retrieved_chunks": retrieved_chunks,
                "metadata": metadata,
                "context": context,
            }

    def clear(self) -> None:
        """Clear all documents and embeddings."""
        self.embedding_engine.clear()
        self.conversation_history = []

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return self.embedding_engine.get_stats()
