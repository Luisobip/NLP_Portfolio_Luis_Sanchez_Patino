"""Gradio interface for RAG system."""

import gradio as gr
import os
from .rag_pipeline import RAGPipeline
from .config import GRADIO_SHARE, GRADIO_DEBUG


class RAGInterface:
    """Gradio interface for RAG system."""

    def __init__(self):
        """Initialize the interface."""
        self.pipeline = RAGPipeline()

    def upload_files(self, files):
        """Handle file uploads - returns status string only."""
        try:
            if not files:
                return "No files uploaded."

            file_contents = {}
            for file in files:
                filename = os.path.basename(file.name)
                with open(file.name, "rb") as f:
                    file_contents[filename] = f.read()

            stats = self.pipeline.add_documents_from_bytes(file_contents)
            num_docs = stats.get('num_documents', 0)
            num_chunks = stats.get('num_chunks', 0)
            msg = f"✓ Loaded {num_docs} documents with {num_chunks} chunks."
            return msg
        except Exception as e:
            return f"✗ Error: {str(e)}"

    def get_answer(self, question):
        """Get answer - returns answer text only."""
        try:
            if not self.pipeline.get_stats().get("num_chunks", 0):
                return "Please upload documents first."

            if not question.strip():
                return "Please enter a question."

            result = self.pipeline.answer_question(question, stream=False)
            
            if isinstance(result, str):
                return result
            
            if isinstance(result, dict):
                answer = result.get("answer", "No answer generated")
                return answer if isinstance(answer, str) else str(answer)
            
            return "Unexpected response format"
        except Exception as e:
            return f"Error: {str(e)}"

    def get_sources(self, question):
        """Get sources - returns sources text only."""
        try:
            if not self.pipeline.get_stats().get("num_chunks", 0):
                return "No documents loaded"

            if not question.strip():
                return ""

            result = self.pipeline.answer_question(question, stream=False)
            
            if isinstance(result, str) or not isinstance(result, dict):
                return ""

            metadata = result.get("metadata", [])
            if not metadata:
                return "No sources retrieved"

            sources_text = "Retrieved Sources:\n"
            for i, chunk_info in enumerate(metadata, 1):
                doc = chunk_info.get('document', 'Unknown')
                score = chunk_info.get('score', 0)
                sources_text += f"{i}. {doc} ({score:.1%})\n"

            return sources_text if sources_text != "Retrieved Sources:\n" else "No sources"
        except Exception as e:
            return f"Error retrieving sources: {str(e)}"

    def build_interface(self):
        """Build the Gradio interface."""
        with gr.Blocks(title="RAG Q&A System") as demo:
            gr.Markdown("# RAG-Based Document Q&A")

            file_upload = gr.File(label="Upload Documents", file_count="multiple")
            upload_button = gr.Button("Load Documents")
            upload_status = gr.Textbox(label="Status", interactive=False, lines=2)

            question_input = gr.Textbox(label="Question", lines=2)
            answer_button = gr.Button("Get Answer")
            answer_output = gr.Textbox(label="Answer", interactive=False, lines=5)
            sources_output = gr.Textbox(label="Sources", interactive=False, lines=3)

            # Simple callbacks that return only strings
            upload_button.click(
                fn=self.upload_files,
                inputs=file_upload,
                outputs=upload_status,
                queue=False
            )

            answer_button.click(
                fn=self.get_answer,
                inputs=question_input,
                outputs=answer_output,
                queue=False
            )
            
            answer_button.click(
                fn=self.get_sources,
                inputs=question_input,
                outputs=sources_output,
                queue=False
            )

        return demo


def launch_app(share=GRADIO_SHARE, debug=GRADIO_DEBUG):
    """Launch the RAG application."""
    interface = RAGInterface()
    demo = interface.build_interface()
    demo.launch(share=share, debug=debug, server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    launch_app()
