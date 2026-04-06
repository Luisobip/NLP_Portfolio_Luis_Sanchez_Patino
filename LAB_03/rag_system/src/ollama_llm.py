"""LLM integration module for Ollama."""

import requests
from typing import Generator, Optional
import json

from .config import OLLAMA_BASE_URL, LLM_CONFIG


class OllamaLLM:
    """Interface for Ollama local LLM."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = LLM_CONFIG["model"],
        temperature: float = LLM_CONFIG["temperature"],
    ):
        """
        Initialize Ollama LLM.

        Args:
            base_url: URL where Ollama is running
            model: Model name
            temperature: Generation temperature (0-1)
        """
        self.base_url = base_url
        self.model = model
        self.temperature = temperature

    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            return False

    def get_available_models(self) -> list:
        """Get list of available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            data = response.json()
            return [model["name"].split(":")[0] for model in data.get("models", [])]
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """
        Generate text using the LLM.

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt for context
            stream: Whether to stream response

        Returns:
            Generated text or generator of text chunks
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9,
                "num_ctx": 4096,
                "num_predict": 500,
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=stream,
            )

            if stream:
                return self._stream_response(response)
            else:
                data = response.json()
                return data.get("message", {}).get("content", "")

        except requests.exceptions.ConnectionError:
            return f"Error: Cannot connect to Ollama at {self.base_url}. Is it running?"
        except Exception as e:
            return f"Error: {str(e)}"

    def _stream_response(self, response):
        """Generator for streaming responses."""
        try:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data:
                        yield data["message"]["content"]
        except Exception as e:
            yield f"Error in streaming: {str(e)}"

    def generate_with_context(
        self,
        question: str,
        context: str,
        system_prompt: str,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """
        Generate answer based on question and context.

        Args:
            question: User question
            context: Retrieved document context
            system_prompt: System prompt
            stream: Whether to stream response

        Returns:
            Generated answer
        """
        prompt = f"""Context:
{context}

Question: {question}

Answer:"""
        return self.generate(prompt, system_prompt=system_prompt, stream=stream)
