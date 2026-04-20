"""Local Speech-to-Text using OpenAI Whisper"""

import whisper
from pathlib import Path


class LocalWhisperSTT:
    """Local speech-to-text with Whisper"""
    
    def __init__(self, model_name: str = "base"):
        """Load Whisper model
        
        Args:
            model_name: "tiny", "base", "small", "medium", or "large"
        """
        self.model = whisper.load_model(model_name)
        self.model_name = model_name
    
    def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio file to text
        
        Returns: {"text": "...", "language": "en", "segments": [...]}
        """
        result = self.model.transcribe(audio_path)
        return result
    
    def translate(self, audio_path: str) -> str:
        """Transcribe and translate to English"""
        result = self.model.transcribe(audio_path, task="translate")
        return result["text"]
