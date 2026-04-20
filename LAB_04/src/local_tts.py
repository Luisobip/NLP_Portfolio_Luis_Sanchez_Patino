"""Local Text-to-Speech using gTTS (Google Text-to-Speech)"""

from pathlib import Path


class LocalPiperTTS:
    """Local text-to-speech with gTTS"""
    
    def __init__(self, voice: str = "en_US-lessac-medium", output_dir: str = "output"):
        """Initialize TTS with gTTS
        
        Args:
            voice: Voice name (for compatibility, gTTS uses built-in voices)
            output_dir: Directory for output audio files
        """
        self.voice = voice
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def synthesize(self, text: str, output_file: str = None) -> str:
        """Convert text to speech using gTTS
        
        Args:
            text: Text to synthesize
            output_file: Output file path (auto-generated if None)
            
        Returns: Path to generated audio file
        """
        if output_file is None:
            output_file = self.output_dir / "output.mp3"
        else:
            output_file = self.output_dir / output_file
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            from gtts import gTTS
            
            # Create gTTS object
            tts = gTTS(text=text, lang="en", slow=False)
            
            # Save to file
            tts.save(str(output_file))
            
            return str(output_file)
        
        except Exception as e:
            raise RuntimeError(f"gTTS synthesis failed: {e}")
