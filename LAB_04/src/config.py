"""Simple configuration"""

from pathlib import Path
import os


class Settings:
    """Project settings"""
    
    def __init__(self):
        self.PROJECT_ROOT = Path(__file__).parent.parent
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.OUTPUT_DIR = self.PROJECT_ROOT / "output"
        self.AUDIO_DIR = self.DATA_DIR / "audio"
        
        # Create directories
        for d in [self.DATA_DIR, self.OUTPUT_DIR, self.AUDIO_DIR]:
            d.mkdir(exist_ok=True)
        
        # Defaults
        self.WHISPER_MODEL = "base"
        self.PIPER_VOICE = "en_US-lessac-medium"
        self.SAMPLE_RATE = 16000


def get_settings():
    """Get settings instance"""
    return Settings()

    """Get application settings."""
    return Settings()


def setup_directories():
    """Create necessary directories."""
    settings = get_settings()
    settings.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
