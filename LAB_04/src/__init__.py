"""LAB_04: Speech Processing System"""

from .config import Settings, get_settings
from .local_stt import LocalWhisperSTT
from .local_tts import LocalPiperTTS
from .external_apis import ExternalSTT, ExternalTTS
from .advanced_tasks import AdvancedTasks

__all__ = [
    "get_settings",
    "Settings",
    "LocalWhisperSTT",
    "LocalPiperTTS",
    "ExternalSTT",
    "ExternalTTS",
    "AdvancedTasks",
]
