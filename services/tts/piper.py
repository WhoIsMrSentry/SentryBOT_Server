import logging
import io
import wave
import tempfile
import os
from services.base import BaseService
from config import settings

logger = logging.getLogger(__name__)

class PiperService(BaseService):
    def __init__(self):
        self.model = None
        self.model_path = settings.piper_model_path

    def initialize(self):
        try:
            from piper import PiperVoice
            if not os.path.exists(self.model_path):
                logger.error(f"Piper model not found at {self.model_path}")
                return
            
            self.model = PiperVoice.load(self.model_path, config_path=self.model_path + ".json", use_cuda=settings.piper_cuda)
            logger.info(f"Piper model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load Piper model: {e}")

    def synthesize(self, text: str, voice: str = None, language: str = "tr") -> bytes:
        if not self.model:
            raise RuntimeError("Piper model not initialized")
        
        try:
            # Audio chunks generator
            audio_chunks = self.model.synthesize(text)
            
            # Merge chunks (raw 16-bit PCM)
            pcm_data = b''.join(chunk.audio_int16_bytes for chunk in audio_chunks)
            
            # Convert to WAV
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(22050) # Standard for Piper medium models
                    wf.writeframes(pcm_data)
                return wav_buffer.getvalue()
        except Exception as e:
            import traceback
            logger.error(f"Piper synthesize failed: {e}")
            logger.error(traceback.format_exc())
            raise

    def health_check(self) -> dict:
        return {"status": "ready" if self.model else "error", "model": self.model_path}
