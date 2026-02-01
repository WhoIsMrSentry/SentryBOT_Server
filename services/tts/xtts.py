import logging
import io
import os
import torch
import tempfile
from services.base import BaseService
from config import settings

logger = logging.getLogger(__name__)

class XTTSService(BaseService):
    def __init__(self):
        self.model = None
        self.model_name = settings.xtts_model_name
        self.device = "cuda" if settings.xtts_use_gpu and torch.cuda.is_available() else "cpu"

    def initialize(self):
        try:
            from TTS.api import TTS
            logger.info(f"Loading XTTS model {self.model_name} on {self.device}...")
            # progress_bar=False to avoid cluttering logs
            self.model = TTS(model_name=self.model_name, progress_bar=False).to(self.device)
            logger.info("XTTS model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load XTTS model: {e}")

    def synthesize(self, text: str, voice: str = None, language: str = "tr") -> bytes:
        if not self.model:
            raise RuntimeError("XTTS model not initialized")
        
        speaker_wav = voice
        if not speaker_wav:
             speaker_wav = settings.speaker_wav_path
        
        if not os.path.exists(speaker_wav):
            raise FileNotFoundError(f"Speaker wav not found: {speaker_wav}")

        # TTS.api saves to file usually, but we want bytes.
        # tts_to_file saves to disk. We can use a temp file.
        # Alternatively, deep access to model for direct tensor output is complex.
        # Stick to temp file for reliability with 'TTS' wrapper.
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            kwargs = {
                "text": text,
                "language": language,
                "file_path": tmp_path
            }
            try:
                self.model.tts_to_file(**kwargs, speaker_wav=speaker_wav)
            except TypeError as te:
                if "unexpected keyword argument 'speaker_wav'" in str(te):
                    logger.warning("XTTS: 'speaker_wav' not accepted, trying 'speaker'")
                    self.model.tts_to_file(**kwargs, speaker=speaker_wav)
                else:
                    raise
            
            with open(tmp_path, "rb") as f:
                data = f.read()
            return data
        except Exception as e:
            import traceback
            logger.error(f"XTTS synthesize failed: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def health_check(self) -> dict:
        return {"status": "ready" if self.model else "error", "device": self.device}
