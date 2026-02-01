from services.tts.piper import PiperService
from services.tts.xtts import XTTSService
from services.llm import LLMService
from services.vision import VisionService
from config import settings
import logging

logger = logging.getLogger(__name__)

class ServiceManager:
    def __init__(self):
        self.settings = settings
        self.tts_service = None
        self.llm_service = None
        self.vision_service = None

    def initialize_all(self):
        # LLM
        logger.info("Initializing LLM Service...")
        self.llm_service = LLMService()
        self.llm_service.initialize()

        # TTS
        if self.settings.tts_enabled:
            logger.info(f"Initializing TTS Service ({self.settings.tts_engine})...")
            if self.settings.tts_engine == "xtts":
                self.tts_service = XTTSService()
            else:
                self.tts_service = PiperService()
            self.tts_service.initialize()

        # Vision
        if self.settings.vision_enabled:
            logger.info("Initializing Vision Service...")
            self.vision_service = VisionService()
            self.vision_service.initialize()
        
        logger.info("All services initialized.")

manager = ServiceManager()

def get_manager():
    return manager
