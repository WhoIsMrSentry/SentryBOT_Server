from services.tts.piper import PiperService
from services.tts.xtts import XTTSService
from services.llm import LLMService
from services.vision_service import VisionService
from config import settings
import logging

logger = logging.getLogger(__name__)

class ServiceManager:
    def __init__(self):
        self.settings = settings
        self.tts_service = None
        self.tts_services = {}
        self.llm_service = None
        self.vision_service = None

    def initialize_all(self):
        # LLM
        logger.info("Initializing LLM Service...")
        self.llm_service = LLMService()
        self.llm_service.initialize()

        # TTS
        if self.settings.tts_enabled:
            if self.settings.tts_preload_all:
                logger.info("Initializing TTS Services (piper + xtts)...")
                self._init_tts_engine("piper")
                self._init_tts_engine("xtts")
            else:
                logger.info(f"Initializing TTS Service ({self.settings.tts_engine})...")
                self._init_tts_engine(self.settings.tts_engine)
            self.tts_service = self.tts_services.get(self.settings.tts_engine)

        # Vision
        if self.settings.vision_enabled:
            logger.info("Initializing Vision Service...")
            self.vision_service = VisionService()
            self.vision_service.initialize()
        
        logger.info("All services initialized.")

    def _init_tts_engine(self, engine: str):
        eng = (engine or "").lower()
        if eng in self.tts_services:
            return self.tts_services[eng]
        if eng == "xtts":
            svc = XTTSService()
        else:
            svc = PiperService()
        svc.initialize()
        self.tts_services[eng] = svc
        return svc

    def get_tts_service(self, engine: str | None = None):
        eng = (engine or self.settings.tts_engine or "piper").lower()
        return self._init_tts_engine(eng)

    def reload_llm(self, model: str = None, base_url: str = None, system_prompt: str = None, enabled: bool = None):
        if enabled is not None:
            self.settings.llm_enabled = enabled
        if model:
            self.settings.llm_model = model
        if base_url:
            self.settings.llm_base_url = base_url
        if system_prompt:
            self.settings.llm_system_prompt = system_prompt

        if not self.settings.llm_enabled:
            self.llm_service = None
            return

        self.llm_service = LLMService()
        self.llm_service.initialize()

    def reload_tts(
        self,
        engine: str = None,
        piper_model_path: str = None,
        piper_cuda: bool = None,
        xtts_model_name: str = None,
        xtts_use_gpu: bool = None,
        speaker_wav_path: str = None,
        enabled: bool = None,
    ):
        if enabled is not None:
            self.settings.tts_enabled = enabled
        if engine:
            self.settings.tts_engine = engine
        if piper_model_path:
            self.settings.piper_model_path = piper_model_path
        if piper_cuda is not None:
            self.settings.piper_cuda = piper_cuda
        if xtts_model_name:
            self.settings.xtts_model_name = xtts_model_name
        if xtts_use_gpu is not None:
            self.settings.xtts_use_gpu = xtts_use_gpu
        if speaker_wav_path:
            self.settings.speaker_wav_path = speaker_wav_path

        if not self.settings.tts_enabled:
            self.tts_service = None
            return

        self.tts_services = {}
        if self.settings.tts_preload_all:
            self._init_tts_engine("piper")
            self._init_tts_engine("xtts")
        else:
            self._init_tts_engine(self.settings.tts_engine)
        self.tts_service = self.tts_services.get(self.settings.tts_engine)

    def reload_vision(
        self,
        yolo_model_path: str = None,
        enabled: bool = None,
        robot_gateway_url: str = None,
        robot_vision_auth_token: str = None,
    ):
        if enabled is not None:
            self.settings.vision_enabled = enabled
        if yolo_model_path:
            self.settings.yolo_model_path = yolo_model_path
        if robot_gateway_url is not None:
            self.settings.robot_gateway_url = robot_gateway_url
        if robot_vision_auth_token is not None:
            self.settings.robot_vision_auth_token = robot_vision_auth_token

        if not self.settings.vision_enabled:
            self.vision_service = None
            return

        self.vision_service = VisionService()
        self.vision_service.initialize()

manager = ServiceManager()

def get_manager():
    return manager
