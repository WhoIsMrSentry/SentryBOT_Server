import os
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    # Core
    start_services: bool = True  # Auto-start services on launch
    server_host: str = "0.0.0.0"
    server_port: int = 5000

    # LLM
    llm_enabled: bool = True
    llm_model: str = "llama3.2:3b"
    llm_base_url: str = "http://127.0.0.1:11435"
    llm_system_prompt: str = "You are SentryBOT, an advanced AI robot assistant. Be concise, helpful, and maintain your friendly robot personality."

    # TTS
    tts_enabled: bool = True
    tts_engine: str = "xtts"  # 'piper' or 'xtts'
    
    # Piper
    piper_model_path: str = r"C:\Users\emirh\OneDrive\Masaüstü\PiperTTS\GLaDOS\glados_piper_medium.onnx"
    # Note: piper binary path might be needed if not in env
    piper_cuda: bool = False

    # XTTS
    xtts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    xtts_use_gpu: bool = True
    # Default speaker for XTTS if none provided
    speaker_wav_path: str = r"c:\Users\emirh\SentryBOT\server_app\tts\TTS\xTTS\ses_kullanılabilir.wav" 

    # Vision
    vision_enabled: bool = True
    yolo_model_path: str = "yolov8n.pt"

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
