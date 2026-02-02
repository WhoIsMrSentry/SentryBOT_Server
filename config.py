import os
try:
    from pydantic_settings import BaseSettings
except Exception:
    # Pydantic v2 compatibility fallback
    from pydantic.v1 import BaseSettings

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
    tts_preload_all: bool = True
    
    # Piper
    _base_dir: str = os.path.dirname(__file__)
    piper_model_path: str = os.path.join(_base_dir, "tts", "TTS", "PiperTTS", "GLaDOS", "glados_piper_medium.onnx")
    # Note: piper binary path might be needed if not in env
    piper_cuda: bool = False

    # XTTS
    xtts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    xtts_use_gpu: bool = True
    xtts_preload_on_start: bool = True
    xtts_max_chars_per_chunk: int = 200
    # Default speaker for XTTS if none provided
    speaker_wav_path: str = "" 

    # Vision
    vision_enabled: bool = True
    yolo_model_path: str = "yolov8n.pt"
    vision_face_model_path: str = os.path.join(_base_dir, "models", "face_detection_yunet_2023mar.onnx")
    vision_age_model_path: str = os.path.join(_base_dir, "models", "age_net.caffemodel")
    vision_age_proto_path: str = os.path.join(_base_dir, "models", "age_deploy.prototxt")
    vision_emotion_model_path: str = os.path.join(_base_dir, "models", "emotion-ferplus-8.onnx")
    vision_face_cascade_path: str = os.path.join(_base_dir, "models", "haarcascade_frontalface_default.xml")
    vision_face_dataset_dir: str = os.path.join(_base_dir, "models", "faces")
    vision_face_recognize_threshold: float = 70.0
    vision_owner_name: str = "owner"
    vision_face_encodings_path: str = os.path.join(_base_dir, "models", "encodings.pickle")
    vision_face_encoding_tolerance: float = 0.6

    # Robot Vision Push
    robot_gateway_url: str = ""  # e.g., http://ROBOT_IP:8080
    robot_vision_auth_token: str = ""  # optional X-Auth-Token

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
