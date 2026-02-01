from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response
from pydantic import BaseModel
from services.manager import get_manager, ServiceManager
from config import settings

router = APIRouter(prefix="/tts", tags=["tts"])

class TTSRequest(BaseModel):
    text: str
    voice: str = None
    language: str = "tr"
    engine: str | None = None


class TTSConfigRequest(BaseModel):
    engine: str | None = None
    piper_model_path: str | None = None
    piper_cuda: bool | None = None
    xtts_model_name: str | None = None
    xtts_use_gpu: bool | None = None
    speaker_wav_path: str | None = None
    enabled: bool | None = None

@router.post("/speak")
async def speak(req: TTSRequest, mgr: ServiceManager = Depends(get_manager)):
    if not mgr.tts_service:
        raise HTTPException(status_code=503, detail="TTS service not enabled")
    
    try:
        svc = mgr.get_tts_service(req.engine)
        audio_bytes = svc.synthesize(req.text, voice=req.voice, language=req.language)
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health(mgr: ServiceManager = Depends(get_manager)):
    if not mgr.tts_service:
        return {"status": "disabled"}
    return mgr.tts_service.health_check()


@router.post("/config")
def update_config(req: TTSConfigRequest, mgr: ServiceManager = Depends(get_manager)):
    try:
        mgr.reload_tts(
            engine=req.engine,
            piper_model_path=req.piper_model_path,
            piper_cuda=req.piper_cuda,
            xtts_model_name=req.xtts_model_name,
            xtts_use_gpu=req.xtts_use_gpu,
            speaker_wav_path=req.speaker_wav_path,
            enabled=req.enabled,
        )
        return {"ok": True, "engine": settings.tts_engine, "enabled": settings.tts_enabled}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
