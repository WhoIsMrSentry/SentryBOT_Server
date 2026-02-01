from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response
from pydantic import BaseModel
from services.manager import get_manager, ServiceManager

router = APIRouter(prefix="/tts", tags=["tts"])

class TTSRequest(BaseModel):
    text: str
    voice: str = None
    language: str = "tr"

@router.post("/speak")
async def speak(req: TTSRequest, mgr: ServiceManager = Depends(get_manager)):
    if not mgr.tts_service:
        raise HTTPException(status_code=503, detail="TTS service not enabled")
    
    try:
        # Use simple synthesize which manages temp files/bytes internally
        audio_bytes = mgr.tts_service.synthesize(req.text, voice=req.voice, language=req.language)
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health(mgr: ServiceManager = Depends(get_manager)):
    if not mgr.tts_service:
        return {"status": "disabled"}
    return mgr.tts_service.health_check()
