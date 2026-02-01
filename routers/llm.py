from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from services.manager import get_manager, ServiceManager
from config import settings

router = APIRouter(prefix="/llm", tags=["llm"])

class ChatRequest(BaseModel):
    query: str
    system: str = settings.llm_system_prompt
    model: str | None = None
    base_url: str | None = None

@router.post("/chat")
def chat(req: ChatRequest, mgr: ServiceManager = Depends(get_manager)):
    if not mgr.llm_service:
         raise HTTPException(status_code=503, detail="LLM service not enabled")
    
    try:
        resp = mgr.llm_service.chat(
            req.query,
            system_prompt=req.system,
            model=req.model,
            base_url=req.base_url,
        )
        return {"response": resp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class LLMConfigRequest(BaseModel):
    model: str | None = None
    base_url: str | None = None
    system_prompt: str | None = None
    enabled: bool | None = None


@router.post("/config")
def update_config(req: LLMConfigRequest, mgr: ServiceManager = Depends(get_manager)):
    try:
        mgr.reload_llm(
            model=req.model,
            base_url=req.base_url,
            system_prompt=req.system_prompt,
            enabled=req.enabled,
        )
        return {
            "ok": True,
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
            "enabled": settings.llm_enabled,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health(mgr: ServiceManager = Depends(get_manager)):
    if not mgr.llm_service:
        return {"status": "disabled"}
    return mgr.llm_service.health_check()
