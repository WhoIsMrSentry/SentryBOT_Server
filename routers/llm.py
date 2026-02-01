from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from services.manager import get_manager, ServiceManager
from config import settings

router = APIRouter(prefix="/llm", tags=["llm"])

class ChatRequest(BaseModel):
    query: str
    system: str = settings.llm_system_prompt

@router.post("/chat")
def chat(req: ChatRequest, mgr: ServiceManager = Depends(get_manager)):
    if not mgr.llm_service:
         raise HTTPException(status_code=503, detail="LLM service not enabled")
    
    try:
        resp = mgr.llm_service.chat(req.query, system_prompt=req.system)
        return {"response": resp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
