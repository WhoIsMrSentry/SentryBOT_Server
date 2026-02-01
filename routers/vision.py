from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from typing import List, Optional
from services.manager import get_manager, ServiceManager

router = APIRouter(prefix="/vision", tags=["vision"])

@router.post("/process")
async def process_image(
    file: UploadFile = File(...),
    modalities: Optional[List[str]] = None,
    mgr: ServiceManager = Depends(get_manager)
):
    if not mgr.vision_service:
        raise HTTPException(status_code=503, detail="Vision service not enabled")
    
    try:
        content = await file.read()
        results = mgr.vision_service.process_image(content, modalities=modalities)
        return {"ok": True, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health(mgr: ServiceManager = Depends(get_manager)):
    if not mgr.vision_service:
        return {"status": "disabled"}
    return mgr.vision_service.health_check()
