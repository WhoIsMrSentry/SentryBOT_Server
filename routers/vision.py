from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from typing import List, Optional
from services.manager import get_manager, ServiceManager
from config import settings

router = APIRouter(prefix="/vision", tags=["vision"])

@router.post("/process")
async def process_image(
    file: UploadFile = File(...),
    mode: Optional[str] = None,
    modalities: Optional[List[str]] = None,
    mgr: ServiceManager = Depends(get_manager)
):
    if not mgr.vision_service:
        raise HTTPException(status_code=503, detail="Vision service not enabled")
    
    try:
        content = await file.read()
        results = mgr.vision_service.process_image(content, mode=mode, modalities=modalities)
        return {"ok": True, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health(mgr: ServiceManager = Depends(get_manager)):
    if not mgr.vision_service:
        return {"status": "disabled"}
    return mgr.vision_service.health_check()


@router.post("/config")
def update_config(
    yolo_model_path: Optional[str] = None,
    enabled: Optional[bool] = None,
    robot_gateway_url: Optional[str] = None,
    robot_vision_auth_token: Optional[str] = None,
    mgr: ServiceManager = Depends(get_manager),
):
    try:
        mgr.reload_vision(
            yolo_model_path=yolo_model_path,
            enabled=enabled,
            robot_gateway_url=robot_gateway_url,
            robot_vision_auth_token=robot_vision_auth_token,
        )
        return {
            "ok": True,
            "enabled": settings.vision_enabled,
            "yolo_model_path": settings.yolo_model_path,
            "robot_gateway_url": settings.robot_gateway_url,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
