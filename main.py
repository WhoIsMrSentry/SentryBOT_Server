import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from services.manager import get_manager
from routers import llm, tts, vision
from config import settings
from gui_app import build_gui
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("server_app")

app = FastAPI(
    title="SentryBOT Server App",
    description="Backend processing for SentryBOT (LLM, TTS, Vision)"
)

# Include routers
app.include_router(llm.router)
app.include_router(tts.router)
app.include_router(vision.router)

# GUI
build_gui(app)

@app.on_event("startup")
def startup_event():
    logger.info("🚀 Starting SentryBOT Server Application...")
    if settings.start_services:
        try:
            mgr = get_manager()
            mgr.initialize_all()
            logger.info("✅ All core services initialized.")
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")

@app.get("/")
def root_redirect():
    return RedirectResponse(url="/ui")


@app.get("/api/status")
def api_status():
    return {
        "status": "online",
        "app": "SentryBOT Server",
        "services": {
            "llm": settings.llm_enabled,
            "tts": settings.tts_enabled,
            "vision": settings.vision_enabled
        }
    }

def start():
    """Entry point for script or console_scripts"""
    uvicorn.run(
        "main:app", 
        host=settings.server_host, 
        port=settings.server_port, 
        reload=False
    )

if __name__ == "__main__":
    start()
