import logging
import requests
from typing import Dict, Any
from services.base import BaseService
from config import settings

logger = logging.getLogger(__name__)

class LLMService(BaseService):
    def __init__(self):
        self.base_url = settings.llm_base_url
        self.model = settings.llm_model

    def initialize(self):
        # Ollama is external, so we just check connection
        try:
            self.health_check()
        except Exception as e:
            logger.warning(f"Ollama might not be running: {e}")

    def chat(self, prompt: str, system_prompt: str = None, model: str = None, base_url: str = None) -> str:
        url = f"{(base_url or self.base_url)}/api/chat"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False
        }

        try:
            logger.info(f"LLM: Sending request to Ollama ({self.model})...")
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"LLM: Request failed: {e}")
            raise

    def health_check(self) -> dict:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return {"status": "online", "models": [m['name'] for m in resp.json().get('models', [])]}
        except Exception as e:
            return {"status": "offline", "error": str(e)}
