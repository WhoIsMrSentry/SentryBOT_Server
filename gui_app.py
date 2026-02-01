import asyncio
import json
import logging
from typing import List, Optional

import httpx
from nicegui import ui

from config import settings
from services.manager import get_manager, ServiceManager

logger = logging.getLogger(__name__)


def _build_css() -> str:
    return """
    body {
        background: linear-gradient(135deg, #1a0005 0%, #4d0011 45%, #88001b 100%);
        color: #f6e8ec;
    }
    .card {
        background: rgba(16, 8, 10, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
    }
    .accent {
        color: #ffb3c0;
    }
    .btn-accent {
        background: linear-gradient(135deg, #88001b 0%, #b3002a 100%);
        color: #ffffff;
    }
    .muted {
        color: #d6c3c8;
        font-size: 0.9rem;
    }
    """


async def _fetch_json(url: str, method: str = "GET", payload: Optional[dict] = None) -> dict:
    async with httpx.AsyncClient(timeout=20) as client:
        if method == "POST":
            resp = await client.post(url, json=payload)
        else:
            resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


def build_gui(app):
    api_host = settings.server_host
    if api_host in ("0.0.0.0", "::"):
        api_host = "127.0.0.1"
    api_base = f"http://{api_host}:{settings.server_port}"

    @ui.page("/ui")
    def page():
        ui.add_head_html(f"<style>{_build_css()}</style>")
        mgr: ServiceManager = get_manager()

        ui.label("SentryBOT Server Control Panel").classes("text-2xl font-bold accent")
        ui.label("LLM • Vision • TTS • SSH").classes("muted")

        with ui.row().classes("w-full gap-6"):
            with ui.card().classes("card w-full").style("min-width: 320px"):
                ui.label("Sunucu Durumu").classes("text-lg font-bold")
                status_area = ui.textarea(value="", placeholder="Durumlar yükleniyor...").classes("w-full")
                status_area.props("readonly")

                async def refresh_status():
                    try:
                        root = await _fetch_json(f"{api_base}/api/status")
                        llm = await _fetch_json(f"{api_base}/llm/health")
                        tts = await _fetch_json(f"{api_base}/tts/health")
                        vision = await _fetch_json(f"{api_base}/vision/health")
                        status_area.value = json.dumps({
                            "server": root,
                            "llm": llm,
                            "tts": tts,
                            "vision": vision,
                        }, indent=2, ensure_ascii=False)
                    except Exception as e:
                        status_area.value = f"Durum alınamadı: {e}"

                ui.button("Yenile", on_click=refresh_status).classes("btn-accent")
                ui.timer(1.0, refresh_status, once=True)

            with ui.card().classes("card w-full").style("min-width: 320px"):
                ui.label("Ollama / LLM").classes("text-lg font-bold")
                model_select = ui.select(options=[], label="Model")
                system_prompt = ui.textarea(label="System Prompt", value=settings.llm_system_prompt).classes("w-full")
                prompt_input = ui.textarea(label="Kullanıcı Mesajı").classes("w-full")
                response_area = ui.textarea(label="Yanıt", value="").classes("w-full")
                response_area.props("readonly")

                async def refresh_models():
                    try:
                        data = await _fetch_json(f"{api_base}/llm/health")
                        models = data.get("models", [])
                        model_select.options = models
                        if models and not model_select.value:
                            model_select.value = models[0]
                    except Exception as e:
                        response_area.value = f"Model listesi alınamadı: {e}"

                async def send_prompt():
                    payload = {
                        "query": prompt_input.value,
                        "system": system_prompt.value,
                        "model": model_select.value,
                    }
                    try:
                        data = await _fetch_json(
                            f"{api_base}/llm/chat",
                            method="POST",
                            payload=payload,
                        )
                        response_area.value = data.get("response", "")
                    except Exception as e:
                        response_area.value = f"Hata: {e}"

                with ui.row():
                    ui.button("Modelleri Yenile", on_click=refresh_models).classes("btn-accent")
                    ui.button("Gönder", on_click=send_prompt).classes("btn-accent")
                ui.timer(1.0, refresh_models, once=True)

            with ui.card().classes("card w-full").style("min-width: 320px"):
                ui.label("TTS (Piper / XTTS)").classes("text-lg font-bold")
                engine_select = ui.select(options=["piper", "xtts"], value=settings.tts_engine, label="Engine")
                language_input = ui.input(label="Dil", value="tr")
                voice_input = ui.input(label="Voice / Speaker WAV", value="")
                text_input = ui.textarea(label="Metin").classes("w-full")
                tts_status = ui.textarea(label="Durum", value="").classes("w-full")
                tts_status.props("readonly")

                async def synthesize():
                    payload = {
                        "text": text_input.value,
                        "voice": voice_input.value or None,
                        "language": language_input.value or "tr",
                    }
                    try:
                        async with httpx.AsyncClient(timeout=60) as client:
                            resp = await client.post(
                                f"{api_base}/tts/speak",
                                json=payload,
                            )
                            resp.raise_for_status()
                            tts_status.value = f"Ses üretildi ({len(resp.content)} bytes)"
                    except Exception as e:
                        tts_status.value = f"Hata: {e}"

                async def apply_tts_config():
                    payload = {
                        "engine": engine_select.value,
                        "speaker_wav_path": voice_input.value or None,
                    }
                    try:
                        data = await _fetch_json(
                            f"{api_base}/tts/config",
                            method="POST",
                            payload=payload,
                        )
                        tts_status.value = json.dumps(data, indent=2, ensure_ascii=False)
                    except Exception as e:
                        tts_status.value = f"Hata: {e}"

                with ui.row():
                    ui.button("Ayarları Uygula", on_click=apply_tts_config).classes("btn-accent")
                    ui.button("Sentezle", on_click=synthesize).classes("btn-accent")

            with ui.card().classes("card w-full").style("min-width: 320px"):
                ui.label("Vision (Robot Stream)").classes("text-lg font-bold")
                stream_url = ui.input(label="Kamera Stream URL", value="")
                snapshot_url = ui.input(label="Tek Kare URL", value="")
                modalities_select = ui.select(options=["object", "face", "hand", "attributes"],
                                              value=["object", "face"], label="Modlar", multiple=True)
                vision_output = ui.textarea(label="Sonuç", value="").classes("w-full")
                vision_output.props("readonly")
                preview_img = ui.image().classes("w-full")

                def show_stream():
                    if stream_url.value:
                        preview_img.set_source(stream_url.value)

                async def process_snapshot():
                    if not snapshot_url.value:
                        vision_output.value = "Tek kare URL gerekli"
                        return
                    try:
                        async with httpx.AsyncClient(timeout=30) as client:
                            img_resp = await client.get(snapshot_url.value)
                            img_resp.raise_for_status()
                            files = {"file": ("frame.jpg", img_resp.content, "image/jpeg")}
                            params = [("modalities", m) for m in (modalities_select.value or [])]
                            resp = await client.post(
                                f"{api_base}/vision/process",
                                files=files,
                                params=params,
                            )
                            resp.raise_for_status()
                            vision_output.value = json.dumps(resp.json(), indent=2, ensure_ascii=False)
                    except Exception as e:
                        vision_output.value = f"Hata: {e}"

                with ui.row():
                    ui.button("Stream Göster", on_click=show_stream).classes("btn-accent")
                    ui.button("Kareyi İşle", on_click=process_snapshot).classes("btn-accent")

            with ui.card().classes("card w-full").style("min-width: 320px"):
                ui.label("SSH Mini Terminal").classes("text-lg font-bold")
                host_input = ui.input(label="Host", value="")
                port_input = ui.number(label="Port", value=22)
                user_input = ui.input(label="User", value="pi")
                password_input = ui.input(label="Password", password=True, password_toggle_button=True)
                key_input = ui.input(label="Key Path (opsiyonel)")
                cmd_input = ui.input(label="Komut", value="uname -a")
                terminal_output = ui.textarea(label="Çıktı", value="").classes("w-full")
                terminal_output.props("readonly")

                async def run_ssh():
                    import paramiko

                    def _execute():
                        client = paramiko.SSHClient()
                        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                        try:
                            if key_input.value:
                                client.connect(
                                    hostname=host_input.value,
                                    port=int(port_input.value or 22),
                                    username=user_input.value,
                                    key_filename=key_input.value,
                                    timeout=10,
                                )
                            else:
                                client.connect(
                                    hostname=host_input.value,
                                    port=int(port_input.value or 22),
                                    username=user_input.value,
                                    password=password_input.value,
                                    timeout=10,
                                )
                            _, stdout, stderr = client.exec_command(cmd_input.value, timeout=20)
                            out = stdout.read().decode("utf-8", errors="ignore")
                            err = stderr.read().decode("utf-8", errors="ignore")
                            return out, err
                        finally:
                            client.close()

                    try:
                        out, err = await asyncio.to_thread(_execute)
                        terminal_output.value = f"{out}\n{err}".strip()
                    except Exception as e:
                        terminal_output.value = f"SSH Hatası: {e}"

                ui.button("Komut Çalıştır", on_click=run_ssh).classes("btn-accent")

    ui.run_with(app)


__all__ = ["build_gui"]
