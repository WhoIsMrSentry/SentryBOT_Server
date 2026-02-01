# SentryBOT Server App

Modern, modular backend for offloading heavy processing from Robot/Jetson to a PC.

## Modules

### LLM (Ollama)
Wraps Ollama API to provide personality-driven responses using `llama3.2:3b`.

### TTS
Supports dual engines:
- **Piper**: Fast, local synthesis (includes GLaDOS personality support).
- **XTTS**: High-quality voice cloning (Optimized to keep model in memory).

### Vision
Unified vision processing including:
- **Objects**: YOLOv8.
- **Hands**: MediaPipe (Gesture ready).
- **Faces**: DeepFace (Recognition, Age, Gender, Emotion/Mood).

## Installation

1. Install [Ollama](https://ollama.com/) and pull `llama3.2:3b`.
2. Install dependencies:
   ```bash
   pip install -r server_app/requirements.txt
   ```
3. Configure `server_app/config.py` (Paths to ONNX models, speaker files, etc.).

## Running

Run the main module from the project root:
```bash
python -m server_app.main
```
Or use the provided `start_server.bat`.

## GUI

PyQt6 masaüstü GUI:

```bash
python qt_main.py
```
