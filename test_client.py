import requests
import time
import os

BASE_URL = "http://localhost:5000"

def test_root():
    print("[TEST] Testing Root Endpoint...")
    try:
        resp = requests.get(f"{BASE_URL}/")
        print(f"Response: {resp.json()}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_llm():
    print("\n[TEST] Testing LLM Chat Endpoint...")
    payload = {
        "query": "Merhaba SentryBOT, nasılsın?",
        "system": "Sen SentryBOT'sun, yardımsever bir robotsun."
    }
    try:
        resp = requests.post(f"{BASE_URL}/llm/chat", json=payload)
        print(f"Response: {resp.json()}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_tts(engine="piper"):
    print(f"\n[TEST] Testing TTS Speak Endpoint ({engine})...")
    # Note: XTTS might need a speaker_wav if the default in config is missing
    payload = {
        "text": "Merhaba, test mesajı seslendiriliyor.",
        "language": "tr"
    }
    try:
        resp = requests.post(f"{BASE_URL}/tts/speak", json=payload)
        if resp.status_code == 200:
            filename = f"test_out_{engine}.wav"
            with open(filename, "wb") as f:
                f.write(resp.content)
            print(f"Success! Audio saved to {filename}")
            return True
        else:
            print(f"Error ({resp.status_code}): {resp.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("--- SentryBOT Server Integration Tests ---")
    if test_root():
        test_llm()
        # Test Piper
        test_tts(engine="piper")
        # Test XTTS (if enabled)
        # Note: we might need to change settings to test XTTS properly
        # but let's try calling it anyway or just focus on the active one
    else:
        print("Server is not running or unreachable at localhost:5000")
