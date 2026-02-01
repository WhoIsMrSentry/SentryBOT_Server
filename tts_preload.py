import time
import logging
import os
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tts_preload")

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
if not settings.xtts_use_gpu:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def main():
    try:
        from piper import PiperVoice
        if settings.piper_model_path:
            logger.info("Loading Piper model: %s", settings.piper_model_path)
            PiperVoice.load(settings.piper_model_path, config_path=settings.piper_model_path + ".json", use_cuda=False)
    except Exception as e:
        logger.warning("Piper preload failed: %s", e)

    try:
        from TTS.api import TTS
        try:
            import torch
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import XttsAudioConfig
            from TTS.config.shared_configs import BaseDatasetConfig
            torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig])
        except Exception:
            pass
        if settings.xtts_use_gpu:
            logger.info("XTTS preload atlandı (GPU kullanılacak).")
        else:
            logger.info("Loading XTTS model...")
            TTS(model_name=settings.xtts_model_name, progress_bar=False, gpu=settings.xtts_use_gpu)
    except Exception as e:
        logger.warning("XTTS preload failed: %s", e)

    logger.info("TTS preload ready. Keep window open.")
    while True:
        time.sleep(5)


if __name__ == "__main__":
    main()
