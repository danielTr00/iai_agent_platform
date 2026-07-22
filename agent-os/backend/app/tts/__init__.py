"""TTS-Registry + Fallback-Kette."""
from .. import config
from .elevenlabs import ElevenLabsTTS
from .piper import PiperTTS

_REGISTRY = {
    "elevenlabs": ElevenLabsTTS,
    "piper": PiperTTS,
}


async def pick():
    chain = config.load()["tts"]["chain"]
    for name in chain:
        cls = _REGISTRY.get(name)
        if not cls:
            continue
        backend = cls()
        if await backend.available():
            return backend
    raise RuntimeError(f"Kein TTS-Backend verfuegbar (chain={chain})")
