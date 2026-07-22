"""STT-Registry + Fallback-Kette."""
from .. import config
from .elevenlabs import ElevenLabsSTT
from .local_whisper import LocalWhisperSTT

_REGISTRY = {
    "elevenlabs": ElevenLabsSTT,
    "local": LocalWhisperSTT,
}


async def pick():
    """Erstes verfuegbares Backend gemaess config.stt.chain."""
    chain = config.load()["stt"]["chain"]
    for name in chain:
        cls = _REGISTRY.get(name)
        if not cls:
            continue
        backend = cls()
        if await backend.available():
            return backend
    raise RuntimeError(f"Kein STT-Backend verfuegbar (chain={chain})")
