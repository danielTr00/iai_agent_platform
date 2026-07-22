"""Lokales STT via faster-whisper (kein Key noetig)."""
import asyncio
import tempfile

from .. import config


class LocalWhisperSTT:
    name = "local"

    def __init__(self) -> None:
        cfg = config.load()["stt"]["local"]
        self.model_size = cfg.get("model", "base")
        self.device = cfg.get("device", "cpu")
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # lazy: schwerer Import
            compute = "int8" if self.device == "cpu" else "float16"
            self._model = WhisperModel(self.model_size, device=self.device,
                                       compute_type=compute)
        return self._model

    async def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    async def transcribe(self, audio: bytes, mime: str = "audio/webm") -> str:
        def _work() -> str:
            model = self._ensure_model()
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as tmp:
                tmp.write(audio)
                tmp.flush()
                segments, _ = model.transcribe(tmp.name)
                return " ".join(s.text for s in segments).strip()

        return await asyncio.to_thread(_work)
