"""STT-Backend-Protokoll (Vorbild: Hermes STTBackend)."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class STTBackend(Protocol):
    name: str

    async def available(self) -> bool:
        """Ist dieses Backend nutzbar (Key/Modell vorhanden)?"""
        ...

    async def transcribe(self, audio: bytes, mime: str = "audio/webm") -> str:
        """Rohes Audio -> Text."""
        ...
