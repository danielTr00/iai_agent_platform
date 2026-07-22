"""TTS-Backend-Protokoll (Vorbild: Hermes TTSBackend)."""
from typing import AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class TTSBackend(Protocol):
    name: str

    async def available(self) -> bool:
        ...

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Text -> Audio-Frames (streamend, MP3/PCM je nach Backend)."""
        ...
