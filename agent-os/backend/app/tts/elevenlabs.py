"""ElevenLabs TTS (Cloud, Streaming)."""
from typing import AsyncIterator

import httpx

from .. import config


class ElevenLabsTTS:
    name = "elevenlabs"

    def __init__(self) -> None:
        cfg = config.load()["tts"]["elevenlabs"]
        self.key = config.env("ELEVENLABS_API_KEY")
        self.voice_id = cfg["voice_id"]
        self.model_id = cfg["model_id"]

    async def available(self) -> bool:
        return bool(self.key)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        url = (f"https://api.elevenlabs.io/v1/text-to-speech/"
               f"{self.voice_id}/stream")
        payload = {"text": text, "model_id": self.model_id}
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST", url,
                headers={"xi-api-key": self.key, "accept": "audio/mpeg"},
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk
