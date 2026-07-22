"""ElevenLabs Scribe STT (Cloud)."""
import httpx

from .. import config


class ElevenLabsSTT:
    name = "elevenlabs"

    def __init__(self) -> None:
        self.key = config.env("ELEVENLABS_API_KEY")
        self.model_id = config.load()["stt"]["elevenlabs"]["model_id"]

    async def available(self) -> bool:
        return bool(self.key)

    async def transcribe(self, audio: bytes, mime: str = "audio/webm") -> str:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": self.key},
                data={"model_id": self.model_id},
                files={"file": ("audio", audio, mime)},
            )
            resp.raise_for_status()
            return resp.json().get("text", "")
