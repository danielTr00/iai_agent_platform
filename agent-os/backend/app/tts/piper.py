"""Lokales TTS via Piper (kein Key, CPU-tauglich).

Erzeugt WAV-PCM. Erwartet ein installiertes Piper-Modell (Name aus config).
"""
import asyncio
from typing import AsyncIterator

from .. import config


class PiperTTS:
    name = "piper"

    def __init__(self) -> None:
        self.voice = config.load()["tts"]["piper"]["voice"]

    async def available(self) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                "piper", "--help",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return proc.returncode == 0
        except FileNotFoundError:
            return False

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        proc = await asyncio.create_subprocess_exec(
            "piper", "--model", self.voice, "--output_file", "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        assert proc.stdin and proc.stdout
        proc.stdin.write(text.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()
        async for chunk in proc.stdout:
            yield chunk
        await proc.wait()
