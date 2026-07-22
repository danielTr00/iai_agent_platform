"""Satz-Chunker + TTS-Streaming (aus dem Hermes-Muster).

Claude liefert Text-Deltas. Wir sammeln bis Satzende (min. N Zeichen),
strippen Markdown/`<think>` und synthetisieren satzweise -> geringe Latenz.
"""
import re
from typing import AsyncIterator

from . import config

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_MD = re.compile(r"[*_`#>]+")
_SENT_END = re.compile(r"[.!?…]\s|\n")


def _clean(text: str) -> str:
    cfg = config.load()["tts"]
    text = _THINK.sub("", text)
    if cfg.get("strip_markdown", True):
        text = _MD.sub("", text)
    return text


def sentence_chunks(buffer: str) -> tuple[list[str], str]:
    """Zerlegt buffer in vollstaendige Saetze + Rest. Rest bleibt im Puffer."""
    min_chars = config.load()["tts"].get("min_chunk_chars", 20)
    chunks: list[str] = []
    while True:
        m = _SENT_END.search(buffer)
        if not m:
            break
        cut = m.end()
        candidate = buffer[:cut].strip()
        if len(candidate) >= min_chars:
            chunks.append(candidate)
            buffer = buffer[cut:]
        else:
            break
    return chunks, buffer


async def stream_tts(text_deltas: AsyncIterator[str], tts_backend) -> AsyncIterator[bytes]:
    """Nimmt Text-Deltas, gibt Audio-Frames aus."""
    buffer = ""
    async for delta in text_deltas:
        buffer += delta
        chunks, buffer = sentence_chunks(buffer)
        for chunk in chunks:
            spoken = _clean(chunk)
            if spoken.strip():
                async for audio in tts_backend.synthesize(spoken):
                    yield audio
    # Rest ausspielen
    tail = _clean(buffer).strip()
    if tail:
        async for audio in tts_backend.synthesize(tail):
            yield audio
