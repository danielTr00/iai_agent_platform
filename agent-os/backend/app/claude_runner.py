"""Bindet Claude Code an das Harness.

Treibt die Claude Code CLI im `claude-core`-Container per `docker exec` und parst
den `--output-format stream-json`-Stream in Text-Deltas. Bewusst die CLI (nicht
das SDK), damit die Abo-Auth ueber CLAUDE_CODE_OAUTH_TOKEN garantiert greift.
`--bare` wird NICHT genutzt (bare erzwingt einen API-Key).
"""
import asyncio
import json
from typing import AsyncIterator

from . import config


def _base_cmd(prompt: str, session_id: str | None) -> list[str]:
    cfg = config.load()["claude"]
    inner = [
        "claude", "-p", prompt,
        "--output-format", "stream-json",
        "--verbose", "--include-partial-messages",
    ]
    if session_id:
        inner += ["--resume", session_id]
    if cfg.get("mode") == "docker":
        return [
            "docker", "exec", "-i",
            "-w", cfg.get("workdir", "/workspace"),
            cfg["container"], *inner,
        ]
    return inner  # lokaler Modus (Claude Code direkt im Backend-Container)


async def run(prompt: str, session_id: str | None = None) -> AsyncIterator[dict]:
    """Yieldet Events: {"type": "text", "text": ...}, {"type": "session", "id": ...},
    {"type": "result", ...}. Text-Deltas kommen als stream_event/text_delta rein.
    """
    proc = await asyncio.create_subprocess_exec(
        *_base_cmd(prompt, session_id),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdout is not None
    async for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = evt.get("type")
        # Session-ID aus dem init-Event festhalten (fuer --resume beim naechsten Turn)
        if etype == "system" and evt.get("subtype") == "init":
            sid = evt.get("session_id")
            if sid:
                yield {"type": "session", "id": sid}
        # Streaming-Text-Deltas
        elif etype == "stream_event":
            delta = evt.get("event", {}).get("delta", {})
            if delta.get("type") == "text_delta" and delta.get("text"):
                yield {"type": "text", "text": delta["text"]}
        # Abschluss
        elif etype == "result":
            yield {"type": "result", "result": evt.get("result"),
                   "session_id": evt.get("session_id")}

    err = await proc.stderr.read() if proc.stderr else b""
    rc = await proc.wait()
    if rc != 0:
        yield {"type": "error", "message": err.decode(errors="replace")[:2000]}
