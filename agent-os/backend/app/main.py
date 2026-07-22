"""Agent OS Harness — FastAPI + WebSocket.

Bindet Frontend (PWA) an Claude Code + Voice-Pipeline.

Protokoll (WS, JSON-Frames vom Client):
  {"type": "auth",  "token": "<HARNESS_TOKEN>"}
  {"type": "text",  "text": "..."}                 -> Textturn
  {"type": "audio", "data": "<base64>", "mime": "audio/webm"}  -> Voiceturn
  {"type": "voice_on" | "voice_off"}               -> TTS-Ausgabe an/aus

Server -> Client:
  {"type": "transcript", "text": ...}   (STT-Ergebnis des Users)
  {"type": "text", "text": ...}         (Claude-Text-Delta)
  {"type": "audio", "data": "<base64>"} (TTS-Frame)
  {"type": "done"} | {"type": "error", "message": ...}
"""
import asyncio
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import claude_runner, config, stt, tts, voice_pipeline

app = FastAPI(title="Agent OS Harness")

HARNESS_TOKEN = config.env("HARNESS_TOKEN")


class Session:
    """Ein Client = eine Claude-Session (Kontext ueber Turns via --resume)."""
    def __init__(self) -> None:
        self.claude_session_id: str | None = None
        self.voice_out: bool = True


async def _claude_text_stream(session: Session, prompt: str, ws: WebSocket):
    """Streamt Claude-Text an den Client und liefert Deltas fuer TTS zurueck."""
    queue: asyncio.Queue = asyncio.Queue()

    async def produce():
        async for evt in claude_runner.run(prompt, session.claude_session_id):
            if evt["type"] == "session":
                session.claude_session_id = evt["id"]
            elif evt["type"] == "text":
                await ws.send_json({"type": "text", "text": evt["text"]})
                await queue.put(evt["text"])
            elif evt["type"] == "result" and evt.get("session_id"):
                session.claude_session_id = evt["session_id"]
            elif evt["type"] == "error":
                await ws.send_json({"type": "error", "message": evt["message"]})
        await queue.put(None)  # Sentinel

    async def deltas():
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    return produce, deltas


async def handle_text_turn(session: Session, prompt: str, ws: WebSocket):
    produce, deltas = await _claude_text_stream(session, prompt, ws)

    if session.voice_out:
        tts_backend = await tts.pick()
        prod_task = asyncio.create_task(produce())
        async for audio in voice_pipeline.stream_tts(deltas(), tts_backend):
            b64 = base64.b64encode(audio).decode("ascii")
            await ws.send_json({"type": "audio", "data": b64})
        await prod_task
    else:
        # Nur Text: Deltas verwerfen (schon live gesendet)
        prod_task = asyncio.create_task(produce())
        async for _ in deltas():
            pass
        await prod_task

    await ws.send_json({"type": "done"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = Session()
    authed = False
    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")

            if mtype == "auth":
                authed = msg.get("token") == HARNESS_TOKEN
                await ws.send_json({"type": "auth", "ok": authed})
                if not authed:
                    await ws.close(code=4401)
                    return
                continue

            if not authed:
                await ws.close(code=4401)
                return

            if mtype == "voice_on":
                session.voice_out = True
            elif mtype == "voice_off":
                session.voice_out = False
            elif mtype == "text":
                await handle_text_turn(session, msg["text"], ws)
            elif mtype == "audio":
                audio = base64.b64decode(msg["data"])
                stt_backend = await stt.pick()
                text = await stt_backend.transcribe(
                    audio, msg.get("mime", "audio/webm"))
                await ws.send_json({"type": "transcript", "text": text})
                if text.strip():
                    await handle_text_turn(session, text, ws)
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        try:
            await ws.send_json({"type": "error", "message": str(e)[:2000]})
        except Exception:
            pass


@app.get("/health")
async def health():
    return {"ok": True}


# Frontend (PWA) ausliefern — Verzeichnis wird per Volume nach /app/frontend gemountet
app.mount("/", StaticFiles(directory="/app/frontend", html=True),
          name="frontend")
