# Harness-Anbindung (BE + FE)

Bezieht sich auf das bestehende `agent-os`-Repo (FastAPI-Harness + PWA).

## 1. Was sich im Backend ändert

**Vorher:** `main.py` → `claude_runner.py` → `docker exec claude -p --output-format stream-json`
(Eigenprotokoll, eigenes Parsing).

**Nachher:** `main.py` → **OpenAI-SDK** gegen das Gateway:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://openai-gateway:8080/v1",   # compose-internes Netz
    api_key=os.environ["GATEWAY_API_KEY"],
)

async def claude_stream(session_key: str, text: str):
    stream = await client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": text}],
        user=session_key,          # Session-Mapping Strategie A
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
```

- `claude_runner.py` entfällt komplett (≈ 80 Zeilen Eigenprotokoll weg).
- Der Docker-Socket-Mount **wandert vom Harness ins Gateway** — das Harness-BE braucht ihn
  nicht mehr (Sicherheitsgewinn, siehe SECURITY.md §4).
- Session-Verwaltung: das Harness generiert pro FE-Client einen stabilen `session_key`
  (UUID im LocalStorage der PWA) und gibt ihn als `user` mit — Resume macht das Gateway.

## 2. Voice über Standard-Endpoints

Die Voice-Pipeline (Satz-Chunker) bleibt im Harness, aber STT/TTS-Aufrufe gehen ebenfalls
über das Gateway:

```python
# STT
tr = await client.audio.transcriptions.create(model="whisper-1", file=audio_file)
# TTS (pro Satz-Chunk aus voice_pipeline.sentence_chunks)
speech = await client.audio.speech.create(model="tts-1", voice="default", input=sentence)
```

Effekt: Das Harness kennt **keine Provider-Details mehr** (ElevenLabs-Keys etc. leben nur
noch im Gateway). Provider-Wechsel = Gateway-Config, kein Harness-Deploy.

## 3. Was sich im Frontend ändert

**Nichts Notwendiges.** Die PWA spricht weiter WebSocket mit dem Harness-BE.

**Optional (späterer Ausbau):** Das FE kann Streaming direkt vom Gateway holen
(SSE via `fetch`), das Harness-BE bleibt für Auth/Sessions/Doc-UX zuständig. Erst relevant,
wenn WebSocket-Hop messbar Latenz kostet — vorher nicht optimieren.

## 4. Compose-Verdrahtung

```yaml
services:
  claude-core:     # unverändert (Abo-Token, Skills-Volume)
  openai-gateway:
    build: ../agent-os-openai-gateway/gateway
    environment:
      - GATEWAY_API_KEY=${GATEWAY_API_KEY}
      - ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY:-}
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock   # nur hier, nicht im Harness
      - ./claude-core/workspace:/workspace:ro
    expose: ["8080"]          # NICHT publishen — nur compose-intern + Tailnet
  harness:
    environment:
      - OPENAI_BASE_URL=http://openai-gateway:8080/v1
      - GATEWAY_API_KEY=${GATEWAY_API_KEY}
    ports: ["8000:8000"]
```

## 5. Migrationsreihenfolge (kein Big-Bang)

1. Gateway parallel zum bestehenden `claude_runner`-Pfad deployen (G1–G2).
2. Harness-BE: Feature-Flag `USE_GATEWAY=1` — beide Pfade lauffähig.
3. Voice auf `/v1/audio/*` umstellen (G3).
4. Alt-Pfad + Docker-Socket aus dem Harness entfernen (G4).
5. Härtungs-Checkliste abarbeiten (G5).

## 6. Bonus: Fremd-Clients

Weil das Gateway Standard-OpenAI spricht, funktionieren im Tailnet sofort auch:
- **Continue.dev / IDE-Plugins** (base_url aufs Gateway),
- **Open WebUI / LibreChat** als alternative Chat-Oberfläche,
- jedes Skript mit `openai`-SDK (`OPENAI_BASE_URL` + Gateway-Key).

Alle unter derselben Single-User-Prämisse (siehe SECURITY.md §1).
