# Agent OS — Voice Harness für Claude Code

Ein eigenes Voice-Frontend + Harness um **Claude Code** — Backbone-Muster orientiert am
[Hermes Agent](https://hermes-agent.nousresearch.com) (pluggable STT/TTS, Fallback-Ketten,
satzweises TTS-Streaming). Claude Code läuft im Docker-Container auf dem VPS, abgerechnet über
dein **Max-Abo** (OAuth-Token, kein API-Key). Zugriff von Laptop + Handy über eine PWA.

> ⚠️ **Nur Single-User.** Dein Abo-Token, deine Geräte = erlaubte persönliche Nutzung.
> Sobald ein zweiter Mensch zugreift, verletzt das Anthropics Consumer Terms (Account-Sharing).
> Für Teams: Team-Plan + API.

Vollständige Architektur: siehe [`ARCHITECTURE.md`](./ARCHITECTURE.md).

## Komponenten

- **`claude-core/`** — Container mit Claude Code CLI. Auth via `CLAUDE_CODE_OAUTH_TOKEN`.
- **`backend/`** — FastAPI-Harness: WebSocket ↔ `claude -p` (stream-json), pluggable STT/TTS.
- **`frontend/`** — PWA (Vanilla JS): Text- + Voice-Chat, installierbar auf Handy & Laptop.

## Setup

### 1. Abo-Token erzeugen (einmal, am Laptop mit Browser)

```bash
claude setup-token          # gibt CLAUDE_CODE_OAUTH_TOKEN aus
```

### 2. Env anlegen

```bash
cp .env.example .env
# .env füllen:
#   CLAUDE_CODE_OAUTH_TOKEN=...   (aus Schritt 1)
#   HARNESS_TOKEN=...             (python -c "import secrets; print(secrets.token_urlsafe(32))")
#   ELEVENLABS_API_KEY=...        (optional, sonst lokales Piper/Whisper)
```

### 3. Skills einlegen

Deine Doku-Skills nach `claude-core/workspace/.claude/skills/<name>/SKILL.md`.
Claude Code lädt sie automatisch (auch im `-p`-Modus).

### 4. Starten

```bash
docker compose up -d --build
docker exec agent-os-claude-core claude /status   # muss das Abo zeigen, NICHT API
```

### 5. Zugriff

- **Empfohlen — Tailscale:** VPS ins Tailnet (`tailscale up`), dann von Handy/Laptop
  `http://<vps-tailscale-name>:8000` öffnen → „Zum Home-Bildschirm hinzufügen" (PWA).
- **Alternative — Public HTTPS:** `caddy/Caddyfile` aktivieren (Domain eintragen),
  im `docker-compose.yml` den `caddy`-Service einkommentieren.

## Status / Roadmap (siehe ARCHITECTURE.md §5)

- [x] Phase 0/1-Gerüst: Container, Harness, WebSocket, Text-Turn, Session-Resume
- [x] Phase 2: TTS raus (ElevenLabs + Piper, satzweises Streaming)
- [x] Phase 3: STT rein (ElevenLabs Scribe + faster-whisper), Push-to-talk
- [ ] Phase 4: Silero-VAD im Browser, Barge-in/Interrupt, echtes Streaming-Playback
- [ ] Phase 5: Doc-Preview/Download-UX, Skill-Trigger-Buttons

## Sicherheits-Hinweise

- `.env` niemals committen (siehe `.gitignore`).
- Der Docker-Socket ist im Backend gemountet (für `docker exec`) — Backend nicht öffentlich
  ohne Auth/Tailscale exponieren.
