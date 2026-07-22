# Agent OS — Voice Harness für Claude Code

**Ziel:** Claude Code als Agent-Kern in einem Docker-Container auf einem VPS, gebunden an ein
eigenes Harness mit Voice-Frontend (Laptop + Handy als PWA). Modellabrechnung über das eigene
Max-Abo (kein API-Key). Vorbild-Architektur: **Hermes Agent (NousResearch)** — pluggable
STT/TTS-Backends, Provider-Auswahl per Config, Streaming-TTS. Wir ersetzen nur die LLM-Loop
durch Claude Code.

> **ToS-Rahmen:** Ausschließlich Single-User (nur du). Dein OAuth-Token, deine Geräte. Das ist
> persönliche Nutzung und erlaubt. Sobald ein zweiter Mensch zugreift, kippt es in den
> verbotenen Bereich (Account-Sharing) — dann bräuchte es Team-Plan + API.

---

## 1. Was wir vom Hermes-Vorbild übernehmen

| Hermes-Konzept | Übernahme in unser Harness |
|---|---|
| `STTBackend` / `TTSBackend` als Protocol-Interfaces | Gleiches Muster: austauschbare Backends |
| Provider-Kette lokal → Cloud (config.yaml) | `local (Whisper/Piper)` → `ElevenLabs` Fallback |
| Streaming-TTS satzweise (min ~20 Zeichen, Markdown/`<think>` strippen) | 1:1 übernehmen — senkt gefühlte Latenz |
| VAD / Silence-Detection für Turn-Taking | Ins Browser-Frontend verlagert (Silero-VAD) |
| Config: `config.yaml` + `.env` für Keys | Gleiches Layout |

**Kernunterschied zu Hermes:** Hermes nimmt Audio lokal per PortAudio auf (Server-Mikro). Wir
brauchen Laptop **und Handy** → Audio wird **im Browser** aufgenommen und über WebSocket zum
VPS gestreamt. STT/TTS laufen im Backend, Claude Code im Container.

---

## 2. Systemarchitektur

```
┌─────────────────────────────┐         WSS / HTTPS          ┌──────────────────────────────────┐
│  Frontend (PWA)             │  ◄────────────────────► │  VPS                              │
│  Laptop-Browser + Handy     │   (über Tailscale-Netz)      │                                  │
│                             │                              │  ┌──────────────────────────────┐  │
│  • Mic-Capture (WebAudio)   │ ── Audio-Chunks ─────────► │  │ harness-api (FastAPI, WS)  │  │
│  • Silero-VAD (Utterance-   │                              │  │                            │  │
│    Ende erkennen)           │ ◄── TTS-Audio-Frames ─────── │  │  STT-Backend (pluggable)   │  │
│  • Chat-UI + Transkript     │                              │  │  TTS-Backend (pluggable)   │  │
│  • Doc-Preview/Download     │ ◄── Text-Deltas + Events ─── │  │  voice_pipeline            │  │
│  • Push-to-talk / VAD-Modus │                              │  │  claude_runner ───────────┐│  │
└─────────────────────────────┘                              │  └──────────────────────────────┼┘  │
                                                             │                              │   │
                                                             │  ┌──────────────────────────▼─┐ │
                                                             │  │ claude-core (Docker)         │ │
                                                             │  │  Claude Code CLI             │ │
                                                             │  │  Auth: CLAUDE_CODE_OAUTH_    │ │
                                                             │  │        TOKEN (Max-Abo)       │ │
                                                             │  │  /workspace (Volume)         │ │
                                                             │  │   └ .claude/skills/  (Doku)  │ │
                                                             │  │   └ output/  (generierte Docs)│ │
                                                             │  └─────────────────────────────┘ │
                                                             └────────────────────────────────┘
```

### Komponenten

**A) `claude-core` (Container mit Claude Code)**
- Base: Ubuntu + Node + Claude Code CLI.
- Auth: `CLAUDE_CODE_OAUTH_TOKEN` (per `claude setup-token` einmal am Laptop erzeugt) als Docker-Secret.
  **Kein** `ANTHROPIC_API_KEY` im Container (sonst greift API-Abrechnung).
- **Nicht** `--bare` verwenden (bare erzwingt API-Key) → normaler Modus, damit das Abo greift.
- Volume `/workspace`: enthält `.claude/skills/` (deine Doku-Skills) und `output/`.
- Ansteuerung durch das Harness via CLI:
  `claude -p --output-format stream-json --resume <session-id> "<prompt>"`
  → liefert newline-delimited JSON-Events (Text-Deltas, Tool-Use, Result).
- Skills funktionieren in `-p`-Mode (auto-getriggert + `/skill-name` im Prompt).

**B) `harness-api` (das eigentliche Harness, FastAPI)**
- **WebSocket-Endpoint** für das Frontend (bidirektional: Audio rein, Text+Audio raus).
- `claude_runner.py`: spawnt `claude -p …`, parst den `stream-json`-Stream, mappt User-Session ↔ Claude-Session-ID (`--resume` für Kontext über Turns).
- `stt/` — Protocol `STTBackend` mit Implementierungen:
  - `LocalWhisper` (faster-whisper, CPU/GPU, kein Key)
  - `ElevenLabsSTT` (Scribe) / `GroqWhisper` (Cloud, schnell)
- `tts/` — Protocol `TTSBackend` mit Implementierungen:
  - `ElevenLabsTTS` (Premium, Key)
  - `PiperTTS` (lokal, schnell, CPU) / `KokoroTTS` (lokal, höhere Qualität)
- `voice_pipeline.py`: Satz-Chunker (min ~20 Zeichen, Markdown/`<think>` strippen) → TTS pro Satz → Audio-Frames streamen. Direkt aus Hermes übernommen.
- `config.yaml` + `.env`: Provider-Auswahl + Fallback-Kette + Keys.
- Auth (Single-User): ein starkes Bearer-Token / Passkey. Hinter Tailscale minimal-invasiv.

**C) `harness-fe` (die „Agent OS"-Oberfläche)**
- **PWA** (React + Vite + Tailwind) → installierbar als App-Icon auf Handy & Laptop.
- Voice: Mic über Web Audio API, **Silero-VAD im Browser** (`@ricky0123/vad-web`) erkennt Sprech-Ende, streamt Audio über WS. Empfangenes TTS-Audio wird direkt abgespielt.
- Modi: Push-to-talk **und** Always-on-VAD (umschaltbar).
- Text-Chat-Modus parallel. Transkript-Ansicht. Preview/Download generierter Dokumente.

**D) Netzwerk (Empfehlung: Tailscale)**
- VPS, Laptop, Handy in einem privaten Mesh-VPN → keine öffentliche Angriffsfläche, kein offener Port.
- Alternative bei Bedarf echter public URL: **Caddy** Reverse-Proxy (Auto-HTTPS) + harte Auth.

---

## 3. Datenfluss (ein Voice-Turn)

1. FE: Mic → VAD erkennt Utterance-Ende → Audio-Blob über WebSocket an Backend.
2. Backend STT (`LocalWhisper` oder `ElevenLabsSTT`) → Text.
3. Backend: `claude -p --resume <sid> --output-format stream-json "<text>"` (Skills verfügbar).
4. Claude-Text-Deltas → Satz-Chunker → TTS pro Satz → Audio-Frames an FE streamen.
5. FE: spielt Audio sofort ab, zeigt Transkript; generierte Docs erscheinen als Link/Preview.
6. Docs landen in `/workspace/output` → FE bekommt Download/Preview.

---

## 4. Repo-Struktur (Prototyp)

```
agent-os/
  docker-compose.yml
  .env.example                 # CLAUDE_CODE_OAUTH_TOKEN, ELEVENLABS_API_KEY, HARNESS_TOKEN …
  claude-core/
    Dockerfile                 # node + claude code
    workspace/
      .claude/skills/          # deine Doku-Skills (gemountet)
  backend/
    Dockerfile
    app/
      main.py                  # FastAPI + WebSocket
      claude_runner.py         # spawnt & streamt `claude -p`
      voice_pipeline.py        # Satz-Chunker + TTS-Streaming
      stt/  __init__.py  base.py  local_whisper.py  elevenlabs.py
      tts/  __init__.py  base.py  elevenlabs.py  piper.py
      config.py
    config.yaml
  frontend/
    (React PWA: Vite, Tailwind, vad-web, WS-Client)
  caddy/ | tailscale/          # je nach Netzwahl
  README.md  ARCHITECTURE.md
```

---

## 5. Phasenplan

| Phase | Inhalt | Ergebnis |
|---|---|---|
| **0 — Fundament** | VPS + Docker + Tailscale; `claude-core` bauen; `setup-token` rein; `/status` prüft Abo-Abrechnung | Claude Code läuft im Container auf deinem Max-Abo |
| **1 — Harness (Text)** | WebSocket ↔ `claude -p` streaming; Session-Mapping; Skills gemountet | Vom Handy tippen → Claude Code generiert Doc |
| **2 — TTS raus** | Satz-Chunker + ElevenLabs/Piper; Audio-Streaming ins FE | Claude „spricht" |
| **3 — STT rein** | Browser-Mic → Whisper → Claude | Voller Voice-Loop |
| **4 — Voice-UX** | Silero-VAD, Barge-in/Interrupt, PWA-Polish, lokale Fallbacks | Flüssiger Voicechat, installierbare App |
| **5 — Doc-UX** | Preview/Download generierter Dokumente, Skill-Trigger-Buttons | Fertiger Doku-Workflow per Stimme |

---

## 6. Risiken & offene Validierungen

- **Abo-Auth im Container:** OAuth-Token ist langlebig, aber nicht ewig → Re-Mint-Prozedur dokumentieren. In Phase 0 verifizieren, dass `/status` das Abo (nicht API) zeigt.
- **`-p` Startup-Latenz pro Turn:** mit `--resume` Session halten; Streaming senkt gefühlte Latenz. Falls zu träge: persistenten Claude-Prozess statt per-Turn-Spawn evaluieren.
- **SDK vs. CLI:** Für garantierte Abo-Auth treiben wir Claude Code über die **CLI** (`claude -p`, nicht bare). Ob die Python/TS-SDK-Pakete den OAuth-Token ebenso honorieren → in Phase 1 testen, sonst bei CLI bleiben.
- **Single-User bleibt Pflicht** (ToS).

---

## 7. Empfohlener Default-Stack (anpassbar)

- Backend: **Python / FastAPI** (Voice-Ökosystem ist Python-nativ: faster-whisper, piper, sounddevice).
- STT lokal: **faster-whisper** (base/small) · Cloud-Premium: **ElevenLabs Scribe** oder **Groq Whisper**.
- TTS lokal: **Piper** (schnell, CPU) · Premium: **ElevenLabs**.
- Frontend: **React PWA** (Vite + Tailwind + vad-web).
- Netz: **Tailscale**.
- VPS-OS: **Ubuntu 22.04/24.04**.
