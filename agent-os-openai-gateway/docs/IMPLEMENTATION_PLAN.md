# Umsetzungsplan — OpenAI-Gateway für Claude Code

## 1. Problemstellung

Das Agent-OS-Harness (Repo `agent-os`) spricht Claude Code heute direkt über ein
Eigenprotokoll (`claude_runner.py` → `docker exec` → `stream-json`-Parsing). Das koppelt
FE/BE hart an unsere Implementierung. Ziel: eine **standardisierte Zwischenschicht** im
OpenAI-Format, sodass:

1. das Harness-BE nur noch ein OpenAI-SDK braucht (weniger Eigencode, testbar gegen Mocks),
2. beliebige OpenAI-kompatible Clients/Tools die Claude-Code-Instanz nutzen können,
3. STT/TTS ebenfalls hinter Standard-Endpoints (`/v1/audio/*`) liegen und austauschbar bleiben
   (ElevenLabs ↔ lokale Modelle), ohne dass das FE etwas davon merkt.

## 2. Zielarchitektur

Drei Container (docker-compose), ein privates Netz (Tailscale):

| Container | Aufgabe | Tech |
|---|---|---|
| `claude-core` | Claude Code CLI + Skills + Workspace | Node 22 + `@anthropic-ai/claude-code` |
| `openai-gateway` | Protokoll-Übersetzung OpenAI ↔ Claude Code, SSE, Session-Mapping, Audio-Endpoints | Python 3.12 / FastAPI |
| `harness` (BE+FE) | Business-Logik, WebSocket zum FE, Voice-Pipeline-Orchestrierung, PWA-Auslieferung | FastAPI + Vanilla-PWA |

Entscheidung: Das Gateway ist ein **eigener Prozess/Container**, nicht Teil des Harness-BE.
Gründe: unabhängige Härtung/Ratelimits, eigenständig nutzbar (Continue.dev & Co.),
klare Zuständigkeit (Gateway = Protokoll, Harness = Produkt-UX).

## 3. Kernprobleme & Lösungen

### 3.1 Stateless OpenAI vs. stateful Claude Code

OpenAI-Clients senden bei jedem Request die **komplette `messages`-Historie**; Claude Code
führt dagegen **Sessions** (`--resume <session_id>`). Drei Mapping-Strategien, wir
implementieren A als Default mit B als Fallback:

- **A — expliziter Session-Schlüssel:** Client übergibt `user`-Feld oder Header
  `X-Session-Id`. Gateway hält Map `session_key → claude_session_id` (SQLite).
  Bei Request: nur die **letzte** User-Message an `claude -p --resume` geben.
  (Muster aus RichardAtCT-Wrapper, bewährt.)
- **B — Historien-Fingerprint:** Kein Session-Schlüssel? Dann Hash über alle Messages außer
  der letzten → Lookup. Treffer ⇒ resume; kein Treffer ⇒ neue Session, komplette Historie
  als Kontext-Präambel in den ersten Prompt.
- **C — stateless-pur (nicht gewählt):** jede Anfrage neue Session mit voller Historie.
  Einfach, aber verliert Claude-Code-interne Zustände (Tool-Ergebnisse, Arbeitsverzeichnis).

### 3.2 Streaming-Übersetzung

`stream-json`-Events → OpenAI-SSE-Chunks (`chat.completion.chunk`), Ende mit `data: [DONE]`.
Mapping-Tabelle in `docs/API_SPEC.md` §4. Wichtig: `usage` aus dem `result`-Event in den
letzten Chunk übernehmen (`stream_options.include_usage`-Kompatibilität).

### 3.3 Tool-Nutzung von Claude Code

Claude Code führt Tools (Bash, Edit, Skills …) **intern** aus — der OpenAI-Client soll sie
nicht ausführen. Default: `tool_use`/`tool_result`-Events **nicht** als OpenAI-`tool_calls`
emittieren, sondern optional als Status-Text (`[tool] Bash: …`) bzw. als eigene
SSE-Kommentar-Events. Konfigurierbar (`GATEWAY_EMIT_TOOL_EVENTS=status|silent|raw`).

### 3.4 Modelle & Effort

`/v1/models` liefert die konfigurierte Modellliste (z. B. `claude-opus-4-8`,
`claude-sonnet-5`). Das `model`-Feld des Requests wird via `/model <x>`-Prompt-Präfix bzw.
CLI-Flag an Claude Code durchgereicht; unbekannte Modelle → 404 `model_not_found`
(OpenAI-konformes Fehlerobjekt).

### 3.5 Dokument-Artefakte

Doku-Skills schreiben nach `/workspace/output`. Gateway bietet dafür einen (Nicht-OpenAI-)
Zusatzendpoint `GET /artifacts/{path}` (Bearer-geschützt, read-only auf `output/`).
Das FE verlinkt generierte Dokumente direkt.

## 4. Phasenplan

| Phase | Inhalt | Definition of Done |
|---|---|---|
| **G0** | Diese Spezifikation reviewen/einfrieren | Docs gemergt |
| **G1** | FastAPI-Skeleton: `/v1/models`, `/v1/chat/completions` (non-stream), Bearer-Auth, OpenAI-Fehlerformat | `curl` + `openai`-SDK (Python) liefern korrekte Antworten |
| **G2** | SSE-Streaming, Session-Mapping A+B (SQLite), Usage-Reporting | Streaming-Client zeigt Live-Deltas; Folgerequest resumed nachweislich |
| **G3** | `/v1/audio/transcriptions` (STT) + `/v1/audio/speech` (TTS) auf die bestehenden pluggablen Backends (ElevenLabs/faster-whisper/Piper) | Roundtrip Audio→Text→Claude→Text→Audio rein über OpenAI-Endpoints |
| **G4** | Harness-BE: `claude_runner.py` ersetzen durch OpenAI-SDK (`base_url` = Gateway); FE unverändert | Bestehende PWA funktioniert identisch über das Gateway |
| **G5** | Härtung nach `SECURITY.md`-Checkliste; Compose-Profile `tailscale` / `public+caddy`; Lasttest Single-User | Checkliste vollständig abgehakt |

## 5. Repo-Struktur (Zielbild Implementierung)

```
agent-os-openai-gateway/
  docker-compose.yml            # gateway + claude-core (harness bleibt im agent-os-Repo)
  .env.example
  gateway/
    Dockerfile
    app/
      main.py                   # FastAPI, Router-Mounting, Auth-Middleware
      routes/
        models.py               # GET /v1/models
        chat.py                 # POST /v1/chat/completions (stream + non-stream)
        audio.py                # POST /v1/audio/speech, /v1/audio/transcriptions
        artifacts.py            # GET /artifacts/{path}
      translate/
        openai_to_claude.py     # Request-Normalisierung, Prompt-Bau
        claude_to_openai.py     # stream-json → chunk/completion-Objekte
      sessions.py               # SQLite-Map, Fingerprint-Fallback, TTL/GC
      claude_exec.py            # docker exec Subprozess-Treiber (aus agent-os übernommen)
      errors.py                 # OpenAI-konforme Fehlerobjekte
      ratelimit.py              # Token-Bucket pro API-Key
    tests/
      test_chat_nonstream.py    # gegen gemockten claude_exec
      test_chat_stream.py
      test_sessions.py
      test_openai_sdk_compat.py # echter openai-Python-Client gegen TestServer
  docs/ …  SECURITY.md  README.md
```

## 6. Risiken

| Risiko | Gegenmaßnahme |
|---|---|
| CLI-Startlatenz pro Request (~2–4 s) | Sessions resumen (kein Cold-Context); später: warmgehaltener Prozess-Pool evaluieren |
| `stream-json`-Format ändert sich mit CLI-Versionen | CLI-Version im Image pinnen; Contract-Test `test_stream_format.py` bei Upgrade |
| OAuth-Token läuft ab | Health-Endpoint prüft Auth-Status; dokumentierte Re-Mint-Prozedur (`claude setup-token`) |
| Client sendet Riesen-Historie (Fallback B) | Request-Size-Limit + Historie-Trunkierung mit Warnung im Response-Header |
| Versehentlicher API-Key im Env | Startup-Guard: Prozess verweigert Start, wenn `ANTHROPIC_API_KEY` gesetzt ist (Abo-Schutz) |
| Mehrbenutzer-Missbrauch des Endpoints | Genau **ein** API-Key, Betrieb nur im Tailnet, siehe SECURITY.md §1 |
