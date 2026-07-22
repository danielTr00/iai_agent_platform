# API-Spezifikation — OpenAI-konforme Endpoints

Alle Endpoints verlangen `Authorization: Bearer <GATEWAY_API_KEY>`.
Fehler folgen dem OpenAI-Fehlerformat: `{"error": {"message", "type", "param", "code"}}`.

## 1. `GET /v1/models`

```json
{ "object": "list", "data": [
    { "id": "claude-opus-4-8",  "object": "model", "owned_by": "agent-os-gateway" },
    { "id": "claude-sonnet-5",  "object": "model", "owned_by": "agent-os-gateway" },
    { "id": "claude-haiku-4-5", "object": "model", "owned_by": "agent-os-gateway" }
] }
```
Liste kommt aus `config.yaml`; sie spiegelt, was das Abo/CLI anbietet.

## 2. `POST /v1/chat/completions`

### Request (unterstützte Felder)

| Feld | Behandlung |
|---|---|
| `model` | → Modellwahl in Claude Code (`/model`-Mechanik). Unbekannt ⇒ 404 |
| `messages` | System-Messages → `--append-system-prompt`; Historie → Session-Mapping (s. §3); letzte User-Message → Prompt |
| `stream` | `true` ⇒ SSE (§4), sonst Vollantwort |
| `stream_options.include_usage` | Usage im letzten Chunk |
| `user` | Session-Schlüssel (Strategie A) |
| `temperature`, `top_p`, `max_tokens` | **Ignoriert** (CLI steuert das nicht) — dokumentiert, kein Fehler; `max_tokens` wird best-effort als Längenhinweis in den Prompt übersetzt |
| `tools`, `tool_choice` | 400 `unsupported` — Claude Code führt eigene Tools intern aus (siehe Plan §3.3) |

Zusätzlich (Nicht-Standard, optional): Header `X-Session-Id` (überschreibt `user`),
Header `X-Effort: low|medium|high`.

### Response (non-streaming)

OpenAI-konformes `chat.completion`-Objekt; `usage` aus dem `result`-Event der CLI;
`finish_reason: "stop"`; `id` = `chatcmpl-<uuid>`; zusätzlich Nicht-Standard-Feld
`x_claude_session_id` für Clients, die explizit resumen wollen.

## 3. Session-Mapping

```
key = header X-Session-Id  ||  body.user  ||  fingerprint(messages[:-1])
map: key → { claude_session_id, last_seen }        (SQLite, TTL z. B. 7 Tage, GC-Job)
```
- Treffer ⇒ `claude -p --resume <sid> "<letzte User-Message>"`
- Kein Treffer ⇒ neue Session; wenn Historie mitkam: als nummerierte Kontext-Präambel
  („Bisheriger Verlauf: …") vor die aktuelle Message setzen.
- Fingerprint = SHA-256 über `role+content` aller Messages außer der letzten.

## 4. Streaming: `stream-json` → SSE

Jede SSE-Zeile: `data: {chat.completion.chunk}` · Abschluss: `data: [DONE]`.

| Claude-Code-Event | OpenAI-Chunk |
|---|---|
| `system/init` (`session_id`) | erster Chunk: `delta: {"role": "assistant"}` (+ `x_claude_session_id`) |
| `stream_event` → `text_delta` | `delta: {"content": "<text>"}` |
| `tool_use` / `tool_result` | je nach `GATEWAY_EMIT_TOOL_EVENTS`: `delta.content = "\n[tool] …"` oder unterdrückt |
| `system/api_retry` | SSE-Kommentar `: retry attempt n` (kein Chunk — Clients ignorieren Kommentare) |
| `result` | letzter Chunk: `finish_reason: "stop"` + `usage` (falls angefordert) → dann `[DONE]` |
| `error` / Prozess-RC ≠ 0 | Fehler-Chunk mit `finish_reason: "stop"` + SSE-Ende; non-stream: 502 mit OpenAI-Fehlerobjekt |

Timeout-Regeln: kein Delta für > `GATEWAY_STREAM_IDLE_TIMEOUT` (Default 300 s, Claude-Code-
Tool-Läufe können lange sein!) ⇒ Abbruch mit Fehler-Chunk; Client-Disconnect ⇒ SIGTERM an
CLI-Subprozess (sauberer Turn-Abbruch, Session bleibt resumebar).

## 5. `POST /v1/audio/transcriptions` (STT)

OpenAI-Whisper-konform: `multipart/form-data` mit `file`, `model` (`whisper-1` ⇒ Kette aus
config: ElevenLabs Scribe → faster-whisper), optional `language`.
Antwort: `{"text": "..."}`.

## 6. `POST /v1/audio/speech` (TTS)

OpenAI-konform: `{"model": "tts-1", "input": "<text>", "voice": "<name>"}` →
Audio-Bytes (`audio/mpeg`). `voice` mappt auf die konfigurierte Backend-Stimme
(ElevenLabs-Voice-ID bzw. Piper-Modell). Streaming via Chunked-Response.

## 7. Zusatz (Nicht-OpenAI): `GET /artifacts/{path}`

Read-only-Zugriff auf `/workspace/output` (generierte Dokumente). Path-Traversal-sicher
(Normalisierung + Prefix-Check), Content-Disposition für Downloads.

## 8. Kompatibilitätstests (Pflicht in CI)

1. `openai`-Python-SDK: `client.chat.completions.create(...)` non-stream + stream.
2. `curl`-SSE-Roh-Test: Chunk-Framing, `[DONE]`, Kommentar-Zeilen.
3. LibreChat/Open WebUI Smoke-Test (manuell, dokumentiert).
4. Session-Resume-Nachweis: zwei Requests mit gleichem `user` ⇒ zweiter kennt Kontext.
