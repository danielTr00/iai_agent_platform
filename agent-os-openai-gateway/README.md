# Agent OS — OpenAI-kompatibles Gateway für Claude Code

**Ziel:** Eine Claude-Code-Instanz (Docker auf VPS, Abo-Auth) hinter **OpenAI-API-konformen
Endpoints** (`/v1/chat/completions`, `/v1/models`, `/v1/audio/*`) verfügbar machen und das
bestehende **Agent-OS-Harness** (BE/FE, Voice) daran anbinden.

**Warum OpenAI-konform?** Das OpenAI-Chat-Completions-Protokoll ist der De-facto-Standard.
Sobald das Gateway es spricht, funktioniert *jeder* OpenAI-Client gegen unsere Claude-Code-
Instanz: unser eigenes Harness-FE, aber auch Standard-Tools (LibreChat, Open WebUI,
Continue.dev, beliebige SDKs via `base_url`-Override). Das Harness-BE wird dadurch schlanker:
statt eigenem `docker exec`-Protokoll spricht es einfach das OpenAI-SDK gegen das Gateway.

> ⚠️ **Single-User (ToS).** Das Gateway läuft auf dem persönlichen Abo-OAuth-Token
> (`CLAUDE_CODE_OAUTH_TOKEN`). Ausschließlich der Account-Inhaber darf zugreifen.
> Kein Team-Zugriff, kein Weiterverkauf, keine Weitergabe des Endpoints.
> Für Team-Szenarien: Anthropic API-Key + Team-Plan. Details: [`SECURITY.md`](./SECURITY.md).

## Dokumente

| Dokument | Inhalt |
|---|---|
| [`docs/IMPLEMENTATION_PLAN.md`](./docs/IMPLEMENTATION_PLAN.md) | Phasenplan, Architektur, Komponenten, Risiken |
| [`docs/API_SPEC.md`](./docs/API_SPEC.md) | Endpoint-Spezifikation + Mapping OpenAI ↔ Claude Code `stream-json` |
| [`docs/HARNESS_INTEGRATION.md`](./docs/HARNESS_INTEGRATION.md) | Anbindung BE/FE des Agent-OS-Harness (inkl. Voice) |
| [`SECURITY.md`](./SECURITY.md) | Vorgeplante Sicherheitsthemen (Auth, Netz, Secrets, ToS) |

## Architektur (Kurzfassung)

```
FE (PWA / beliebiger OpenAI-Client)
        │  OpenAI-Protokoll (HTTPS/SSE, Bearer)
        ▼
Harness-BE ──────────────┐  (nutzt OpenAI-SDK, base_url = Gateway)
        │                │
        ▼                ▼
┌───────────────────────────────┐
│ openai-gateway (FastAPI)      │   /v1/models · /v1/chat/completions
│  • Request-Übersetzer         │   /v1/audio/speech · /v1/audio/transcriptions
│  • SSE-Streamer               │
│  • Session-Mapper             │
└──────────────┬────────────────┘
               │ claude -p --output-format stream-json (docker exec)
               ▼
┌───────────────────────────────┐
│ claude-core (Docker)          │   CLAUDE_CODE_OAUTH_TOKEN (Max-Abo)
│  .claude/skills/ (Shared)     │   /workspace/output (Dokumente)
└───────────────────────────────┘
```

## Referenzprojekte (Stand der Technik, geprüft Juli 2026)

Das Muster „Claude Code CLI hinter OpenAI-API" ist erprobt — wir bauen es passend zu unserem
Harness nach, statt blind zu forken:

- [RichardAtCT/claude-code-openai-wrapper](https://github.com/RichardAtCT/claude-code-openai-wrapper) — `/v1/chat/completions` mit `session_id`-Fortführung, zusätzlich Anthropic-`/v1/messages`
- [wende/claude-max-api-proxy](https://github.com/wende/claude-max-api-proxy) — CLI als Subprozess, OpenAI-kompatibel, für Continue.dev u. a.
- [bethington/claude-code-api](https://github.com/bethington/claude-code-api) — Streaming + persistierte, wiederaufnehmbare Sessions
- [i-am-logger/claude-code-proxy](https://github.com/i-am-logger/claude-code-proxy) — `/v1/chat/completions` und `/v1/responses`

## Status

- [ ] Phase G0 — Spezifikation eingefroren (dieses Repo)
- [ ] Phase G1 — Gateway-Skeleton: `/v1/models`, `/v1/chat/completions` non-streaming
- [ ] Phase G2 — SSE-Streaming + Session-Mapping
- [ ] Phase G3 — Voice-Endpoints (`/v1/audio/*`) auf die pluggablen STT/TTS-Backends
- [ ] Phase G4 — Harness-BE auf OpenAI-SDK umstellen, FE unverändert
- [ ] Phase G5 — Härtung (SECURITY.md-Checkliste), Betrieb hinter Tailscale
