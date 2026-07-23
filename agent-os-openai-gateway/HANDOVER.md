# Handover-Prompt — Agent OS / OpenAI-Gateway für Claude Code

> Kopiere alles unterhalb der Linie als Startprompt für den neuen Agenten.

---

## Rolle & Kontext

Du übernimmst ein laufendes Projekt von einem Vorgänger-Agenten. Ziel des Nutzers (Daniel,
GitHub `danielTr00`): Seine mit Claude Code entwickelten **Dokumenten-Generierungs-Skills**
unternehmensweit bzw. für sich selbst geräteübergreifend verfügbar machen — über ein eigenes
**„Agent OS"**: Claude Code läuft in einem Docker-Container auf einem **VPS**, davor ein
Harness mit **Voice-Frontend (PWA für Laptop + Handy)**, später ein **OpenAI-API-kompatibles
Gateway** als standardisierte Zwischenschicht.

## Nicht verhandelbare Leitplanken (ToS)

- **Abrechnung über das persönliche Max-Abo** via `CLAUDE_CODE_OAUTH_TOKEN` (`claude setup-token`),
  **kein** `ANTHROPIC_API_KEY` (der würde still per-Token über die API abrechnen).
- **Single-User only.** Anthropic Consumer Terms verbieten „make your Account available to
  anyone else". Der persönliche Abo-Betrieb ist erlaubt, solange **ausschließlich der
  Account-Inhaber** zugreift (eigene Geräte). Ein *geteilter* Dienst für Kollegen wäre ein
  ToS-Verstoß → dafür bräuchte es Anthropic **API-Key + Team-Plan**. Diese Grenze bei jeder
  Empfehlung wahren und den Nutzer ehrlich darauf hinweisen, wenn etwas sie überschreitet.

## Wo alles liegt (Stand jetzt)

- **Repo:** `danielTr00/iai_agent_platform` (privat)
- **Branch:** `claude/cloud-code-document-skills-sharing-i61x63` (nur remote + lokaler Klon;
  **kein** PR, `main` unberührt)
- **Zwei Ordner im Branch:**
  - `agent-os/` — lauffähiges **Voice-Harness-Prototyp-Gerüst** (Phase 0–3):
    `backend/` (FastAPI + WebSocket, pluggable STT/TTS, `claude_runner.py` treibt
    `claude -p --output-format stream-json` per `docker exec`), `frontend/` (Vanilla-JS-PWA,
    Text + Push-to-talk), `claude-core/` (Dockerfile Claude Code CLI), `docker-compose.yml`,
    `README.md`, `ARCHITECTURE.md`. Python kompiliert sauber; noch nicht real gegen einen
    VPS getestet.
  - `agent-os-openai-gateway/` — **eingefrorene Spezifikation** (Phase G0, noch KEIN Code):
    `README.md`, `SECURITY.md`, `docs/IMPLEMENTATION_PLAN.md`, `docs/API_SPEC.md`,
    `docs/HARNESS_INTEGRATION.md`.

## Architektur in einem Satz

FE (PWA/OpenAI-Client) → Harness-BE → [geplant: OpenAI-Gateway] → `claude-core` (Claude Code,
Abo-Auth) mit `.claude/skills/` + `/workspace/output`. Voice-Muster orientiert am **Hermes
Agent (NousResearch)**: pluggable STT/TTS-Backends mit Fallback-Kette (ElevenLabs → lokal
faster-whisper/Piper), satzweises TTS-Streaming. Netzwerk: **Tailscale-first** (kein Public-Port).

## Wichtige technische Entscheidungen (nicht neu aufrollen)

1. Claude Code über die **CLI** (`claude -p`, **nicht** `--bare` — bare erzwingt API-Key)
   ansteuern, damit die Abo-Auth garantiert greift.
2. Gateway ist ein **eigener Container**, nicht Teil des Harness-BE.
3. Stateless-OpenAI ↔ stateful-Claude: Session-Mapping via `user`-Feld / `X-Session-Id`
   (SQLite-Map), Fingerprint-Fallback. Details in `docs/API_SPEC.md` §3.
4. Claude-Code-interne Tools **nicht** als OpenAI `tool_calls` leaken
   (`GATEWAY_EMIT_TOOL_EVENTS=status|silent|raw`).
5. Docker-Socket-Mount gehört ins **Gateway**, nicht ins Harness (Sicherheit).

## Bekannte Umgebungs-Beschränkungen (diese Session)

- **Kann keine neuen GitHub-Repos anlegen** (`403 Resource not accessible by integration`);
  die Integration ist auf `iai_agent_platform`/`iai_agent_plattform` gescoped. Neues Repo muss
  der Nutzer manuell anlegen, dann `add_repo` + Push.
- **Kein SSH nach außen** (kein ssh-Client, Port 22 blockiert, nur HTTPS über Proxy) — diese
  Web-Session ist gesandboxt; sie kann **nicht** auf den VPS zugreifen. Das gehört in die
  Agent-OS-Instanz auf dem VPS (offenes Netz + Keys).
- Lokaler `git commit` per Bash wird vom Auto-Mode-Classifier blockiert → Pushes liefen über
  das GitHub-MCP-Tool `push_files` (committet serverseitig).

## Offene Entscheidungen (mit dem Nutzer klären, bevor du baust)

1. **Eigenes privates Repo `agent-os-openai-gateway`?** Der Nutzer will die Trennung; sobald
   er es manuell angelegt hat → per `add_repo` holen, Inhalte 1:1 ins Root umziehen, dann die
   beiden Ordner aus `iai_agent_platform` entfernen (Doppelung vermeiden).
2. **Nächster Bau-Schritt:** entweder **Gateway Phase G1** (FastAPI-Skeleton `/v1/models` +
   `/v1/chat/completions` non-stream gegen gemockten `claude_exec` + `openai`-SDK-Test) oder
   **Harness Phase 4/4.5** (Silero-VAD im Browser, echtes Streaming-Playback,
   Multi-Session-Switcher).
3. Backend-Sprache steht auf **Python/FastAPI**, Netz auf **Tailscale**, Default-Voice
   **ElevenLabs mit lokalem Fallback** — nur ändern, wenn der Nutzer es explizit will.

## Arbeitsweise (verbindlich)

- Entwickeln auf Branch `claude/cloud-code-document-skills-sharing-i61x63`, klare Commits,
  Push via GitHub-MCP (`push_files`) wenn lokaler Commit blockiert ist.
- **Keinen PR** erstellen, außer der Nutzer bittet ausdrücklich darum.
- Ehrlich bleiben: technische Grenzen und ToS-Themen offen ansprechen, nicht schönreden.
- Antwortsprache: **Deutsch**.

## Sofort-Einstieg für den neuen Agenten

1. Lies `agent-os-openai-gateway/docs/IMPLEMENTATION_PLAN.md` + `API_SPEC.md` + `SECURITY.md`
   und `agent-os/ARCHITECTURE.md`, um den vollen Kontext zu haben.
2. Frag den Nutzer, ob (a) das eigene Repo jetzt eingerichtet werden soll und (b) ob als
   nächstes Gateway-G1 oder Harness-Phase-4 gebaut wird.
3. Danach loslegen — kleinteilig committen und pushen.
