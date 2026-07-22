# Sicherheitskonzept (vorgeplant, Pflicht vor Go-Live = Phase G5)

Konsolidiert die Sicherheitsthemen aus den bestehenden Skills/Planungen (agent-os
ARCHITECTURE.md) und ergänzt Gateway-spezifisches.

## 1. Lizenz-/ToS-Leitplanken (nicht verhandelbar)

- **Single-User:** Der Gateway-Endpoint wird ausschließlich vom Account-Inhaber genutzt.
  Anthropic Consumer Terms verbieten „make your Account available to anyone else".
  Automatisierter Zugriff ist nur via API-Key vorgesehen — der persönliche Abo-Betrieb
  hier ist die Eigen-Nutzung über eigene Geräte, nicht mehr.
- **Kein Weiterreichen:** Gateway-Key nie teilen, Endpoint nie öffentlich verlinken.
- **Team-Ausbau später:** Umstieg auf `ANTHROPIC_API_KEY` (Console) + eigene Mandanten-Auth.
  Architektur ist darauf vorbereitet (ein Env-Switch im Gateway).

## 2. Authentifizierung & Zugriff

- **Ein** Gateway-API-Key (`GATEWAY_API_KEY`), ≥ 32 Byte Zufall, Rotation dokumentiert.
- Constant-time-Vergleich (`secrets.compare_digest`), kein Key in Logs/URLs.
- **Tailscale-first:** Gateway-Port nicht publishen; erreichbar nur compose-intern und im
  Tailnet. Public-Betrieb nur mit Caddy (TLS) **und** zusätzlicher Auth-Schicht.
- Ratelimit (Token-Bucket) auch im Single-User-Betrieb: schützt Abo-Kontingent vor
  Amok-Clients/Schleifen.
- CORS: Default `same-origin`; nur die PWA-Origin whitelisten.

## 3. Secrets-Handling

- Alle Secrets via `.env` (gitignored) bzw. Docker-Secrets; `.env.example` ohne Werte.
- **Startup-Guard im Gateway:** Abbruch, wenn `ANTHROPIC_API_KEY` gesetzt ist
  (verhindert stille API-Abrechnung am Abo vorbei).
- OAuth-Token (`CLAUDE_CODE_OAUTH_TOKEN`) nur im `claude-core`-Container, nie im
  Harness/FE; Re-Mint-Prozedur (`claude setup-token`) im Runbook.
- Provider-Keys (ElevenLabs, Groq) nur im Gateway; Harness/FE kennen sie nicht.

## 4. Container-Härtung

- Docker-Socket-Mount **nur** im Gateway (nötig für `docker exec`), nicht im Harness.
  - Ausbaustufe G5+: Socket-Proxy (z. B. `tecnativa/docker-socket-proxy`) mit
    Whitelist nur für `exec` auf `claude-core` — voller Socket = root-äquivalent.
- `claude-core`: non-root-User, Workspace-Volume ist die einzige Schreibfläche,
  `--allowedTools`-Restriktion auf das, was die Doku-Skills brauchen.
- Gateway/Harness: non-root, `read_only: true` Root-FS wo möglich, `no-new-privileges`.
- Images: Versionen pinnen (CLI-Version!), regelmäßige Rebuilds für Patches.

## 5. Eingabe-/Ausgabe-Sicherheit

- **Prompt-Injection-Bewusstsein:** Claude Code führt Tools aus. Eingaben kommen zwar nur
  vom Inhaber, aber verarbeitete Inhalte (Webseiten, Dokumente) können Anweisungen
  enthalten → Tools restriktiv halten, Netzwerkzugriff des `claude-core`-Containers auf
  das Nötige begrenzen.
- `/artifacts/{path}`: Path-Traversal-Schutz (realpath + Prefix-Check), read-only, nur
  `output/`-Verzeichnis.
- Request-Limits: Body ≤ 10 MB, Audio ≤ 25 MB, Historien-Trunkierung mit Warn-Header.
- Fehlerantworten ohne interne Pfade/Stacktraces (Mapping in `errors.py`).

## 6. Beobachtbarkeit & Betrieb

- Audit-Log pro Request: Zeit, Route, session_key (gehasht), Dauer, Usage-Tokens —
  **keine Prompts/Inhalte** im Log (Dokumente können vertraulich sein).
- `/health`: Liveness + Auth-Status des Abo-Tokens (erkennt abgelaufenen Token früh).
- Backups: SQLite-Session-Map + `workspace/output` in die bestehende VPS-Backup-Routine.
- Update-Runbook: CLI-Update ⇒ Contract-Test für `stream-json` vor Redeploy.

## 7. Checkliste Go-Live (G5)

- [ ] Gateway nur im Tailnet erreichbar (Portscan von extern: zu)
- [ ] `ANTHROPIC_API_KEY`-Guard getestet
- [ ] Key-Rotation einmal durchgespielt
- [ ] Socket-Proxy aktiv oder Risiko dokumentiert akzeptiert
- [ ] Ratelimit + Idle-Timeout getestet
- [ ] Audit-Log inhaltsfrei verifiziert
- [ ] Token-Ablauf-Alarm im `/health`-Monitoring
- [ ] ToS-Absatz (§1) im README sichtbar
