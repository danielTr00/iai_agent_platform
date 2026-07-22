FROM node:22-bookworm-slim

# Basis-Tools + git (Claude Code nutzt es fuer viele Workflows)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Claude Code CLI installieren
RUN npm install -g @anthropic-ai/claude-code

# Nicht-root-User fuer die Ausfuehrung
RUN useradd -ms /bin/bash agent
USER agent
WORKDIR /workspace

# Auth kommt zur Laufzeit ueber CLAUDE_CODE_OAUTH_TOKEN (docker-compose).
# Verifikation nach dem Start:  docker exec agent-os-claude-core claude /status
