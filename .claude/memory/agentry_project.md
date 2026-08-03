---
name: agentry-project
description: "Persistent OpenAI-compatible proxy in front of `copilot --acp`, built to kill the per-turn spawn cost when using CLI coding-agents as automation backends. Lives at C:\\devel\\aweussom\\python\\agentry, published to github.com/aweussom/agentry."
metadata: 
  node_type: memory
  type: reference
  originSessionId: 926aecde-d39c-4168-810c-9325f0126746
---

**Project:** `agentry` — local Flask proxy that wraps `copilot --acp` (and
eventually other coding-agent CLIs) as an OpenAI-compatible chat API over
localhost. Renamed mid-spike from `copilot-proxy`.

**Location:**
  - Local: `C:\devel\aweussom\python\agentry\`
  - Remote: `git@github.com:aweussom/agentry.git` (private, user's personal
    GitHub `aweussom`)

**Why it exists:** the user runs `claude-code -p` (and similar coding-agent
CLIs) heavily in scripted workflows. Each invocation pays ~5 s of fixed
overhead — process spawn, ACP/MCP handshakes, session creation — before the
model produces a single token. Agentry holds one CLI subprocess persistent
across requests and routes prompts via the Agent Client Protocol, dropping
turn time from ~8 s (`-p` mode) to ~2-3 s (`api_ms` floor) for short
prompts.

**Architecture in one paragraph:** Flask server + minimal JSON-RPC client
over the subprocess's stdio. Agent->client requests for tools / files /
permissions are auto-denied with `-32601` to keep the proxy a pure chat
client. Per-session reasoning effort applied via `session/set_config_option`
because the `--reasoning-effort` CLI flag is silently ignored in `--acp`
mode. Local `.github/copilot-instructions.md` shipped with the project to
override global custom instructions that were leaking `<system_reminder>`
context into prompts.

**Status at handoff (2026-05-26):** working API + bundled web UI for
verification; tested on the free GitHub tier (`gpt-5-mini` / `gpt-4.1` /
`claude-haiku-4.5`). Web UI is a verification tool — the real surface is
the OpenAI-compatible HTTP endpoints. See `TODO.md` for backlog: chiefly
evaluating alternative backends (claude-code, qwen3-code, antigravity-cli,
codex) and a backend-as-plugin refactor for multi-backend support.

**Spun out from this NoLlama working dir.** The two projects share no
code; agentry has its own git repo at its own path. If the user mentions
"agentry" or "the copilot proxy" or "that persistent wrapper thing", that
is this project.

Related memories: [[copilot-cli-no-thinking-trace]],
[[vendor-cli-api-wrap-tos-risk]].
