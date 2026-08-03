---
name: copilot-cli-no-thinking-trace
description: "GitHub Copilot CLI (`copilot.exe`) does not expose model chain-of-thought; only a literal \"...thinking\" placeholder, in both interactive and -p modes."
metadata: 
  node_type: memory
  type: reference
  originSessionId: 926aecde-d39c-4168-810c-9325f0126746
---

GitHub Copilot CLI (`copilot.exe`) does NOT stream reasoning/thinking traces.
Both interactive mode and non-interactive `-p` mode display only a literal
`...thinking` placeholder while the model reasons, then emit just the final
answer text. Confirmed by direct test 2026-05-26 against company SSO account.

**Why:** the CLI is built for terminal UX; intermediate reasoning is not part
of its surfaced output. This is a CLI limitation, not a model limitation —
Haiku/Sonnet/Opus all support extended thinking via the Anthropic API directly.

**How to apply:** Any wrapper around `copilot.exe` (see `C:\devel\aweussom\python\copilot-proxy\`)
cannot render `<think>` blocks or reasoning summaries — the data is not in stdout.
NoLlama-style think-block UI degrades gracefully (no tags → no block) but
features like "Just answer me, dammit" are inert against this backend. If
thinking visibility matters, the alternatives are the Anthropic SDK directly
(extended thinking parameter) or the `claude` CLI.
