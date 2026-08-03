---
name: openclaw-nollama-integration
description: "How OpenCLAW is wired to NoLlama, the heartbeat/CPU/parser changes it required, and the prefill wall on the 285K desktop iGPU."
metadata: 
  node_type: memory
  type: project
  originSessionId: be7d191b-97b8-4ef4-ad76-6439300bf3c1
---

Getting the **OpenCLAW** coding agent running on **NoLlama** (Intel local inference), worked through 2026-06-27/28. The integration is proven correct end-to-end; the practical limit is prefill throughput on weak hardware.

**Setup (all done):**
- OpenCLAW installed via `npm i -g openclaw@latest` (v2026.6.10). Config at `~/.openclaw/openclaw.json`; manage non-interactively with `openclaw config patch/get/set/schema/validate` (bare `openclaw config` launches the interactive wizard — avoid in scripts).
- Provider wired: `models.providers.nollama` = `{baseUrl: "http://localhost:8000/v1", api: "openai-completions", apiKey: "local-no-auth"}`, model id `Qwen2.5-Coder-14B-Instruct@GPU`, `agents.defaults.model.primary: "nollama/<id>"`.
- Gateway made non-interactive: `gateway.auth.mode=none` + `gateway.bind=loopback` (single-user loopback). `openclaw agent --local` runs embedded (no gateway service needed); `openclaw chat` is the TUI.
- Prompt trim: `tools.profile=coding` (+ web search / memorySearch / startupContext off). NOTE: `tools.allow` needs exact tool ids not in the schema — don't guess; the `coding` profile only removed 5 minor tools (agents_list, gateway, message, nodes, tts), so it's a small cut.

**NoLlama changes this required (all committed to main):**
- **Heartbeat** (`_sse_tool_stream` + `stream_llm`): tool turns are buffered (whole generation before `tool_calls`), and agent clients have a **~120s idle watchdog**. On a slow prefill the client aborted with no output (and OpenVINO can't cancel a blocked prefill, so the abandoned run kept churning). Fix: run generation in a background thread, emit an **empty-content delta chunk** every `HEARTBEAT_SECS` (15s). Empty-content delta, NOT an SSE `: ping` comment — OpenCLAW's watchdog says "produced no reply", so it likely tracks reply events, not bytes.
- **CPU tool support**: `_tools_supported` now allows `GPU` and `CPU` (excludes only NPU). The old GPU-only gate over-excluded CPU; a capable coder on a strong desktop CPU drives tool loops fine.
- **Parser**: Qwen2.5-Coder emits bare `<function=...>` with **no `<tool_call>` wrapper** — `parse_tool_calls` now handles that. (Live-test find; unit tests had used the wrapped form.)

**Prefix caching (landed 2026-06-28, default-on):** GPU/CPU LLM slots load via the continuous-batching backend with `SchedulerConfig(enable_prefix_caching=True, cache_size=PROMPT_CACHE_GB)` — the identical agent system prompt is prefilled once, not every turn. Measured ~47× on a cached turn (24.4s→0.5s, ~2k-token prefix, 285K CPU). `--no-prompt-cache` / `--cache-size-gb`. This is the real fix for the prefill wall below — after turn 1, agent turns are fast. Sanctioned despite the general anti-cache lean ([[avoid-caching-prefer-wait]]) because it auto-invalidates (no staleness). **`--prewarm <file>`** (added same day) prefills a saved prompt at startup so even turn 1 is a cache hit — self-bootstrapping (auto-captures the first big system prompt to the file; run once → restart with `--prewarm`). Verified end-to-end: OpenCLAW turn 1 cold ~3m25s (one watchdog retry; the abandoned attempt warmed the cache so the retry prefilled in 68s), turn 2 (cached) ~14s wall, clean. Further toolset trimming is low-value now that the prompt is cached — only X-search was disabled; the core coding tool schemas must stay.

**Best device is hardware-specific (confirmed by running OpenCLAW on both machines):**
- **Laptop (Lunar Lake, ARC 140V — 8-core iGPU + LPDDR5X):** the **iGPU runs the agent well**; the **CPU is more or less useless** for it. So GPU is the pick.
- **Desktop (Arrow Lake 285K, weak 4-core Xe-LPG iGPU + DDR5):** the **CPU beats the iGPU** (the 382s-prefill wall was the iGPU; CPU is faster there).
So `start-openclaw.ps1 -Device Auto` (default) prefers a real GPU when present — correct for laptop ARC / discrete ARC (the common case); the weak desktop iGPU is the documented exception (`-Device CPU`). Don't assume CPU-or-GPU universally; it flips by machine.

**The wall (Core Ultra 9 285K desktop):** OpenCLAW's system prompt is **~21k tokens (84k chars)**; the desktop iGPU is the weak 4-core Xe-LPG and prefilled it in **~382s** — far over the 120s watchdog. Levers: smaller coder (Qwen2.5-Coder **7B** is in the menu), **CPU** (this 285K's CPU out-prefills its tiny iGPU — README shows CPU > iGPU here), trim the client prompt. The heartbeat makes turns *complete* regardless; speed still needs a smaller model / CPU. The **laptop ARC 140V** (8 cores + LPDDR5X) is the better device but may still be ~150-250s prefill with 14B.

**`@`-fragility:** OpenCLAW strips `@GPU` from the model id (sends bare `Qwen2.5-Coder-14B-Instruct`). Works here only because the bare name uniquely matches the GPU slot (NPU is Qwen3-8B). The robust general fix is a server-side **tools⇒GPU routing override** (route any tools-bearing request to a GPU/CPU slot) — see [[nollama-npu-niche-scope]]. Documented in OPENCLAW-PLAN.md.
