# NoLlama

OpenAI-compatible LLM/VLM server for Intel hardware. NPU-first.

## Architecture

- `nollama.py` — Flask server, DeviceSlot class per device, auto-detects VLM/LLM from config.json
- NPU: LLMPipeline with MAX_PROMPT_LEN=4096, streaming via SSE
- GPU: VLMPipeline (images) or LLMPipeline (text). Both stream as of openvino-genai 2026.1 — verified on Arc 140V iGPU.
- Prefix (KV) caching: **default on** for GPU/CPU **LLM** slots — they load via the
  continuous-batching backend (`LLMPipeline(..., scheduler_config=SchedulerConfig(
  enable_prefix_caching=True, cache_size=PROMPT_CACHE_GB))`). A repeated prompt prefix (an
  agent's fixed system prompt + tool schemas, identical every turn) is prefilled once, not
  every turn — measured ~47× faster on a cached turn (24.4s→0.5s for a ~2k-token prefix on
  the 285K CPU). Auto-invalidated by any prefix change (no staleness). `--no-prompt-cache`
  disables it; `--cache-size-gb N` sizes the pool (default 2). NPU and VLM slots keep the
  plain pipeline (NPU has no CB path; it keeps MAX_PROMPT_LEN). Falls back to the plain
  pipeline with a warning if a device can't build the CB backend. `--prewarm <file>`
  prefills a saved agent prompt at startup (the file auto-captures the first big prompt
  served via `_maybe_capture_prewarm`, so: run once → restart with `--prewarm`) so even the
  first turn is a cache hit instead of a cold prefill that can trip a client's idle watchdog.
- Whisper: WhisperSlot + WhisperPipeline for STT, `POST /v1/audio/transcriptions`, CPU or GPU
- OpenVINO GenAI may unify VLM/LLMPipeline — when that happens, simplify the dual-pipeline routing
- Routing: images go to GPU, text goes to NPU (or GPU if no NPU)
- Web UI: `templates/index.html` + `static/css/style.css` + `static/js/app.js`
- Collapsible `<think>` blocks, "Just answer me, dammit!" button, temperature slider
- `threaded=True` on Flask, concurrency via per-device locks
- `models.json` — curated model registry (npu, gpu_vlm, gpu_llm, whisper categories)
- `install.ps1` detects devices, shows model menu, generates `start.ps1`
- Tool calling: **GPU/iGPU + CPU** (gated by `_tools_supported`, i.e.
  `device_name in ("GPU","CPU")`); the **NPU is excluded** — it has a hard prompt cap and
  small NPU-class models can't drive agent loops, so when the NPU serves the request we
  ignore `tools` and answer as plain chat. `/api/show` advertises the `tools` capability only
  for GPU/CPU slots (so Copilot won't offer NPU models for agent mode). CPU is viable for
  agents on strong desktops (e.g. Core Ultra 9, many cores) where prefill can beat a weak
  iGPU. Tool specs from the request `tools` array are rendered into a system prompt
  (Qwen3-Coder native format); the model's emitted call is parsed back into OpenAI/Ollama
  `tool_calls`. `parse_tool_calls` recognizes several native formats, since a model
  often ignores our prompt and falls back to what it was trained on: Qwen3-Coder XML, Hermes
  JSON-in-`<tool_call>`, **bare `<function=>` with no wrapper (Qwen2.5-Coder native)**, Mistral
  `[TOOL_CALLS]`, Llama `<|python_tag|>`, DeepSeek `<｜tool▁calls▁begin｜>` blocks, plus a
  bare-JSON fallback. See `render_tools_prompt` / `parse_tool_calls`. Copilot Chat 0.53+ hits
  `/v1/chat/completions` (delegates to `chat_completions`); `/api/chat` also handled.

## Environment

- Primary: Windows 11, Python 3.10+
- Cross-platform: scripts use `#requires -Version 7.0` and branch on
  `$IsWindows`. Linux + PowerShell 7 is confirmed working (user-reported
  on Core Ultra 7 258V with NPU + GPU, issue #6). There is no install.sh —
  Linux runs the same install.ps1 via pwsh. On Linux, NPU/GPU need the
  Intel userspace drivers installed or only CPU is detected; the Linux NPU
  stack (`intel-npu-driver`) is less mature than Windows.
- Intel Core Ultra (NPU) + Intel ARC 140V 16GB (GPU)
- OpenVINO 2026.1+ with openvino_genai
- venv in `venv/`, activate before running

## Development preferences

- Read `TODONT.md` before proposing anything structural — it records approaches
  already rejected, with the reason. Add an entry whenever one is abandoned.
- Keep it simple. One file (`nollama.py`) is fine. Don't split into modules unless it gets unwieldy.
- PowerShell for install/launch scripts (Windows-native users).
- Runtime flags over hardcoded config (e.g. `--port`, `--device`).
- When testing, use small payloads / short prompts. Don't run full model loads unless needed.
- VLM prompts must be dead simple for small models (3B). One question, one answer, minimal JSON. All logic in Python, not in the prompt.
- Qwen3-VL is now pre-exported by Intel (OpenVINO/Qwen3-VL-8B-Instruct-int4-ov, May 2026) — not yet tested here. Earlier note about optimum-intel support is obsolete.

## Known issues

- NPU default prompt limit is 1024 tokens — we override to MAX_PROMPT_LEN=4096
- (resolved 2026-05-25) VLMPipeline gained streaming support in openvino-genai 2026.1; verified on Arc 140V iGPU at ~11 tok/s decode.
- Qwen3 thinking models can exhaust token budget on `<think>` before producing an answer
- Cancel (`/v1/cancel`) relies on OpenVINO invoking the streamer callback. If the native code blocks without yielding, cancel won't take effect — generation completes naturally.
- Chat history unbounded in web UI — user clears with Ctrl+N when long sessions approach MAX_PROMPT_LEN
- Tool-enabled turns are buffered, not token-streamed: we must see the whole tool-call block
  before emitting a structured `tool_calls` delta, so the full generation is collected before
  the result is sent (no incremental tokens that turn). To stop a slow prefill on a big agent
  prompt from tripping client idle watchdogs (Copilot/OpenClaw abort with no output after
  ~120s), the streaming tool path runs generation in a background thread and emits SSE
  keep-alive pings every `HEARTBEAT_SECS` (`_sse_tool_stream`); the plain stream path
  (`stream_llm`) pings the same way during a long prefill. True token streaming on tool turns
  (stream until a tool-call prefix appears) is still TODO.
- Big agent prompts (OpenClaw ships ~21k-token system prompts) prefill slowly on weak iGPUs
  (~6 min TTFT on the desktop 285K Xe-LPG). Mitigations: smaller coder model, CPU on strong
  desktops, trimming the client's tool set, and the keep-alive above so turns complete instead
  of aborting. OpenVINO can't cancel a blocked prefill, so an aborted client leaves the
  generation churning — another reason to keep clients connected via heartbeat.

## Verified models

- Qwen3-8B (INT4-CW) on NPU — recommended, needs MAX_PROMPT_LEN=4096
- Phi 3.5 Mini (INT4-CW) on NPU — smaller, faster
- DeepSeek-R1-1.5B (INT4-CW) on NPU — works but terrible quality (testing only)
- Gemma 3 4B Vision (INT4) on GPU — fast VLM
- Qwen2.5-VL-3B/7B (INT4/INT8) on GPU — proven for image tasks
- Qwen3-30B-A3B on GPU — needs >16GB VRAM, falls back to CPU silently on 16GB cards
