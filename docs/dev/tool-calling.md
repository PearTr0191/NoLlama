# Tool calling / agent turns

Read this before touching `tools` handling, `render_tools_prompt`,
`parse_tool_calls`, or the SSE heartbeat.

## Where it works

**GPU/iGPU + CPU**, on both LLM and VLM slots. VLM tool turns — including
images alongside tools — landed 2026-08-18; they are buffered like LLM tool
turns and generated through the VLM pipeline.

Gated by `_tools_supported`, i.e. `device_name in ("GPU","CPU")`. The
**NPU is excluded**: it has a hard prompt cap and small NPU-class models
can't drive agent loops, so when the NPU serves the request we ignore
`tools` and answer as plain chat. `/api/show` advertises the `tools`
capability only for GPU/CPU slots, so Copilot won't offer NPU models for
agent mode.

CPU is viable for agents on strong desktops (e.g. Core Ultra 9, many
cores) where prefill can beat a weak iGPU.

## How a turn works

Tool specs from the request's `tools` array are rendered into a system
prompt (Qwen3-Coder native format); the model's emitted call is parsed back
into OpenAI/Ollama `tool_calls`.

`parse_tool_calls` recognizes several native formats, because a model often
ignores our prompt and falls back to whatever it was trained on:

- Qwen3-Coder XML
- Hermes JSON-in-`<tool_call>`
- bare `<function=>` with no wrapper (Qwen2.5-Coder native)
- Mistral `[TOOL_CALLS]`
- Llama `<|python_tag|>`
- DeepSeek `<｜tool▁calls▁begin｜>` blocks
- bare-JSON fallback

See `render_tools_prompt` / `parse_tool_calls`.

Client surfaces: Copilot Chat 0.53+ hits `/v1/chat/completions` (delegates
to `chat_completions`); `/api/chat` is also handled.

## Buffered, not token-streamed (known limitation)

Tool-enabled turns are **buffered**: we must see the whole tool-call block
before emitting a structured `tool_calls` delta, so the full generation is
collected before the result is sent — no incremental tokens that turn.
True token streaming on tool turns (stream until a tool-call prefix
appears) is still TODO.

## Heartbeat — why it exists

A slow prefill on a big agent prompt trips client idle watchdogs
(Copilot/OpenClaw abort with no output after ~120s). So:

- the streaming tool path runs generation in a background thread and emits
  SSE keep-alive pings every `HEARTBEAT_SECS` (`_sse_tool_stream`);
- the plain stream path (`stream_llm`) pings the same way during a long
  prefill.

Big agent prompts (OpenClaw ships ~21k-token system prompts) prefill slowly
on weak iGPUs — ~6 min TTFT on the desktop 285K Xe-LPG. Mitigations: a
smaller coder model, CPU on strong desktops, trimming the client's tool
set, and the keep-alive above so turns complete instead of aborting.

OpenVINO **can't cancel a blocked prefill**, so an aborted client leaves
the generation churning — another reason to keep clients connected via
heartbeat. (Same root cause as the `/v1/cancel` caveat: cancel relies on
OpenVINO invoking the streamer callback.)

Note the "minutes of prefill" worry does not generalize: on the B60 class,
33k tokens prefill in ~9s on the plain pipeline.
