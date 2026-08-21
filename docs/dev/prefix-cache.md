# Prefix cache, KV pool, prewarm

Read this before touching caching, the KV pool, prewarm, `--idle-timeout`,
TTFT logging or the memory preflight.

## Prefix (KV) caching — default on

**Default on** for GPU/CPU **LLM and VLM** slots. They load via the
continuous-batching backend:

```python
LLMPipeline/VLMPipeline(..., scheduler_config=SchedulerConfig(
    enable_prefix_caching=True, cache_size=slot.kv_pool_gb))
```

A repeated prompt prefix (an agent's fixed system prompt + tool schemas,
identical every turn) is prefilled once, not every turn — measured **~47×
faster** on a cached turn (24.4s→0.5s for a ~2k-token prefix on the 285K
CPU). Auto-invalidated by any prefix change, so no staleness.
`--no-prompt-cache` disables it.

VLMPipeline honoring `scheduler_config` was verified 2026-08-18 on 2026.3
release (140V: ~9k-token prefix 21.7s→3.9s TTFT) and 2026.4 nightly (B60:
33k tokens 54.5s→1.3s) — the earlier **"CB backend is LLM-only" note was
stale**. Documented upstream gap: openvino.genai#4343.

NPU slots keep the plain pipeline (no CB path; NPU keeps MAX_PROMPT_LEN).
Any device or runtime that can't build the CB backend falls back to the
plain pipeline with a warning.

**It is not only devices and runtimes — an individual IR can refuse.** The
CB path is built by rewriting the graph (`SDPAToPagedAttention`), so a model
exported without an SDPA op cannot take it at all:

```
No ScaledDotProductAttention operation observed in the graph,
cannot perform the SDPAToPagedAttention transformation.
); using plain pipeline
```

Measured 2026-08-21 on the B60 / 2026.3 release across Intel's three Gemma 4
exports — and it tracks the **exporter**, not the model size:

| IR | exported with | CB backend |
|---|---|---|
| `gemma-4-E2B-it-int4-ov` | transformers 5.5.4 | builds |
| `gemma-4-26b-a4b-it-int4-ov` | transformers 5.5.4 | builds |
| `gemma-4-E4B-it-int8-ov` | transformers **5.5.0** | **refuses** |

So a model can lose prefix caching for reasons invisible in its name, size
or precision. Check the load log rather than assuming; `/health`'s per-slot
`kv_pool_gb` is the other tell (null = fell back).

Cold-prefill trade-off, measured on the B60 with a 33k prompt: ~8.7s on the
plain pipeline vs 53.7s under CB (then 1.4s per repeat). **Agents win from
turn two; one-shot prompts pay more once.**

## KV pool sizing

**Auto-sized per slot** (`_resolve_kv_pool`): a third of what the weights
leave free in the device budget, floor 2 GB (`AUTO_KV_MIN_GB`), cap ~64k
tokens of the model's KV geometry (`AUTO_KV_TOKENS`). Sized from the
*total* budget, not free RAM, so it's stable across restarts and reloads.
The CB backend grows into the pool rather than allocating upfront, but
prefix-cached blocks are never released — hence the fraction.

`--cache-size-gb N` pins it and skips auto.

## Prewarm

`--prewarm <file>` prefills a saved agent prompt at startup, so even the
first turn is a cache hit instead of a cold prefill that can trip a
client's idle watchdog. The file **auto-captures** the first big prompt
served (`_maybe_capture_prewarm`, on both the OpenAI and Ollama chat
paths) — so the workflow is: run once → restart with `--prewarm`.

- Auto-enabled as `prewarm-<port>.json` when `--idle-timeout 0` (opt out
  with `--no-prewarm`).
- `--prewarm` implies `--idle-timeout 0`. An explicit nonzero timeout
  alongside it is **REFUSED at startup** (2026-08-18; was a warning).
  Unload discards the warmed cache, and the reload path deliberately does
  **not** re-warm — a synchronous re-warm would stall the triggering
  request pre-SSE and trip exactly the client watchdogs the heartbeat
  exists to defeat.
- Covers **VLM slots** too (2026-08-18): capture on both API surfaces, and
  the startup prefill replays through `parse_messages`' flattening so the
  cached token prefix matches real requests. Measured, Glimmer/B60: first
  turn after restart 12.4s → 0.65s TTFT.
- Slots whose runtime fell back to the plain pipeline report a **null**
  `kv_pool_gb` at
  load, so prewarm skips them instead of burning a 30B-scale prefill for
  nothing (this also keeps `/health` honest about a dead cache).

## Observability

Per-request log lines include TTFT — streaming: wall-clock to first token;
non-streaming: `perf_metrics` via `extract_perf`. A prefix-cache hit is
sub-second against a cold multi-second/minute prefill, so hits and misses
are visible **without instrumentation**.

`/health` carries `prompt_cache_info` (pinned `pool_gb` or null, plus
`auto`, plus the prewarm file) and per-slot `kv_pool_gb` (resolved size),
`last_ttft_ms` and `prewarmed`. `prompt_cache` stays a bare bool —
`start-openclaw.ps1` truth-tests it.

## Memory preflight

`_preflight_memory` at load **warns, never blocks**, when weights + KV pool
exceed the device budget (GPU: `GPU_DEVICE_TOTAL_MEM_SIZE`, which reflects
Windows' ~half-RAM iGPU policy and Intel's "Shared GPU Memory Override"
driver setting; CPU: total RAM). It logs the KV pool's token capacity from
`config.json` geometry — ~56 KB/token for a 7B coder, ~96 KB for 30B.

A too-small pool **hard-fails** generation with `Got unfinished
GenerationStatus` (issue #21); `explain_genai_error` annotates that error
with a `--cache-size-gb` hint wherever it surfaces.

VLM configs nest geometry under `text_config` — `_text_config` handles
that, which is what fixed the KV half of this preflight silently
no-op'ing on every VLM.
