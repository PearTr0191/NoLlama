# TODONT

Things we tried that didn't work, or that work but aren't worth doing. Each
entry explains *why not* so we don't re-litigate it in six months.

## `--cpu-model-dir` — a third generative slot in one process (2026-08-03)

Idea: add a third `DeviceSlot` so one NoLlama process could serve chat +
vision + coding at once (e.g. NPU chat, iGPU vision, CPU coder). Prompted by
a user question (Manuel Destouesse, email 2026-08-02) asking whether three
models could run simultaneously.

**Verdict:** works in principle, don't build it.

**Why not:**
- It adds a slot to serve the exact case we should be recommending *away*
  from NoLlama. CPU is the one device where Ollama is unambiguously the
  better tool (see the next entry) — so the feature's whole purpose is to
  do badly what a `ollama serve` next door does well.
- `_route_request` (`nollama.py:1142`) is built around exactly two
  generative slots: `for slot in (primary, secondary)` for explicit
  `model@DEVICE` selection, then a two-way heuristic (images → whichever
  slot is a VLM, text → the GPU if it holds an LLM, else primary). A third
  slot turns that heuristic into a policy question — with a coder model
  loaded, which slot gets an unlabelled text request? There's no good
  default, so it becomes config, which is the complexity we're avoiding.
- Memory-bound anyway on the target hardware. The NPU and iGPU both draw
  on system RAM, so three resident 4-bit models plus KV caches are
  competing for one pool on a 32 GB laptop. The device count was never the
  scarce resource.
- **Zero-code alternative already works:** two NoLlama instances on
  different `--port`/`--ollama-port` values, or the recommended split
  (NoLlama for NPU+iGPU, Ollama for the CPU model). Both are documented in
  README "When to use NoLlama, and when to use Ollama".

Re-evaluate if: a single-device machine ever needs three models on that one
device (the two-slot cap, not the device count, would then be the real
blocker) — or if OpenVINO's CPU path decisively beats llama.cpp, which
would reverse the entry below and with it this one's first argument.

## Recommending / building out NoLlama's CPU path (2026-08-03)

Recurring temptation: NoLlama already runs on CPU, the `--device CPU` path
is tested and works, and the desktop 285K benchmarks are respectable
(17.8 tok/s on Qwen3-8B INT4, faster than that box's iGPU *and* NPU). So
it's tempting to present CPU as a first-class NoLlama target and invest in
it — tool-calling on CPU is already enabled (`_tools_supported`).

**Verdict:** keep the CPU path as a working fallback, but recommend Ollama
for CPU-only users, and don't invest further in it.

**Why not:**
- Ollama's llama.cpp CPU backend is far more mature than our OpenVINO CPU
  path, `ollama pull` avoids the conversion/export problem entirely, and
  its tool calling uses per-model chat templates rather than our
  `render_tools_prompt` + `parse_tool_calls` regex approach. That parser
  already needs to recognize six native formats (Qwen3-Coder XML, Hermes,
  bare `<function=>`, Mistral, Llama, DeepSeek) precisely because models
  ignore our prompt — that's a maintenance treadmill Ollama doesn't have.
- ~~It contradicts the project's stated scope. NoLlama exists for the Intel
  **NPU**; GPU/CPU are explicitly provisional (README "Roadmap note"),
  kept only while OpenVINO is meaningfully faster. Advertising CPU dilutes
  the one claim nothing else makes.~~ *(Update 2026-08-04: the provisional
  stance is reversed — GPU/CPU are committed long-term, since no
  OpenVINO-class Ollama Intel backend is coming and most users run agents
  (OpenClaw) on GPU/CPU. This argument no longer applies; the entry's
  verdict still stands on the ecosystem-maturity argument above.)*
- Coexistence is free: Ollama keeps 11434, and NoLlama's port check
  (`nollama.py:2193`) already detects that and disables its own Ollama
  shim rather than failing. There is no integration cost to pay.

**Caveat — this verdict is not measured.** We have benchmarked NoLlama vs
Ollama on the Arc 140V iGPU (NoLlama ~1.6× faster on decode, 2026-06-16)
but **never on CPU**. `bench-results/` has `cpu-qwen3-at-CPU-*.json` for
NoLlama only. The recommendation above rests on ecosystem maturity and
scope, not on a throughput comparison.

Re-evaluate if: someone runs `benchmark.py --backend ollama` on CPU against
the same model/quantization and OpenVINO wins by a wide margin — that would
make CPU worth defending on the same measured grounds as the iGPU. Until
then, don't claim a speed verdict on CPU in docs or in replies to users.
