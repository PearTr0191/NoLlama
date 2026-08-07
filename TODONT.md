# TODONT

Things we tried that didn't work, or that work but aren't worth doing. Each
entry explains *why not* so we don't re-litigate it in six months.

## Gemma 4 on the NPU (2026-08-07)

Idea: Gemma 4 launched this week; the E-series (E2B/E4B) are edge-sized
multimodal models with Intel pre-exports — natural NPU candidates, and the
blog post promised we'd test them.

**Verdict:** no Gemma 4 on the NPU for now, on any precision. CPU (and
presumably XMX GPU) is the way to run them.

**Why not (285K NPU, driver 32.0.100.4778, genai 2026.3):**
- `gemma-4-E4B-it-int8-ov` (Intel's own export) **compiles** for NPU
  (103 s) but generates **garbage at 0.5 tok/s** — multilingual token
  salad, three identical runs. The same file on CPU: 13.0 tok/s, perfectly
  coherent. Export is sound; the NPU path is numerically broken for
  `Gemma4ForConditionalGeneration`.
- int4 variants are already documented (zenn.dev, 2026-08) to crash the
  vpux compiler with the duplicated-names bug; we did not re-prove that.
- Both failure modes differ from the LFM int8 traps (fast-garbage /
  slow-correct) — this is slow-AND-garbage, a distinct NPU-path defect.

The models themselves are good: `gemma-4-26b-a4b-it-int4-ov` (VLM MoE,
128 experts) does **21.0 tok/s steady-state on the 285K CPU**, coherent,
16 s load. E4B int8 does 13.0 on CPU. Gemma 4 belongs in the CPU/GPU
columns, not the NPU column.

**Update 2026-08-07 (laptop NPU4, Arc 140V machine):** same E4B int8 on
the newer NPU generation produces **coherent** output — but at
**0.1 tok/s** (8 minutes per answer, three consistent runs). So two
separate defects: the numerical garbage is specific to the older NPU
arch/driver (3720 wrong, NPU4 right), while the speed is broken on BOTH
generations (0.1-0.5 tok/s smells like most of the graph falling back off
the NPU via NPUW partitioning). Verdict unchanged — no Gemma 4 on any NPU
we own — but the upstream report can now be precise: wrong-on-3720,
~100x-too-slow-everywhere. For comparison the same file does 16.4 tok/s
on the same laptop's GPU and 13.0 on the desktop CPU.

Re-evaluate if: an NPU driver or openvino release notes gemma4 fixes —
retest is `scripts/vlm-bench.py`, three minutes; or Intel ships a
`-int4-cw-ov` build of a gemma-4 (none exist today, unlike gemma-3).

## OFFLOAD_RATIO (2026.3 MoE disk offload) on the desktop 285K iGPU (2026-08-06)

Idea: OpenVINO 2026.3's MoE disk offload ("30B on 16 GB of memory") should
let big MoE models (Qwen3.6-35B-A3B, Qwen3-30B-A3B, and Dmitriy's 74 GB
Qwen3-Coder-Next from #19) run on this 33 GB-shared-memory iGPU.

**Verdict:** could not be made to work on this machine, on ANY model, at ANY
ratio, after a full day of controlled experiments. Do not recommend it to
users (incl. #19) as more than "exists upstream, unverified by us."

**What was measured (genai 2026.3.0, iGPU shared mem 33 GB, 64 GB RAM,
141 GB pagefile):**
- Qwen3.6-35B-A3B int4 VLM (2026.2 export): USM **Device** OOM (512 MB
  alloc) at ratio absent/40/90 — identical failure, ~9.5 min in. Compiling
  its language model directly with the property (no VLM wrapper) fails the
  same, so it is not a property-forwarding problem.
- Qwen3-30B-A3B-int4-ov, Intel pre-convert (2026.0 export): ratio 0 → USM
  **Host** OOM (384 MB); ratio 90 → USM Device OOM, one minute later and
  after staging ~120 GB of host commit. The offload machinery clearly
  *engages* — and still fails.
- LFM2-24B-A2B-int4-ov, Intel pre-convert (2026.2 export, **11.6 GB** —
  fits the 33 GB pool three times over): ratio 0 AND 90 → USM Host OOM
  (384 MB). An 11.6 GB model failing a 33 GB device on load is the smoking
  gun: the failure is in the GPU plugin's **weight-staging phase**, before
  any device-residency savings from offload can apply.
- Control that the pool itself works: Qwen3-8B int4 (~5 GB) and
  Qwen2.5-Coder-14B (~8 GB) load and generate fine on this iGPU. The
  practical ceiling on this box sits between ~8 and ~11.6 GB for MoE IRs.

**ROOT CAUSE (definitive, from source + device query):** the entire MoE
fusion path is gated in `transformations_pipeline.cpp`:

```cpp
// Gated on supports_immad (systolic-only) and oneDNN (required for expert GEMM dispatch).
if (device_info.supports_immad && config.get_use_onednn() && !config.get_moe_disable_fusion())
```

`supports_immad` = XMX/DPAS systolic hardware. The desktop 285K's Xe-LPG
iGPU has none (`OPTIMIZATION_CAPABILITIES` lists no `GPU_HW_MATMUL`;
verified 2026-08-06). No XMX → no TiledMoeBlock→MOECompressed fusion →
`OFFLOAD_RATIO` is a **silent no-op**, and experts stay as giant plain
constants — which is also why big-MoE loads OOM in staging on this device.
Proven end-to-end on a fusable IR: LFM2-8B-A1B exported fresh with the
2026.3 stack (tiled `u4 [32,1792,16,128]` expert constants confirmed in
the XML) loads fine and shows byte-identical device memory (14.91 GB) and
identical tok/s at ratio 0 and 90.

Intel's demos run on XMX-capable GPUs (Lunar Lake Arc 140V, Panther Lake,
Arc dGPUs). The release notes never mention the hardware gate.

**Consequence:** MoE disk offload is a hardware capability, not a software
setting, on this box. Raising the pagefile to re-export Qwen3-30B-A3B is
pointless *for offload on this machine* (the export itself would still be
useful only on an XMX-capable device). The Arc 140V laptop (original
NoLlama dev machine) HAS XMX — that is the machine to validate offload on.

Re-evaluate if: (a) testing on an XMX GPU (Arc 140V laptop / any Arc dGPU)
— use a fresh-stack export, ratio 0 vs 90, `GPU_MEMORY_STATISTICS`;
(b) Intel lifts the immad gate for non-systolic GPUs in a future release
(watch `transformations_pipeline.cpp`); (c) recommending it to anyone —
ask for their GPU model first, `OPTIMIZATION_CAPABILITIES` containing
`GPU_HW_MATMUL` is the tell.

**Update 2026-08-06 (same evening):** condition (a) tested on the Arc 140V
laptop (Core Ultra 7 258V, XMX confirmed) — **offload works exactly as
advertised there**. LFM2-8B-A1B int4: 4.10 GB resident at ratio 0 →
0.70 GB at ratio 90 (−83%), SSD streaming visible. Qwen3-30B-A3B int4
(15.2 GB weights, Intel's 2026.0 pre-convert — so old IRs DO fuse on XMX;
the tiled layout was never the blocker): loads and generates at ratio 90
with **2.35 GB resident**, 2.5 tok/s (ratio was oversized; tuning the knee
is follow-up). The verdict above is thus purely about non-XMX hardware —
the feature itself is real, first reproduction outside Intel we know of.
NoLlama grew `--offload-ratio` the same evening, with a startup warning on
non-XMX GPUs. install.ps1 surfaces XMX at device detection.

**Update 2026-08-07 (steady-state correction — the evening numbers above
were 2-5× too pessimistic):** the offload LRU needs ~60 tokens to warm,
and single-generate measurements reported cold-cache speed as the verdict.
Proper steady-state on the 140V, Qwen3-30B int4: ratio 30 → **25.3 tok/s**
(interactive — matches the 24-core desktop CPU running the same model
resident), 50 → 22.1, 90 → 5.1. Two benchmark bugs fixed the same morning:
warm-up contamination, and rate computed from ASSUMED token counts (a
4-token "Hello!" + EOS once reported 645 tok/s — real LFM2-8B GPU number
is 86.8). Also found: **a second generate() on an offload-active plain
pipeline hangs in native code, uninterruptible** (140V, 30B ratio 50) —
upstream-repro-worthy, and it means NoLlama's own `--offload-ratio` serving
path (which reuses one pipeline across requests, though via the CB backend,
not the plain pipeline) MUST be verified with two sequential chat requests
before recommending the flag in production.

## int8 exports of LFM2 / LFM2.5 for the NPU (2026-08-06)

Idea: channel-wise int4 is the lossiest int4 variant and the NPU forces it,
so ship int8 builds of the LFM models for quality-sensitive use — it worked
for SmolLM3-3B (int8-cw-sym: coherent, 12.3 tok/s vs int4-cw's 23.3 on the
285K NPU, a fair trade).

**Verdict:** no publishable int8 variant exists for LFM2-family on NPU.
int4-cw is the only good configuration. The SmolLM3 result does NOT
generalize.

**Why not (both variants measured on 285K NPU, genai 2026.3):**
- `--weight-format int8 --sym --group-size -1` (mirroring the int4-cw
  recipe): compiles and runs FAST (32-33 tok/s) but generates garbage —
  LFM2-1.2B emits whitespace, LFM2.5-1.2B-Instruct emits "BY-AL-AN-AN-…"
  loops. Silent numerical breakage, not a crash: the worst failure mode.
- `--weight-format int8` (asymmetric, Intel's own recipe — their
  LFM2.5-350M-int8-ov uses it): output is coherent but decode is
  **1.4 tok/s** (119 tokens in 89 s). Intel's own 350M reference runs
  4.5 tok/s the same way — asymmetric zero-points evidently fall off the
  NPU fast path. Correct but unusable.
- SmolLM3-3B int8-cw-sym is fine (fast AND coherent), so this is
  LFM-architecture-specific (its short-conv/linear-attention blocks),
  not a general int8-on-NPU rule.

Re-evaluate if: a newer NPU driver or openvino release changes either half
(retest is two 5-minute benches with scratchpad `npu_bench.py`-style
timing), or Intel publishes a fast LFM int8 NPU build — read its rt_info
for the recipe before assuming ours was wrong.

## Qwen3.6-35B-A3B (Qwen3.5-MoE arch) on the NPU (2026-08-06)

Idea: with OpenVINO 2026.3 passing regression, put the new Qwen3.6-35B-A3B
INT4 export on the NPU — NPU coverage is the stated priority of the 2026.3
move, and an A3B MoE (3B active) looks NPU-sized on paper.

**Verdict:** doesn't load. Not a memory problem — an architecture-vs-plugin
incompatibility. Serve this model on GPU/CPU only until the NPU plugin
catches up.

**Why not:**
- Both `VLMPipeline` and `LLMPipeline` on NPU fail in ~3 s at shape
  inference, before compile, with
  `Check '!dim::is_empty(minus_one_dim)' failed ...
  reshape_shape_inference.hpp:357` on node
  `__module.model.model.language_model/aten::index/Reshape`
  ("Non-'-1' output dimensions do not evenly divide the input dimensions").
  The NPU's static-shape import can't reshape a boolean-mask `aten::index`
  in the Qwen3_5Moe language model. genai 2026.3.0.0-3277, 285K NPU
  ("AI Boost"), driver as of 2026-08-06.
- It is *not* the earlier commit failure: that was fixed (141 GB pagefile,
  33 GB iGPU shared-memory override) and this failure reproduces identically
  with memory to spare. Don't respond to this error by adding RAM/pagefile.
- Nothing NoLlama can patch: the export is Intel-toolchain-fresh
  (OpenVINO 2026.2 export, optimum-intel 1.27.0.dev0) and the failure is in
  OpenVINO's NPU plugin shape inference, upstream of anything we configure
  (`MAX_PROMPT_LEN` etc. never comes into play).

Re-evaluate if: a later OpenVINO release notes NPU support for Qwen3.5-MoE /
`Qwen3_5MoeForConditionalGeneration` (retest is one `--scan`-verified dir +
a 3-second load attempt), or Intel publishes an NPU-targeted export of this
family.

## `--model-name` / `--model-description` override flags (2026-08-06)

Idea: let the user set the name shown in the web UI and reported as the
model ID, since renaming the model folder appeared to do nothing. Raised by
Dmitriy Teteruk (issue #19) after converting Qwen3-Coder-Next himself and
wanting it to show up as something sensible.

**Verdict:** don't add the flags. Fix the rename and add `--scan` instead.

**Why not:**
- **The bug was ours, not the interface's.** `model_display_name()` called
  `os.path.realpath()` unconditionally, so on a junction (which is what
  `install.ps1` creates) the name came from the *link target* and the user's
  rename was silently discarded. Renaming a directory is already the naming
  interface — it needed no documentation, no flag, and no knowledge. It just
  had to work. Now it does: the given name wins, and the link is only
  followed when the directory name is generic (`model/`, `gpu-model/`).
- **A flag puts the cost in the wrong place.** It has to be discovered in
  `--help`, then threaded through the generated `start.ps1`, then kept in
  sync per slot (primary, GPU, whisper). The person most likely to need it
  is the person least likely to be editing launch scripts — the exact user
  who reported it.
- **Most of what a description would say is already on disk, and more
  reliably.** The IR's model-level `<rt_info>` records the real nncf
  weight-compression mode, group size, ratio and AWQ flag; `config.json`
  gives architecture, layer count, context and MoE expert counts. `--scan`
  reports those as facts. A hand-typed description would just be an
  opportunity to be wrong — a folder named `-int4-ov` holding int8 weights
  is exactly the confusion the feature would have entrenched.

**The one thing detection genuinely cannot do:** recover the *variant*.
`config.json` in an OpenVINO export has no `_name_or_path`, and
Qwen3-Coder-Next vs Qwen3-Next-Instruct are identical in architecture and
geometry — indistinguishable from the files. That's precisely why the
directory name must stay authoritative for naming instead of being
second-guessed by a heuristic.

Re-evaluate if: someone needs two directories with the same basename served
under different IDs (two quantizations of one model in one process). That's
a real case a rename can't express — but nobody has asked for it, and the
dual-slot routing (`_route_request`) would need work first anyway.

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
