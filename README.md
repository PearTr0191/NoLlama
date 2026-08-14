# NoLlama

**Local LLM server for the full Intel stack.** NPU, ARC iGPU, ARC discrete, CPU.
OpenAI + Ollama APIs. One server, every Intel device.

No NVIDIA required. No Ollama install. No llama.cpp. **No problem.**

Runs on Intel Core Ultra laptops (NPU + ARC iGPU), desktops with ARC
discrete GPUs (A770, B580), or any Intel CPU. Automatically detects your
hardware, picks the best device, and exposes both OpenAI and Ollama
compatible APIs — so any client that speaks to either just works.

**It drives coding agents, too.** VS Code Copilot Chat and OpenClaw run against
NoLlama with local **tool-calling** on your Intel GPU or CPU — no cloud, no
NVIDIA. See [Agent tools & coding assistants](#agent-tools--coding-assistants-vs-code-copilot-openclaw).

![NoLlama in action](docs/images/nollama-demo.gif)

## When to use NoLlama, and when to use Ollama

NoLlama is not trying to replace Ollama. It exists to cover the Intel
devices Ollama doesn't reach well. Pick per device, not per project:

| You want to run on | Use | Why |
|---|---|---|
| **Intel NPU** (Core Ultra "AI Boost") | **NoLlama** | Ollama can't target the NPU at all. This is NoLlama's reason to exist. |
| **Intel iGPU / ARC**, text | **NoLlama** (for now) | OpenVINO INT4 is ~1.6× faster on decode than Ollama's Vulkan backend on an Arc 140V — [measured below](#nollama-vs-ollama-on-the-arc-140v-igpu). Ollama also needs `OLLAMA_IGPU_ENABLE=1` or it silently falls back to CPU. |
| **Intel iGPU / ARC**, images | **NoLlama** | Local vision models (Qwen3-VL, Gemma 3 Vision) on Intel GPUs — Ollama has no Intel path for these. |
| **CPU only** | **Ollama** | llama.cpp's CPU backend is mature and better supported than NoLlama's OpenVINO CPU path, `ollama pull` is easier than model conversion, and its tool-calling uses proper per-model templates rather than NoLlama's prompt-rendering-plus-parsing. (We haven't benchmarked the two on CPU — the recommendation is about maturity, not measured speed.) |
| **NVIDIA or AMD GPU** | **Ollama** | NoLlama is [Intel-only by design](#intel-only--by-design). Ollama will always do Ollama better. |

**They run side by side.** Ollama keeps its default port 11434; NoLlama
notices that port is taken and disables its own Ollama-compatible shim
automatically (or set `--ollama-port 0` to be explicit), leaving NoLlama's
OpenAI API on port 8000. So a three-role setup — chat, vision, coding — is
one NoLlama process plus Ollama:

```powershell
# NoLlama: NPU chat + iGPU vision, simultaneously → http://localhost:8000/v1
python nollama.py --device NPU --model-dir model --gpu-model-dir gpu-model

# Ollama: coding model on the CPU → http://localhost:11434
ollama serve
ollama pull qwen2.5-coder:7b
```

The binding constraint is memory, not device count: the NPU and iGPU both
draw on system RAM, and Ollama's CPU model adds to the same pool. 32 GB
handles an 8B chat + 4B vision + 7B coder at 4-bit; 64 GB is comfortable.
NoLlama unloads idle slots after 30 minutes (`--idle-timeout`).

> One process serves at most two generative models (a primary on any device
> plus an optional GPU secondary via `--gpu-model-dir`), plus Whisper. Three
> models means two NoLlama instances on different ports, or — better — the
> split above.

## Speed at a glance

Measured steady-state decode, tok/s, int4 weights. Every number is from a
real run on hardware named below — no vendor sheets. Rule of thumb behind
all of them: **decode ≈ practical memory bandwidth ÷ active weight bytes**,
because LLM decode streams the whole model per token.

| Model (int4) | NPU | iGPU | Arc dGPU | CPU (DDR5) | CPU (DDR4) |
|---|---|---|---|---|---|
| SmolLM3-3B (~2 GB) | 23.3 ᵃ | 29.7 ᵃ | *wanted* | 37.5 ᵃ | 23.0 ᵇ |
| Qwen3-8B (~5 GB) | 10.0 ᵃ | 21.7 ᶜ / 15.4 ᵃ | *wanted* | 17.8 ᵃ | *wanted* |
| Qwen3-30B-A3B MoE (~17 GB, ~2 GB active) | n/a — over the NPU's size class | 25.3 ᶜ (`--offload-ratio 30`) | *wanted* (B60 numbers coming) | ~6 † | ‡ |

ᵃ Core Ultra 9 **285K** desktop — DDR5-6400 (~100 GB/s), 4-core Xe-LPG iGPU, NPU 3.
ᵇ AMD Ryzen 9 **5950X** — DDR4 (~50 GB/s). Unsupported-but-measured; see below.
ᶜ Core Ultra 7 **258V** laptop — Arc 140V iGPU on LPDDR5X-8533 (~136 GB/s).
† 285K CPU at *whole-novel context* (~6 tok/s) — KV reads eat bandwidth at that
scale; no clean short-context CPU number for the MoE yet.
‡ Pronounced, not measured: DDR5 already sits at ~6 tok/s (†), and DDR4 has
roughly half the bandwidth — big-MoE-on-DDR4-CPU is below usable at real
context. Not worth the 17 GB download to confirm.

Reading it: the memory column, not the device column, predicts most of the
table (the laptop iGPU beats the desktop's because its *memory* is faster).
Protocols vary slightly across rows — the [benchmark sections](#benchmark-core-ultra-7-258v-arc-140v-16-gb--laptop-lpddr5x)
have the methodology; treat cells as ±10%. For scale: an RTX 5090 does 197
tok/s on the same 8B via Ollama — the [desktop benchmark](#benchmark-core-ultra-9-285k-rtx-5090--desktop-ddr5)
explains why NoLlama doesn't compete there.

Every *wanted* cell is an open invitation: run it, report it in
[#24](https://github.com/aweussom/NoLlama/issues/24), and your number lands
here with credit. **Arc A-series owners especially.**

## Quick start

**No git?** Download the latest release ZIP from
[Releases](https://github.com/aweussom/NoLlama/releases/latest), unzip it, then:

- **Windows** — double-click **`install-windows.bat`**. It checks for
  PowerShell 7 and Python 3.10+, offers to install whatever is missing
  (via winget), then runs the real installer.
- **Linux** — run **`./install-linux.sh`**. Same checks; it prints the
  exact install command for your distro if something is missing.

Already have PowerShell 7 and Python 3.10+ (or cloned the repo):

```powershell
.\install.ps1
.\start.ps1
```

That's it. `install.ps1` detects your hardware, lets you pick a model,
downloads it, and generates `start.ps1`. The launcher waits for the
model to load (with a progress indicator), then opens the built-in
chat UI in your browser at http://localhost:8000.

### Getting PowerShell 7

NoLlama's installer needs PowerShell 7+ — the 5.1 that Windows ships is
too old, and Linux doesn't ship it at all. The shims above handle this
for you; to do it yourself:

```powershell
# Windows 10/11
winget install --id Microsoft.PowerShell --source winget
```

```bash
# Ubuntu
sudo snap install powershell --classic
```

Other distros: [Microsoft's install docs](https://learn.microsoft.com/powershell/scripting/install/installing-powershell-on-linux).

## Recommended models

New here, or re-running `install.ps1`? Pick a **use-case** in the menu — here are
the proven models per role on a Core Ultra laptop (NPU + ARC iGPU):

| Use-case | Role | Pick in the menu | HuggingFace | Size |
|---|---|---|---|---|
| Chat | **NPU chat** | Qwen3 8B (INT4-CW) | `OpenVINO/Qwen3-8B-int4-cw-ov` | ~5 GB |
| Vision | **GPU vision** | Qwen3-VL 8B (INT8) | `OpenVINO/Qwen3-VL-8B-Instruct-int8-ov` | ~9 GB |
| Coding agent | **GPU/CPU coder** | Qwen2.5-Coder 7B (INT4) | `OpenVINO/Qwen2.5-Coder-7B-Instruct-int4-ov` | ~5 GB |

Qwen3 8B is the best-quality text model verified on the NPU. Qwen3-VL 8B
is the matching vision model — the INT8 build keeps fine detail (OCR,
small numbers) and fits a 16 GB ARC; drop to the ~6 GB INT4 build
(`…-int4-ov`) if you're tight on VRAM. For **coding agents** (VS Code Copilot
Chat, OpenClaw), pick the "Coding agent" use-case and a **Qwen2.5-Coder** model —
7B for snappy turns, 14B for stronger multi-step work; it runs on the GPU, or on
the CPU (which beats a weak iGPU on strong desktops). All are pre-exported — **no
conversion step**, though the multi-GB download still takes a while — and returning
users see them flagged **"Already on disk"** (those link instantly).

## Models we publish

Where a model the ecosystem needs doesn't exist in OpenVINO form, we build,
verify and [publish it on HuggingFace](https://huggingface.co/aweussom) —
several of these are the only OpenVINO builds of their model in existence.
All are in the `install.ps1` menu, with measured numbers on real hardware:

| Model | NPU (285K) | Notes |
|---|---|---|
| [`SmolLM3-3B-int4-cw-ov`](https://huggingface.co/aweussom/SmolLM3-3B-int4-cw-ov) | 23.3 tok/s | New in OpenVINO 2026.3; also runs GPU (29.7) / CPU (37.5) |
| [`SmolLM3-3B-int8-cw-ov`](https://huggingface.co/aweussom/SmolLM3-3B-int8-cw-ov) | 12.3 tok/s | Quality-first variant; ~half the speed of int4-cw |
| [`LFM2.5-1.2B-Instruct-int4-cw-ov`](https://huggingface.co/aweussom/LFM2.5-1.2B-Instruct-int4-cw-ov) | **38.8 tok/s** | Fastest model we've verified on an NPU. NPU-only build |
| [`LFM2-1.2B-int4-cw-ov`](https://huggingface.co/aweussom/LFM2-1.2B-int4-cw-ov) | 36.5 tok/s | NPU-only build |
| [`Qwen2.5-VL-3B-Instruct-int8-ov`](https://huggingface.co/aweussom/Qwen2.5-VL-3B-Instruct-int8-ov) | — (GPU VLM) | The proven small vision model, now a download instead of a 10-min conversion. Research license |
| [`LFM2-8B-A1B-int4-ov`](https://huggingface.co/aweussom/LFM2-8B-A1B-int4-ov) | — (GPU MoE) | 87 tok/s resident on an Arc 140V; the disk-offload test model |

The NPU builds are **channel-wise** exports (`-cw`) on purpose: the default
group-quantized int4 that `optimum-cli` produces crashes the Intel NPU
driver compiler (a known vpux bug — `"Found N duplicated names"`). If you
convert your own models for the NPU, use `download-model.ps1 -Weight
int4-cw` (or `int8-cw`), which encodes the working recipe.

## Big MoE models on small GPUs (disk offload)

OpenVINO 2026.3 can stream Mixture-of-Experts weights from disk instead of
keeping them GPU-resident. NoLlama exposes it as `--offload-ratio PCT`
(GPU slots). Measured on an Arc 140V (16 GB) laptop, Qwen3-30B-A3B INT4 —
a 15.2 GB model that doesn't fit resident at all:

| `--offload-ratio` | Resident GPU memory | Steady-state decode |
|---|---|---|
| 30 | 10.8 GB | **25.3 tok/s** |
| 50 | 8.1 GB | 22.1 tok/s |
| 90 | **2.35 GB** | 5.1 tok/s |

(Steady-state, measured after the expert LRU warms up — the first ~60
tokens run 2-5× slower while the cache fills, so don't judge offload by
its first sentence. `scripts/offload-test.py` measures this properly.)

Pick the **smallest ratio that fits** your memory. At moderate ratios this
is genuinely interactive: 25 tok/s from a 15.2 GB model on a 16 GB-class
laptop iGPU matches a 24-core desktop CPU running the same model resident.
High ratios (90) trade speed for extreme footprint — batch/overnight
territory. **Requires an XMX-capable GPU** (Arc, Lunar Lake and newer —
`install.ps1` tells you at device detection); on iGPUs without XMX the
feature silently does nothing, and NoLlama warns at startup instead of
letting you believe your model got smaller.

### Where does your hardware land? (big-MoE routes, measured 2026-08)

Same model family (Qwen3 MoE, A3B-class), steady-state decode, best route
per hardware class — including a CUDA flagship for perspective. Mixed
quants and sizes, so read it as *routes*, not a controlled A/B:

| Hardware | Stack & route | Model | tok/s |
|---|---|---|---|
| RTX 5090 32 GB + CPU (hybrid auto-split) | Ollama/CUDA | Coder-Next Q4, 53 GB | **~73** |
| Arc 140V laptop iGPU, `--offload-ratio 30` | NoLlama/OpenVINO | 30B-A3B int4, 15 GB | 25.3 |
| 24-core desktop CPU (64 GB RAM), model fits | NoLlama/OpenVINO | 30B-A3B int4 | 23.7 |
| 24-core desktop CPU, model **bigger than RAM** | NoLlama/OpenVINO | Coder-Next int8, **74 GB** | 9-11.5 |
| 8-core laptop CPU (LPDDR5X) | NoLlama/OpenVINO | 30B-A3B int4 | 9.1 |
| Non-XMX desktop iGPU | — | any big MoE | won't load |

Takeaways: a dedicated CUDA card is still ~3× the best Intel route — but
every Intel row above is *usable*, runs on hardware you may already own,
and two of them (offload, bigger-than-RAM CPU) were impossible before
OpenVINO 2026.3 and the MoE era. Decode is the whole story here; on
thinking models multiply by your patience.

## Brand-new architectures: the optimum backend

Some architectures land in optimum-intel (export) before openvino_genai
(serving) learns to run them — as of 2026-08 that's **Meta Muse Glimmer**
(`muse_glimmer`, [our int4 export](https://huggingface.co/aweussom/Muse-Glimmer-30B-int4-ov))
and **NVIDIA Nemotron 3.5 Lightning** (`nemotron_h`). NoLlama serves these
through optimum-intel's python runtime instead: detection is automatic
(`--scan` shows a `Backend` line; `--backend` overrides), tool calling
works, and both API surfaces behave identically. Differences from GenAI
slots: **text-only for now** (images get a clean 400), no prefix cache /
prewarm (a GenAI feature), no `--offload-ratio`, and no NPU. GPU support
also depends on the OpenVINO GPU plugin executing the model's
dynamic-shape graph, and as of OpenVINO 2026.3 **no Intel iGPU family runs
Glimmer correctly** — use `--device CPU`:

- **Xe-LPG** (desktop Arrow Lake iGPU): fails loudly at warmup
  (`Count is called for dynamic shape`).
- **Xe2** (Arc 140V, Windows, verified 2026-08-13): loads and warms up
  fine, then **silently computes garbage** — the model half-perceives the
  prompt (drops words, hallucinates a system prompt that was never sent)
  and greedy decoding degenerates into a two-word loop inside the think
  channel. The same IR with the same sampling params comprehends and
  complies perfectly on CPU. There is no error to catch: the only symptom
  is a model that seems drunk.
- **Xe3** (Arc B390 iGPU in Core Ultra X7 358H, **Linux**, community
  report in issue #24, 2026-08-13): identical corruption fingerprint —
  same "the user message is garbled" half-perception, same think-loop
  hang under greedy. Three iGPU generations and two OSes rule out any
  Windows-driver or Xe2-specific theory; tracked upstream as
  [openvinotoolkit/openvino#37419](https://github.com/openvinotoolkit/openvino/issues/37419).
  Discrete Battlemage (dedicated VRAM, different memory path) is the one
  untested configuration — the comprehension test below is its go/no-go.

The catch is the python stack: these models need transformers **from git
main** plus optimum-intel **from git main**, which no NoLlama venv pins.
`install-optimum.ps1` (Windows and Linux, needs git on PATH) builds a
dedicated `venv-optimum/` with the right stack in the right order — the
order matters: optimum-intel pins `transformers<5.6`, so the git
transformers goes in last to override it:

```powershell
.\install-optimum.ps1
venv-optimum\Scripts\python.exe nollama.py --model-dir ~\models\Muse-Glimmer-30B-int4-ov --device CPU --idle-timeout 0
```

(`--device CPU` is deliberate — see the iGPU verdicts above. If upstream
main breaks, pin with `-TransformersRef <commit>` / `-OptimumIntelRef
<commit>`.)

Running a plain install against such a model exits immediately with an
error naming this section instead of failing minutes into the load. When
openvino_genai gains these architectures, `--backend genai` (or just
re-exporting) moves them onto the faster path with prefix caching.

Measured (Muse Glimmer 30B int4, short chat prompts, 2026-08-13):
Core Ultra 7 258V laptop CPU 1.4 tok/s / TTFT 12.9 s; Core Ultra 9 285K
desktop CPU 2.6 tok/s / TTFT 9.6 s. Dense-30B bandwidth physics — fine
for verification, not agent loops; a 24 GB Arc-class card is the real
host. Note Glimmer *always* reasons by default (`reasoning_strength`
defaults to high in its template); the web UI's no-think toggle sends its
native `Reasoning strength: low.` directive.

## What it does

- **OpenAI API** (`/v1/chat/completions`) — works with any OpenAI client, OpenWebUI, etc.
- **Ollama API** (`/api/chat`, `/api/generate`) — works with Ollama clients, OpenWebUI Ollama mode, etc.
- **Auto-detects** NPU, ARC iGPU, ARC discrete, CPU — picks the best available
- **VLM support** — send images via base64 or `file://` URIs for vision models
- **Streaming** — token-by-token for text chat, with collapsible thinking blocks
- **Dual device** — NPU for chat + GPU for vision, simultaneously
- **Tool calling / agents** (GPU/iGPU + CPU, not NPU) — works with VS Code Copilot Chat and OpenClaw; the model drives tools on the ARC GPU or a strong CPU
- **Prefix caching** (on by default) — a repeated prompt prefix (e.g. an agent's fixed system prompt) is prefilled once, not every turn — ~47× faster on cached turns
- **MoE disk offload** (`--offload-ratio`) — run 30B-class MoE models on 16 GB-class XMX GPUs by streaming expert weights from disk (verified: 2.35 GB resident for a 15.2 GB model)
- **Built-in web UI** — chat, image drop zone, model selector, dark theme
- **Model menu** — curated list of verified models, no conversion nightmares

## Web UI

The server includes a built-in chat interface at http://localhost:8000.
No separate install, no Docker, no Node.js.

![NoLlama chat UI](docs/images/nollama-chat.gif)

A native Windows GUI is planned to replace the browser-based UI.

Features:
- Streaming chat with tokens appearing in real-time
- Collapsible "Thinking..." blocks (Qwen3 reasoning models)
- Drag-and-drop / paste images for VLM queries
- Model selector showing loaded models and their devices
- Device badge on each response (`[NPU 1.2s]`, `[GPU 2.8s]`)
- Dark theme
- Keyboard shortcuts: Enter to send, Shift+Enter for newline,
  Ctrl+V to paste images, Ctrl+N for new chat, Escape to cancel

## Device support

| Device | Examples | What it does | Streaming? |
|---|---|---|---|
| NPU (Intel AI Boost) | Core Ultra 7 258V | Text chat via LLMPipeline. Low power, sustained workload sweet spot. | Yes |
| ARC iGPU | ARC 140V (Core Ultra) | Vision + text, or bigger LLM | Yes (VLM streams in 2026.1+) |
| ARC discrete | A770, B580 | Same as iGPU, more VRAM for larger models | Yes (VLM streams in 2026.1+) |
| CPU | Any x86-64 with AVX2 — Intel supported, **AMD works too** (measured: Ryzen 9 5950X on DDR4, 23 tok/s on a 3B) | Fallback for everything. On desktops with DDR5 and many cores, often *faster* than NPU — see benchmarks. Decode is memory-bandwidth-bound: DDR4 boxes should size models accordingly. | Yes |

### Intel only — by design

NoLlama is Intel-hardware-only and will stay that way. Non-Intel GPUs
(NVIDIA, AMD) are filtered out of device detection on purpose, even
though OpenVINO 2026 now ships an experimental NVIDIA plugin via
[`openvino-extensibility`](https://docs.openvino.ai/2026/documentation/openvino-extensibility/openvino-plugin-library/plugin.html).
That path drags CUDA/cuDNN into the stack — it's a developer-backend
extension, not a drop-in user feature, and it loses every reason
NoLlama exists in the first place (NPU-first, Intel-first, no CUDA).

If you have an NVIDIA GPU, **use Ollama**. Ollama will always do
Ollama better than NoLlama could, and that's the right tool for that
hardware. NoLlama's value is specifically the Intel NPU / ARC story
that Ollama doesn't tell.

One honest footnote: the **CPU path happens to run on any AVX2 x86**,
because OpenVINO's CPU plugin doesn't actually care whose name is on
the silicon. Measured on an AMD Ryzen 9 5950X with DDR4 (fresh
Windows 11, installed from the release ZIP): 23 tok/s, TTFT 360 ms, on
a small thinking model (SmolLM3-3B int4). Not supported, not tuned
for, works fine — but read the number honestly: decode is
memory-bandwidth-bound, DDR4 dual-channel is roughly half a DDR5
desktop's bandwidth, so a 3B at 23 tok/s is the expectation-setter,
not a promise about 30Bs. Non-Intel GPUs stay filtered out, but nobody
will stop your CPU.

### Benchmark (Core Ultra 7 258V, ARC 140V 16 GB) — laptop, LPDDR5X

Tested with `benchmark.py` — 1 warmup + 5 runs, outliers discarded.

```powershell
# Text-only (no images required)
python benchmark.py --llm-only

# With VLM tests — provide 4 images: two "same vehicle" + two "different"
python benchmark.py --images-dir C:\path\to\images
python benchmark.py --same-1 a.jpg --same-2 b.jpg --diff-1 c.jpg --diff-2 d.jpg
```

**LLM text (Qwen3 8B INT4-CW, same model on NPU and CPU):**

| Test | NPU | CPU |
|---|---|---|
| "Say hello" (thinking) | 11.7s, 5.2 tok/s | 8.1s, 7.4 tok/s |
| "Say hello" (no-think) | 10.6s, 4.6 tok/s | 8.6s, 7.3 tok/s |
| "What is 2+2?" (thinking) | 11.7s, 5.3 tok/s | 9.0s, 7.0 tok/s |
| "What is 2+2?" (no-think) | 5.5s, 0.7 tok/s | 2.7s, 1.5 tok/s |

**GPU (Qwen2.5-VL 3B on ARC 140V, non-streaming):**

| Test | Time |
|---|---|
| "Say hello" (thinking) | 2.6s |
| "Say hello" (no-think) | 2.6s |
| "What is 2+2?" (thinking) | 2.6s |
| "What is 2+2?" (no-think) | 2.4s |
| Same vehicle? (2 images) | 3.8s |
| Different vehicles? (2 images) | 3.8s |

Above benchmarks were captured before VLMPipeline gained streaming
support (openvino-genai 2026.1). VLM now streams on Arc 140V at
roughly 11 tok/s decode after prefill — see
`benchmark.py --backend vlm` for fresh numbers.

CPU beats NPU on throughput (~7.4 vs ~5.2 tok/s) for this model.
GPU text is fast but runs a smaller 3B model (not directly comparable).
VLM image responses take ~3-4s regardless of answer length.

### NoLlama vs Ollama on the Arc 140V iGPU

Ollama now runs on Intel iGPUs via its Vulkan backend, so this is the
direct apples-to-apples question: **same Qwen3-8B, same 4-bit, same
Arc 140V iGPU.** Measured 2026-06-16 with `benchmark.py` (3 runs), using
the `count 1-100` test as the steady-state decode metric.

| | NoLlama (OpenVINO INT4-CW) | Ollama 0.30.8 (Vulkan GGUF Q4) |
|---|---|---|
| **Decode tok/s** (count 1-100) | **21.7** | 13.4 |
| Decode tok/s (2+2, thinking) | 18.6 | 11.2 |
| TTFT (prefill) | 3.2s | **1.85s** |

**NoLlama's OpenVINO GPU path is ~1.6× faster on decode**; Ollama wins
time-to-first-token. Two caveats that matter in practice:

- **Ollama drops the iGPU by default** — it needs `OLLAMA_IGPU_ENABLE=1`,
  or it silently runs on CPU. The out-of-the-box Ollama experience on
  this laptop is *CPU*, not GPU.
- Ollama can't use the **NPU** at all, and has no local **vision** model
  on Intel — both are NoLlama-only.

> **Roadmap note — GPU/CPU support is here to stay** *(updated 2026-08:
> this reverses the earlier "provisional" stance)*. NoLlama's original
> reason to exist is the Intel **NPU** (which Ollama doesn't support), and
> the plan was to drop GPU/CPU once Ollama's Intel performance caught up.
> That hasn't happened and isn't on the horizon: Ollama's Intel path runs
> through a non-OpenVINO shim and remains much slower, while most real
> NoLlama users drive coding agents (OpenClaw, Copilot) on the GPU/CPU
> path. So GPU/CPU — and with them tool calling, prefix caching, and
> prewarm — are supported for the foreseeable future. If you outgrow a
> single-user local server (multi-user, production serving of 30B+
> models), the step up is [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server)
> — same runtime underneath, built for that job.

### Benchmark (Core Ultra 9 285K, RTX 5090) — desktop, DDR5

Same Qwen3 8B INT4-CW model on every Intel device, plus the same model
served via Ollama (GGUF Q4_K_M) on the RTX 5090 for context. 1 warmup +
3 runs. The "count 1-100" test (`max_tokens=4096`, no-think) is the
cleanest cross-stack number — long output, steady-state, no thinking confound.

```powershell
# Each NoLlama device — restart the server with --device <name> first
python benchmark.py --label npu --runs 3 --llm-only
python benchmark.py --label igpu --runs 3 --llm-only
python benchmark.py --label cpu --runs 3 --llm-only

# Ollama (any backend it's running on — CUDA, ROCm, CPU)
python benchmark.py --backend ollama --model qwen3:8b --label rtx5090 --runs 3 --llm-only
```

**Decode throughput, count-1-100 test:**

| Backend | Device | TTFT | Decode tok/s | Speed vs CPU |
|---|---|---|---|---|
| Ollama (GGUF/CUDA) | RTX 5090 | 0.19s | 197 | 11.1× |
| NoLlama (OpenVINO) | CPU (8P + 16E @ DDR5) | 3.84s | 17.8 | 1.0× |
| NoLlama (OpenVINO) | iGPU (Xe-LPG, 4 cores) | 4.01s | 15.4 | 0.87× |
| NoLlama (OpenVINO) | NPU 3 (Intel AI Boost) | 10.6s | 10.0 | 0.56× |

**Surprises on this hardware:**

- **CPU beats iGPU.** Arrow Lake's 285K (8P + 16E at high clocks) plus
  OpenVINO's tuned INT4 CPU kernels add up to more decode throughput
  than the small Xe-LPG iGPU (only 4 Xe cores on the desktop part —
  the laptop's ARC 140V has 8). Both share the same DDR5 pool, so the
  iGPU has no bandwidth advantage, only a compute disadvantage.
- **NPU is the slowest Intel device on desktop**, opposite of the laptop
  story. NPU's value is power efficiency (laptop on battery), not
  throughput on mains.
- **Prefill scales differently than decode.** RTX 5090's TTFT advantage
  over NPU is ~55× (0.19s vs 10.6s); its decode advantage is ~20×.
  Long prompts amplify the gap.
- **The dGPU dominates** — if you have one, use it. NoLlama's CPU
  fallback is good for "Intel-only laptop on battery", not for
  competing with a discrete card.

**Why the desktop iGPU/NPU are slower than the laptop's:**
LPDDR5X-8533 (laptop, ~136 GB/s) vs DDR5-6400 dual-channel (desktop,
~100 GB/s). Decode throughput on INT4 LLMs is memory-bandwidth-bound,
so the laptop's faster system memory closes some of the gap that
silicon size alone would suggest. (The Core Ultra 7 258V Lunar Lake
NPU also has more compute units than the 285K Arrow Lake NPU.)

**Practical guidance:**

| Hardware | Best NoLlama device |
|---|---|
| Intel Core Ultra laptop (Lunar Lake) | NPU (efficiency) or ARC 140V iGPU |
| Intel Arrow Lake desktop, no dGPU | **CPU** — surprisingly best |
| Intel + ARC discrete (A770, B580) | ARC discrete |
| Intel + NVIDIA discrete | Use Ollama for the dGPU; NoLlama on CPU/NPU/iGPU as fallback |

### Dual mode (NPU + GPU)

When you have both, text requests go to the NPU (streaming) and image
requests go to the GPU (VLM). Or put a bigger LLM on the GPU for
smarter chat. The routing is automatic — send a request and the right
device handles it.

```
POST /v1/chat/completions
  "What is the capital of Norway?"  --> NPU (streaming)
  [image + "What vehicle is this?"] --> GPU (VLM)
```

## Why not OpenVINO Model Server (OVMS)?

Intel already ships OVMS — a production-grade OpenVINO inference server.
If you're deploying LLMs in a datacenter or on Kubernetes, use OVMS.
NoLlama is a different target: your laptop.

| | OVMS | NoLlama |
|---|---|---|
| Target | Production, datacenter, K8s | Laptop, desktop, local |
| Runtime | C++ | Python (Flask) |
| OpenAI API | Yes (recent versions) | Yes |
| Ollama API | No | **Yes** |
| Built-in web UI | No (add OpenWebUI) | **Yes** |
| Auto device detection | No | **Yes** |
| Dual-device routing | One model per instance | **NPU chat + GPU vision, simultaneously** |
| Config | JSON, manual | Zero — `install.ps1` and go |

OVMS is a proper inference server. NoLlama is the thing that makes
your Core Ultra feel like Ollama already ran on it.

### ...and why not llm-scaler-vllm?

Same answer, different Intel stack. [`intel/llm-scaler`](https://github.com/intel/llm-scaler)
(vLLM + IPEX, the Battlematrix software) is Intel's official serving
path for **Arc Pro B-series** cards — and if you're building a
dedicated Linux inference box around them, use it: multi-card
tensor-parallel serving is its home game. It's also Ubuntu-with-a-
specific-kernel, Docker, and Linux-only for the vLLM path.

The axis that actually decides is streams × precision. LLM decode is
memory-bandwidth-bound, and 4-bit weights move roughly a quarter of
the bytes per token — INT4 IR is openvino-genai's native format, so
**single-user quantized decode on Intel silicon is NoLlama's tier**:
one to a few streams, the machine you sit at. Moderate shared
concurrency is OVMS's tier (continuous batching, same INT4 IR).
Multi-GPU tensor-parallel on Linux is llm-scaler's. Different jobs,
all three real.

## Usage

```powershell
# Auto-detect (picks best device)
python nollama.py

# Force a specific device
python nollama.py --device NPU
python nollama.py --device GPU
python nollama.py --device CPU

# Dual mode: NPU chat + GPU vision
python nollama.py --model-dir model --gpu-model-dir gpu-model

# Different port
python nollama.py --port 9000

# Change the default idle-unload timeout (default is 1800 = 30 min)
python nollama.py --idle-timeout 600     # unload after 10 min idle
python nollama.py --idle-timeout 0       # never unload — keep models loaded forever

# Log every inbound API request (method, path, User-Agent, body) — handy when
# wiring up a new agent client and you need to see exactly what it sends
python nollama.py --debug

# Report a real Ollama version on /api/version so VS Code's Ollama client
# accepts the server (needed for VS Code Copilot Chat in Ollama mode)
python nollama.py --vscode-compat

# Prefix (KV) caching is ON by default for GPU/CPU LLM slots — a repeated prompt
# prefix is prefilled once, not every turn (big win for agent loops, ~47x on a
# cached turn). The pool is auto-sized per device from its memory budget and
# the model's KV geometry (a third of what the weights leave free, floor 2 GB,
# cap ~64k tokens' worth) — the startup log shows the chosen size and its
# token capacity. Pin the size, or disable caching:
python nollama.py --cache-size-gb 4     # pin the KV-cache pool (skips auto-sizing)
python nollama.py --no-prompt-cache     # disable prefix caching

# Pre-warm the cache at startup so the FIRST agent turn is fast too (not just
# turn 2+). The file auto-populates from the first big prompt served, so the
# workflow is: run once, then restart with --prewarm to skip the cold prefill.
# With --idle-timeout 0 this is automatic (as prewarm-<port>.json; opt out
# with --no-prewarm) — combining --prewarm with idle unload gets a warning,
# because the warmed cache is thrown away when the model idle-unloads.
python nollama.py --prewarm prewarm.json

# What models do I actually have? Reports each model directory's real
# contents — no server started, no model loaded.
python nollama.py --scan
python nollama.py --scan D:\models      # search somewhere else
```

### What have I got? (`--scan`)

`--scan` answers that from the files on disk rather than from what a folder
is called:

```
  C:\Users\you\models\Qwen3-Coder-Next-int8-ov
    Name in API/UI : Qwen3-Coder-Next      (from directory name)
    Kind           : LLM (text)
    Architecture   : Qwen3NextForCausalLM / qwen3_next
    Weights        : INT8 (asymmetric, channel-wise)   80.1 GB on disk
    MoE            : 512 experts, 10 active per token
    Geometry       : 48 layers, 262,144-token context, 32 KB/token KV
    Exported with  : OpenVINO 2026.1.0, optimum-intel 1.27.0, transformers 4.57.6
    Agent mode     : tool calling on GPU/CPU; never on NPU (hard prompt cap)
    Integrity      : weights complete
```

The precision comes from the IR's own `nncf` record, not the directory name
— a folder called `-int4-ov` can contain anything, and `--scan` reports what
the weights actually are (including partial quantization and AWQ). It also
runs the truncation check, so it's the quickest way to tell a bad download
from a bad model.

**To rename a model,** rename its directory: that name is what the web UI
shows and what clients request as the model ID. There's deliberately no
`--model-name` flag — see `TODONT.md`.

**The KV pool sizes itself.** The pool must hold the whole conversation:
bytes-per-token scale with the model's layer/head geometry (~56 KB/token
for a 7B coder, ~96 KB/token for Qwen3-Coder-30B). Too small doesn't just
evict cache — generation on a big agent prompt **fails outright**
(`Got unfinished GenerationStatus`, see issue #21). So NoLlama sizes the
pool per device at load: a third of what the weights leave free in the
device's memory budget, floored at 2 GB and capped at ~64k tokens of the
model's geometry (enough for a big agent system prompt plus a long
session). It's a ceiling the cache grows into, not an upfront allocation
— and the fraction leaves RAM for the compilers and tests an agent runs
on the same machine. The startup log shows the chosen size and its token
capacity; `--cache-size-gb N` pins it when you know better (e.g. whole-book
contexts). The preflight still warns when agent prompts would exhaust the
pool, and when model + pool exceed the device budget entirely. On Core
Ultra iGPUs that budget is ~half of system RAM by default — raise it with
Intel Graphics Software's "Shared GPU Memory Override" (driver 101.6987+).
Per-request log lines include TTFT, so a prefix-cache hit (sub-second) vs
a cold prefill (seconds-to-minutes) is visible directly; `/health` reports
the cache config (`prompt_cache_info.auto`, per-slot `kv_pool_gb`) and
each slot's last TTFT.

### Idle unload

NoLlama frees model memory after **30 minutes of inactivity by default**
(an 8B INT4 model holds ~5 GB of RAM; a VLM another ~3 GB). The next
request automatically reloads the model — the client just sees a slow
first response (~30-60s for an 8B model on NPU). The web UI shows
"Reloading model..." while it waits.

Change with `--idle-timeout <seconds>`. Use `0` to keep models loaded
forever (the old behavior) — recommended for agent use: it also
auto-enables `--prewarm`, and the warmed prefix cache survives (an idle
unload discards it until the next restart, which is why mixing
`--prewarm` with idle unload prints a warning).

`/health` reports `idle_unloaded` slots; the overall status stays
`ready` because requests can still be served (with a reload).

## API

Standard OpenAI `/v1/chat/completions`. Works with any OpenAI client.

### Text chat

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

### Image (VLM, requires GPU with vision model)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":[
      {"type":"text","text":"What is in this image?"},
      {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}
    ]}]
  }'
```

### Local file shortcut

When client and server are on the same machine, skip base64:

```python
{"type": "image_url", "image_url": {"url": "file:///C:/path/to/image.jpg"}}
```

**Note:** `file://` URIs only work locally. Remote clients must use base64.

### Streaming

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Tell me a story"}],"stream":true}'
```

### Other endpoints

- `GET /health` — device status, model names, readiness
- `GET /v1/models` — list loaded models (OpenAI format)

### Response headers

Every response includes `X-Device` and `X-Model` headers so you can
see which device handled it:

```
X-Device: NPU
X-Model: qwen3-8b
```

## Using with the openai Python package

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="qwen3-8b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

## Ollama API

NoLlama also serves a full Ollama-compatible API on port 11434 (the
Ollama default). Any tool or client that talks to Ollama works without
modification — it thinks it's talking to a real Ollama instance.

Supported endpoints:

- `POST /api/chat` — chat with streaming (newline-delimited JSON)
- `POST /api/generate` — single-turn completion
- `GET /api/tags` — list models
- `POST /api/show` — model info

```bash
curl http://localhost:11434/api/chat \
  -d '{"model":"qwen3-8b-int4-cw","messages":[{"role":"user","content":"Hello!"}]}'
```

Disable with `--ollama-port 0` if you don't need it or port 11434 is taken.

## Using with OpenWebUI

OpenWebUI can connect via either API:

**OpenAI mode** (recommended):

| Field | Value |
|---|---|
| Base URL | `http://host.docker.internal:8000/v1` |
| API Key | `not-needed` |

**Ollama mode** (no config needed if NoLlama runs on default port):

| Field | Value |
|---|---|
| Ollama Base URL | `http://host.docker.internal:11434` |

## Agent tools & coding assistants (VS Code Copilot, OpenClaw)

NoLlama can drive tool-calling coding agents — the model emits function calls,
NoLlama parses them into OpenAI/Ollama `tool_calls`, and the agent acts on the
results.

![OpenClaw running locally against NoLlama on an Intel iGPU — start-openclaw.ps1 brings up NoLlama (GPU, Qwen2.5-Coder-7B), pre-warms the cache, and the agent replies](screenshots/openclaw-1-Skjermbilde2026-06-28_113203.png)
*OpenClaw driving a local Qwen2.5-Coder model on an Intel iGPU via NoLlama — one command (`./start-openclaw.ps1`), no cloud, no NVIDIA.*

> **Tool calling runs on GPU/iGPU and CPU — not the NPU.** The NPU has a hard
> prompt cap and small NPU-class models can't reliably drive agent loops, so
> NoLlama ignores `tools` there and answers as plain chat; `/api/show` advertises
> the `tools` capability only for GPU/CPU slots. Load a coder LLM on the GPU, or
> on a strong desktop CPU (many-core Core Ultra) where prefill can beat a weak
> iGPU. The Qwen2.5-Coder GPU builds in the menu work well; pick a smaller size
> (7B) for snappier prefill on big agent prompts.
>
> Tool turns are **buffered** (the whole reply is generated before the structured
> `tool_calls` are sent), but the server emits SSE keep-alive pings during a long
> prefill so agent clients (Copilot/OpenClaw) don't hit their idle timeout and
> abort. Big agent system prompts (~20k tokens) prefill slowly on weak iGPUs — a
> smaller model, the CPU, or trimming the client's tool set all help. And
> **prefix caching is on by default**, so that big system prompt is prefilled
> once, not every turn — after the first turn, agent turns are fast (~47x on the
> cached prefix). Disable with `--no-prompt-cache`.

The tool prompt is rendered in Qwen3-Coder native format, and `parse_tool_calls`
also understands Hermes, Mistral `[TOOL_CALLS]`, Llama `<|python_tag|>`, DeepSeek,
and bare-JSON outputs — so most instruct/coder models work.

**VS Code Copilot Chat** (0.53+) — point it at the Ollama API and start the
server with `--vscode-compat` so VS Code accepts the version handshake:

```powershell
python nollama.py --gpu-model-dir gpu-coder-model --vscode-compat
```

Then in VS Code set the Ollama base URL to `http://localhost:11434` and pick the
GPU model. (Add `--debug` while wiring it up to see exactly what Copilot sends.)

**OpenClaw** — speaks the OpenAI chat-completions API NoLlama already serves; it
runs against a NoLlama GPU slot with no code changes, just config. See
[OPENCLAW-PLAN.md](OPENCLAW-PLAN.md) for the step-by-step setup (the one gotcha:
address the model as `<name>@GPU` so tool requests hit the GPU, not the NPU).

**Install OpenClaw** (once):

```powershell
npm install -g openclaw@latest
openclaw onboard --install-daemon
```

Then **`start-openclaw.ps1`** is the one-command launcher (the NoLlama equivalent
of `ollama launch openclaw`):

```powershell
./start-openclaw.ps1 -Setup -Device GPU     # -Setup writes the `nollama` provider into openclaw.json
./start-openclaw.ps1 -Device GPU            # subsequent runs
```

It starts NoLlama with the agent flags (`--device`, `--prewarm`, keep-loaded),
waits until ready, then runs OpenClaw. If a NoLlama is **already** on the port it
**verifies** it (prefix caching on + a tool-capable GPU/CPU slot) and reuses it —
or, if it's misconfigured, tells you why and offers to restart it correctly
(`-Force` to skip the prompt). `-Warmup` fires one throwaway turn first so even
the first real turn is fast.

> **NoLlama runs OpenClaw in a deliberately constrained mode — by design.** A
> coding-agent prompt is large (~21k tokens of system prompt + tool schemas), which
> is a lot for a small local model on weak Intel hardware. So `-Setup` doesn't just
> point OpenClaw at NoLlama — it also **trims OpenClaw** to fit: it selects the
> `coding` tool profile and turns off web search, X search, memory search, and the
> startup-context prelude. This shrinks the prompt and tool surface so a 7B coder on
> an iGPU/CPU can actually drive the loop. It's all plain config in
> `~/.openclaw/openclaw.json` — re-enable anything if your hardware can handle a
> bigger prompt, and re-run `-Setup` to restore the trimmed defaults. Package
> updates (`npm i -g openclaw@latest`) don't touch this config; only re-running
> `openclaw onboard` might, in which case re-run `-Setup`.

## Models

`install.ps1` shows a curated menu of models known to work on Intel
hardware. All pre-exported models are download-only (no conversion).
The menu is defined in `models.json` — add entries when new models
are verified.

### Gated or private models (HuggingFace token)

The curated `OpenVINO/…` models are public and download anonymously — no
token needed. You only need a [HuggingFace
token](https://huggingface.co/settings/tokens) (the `hf_…` string) for
**gated** models (ones that make you accept a license, e.g. Llama) or
**private** repos. Pass it with `-HfToken`:

```powershell
.\install.ps1 -HfToken hf_xxxxxxxxxxxxxxxxxxxxx
.\download-model.ps1 some-org/gated-model -HfToken hf_xxxxxxxxxxxxxxxxxxxxx
```

Note: `hf auth login` won't help on a first run — `install.ps1` is what
installs the `hf` CLI in the first place, so there's no `hf` to log in
with yet. `-HfToken` works on a clean machine because it sets `HF_TOKEN`
before the download (which `huggingface_hub` reads automatically). If you
already have an `hf auth login` token stored from elsewhere, that's used
too — `-HfToken` is just the bootstrap-proof way.

### Adding models outside the menu

Use `download-model.ps1` to grab any HuggingFace model:

```powershell
# Pre-exported OpenVINO model (just download)
.\download-model.ps1 OpenVINO/Qwen3-8B-int4-cw-ov

# Convert a HuggingFace model to OpenVINO (PowerShell flags: single dash)
.\download-model.ps1 Qwen/Qwen2.5-VL-3B-Instruct -Convert -Weight int8

# With trust-remote-code (some models require this)
.\download-model.ps1 Qwen/Qwen2.5-VL-3B-Instruct -Convert -Weight int4 -Trust
```

Models download to `~/models/<name>/`. Point NoLlama at them:

```powershell
python nollama.py --model-dir ~/models/my-model --device GPU
python nollama.py --gpu-model-dir ~/models/my-vlm
```

Model folders are sanity-checked both at install time and at server start:
the `openvino_model.bin` + `.xml` pair must be present, and the `.bin` must
be at least as large as the `.xml` says it should be (byte-exact — catches
interrupted downloads and half-synced copies). A broken folder is re-fetched
by the installer, or refused at load with an error that says exactly what's
missing — so if you assembled a model directory by hand and NoLlama rejects
it, trust the message, not the folder listing.

### Model won't load? Run the canary first

Before debugging anything else, establish whether the problem is **your
model** or **your stack**. The registry's smallest model is a ~1 GB
known-good canary — output quality is terrible, that's not the point;
it loads everywhere:

```powershell
.\download-model.ps1 OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov
python nollama.py --model-dir "~/models/DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov" --device NPU   # or GPU / CPU
```

- **Canary loads, your model doesn't** → the stack is healthy; the model
  is the problem. NPU limits are real: INT4-**CW** quantization, ≤ 8B
  params (~6 GB on disk). Bigger or group-quantized INT4 models die in
  the NPU compiler ("Compilation failed", `vpux-compiler` errors). Run
  that model on GPU/CPU instead, or pick an NPU model from the menu.
- **Canary fails too** → driver/stack problem, not the model. Windows:
  update the Intel NPU driver. Linux: the `intel-npu-driver` and
  `intel-npu-compiler` versions must match each other and OpenVINO —
  distro-repo packages often lag; use the
  [intel/linux-npu-driver releases](https://github.com/intel/linux-npu-driver/releases)
  compatibility table.

The install-menu models are the proven set; anything you bring via
`download-model.ps1 -Convert` is best-effort territory — the server's
startup log now says explicitly what's wrong (missing/truncated files,
memory that won't fit, KV pool too small, NPU compiler rejection)
instead of generic errors, so read it before opening an issue. 🙂

### Finding newer/better models

The model menus rot fast — new architectures appear monthly. The
authoritative place to look is the OpenVINO org on HuggingFace:

**[huggingface.co/OpenVINO](https://huggingface.co/OpenVINO)**

These are pre-exported by Intel, so there's **no conversion step** — just a
download (still slow for multi-GB models, but no 5-20 min `optimum-cli` export).
What to look for:

| Suffix | Where it runs | Notes |
|---|---|---|
| `-int4-cw-ov` | NPU + GPU | Channel-wise INT4. NPU's preferred format. |
| `-int4-ov` | GPU only | Standard INT4. Not always NPU-compatible. |
| `-int8-ov` | GPU + CPU | Better fine-detail retention than INT4 (OCR, numbers). |
| `-fp16-ov` | GPU + CPU | Full precision. Largest, slowest, sharpest. |

Quick rules of thumb:
- **NPU chat:** must be `-int4-cw-ov` and ≤ 8B params (~6 GB on disk) —
  a 14B INT4 fits the old size advice but fails in the NPU compiler (#20).
- **GPU vision (VLM):** any `-int4-ov` or `-int8-ov` model marked
  "Image-Text-to-Text" on HF.
- **GPU LLM (smarter than NPU):** any `-int4-ov` model up to your
  VRAM. Above ~16 GB falls back to CPU silently.
- **Whisper (STT):** OpenVINO ships pre-quantized whisper variants
  (`whisper-{tiny,base,small,medium,large-v3}-{int4,int8,fp16}-ov`).

Once a model proves itself, add it to `models.json` so it appears in
the install menu. Keep "Untested" tags on entries that haven't been
verified yet — be honest about what's measured vs. assumed.

> **Recommended VLM:** OpenVINO ships
> [Qwen3-VL-8B](https://huggingface.co/OpenVINO/Qwen3-VL-8B-Instruct-int8-ov)
> pre-exported in INT4/INT8/FP16 — the natural vision sibling to the
> proven Qwen3-8B NPU chat model. The INT8 build is verified here on the
> Arc 140V in dual mode (2026-06-16) and is the default GPU vision pick
> (see [Recommended models](#recommended-models)); INT4 is the lighter
> ~6 GB option.

### NPU models (chat)

| Model | Size | Notes |
|---|---|---|
| Qwen3 8B (INT4-CW) | ~5 GB | Recommended. Best quality. |
| Phi 3.5 Mini (INT4-CW) | ~2 GB | Smaller, faster. |
| DeepSeek R1 Distill 7B (INT4-CW) | ~4 GB | Reasoning. |
| DeepSeek R1 Distill 1.5B (INT4-CW) | ~1 GB | Testing only. |
| Mistral 7B v0.3 (INT4-CW) | ~4 GB | General purpose. |

### GPU vision models

| Model | Size | Notes |
|---|---|---|
| Qwen3-VL 8B (INT8) | ~9 GB | Recommended pairing for 16 GB ARC. Keeps fine detail (OCR, numbers). |
| Qwen3-VL 8B (INT4) | ~6 GB | Lighter alternative. Newer Qwen-VL generation; verified on Xe-LPG. |
| Qwen2.5-VL 3B (INT8, convert) | ~4 GB | Proven. INT8 better at fine detail (OCR, numbers). |
| Gemma 3 4B Vision (INT4) | ~3 GB | Untested. |
| Gemma 3 12B Vision (INT4) | ~7 GB | Untested. Needs ~12 GB RAM with KV cache. |
| InternVL2 4B (INT4) | ~3 GB | Untested. |
| Phi 3.5 Vision (INT4) | ~3 GB | Untested. |

### GPU large LLMs (smarter than NPU)

| Model | Size | Notes |
|---|---|---|
| Qwen3 14B (INT4) | ~8 GB | Great reasoning. |
| Qwen3 30B-A3B MoE (INT4) | ~17 GB | 30B brain, 3B speed. |
| Phi 4 (INT4) | ~8 GB | Strong reasoning. |
| Phi 4 Reasoning (INT4) | ~8 GB | Chain-of-thought. |

## How it works

The server auto-detects your model type (VLM or LLM) from
`config.json` and loads the right OpenVINO GenAI pipeline:

- **VLMPipeline** for vision models — handles images + text
- **LLMPipeline** for text models — handles chat with streaming

In dual mode, both pipelines run on separate devices with separate
locks. They don't interfere with each other.

> **Future simplification:** OpenVINO GenAI may unify VLMPipeline and
> LLMPipeline into a single pipeline that handles both text and images.
> When that lands, the dual-pipeline detection and routing logic in
> NoLlama can be collapsed into one code path.

## Files

```
nollama.py              The server
install-windows.bat     Windows entry point — double-click me (installs
                        PowerShell 7 + Python if missing, runs install.ps1)
install-linux.sh        Linux entry point — checks pwsh/python3, prints the
                        install command for your distro, runs install.ps1
install.ps1             Setup wizard (cross-platform; the shims above call it)
download-model.ps1      Download/convert any HuggingFace model
benchmark.py            Device performance benchmark
start.ps1               Auto-generated launcher (after install)
start-openclaw.ps1      Launch NoLlama (caching + pre-warm) + OpenClaw together
models.json             Curated model registry
model/                  Primary model (NPU or GPU)
gpu-model/              Secondary GPU model (dual mode)
venv/                   Python virtual environment
```

`model/`, `gpu-model/`, `venv/`, and `start.ps1` are gitignored.
The repo is pure code.

## Requirements

- PowerShell 7+ (Windows PowerShell 5.1 is not supported; on Linux,
  see [Microsoft's install
  instructions](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-linux))
- Python 3.10+
- OpenVINO 2026.1+ with openvino-genai
- At least one of:
  - Intel Core Ultra (NPU + ARC iGPU)
  - Intel ARC discrete GPU (A770, B580, etc.)
  - Any Intel CPU (slower, but works)
- ~1-17 GB disk per model

`install.ps1` handles the venv, dependencies, and model download. It *is*
the installer on every platform — `install-windows.bat` and
`install-linux.sh` are thin entry-point shims that check the
prerequisites above (offering to install what's missing) and then hand
off to it. There is deliberately no separate Bash installer: on
Linux/macOS the shim runs `pwsh ./install.ps1` for you (flags like
`-HfToken` pass through); paths and link creation branch on
`$IsWindows`. Windows is the primary platform, but
**Linux is confirmed working** by user reports (Core Ultra 7 258V, NPU +
GPU detected — see [#6](https://github.com/aweussom/NoLlama/issues/6));
macOS is untested. On Linux, NPU and GPU detection needs the Intel
userspace drivers installed (`intel-npu-driver` for the NPU, the GPU
compute runtime for the iGPU) — without them only the CPU shows up. The
NPU Linux stack is less battle-tested than Windows.

## Known limitations

These are known and intentionally not fixed — either because the cause
is upstream, the fix would hurt simplicity, or it doesn't matter for a
local single-user tool.

- **Cancel may not interrupt mid-generation.** The cancel endpoint
  signals OpenVINO's streamer callback to stop. If OpenVINO is blocked
  inside a native call and not invoking the callback, there's no way
  to interrupt it from Python. Generation completes; lock releases
  when it does.
- **NPU prompt limit is 4096 tokens.** Long chat histories will
  eventually exceed this. The UI doesn't trim history — use Ctrl+N to
  start fresh if you hit the limit.
- **Vision runs on the GPU, not the NPU — by design.** The NPU *can*
  load a VLM (Qwen2.5-VL-3B compiles and runs via VLMPipeline; Qwen3.5
  and MiniCPM-V don't compile at all), but the NPU caps the prompt at
  ~1024 tokens *including image tokens*, and Qwen2.5-VL spends one token
  per 28×28 px. That leaves a usable ceiling around **768×768 (~784
  image tokens)**: at that size — or smaller — it answers correctly, so
  NPU vision works **well-ish on very small images** (a 256–512px crop is
  fine). But prefill already takes ~17s at the ceiling, and a plain
  1024×768 photo overflows the cap and fails outright (720p/1080p never
  stand a chance). So vision stays on the GPU, which has no such cap,
  runs at full resolution, and is faster. Measured with
  `test_npu_vlm_imagesize.py`.
- **Ollama management endpoints are stubs.** `/api/pull`, `/api/delete`,
  `/api/copy` return success but don't do anything. Model management is
  via `install.ps1` or `download-model.ps1`, not the API.
- **No graceful shutdown.** Ctrl+C is abrupt. If you hit it mid-load,
  NPU/GPU resources may not free cleanly — usually resolves on next
  launch, occasionally needs a reboot.
- **Flask dev server, not production.** Single-user local tool. Don't
  put it on the internet without a reverse proxy.

## A note about small models

During initial NPU testing with DeepSeek R1 1.5B, we asked:
"What is the capital of Norway?"

The model's response:

> "I need to figure out the capital of Norway. I know it's a country
> in Norway. I remember that Norway is a small island..."

Norway is, in fact, not a small island.

Or *is* it? To paraphrase the greatest detective of all time, Ford
Fairlane: "...an island in an ocean of diarrhea."

The point: 1.5B parameter models are for testing the plumbing, not
for geography. Use Qwen3-8B or larger for actual chat. The small
models will catch up — they're getting smarter every month.

## License

MIT

## Author

Tommy Leonhardsen
