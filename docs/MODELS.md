# Models

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
