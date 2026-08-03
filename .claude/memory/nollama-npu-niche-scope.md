---
name: nollama-npu-niche-scope
description: "NoLlama's planned scope narrowing — if Ollama ships working Intel ARC GPU support, drop GPU+CPU and keep NPU as the defensible niche."
metadata: 
  node_type: memory
  type: project
  originSessionId: 2f00d17b-a999-4fcc-92e1-677fc1e6c89f
---

If mainline Ollama delivers solid Intel ARC GPU (and CPU) support on shared-memory iGPU, Tommy plans to **remove the GPU+CPU paths from NoLlama and keep NPU as the sole target** — "Ollama does Ollama better than I can." The NPU is the moat: llama.cpp/Ollama target Intel GPU (SYCL/Vulkan) and CPU but **not** the NPU, so nobody else serves it.

**Why:** Stop competing where a better-resourced project wins; keep only the part where NoLlama is the only option. Matches the existing "NPU-first" tagline and collapses a lot of `nollama.py` (dual VLM/LLM routing, GPU streaming, image→GPU/text→NPU dispatch, DeviceSlot juggling).

**How to apply:** Don't over-invest in GPU code paths for future NoLlama work. Two caveats raised: (1) NPU-only likely means **text-only** — vision (VLMPipeline) and GPU-Whisper run on GPU, so dropping GPU also drops those capabilities; decide deliberately. (2) Don't delete on a promise — verify the 140V actually runs the verified models under mainline Ollama at parity before removing working code. Related: [[npu_memory_path_isolation]].

**Update (2026-06-27) — benchmark in, drop-GPU plan REVERSED:** Tommy ran the benchmark this note called for. On his Intel iGPU, **NoLlama (OpenVINO GenAI) is ≥30% faster than mainline Ollama (Vulkan)**. So the condition for dropping GPU — "Ollama does GPU better than I can" — is *false*: it doesn't. The GPU path is now a measured moat, not just NPU. Working hypothesis: real NoLlama users are driving Intel **graphics cards** (least-painful way), not NPUs. **Revised scope:** keep GPU as a first-class target; the speed edge over Ollama-Vulkan is the selling point. This also revives the Copilot-Chat/tool-calling PR (#9) — viable on GPU (not NPU), positioning NoLlama as "the fastest Intel-GPU local Copilot backend." Pair with a Coder model (parser is Qwen3-Coder-native). See [[intel_llm_stack_landscape]].

**Update (2026-06-27) — why the NPU moat holds, and the real tripwire:** Vulkan is a **GPU compute API**; the Intel NPU is **not a Vulkan device** (driven by OpenVINO / Level Zero / `intel-npu-driver`). So Ollama's Vulkan path *structurally cannot* reach the NPU — it's a wrong-tool barrier, not a "catch up" race. Vulkan is exactly how Ollama closes the **GPU** gap, so it reinforces "NPU is the defensible part," not "Ollama is behind." The actual threat to the NPU moat is **not** Vulkan but a possible **llama.cpp OpenVINO backend** (open discussion: `ggml-org/llama.cpp#15883`, covers CPU/GPU/NPU via OpenVINO) — watch that, not Vulkan release notes. Tommy's stated stance: **"I am not in competition. The second Ollama does this better than me, I will walk away."** So the trigger to narrow/abandon scope is genuine parity, not territory defense.

**Status (checked 2026-06-10):** Condition PARTIALLY met. Mainline Ollama now has native Intel Arc support via an **experimental Vulkan backend** (added `0.12.6-rc0` Oct 2025, Arc firmed up ~`0.12.11`, baked into current `v0.30.7` June 7 2026). This is the upstream/maintained path, NOT the archived ipex-llm fork (see [[intel_llm_stack_landscape]]) — so it won't bit-rot. BUT: Vulkan is the lowest-common-denominator backend, likely slower than NoLlama's OpenVINO GenAI path; a separate SYCL PR (#11160) for better Intel perf is not yet merged; and shared-memory iGPU (140V/Lunar Lake) is exactly Vulkan's weakest case. **NPU: still zero support anywhere in Ollama — moat fully intact.** Next step before any deletion: benchmark Qwen3-8B + Gemma 3 4B VLM on the 140V via Ollama-Vulkan vs NoLlama tok/s. (Note: `v0.30.7` itself is just Hermes Desktop + minor fixes — not the Arc release; Tommy's "launched 2 days ago with ARC support" conflated the cadence.)
