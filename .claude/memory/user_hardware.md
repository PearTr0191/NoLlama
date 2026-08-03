---
name: User hardware setup
description: Tommy's actual workstation specs — relevant when discussing performance, model sizing, or device selection
type: user
originSessionId: 4ff37bf5-d22d-4841-a7ce-4f108408970f
---
Tommy has two machines that both run NoLlama-relevant workloads:

**Desktop (Arrow Lake-S):**
- CPU: Intel Core Ultra 9 285K (NPU 3 + Xe-LPG iGPU on-die)
- Discrete GPU: NVIDIA RTX 5090 (GDDR7, ~1.8 TB/s)
- System memory: standard DDR5 (~100 GB/s dual-channel)
- Runs Ollama on the 5090 for Qwen3-series LLMs

**Laptop (Lunar Lake) — what NoLlama's CLAUDE.md was originally written for:**
- Intel Core Ultra + ARC 140V 16GB iGPU + NPU
- LPDDR5X-8533 on-package (~136 GB/s)

Important benchmarking caveat: for memory-bandwidth-bound LLM decode, the *laptop's* iGPU/NPU may actually beat the *desktop's* iGPU/NPU because LPDDR5X >> DDR5. Don't assume the desktop wins just because the silicon is "newer/bigger" — the desktop wins on CPU compute and the discrete 5090, but loses memory bandwidth on the integrated devices. Treat CLAUDE.md hardware lines as describing the laptop, and confirm which machine before quoting ARC 140V details.
