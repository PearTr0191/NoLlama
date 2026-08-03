# Memory Index
- [Intel LLM stack landscape](intel_llm_stack_landscape.md) — ipex-llm archived (2026); OpenVINO GenAI is the surviving first-party Intel LLM path. Strengthens the Ollama-wrapper spinoff case.
- [Avoid caching, prefer wait](avoid_caching_prefer_wait.md) — feedback: default to re-computation; a brief startup wait beats the long-tail cost of cache staleness/cleverness.
- [NPU memory path isolation](npu_memory_path_isolation.md) — Intel NPU has its own DMA; CPU/GPU benchmarks are best-case-idle, NPU is steady under contention. Affects device-routing UX framing.
- [Copilot CLI no thinking trace](copilot_cli_no_thinking_trace.md) — `copilot.exe` only emits a `...thinking` placeholder, never the actual reasoning. Hard ceiling on wrappers like `copilot-proxy`.
- [Vendor CLI API-wrap ToS risk](vendor_cli_api_wrap_tos_risk.md) — flag ToS/account-risk concerns BEFORE building when user wants to wrap a vendor's interactive CLI as an API. Lesson from copilot-proxy spike (2026-05-26).
- [Agentry project](agentry_project.md) — persistent ACP proxy in front of `copilot --acp`, killing per-turn spawn cost for CLI coding-agents. Lives at C:\devel\aweussom\python\agentry, repo at github.com/aweussom/agentry.
- [NoLlama NPU niche scope](nollama-npu-niche-scope.md) — if Ollama ships working Intel ARC support, drop GPU+CPU from NoLlama and keep NPU as the sole (defensible) target.
- [OpenCLAW + NoLlama integration](openclaw-nollama-integration.md) — how OpenCLAW is wired to NoLlama; heartbeat/CPU-tools/parser changes; the 21k-token prefill wall on the 285K iGPU.
- [User hardware setup](user_hardware.md) — 285K + RTX 5090 desktop; CLAUDE.md's ARC 140V line is stale, don't assume it applies
