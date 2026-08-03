---
name: npu-memory-path-isolation
description: Intel NPU has its own DMA path to system memory; performance vs CPU/GPU is workload- AND contention-dependent
metadata: 
  node_type: memory
  type: project
  originSessionId: a8a4325f-377f-4561-8637-1c6db75d6fcd
---

**Intel AI Boost NPU uses a dedicated DMA path to system memory, separate from the CPU and GPU memory controllers.** This breaks the simple "device X is fastest" benchmarking framing.

**Why:** Memory bandwidth on the CPU and GPU is shared with everything else the system is doing — browser tabs, builds, background services, file I/O, game rendering. The NPU isn't on those buses. Its bandwidth is consistent regardless of how loaded the rest of the system is.

**Consequences for NoLlama benchmarking and device routing:**

- "CPU > NPU on Arrow Lake" (`0bbb948`'s finding for text LLM decode) was measured on an idle system. Best-case for CPU. **Under load, the gap closes or inverts** because CPU loses bandwidth to other tenants while NPU's path stays clean.
- "GPU > CPU for VLM" (2026-05-26 finding) is more robust because VLM prefill is compute-bound on the vision encoder, not bandwidth-bound — but the GPU's memory bandwidth is still shared, so heavy contention can still degrade it.
- NPU is a strong fit for "always-on agent" / "while-I-work" workloads where the user is doing other things on the same machine. Pure "give me the absolute fastest single-prompt response on an idle box" is a different optimization target.

**How to apply:** When proposing device defaults or recommending a device for a given workload, ask: is this a benchmark workload (idle box, pure throughput) or a real-use workload (system doing other things in parallel)? For the latter, NPU is undervalued by idle benchmarks. The install-time UX should frame the choice as "what matches your usage pattern" rather than "which is fastest."

Related TODO: `TODO.md` "Offer CPU as install choice even when NPU/GPU detected" — captures this nuance in the proposed install UX framing.

Source: hardware fact about Intel Core Ultra NPU architecture, surfaced in a parallel claude.ai conversation 2026-05-26.
