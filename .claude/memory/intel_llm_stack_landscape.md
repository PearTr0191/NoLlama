---
name: intel-llm-stack-landscape
description: "Intel's LLM-on-Intel-hardware landscape as of 2026-05 — ipex-llm archived, OpenVINO GenAI is the surviving first-party path"
metadata: 
  node_type: memory
  type: reference
  originSessionId: a8a4325f-377f-4561-8637-1c6db75d6fcd
---

**ipex-llm is archived** (github.com/intel/ipex-llm — Intel announced no further maintenance, bug fixes, or releases). It was Intel's blessed bridge for running llama.cpp / Ollama / vLLM on Arc and Iris Xe GPUs via SYCL/Level Zero.

With ipex-llm gone, the Intel-LLM-serving landscape narrows to:
- **OpenVINO GenAI** — first-party, actively maintained, what NoLlama is built on. Supports NPU + GPU + CPU.
- **Community Ollama/llama.cpp on Intel GPU** — still works against pinned ipex-llm versions, but no driver-rev fixes or new model arch support going forward.
- **NPU was never on the ipex-llm roadmap** — nothing lost there.

**Why this matters for NoLlama:**
- Reinforces OpenVINO as the correct bet for Intel-hardware LLM serving.
- Strengthens the case for the Ollama-wrapper spinoff (see commit `4f00190 TODO: honest reassessment of Ollama-wrapper spinoff`) — Ollama users on Intel hardware now have no vendor-supported acceleration path, so an OpenVINO-backed Ollama-compatible API fills a real gap.

**How to apply:** When comparing stacks or evaluating "should we support backend X on Intel," treat ipex-llm/Ollama-on-Intel-GPU as a dead end for new work. When discussing the spinoff TODO, factor in that the competitive Intel-GPU runtime just lost vendor support.

Source: dev.to/khusyasy/local-llm-model-on-intel-iris-xe-using-ollama-4hmn (the recipe that ipex-llm enabled) and github.com/intel/ipex-llm archival banner, surfaced 2026-05-21.
