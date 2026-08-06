---
name: ov-2026-3-offload-ratio-gpu-only
description: "OpenVINO MoE disk offload (OFFLOAD_RATIO) requires an XMX-capable Intel GPU — silently no-ops on Xe-LPG iGPUs like the 285K's; also GPU-plugin-only (no NPU/CPU)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 84f71153-d48b-4f74-911c-1ee6cb4055e2
  modified: 2026-08-06T15:34:19.294Z
---

OpenVINO 2026.3's "MoE offloading to disk" (`OFFLOAD_RATIO`, GPU plugin
property, % of MoE experts streamed from disk via LRU) has TWO gates:

1. **GPU-plugin-only** — NPU and CPU don't support the property (verified
   via SUPPORTED_PROPERTIES on the installed runtime, 2026-08-06).
2. **XMX/systolic hardware required** — the TiledMoeBlock→MOECompressed
   fusion it depends on is gated `if (device_info.supports_immad && oneDNN)`
   in `intel_gpu/src/plugin/transformations_pipeline.cpp`. On non-XMX GPUs
   (desktop Arrow Lake Xe-LPG, Meteor Lake iGPUs) it is a **silent no-op**:
   identical memory and speed at any ratio, no warning. Verified end-to-end
   on the 285K desktop with a confirmed-tiled fresh export (LFM2-8B-A1B).

**Why:** a full day of experiments (2026-08-06) chased export vintage and
memory settings before finding the hardware gate in source. The failure is
silent, so nothing points at it.

**How to apply:** before recommending offload (e.g. NoLlama issue #19),
check the target GPU: `OPTIMIZATION_CAPABILITIES` must include
`GPU_HW_MATMUL` (Arc dGPU, Lunar Lake 140V, Panther Lake = yes; desktop
Xe-LPG / Meteor Lake = no). Tommy's Arc 140V laptop qualifies; his 285K
desktop does not. Full experiment log in NoLlama TODONT.md.
Related: [[user_hardware]].
