# MoE disk offload (`--offload-ratio`, 2026-08-06)

`--offload-ratio PCT` streams PCT% of MoE expert weights from disk on GPU
slots (OpenVINO 2026.3 `OFFLOAD_RATIO`).

**Requires XMX** (Arc / Lunar Lake — `GPU_HW_MATMUL` in
`OPTIMIZATION_CAPABILITIES`). Without it the property is a **silent no-op**,
so NoLlama warns at startup. Non-XMX iGPUs can't load big MoE at all (USM
staging OOM) — full story in `TODONT.md`, which also records that
OFFLOAD_RATIO could not be validated on the desktop iGPU.

Verified on Arc 140V, Qwen3-30B-A3B int4, steady state:

| ratio | resident | decode |
|---|---|---|
| 30 | 10.8 GB | 25.3 tok/s (interactive) |
| 90 | 2.35 GB | 5.1 tok/s |

Pick the smallest ratio that fits. The expert LRU needs **~60 tokens to
warm** — benchmark steady state, not the first sentence.

Known upstream bug: a **second** `generate()` on an offload-active **plain**
pipeline hangs in native code, uninterruptible. NoLlama's serving path is
unaffected: the CB backend it uses was verified with sequential requests
(140V, ratio 30 — 12.5 then 15.9 tok/s, prefix-cache TTFT 8.0s→1.9s).
