# NEXT STEPS (2026-08-06, OpenVINO 2026.3 testing session)

## OFFLOAD INVESTIGATION: CLOSED — root cause found, swap raise NOT needed

`OFFLOAD_RATIO` requires an **XMX-capable GPU**: the MoE fusion it depends on is gated
`if (device_info.supports_immad && oneDNN)` in the GPU plugin
(transformations_pipeline.cpp). The 285K's Xe-LPG iGPU has no XMX
(OPTIMIZATION_CAPABILITIES lists no GPU_HW_MATMUL — verified) → silent no-op on this
machine, regardless of export vintage, pagefile, or ratio. Proven end-to-end: fresh
2026.3-stack LFM2-8B-A1B export (tiled expert constants confirmed in the IR) loads and
runs 27 tok/s on the iGPU with byte-identical 14.91 GB device memory at ratio 0 and 90.

Consequences:
- **Do NOT bother raising swap for the 30B re-export** — it cannot offload here anyway.
- The **Arc 140V laptop HAS XMX** — that's the machine to validate offload on (fresh-stack
  export + ratio 0 vs 90 + GPU_MEMORY_STATISTICS). The LFM2-8B-A1B-int4-2026.3 export
  (~4.5 GB, `~/models`) is the ready-made test artifact to copy over.
- For #19: ask Dmitriy what GPU his laptop has before promising anything — 128 GB RAM
  suggests Meteor/Arrow Lake-H (no XMX → no offload for him either). Draft updated.
- Big-MoE loads OOM in staging on non-XMX iGPUs (11.6 GB model on 33 GB device fails);
  that's the same missing fusion, not a memory setting. Full log in TODONT.md.

## Waiting on Tommy

1. **Upload go-ahead**: 4 validated builds with model cards + LICENSEs ready in
   `~/models/{SmolLM3-3B-int4-cw, SmolLM3-3B-int8-cw, LFM2-1.2B-int4-cw, LFM2.5-1.2B-Instruct-int4-cw}`
   → `hf upload` under your namespace, then add to `models.json` (npu category).
2. **#19 comment**: draft at scratchpad `issue19-comment-draft.md` — update with the
   LFM2-8B-A1B result before posting.
3. **Revoke unattended permissions**: delete `"PowerShell"`, `"Bash"`, `"WebFetch"` from
   `.claude/settings.local.json`.
4. **Disk cleanup** (~40 GB of experiment leftovers in `~/models`, nothing deleted):
   `LFM2-1.2B-int8-cw`, `LFM2.5-1.2B-Instruct-int8-cw` (NPU garbage), `LFM2-1.2B-int8-asym`,
   `LFM2-1.2B-int8-new`, `LFM2-1.2B-int4-cw-v2` (experiments), `SmolLM3-3B` (superseded),
   `Qwen3-30B-A3B-int4-ov`, `LFM2-24B-A2B-int4-ov` (failed controls, ~27 GB);
   keep `_src-Qwen3-30B-A3B` until the export succeeds.

## Session verdicts (full detail in TODONT.md + CLAUDE.md)

- NPU: SmolLM3-3B / LFM2-1.2B / LFM2.5-1.2B all work on 2026.3 — but ONLY channel-wise
  exports (`download-model.ps1 -Weight int4-cw|int8-cw`); default int4 crashes the vpux
  compiler. LFM int8 is a trap (sym=garbage, asym=1.4 tok/s). LFM builds are NPU-only.
- EAGLE-3: works; +6% GPU, +14% CPU on Qwen3-8B. Draft export needs venv-2026.3 stack.
- OFFLOAD_RATIO: not validated on this box — every big-MoE load dies in USM Host staging
  before offload matters (even 11.6 GB on the 33 GB iGPU); old-vintage IRs additionally
  can't fuse to MOECompressed. Mechanism notes in TODONT.md.
- Qwen3.6-35B-A3B: NPU dead (shape inference), iGPU dead (staging OOM). Parked.
