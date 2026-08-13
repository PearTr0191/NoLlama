# Next steps

Carried over after the optimum-backend merge (2026-08-13); everything else
from that branch's checklist shipped or is recorded in README/TODONT.

- Qwen3.5-4B vision verdict for the registry note (models.json).
- SmolLM3 registry notes could mention thinking-mode + `/no_think`.
- Nemotron Lightning: still blocked upstream (PR #1789 merged descoped — no
  `nemotron_h` exporter). Decide whether to file the optimum-intel feature
  request offering to test (the Glimmer issue #1927 pattern that worked).
- OpenVINO GPU-plugin bug for Glimmer-on-iGPU filed/tracked — re-run the
  TODONT comprehension test on each new OpenVINO release.
- When the B60 arrives (PostNord pickup ~2026-08-16): run the comprehension
  test FIRST (iGPUs across Xe-LPG/Xe2/Xe3 all fail — see TODONT; dGPU with
  dedicated VRAM is the untested memory path); post the verdict to
  openvino#37419 either way. Only if it passes, benchmark. Glimmer int4
  fits resident (17 GB in 24 GB) — expect 12-18 tok/s if correct.
- Glimmer in install.ps1/models.json: **deliberately absent** until the
  B60 verdict. CPU-only at 1.4-2.6 tok/s behind a menu item is a
  disappointment machine (17 GB + a separate venv the generated start.ps1
  can't use); the README manual path is the honest offering for the
  self-selecting few. If the B60 passes, Glimmer@GPU becomes a flagship
  entry and earns the installer wiring (with device gating) — decide then.
