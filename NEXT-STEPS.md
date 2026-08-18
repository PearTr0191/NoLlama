# Next steps

Carried over after the optimum-backend merge (2026-08-13); everything else
from that branch's checklist shipped or is recorded in README/TODONT.

- **Arc dGPU benchmarks: DONE 2026-08-18.** The README's dGPU column is filled
  (SmolLM3 77.9, Qwen3-8B 64.5, 30B-A3B 50.8 resident, `count 1-100` test, Arc
  Pro B60). Two things worth carrying forward:
  - **Re-measure the older rows.** They were taken with a benchmark harness
    whose TTFT included stream buffering, so their decode figures read a few
    percent high, and Windows TTFT measured before the dual-stack bind carried
    a fixed ~2 s loopback penalty. The dGPU row is the only one measured
    cleanly. Cross-row TTFT comparison is not meaningful until the rest are
    redone — the README says so, but the numbers themselves are still old.
  - **Offload non-determinism is unexplained.** Under `--offload-ratio 30` on
    the B60, greedy decoding returned 87-2040 tokens for the same prompt across
    five runs (resident: 478 every time). Recorded in TODONT and the README as
    an observation. Nobody has looked at whether the *content* is wrong or
    merely different.
- **Benchmark harness gotcha, for whoever automates this next:** a venv built
  from the Microsoft Store Python has a redirector at `venv\Scripts\python.exe`,
  so `Start-Process -PassThru` hands back the launcher's pid, not the server's.
  Killing it leaves the port held, the next server fails to bind, and the
  benchmark silently keeps talking to the previous model. `bench-b60.ps1` now
  kills by port owner and asserts `/health` reports the expected model. Any new
  orchestration script needs both.
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
- Glimmer in install.ps1/models.json: **deliberately absent** until BOTH
  gates close. (1) Stack gate — muse_glimmer in *released*
  transformers + optimum-intel, so the standard venv serves it with a
  requirements bump; an installer that builds a second venv from git
  main promises reproducibility it can't keep. (2) Device gate — a
  device where it's both correct and fast enough (B60 verdict Saturday,
  or an OpenVINO iGPU fix). Until then the README manual path
  (install-optimum.ps1) is the honest offering for the self-selecting
  few; CPU-only at 1.4-2.6 tok/s behind a menu item is a disappointment
  machine. Re-check the stack gate alongside the Nemotron watch. Any
  automation experiment lives on its own branch (the optimum-backend
  pattern: merge when verified, not before) — and don't cut that branch
  until at least one gate has closed, or it rots against install.ps1.
