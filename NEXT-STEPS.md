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
- ~~When the B60 arrives: run the comprehension test.~~ **Done 2026-08-15,
  and it ended better than expected.** On OpenVINO 2026.3 the B60 fails
  exactly like Xe2/Xe3 (posted to openvino#37419, killing the
  shared-memory theory). On **2026.4.0.dev20260814 it is FIXED** — the
  issue's own repro quotes the prompt verbatim on GPU, at 8-9 tok/s vs 1.0
  on the CPU control. Remaining: post the fix confirmation to #37419.
- **Open question from that run, don't state it as settled:** under greedy
  decoding the GPU and CPU traces are identical for ~40 tokens and then
  diverge into different-but-coherent reasoning, both reaching the correct
  answer. That divergence is **observed, not explained.** It is plausibly
  ordinary cross-plugin numerics (an argmax near-tie flipping, then
  cascading), but nothing here establishes that. Two cheap experiments
  would: (1) run `scripts/glimmer-37419-repro.py` twice on the SAME device
  and confirm byte-identical output, proving the runs are deterministic at
  all; (2) run a known-good model GPU-vs-CPU greedy and see whether it
  diverges the same way — if other models match exactly, Glimmer's
  divergence is suspicious rather than normal. Until then, report it to
  upstream as an observation and let them judge.
- Glimmer's **device gate is now open on 2026.4** (8-9 tok/s is usable, not
  just verifiable) — but that is a *nightly*, so it converts the old
  two-gate problem into a waiting game on the 2026.4 release. When 2026.4
  ships stable, re-run the comprehension test on it and the device gate
  closes for real. The stack gate (muse_glimmer in released transformers +
  optimum-intel) is unaffected and still shut.
- Glimmer in install.ps1/models.json: **deliberately absent** until BOTH
  gates close. (1) Stack gate — muse_glimmer in *released*
  transformers + optimum-intel, so the standard venv serves it with a
  requirements bump; an installer that builds a second venv from git
  main promises reproducibility it can't keep. (2) Device gate — a
  device where it's both correct and fast enough. **The B60 did not close
  this** (2026-08-15, corrupt); it now needs an OpenVINO GPU-plugin fix,
  on any device. Until then the README manual path
  (install-optimum.ps1) is the honest offering for the self-selecting
  few; CPU-only at 1.4-2.6 tok/s behind a menu item is a disappointment
  machine. Re-check the stack gate alongside the Nemotron watch. Any
  automation experiment lives on its own branch (the optimum-backend
  pattern: merge when verified, not before) — and don't cut that branch
  until at least one gate has closed, or it rots against install.ps1.
