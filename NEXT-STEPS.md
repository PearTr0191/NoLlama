# Next steps

Carried over after the optimum-backend merge (2026-08-13); everything else
from that branch's checklist shipped or is recorded in README/TODONT.

## B60 test campaign — status at 2026-08-16

Short version: **the correctness work is done. The benchmarking is not.**
Detail for each item is in the bullets further down; this is the map.

Done, and posted or documented:

1. **Qwen3.8-27B (GenAI path) works on the B60.** Loads in 2.0 s, ~17-19
   tok/s, passes comprehension well past the `HELLO!` screen. Needs the
   OpenVINO nightly (branch `qwen38-nightly`).
2. **Glimmer corrupts on the B60 under OpenVINO <=2026.3** — same
   fingerprint as Xe2/Xe3, so dedicated VRAM does not escape it and the
   shared-memory theory is dead. Posted to openvino#37419.
3. **Glimmer is FIXED on 2026.4.0.dev20260814**, 8-11 tok/s. Verified with
   the issue's own repro plus harder prompts, against a same-venv CPU
   control. Posted to openvino#37419.
4. **The GPU-vs-CPU divergence is ordinary**, not a residue — a known-good
   model diverges the same way, at an argmax near-tie.
5. **XMX confirmed present** on the B60 (first discrete-Battlemage
   datapoint; every previous XMX measurement was the 140V iGPU).

Open, in the order they are worth doing:

- **The workstation has a hardware fault.** It crashed three times, most
  recently while merely installing an npm package — no GPU, no model, no
  memory pressure. That is not ours. MemTest86 is running. Also pending:
  the `BugcheckCode` field on the Kernel-Power 41 events (gives the stop
  code even though the dumps failed), and enlarging the 2 GB pagefile so
  future crashes leave evidence at all. **Everything else on this list is
  blocked behind trusting that machine.**
- **No benchmark numbers were ever captured.** This is the real gap. The
  README speed table still reads *wanted* in the Arc dGPU column, and this
  is the project's first discrete card. The figures quoted above are
  ad-hoc single runs from the web UI, NOT `benchmark.py` output, and should
  not be copied into the table as if they were. Wanted: SmolLM3-3B,
  Qwen3-8B, and Qwen3-30B-A3B (which also exercises `--offload-ratio` on
  real XMX hardware for the first time).
- **Post the divergence resolution to openvino#37419.** The comment there
  leaves it explicitly open ("I do not know whether..."); it now has an
  answer, and the answer costs them nothing.
- Two branches are unmerged: `qwen38-nightly` (installer + Web UI, whose
  `</think>` fix is already backported to main) and `glimmer-b60-verdict`
  (all the Glimmer/B60 material). Neither has been reviewed as a whole.

Deliberately dropped: the Qwen3.8 GPU-vs-CPU divergence datapoint. It would
only corroborate a question already answered, and it is the workload that
was running during two of the crashes.

- **transformers main breaks the optimum backend's TEXT-ONLY path**
  (found 2026-08-15). `transformers 5.16.0.dev0` calls
  `get_experts_implementation()` from `_optimize_model_for_decode()` on
  every `_sample()`; `OVModelForCausalLM` doesn't implement it, so
  `generate()` dies with AttributeError. `OVModelForVisualCausalLM` has its
  own `generate()` and is unaffected — which is the only reason Glimmer
  (VLM-shaped export) works, and why this went unnoticed. Affects
  `venv-optimum` and `venv-optimum-nightly` alike; it's a transformers
  regression, not an OpenVINO one. Consequences: any future text-only
  optimum-backend model (nemotron_h) will hit it, and
  `install-optimum.ps1`'s `-TransformersRef main` default is the exposure.
  Decide between pinning a known-good transformers ref as the default and
  waiting for optimum-intel to catch up. `scripts/device-divergence.py`
  carries a loudly-announced shim for dense models.
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
  cascading), but nothing here establishes that.
  - Experiment (1) **done 2026-08-15: runs are deterministic.** A second
    run of `scripts/glimmer-37419-repro.py` was byte-identical to the first
    on *both* devices, so the divergence is reproducible, not noise.
  - Experiment (2) **done 2026-08-16: RESOLVED, the divergence is ordinary.**
    SmolLM3-3B (known-good) on the same B60 via openvino_genai, greedy,
    subprocess-isolated: GPU and CPU diverge too. The content is the proof
    — at the divergence point the prompt is "The sky appears blue because
    ___", GPU picks `' shorter'` and CPU picks `' blue'`, and both
    continuations are correct and near-equivalent. That is an argmax
    near-tie flipping, which is what "ordinary cross-plugin numerics"
    actually looks like. Glimmer needs no special explanation.
    **Caveat:** the index is NOT a stable quantity — the same model gave 42
    in one session and 58 in another (reboot in between; CPU was
    byte-identical across both). Cite it as "divergence occurs in this
    depth range", never as a precise figure, and never compare two indices
    from different sessions.
- Correctness on 2026.4/GPU verified well past the `HELLO!` test: a
  409-char multi-step word problem quoted back verbatim and solved with
  correct intermediates (30%), 17*23=391, sum 2..8=35, no non-terminating
  runs. Worth remembering the method — the short prompt was only ever the
  cheap screen; a long prompt with numbers in it is the real test, since
  the 2026.3 failure could not hold 30 characters intact.
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
  device where it's both correct and fast enough. The B60 closed this on
  **2026.4**, but only on the nightly, so it does not count yet.
  **Standing rule from the user (2026-08-15): not until 2026.4 is
  *released*. "We want to be leading edge, not BLEEDING edge."** Shipping
  a menu item that needs a nightly wheel is the definition of bleeding.
  Docs may say we KNOW Glimmer will work soon — that is honest and
  useful; the installer may not act on it. Until then the README path
  (install-optimum.ps1) is the honest offering for the self-selecting
  few; CPU-only at 1.4-2.6 tok/s behind a menu item is a disappointment
  machine. Re-check the stack gate alongside the Nemotron watch. Any
  automation experiment lives on its own branch (the optimum-backend
  pattern: merge when verified, not before) — and don't cut that branch
  until at least one gate has closed, or it rots against install.ps1.
