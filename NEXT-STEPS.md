# Next steps — after the laptop reboot (registry override)

Context (2026-08-13, Svalbard): `HKLM\SOFTWARE\Intel\GMM\DedicatedSegmentSize =
24576` (24 GB) was set because Intel Graphics Software crashes; a reboot makes
the driver pick it up. These tests close out the `optimum-backend` branch.

## 1. Did the override stick?

```powershell
venv\Scripts\python.exe -c "import openvino as ov; print(ov.Core().get_property('GPU','GPU_DEVICE_TOTAL_MEM_SIZE')/2**30, 'GB')"
```

- **~24 GB** → override works, continue to test 2.
- **Still ~16 GB** → corporate policy re-stamped the registry or the key is
  ignored on this driver. GPU test is then pointless for the 17 GB model —
  skip to test 3, and delete the key
  (`Remove-ItemProperty HKLM:\SOFTWARE\Intel\GMM -Name DedicatedSegmentSize`).

## 2. Glimmer on the 140V GPU — the Xe2 dynamic-shape question

**DONE 2026-08-13 — verdict: a third outcome nobody listed.** Warmup
completes (2.9s, ~2.8 tok/s streaming) but inference is **numerically
corrupt**: the model half-perceives the prompt, hallucinates unsent content,
greedy-loops in the think channel. Same IR + same params on CPU is exact.
Recorded in README optimum section + TODONT (with the comprehension test to
re-run on new OpenVINO releases and on the B60). Optimum backend stays
CPU-only for Glimmer. Diagnosis unearthed two real NoLlama bugs, both fixed:
assistant `<think>` content round-tripping into history (server now strips
in `parse_messages`), and failed sends lingering in web-UI chatHistory
(consecutive user turns read as a corrupted transcript).

## 3. no-think directive check (any machine, browser only)

After `git pull` + Ctrl+F5: toggle no-think on, ask something simple.

- Thinking gone or tiny → `Reasoning strength: low.` is valid vocabulary. Done.
- Thinking merely shorter → edit `NO_THINK_PROMPT` in `static/js/app.js` to
  try `minimal` (then `off`/`none` as further candidates).
- No change at all → the directive needs to be the *whole* control line;
  check the rendered prompt with `--debug` to see what the template emitted.

## Carried over from the 08-06/07 session log (previous content of this file,
## preserved in git history — everything else there shipped or was archived
## into README/TODONT/CLAUDE.md)

- Qwen3.5-4B vision verdict for the registry note (models.json).
- SmolLM3 registry notes could mention thinking-mode + `/no_think`.

## 4. Then

- Merge `optimum-backend` → `main` when satisfied (all verification was green;
  GenAI regression included).
- Delete this file as part of the merge.
- Nemotron Lightning: still blocked upstream (PR #1789 merged descoped — no
  `nemotron_h` exporter). Decide whether to file the optimum-intel feature
  request offering to test (the Glimmer issue #1927 pattern that worked).
- When the B60 arrives: Glimmer int4 fits resident (17 GB in 24 GB) — expect
  12-18 tok/s; that's the real serving host for the optimum backend.
