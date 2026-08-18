# Next steps

State after the 2026-08-18 merge. Anything settled lives in README, TODONT or
the docs — this file is only what's still open.

## Open

- **Glimmer runs on the GenAI path now — reroute it.** Intel published an
  official OpenVINO export on 2026-08-12
  (`OpenVINO/Muse-Glimmer-30B-int4-ov`, 17 GB, INT4_ASYM g64) and it is
  **VLM-shaped**: separate vision / text / language IRs. Verified 2026-08-18 on
  the B60 via `openvino_genai.VLMPipeline` on **GPU**: loads, quotes the
  instruction back verbatim, answers 2+2 correctly, ~14 tok/s (against 8-11 on
  the optimum path). No optimum backend, no transformers-from-git, and the
  optimum-path GPU corruption never applied to GenAI.

  `NEEDS_OPTIMUM = {"muse_glimmer", "nemotron_h"}` (nollama.py:129) now blocks
  this. Its stated reason — *"muse_glimmer's language model takes inputs_embeds
  (no LLMPipeline)"* — is an argument about **LLMPipeline**; a VLM-shaped export
  goes to **VLMPipeline**, which feeds a language model from embeddings by
  design. Fix is structural, not a special case: **`is_vlm` should override the
  architecture blocklist**, which keeps our own LLM-shaped export on optimum
  where it belongs.

  Still needs the nightly runtime — Intel exported it with a
  `2026.4.0-...-muse_onyx` build and the card wants 2026.3.1+ with a genai
  pre-release. So Glimmer's stack gate becomes **"2026.4 ships stable"** rather
  than "muse_glimmer in released transformers", which is the same gate Qwen3.8
  is already waiting on. Note VLM slots get no prefix cache (the CB backend is
  LLM-only), so that win does not arrive with this.

- **The ATEM filter cannot simply be ported to the GenAI path.**
  `_AtemStreamFilter` keys on `<|start|>`, `<|message|>`, `<|eom|>`, `<|eot|>`.
  `VLMPipeline.generate()` strips special tokens and `GenerationConfig` has no
  `skip_special_tokens` (only `Tokenizer.decode` does, which the pipeline does
  not expose). What survives is the plain-text routing — `to=self`,
  `assistant to=user` — so a GenAI-path filter has to parse bare words that
  could legitimately appear in content. More fragile than the existing one.
  Gate it on the flag that already exists (`model_type == "muse_glimmer"`,
  nollama.py:1881) rather than adding a second one.

  This is not an Intel bug: stripping special tokens is correct detokenisation,
  and the routing words are plain text the model emitted. Serving a
  harmony-channel model is the application's job — vLLM and TGI would show the
  same thing.

- **Loading a big model stages through host memory first.** Watched on the B60
  (17 GB Glimmer): shared GPU memory ramps to near its 16 GB ceiling and holds
  there while dedicated VRAM stays flat, then dedicated fills, then shared
  drains. So **peak host RAM during load is roughly model-sized even on a
  discrete card** — worth knowing before assuming 24 GB of VRAM makes system RAM
  irrelevant.

- **`hf download` stalls on large files via Xet.** It sat at 0.00 CPU with a
  `.lock` on the 14.9 GB blob. `HF_HUB_DISABLE_XET=1` resumed it and ran at
  ~78 MB/s. Also leaves an abandoned partial in `.cache/huggingface/download`
  that has to be deleted by hand (17 GB of files, 28.7 GB on disk until then).

- **Glimmer into `install.ps1`/`models.json`: waits for OpenVINO 2026.4 as a
  *release*.** Standing rule: leading edge, not bleeding edge. The device gate
  closed on the 2026.4 nightly (GPU output correct, 8-11 tok/s), but a menu item
  that needs a nightly wheel is bleeding. The stack gate is still shut too —
  `muse_glimmer` isn't in released transformers/optimum-intel, so
  `install-optimum.ps1` remains the honest path. Docs may say we know it will
  work; the installer may not act on it.
- **`transformers` main breaks the optimum backend's text-only path.**
  `5.16.0.dev0` calls `get_experts_implementation()` from
  `_optimize_model_for_decode()`; `OVModelForCausalLM` doesn't implement it, so
  `generate()` dies. `OVModelForVisualCausalLM` has its own `generate()` and is
  unaffected — the only reason Glimmer works. This will bite `nemotron_h`, which
  is text-only. `install-optimum.ps1 -TransformersRef main` is the exposure:
  decide between pinning a known-good ref and waiting for optimum-intel.
- **Offload non-determinism on the B60, unexplained.** At
  `--offload-ratio 30`, greedy decoding returned 87-2040 tokens for the same
  prompt across five runs (resident: 478 every time). Varying length proves
  something varies; nobody has looked at whether the content is wrong or merely
  different. Detail in TODONT.
- **The offload split didn't track the ratio.** Ratio 30 left 3.2 GB of 15.2 GB
  resident (~24%) where the 140V measured 10.8 GB (~71%), with 21 GB of VRAM
  free. Either the ratio is a ceiling a demand-driven expert LRU never fills, or
  it behaves differently on discrete hardware. Runs at 50 and 90 would tell.
- **Ollama head-to-head needs redoing with the temperature pin.** The old
  comparison had Ollama sampling (its default 0.8) against NoLlama greedy (0.0),
  because `benchmark.py` sent no temperature. Fixed now. The 1.6× decode figure
  probably survives; the *task-time* reading of it does not, because Ollama's
  build ignores `/no_think` and spends ~1755 tokens on a 291-character answer
  where NoLlama spends 293. Needs Ollama on the 140V.
- **Nemotron Lightning: still blocked upstream.** PR #1789 merged descoped — no
  `nemotron_h` exporter. Decide whether to file the optimum-intel feature request
  offering to test (the pattern that worked for Glimmer, issue #1927).
- Re-run the TODONT comprehension test on each new OpenVINO release.
- Qwen3.5-4B vision verdict for the registry note (`models.json`).
- SmolLM3 registry notes could mention thinking-mode + `/no_think`.

## Benchmarking notes for whoever runs the next one

- **Use the 285K or the B60 box, not the laptop.** A busy 140V reads ~30% low
  (Qwen3-8B int4-cw: 14.8 tok/s with a browser and chat apps running, 19.4
  quiet). Decode figures across the table were verified sound on the 285K
  (SmolLM3 iGPU 29.4 vs 29.7 published, Qwen3-8B 14.6 vs 15.4).
- **Kill servers by port owner, not by pid.** A venv built from the Microsoft
  Store Python has a redirector at `venv\Scripts\python.exe`, so
  `Start-Process -PassThru` returns the launcher's pid and the real server
  survives being stopped. The next server then fails to bind and the benchmark
  quietly keeps talking to the previous model. `scripts/bench-b60.ps1` kills by
  port and asserts `/health` reports the expected model; copy both.
- **Detached `pwsh` launched over SSH dies when the session ends.** Long
  orchestration runs need to be started locally, or driven one step per SSH
  call.
