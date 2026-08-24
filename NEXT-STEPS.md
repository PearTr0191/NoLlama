# Next steps

State after the 2026-08-18 merge. Anything settled lives in README, TODONT or
the docs — this file is only what's still open.

## Open

- **Docker/#31: Phases 0-2 measured 2026-08-24, container path works.** Full
  results in `DOCKER-INSTALL.md`; the short version is that an Intel GPU is
  usable from a container at native throughput (74-79 vs 76-78 tok/s, prefix
  cache 1.9s → 0.3s vs 2.1s → 0.2s native), and two NoLlama bugs surfaced and
  were fixed on the way:

  1. **cgroup-blind memory sizing** (`_cgroup_mem_limit_bytes`): a
     `--memory=4g` container sized a 4 GB KV pool from the host's 23.5 GB
     `MemTotal`. Now `min(MemTotal, cgroup limit)`, v2 and v1.
  2. **WSL `/dev/dxg` 1 GiB allocation cap** (`_gpu_large_alloc_props`): the
     same B60 reports 25,055,051,776 bytes max-alloc natively and exactly
     1,073,741,824 through a container, so Gemma 4 E2B's 2.2 GB per-layer
     embedding table killed the load. A GPU whose max-alloc is below its own
     total budget now gets `GPU_ENABLE_LARGE_ALLOCATIONS`. Native installs
     take no hint and are unaffected.

  Still open, in rough priority order:

  - **`gemma-4-26b-a4b-it-int4-ov` produces deterministic garbage on the
    container GPU path.** Correct natively, correct on CPU inside the same
    container, correct on GPU for every other model tried including a 16 GB
    MoE. Byte-identical gibberish across pipelines and runs, so it is a
    compute defect, not corruption. Worth an upstream report against NEO
    26.31 on the WSL /dev/dxg path — not filed.
  - **Phase 3 packaging** — `Dockerfile` + `compose.yml` into the repo. The
    working image is still only in a session scratchpad. Compose must bind
    models under their real directory names, publish 8000 **and** 11434, and
    give `/app` a writable volume or prewarm silently never persists.
  - **NPU in a container: closed 2026-08-24, the answer is no.** 285K taken
    to WSL 2.9.8.0 (WSL Containers preview): no `/dev/accel*` in either
    channel, and `wslc run` exposes `--gpus` and no device flag at all, so
    there is no way to even ask for one. See TODONT.
  - **Native Linux `/dev/dri` is still untested**, and it is what #31 asks
    for. Neither limitation found here predicts the native answer.
  - Why models needing the large-allocation hint load ~3x slower (E2B 11.5s
    native vs 33-42.5s in-container; SmolLM3, which needs no hint, is at
    parity).
  - `_maybe_capture_prewarm` swallows `OSError`, so a read-only rootfs is a
    silent cold start forever. Wants a log line.

- **VLM slots are agent-grade (merged as PR #30 + the prewarm commit).**
  Three changes, all verified end-to-end 2026-08-18:

  0. **Prewarm on VLM slots** (followed the PR straight onto main): capture
     now happens on the VLM paths of both API surfaces, and the startup
     prefill replays through `parse_messages`' flattening so the cached
     token prefix matches real requests. Measured Glimmer/B60 through the
     network API: first turn after a restart 12.4s → **0.65s** TTFT
     (startup prewarm cost 12.1s, paid before the port answers requests).
     Slots whose runtime fell back to the plain pipeline zero `kv_pool_gb`
     at load, so prewarm skips them rather than burning a 30B prefill for
     nothing (this also makes `/health` honest about a dead cache).

  1. **Tool calling on VLM slots** (both API surfaces; buffered like LLM
     tool turns; images may ride along with tools). Qwen3.5-4B on the 140V
     and Glimmer on the B60 both return structured
     `get_weather({"city":"Oslo"})` with `finish_reason=tool_calls`;
     Glimmer's reasoning stays in `<think>` with no channel leak
     (`_AtemPlainFilter` also closes the think block at
     `<atem:function_calls>` so the tool XML reaches `parse_tool_calls`).
     This un-does the one regression the GenAI reroute had — Glimmer agent
     use no longer wants `--backend optimum`.
  2. **Prefix caching on VLM slots.** VLMPipeline honors `scheduler_config`
     — the long-standing "CB backend is LLM-only" belief was stale. Verified
     on 2026.3 *release* (140V, ~9k-token prefix 21.7s→3.9s TTFT) and the
     2026.4 nightly (B60/Glimmer, 33k-token prefix 53.7s→1.4s through
     NoLlama's serving path). Runtimes that reject the property fall back to
     the plain pipeline with a log line, like the LLM branch.

  Honest observations from the measurements:
  - **CB VLM prefill is slower cold**: the same 33k prompt prefilled in
    ~8.7s on the plain pipeline vs 53.7s under CB (then 1.4s per repeat).
    Agents win from turn two; one-shot prompts pay more once.
    `--no-prompt-cache` restores the plain pipeline if that bites.
  - The plain pipeline **OOM'd on the first 33k-token request** on the B60
    (16 GB USM allocation failed; the immediate retry succeeded). Under CB
    the same request completed first try. Unexplained — file upstream if it
    reproduces.
  - The "minutes of prefill" worry for agent prompts was wrong for the B60
    class: 33k tokens prefill in ~9s on the plain pipeline.

- **Intel docs gap — filed upstream as openvino.genai#4343 (2026-08-18).**
  The VLMPipeline API docs describe its kwargs only as "Device properties"
  and never mention `scheduler_config`/prefix caching; the GenAI guide shows
  SchedulerConfig on LLMPipeline only. The feature works (our measurements
  above, on 2026.3 release AND 2026.4 nightly) — undocumented, not
  unsupported. The issue also flags the slow cold CB prefill (~54s vs ~9s
  plain, same prompt/HW) as an observation; if Intel asks, offer the
  standalone repro. (Track: Intel has historically fixed our reports
  within a day.)

- **USM OOM: filed upstream as openvino.genai#4344 (2026-08-18).**
  Raw VLMPipeline (plain, no scheduler_config), Glimmer int4 on the B60:
  first ~33k-token generate fails with a USM Device allocation error;
  identical retry succeeds. 100% reproducible, with or without short
  generates first. Both observed failure sizes divide by exactly 1.1:
  16,049,884,928 = 14,590,804,480 × 1.1 and 16,031,296,512 =
  14,573,905,920 × 1.1 — the GPU plugin's ShapePredictor "percentage
  preallocation" (`buffers_preallocation_ratio = 1.1`, options.inl:61)
  applied to a huge dynamic buffer (full-sequence logits or attention
  scratch). The CB path avoids it because chunked prefill never allocates
  the full-sequence buffer — consistent with scheduler_config being the
  workaround AND with CB's slower cold prefill. Bonus bug found while
  testing: setting `OV_GPU_SHAPE_PREDICTOR_SETTINGS` (a RELEASE_INTERNAL
  option) crashes pipeline construction — `ShapePredictor::Settings` has no
  string parser ("Bad as from std::string"), so the env knob is unusable
  and a bad value kills the load. Weight staging through host/shared memory
  is by design (two-stage allocation, memory_allocation_gpu_plugin.md); no
  public knob for device-direct loading; `usm_policy`/`disable_usm` are
  debug-caps-only. Windows "shared GPU memory" is the WDDM half-of-RAM
  budget — discrete GPUs have it too, no iGPU required.

- **Local sparse checkouts of Intel sources** (for grepping docs + GPU
  plugin internals): `C:\devel\intel\openvino` (docs/articles_en +
  src/plugins/intel_gpu, shallow) and `C:\devel\intel\openvino.genai`
  (site + src). Machine has no git-lfs — clone with
  `GIT_LFS_SKIP_SMUDGE=1` and LFS filters disabled; partial-clone sparse
  blob fetch dies on this network, plain `--depth 1` works.

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
  *release*.** Standing rule: leading edge, not bleeding edge. The GenAI
  reroute works on the 2026.4 nightly, but a menu item that needs a nightly
  wheel is bleeding. When 2026.4 releases, the entry is Intel's
  `OpenVINO/Muse-Glimmer-30B-int4-ov` with `"requires_nightly"` dropped —
  the manual path until then is `install.ps1 -Nightly` plus a hand
  download (`install-optimum.ps1` is no longer the recommended Glimmer
  path, only the `--backend optimum` fallback). Docs may say we know it
  will work; the installer may not act on it.

  **Qwen3.8 27B rides the same gate** (removed from `models.json`
  2026-08-21). It had a menu entry carrying `requires_nightly: true`, which
  contradicted this very rule; the rule wins. Re-add
  `OpenVINO/Qwen3.8-27B-int4-ov` when 2026.4 ships as a release — and test
  it first, since it was never run here.
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
