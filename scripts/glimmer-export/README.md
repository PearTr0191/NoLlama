# Muse Glimmer → OpenVINO int4 export

> **SUPERSEDED (2026-08-18).** Intel now publishes an official export —
> [`OpenVINO/Muse-Glimmer-30B-int4-ov`](https://huggingface.co/OpenVINO/Muse-Glimmer-30B-int4-ov)
> (2026-08-12) — which is VLM-shaped and runs on GenAI's `VLMPipeline`
> (verified B60, 2026-08-18, ~14 tok/s). Download that instead of running
> these scripts. This directory stays because it records the working
> export recipe and the RAM-bound conversion lesson (60 GB BF16 through
> host RAM), and because our published export
> ([`aweussom/Muse-Glimmer-30B-int4-ov`](https://huggingface.co/aweussom/Muse-Glimmer-30B-int4-ov))
> is the repro model named in openvino#37419.

Meta's Muse Glimmer 30B (best-in-class local agent model) has upstream OpenVINO
export support since optimum-intel PR #1924 (merged 2026-08-11) — tracked in
our issue huggingface/optimum-intel#1927. These scripts run that recipe.

Two steps, same `-Workspace` (default `C:\devel\aweussom\glimmer-port`,
needs ~80 GB free — weights, venv and output live there, not in this repo):

```powershell
.\oslo-prep.ps1      -Workspace D:\glimmer-port   # clones, wheels, 60 GB weights (resumable)
.\export-glimmer.ps1 -Workspace D:\glimmer-port   # BF16 -> int4 IR (~16 GB output)
```

Notes:
- `muse_glimmer` is not in any *released* transformers (the checkpoint was made
  by `5.15.0.dev0`) — export-glimmer installs transformers from the git clone.
- The export loads 60 GB BF16 through RAM: fine on a 64–128 GB workstation,
  hours of pagefile grinding on a 32 GB laptop. Not resumable mid-run.
- `run-export.py` invokes optimum-cli's parser via Python because corporate
  AV blocks pip's `.exe` launcher stubs (`hf.exe`, `optimum-cli.exe`).
- Before converting, check whether Intel beat us to it:
  an official `OpenVINO/Muse-Glimmer-*-ov` drop makes all of this unnecessary
  (the model-watch workflow opens an issue if that happens).
