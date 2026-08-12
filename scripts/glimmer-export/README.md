# Muse Glimmer → OpenVINO int4 export

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
- The export loads 60 GB BF16 through RAM: fine on a 64–128 GB workstation,
  hours of pagefile grinding on a 32 GB laptop. Not resumable mid-run.
- `run-export.py` invokes optimum-cli's parser via Python because corporate
  AV blocks pip's `.exe` launcher stubs (`hf.exe`, `optimum-cli.exe`).
- Before converting, check whether Intel beat us to it:
  an official `OpenVINO/Muse-Glimmer-*-ov` drop makes all of this unnecessary
  (the model-watch workflow opens an issue if that happens).
