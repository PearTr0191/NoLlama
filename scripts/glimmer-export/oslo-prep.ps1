#requires -Version 7.0
<#
Stage the Muse Glimmer -> OpenVINO export workspace (repos, wheels, weights).
Upstream support landed in optimum-intel PR #1924 (merged 2026-08-11), so the
job is export + test, not porting. Needs: optimum-intel from git main,
transformers==5.15, opencv-python.

Staged smallest-first so an interrupted session still completes the essentials;
every stage is idempotent and the weights download resumes on re-run.

  .\oslo-prep.ps1                             # workspace: C:\devel\aweussom\glimmer-port
  .\oslo-prep.ps1 -Workspace D:\glimmer-port  # e.g. big disk on the workstation
#>
param(
    [string]$Workspace = 'C:\devel\aweussom\glimmer-port'
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force $Workspace | Out-Null

# --- Stage 0: disk sanity -------------------------------------------------
$drive = (Resolve-Path $Workspace).Path.Substring(0, 1)
$freeGB = [math]::Round((Get-PSDrive $drive).Free / 1GB, 1)
Write-Host "Workspace: $Workspace  (drive ${drive}: $freeGB GB free)"
if ($freeGB -lt 80) {
    Write-Warning ("~80 GB needed for weights + workspace; ${drive}: has $freeGB GB. " +
                   "Point -Workspace at a bigger drive or free space first.")
    if ($freeGB -lt 65) { throw "Not enough space for the 60 GB weights alone. Aborting." }
}

# --- Stage 1: repos (a few minutes) ----------------------------------------
if (-not (Test-Path "$Workspace\transformers")) {
    Write-Host "`n[1/4] Cloning transformers (Glimmer needs 5.15)..."
    git clone https://github.com/huggingface/transformers "$Workspace\transformers"
} else { Write-Host "`n[1/4] transformers already cloned - skipping." }

if (-not (Test-Path "$Workspace\optimum-intel")) {
    Write-Host "[2/4] Cloning optimum-intel (main has Glimmer support, PR #1924)..."
    git clone https://github.com/huggingface/optimum-intel "$Workspace\optimum-intel"
} else { Write-Host "[2/4] optimum-intel already cloned - skipping." }

# --- Stage 2: wheelhouse (offline venv rebuilds) ----------------------------
Write-Host "[3/4] Filling wheelhouse..."
New-Item -ItemType Directory -Force "$Workspace\wheels" | Out-Null
python -m pip download -d "$Workspace\wheels" `
    "transformers==5.15" opencv-python nncf openvino openvino-genai `
    einops accelerate safetensors huggingface_hub
# optimum-intel itself: installed from the git clone by export-glimmer.ps1 —
# the PyPI release predates Glimmer support.

# --- Stage 3: the 60 GB. Start last, resumes on re-run ----------------------
Write-Host "[4/4] Downloading meta-models/Muse-Glimmer-30B (~60 GB, resumable)..."
python -m pip install -q huggingface_hub
# Via the Python API, not hf.exe — corporate AV can block pip's exe launcher stubs.
python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-models/Muse-Glimmer-30B', local_dir=r'$Workspace\Muse-Glimmer-30B')"

Write-Host "`nAll stages complete. Re-run any time; finished stages are skipped."
Write-Host "Next: .\export-glimmer.ps1 -Workspace `"$Workspace`""
