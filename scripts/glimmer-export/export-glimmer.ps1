#requires -Version 7.0
<#
Export Muse Glimmer 30B -> OpenVINO int4, per optimum-intel PR #1924.
Run oslo-prep.ps1 first (same -Workspace). Idempotent: finished stages are
skipped on re-run. The export stage itself is all-or-nothing (no mid-run
resume). Memory-hungry: loads 60 GB BF16 through RAM — on a 32 GB machine
expect hours of pagefile grinding; a 64-128 GB workstation is much happier.
Log: export.log in the workspace.

  .\export-glimmer.ps1
  .\export-glimmer.ps1 -Workspace D:\glimmer-port
#>
param(
    [string]$Workspace = 'C:\devel\aweussom\glimmer-port'
)

$ErrorActionPreference = 'Stop'
$src  = "$Workspace\Muse-Glimmer-30B"
$out  = "$Workspace\Muse-Glimmer-30B-int4-ov"
$venv = "$Workspace\venv-export"
$log  = "$Workspace\export.log"

if (-not (Test-Path "$src\model.safetensors.index.json")) {
    throw "Weights not found at $src - run oslo-prep.ps1 -Workspace `"$Workspace`" first."
}

# Keep the machine awake on AC for the duration (0 = never sleep).
powercfg /change standby-timeout-ac 0 2>$null

# --- Stage 1: venv + deps (resumable, ~minutes) ------------------------------
if (-not (Test-Path "$venv\Scripts\python.exe")) {
    Write-Host "[1/2] Creating export venv..."
    python -m venv $venv
}
$py = "$venv\Scripts\python.exe"
# muse_glimmer is NOT in any released transformers (the checkpoint was made by
# 5.15.0.dev0) — it must come from the git clone. Verify that, not just imports.
& $py -c "import optimum.intel; from transformers.models.auto.configuration_auto import CONFIG_MAPPING; assert 'muse_glimmer' in CONFIG_MAPPING" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[1/2] Installing deps (transformers from git clone - Glimmer needs main)..."
    & $py -m pip install --find-links "$Workspace\wheels" `
        opencv-python nncf openvino accelerate safetensors einops
    & $py -m pip install -e "$Workspace\optimum-intel"
    # LAST, deliberately: optimum-intel pins transformers<5.6, and pip enforces
    # that by downgrading whatever is present. Glimmer only exists on
    # transformers main, so override the pin after; pip warns but obeys.
    & $py -m pip install "$Workspace\transformers"
} else { Write-Host "[1/2] Deps already installed - skipping." }

# --- Stage 2: the grind (NOT resumable mid-run) ------------------------------
if (Test-Path "$out\openvino_language_model.xml") {
    Write-Host "[2/2] Export already complete at $out - nothing to do."
    exit 0
}
if (Test-Path $out) {
    Write-Host "[2/2] Incomplete previous export found - starting over."
    Remove-Item -Recurse -Force $out
}
Write-Host "[2/2] Exporting int4 (recipe from optimum-intel PR #1924)."
Write-Host "      Heavy RAM/pagefile use during the BF16 load; log: $log"
Write-Host "      If it dies with a MemoryError/'paging file too small':"
Write-Host "      set the pagefile ~150 GB (admin), reboot, re-run."

& $py "$PSScriptRoot\run-export.py" $src $out 2>&1 | Tee-Object -FilePath $log

if (Test-Path "$out\openvino_language_model.xml") {
    Write-Host "`nDone. int4 IR at: $out"
    Get-ChildItem $out | Sort-Object Length -Descending |
        Select-Object Name, @{n='GB';e={[math]::Round($_.Length/1GB,2)}} -First 8
    Write-Host "Upload when happy:  hf upload aweussom/Muse-Glimmer-30B-int4-ov `"$out`" ."
} else {
    Write-Warning "Export did not produce the expected IR - check $log (tail below)."
    Get-Content $log -Tail 20
}
