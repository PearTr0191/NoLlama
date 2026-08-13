#requires -Version 7.0
<#
Set up the python venv for NoLlama's optimum backend — the runtime for
brand-new architectures that openvino_genai cannot serve yet (Muse Glimmer;
see README "Brand-new architectures"). Serving only: for EXPORTING such a
model see scripts\glimmer-export\.

What lands in the venv, and why this exact order:
  - openvino + openvino-genai + flask + pillow: NoLlama's normal serving deps
  - torch, CPU build: the optimum runtime rides on it; CUDA is dead weight here
  - optimum-intel from git (muse_glimmer support is not in a release yet)
  - transformers from git, LAST: optimum-intel pins transformers<5.6 and pip
    enforces the pin by downgrading whatever is present; Glimmer only exists
    on transformers main, so we override the pin afterwards (pip warns, obeys)

Needs git on PATH (pip installs two packages straight from GitHub).
Idempotent: re-running on a healthy venv exits early.

  .\install-optimum.ps1                       # creates ./venv-optimum
  .\install-optimum.ps1 -Python python3.12    # pick the python to seed from
Then serve with it:
  Windows: venv-optimum\Scripts\python.exe nollama.py --model-dir <dir> --device CPU
  Linux:   venv-optimum/bin/python nollama.py --model-dir <dir> --device CPU
#>
param(
    [string]$Python = 'python',
    [string]$VenvDir = "$PSScriptRoot\venv-optimum",
    # Pin these to a commit/tag if main breaks; defaults track upstream.
    [string]$TransformersRef = 'main',
    [string]$OptimumIntelRef = 'main'
)

$ErrorActionPreference = 'Stop'
function Invoke-Step($desc, [scriptblock]$cmd) {
    Write-Host $desc
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "failed: $desc" }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'git is required on PATH (transformers/optimum-intel install from GitHub).'
}

$py = $IsWindows ? "$VenvDir\Scripts\python.exe" : "$VenvDir/bin/python"
if (-not (Test-Path $py)) {
    Invoke-Step "[1/4] Creating venv at $VenvDir..." { & $Python -m venv $VenvDir }
} else {
    Write-Host "[1/4] venv exists at $VenvDir - reusing."
}

# The assert mirrors export-glimmer.ps1: muse_glimmer registered proves the
# git transformers survived (a pip run that re-pinned it would drop the arch).
& $py -c "import optimum.intel, openvino_genai, flask; from transformers.models.auto.configuration_auto import CONFIG_MAPPING; assert 'muse_glimmer' in CONFIG_MAPPING" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host 'Venv already serves the optimum backend - nothing to do.'
    exit 0
}

Invoke-Step '[2/4] Serving deps (openvino, openvino-genai, flask, torch CPU)...' {
    & $py -m pip install --upgrade pip --quiet
    if ($LASTEXITCODE -ne 0) { return }
    & $py -m pip install openvino openvino-genai flask pillow einops accelerate
    if ($LASTEXITCODE -ne 0) { return }
    if ($IsWindows) {
        & $py -m pip install torch   # Windows default wheels are CPU-only
    } else {
        & $py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    }
}

Invoke-Step "[3/4] optimum-intel @ $OptimumIntelRef (git)..." {
    & $py -m pip install "optimum-intel @ git+https://github.com/huggingface/optimum-intel.git@$OptimumIntelRef"
}

Invoke-Step "[4/4] transformers @ $TransformersRef (git) - overrides the <5.6 pin..." {
    & $py -m pip install "transformers @ git+https://github.com/huggingface/transformers.git@$TransformersRef"
}

& $py -c "import openvino, transformers, optimum.intel; from transformers.models.auto.configuration_auto import CONFIG_MAPPING; assert 'muse_glimmer' in CONFIG_MAPPING, 'muse_glimmer arch missing - transformers pin was not overridden'; print('openvino', openvino.__version__); print('transformers', transformers.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'verification failed - see errors above' }

Write-Host ''
Write-Host 'Done. Serve an optimum-backend model with:'
if ($IsWindows) {
    Write-Host "  $VenvDir\Scripts\python.exe nollama.py --model-dir <model-dir> --device CPU"
} else {
    Write-Host "  $VenvDir/bin/python nollama.py --model-dir <model-dir> --device CPU"
}
Write-Host '(--device CPU is deliberate: no current Intel iGPU runs Glimmer correctly - see README.)'
