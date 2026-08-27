#requires -Version 7.0
# start.ps1 — NoLlama launcher
# Activates the venv and runs nollama.py. nollama.py prints its own
# device-detection, per-model loading progress, and the "NoLlama ready"
# banner with the URL — the launcher does not poll /health or auto-open
# the browser. Open the URL from the banner yourself.
#
# Args are set by install.ps1 in the generated start.ps1.

param(
    [string]$ServerArgs = "",
    # Which venv to launch from. install.ps1 bakes this into the generated
    # start.ps1: "venv" normally, "venv-nightly" for an -Nightly install.
    # Defaults to "venv" so a start.ps1 generated before this parameter
    # existed keeps working unchanged.
    [string]$VenvName = "venv",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Activate venv (Scripts on Windows, bin on POSIX)
$VenvBinDir = if ($IsWindows) { "Scripts" } else { "bin" }
$ActivatePath = Join-Path $ScriptDir $VenvName $VenvBinDir "Activate.ps1"
if (-not (Test-Path $ActivatePath)) {
    Write-Host "ERROR: no venv at $(Join-Path $ScriptDir $VenvName)" -ForegroundColor Red
    Write-Host "  Run .\install.ps1$(if ($VenvName -eq 'venv-nightly') { ' -Nightly' }) to create it." -ForegroundColor Yellow
    exit 1
}
& $ActivatePath

$AllArgs = @((Join-Path $ScriptDir "nollama.py"))
if ($ServerArgs) {
    $AllArgs += $ServerArgs.Split(" ", [StringSplitOptions]::RemoveEmptyEntries)
}
if ($ExtraArgs) {
    $AllArgs += $ExtraArgs  # user overrides from the start.ps1 command line, e.g. --port 8091
}

& python @AllArgs
