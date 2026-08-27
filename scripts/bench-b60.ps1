# bench-b60.ps1 — fill the "Arc dGPU" column of the README speed table.
#
# Same protocol as the existing table entries (1 warmup + 5 runs, IQR outliers
# discarded) so the numbers slot in rather than sitting awkwardly beside them.
#
# Two hard-won details, both from a run that silently benchmarked the SAME
# model three times (2026-08-18):
#
#  1. KILL BY PORT, NOT BY PID. A venv created from the Microsoft Store Python
#     has a redirector at venv\Scripts\python.exe: Start-Process -PassThru
#     returns the LAUNCHER's pid ("python"), while the server runs as a child
#     ("python3.12"). Stopping the launcher leaves the server holding the port,
#     every later server fails to bind, and the benchmark quietly keeps talking
#     to the first one.
#
#  2. ASSERT THE MODEL. /health returning "ready" only proves *a* server is
#     there. Checking that it serves the model we asked for is what turns a
#     silent wrong-model run into a loud failure.
param(
    [string]$Repo    = 'C:\devel\aweussom\NoLlama',
    [string]$Root    = "$env:USERPROFILE\models",
    [string]$Label   = 'b60',
    [int]$Runs       = 5,
    [int]$Port       = 8000,
    [string]$LogFile = 'C:\devel\bench.log'
)

# Codex CLI's PATH junction breaks child processes under an SSH token.
$env:Path = (($env:Path -split ';') | Where-Object { $_ -notlike '*OpenAI*' -and $_ -notlike '*Codex*' }) -join ';'

$py = Join-Path $Repo 'venv\Scripts\python.exe'
function Log($m) {
    $l = "{0}  {1}" -f (Get-Date -Format 'HH:mm:ss'), $m
    Write-Output $l; Add-Content -Path $LogFile -Value $l
}

function Stop-Port([int]$p) {
    # Whoever actually owns the listening socket, whatever it is called.
    Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique |
        ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
    for ($i = 0; $i -lt 30; $i++) {
        if (-not (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue)) { return $true }
        Start-Sleep -Seconds 1
    }
    return $false
}

$models = @('SmolLM3-3B-int4-cw-ov', 'Qwen3-8B-int4-ov', 'Qwen3-30B-A3B-int4-ov')

Log "=== BENCH START  label=$Label runs=$Runs port=$Port ==="
if (-not (Stop-Port $Port)) { Log "FATAL: port $Port still held before we began"; exit 1 }

foreach ($m in $models) {
    $dir = Join-Path $Root $m
    if (-not (Test-Path $dir)) { Log "SKIP $m (not on disk)"; continue }

    Log "--- $m : starting server on GPU ---"
    Start-Process $py -ArgumentList "$Repo\nollama.py","--model-dir",$dir,
        "--device","GPU","--port","$Port","--ollama-port","0" `
        -WorkingDirectory $Repo -WindowStyle Hidden `
        -RedirectStandardOutput "C:\devel\server-$m.log" `
        -RedirectStandardError  "C:\devel\server-$m.err" | Out-Null

    $ready = $false
    for ($i = 1; $i -le 120; $i++) {          # 17 GB off NVMe + compile: allow 20 min
        Start-Sleep -Seconds 10
        try {
            $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 5
            if ($h.status -eq 'ready') { $ready = $true; break }
        } catch { }
    }
    if (-not $ready) {
        Log "FAIL $m : not ready after 20 min (see server-$m.log)"
        Stop-Port $Port | Out-Null
        continue
    }

    # Guard: is this OUR model, or a survivor from the previous iteration?
    $served = ($h.devices.PSObject.Properties | ForEach-Object { $_.Value.model }) -join ','
    $expect = ($m -replace '-ov$', '') -replace '-int4$', ''
    if ($served -notlike "*$($m.Split('-')[0])*") {
        Log "FAIL $m : server is serving '$served', not this model — aborting"
        Log "        (a previous server almost certainly still holds port $Port)"
        Stop-Port $Port | Out-Null
        continue
    }

    Log "$m ready after $($i * 10)s, serving '$served' — benchmarking"
    & $py "$Repo\benchmark.py" --llm-only --label $Label --runs $Runs 2>&1 |
        ForEach-Object { Log $_ }

    if (-not (Stop-Port $Port)) { Log "WARN: port $Port still held after stop" }
    Start-Sleep -Seconds 10       # let the driver release VRAM
    Log "--- $m done ---"
}
Log '=== BENCH ALL DONE ==='
