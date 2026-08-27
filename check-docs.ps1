# check-docs.ps1 — the canonical implementation.
#
# Verifies that functions are documented and that logic diagrams still match the
# code they claim to cover. Run before committing.
#
# CANONICAL means: when this and check_docs.py disagree, THIS is right and the
# port gets fixed. This is the version that was debugged against a real
# codebase. tests/test_parity.py runs both and diffs the output, so a
# disagreement is a caught failure rather than a discovered one.
#
# Configure the repo by editing docs-toolkit.json, NOT this file. Nothing
# project-specific belongs in here — that config file is what makes the two
# implementations comparable at all.
#
# Read conventions/DOCUMENTING-A-CODEBASE.md first. It is the order of
# operations and the traps, and it is shorter than this script.
#
# Runs on PowerShell 7, which is cross-platform: `pwsh ./check-docs.ps1` works
# on Linux and macOS as well as Windows. No modules required.
#
# Four checks:
#   1. Docstrings          - every def/class in an enforced file has one.
#   2. Doc blocks + syntax - same for brace languages, plus a free syntax gate.
#   3. covers: integrity   - every function named in a diagram's `%% covers:`
#                            header still exists in the file it names.
#   4. Diagram staleness   - which diagrams your working diff has invalidated.
#
# Usage:
#   .\check-docs.ps1                # checks 1-3 on enforced files + check 4 on the diff
#   .\check-docs.ps1 -All           # checks 1-3 only, ignore git state
#   .\check-docs.ps1 -Deep          # check 4 narrowed to changed *functions*, not files
#   .\check-docs.ps1 -Audit         # coverage for EVERY source file, enforced or not
#   .\check-docs.ps1 -Render        # parse every .mmd AND every inline mermaid fence
#   .\check-docs.ps1 -Quiet         # exit code only, no output (for a hook)
#
# Exit code is 1 when any check reports a finding, 0 otherwise.
#
# HONEST NOTE, and do not paper over it: check 4 tells you a diagram *might*
# be stale. It cannot tell you it *is* - only a human reading the diagram can.
# If someone rubber-stamps the output, the gate stops working. That is design
# intent, not a bug to fix.
#
# WHAT THIS SCRIPT CANNOT PROTECT, AND WHY IT MATTERS MOST:
# It counts whether a docstring EXISTS. It has no idea whether the docstring
# still says anything true, and it cannot stop a future rewrite from quietly
# replacing hard-won evidence with fluent paraphrase.
#
# The countermeasure is not tooling, it is a convention: epistemic tags.
# Mark the statements that came from a spec, a production run, or domain
# knowledge - the ones that CANNOT be recovered from the code - as
# [DOCUMENTED] / [OBSERVED <date>] / [INFERRED] / [GUESS], and write into your
# agent/contributor doc that a tagged line is evidence rather than phrasing:
# never reworded or dropped as part of an unrelated edit, never promoted
# without a run to justify it.
#
# The failure mode is OVER-tagging, not under-tagging. Tagging
# "returns a list of ints" is syntactically fine and destroys the whole point.
# Review the first regenerated file for over-tagging first.
#
# Full rationale and a worked example: conventions/DOCUMENTING-A-CODEBASE.md
# -> "Epistemic tagging - the part that survives a rewrite".
#
# Dependencies: git, plus whatever interpreter each language adapter shells to
# (python for the AST scan; node for the optional JS syntax gate - both
# skippable). Deliberately no lint/doc-tool dependency; see
# DOCUMENTING-A-CODEBASE.md on why that was rejected rather than assumed.

[CmdletBinding()]
param(
    [switch]$All,
    [switch]$Deep,
    [switch]$Audit,
    [switch]$Render,
    [switch]$Quiet,
    [switch]$NoOpen,
    [string]$Root
)

$ErrorActionPreference = 'Stop'

# --- Repo root ---------------------------------------------------------------
# Nearest ancestor holding docs-toolkit.json, else one holding .git, else the
# script's own directory. The config file wins because it is what the two
# implementations share; .git is the fallback for a repo not yet configured.
function Find-RepoRoot {
    param([string]$start)
    $d = Resolve-Path $start
    for ($p = $d; $p; $p = (Split-Path $p -Parent)) {
        if (Test-Path (Join-Path $p 'docs-toolkit.json')) { return $p }
        if ((Split-Path $p -Parent) -eq $p) { break }
    }
    for ($p = $d; $p; $p = (Split-Path $p -Parent)) {
        if (Test-Path (Join-Path $p '.git')) { return $p }
        if ((Split-Path $p -Parent) -eq $p) { break }
    }
    return $d
}

$root = if ($Root) { (Resolve-Path $Root).Path }
        else { (Find-RepoRoot (Get-Location)).ToString() }

# --- Configuration: SHARED with check_docs.py --------------------------------
# The ratchet, the skip list and the diagram dirs all live in docs-toolkit.json,
# NOT in this script. That file existing is what makes the two implementations
# comparable at all — hardcoding the lists here would guarantee they drift, which
# is the exact failure this whole toolkit is about.
#
# Configure the repo by editing docs-toolkit.json. Do not add project paths here.
#
# A missing config is not an error: an unconfigured repo still runs, with empty
# enforced lists, which is the intended starting state. A MALFORMED config is an
# error, because falling back to defaults would hide a typo'd path list and
# report a false clean.

$DEFAULT_SKIP = @('.git', '.venv', 'venv', 'node_modules', '__pycache__',
                  'site-packages', 'dist', 'build', '.pytest_cache', '.mypy_cache')

$cfgPath = Join-Path $root 'docs-toolkit.json'
$ENFORCED_PY = @(); $ENFORCED_JS = @(); $ENFORCED_GO = @()
$SKIP_DIRS = $DEFAULT_SKIP
$DIAGRAM_DIRS = @('docs')
$SCAN_MD_DIAGRAMS = $true
$JS_SYNTAX_GATE = $true

if (Test-Path $cfgPath) {
    try {
        $cfg = Get-Content $cfgPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        Write-Host "docs-toolkit.json is unreadable: $($_.Exception.Message)"
        exit 2
    }
    if ($cfg.enforced) {
        if ($null -ne $cfg.enforced.python)     { $ENFORCED_PY = @($cfg.enforced.python) }
        if ($null -ne $cfg.enforced.javascript) { $ENFORCED_JS = @($cfg.enforced.javascript) }
        if ($null -ne $cfg.enforced.go)         { $ENFORCED_GO = @($cfg.enforced.go) }
    }
    if ($cfg.skip_dirs)    { $SKIP_DIRS = @($cfg.skip_dirs) }
    if ($cfg.diagram_dirs) { $DIAGRAM_DIRS = @($cfg.diagram_dirs) }
    if ($null -ne $cfg.scan_markdown_diagrams) { $SCAN_MD_DIAGRAMS = [bool]$cfg.scan_markdown_diagrams }
    if ($null -ne $cfg.js_syntax_gate)         { $JS_SYNTAX_GATE = [bool]$cfg.js_syntax_gate }
}

# Component match, not substring: a file called build_tools.py must not be
# skipped because 'build' is in the list. Getting this wrong silently drops real
# files from the audit.
function Test-Skipped {
    param([string]$relPath)
    foreach ($part in ($relPath -replace '\\', '/').Split('/')) {
        if ($SKIP_DIRS -contains $part) { return $true }
    }
    return $false
}

# --- Output helpers ----------------------------------------------------------
$script:findings = 0

function Write-Line {
    param([string]$text = '')
    if (-not $Quiet) { Write-Host $text }
}

function Write-Finding {
    param([string]$text)
    $script:findings++
    if (-not $Quiet) { Write-Host "  $text" }
}

function Write-Section {
    param([string]$title)
    Write-Line ''
    Write-Line "=== $title ==="
}

# --- Python scanning ---------------------------------------------------------
# Shelled out to python's own `ast` rather than regexed, because decorators,
# multi-line signatures and nested classes all defeat a regex and this repo has
# all three. Emits one pipe-separated record per definition.
#
# Nested defs (closures inside functions) are NOT reported — they are usually
# three lines inside an already-documented parent. Module-level functions and
# direct class methods are.

$pyScanSource = @'
import ast, sys

def emit(path, node, kind):
    doc = ast.get_docstring(node)
    end = getattr(node, "end_lineno", node.lineno)
    print("%s|%d|%d|%s|%s|%s" % (
        path, node.lineno, end, node.name, kind,
        "HAS" if doc else "MISS"))

for path in sys.argv[1:]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except (OSError, SyntaxError) as exc:
        print("%s|0|0|-|ERROR|%s" % (path, exc))
        continue
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            emit(path, node, "func")
        elif isinstance(node, ast.ClassDef):
            emit(path, node, "class")
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    emit(path, sub, "method")
'@

$pyScanPath = Join-Path $env:TEMP 'check-docs-ast-scan.py'
[System.IO.File]::WriteAllText($pyScanPath, $pyScanSource,
    (New-Object System.Text.UTF8Encoding($false)))

function Get-PyDefs {
    param([string[]]$paths)
    $existing = @($paths | Where-Object { Test-Path (Join-Path $root $_) })
    if ($existing.Count -eq 0) { return @() }
    # @() is load-bearing: a single-element pipeline collapses to a scalar
    # string, and splatting a scalar string passes it one CHARACTER per
    # argument. Scanning one file then hands python 51 one-char filenames and
    # every definition comes back as an ERROR record.
    $full = @($existing | ForEach-Object { Join-Path $root $_ })

    # Batched because Windows caps a command line at ~32k characters. Blowing
    # that cap does not fail cleanly — PowerShell reports "StandardOutputEncoding
    # is only supported when standard output is redirected", which points
    # nowhere near the real cause. 200 paths per batch leaves plenty of margin.
    $out = @()
    for ($i = 0; $i -lt $full.Count; $i += 200) {
        $batch = $full[$i..([math]::Min($i + 199, $full.Count - 1))]
        $out += & python $pyScanPath @batch 2>&1
    }
    $defs = @()
    foreach ($line in $out) {
        $parts = "$line" -split '\|'
        if ($parts.Count -lt 6) { continue }
        $rel = ($parts[0] -replace [regex]::Escape($root + '\'), '') -replace '\\', '/'
        $defs += [pscustomobject]@{
            File    = $rel
            Line    = [int]$parts[1]
            EndLine = [int]$parts[2]
            Name    = $parts[3]
            Kind    = $parts[4]
            HasDoc  = ($parts[5] -eq 'HAS')
        }
    }
    return $defs
}

# --- JavaScript scanning -----------------------------------------------------
# Regex, deliberately crude. No npm in this repo and none coming, so there is no
# real parser available. False positives are preferred to silence — the same
# trade the memory-discovery pass makes. If it flags something that genuinely
# needs no block, document it anyway or rename it to something self-evident.
#
# Recognised shapes:
#   function name(              async function name(
#   const name = function       const name = (a, b) =>      const name = async (
#   name: function(             name(arg) {   (object/IIFE-returned methods)

function Get-JsDefs {
    param([string]$relPath)
    $full = Join-Path $root $relPath
    if (-not (Test-Path $full)) { return @() }
    $lines = [System.IO.File]::ReadAllLines($full)

    $patterns = @(
        '^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(',
        '^\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function\s*\(|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)',
        '^\s*([A-Za-z_$][\w$]*)\s*:\s*(?:async\s+)?function\s*\('
    )

    $defs = @()
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $name = $null
        foreach ($pat in $patterns) {
            $m = [regex]::Match($lines[$i], $pat)
            if ($m.Success) { $name = $m.Groups[1].Value; break }
        }
        if (-not $name) { continue }

        # Exempt: a single-expression arrow whose whole body is on the same
        # line — `const toRad = d => d * Math.PI / 180;`. The convention in
        # CLAUDE.md excuses trivial helpers whose name says everything, and a
        # six-line block above a one-line conversion is the noise that trains
        # people to stop reading the blocks that matter.
        #
        # Detected as: an arrow with content after it, not opening a body brace.
        # A multi-line arrow (`=> {` or a bare trailing `=>`) is NOT exempt.
        if ($lines[$i] -match '=>\s*[^\s{]' -and $lines[$i].TrimEnd() -notmatch '\{$') {
            continue
        }

        # Documented == the nearest preceding non-blank line closes a block
        # comment. Bare `//` lines do not count: the convention is /** */ so
        # that a doc block is visually distinct from an inline aside.
        $hasDoc = $false
        for ($j = $i - 1; $j -ge 0; $j--) {
            $prev = $lines[$j].Trim()
            if ($prev -eq '') { continue }
            $hasDoc = $prev.EndsWith('*/')
            break
        }

        $defs += [pscustomobject]@{
            File    = $relPath
            Line    = $i + 1
            EndLine = 0          # filled in below
            Name    = $name
            Kind    = 'func'
            HasDoc  = $hasDoc
        }
    }

    # Approximate each function's extent as "until the next one starts". Good
    # enough for -Deep hunk overlap; nothing else depends on it.
    for ($k = 0; $k -lt $defs.Count; $k++) {
        $defs[$k].EndLine = if ($k -lt $defs.Count - 1) { $defs[$k + 1].Line - 1 }
                            else { $lines.Count }
    }
    return $defs
}

# --- Go scanning -------------------------------------------------------------
# THE WORKED EXAMPLE OF AN ADAPTER. Adding a language is one Get-<Lang>Defs
# function plus one line in Get-DefsFor. The record contract each adapter must
# return is documented in conventions/DOCUMENTING-A-CODEBASE.md ->
# "Adapting the script to another language".
#
# Use a real parser when the language ships one (see Get-PyDefs). Regex is
# acceptable when it does not, as here and for JS — false positives are
# preferred to silence.
#
# Go convention differs from the other two: the doc comment is `//` lines
# immediately above the declaration, starting with the identifier's own name.
# We accept any `//` line, because enforcing GoDoc phrasing is a different job.

function Get-GoDefs {
    param([string]$relPath)
    $full = Join-Path $root $relPath
    if (-not (Test-Path $full)) { return @() }
    $lines = [System.IO.File]::ReadAllLines($full)

    $defs = @()
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $m = [regex]::Match($lines[$i],
            '^func\s+(?:\([^)]*\)\s*)?([A-Za-z_][\w]*)\s*[\(\[]')
        if (-not $m.Success) { continue }

        $hasDoc = $false
        for ($j = $i - 1; $j -ge 0; $j--) {
            $prev = $lines[$j].Trim()
            if ($prev -eq '') { continue }
            $hasDoc = $prev.StartsWith('//')
            break
        }

        $defs += [pscustomobject]@{
            File = $relPath; Line = $i + 1; EndLine = 0
            Name = $m.Groups[1].Value; Kind = 'func'; HasDoc = $hasDoc
        }
    }
    for ($k = 0; $k -lt $defs.Count; $k++) {
        $defs[$k].EndLine = if ($k -lt $defs.Count - 1) { $defs[$k + 1].Line - 1 }
                            else { $lines.Count }
    }
    return $defs
}

function Get-DefsFor {
    param([string]$relPath)
    if ($relPath -match '\.py$')      { return Get-PyDefs @($relPath) }
    elseif ($relPath -match '\.js$')  { return Get-JsDefs $relPath }
    elseif ($relPath -match '\.go$')  { return Get-GoDefs $relPath }
    else                              { return @() }
}

# --- Check 1 + 2: documentation coverage -------------------------------------

function Test-Coverage {
    Write-Section 'Checks 1-2: function documentation (enforced files)'

    if ($ENFORCED_PY.Count -eq 0 -and $ENFORCED_JS.Count -eq 0 -and $ENFORCED_GO.Count -eq 0) {
        Write-Line '  (enforced list is empty — nothing checked strictly yet.)'
        Write-Line '  Run with -Audit for a coverage report across the whole repo.'
        return
    }

    $pyDefs = Get-PyDefs $ENFORCED_PY
    foreach ($d in $pyDefs) {
        if ($d.Kind -eq 'ERROR') { Write-Finding "$($d.File): parse error"; continue }
        if ($d.Name -like 'test_*') { continue }
        if (-not $d.HasDoc) {
            Write-Finding "$($d.File):$($d.Line)  $($d.Name)  — no docstring"
        }
    }

    foreach ($rel in $ENFORCED_JS) {
        foreach ($d in (Get-JsDefs $rel)) {
            if (-not $d.HasDoc) {
                Write-Finding "$($d.File):$($d.Line)  $($d.Name)  — no /** */ block"
            }
        }
    }

    foreach ($rel in $ENFORCED_GO) {
        foreach ($d in (Get-GoDefs $rel)) {
            if (-not $d.HasDoc) {
                Write-Finding "$($d.File):$($d.Line)  $($d.Name)  — no // doc comment"
            }
        }
    }

    # Syntax-check the JS while we are here. `node --check` needs no packages,
    # so it costs nothing and needs no package.json. A documentation sweep's
    # realistic failure is an unbalanced /** */, which IS a syntax error.
    # Equivalents if you add a language: python -m py_compile, gofmt -e,
    # bash -n, tsc --noEmit. Cheap, and it catches the one thing that would
    # otherwise only surface at runtime. Toggle with js_syntax_gate in the config.
    # Silently skipped when node is absent.
    if ($ENFORCED_JS.Count -gt 0 -and $JS_SYNTAX_GATE -and (Get-Command node -ErrorAction SilentlyContinue)) {
        foreach ($rel in $ENFORCED_JS) {
            $full = Join-Path $root $rel
            if (-not (Test-Path $full)) { continue }
            $err = & node --check $full 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Finding "$rel — SYNTAX ERROR: $($err -join ' ')"
            }
        }
    }

    if ($script:findings -eq 0) { Write-Line '  All enforced files fully documented.' }
}

# --- Audit: coverage across everything, enforced or not ----------------------
# Does not count as a finding. This is the measuring instrument for planning a
# sweep, not a gate.

function Show-Audit {
    Write-Section 'Audit: documentation coverage, whole repo'

    # Skip list comes from docs-toolkit.json via Test-Skipped — matched per path
    # COMPONENT, not as a substring, so build_tools.py is not skipped because
    # 'build' is listed. A repo-root virtualenv holds thousands of third-party
    # files that are not yours to document, and including them also blows the
    # command-line length limit on Windows.
    $pyFiles = Get-ChildItem $root -Recurse -Filter '*.py' -File |
        ForEach-Object { ($_.FullName.Substring($root.Length + 1)) -replace '\\', '/' } |
        Where-Object { -not (Test-Skipped $_) } |
        Where-Object { $_ -notmatch '(^|/)tests?/' -and $_ -notmatch '__init__\.py$' } |
        Sort-Object

    $jsFiles = Get-ChildItem $root -Recurse -Filter '*.js' -File |
        ForEach-Object { ($_.FullName.Substring($root.Length + 1)) -replace '\\', '/' } |
        Where-Object { -not (Test-Skipped $_) } |
        Sort-Object

    $goFiles = Get-ChildItem $root -Recurse -Filter '*.go' -File |
        ForEach-Object { ($_.FullName.Substring($root.Length + 1)) -replace '\\', '/' } |
        Where-Object { -not (Test-Skipped $_) } |
        Sort-Object

    $allPyDefs = Get-PyDefs $pyFiles
    $totalHas = 0; $totalAll = 0

    Write-Line ''
    Write-Line '  Python'
    foreach ($f in $pyFiles) {
        $defs = @($allPyDefs | Where-Object { $_.File -eq $f -and $_.Kind -ne 'ERROR' })
        if ($defs.Count -eq 0) { continue }
        $has = @($defs | Where-Object { $_.HasDoc }).Count
        $totalHas += $has; $totalAll += $defs.Count
        $pct = [math]::Round(100 * $has / $defs.Count)
        $mark = if ($pct -eq 100) { 'ok  ' } else { '    ' }
        Write-Line ("  {0}{1,4}/{2,-4} {3,3}%  {4}" -f $mark, $has, $defs.Count, $pct, $f)
    }

    Write-Line ''
    Write-Line '  JavaScript'
    foreach ($f in $jsFiles) {
        $defs = @(Get-JsDefs $f)
        if ($defs.Count -eq 0) { continue }
        $has = @($defs | Where-Object { $_.HasDoc }).Count
        $totalHas += $has; $totalAll += $defs.Count
        $pct = [math]::Round(100 * $has / $defs.Count)
        $mark = if ($pct -eq 100) { 'ok  ' } else { '    ' }
        Write-Line ("  {0}{1,4}/{2,-4} {3,3}%  {4}" -f $mark, $has, $defs.Count, $pct, $f)
    }

    if ($goFiles) {
        Write-Line ''
        Write-Line '  Go'
        foreach ($f in $goFiles) {
            $defs = @(Get-GoDefs $f)
            if ($defs.Count -eq 0) { continue }
            $has = @($defs | Where-Object { $_.HasDoc }).Count
            $totalHas += $has; $totalAll += $defs.Count
            $pct = [math]::Round(100 * $has / $defs.Count)
            $mark = if ($pct -eq 100) { 'ok  ' } else { '    ' }
            Write-Line ("  {0}{1,4}/{2,-4} {3,3}%  {4}" -f $mark, $has, $defs.Count, $pct, $f)
        }
    }

    if ($totalAll -gt 0) {
        Write-Line ''
        Write-Line ("  Total: {0}/{1} ({2}%)" -f $totalHas, $totalAll,
            [math]::Round(100 * $totalHas / $totalAll))
    }
}

# --- Diagram parsing ---------------------------------------------------------
# Each .mmd declares its own coverage in its %% header:
#   %% covers: src/app.py:some_function,another_function
# One line per source file, comma-separated names, no spaces. Nothing but this
# script reads them.

function Get-Diagrams {
    $out = @()
    # Diagram dirs and the skip list both come from docs-toolkit.json. Sorted so
    # output ordering is stable across platforms and between the two
    # implementations — the parity test compares line by line.
    $dirs = @()
    foreach ($dirName in $DIAGRAM_DIRS) {
        $dirs += Get-ChildItem $root -Recurse -Directory -Filter $dirName -ErrorAction SilentlyContinue |
            Where-Object { -not (Test-Skipped ($_.FullName.Substring($root.Length + 1))) }
    }
    $dirs = $dirs | Sort-Object FullName

    foreach ($dir in $dirs) {
        foreach ($mmd in (Get-ChildItem $dir.FullName -Filter '*.mmd' -File | Sort-Object Name)) {
            $rel = ($mmd.FullName.Substring($root.Length + 1)) -replace '\\', '/'
            $covers = @()
            foreach ($line in [System.IO.File]::ReadAllLines($mmd.FullName)) {
                if ($line -notmatch '^\s*%%\s*covers:\s*(.+)$') { continue }
                $spec = $Matches[1].Trim()
                $idx = $spec.LastIndexOf(':')
                if ($idx -lt 1) { continue }
                $covers += [pscustomobject]@{
                    SourceFile = $spec.Substring(0, $idx).Trim()
                    Functions  = @($spec.Substring($idx + 1).Split(',') |
                                   ForEach-Object { $_.Trim() } |
                                   Where-Object { $_ })
                }
            }
            $out += [pscustomobject]@{ Path = $rel; Covers = $covers }
        }
    }
    return $out
}

# --- Check 3: covers: integrity ----------------------------------------------
# This is the check that actually bites. Rename a function and the diagram's
# covers: line goes red on the next run, which is the only automatic signal
# that a hand-drawn diagram has fallen behind.

function Test-CoversIntegrity {
    param($diagrams)
    Write-Section 'Check 3: covers: headers point at code that exists'

    if ($diagrams.Count -eq 0) {
        Write-Line '  (no .mmd files found under any docs/ directory.)'
        return
    }

    $defCache = @{}
    $checked = 0

    foreach ($diag in $diagrams) {
        if ($diag.Covers.Count -eq 0) {
            Write-Finding "$($diag.Path)  — no `%% covers:` header"
            continue
        }
        foreach ($c in $diag.Covers) {
            if (-not (Test-Path (Join-Path $root $c.SourceFile))) {
                Write-Finding "$($diag.Path)  — covers missing file: $($c.SourceFile)"
                continue
            }
            if (-not $defCache.ContainsKey($c.SourceFile)) {
                $defCache[$c.SourceFile] = @(Get-DefsFor $c.SourceFile | ForEach-Object { $_.Name })
            }
            $known = $defCache[$c.SourceFile]
            foreach ($fn in $c.Functions) {
                $checked++
                if ($known -notcontains $fn) {
                    Write-Finding "$($diag.Path)  — $($c.SourceFile) has no `'$fn`' (renamed or removed?)"
                }
            }
        }
    }

    if ($script:findings -eq 0) {
        Write-Line "  $checked function reference(s) across $($diagrams.Count) diagram(s) all resolve."
    }
}

# --- Check 4: staleness against the working diff -----------------------------

function Get-ChangedFiles {
    $changed = @()
    $changed += & git -C $root diff --name-only HEAD 2>$null
    $changed += & git -C $root diff --name-only --cached 2>$null
    return @($changed | Where-Object { $_ } | Sort-Object -Unique)
}

# Changed line ranges per file, from a zero-context diff. Used by -Deep to skip
# diagrams whose covered functions weren't actually touched.
function Get-ChangedRanges {
    param([string]$relPath)
    $ranges = @()
    $diff = & git -C $root diff -U0 HEAD -- $relPath 2>$null
    $diff += & git -C $root diff -U0 --cached -- $relPath 2>$null
    foreach ($line in $diff) {
        if ($line -notmatch '^@@ -\S+ \+(\d+)(?:,(\d+))? @@') { continue }
        $start = [int]$Matches[1]
        $count = if ($Matches[2]) { [int]$Matches[2] } else { 1 }
        # A zero-length new-side hunk (pure deletion) still points at a real
        # line; treat it as one line so the deletion is attributed somewhere.
        if ($count -eq 0) { $count = 1 }
        # $end is computed BEFORE the array is built on purpose: PowerShell's
        # unary comma binds looser than arithmetic, so `,@($a, $a + $n - 1)`
        # parses as `(,@($a, $a + $n)) - 1` and dies with op_Subtraction on
        # Object[]. Named fields sidestep the whole question.
        $end = $start + $count - 1
        $ranges += [pscustomobject]@{ From = $start; To = $end }
    }
    return $ranges
}

function Test-Staleness {
    param($diagrams)
    $mode = if ($Deep) { 'changed functions' } else { 'changed files' }
    Write-Section "Check 4: diagrams to review ($mode)"

    $changed = Get-ChangedFiles
    if ($changed.Count -eq 0) {
        Write-Line '  Working tree clean — nothing to review.'
        return
    }

    $hits = @{}
    foreach ($diag in $diagrams) {
        foreach ($c in $diag.Covers) {
            if ($changed -notcontains $c.SourceFile) { continue }

            $reason = $c.SourceFile
            if ($Deep) {
                $ranges = Get-ChangedRanges $c.SourceFile
                if ($ranges.Count -eq 0) { continue }
                $defs = @(Get-DefsFor $c.SourceFile)
                $touched = @()
                foreach ($fn in $c.Functions) {
                    $d = $defs | Where-Object { $_.Name -eq $fn } | Select-Object -First 1
                    if (-not $d) { continue }
                    foreach ($r in $ranges) {
                        if ($r.From -le $d.EndLine -and $r.To -ge $d.Line) { $touched += $fn; break }
                    }
                }
                if ($touched.Count -eq 0) { continue }
                $reason = "$($c.SourceFile) → $($touched -join ', ')"
            }

            if (-not $hits.ContainsKey($diag.Path)) { $hits[$diag.Path] = @() }
            $hits[$diag.Path] += $reason
        }
    }

    if ($hits.Count -eq 0) {
        Write-Line '  No diagram covers anything in this diff.'
        if (-not $Deep) { Write-Line '  (Nothing changed under a covers: header.)' }
        return
    }

    foreach ($path in ($hits.Keys | Sort-Object)) {
        Write-Finding "$path"
        foreach ($r in ($hits[$path] | Sort-Object -Unique)) { Write-Line "      $r" }
    }
    Write-Line ''
    Write-Line '  Update these in the same commit, or say in the commit message why not.'
}

# --- Render check: does every diagram actually parse? ------------------------
# There is no offline Mermaid linter without npm. So: generate a self-contained
# page that calls mermaid.parse() on every diagram and reports the failures,
# then open it. Costs a browser and one CDN fetch. If npm is already present
# in your repo, `mmdc --parse` removes the browser step.
#
# Not theoretical. First run against eight hand-written diagrams found NINE
# real parse errors, in files that had been read carefully and looked fine:
#   · a bare `%%` separator line survives Mermaid's comment stripper (it wants
#     at least one character after the %%) and then collides with the graph
#     header. Always put text after %% — `%% ---` works.
#   · `A -. .-> B` is not an unlabelled dotted edge. `A -.-> B` is.

function Invoke-RenderCheck {
    # Collected as {Label, Source} so standalone .mmd files and inline
    # ```mermaid fences in markdown go through the same parser.
    $diagrams = @()

    $dirs = Get-ChildItem $root -Recurse -Directory -Filter 'docs' -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '[\\/](\.git|node_modules)[\\/]' }
    foreach ($dir in $dirs) {
        foreach ($f in (Get-ChildItem $dir.FullName -Filter '*.mmd' -File | Sort-Object Name)) {
            $rel = ($f.FullName.Substring($root.Length + 1)) -replace '\\', '/'
            $diagrams += [pscustomobject]@{
                Label  = $rel
                Source = [System.IO.File]::ReadAllText($f.FullName)
            }
        }
    }

    # Inline fences too: architecture-level diagrams embedded in markdown break
    # exactly as easily as standalone .mmd files, and nothing else would ever
    # parse them. Gated on scan_markdown_diagrams in docs-toolkit.json.
    if ($SCAN_MD_DIAGRAMS) {
        foreach ($md in (Get-ChildItem $root -Recurse -Filter '*.md' -File |
                         Where-Object { -not (Test-Skipped ($_.FullName.Substring($root.Length + 1))) } |
                         Sort-Object FullName)) {
            $text = [System.IO.File]::ReadAllText($md.FullName)
            if ($text -notmatch '(?m)^\s*```mermaid') { continue }
            $rel = ($md.FullName.Substring($root.Length + 1)) -replace '\\', '/'
            $n = 0
            foreach ($m in [regex]::Matches($text, '(?ms)^[ \t]*```mermaid[ \t]*\r?\n(.*?)^[ \t]*```')) {
                $n++
                $diagrams += [pscustomobject]@{
                    Label  = "$rel (inline #$n)"
                    Source = $m.Groups[1].Value
                }
            }
        }
    }

    if ($diagrams.Count -eq 0) {
        Write-Line 'No Mermaid diagrams found.'
        return
    }

    $sb = [System.Text.StringBuilder]::new()
    [void]$sb.AppendLine('<!doctype html><html><head><meta charset="utf-8">')
    [void]$sb.AppendLine('<title>diagram render check</title>')
    [void]$sb.AppendLine('<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>')
    [void]$sb.AppendLine('<style>body{font:14px system-ui;margin:2rem}h1{position:sticky;top:0;background:#fff;padding:.5rem 0}#errors{color:#a00;white-space:pre-wrap}</style>')
    [void]$sb.AppendLine('</head><body><h1 id="verdict">rendering…</h1><pre id="errors"></pre>')
    foreach ($d in $diagrams) {
        $esc = $d.Source -replace '&', '&amp;' -replace '<', '&lt;' -replace '>', '&gt;'
        $lbl = $d.Label -replace '"', '&quot;'
        [void]$sb.AppendLine("<h2>$lbl</h2><pre class=""mermaid"" data-src=""$lbl"">$esc</pre>")
    }
    [void]$sb.AppendLine(@'
<script type="module">
  const errs = [];
  mermaid.initialize({ startOnLoad: false });
  for (const p of document.querySelectorAll('pre.mermaid')) {
    try { await mermaid.parse(p.textContent); }
    catch (e) { errs.push(p.dataset.src + ' :: ' + (e && e.message || e)); }
  }
  const n = document.querySelectorAll('pre.mermaid').length;
  document.getElementById('verdict').textContent =
    errs.length ? ('PARSE ERRORS: ' + errs.length + ' of ' + n) : ('ALL OK (' + n + ' diagrams)');
  document.getElementById('errors').textContent = errs.join('\n\n');
  await mermaid.run({ querySelector: 'pre.mermaid' });
</script></body></html>
'@)

    $out = Join-Path ([System.IO.Path]::GetTempPath()) 'diagram-render-check.html'
    [System.IO.File]::WriteAllText($out, $sb.ToString(), (New-Object System.Text.UTF8Encoding($false)))
    Write-Line "Wrote render harness for $($diagrams.Count) diagram(s):"
    Write-Line "  $out"
    Write-Line ''
    Write-Line 'Opening in the default browser. The heading reads ALL OK or lists'
    Write-Line 'the parse errors; the diagrams themselves render below it.'
    if (-not $NoOpen) { Start-Process $out }
}

# --- Run ---------------------------------------------------------------------

if ($Render) {
    Invoke-RenderCheck
    exit 0
}

if ($Audit) {
    Show-Audit
    exit 0
}

$diagrams = Get-Diagrams

Test-Coverage
Test-CoversIntegrity $diagrams
if (-not $All) { Test-Staleness $diagrams }

Write-Line ''
if ($script:findings -eq 0) {
    Write-Line 'check-docs: clean.'
    exit 0
} else {
    Write-Line "check-docs: $($script:findings) finding(s)."
    exit 1
}
