#!/usr/bin/env python3
"""check_docs.py — port of check-docs.ps1. Same checks, same output.

check-docs.ps1 is canonical: it is the version that was debugged against a real
codebase. This port exists so the toolkit works where PowerShell is not
installed, and it is held to the canonical script's OUTPUT, byte for byte, by
tests/test_parity.py.

If you change behaviour, change both and run the parity test. If you cannot
make them agree, the canonical script wins and this one gets fixed.

Four checks:
  1. Docstrings          — every def/class in an enforced file has one.
  2. Doc blocks + syntax — same for brace languages, plus a free syntax gate.
  3. covers: integrity   — every function named in a diagram's `%% covers:`
                           header still exists in the file it names.
  4. Diagram staleness   — which diagrams your working diff has invalidated.

Usage:
  python check_docs.py                # checks 1-3 on enforced files + 4 on the diff
  python check_docs.py --all          # checks 1-3 only, ignore git state
  python check_docs.py --deep         # check 4 narrowed to changed *functions*
  python check_docs.py --audit        # coverage for EVERY source file
  python check_docs.py --render       # parse every diagram, incl. markdown fences
  python check_docs.py --quiet        # exit code only

Exit code is 1 when any check reports a finding, 0 otherwise.

HONEST NOTE, and do not paper over it: check 4 tells you a diagram *might* be
stale. It cannot tell you it *is* — only a human reading the diagram can. If
someone rubber-stamps the output, the gate stops working. That is design intent,
not a bug to fix.

WHAT THIS CANNOT PROTECT: it counts whether a docstring EXISTS. It has no idea
whether the docstring still says anything true, and it cannot stop a rewrite
replacing hard-won evidence with fluent paraphrase. The countermeasure is
epistemic tags — see conventions/CLAUDE-snippet.md.

Configuration is read from docs-toolkit.json at the repo root, SHARED with the
PowerShell version. Nothing project-specific belongs in this file.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_NAME = "docs-toolkit.json"

# Force UTF-8 on stdout. Windows defaults a PIPED stdout to the ANSI codepage
# (cp1252), which turns the em-dash in every finding into a replacement
# character — so redirecting output to a file or through another tool silently
# corrupts it. Harmless where UTF-8 is already the default.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_CONFIG = {
    "enforced": {"python": [], "javascript": [], "go": []},
    "skip_dirs": [".git", ".venv", "venv", "node_modules", "__pycache__",
                  "site-packages", "dist", "build", ".pytest_cache",
                  ".mypy_cache"],
    "diagram_dirs": ["docs"],
    "scan_markdown_diagrams": True,
    "js_syntax_gate": True,
}


# --- Records ------------------------------------------------------------------

@dataclass
class Definition:
    """One documentable definition found by a language adapter.

    The contract every adapter must return. `end_line` is only used by --deep
    hunk overlap, so an approximation is acceptable where a real parser is not
    available.
    """

    file: str
    line: int
    end_line: int
    name: str
    kind: str          # 'func' | 'class' | 'method' | 'ERROR'
    has_doc: bool


@dataclass
class Cover:
    """One `%% covers:` entry: a source file and the functions claimed in it."""

    source_file: str
    functions: list[str]


@dataclass
class Diagram:
    """A .mmd file and the coverage it declares."""

    path: str
    covers: list[Cover] = field(default_factory=list)


# --- Output -------------------------------------------------------------------

class Reporter:
    """Collects findings and prints them, matching check-docs.ps1 exactly.

    Why a class rather than prints: the parity test compares stdout, so every
    string lives in one place where it can be kept identical to the canonical
    script. Resist the urge to "improve" the wording on one side only.
    """

    def __init__(self, quiet: bool = False) -> None:
        """In: quiet, which suppresses output but NOT the finding count — the
        exit code still has to be right for a hook to act on."""
        self.quiet = quiet
        self.findings = 0

    def line(self, text: str = "") -> None:
        """Print one plain line. No-op under --quiet."""
        if not self.quiet:
            print(text)

    def section(self, title: str) -> None:
        """Print a blank line then a `=== title ===` header.

        The leading blank is part of the format the parity test compares; do not
        move it to the call sites.
        """
        self.line("")
        self.line(f"=== {title} ===")

    def finding(self, text: str) -> None:
        """Record a finding and print it indented two spaces.

        Counting is separate from printing on purpose, so --quiet still produces
        a correct exit code.
        """
        self.findings += 1
        if not self.quiet:
            print(f"  {text}")


# --- Config -------------------------------------------------------------------

def find_repo_root(start: Path) -> Path:
    """Locate the repo root: nearest ancestor with docs-toolkit.json, else .git.

    Why both: the config file is authoritative because it is what the two
    implementations share, but a repo that has not been configured yet should
    still work from anywhere inside it.

    In: any directory. Out: the root Path. Falls back to `start` when neither
    marker is found, so the tool degrades to "treat cwd as root" rather than
    guessing wildly.
    """
    for candidate in [start, *start.parents]:
        if (candidate / CONFIG_NAME).is_file():
            return candidate
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return start


def load_config(root: Path) -> dict:
    """Read docs-toolkit.json, merged over the defaults.

    In: repo root. Out: the merged config dict.

    A missing file is NOT an error — an unconfigured repo still runs, with empty
    enforced lists, which is exactly the intended starting state. A malformed
    file IS an error, because silently falling back to defaults would hide a
    typo'd path list and report a false clean.
    """
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    path = root / CONFIG_NAME
    if not path.is_file():
        return cfg
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{CONFIG_NAME} is unreadable: {exc}\n")
        raise SystemExit(2)
    for key, value in raw.items():
        if key.startswith("_") or key == "schema_version":
            continue
        if key == "enforced" and isinstance(value, dict):
            cfg["enforced"].update(value)
        else:
            cfg[key] = value
    return cfg


def is_skipped(rel_path: str, skip_dirs: list[str]) -> bool:
    """Is any path component a skipped directory?

    Component matching, not substring: a file called `build_tools.py` must not be
    skipped because `build` is in the list. Getting this wrong silently drops
    real files from the audit.
    """
    parts = Path(rel_path).parts
    return any(part in skip_dirs for part in parts)


# --- Adapters -----------------------------------------------------------------
#
# Adding a language is one function returning list[Definition], plus one line in
# defs_for(). The contract is documented on Definition.
#
# Use a real parser when the language ships one (see python_defs). Regex is
# acceptable when it does not — false positives are preferred to silence.


def python_defs(root: Path, rel_path: str) -> list[Definition]:
    """Module-level functions and classes (plus direct methods) via `ast`.

    Why ast rather than regex: decorators, multi-line signatures and nested
    classes all defeat a regex, and any real codebase has all three.

    Nested defs (closures inside functions) are deliberately NOT reported — they
    are usually three lines inside an already-documented parent.

    In: repo root and a repo-relative path. Out: one Definition per definition.
    A parse or read failure yields a single ERROR record rather than raising, so
    one broken file cannot abort a whole audit.
    """
    full = root / rel_path
    try:
        tree = ast.parse(full.read_text(encoding="utf-8"), filename=str(full))
    except (OSError, SyntaxError, ValueError) as exc:
        return [Definition(rel_path, 0, 0, "-", "ERROR", False)]

    out: list[Definition] = []

    def emit(node, kind: str) -> None:
        out.append(Definition(
            file=rel_path,
            line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
            name=node.name,
            kind=kind,
            has_doc=ast.get_docstring(node) is not None,
        ))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            emit(node, "func")
        elif isinstance(node, ast.ClassDef):
            emit(node, "class")
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    emit(sub, "method")
    return out


JS_PATTERNS = (
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\("),
    re.compile(
        r"^\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?"
        r"(?:function\s*\(|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)"
    ),
    re.compile(r"^\s*([A-Za-z_$][\w$]*)\s*:\s*(?:async\s+)?function\s*\("),
)

TRIVIAL_ARROW = re.compile(r"=>\s*[^\s{]")


def javascript_defs(root: Path, rel_path: str) -> list[Definition]:
    """Functions in a JS file, by regex. Deliberately crude.

    There is no parser available without adding an npm dependency, so this
    accepts false positives in exchange for never being silent. If it flags
    something that genuinely needs no block, document it anyway or add an
    exemption rule.

    In: repo root and a repo-relative path. Out: one Definition per match;
    empty list when the file is missing.

    Documented == the nearest preceding non-blank line ends with `*/`. Bare `//`
    lines do NOT count: the convention is a block comment so a contract is
    visually distinct from an inline aside.

    Exempt: a single-expression arrow whose whole body is on the same line
    (`const toRad = d => d * Math.PI / 180`). A six-line block over a one-line
    conversion is the noise that trains people to stop reading blocks.
    `end_line` is approximated as "until the next definition starts".
    """
    full = root / rel_path
    if not full.is_file():
        return []
    lines = full.read_text(encoding="utf-8", errors="replace").splitlines()

    out: list[Definition] = []
    for i, text in enumerate(lines):
        name = None
        for pattern in JS_PATTERNS:
            match = pattern.match(text)
            if match:
                name = match.group(1)
                break
        if name is None:
            continue
        if TRIVIAL_ARROW.search(text) and not text.rstrip().endswith("{"):
            continue

        has_doc = False
        for j in range(i - 1, -1, -1):
            prev = lines[j].strip()
            if not prev:
                continue
            has_doc = prev.endswith("*/")
            break

        out.append(Definition(rel_path, i + 1, 0, name, "func", has_doc))

    for k, definition in enumerate(out):
        definition.end_line = (out[k + 1].line - 1) if k + 1 < len(out) else len(lines)
    return out


GO_FUNC = re.compile(r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*[\(\[]")


def go_defs(root: Path, rel_path: str) -> list[Definition]:
    """Top-level and method funcs in a Go file, by regex.

    The worked example of a second adapter. Go's convention differs from the
    brace-language default: the doc comment is `//` lines immediately above the
    declaration, conventionally starting with the identifier's own name. Any
    `//` line is accepted — enforcing GoDoc phrasing is a different job.

    In: repo root and a repo-relative path. Out: one Definition per func.
    """
    full = root / rel_path
    if not full.is_file():
        return []
    lines = full.read_text(encoding="utf-8", errors="replace").splitlines()

    out: list[Definition] = []
    for i, text in enumerate(lines):
        match = GO_FUNC.match(text)
        if not match:
            continue
        has_doc = False
        for j in range(i - 1, -1, -1):
            prev = lines[j].strip()
            if not prev:
                continue
            has_doc = prev.startswith("//")
            break
        out.append(Definition(rel_path, i + 1, 0, match.group(1), "func", has_doc))

    for k, definition in enumerate(out):
        definition.end_line = (out[k + 1].line - 1) if k + 1 < len(out) else len(lines)
    return out


LANG_BY_SUFFIX = {".py": "python", ".js": "javascript", ".go": "go"}
ADAPTERS = {"python": python_defs, "javascript": javascript_defs, "go": go_defs}


def defs_for(root: Path, rel_path: str) -> list[Definition]:
    """Dispatch to the right adapter by file extension.

    In: repo root and a repo-relative path. Out: definitions, or [] for a
    language with no adapter — an unknown extension is not an error, it is
    simply out of scope.
    """
    lang = LANG_BY_SUFFIX.get(Path(rel_path).suffix)
    if lang is None:
        return []
    return ADAPTERS[lang](root, rel_path)


# --- Checks 1 + 2: documentation coverage -------------------------------------

def check_coverage(root: Path, cfg: dict, rep: Reporter) -> None:
    """Docstrings and doc blocks on the enforced files, plus a syntax gate.

    In: repo root, config, reporter. Out: nothing; reports findings.

    Prints the "enforced list is empty" hint rather than a misleading clean
    result when nothing is configured — a green tick on zero files is the single
    most confusing thing this tool could say.
    """
    rep.section("Checks 1-2: function documentation (enforced files)")

    enforced = cfg["enforced"]
    py_files = list(enforced.get("python") or [])
    js_files = list(enforced.get("javascript") or [])
    go_files = list(enforced.get("go") or [])

    if not (py_files or js_files or go_files):
        rep.line("  (enforced list is empty — nothing checked strictly yet.)")
        rep.line("  Run with --audit for a coverage report across the whole repo.")
        return

    before = rep.findings

    for rel in py_files:
        if not (root / rel).is_file():
            continue
        for d in python_defs(root, rel):
            if d.kind == "ERROR":
                rep.finding(f"{d.file}: parse error")
                continue
            if d.name.startswith("test_"):
                continue
            if not d.has_doc:
                rep.finding(f"{d.file}:{d.line}  {d.name}  — no docstring")

    for rel in js_files:
        for d in javascript_defs(root, rel):
            if not d.has_doc:
                rep.finding(f"{d.file}:{d.line}  {d.name}  — no /** */ block")

    for rel in go_files:
        for d in go_defs(root, rel):
            if not d.has_doc:
                rep.finding(f"{d.file}:{d.line}  {d.name}  — no // doc comment")

    # Syntax gate. `node --check` needs no packages and no package.json. A
    # documentation sweep's realistic failure is an unbalanced /** */, which IS
    # a syntax error. Skipped silently when node is absent.
    if js_files and cfg.get("js_syntax_gate", True) and shutil.which("node"):
        for rel in js_files:
            full = root / rel
            if not full.is_file():
                continue
            proc = subprocess.run(["node", "--check", str(full)],
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                err = " ".join((proc.stderr or proc.stdout or "").split())
                rep.finding(f"{rel} — SYNTAX ERROR: {err}")

    if rep.findings == before:
        rep.line("  All enforced files fully documented.")


# --- Audit --------------------------------------------------------------------

def show_audit(root: Path, cfg: dict, rep: Reporter) -> None:
    """Coverage across every source file, enforced or not. Gates nothing.

    In: repo root, config, reporter. Out: nothing; prints a table.

    This is the measuring instrument, not a check — run it before starting a
    sweep and keep the number. Tests and package markers are excluded from the
    Python count because they are exempt by convention.
    """
    rep.section("Audit: documentation coverage, whole repo")
    skip = cfg["skip_dirs"]
    total_has = total_all = 0

    groups = (
        ("Python", ".py", python_defs,
         lambda r: not re.search(r"(^|/)tests?/", r) and not r.endswith("__init__.py")),
        ("JavaScript", ".js", javascript_defs, lambda r: True),
        ("Go", ".go", go_defs, lambda r: True),
    )

    for label, suffix, adapter, keep in groups:
        files = sorted(
            str(p.relative_to(root)).replace("\\", "/")
            for p in root.rglob(f"*{suffix}")
            if p.is_file()
        )
        files = [f for f in files if not is_skipped(f, skip) and keep(f)]
        rows = []
        for rel in files:
            defs = [d for d in adapter(root, rel) if d.kind != "ERROR"]
            if not defs:
                continue
            has = sum(1 for d in defs if d.has_doc)
            total_has += has
            total_all += len(defs)
            pct = round(100 * has / len(defs))
            mark = "ok  " if pct == 100 else "    "
            rows.append(f"  {mark}{has:>4}/{len(defs):<4} {pct:>3}%  {rel}")
        if rows:
            rep.line("")
            rep.line(f"  {label}")
            for row in rows:
                rep.line(row)

    if total_all:
        rep.line("")
        rep.line(f"  Total: {total_has}/{total_all} ({round(100 * total_has / total_all)}%)")


# --- Diagram discovery --------------------------------------------------------

COVERS_RE = re.compile(r"^\s*%%\s*covers:\s*(.+)$")


def find_diagrams(root: Path, cfg: dict) -> list[Diagram]:
    """Every .mmd under a configured diagram dir, with its declared coverage.

    In: repo root, config. Out: Diagrams sorted by path, so output ordering is
    stable across platforms and between the two implementations.

    A `covers:` line is `path:fn1,fn2`. The path is split on the LAST colon so a
    Windows-style absolute path would not be mangled — repo-relative paths are
    expected, but being wrong here is silent.
    """
    out: list[Diagram] = []
    for dir_name in cfg.get("diagram_dirs", ["docs"]):
        for d in sorted(root.rglob(dir_name)):
            if not d.is_dir():
                continue
            rel_dir = str(d.relative_to(root)).replace("\\", "/")
            if is_skipped(rel_dir, cfg["skip_dirs"]):
                continue
            for f in sorted(d.glob("*.mmd")):
                rel = str(f.relative_to(root)).replace("\\", "/")
                diagram = Diagram(path=rel)
                for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                    m = COVERS_RE.match(line)
                    if not m:
                        continue
                    spec = m.group(1).strip()
                    idx = spec.rfind(":")
                    if idx < 1:
                        continue
                    fns = [x.strip() for x in spec[idx + 1:].split(",") if x.strip()]
                    diagram.covers.append(Cover(spec[:idx].strip(), fns))
                out.append(diagram)
    return sorted(out, key=lambda x: x.path)


# --- Check 3: covers: integrity ----------------------------------------------

def check_covers(root: Path, diagrams: list[Diagram], rep: Reporter) -> None:
    """Every function named in a `covers:` header still exists.

    This is the check that actually bites. Rename a function and its diagram
    goes red on the next run, which is the ONLY automatic signal that a
    hand-drawn diagram has fallen behind the code.

    In: repo root, diagrams, reporter. Out: nothing; reports findings.
    """
    rep.section("Check 3: covers: headers point at code that exists")

    if not diagrams:
        rep.line("  (no .mmd files found under any docs/ directory.)")
        return

    before = rep.findings
    cache: dict[str, list[str]] = {}
    checked = 0

    for diagram in diagrams:
        if not diagram.covers:
            rep.finding(f"{diagram.path}  — no `%% covers:` header")
            continue
        for cover in diagram.covers:
            if not (root / cover.source_file).is_file():
                rep.finding(f"{diagram.path}  — covers missing file: {cover.source_file}")
                continue
            if cover.source_file not in cache:
                cache[cover.source_file] = [d.name for d in defs_for(root, cover.source_file)]
            known = cache[cover.source_file]
            for fn in cover.functions:
                checked += 1
                if fn not in known:
                    rep.finding(
                        f"{diagram.path}  — {cover.source_file} has no "
                        f"'{fn}' (renamed or removed?)"
                    )

    if rep.findings == before:
        rep.line(
            f"  {checked} function reference(s) across {len(diagrams)} "
            "diagram(s) all resolve."
        )


# --- Check 4: staleness -------------------------------------------------------

def git_lines(root: Path, args: list[str]) -> list[str]:
    """Run a git command in the repo, returning stdout lines. [] on failure.

    Swallows a git failure on purpose: a repo with no commits yet, or no git at
    all, should degrade to "nothing changed" rather than crashing a docs check.
    """
    try:
        proc = subprocess.run(["git", "-C", str(root), *args],
                              capture_output=True, text=True)
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    return [l for l in proc.stdout.splitlines() if l]


def changed_files(root: Path) -> list[str]:
    """Repo-relative paths changed vs HEAD, working tree and index together."""
    out = set(git_lines(root, ["diff", "--name-only", "HEAD"]))
    out |= set(git_lines(root, ["diff", "--name-only", "--cached"]))
    return sorted(out)


HUNK_RE = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@")


def changed_ranges(root: Path, rel_path: str) -> list[tuple[int, int]]:
    """Changed line ranges for one file, from a zero-context diff.

    In: repo root, repo-relative path. Out: (from, to) inclusive pairs.

    A zero-length new-side hunk (a pure deletion) still points at a real line, so
    it is treated as one line — otherwise deletions are attributed nowhere and
    --deep silently misses them.
    """
    lines = git_lines(root, ["diff", "-U0", "HEAD", "--", rel_path])
    lines += git_lines(root, ["diff", "-U0", "--cached", "--", rel_path])
    ranges = []
    for line in lines:
        m = HUNK_RE.match(line)
        if not m:
            continue
        start = int(m.group(1))
        count = int(m.group(2)) if m.group(2) else 1
        if count == 0:
            count = 1
        ranges.append((start, start + count - 1))
    return ranges


def check_staleness(root: Path, diagrams: list[Diagram], rep: Reporter,
                    deep: bool) -> None:
    """Which diagrams the working diff has invalidated.

    In: repo root, diagrams, reporter, and deep. Out: nothing; reports findings.

    Without --deep a diagram is flagged when any file it covers changed. With
    --deep it is flagged only when a covered FUNCTION's line range overlaps a
    changed hunk — fewer false positives, more git calls.

    Remember what this can and cannot say: it names diagrams to REVIEW. Only a
    human can decide whether one is actually wrong.
    """
    mode = "changed functions" if deep else "changed files"
    rep.section(f"Check 4: diagrams to review ({mode})")

    changed = changed_files(root)
    if not changed:
        rep.line("  Working tree clean — nothing to review.")
        return

    hits: dict[str, set[str]] = {}
    for diagram in diagrams:
        for cover in diagram.covers:
            if cover.source_file not in changed:
                continue
            reason = cover.source_file
            if deep:
                ranges = changed_ranges(root, cover.source_file)
                if not ranges:
                    continue
                by_name = {d.name: d for d in defs_for(root, cover.source_file)}
                touched = [
                    fn for fn in cover.functions
                    if fn in by_name
                    and any(lo <= by_name[fn].end_line and hi >= by_name[fn].line
                            for lo, hi in ranges)
                ]
                if not touched:
                    continue
                reason = f"{cover.source_file} → {', '.join(touched)}"
            hits.setdefault(diagram.path, set()).add(reason)

    if not hits:
        rep.line("  No diagram covers anything in this diff.")
        if not deep:
            rep.line("  (Nothing changed under a covers: header.)")
        return

    for path in sorted(hits):
        rep.finding(path)
        for reason in sorted(hits[path]):
            rep.line(f"      {reason}")
    rep.line("")
    rep.line("  Update these in the same commit, or say in the commit message why not.")


# --- Render check -------------------------------------------------------------

MERMAID_FENCE = re.compile(r"^[ \t]*```mermaid[ \t]*\r?\n(.*?)^[ \t]*```",
                           re.MULTILINE | re.DOTALL)

RENDER_TAIL = """
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
  document.getElementById('errors').textContent = errs.join('\\n\\n');
  await mermaid.run({ querySelector: 'pre.mermaid' });
</script></body></html>
"""


def render_check(root: Path, cfg: dict, rep: Reporter, open_browser: bool = True) -> None:
    """Build a page that mermaid.parse()es every diagram, and open it.

    There is no offline Mermaid linter without npm, so this is the workaround:
    a self-contained page plus one CDN fetch. If npm is already in your repo,
    `mmdc --parse` removes the browser step.

    Not theoretical — the first run of this against eight hand-written diagrams
    found NINE real parse errors in files that had been read carefully:
    a bare `%%` line survives Mermaid's comment stripper and collides with the
    graph header, and `A -. .-> B` is not an unlabelled dotted edge.

    In: repo root, config, reporter. Out: nothing; writes the harness and opens
    it. Markdown fences are included because architecture diagrams embedded in
    a README break exactly as easily and nothing else would ever parse them.
    """
    items: list[tuple[str, str]] = []

    for diagram in find_diagrams(root, cfg):
        items.append((diagram.path, (root / diagram.path).read_text(encoding="utf-8")))

    if cfg.get("scan_markdown_diagrams", True):
        for md in sorted(root.rglob("*.md")):
            rel = str(md.relative_to(root)).replace("\\", "/")
            if is_skipped(rel, cfg["skip_dirs"]):
                continue
            text = md.read_text(encoding="utf-8", errors="replace")
            if "```mermaid" not in text:
                continue
            for n, m in enumerate(MERMAID_FENCE.finditer(text), start=1):
                items.append((f"{rel} (inline #{n})", m.group(1)))

    if not items:
        rep.line("No Mermaid diagrams found.")
        return

    parts = [
        "<!doctype html><html><head><meta charset=\"utf-8\">",
        "<title>diagram render check</title>",
        '<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>',
        "<style>body{font:14px system-ui;margin:2rem}"
        "h1{position:sticky;top:0;background:#fff;padding:.5rem 0}"
        "#errors{color:#a00;white-space:pre-wrap}</style>",
        '</head><body><h1 id="verdict">rendering…</h1><pre id="errors"></pre>',
    ]
    for label, source in items:
        parts.append(
            f'<h2>{html.escape(label)}</h2>'
            f'<pre class="mermaid" data-src="{html.escape(label, quote=True)}">'
            f'{html.escape(source)}</pre>'
        )
    parts.append(RENDER_TAIL)

    out = Path(tempfile.gettempdir()) / "diagram-render-check.html"
    out.write_text("\n".join(parts), encoding="utf-8")

    rep.line(f"Wrote render harness for {len(items)} diagram(s):")
    rep.line(f"  {out}")
    rep.line("")
    rep.line("Opening in the default browser. The heading reads ALL OK or lists")
    rep.line("the parse errors; the diagrams themselves render below it.")
    if open_browser:
        webbrowser.open(out.as_uri())


# --- Entry point --------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the selected checks, return an exit code.

    In: argv (None means sys.argv). Out: 0 when clean, 1 on any finding, 2 on a
    malformed config.

    --audit and --render both exit 0 unconditionally: neither is a gate. Only
    the checks gate.
    """
    parser = argparse.ArgumentParser(
        prog="check_docs.py",
        description="Verify functions are documented and diagrams match the code.",
    )
    parser.add_argument("--all", action="store_true",
                        help="checks 1-3 only, ignore git state")
    parser.add_argument("--deep", action="store_true",
                        help="narrow check 4 to changed functions, not files")
    parser.add_argument("--audit", action="store_true",
                        help="coverage for every source file; gates nothing")
    parser.add_argument("--render", action="store_true",
                        help="build a page that parses every diagram")
    parser.add_argument("--quiet", action="store_true",
                        help="exit code only, no output")
    parser.add_argument("--root", default=None,
                        help="repo root (default: discovered from cwd)")
    parser.add_argument("--no-open", action="store_true",
                        help="with --render, write the harness but do not open it")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else find_repo_root(Path.cwd())
    cfg = load_config(root)
    rep = Reporter(quiet=args.quiet)

    if args.render:
        render_check(root, cfg, rep, open_browser=not args.no_open)
        return 0

    if args.audit:
        show_audit(root, cfg, rep)
        return 0

    diagrams = find_diagrams(root, cfg)
    check_coverage(root, cfg, rep)
    check_covers(root, diagrams, rep)
    if not args.all:
        check_staleness(root, diagrams, rep, deep=args.deep)

    rep.line("")
    if rep.findings == 0:
        rep.line("check-docs: clean.")
        return 0
    rep.line(f"check-docs: {rep.findings} finding(s).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
