# The CLAUDE.md block

`install --project` writes everything between the two markers below into your
repo's `CLAUDE.md` (or `AGENTS.md`, or `CONTRIBUTING.md` — `--doc <file>`).
Re-running updates it in place, so it is safe to install repeatedly.

The markers matter: without them a second install duplicates the block, and
nobody notices until the file is 400 lines of the same three rules.

Nothing here is claude-specific despite the filename. It is a contributor
convention that happens to also be read by a coding harness.

---

<!-- docs-toolkit:begin — managed block, edit above or below but not inside -->

## Every function gets documented — no exceptions

Add a function, write its docstring in the same edit. Change a function so its
contract, its failure mode, or the reason it exists moved? The words above it
move too, same edit. Not "before commit" — same edit, while you still remember
why.

Format is identical in every language: one line saying what it does, a `Why:`
giving the reason it exists (the failure it prevents, the upstream quirk it
absorbs, the behaviour it serves), then `In:`/`Out:` in prose naming the edge
case that will bite.

**No `@param`/`@returns`/`Args:` tags** unless something actually consumes them
— a docs site, `mypy --strict`, IDE contract checks. Otherwise they crowd out
the *why*, which is the part you cannot recover from reading the code.

```python
def _msg_signature(msg):
    """Collapse counter drift in a message so it dedups.

    Why: the upstream re-fires with a ticking counter ('for 901 seconds' →
    'for 954 seconds'); grouping on raw text inflates the count 30+x per
    real event.

    In: raw string, or None/''. Out: same string with every digit run
    replaced by 'N'; '' for falsy input (never None).
    """
```

Brace languages use the same prose in a `/** */` block — the block form, not
`//`, so a contract is visually distinct from an inline aside.

**Exempt**: test functions (the test name is the documentation), package
markers, and single-expression one-liners whose name says everything
(`const toRad = d => d * Math.PI / 180`). The checker knows about that last
exemption; it does not need arguing with.

**Cross-file twins get the same docstring**, plus an explicit note on the one
genuine difference. Where two near-identical functions must NOT be unified, say
so — otherwise someone will helpfully unify them.

Bring a file to 100% before adding it to the checker's `$ENFORCED_*` arrays,
never the other way round, or the output fills with known misses and stops
being read.

## Epistemic tagging in docstrings

Docstrings carry two kinds of statement. One is derivable from the code by
reading it. The other is not — it came from a spec, a production run, or domain
knowledge, and **it cannot be recovered if it is lost**. Tag the second kind so
a future rewrite can see the boundary.

- `[DOCUMENTED]` — stated in a vendor spec, RFC, or stdlib docs. Name the
  source.
- `[OBSERVED <date>]` — measured, or seen in a real run. Needs a date and
  enough provenance to re-run it.
- `[INFERRED]` — reasoned from code or evidence, not verified against reality.
  Say what would confirm it.
- `[GUESS]` — no basis. A placeholder asking to be checked or deleted.

### Do not tag what the code already says

**This rule matters more than the definitions above.** `TTL-cached 300s`,
`returns a list of ints`, `raises on HTTP failure` are all visible in the
function. Tagging them dilutes the signal until nobody reads the tags.

The test: **could this line be reconstructed from the code alone?** If yes, no
tag. If it needs a spec, a run, or domain knowledge, tag it.

A design *rationale* is not evidence — it belongs in untagged `Why:` prose.

The failure mode of this convention is **over**-tagging, not under-tagging.
Marking `returns a list of ints` as `[DOCUMENTED]` is syntactically fine and
destroys the entire point, because the tags become decoration and a diff
touching them stops meaning anything. When reviewing a regeneration, check for
over-tagging **first** — that is the direction the error comes from.

### Tags survive rewrites

- Untagged prose may be rewritten freely.
- **A tagged line is evidence, not phrasing.** Do not reword, replace, or drop
  it as part of an unrelated edit.
- If a tagged line must change because the code changed, change the tag with it
  and **say so explicitly in the summary** — never fold it into a diff that
  reads as a wording improvement.
- **Updates preserve or DOWNGRADE a tag; upgrades require cited evidence.**
  `[INFERRED]` becomes `[OBSERVED]` only when a run produced the number, and the
  date comes from that run. Downgrading is always allowed, and is the honest move
  when a claim turns out to rest on less than you thought.

  The drift vector this closes is specific: an eager session "improves" a file
  and rewrites `[INFERRED]` material in `[DOCUMENTED]` voice. Nothing in the diff
  looks wrong — the prose got better — and a guess is now load-bearing.
- Pair a checkable `[OBSERVED]` claim with a real test, reading real data rather
  than a fixture, so it fails when the convention changes. A claim written only
  as prose is a test that never runs.

## Diagrams follow the code

Logic diagrams live in `docs/*.mmd`, indexed by `docs/DIAGRAMS.md`. Node IDs
are the real function names, so every box is greppable. Solid edges are the
normal path; **dashed edges are failure, degraded, dead, or
scheduled-for-removal**. ALL-CAPS annotation nodes carry prose explaining *why*.

Each diagram declares its own coverage:

```
%% covers: src/alarms.py:refresh_state,reconcile_snapshot
```

Touch a function named in a `covers:` line → update that diagram in the same
commit. Add a function to a flow that is already diagrammed → it gets a node
and a `covers:` entry. Add a whole new flow → a new `.mmd`, an index row, and a
link. Not a paragraph of prose pretending to be a diagram.

Before committing:

```powershell
.\check-docs.ps1              # docs + covers integrity + what your diff invalidated
.\check-docs.ps1 -Deep        # narrow the staleness check to changed functions
.\check-docs.ps1 -Render      # confirm every diagram still parses
.\check-docs.ps1 -Audit       # coverage across the repo; gates nothing
```

Fix what it names, **or say in the commit message why you didn't** — a
docstring-only change to a covered function genuinely does not invalidate a
diagram, and saying so is the honest close-out.

Two Mermaid traps, both caught by `-Render`:

- A **bare `%%` line** survives Mermaid's comment stripper (it needs at least
  one character after the `%%`) and then collides with the graph header. Write
  `%% ---` for a separator.
- **`call` is a reserved node ID** (the `click <id> call fn()` directive). So
  are `end`, `graph`, `subgraph`, `class`, `style`. And `A -. .-> B` is not an
  unlabelled dotted edge — `A -.-> B` is.

## Rejected approaches go in TODONT.md

When an approach is abandoned — including ones that sound obviously sensible —
log it in `TODONT.md`: what was tried, the **verdict**, and **why not**, with
the measurement or the concrete failure rather than an opinion. Read it before
proposing anything structural. Update an existing entry rather than duplicating
it when a verdict later narrows or reverses.

A docstring is the wrong home for a decision that spans the codebase; nobody
finds it there.

<!-- docs-toolkit:end -->
