# Main-theorem formalization conventions (`bench/tritonbench_g`)

Every kernel `.lean` file must culminate in a **main theorem** (the "headline",
usually named `<kernel>_output_summary` / `..._compute_correct` / `..._general`)
that meets *all* of the requirements below. These are the standard this repo
holds every per-kernel correctness proof to; use this as the checklist when
adding, reviewing, or hardening a kernel.

---

## 1. Position — the last theorem is the main theorem

The **final `theorem` in the file must be the headline**. A reviewer should be
able to read the last theorem, then walk *up* to the lemmas it references, and
see the whole trust chain. No pinned corollaries, dead helpers, or stray lemmas
after it.

## 2. Dimension-general — no test shapes

- The headline is **universally quantified over all shapes / strides /
  compile-time flags** as symbolic `Nat`/`Bool`/`ℝ` parameters. It must **not**
  be pinned to the benchmark's concrete test dimensions.
- **Zero test-shape declarations.** Pinned per-shape / per-flag instantiations
  (the tell-tale marker is the token `python` in a decl name — e.g.
  `*_python_case4_*`, `*_python_block128_*`, `*_python_128x128_*`,
  `*_python_test_shape_*` — anything that is *not* a `_general` theorem) are
  redundant given the general headline and must be removed, along with any helper
  lemmas that become dead once they are gone.
- A genuine compile-time *case* decomposition (one theorem per autotune/flag
  branch) is fine **only** when each case theorem is itself dimension-general
  (e.g. `..._python_caseN_output_summary_general`); the `python`/`caseN` in the
  name then denotes the source kernel / flag branch, not a test shape.

## 3. Genuine — non-self-referential

The `expected` value (what the theorem claims the output is) must be a
**closed form over INPUT memory** (`s.readMem <input> …`, a `Finset.sum`, a pure
`Math/*` spec, …). It must **never** be `exec(...).readMem …` — i.e. the kernel's
own executed output read back and compared to itself. `scripts/spec_sheet.py`
must report `self-ref-flagged: 0`.

## 4. Structure — `ComputeCorrect.Realizes`

The headline is stated with the standard trust surface

```lean
ComputeCorrect.Realizes
  (kernel := <faithful surface>)
  (initialState := s)
  (write := <WriteMap: which cells to check>)
  (expected := <input-memory closed form>)
```

A multi-output kernel is stated as a **single headline theorem whose conclusion
is a conjunction of `Realizes`**, one conjunct per stored output (scalar outputs
use `fun _ : PUnit => some (region, offset)`; masked tile outputs use
`ComputeCorrect.WriteMap.writeIf mask addr`). Keep the conjunction — do **not**
split the outputs into separate top-level theorems; the file ends on one bundled
headline. Do **not** leave the summary in the raw
`(hExec : exec … = some s') → s'.readMem … = expected` form — that is a proof
artifact, not the public surface. (`Realizes` internalizes the `exec`
quantification, so the `s'`/`hExec` binders drop out of the statement.)

## 5. Axiom-clean

`#print axioms <headline>` must be **exactly `[propext, Classical.choice,
Quot.sound]`**. No `sorry` / `admit` / `native_decide` / `ofReduceBool`, and no
`sorryAx` anywhere in the transitive dependency.

## 6. Honest hypotheses and honest scope

- Real preconditions (region distinctness / no-aliasing, offset injectivity,
  `hundef`, positivity of block dims, …) are stated as **explicit honest
  hypotheses** of the headline — not silently assumed or worked around.
- Modeling boundaries are stated plainly (host launch / grid / autotune are the
  *trusted boundary*; arithmetic is over `ℝ`, not bit-accurate IEEE; casts erase
  to identity at the algorithm layer; …).
- If the model genuinely cannot cover a path (e.g. a faithful `tanh(-∞) = -1`
  makes a masked-lane argument fail), **narrow the theorem's scope and document
  it** — do not fake a spec that does not hold. Prefer *genuine-but-scoped* over
  *pinned-but-annotated*.

### Never lie, never silently ignore

If a kernel (or a path/output of it) **cannot be dimension-generalized or proved
genuinely**, that is a **reported failure — surface it as an explicit ERROR /
blocker**, with the exact goal or the exact obstacle. It is **forbidden** to:

- **mis-report** it — claim a theorem is general / genuine / axiom-clean when it
  is not (e.g. calling a pinned theorem `_general`, or an `exec`-readback spec
  "genuine");
- **silently ignore / drop** it — leave the shape pinned with no signal, quietly
  weaken the spec, or omit the output from the summary without saying so;
- **paper over** it with `sorry`/`admit`/`native_decide` or a vacuous/trivially-
  true statement.

The honest outcomes are exactly two: **(a)** a headline meeting §1–§6 in full, or
**(b)** a clearly-reported blocker (the specific goal that won't close, or the
specific reason generalization is impossible) plus, if partial progress is
landed, an explicitly-scoped theorem whose narrowed scope is stated in its own
hypotheses/docstring. Partial-but-honest is acceptable; anything that reads as
"done" when it isn't is not.

## 7. Shared pure math lives in `VeriTile/Triton/Math/*`

When a genuinely reusable pure-math object exists (e.g.
`crossEntropyLoss = stableLSE − logit_target`, `klDivSpec`), put the **pure
function** in `VeriTile/Triton/Math/*` and connect each kernel with a **bridge
lemma** proving its (kernel-coupled) spec reduces to the pure core in the base
regime. Kernel-coupled specs — those taking `BlockState` / `RegionName` / program
ids — **stay in the kernel file**; only pure `ℝ`-level math is lifted.

## 8. Tidy

- **No dead code**: delete superseded boundary lemmas (e.g. store-slice copies
  once the genuine forward subsumes them) and any helper left unreferenced.
- **Docstrings match the code**: the module's proof-architecture diagram and
  prose must name only declarations that still exist and describe the current
  headline form. No stale references.
- Regenerate spec sheets (`scripts/spec_sheet.py`) after changes;
  `self-ref-flagged: 0` and `no-summary: 0`.

## 9. One Python kernel ↔ one `.lean` file

Keep the 1:1 mapping with the benchmark's Python kernels. Do **not** merge two
distinct benchmark kernels into one file even if their proofs are near-identical
(e.g. `cross_entropy` vs `cross_entropy_ops`).

---

## Verification recipe

Bench files are checked individually (they are not `lake` library targets):

```bash
# 0. build the shared library once (so imports resolve)
lake build VeriTile
lake build VeriTile.Triton             # umbrella prelude (what ports import)

# 1. compile a kernel file — must be exit 0 with zero `error:` lines
lake env lean bench/tritonbench_g/<kernel>/<File>.lean

# 2. no cheats anywhere in the file
grep -nE '\bsorry\b|\badmit\b|native_decide' bench/tritonbench_g/<kernel>/<File>.lean   # -> none

# 3. axioms of the headline — append, run, then remove the line
#    (use the FULLY-QUALIFIED name; #print after `end <ns>` needs it)
#    expect: [propext, Classical.choice, Quot.sound]
printf '\n#print axioms VeriTile.Bench.TritonBenchG.<Ns>.<headline>\n' >> <File>.lean
lake env lean <File>.lean | grep -A2 'depends on axioms'
git checkout -- <File>.lean   # or manually delete the appended line

# 4. spec sheet: no self-reference, has a summary
python3 scripts/spec_sheet.py bench/tritonbench_g/<kernel>/<File>.lean
```

Notes:
- Verify with `lake env lean` (the LSP/`lean-lsp` may be unreliable in some
  environments); do **not** run bare `lake build`/`lake clean`/`lake exe cache`
  while iterating on a kernel — they can invalidate the prebuilt oleans.
- Masked-reduction / log-sum-exp proofs need
  `import VeriTile.Triton.Semantics.{TiledIndexing,MaskedReduction}` +
  `VeriTile.Triton.Math.LogSumExp` and **`open VeriTile.Triton.TiledLogSumExp`**
  (the lemmas `validLanes` / `partialLSE_full` / `sup'_masked_map_eq` / `stableLSE`
  live in that nested namespace).

---

*One line:* the last theorem is a **dimension-general, non-self-referential,
`ComputeCorrect.Realizes`-form (conjunction for multi-output), axiom-clean** main
theorem with honest hypotheses; pure math is factored into `Math/*`; the file
carries no dead code or stale docstrings — and if any of this genuinely cannot be
achieved, that is **reported as an explicit blocker, never faked or silently
ignored**.
