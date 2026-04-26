# VeriTile

**English** | [中文](README_zh.md)

Translation validation of Triton GPU kernels via LLM-assisted Lean 4 proofs.

Full proposal in `Notes/proposal.md` (English) / `Notes/proposal_zh.md` (Chinese). Program plan in `PLAN.md` / `PLAN_zh.md`.

## What this repo demonstrates

Two Triton kernels (naive softmax and numerically-stable softmax) embedded
in Lean via a `triton { ... }` macro, with their **algorithmic equivalence**
proved against our operational semantics — same shape as arm-in-lean's
`tnum_const_refinement`:

```lean
-- The two kernels (Triton-style syntax inside Lean):
def naiveSoftmaxKernel (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange($(N))
  x    := tl.load(X, offs)
  e    := tl.exp(x)
  s    := tl.sum(e)
  y    := e / s
  tl.store(Y, offs, y)
}

def stableSoftmaxKernel (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange($(N))
  x    := tl.load(X, offs)
  m    := tl.max(x)
  e    := tl.exp(x - m)
  s    := tl.sum(e)
  y    := e / s
  tl.store(Y, offs, y)
}

-- The refinement theorem (mirrors `tnum_const_refinement`):
theorem softmax_kernels_refinement
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoaded s N xs) :
    ∀ i : Fin N,
      observeY (exec (naiveSoftmaxKernel N) s) N s.pid i =
      observeY (exec (stableSoftmaxKernel N) s) N s.pid i
```

The theorem says: **for the same input, both kernels write the same values
to `Y`, position-by-position.** This is observational (algorithmic)
equivalence — the two kernels' intermediate registers (`m`, `e`, `s`)
intentionally differ; full BlockState equality would be a strictly stronger
(and false) claim.

## Refinement structure

Same shape as `tnum_const_refinement`: `exec` runs each kernel, an
observation function (`readMem` for us, `readReg` for arm-in-lean) extracts
the output channel, the theorem asserts pointwise equality. The math step
is isolated in `naive_eq_stable`; the operational walk-through reduces
`exec kernel s` to a closed form via simp on our gather/scatter semantics.

```
softmax_kernels_refinement              ✓ FULL PROOF
  ├─ softmax_naive_correct              ✓ FULL PROOF (operational walk-through)
  ├─ softmax_stable_correct             ✓ FULL PROOF (operational walk-through)
  └─ naive_eq_stable                    ✓ FULL PROOF (math; 6-line tactic)
                                            uses Real.exp_sub, Finset.sum_div,
                                            Real.exp_ne_zero, field_simp

Operational support:
  scatter_readback                      ✓ FULL PROOF (List.foldl induction over
                                            distinct offsets, via Nodup +
                                            injectivity of offsetFn)
```

P1 is sorry-free. The math content is isolated in `naive_eq_stable`; the
two `softmax_*_correct` lemmas are pure operational walk-throughs that
reduce `exec kernel s` to its closed-form output via simp on
`exec / stepStmts / stepStmt / evalOp / Value.bop / writeMem`, then
delegate to `scatter_readback` for the final readback step.
`scatter_readback` itself is a `List.foldl` induction: split
`List.finRange n = l₁ ++ i :: l₂`, use `Nodup` + injectivity to show no
later write touches the target cell, and read off the value written at
position `i`.

## Layout

```
README.md / README_zh.md         This file (English / 中文)
PLAN.md / PLAN_zh.md             Program plan (English / 中文)
lakefile.toml, lean-toolchain    Lake build (Lean 4 v4.15.0; works on v4.29.0)
VeriTile.lean                    Top-level library entry

VeriTile/Triton/
  Core.lean                      Op / Stmt / Kernel data types (P1 subset)
  Semantics.lean                 BlockState, evalOp (gather), stepStmt (scatter),
                                   exec, scatter_readback
  DSL.lean                       The `triton { ... }` macro
  Examples.lean                  Hand-built naiveSoftmax kernel (constructor form)

VeriTile/Examples/
  SoftmaxEq.lean                 Naive vs stable softmax + refinement theorem

Notes/
  proposal.md / proposal_zh.md   Project proposal (English / 中文)
  MacroOptions.md                Tech investigation: macro implementation tradeoffs
```

## Build

Requires Elan + Lake. The user shell does not auto-source `.elanrc`, so
prefix `PATH` manually:

```bash
PATH="$HOME/.elan/bin:$PATH" lake update     # ~5–15 min, downloads Mathlib
PATH="$HOME/.elan/bin:$PATH" lake build
```

Expected: clean build, no warnings, no sorries.

## Triton subset (P1 scope)

**Included:**
* `tl.load`, `tl.store` — both with **gather/scatter** (tile-valued offsets).
* `tl.arange`, `tl.broadcast`, `tl.full`, scalar/tensor constants.
* `tl.exp`, `tl.maximum`, basic arithmetic (`+ − * /`).
* `tl.max`, `tl.sum` — block-level reductions (axis = 0), defined via
  Mathlib `Finset.sup'` / `Finset.sum`.
* `tl.program_id`, `tl.constexpr`.
* `assign`, `store`. The `forLoop` AST exists but is operationally `none`.

**Excluded (P2+):**
* `tl.atomic_*`, `tl.dot`, async copy.
* Multi-block coordination, cluster-level ops.
* Hopper / Blackwell ops (TMA, WGMMA).
* `forLoop` operational semantics — needs mutual recursion with `stepStmts`
  and an explicit `termination_by (sizeOf body + 1, n − i)` measure.

**Trust assumptions:** floating-point arithmetic is modelled in `ℝ` (Mathlib
`Real`). IEEE-754 fidelity is out of scope; differential testing against
PyTorch covers it (P5+).

## The `triton { ... }` macro

Modelled on arm-in-lean's `arm64 { ... }` pattern. Takes a Triton-style
block and lowers it to our `Op` / `Stmt` / `Kernel` AST.

Conventions:
* Bare lowercase identifiers → register references (`pid`, `x`, `e`).
* First argument of `tl.load(...)` / `tl.store(...)` → bare identifier
  interpreted as a memory region name (string literal).
* `$(LeanTerm)` → antiquote a Lean-level value (used for the kernel's
  `tl.constexpr` size argument `N`).
* Numeric literals → `Op.const`.
* Statements separated by newlines (`tritonStmt*` syntax category).

`Notes/MacroOptions.md` discusses why we chose the macro syntax over
direct constructor calls / external S-expressions / scoped notation, and
the tradeoffs.

## Roadmap

**P1 (skeleton — done):**
- [x] Project layout, lakefile, lean-toolchain.
- [x] `Op` / `Stmt` / `Kernel` types.
- [x] Operational semantics (`evalOp`, `stepStmt`, `exec`) with gather and
      scatter.
- [x] `triton { ... }` macro.
- [x] Two-kernel softmax example via the macro.
- [x] Math equivalence (`naive_eq_stable`): full proof.
- [x] Refinement theorem (`softmax_kernels_refinement`): full proof.
- [x] `scatter_readback`: full proof (`List.foldl` induction).
- [x] `softmax_naive_correct`, `softmax_stable_correct`: full proof
      (operational walk-through + `scatter_readback`).

**P2 (operational polish):**
- [ ] `forLoop` operational semantics (mutual recursion + `termination_by`).
- [ ] Custom simp lemmas for `Value.bop` on tile-tile of equal length.
- [ ] Auto-derive `Kernel.inputs` / `Kernel.outputs` from the macro body.

**P3+ (research direction — see `proposal.md`):**
- [ ] Python lifter (real `.py` Triton → embedded `triton { ... }` term).
- [ ] LLM-driven proof drafting harness (Vero-style).
- [ ] More equivalence examples: Welford variance, log-sum-exp, scan
      reordering.
- [ ] FlashAttention-style decomposition (stretch).

## Positioning

Closest neighbours:

* **arm-in-lean** — same refinement shape, ARM64 target. We port the
  pattern to Triton GPU kernels.
* **Vero** (preprint, 2026) — Lean-checked verified code generation for
  LLVM transfer functions. We do refinement at the kernel level (not
  compiler-pass) and at the algorithmic layer (not implementation).
* **ATL** (POPL'20) — Coq-embedded tensor DSL with equivalence proofs;
  closest relative for tensor-program equivalence but operates on a
  bespoke DSL, not on production kernels.

VeriTile's intersection: **algorithm-layer + LLM-targeted + Lean-verified
+ executable on production-grade Triton.** None of the three neighbours
covers all four axes simultaneously.
