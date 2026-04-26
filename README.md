# VeriTile

**English** | [中文](README_zh.md)

Translation validation of Triton GPU kernels via LLM-assisted Lean 4 proofs.

Full proposal in `Notes/proposal.md` (English) / `Notes/proposal_zh.md` (Chinese). Program plan in `PLAN.md` / `PLAN_zh.md`.

**Current status:** Phase A complete — see release [`v0.1-tier1`](https://github.com/Lizn-zn/VeriTile/releases/tag/v0.1-tier1). 3 Tier 1 kernel-pair theorems closed, T3 (FlashAttention) feasibility study delivered, LLM proof-drafting wrapper shipped.

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

Two more kernel pairs (LSE direct vs shift-trick, softmax `e/s` vs
precomputed reciprocal) follow the same pattern; see §Refinement structure.

## Environment (prepare)

### Required

- **[Elan](https://github.com/leanprover/elan)** — Lean 4 toolchain manager.
  Install via `curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh`.
- **Lake** — comes with Elan; manages the Lean build.
- **Lean 4** v4.15.0 (pinned via `lean-toolchain`; works on v4.29.0 too).
- **Mathlib** — pulled automatically by `lake update`.

The user shell does not auto-source `.elanrc`, so prefix `PATH` manually
in any shell where you run lake / lean:

```bash
export PATH="$HOME/.elan/bin:$PATH"
```

Or prefix individual commands. Examples in §Usage do this inline.

### Optional (for the LLM proof wrapper)

- **[Claude Code CLI](https://docs.claude.com/en/docs/claude-code)** — `claude` on PATH.
- **[lean4-skills plugin](https://github.com/lean4-skills/lean4-skills)** — provides `/lean4:autoprove`. Install via `claude /plugin add lean4/lean4-skills` (or your preferred plugin install flow).
- **`jq`** — for JSON output parsing in `scripts/prove.sh`. macOS: `brew install jq`.

Without these, you can build and inspect the project just fine — only
`scripts/prove.sh` (and the held-out eval) needs them.

### Optional (for differential testing, planned in later phases)

- **GPU + Triton + flash-attn** — only used in Phases B–D for the differential testing artifact (see PLAN.md §Differential testing). Not required for Phase A.

## Usage

### Build the project

```bash
PATH="$HOME/.elan/bin:$PATH" lake update     # ~5–15 min, first-time Mathlib download
PATH="$HOME/.elan/bin:$PATH" lake build
```

Expected: clean build, no warnings, no sorries.

### Browse a worked example

The fastest way to understand what the project does:

```bash
$EDITOR VeriTile/Examples/SoftmaxEq.lean       # the prototype, well-commented
$EDITOR VeriTile/Examples/LogSumExpEq.lean     # log-sum-exp pair (single scalar output)
$EDITOR VeriTile/Examples/SoftmaxReciprocal.lean   # save N divisions via 1/s
$EDITOR VeriTile/Examples/WelfordMath.lean     # math-only Welford identity (Phase B prep)
```

Each file walks through: kernel definitions via `triton { ... }` → math identity → kernel-correctness lemmas → headline refinement theorem.

### Run the LLM proof wrapper on a held-out theorem

```bash
# Dry run / argument check
scripts/prove.sh --help

# Real run (calls /lean4:autoprove via claude; spends API budget)
scripts/prove.sh bench/llm_eval/softmax_naive_correct_held_out.lean --max-cycles 5
```

Per-trial JSON logs land in `Logs/`; success/fail is determined by re-running `lake env lean` on the (potentially modified) file.

### Run the Phase A 5-trial eval

```bash
mkdir -p bench/llm_eval/results
> bench/llm_eval/results/summary.log
for trial in 1 2 3 4 5; do
  TRIAL_FILE="/tmp/heldout_$(date +%s)_${trial}.lean"
  cp bench/llm_eval/softmax_naive_correct_held_out.lean "${TRIAL_FILE}"
  scripts/prove.sh "${TRIAL_FILE}" --max-cycles 5 \
    >> bench/llm_eval/results/summary.log 2>&1
done
grep -c "\[SUCCESS\]" bench/llm_eval/results/summary.log
```

The Phase A eval report is at `bench/llm_eval/results/phase_a_report.md`.

### Add a new kernel-pair example

Mirror the structure of `LogSumExpEq.lean` (simplest):

1. Define both kernels via the `triton { ... }` macro
2. State the math identity (`theorem ... : math_a = math_b`)
3. State two correctness lemmas (each kernel reduces to a closed form via `simp` on the operational semantics)
4. Compose them into the headline refinement theorem
5. Add the file to `VeriTile.lean`'s import list
6. `lake build` — should compile with no sorries

`scripts/prove.sh` can attempt the math identity for you; the operational walk-throughs are usually a 2–10 line copy-and-adapt from existing kernels.

## Triton subset (current scope)

**Included:**
* `tl.load`, `tl.store` — both with **gather/scatter** (tile-valued offsets).
* `tl.arange`, `tl.broadcast`, `tl.full`, scalar/tensor constants.
* `tl.exp`, `tl.log`, `tl.maximum`, basic arithmetic (`+ − * /`).
* `tl.max`, `tl.sum` — block-level reductions (axis = 0), defined via
  Mathlib `Finset.sup'` / `Finset.sum`.
* `tl.program_id`, `tl.constexpr`.
* `assign`, `store`. The `forLoop` AST exists but is operationally `none`.

**Excluded (Phase B+):**
* `tl.atomic_*`, `tl.dot`, async copy.
* Multi-block coordination, cluster-level ops.
* Hopper / Blackwell ops (TMA, WGMMA).
* `forLoop` operational semantics — needs mutual recursion with `stepStmts`
  and an explicit `termination_by (sizeOf body + 1, n − i)` measure. Phase B work.

**Trust assumptions:** floating-point arithmetic is modelled in `ℝ` (Mathlib
`Real`). IEEE-754 fidelity is out of scope; the differential testing artifact
(Phase B+) catches gross divergence numerically, not bit-level.

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

## Refinement structure

Same shape as `tnum_const_refinement`: `exec` runs each kernel, an
observation function (`readMem` for us, `readReg` for arm-in-lean) extracts
the output channel, the theorem asserts pointwise equality. The math step
is isolated in `naive_eq_stable`; the operational walk-through reduces
`exec kernel s` to a closed form via simp on our gather/scatter semantics.

### Tier 1 (current scope, all closed)

| Theorem | File | Math identity | Status |
|---|---|---|---|
| `softmax_kernels_refinement` | `Examples/SoftmaxEq.lean` | `naive_eq_stable` | ✓ |
| `log_sum_exp_refinement` | `Examples/LogSumExpEq.lean` | `log_sum_exp_shift_invariant` | ✓ |
| `softmax_reciprocal_refinement` | `Examples/SoftmaxReciprocal.lean` | `div_eq_mul_inv_real` | ✓ |
| `welford_eq_two_pass` (math lemma, Phase B prep) | `Examples/WelfordMath.lean` | own | ✓ |

Each kernel-pair theorem composes a math identity with two operational
walk-throughs (one per kernel), all delegating to `scatter_readback` for
the final tile-write readback.

### Detail: softmax (the prototype)

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

The math content is isolated in `naive_eq_stable`; the two
`softmax_*_correct` lemmas are pure operational walk-throughs that
reduce `exec kernel s` to its closed-form output via simp on
`exec / stepStmts / stepStmt / evalOp / Value.bop / writeMem`, then
delegate to `scatter_readback` for the final readback step.
`scatter_readback` itself is a `List.foldl` induction: split
`List.finRange n = l₁ ++ i :: l₂`, use `Nodup` + injectivity to show no
later write touches the target cell, and read off the value written at
position `i`. The other two Tier 1 theorems follow the same structure
(LSE writes a single scalar at offset `pid` rather than a tile, so it
skips `scatter_readback`).

## Layout

```
README.md / README_zh.md         This file (English / 中文)
PLAN.md / PLAN_zh.md             Program plan (English / 中文)
LICENSE                          MIT
lakefile.toml, lean-toolchain    Lake build (Lean 4 v4.15.0; works on v4.29.0)
VeriTile.lean                    Top-level library entry

VeriTile/Triton/
  Core.lean                      Op / Stmt / Kernel data types
  Semantics.lean                 BlockState, evalOp (gather), stepStmt (scatter),
                                   exec, scatter_readback
  DSL.lean                       The `triton { ... }` macro
  Examples.lean                  Hand-built naiveSoftmax kernel (constructor form)

VeriTile/Examples/
  SoftmaxEq.lean                 Naive ↔ stable softmax (Tier 1 #1)
  LogSumExpEq.lean               Direct LSE ↔ shift-trick LSE (Tier 1 #2)
  SoftmaxReciprocal.lean         Stable softmax ↔ precomputed-reciprocal (Tier 1 #3)
  WelfordMath.lean               welford_eq_two_pass math lemma (Phase B prep)

scripts/
  prove.sh                       Bash wrapper around /lean4:autoprove for benchmark eval
  README.md                      Wrapper usage + pinned plugin version

bench/llm_eval/
  softmax_naive_correct_held_out.lean   Held-out copy for LLM eval
  README.md                      Eval protocol
  results/                       Per-trial logs + phase_a_report.md

Notes/
  proposal.md / proposal_zh.md   Project proposal (English / 中文)
  T3_scouting.md                 FlashAttention forward feasibility study
                                   (Phase A deliverable, ~15 pages)
  MacroOptions.md                Tech investigation: macro implementation tradeoffs
  2026-04-26-phase-a-implementation.md   Phase A implementation plan
```

## Roadmap

**Phase A — Tier 1 + LLM tooling + T3 scouting (done, [`v0.1-tier1`](https://github.com/Lizn-zn/VeriTile/releases/tag/v0.1-tier1)):**
- [x] Project layout, lakefile, lean-toolchain
- [x] `Op` / `Stmt` / `Kernel` types + `evalOp`, `stepStmt`, `exec`
- [x] `triton { ... }` macro + `tl.log` (`Op.log`)
- [x] `softmax_kernels_refinement` (#1), `log_sum_exp_refinement` (#2),
      `softmax_reciprocal_refinement` (#3) — full proofs
- [x] `welford_eq_two_pass` math lemma (Phase B prep)
- [x] `scripts/prove.sh` LLM proof-drafting wrapper
- [x] `bench/llm_eval/` held-out evaluation harness
- [x] T3 forward feasibility study (`Notes/T3_scouting.md`)

**Phase B — forLoop semantics + Tier 2 (next):**
- [ ] `forLoop` operational semantics (mutual recursion + `termination_by`)
- [ ] `forLoop_inv` invariant-reasoning lemma (with index register binding)
- [ ] `welford_kernels_refinement` (#4) — kernel-level lift of `welford_eq_two_pass`
- [ ] `online_softmax_recurrence_eq_batch` (#5) — *FlashAttention algorithmic core*
- [ ] `layernorm_kernels_refinement` (#6) — fused single-pass vs two-pass
- [ ] Differential testing artifact for 1–2 selected representative kernels

**Phase C — `tl.dot` + masking + FA forward (Tier 3-A):**
- [ ] `Value.tile2D` 2D-matrix tile constructor + `Op.dot` semantics
- [ ] Comparison ops, `Op.where` for masking, multi-axis broadcast
- [ ] `fa_forward_correct` (#7) — FA-1 forward ≡ standard attention denotation

**Phase D — multi-block + FA-1 ↔ FA-2 + paper (Tier 3-B):**
- [ ] Multi-block grid execution model + disjoint-writes lemma
- [ ] `fa_2_forward_correct` (#8) — FA-2 forward ≡ standard attention denotation
- [ ] `fa1_eq_fa2` headline corollary by spec transitivity
- [ ] PLDI / OOPSLA-spring submission

**Beyond the program (P3+):**
- [ ] Python lifter (real `.py` Triton → embedded `triton { ... }` term)
- [ ] FA-backward, more kernel families (RoPE, paged attention, ...)
- [ ] IEEE-754 fidelity (refining the `ℝ` model)

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

## License

[MIT](./LICENSE) © 2026 Zenan Li.
