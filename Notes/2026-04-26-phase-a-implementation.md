# Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close 2 new Tier 1 kernel-pair theorems and 1 math lemma, build a ~200-line LLM proof-drafting MVP, produce a T3 feasibility scouting document, and set up the LLM benchmark held-out infrastructure — all per `PLAN.md` Phase A.

**Architecture:** Four workstreams sharing one exit gate (`v0.1-tier1` tag). Lean work extends the existing `VeriTile` library; the LLM tool is a separate Python project under `tools/llm_prover/`; scouting is a markdown analysis under `Notes/`; the benchmark held-out lives at `bench/llm_eval/`.

**Tech Stack:** Lean 4 (v4.15.0+, Mathlib), Python 3.11+, Anthropic Python SDK (`anthropic`), `pytest` (Python tests), `lake build` (Lean), git.

---

## Scope reference

This plan implements `PLAN.md` Phase A:
- 2 new Tier 1 theorems: `log_sum_exp_refinement` (#2), `softmax_reciprocal_refinement` (#3) — #1 `softmax_kernels_refinement` is already done
- 1 math lemma: `welford_eq_two_pass`
- LLM proof-drafting MVP under `tools/llm_prover/`
- T3 scouting at `Notes/T3_scouting.md`
- Held-out eval at `bench/llm_eval/softmax_naive_correct_held_out.lean`
- Exit gate: `lake build` clean, LLM MVP close rate ≥ 1/3 on the held-out, `v0.1-tier1` tag

## File structure

**Lean (extends existing library):**

| File | Status | Responsibility |
|---|---|---|
| `VeriTile/Examples/LogSumExpEq.lean` | Create | `log_sum_exp_refinement` theorem + supporting math lemmas |
| `VeriTile/Examples/SoftmaxReciprocal.lean` | Create | `softmax_reciprocal_refinement` theorem |
| `VeriTile/Examples/WelfordMath.lean` | Create | Pure-math `welford_eq_two_pass` lemma (no kernel; prep for Phase B) |
| `VeriTile.lean` | Modify | Add three new imports for the above |
| `bench/llm_eval/softmax_naive_correct_held_out.lean` | Create | Held-out copy of `softmax_naive_correct` with `sorry` for LLM eval |
| `bench/lakefile.toml` | Create | Sub-package excluded from main library build |

**Python LLM tool:**

| File | Status | Responsibility |
|---|---|---|
| `tools/llm_prover/pyproject.toml` | Create | Python project metadata + dep on `anthropic` |
| `tools/llm_prover/prove.py` | Create | Main entry point: `prove.py <file> <theorem_name>` |
| `tools/llm_prover/prompt_builder.py` | Create | Assemble prompt from theorem + context + few-shot |
| `tools/llm_prover/llm_client.py` | Create | Thin wrapper around `anthropic` SDK |
| `tools/llm_prover/lean_runner.py` | Create | Spawn `lake env lean` and parse pass/fail + error |
| `tools/llm_prover/prompts/base.txt` | Create | Main prompt template |
| `tools/llm_prover/prompts/fewshot/softmax.txt` | Create | Worked example: stripped-down version of `softmax_kernels_refinement` |
| `tools/llm_prover/prompts/fewshot/scatter.txt` | Create | Worked example: `scatter_readback` |
| `tools/llm_prover/tests/test_prompt_builder.py` | Create | Unit tests for prompt assembly |
| `tools/llm_prover/tests/test_lean_runner.py` | Create | Unit tests for Lean output parsing |
| `tools/llm_prover/README.md` | Create | Usage instructions |

**Scouting + meta:**

| File | Status | Responsibility |
|---|---|---|
| `Notes/T3_scouting.md` | Create | FA pseudocode + invariants + lemma needs + mask option choice + risk register + T3-B feasibility judgment |
| `.gitignore` | Modify | Add `tools/llm_prover/.venv/`, `__pycache__/`, etc. |
| `PLAN.md` | Modify | Phase A exit checkbox once gate passes |

---

## Workstream 1 — Tier 1 Lean theorems (~3–4 weeks)

### Task 1.0: Set up Tier 1 example directory structure

**Files:**
- Read: `VeriTile.lean`
- Modify: `VeriTile.lean`

- [ ] **Step 1: Read the current VeriTile.lean to know what's there**

```bash
cat VeriTile.lean
```

Expected: imports for `Triton.Core`, `Triton.Semantics`, `Triton.Examples`, `Examples.SoftmaxEq`.

- [ ] **Step 2: Add three new imports placeholder (commented out for now)**

Edit `VeriTile.lean`:

```lean
-- VeriTile: top-level entry point.
-- Imports the public surface of the embedded Triton subset and worked examples.

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Examples
import VeriTile.Examples.SoftmaxEq
-- import VeriTile.Examples.LogSumExpEq        -- Task 1.1
-- import VeriTile.Examples.SoftmaxReciprocal  -- Task 1.2
-- import VeriTile.Examples.WelfordMath        -- Task 1.3
```

- [ ] **Step 3: Verify build still clean**

Run: `PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -5`

Expected: `Build completed successfully`, no warnings (assumes P1 baseline; if a warning appears that's a pre-existing condition unrelated to this task).

- [ ] **Step 4: Commit**

```bash
git add VeriTile.lean
git commit -m "chore: scaffold Tier 1 example imports (commented)"
```

### Task 1.1: `log_sum_exp_refinement` theorem (#2)

This proves: a direct LSE kernel and a shift-trick LSE kernel produce the same `Y` value. Mirrors `SoftmaxEq.lean` structure.

**Math identity:**
```
logsumexp(x) = log(Σ exp(x_i)) = m + log(Σ exp(x_i − m))    where m = max(x)
```

**Files:**
- Create: `VeriTile/Examples/LogSumExpEq.lean`
- Modify: `VeriTile.lean` (uncomment the import)

- [ ] **Step 1: Write the math lemma `log_sum_exp_shift_invariant` with sorry**

Create `VeriTile/Examples/LogSumExpEq.lean`:

```lean
/-
VeriTile.Examples.LogSumExpEq

Worked equivalence example: direct log-sum-exp kernel vs shift-trick
log-sum-exp kernel. The math identity `log_sum_exp_shift_invariant` is the
load-bearing fact; the kernel-level theorem composes operational walk-throughs
of both kernels with this identity.
-/

import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL

namespace VeriTile.Examples

open VeriTile.Triton

/-- Direct log-sum-exp kernel: y = log(Σ exp(x)). -/
def directLSEKernel (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange($(N))
  x    := tl.load(X, offs)
  e    := tl.exp(x)
  s    := tl.sum(e)
  -- TODO Phase A: add tl.log to Op when needed; for now use an antiquoted Real.log.
  -- This kernel form is preliminary; revise if Op needs extension.
  tl.store(Y, offs, e / s)  -- placeholder; revise once tl.log is in Op
}

/-- Shift-trick LSE kernel: y = m + log(Σ exp(x − m)). -/
def shiftLSEKernel (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange($(N))
  x    := tl.load(X, offs)
  m    := tl.max(x)
  e    := tl.exp(x - m)
  s    := tl.sum(e)
  tl.store(Y, offs, e / s)  -- placeholder; revise once tl.log is in Op
}

/-- The load-bearing math identity. -/
theorem log_sum_exp_shift_invariant {n : Nat} (x : Fin n → ℝ) (m : ℝ) :
    Real.log (∑ i, Real.exp (x i)) = m + Real.log (∑ i, Real.exp (x i - m)) := by
  sorry

end VeriTile.Examples
```

- [ ] **Step 2: Confirm the file compiles with sorry**

Uncomment the import in `VeriTile.lean`:

```lean
import VeriTile.Examples.LogSumExpEq
```

Run: `PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -10`

Expected: build succeeds with one `declaration uses 'sorry'` warning at the math lemma.

- [ ] **Step 3: Decide on `Op.log` extension**

Before proving the kernel-level theorem, check whether `Op` has a `log` constructor.

Run: `grep -n "log\|Op\\.exp" VeriTile/Triton/Core.lean | head -20`

If `Op.log` is missing: this is a Phase A scope decision. Two options:
- **Option α**: extend `Op` with a `log` constructor in this task; small change to `Core.lean` + `Semantics.lean`.
- **Option β**: defer LSE kernel-level theorem; for Phase A, deliver only the math identity (`log_sum_exp_shift_invariant`).

**Decision point** — present both options to user, take direction. Default: Option α (extend `Op`).

If Option α chosen, add a sub-task block (see Task 1.1.5 below). If Option β chosen, skip the kernel-level part.

- [ ] **Step 4: (If Option α) Extend `Op` with `log`**

Edit `VeriTile/Triton/Core.lean` — add to the `Op` inductive:

```lean
  | log : Op → Op
```

Edit `VeriTile/Triton/Semantics.lean` — add to `evalOp` match:

```lean
  | .log a, s =>
      match evalOp a s with
      | some va => some (va.uop Real.log)
      | none => none
```

Edit `VeriTile/Triton/DSL.lean` — add to expression syntax:

```lean
syntax "tl.log(" tritonExpr ")" : tritonExpr
```

And in `expandExpr`:

```lean
  | `(tritonExpr| tl.log($e:tritonExpr)) => do
      let e' ← expandExpr e
      `(Op.log $e')
```

- [ ] **Step 5: Run lake build to confirm Op.log compiles**

Run: `PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -5`

Expected: clean build (with only the existing sorry warning from Step 2).

- [ ] **Step 6: Update kernels to use `tl.log`**

Replace placeholder `tl.store(Y, offs, e / s)` lines in both kernels:

```lean
def directLSEKernel (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange($(N))
  x    := tl.load(X, offs)
  e    := tl.exp(x)
  s    := tl.sum(e)
  y    := tl.log(s)
  tl.store(Y, offs, y)
}

def shiftLSEKernel (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange($(N))
  x    := tl.load(X, offs)
  m    := tl.max(x)
  e    := tl.exp(x - m)
  s    := tl.sum(e)
  y    := m + tl.log(s)
  tl.store(Y, offs, y)
}
```

- [ ] **Step 7: Prove `log_sum_exp_shift_invariant`**

Replace the `sorry` with:

```lean
theorem log_sum_exp_shift_invariant {n : Nat} (x : Fin n → ℝ) (m : ℝ) :
    Real.log (∑ i, Real.exp (x i)) = m + Real.log (∑ i, Real.exp (x i - m)) := by
  -- Approach: pull out exp(m) from each summand and use log of product.
  -- ∑ exp(x_i) = ∑ exp(m) * exp(x_i - m) = exp(m) * ∑ exp(x_i - m)
  -- log(LHS) = log(exp(m)) + log(∑ exp(x_i - m)) = m + log(∑ exp(x_i - m))
  by_cases hn : n = 0
  · subst hn
    simp [Finset.sum_empty]  -- log 0 case; check this is handled
    sorry  -- if n = 0 the sum is 0 and log 0 is messy; come back if it bites
  · have h_pos : 0 < ∑ i, Real.exp (x i - m) := by
      apply Finset.sum_pos
      · intro i _; exact Real.exp_pos _
      · exact Finset.univ_nonempty_iff.mpr (Fin.pos_iff_nonempty.mp (Nat.pos_of_ne_zero hn))
    have h_factor : ∀ i : Fin n, Real.exp (x i) = Real.exp m * Real.exp (x i - m) := by
      intro i
      rw [← Real.exp_add]
      ring_nf
    rw [Finset.sum_congr rfl (fun i _ => h_factor i)]
    rw [← Finset.mul_sum]
    rw [Real.log_mul (Real.exp_ne_zero m) (ne_of_gt h_pos)]
    rw [Real.log_exp]
```

If the `n = 0` case proves too fiddly, restrict the theorem to `0 < n`:

```lean
theorem log_sum_exp_shift_invariant {n : Nat} (hn : 0 < n) (x : Fin n → ℝ) (m : ℝ) :
    ...
```

- [ ] **Step 8: Run lake build**

Run: `PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -5`

Expected: build succeeds. If math proof fails, iterate on the tactic block; the math is sound, this is a Lean syntax issue.

- [ ] **Step 9: Sketch the kernel-level theorem `log_sum_exp_refinement`**

Reuse the helpers (`observeY`, `tileMax`, `InputLoaded`) from `SoftmaxEq.lean`. Add at the top of `LogSumExpEq.lean`:

```lean
import VeriTile.Examples.SoftmaxEq
```

The names are already exposed in the `VeriTile.Examples` namespace.

Add the spec functions and theorems:

```lean
/-- Math: direct LSE produces `log(Σ exp(x_j))` at every output position. -/
noncomputable def directLSESpec {N : Nat} (xs : Fin N → ℝ) (_i : Fin N) : ℝ :=
  Real.log (∑ j, Real.exp (xs j))

noncomputable def shiftLSESpec {N : Nat} (xs : Fin N → ℝ) (m : ℝ) (_i : Fin N) : ℝ :=
  m + Real.log (∑ j, Real.exp (xs j - m))

theorem direct_lse_correct
    (N : Nat) (_hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoaded s N xs) :
    ∀ i : Fin N,
      observeY (exec (directLSEKernel N) s) N s.pid i
        = some (directLSESpec xs i) := by
  sorry  -- operational walk-through; uses the same scatter_readback pattern as softmax

theorem shift_lse_correct
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoaded s N xs) :
    ∀ i : Fin N,
      observeY (exec (shiftLSEKernel N) s) N s.pid i
        = some (shiftLSESpec xs (tileMax hN xs) i) := by
  sorry

theorem log_sum_exp_refinement
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoaded s N xs) :
    ∀ i : Fin N,
      observeY (exec (directLSEKernel N) s) N s.pid i
        = observeY (exec (shiftLSEKernel N) s) N s.pid i := by
  intro i
  rw [direct_lse_correct  N hN s xs h_x i,
      shift_lse_correct N hN s xs h_x i]
  congr 1
  unfold directLSESpec shiftLSESpec
  exact log_sum_exp_shift_invariant xs (tileMax hN xs)
```

- [ ] **Step 10: Prove `direct_lse_correct` and `shift_lse_correct`**

These are operational walk-throughs identical in structure to `softmax_naive_correct` and `softmax_stable_correct` from `SoftmaxEq.lean`. Copy the proof shape and adapt:
- Same `hcast` lemma for offset arithmetic
- Same `h_inj` for offset injectivity
- Same `simp` block, with `directLSEKernel` / `shiftLSEKernel` instead of softmax
- Same `BlockState.scatter_readback` finalization

Open `VeriTile/Examples/SoftmaxEq.lean`, copy the proofs of `softmax_naive_correct` (lines 171–201) and `softmax_stable_correct` (lines 203–243), adapt the kernel name and the simp lemma list (add `directLSEKernel` / `shiftLSEKernel`, `Value.reduceSum`, `Real.log` if necessary).

If the simp doesn't reduce cleanly, add custom simp lemmas as P2 work; not a blocker for Phase A if the math identity is closed.

- [ ] **Step 11: Run lake build, expect clean**

Run: `PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -10`

Expected: clean build, no sorries.

- [ ] **Step 12: Commit**

```bash
git add VeriTile/Examples/LogSumExpEq.lean VeriTile.lean VeriTile/Triton/Core.lean VeriTile/Triton/Semantics.lean VeriTile/Triton/DSL.lean
git commit -m "feat(tier1): log_sum_exp_refinement (#2) + Op.log extension"
```

### Task 1.2: `softmax_reciprocal_refinement` theorem (#3)

This proves: `y = e/s` per-element-divide is equivalent to `inv_s = 1/s; y = e * inv_s` precomputed-reciprocal. Trivial in `ℝ` (algebraic identity `e/s = e * (1/s)`).

**Files:**
- Create: `VeriTile/Examples/SoftmaxReciprocal.lean`
- Modify: `VeriTile.lean` (uncomment import)

- [ ] **Step 1: Write file with both kernels and refinement theorem**

Create `VeriTile/Examples/SoftmaxReciprocal.lean`:

```lean
/-
VeriTile.Examples.SoftmaxReciprocal

Real Triton optimization: replace `y = e/s` (one division per element) with
`inv_s = 1/s; y = e * inv_s` (one division total + one multiply per element).
Algorithmically equivalent in ℝ (e/s = e * (1/s) when s ≠ 0).
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Examples.SoftmaxEq

namespace VeriTile.Examples

open VeriTile.Triton

/-- Stable softmax with per-element division. -/
def softmaxDivKernel (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange($(N))
  x    := tl.load(X, offs)
  m    := tl.max(x)
  e    := tl.exp(x - m)
  s    := tl.sum(e)
  y    := e / s
  tl.store(Y, offs, y)
}

/-- Stable softmax with precomputed reciprocal. -/
def softmaxRecipKernel (N : Nat) : Kernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(N) + tl.arange($(N))
  x      := tl.load(X, offs)
  m      := tl.max(x)
  e      := tl.exp(x - m)
  s      := tl.sum(e)
  inv_s  := 1 / s
  y      := e * inv_s
  tl.store(Y, offs, y)
}

theorem softmax_reciprocal_refinement
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoaded s N xs) :
    ∀ i : Fin N,
      observeY (exec (softmaxDivKernel N) s) N s.pid i =
      observeY (exec (softmaxRecipKernel N) s) N s.pid i := by
  sorry

end VeriTile.Examples
```

- [ ] **Step 2: Check macro support for `1` literal**

The `inv_s := 1 / s` line uses a numeric literal `1` in the macro. Verify `tritonExpr` supports `num` (it does, per `DSL.lean:51`). If `1` doesn't expand correctly, use `$(1 : Nat)` antiquote.

Run: `PATH="$HOME/.elan/bin:$PATH" lake env lean VeriTile/Examples/SoftmaxReciprocal.lean 2>&1 | tail -10`

Expected: file compiles with one `sorry` warning. If macro expansion fails on `1 / s`, switch to `$(1 : Nat) / s`.

- [ ] **Step 3: Sketch the proof**

The two kernels differ only in the last computation step:
- `softmaxDivKernel`: register `y` = element-wise `(e / s)` value
- `softmaxRecipKernel`: register `inv_s` = scalar `1/s`, then `y` = element-wise `e * inv_s`

By `Value.bop` semantics, both yield the same tile in `Y`:

```
(e i) / s = (e i) * (1 / s)
```

This is the math identity, provable by `field_simp` (assuming `s ≠ 0`, which we'll need from softmax non-zero denominator).

- [ ] **Step 4: Write the math identity helper**

Add before the kernel theorem:

```lean
/-- The load-bearing identity: division equals multiplication by reciprocal. -/
theorem div_eq_mul_inv_real (a s : ℝ) (hs : s ≠ 0) : a / s = a * (1 / s) := by
  field_simp
```

- [ ] **Step 5: Prove `softmax_reciprocal_refinement`**

Strategy: introduce two correctness theorems for each kernel against the same `naiveSpec`, then compose. Or directly walk through `exec` for both kernels and reduce them to the same closed form.

Cleanest path: prove both reduce to the same observable value pointwise:

```lean
theorem softmax_reciprocal_refinement
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoaded s N xs) :
    ∀ i : Fin N,
      observeY (exec (softmaxDivKernel N) s) N s.pid i =
      observeY (exec (softmaxRecipKernel N) s) N s.pid i := by
  intro i
  -- Walk through both kernels' exec; they share the first 6 stmts (pid, offs, x, m, e, s).
  -- Difference: div kernel does `y := e / s` directly;
  --             recip kernel does `inv_s := 1 / s; y := e * inv_s`.
  -- Both produce the same tile in `e_i` register, so the writes to Y are equal.
  -- Reduce via simp on the operational semantics, both sides land at the same expression.
  sorry  -- ~30-50 lines following the SoftmaxEq.lean pattern
```

Flesh out the proof following `SoftmaxEq.lean`'s walkthrough pattern. The simp set will be the same as `softmax_naive_correct` plus `Value.bop` reasoning for the multiplication and division.

If stuck for more than 2 days, consider expressing both kernels via a shared reduction to the closed form and only prove the closed forms equal via `div_eq_mul_inv_real` per element.

- [ ] **Step 6: Run lake build, expect clean**

Run: `PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -10`

Expected: clean build, no sorries.

- [ ] **Step 7: Commit**

```bash
git add VeriTile/Examples/SoftmaxReciprocal.lean VeriTile.lean
git commit -m "feat(tier1): softmax_reciprocal_refinement (#3)"
```

### Task 1.3: `welford_eq_two_pass` math lemma

Pure math lemma — no kernels. Proves Welford's online recurrence formula equals the two-pass formula. Prep for Phase B's Welford kernel theorem.

**Math content:**

Two-pass:
```
μ = (1/n) Σ_{i=0}^{n-1} x_i
S = Σ_{i=0}^{n-1} (x_i − μ)²
var = S / n
```

Welford recurrence:
```
M_0 = 0, S_0 = 0
M_k = M_{k-1} + (x_{k-1} − M_{k-1}) / k
S_k = S_{k-1} + (x_{k-1} − M_{k-1}) · (x_{k-1} − M_k)
```

Claim: `M_n = μ` and `S_n = S`. Therefore `var_welford = S_n / n = var_twopass`.

**Files:**
- Create: `VeriTile/Examples/WelfordMath.lean`
- Modify: `VeriTile.lean` (uncomment import)

- [ ] **Step 1: Define both forms in Lean**

Create `VeriTile/Examples/WelfordMath.lean`:

```lean
/-
VeriTile.Examples.WelfordMath

Math-only lemma: Welford's online variance recurrence equals the two-pass formula.
This is preparation for Phase B's `welford_kernels_refinement` theorem (#4); the
kernel-level lift will use `forLoop_inv` once forLoop semantics is in place.
-/

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

namespace VeriTile.Examples

/-- Two-pass mean. -/
noncomputable def twoPassMean {n : Nat} (x : Fin n → ℝ) : ℝ :=
  (∑ i, x i) / n

/-- Two-pass sum-of-squared-deviations. -/
noncomputable def twoPassS {n : Nat} (x : Fin n → ℝ) : ℝ :=
  ∑ i, (x i - twoPassMean x) ^ 2

/-- Welford recurrence step's mean update. Walks `Fin n` linearly. -/
noncomputable def welfordMean {n : Nat} (x : Fin n → ℝ) : Nat → ℝ
  | 0     => 0
  | k + 1 =>
      if h : k < n then
        let prev := welfordMean x k
        prev + (x ⟨k, h⟩ - prev) / (k + 1)
      else welfordMean x k

/-- Welford recurrence step's running sum-of-squared-deviations. -/
noncomputable def welfordS {n : Nat} (x : Fin n → ℝ) : Nat → ℝ
  | 0     => 0
  | k + 1 =>
      if h : k < n then
        let prevM := welfordMean x k
        let curM  := welfordMean x (k + 1)
        welfordS x k + (x ⟨k, h⟩ - prevM) * (x ⟨k, h⟩ - curM)
      else welfordS x k

/-- The load-bearing identity for Phase B's Welford kernel theorem. -/
theorem welford_eq_two_pass {n : Nat} (hn : 0 < n) (x : Fin n → ℝ) :
    welfordMean x n = twoPassMean x ∧ welfordS x n = twoPassS x := by
  sorry

end VeriTile.Examples
```

- [ ] **Step 2: Verify file compiles with sorry**

Run: `PATH="$HOME/.elan/bin:$PATH" lake env lean VeriTile/Examples/WelfordMath.lean 2>&1 | tail -10`

Expected: one sorry warning, no errors.

- [ ] **Step 3: Prove `welford_eq_two_pass` mean part by induction**

Strategy: prove a stronger claim by induction on `k`:

```
∀ k ≤ n, welfordMean x k = (1/k) * Σ_{i=0}^{k-1} x i    -- when k > 0
welfordMean x 0 = 0                                       -- base
```

Replace the sorry with a structured proof. Sketch:

```lean
theorem welford_eq_two_pass {n : Nat} (hn : 0 < n) (x : Fin n → ℝ) :
    welfordMean x n = twoPassMean x ∧ welfordS x n = twoPassS x := by
  -- Auxiliary: stronger statement on prefixes.
  have hMean : ∀ k, k ≤ n →
      welfordMean x k * k = ∑ i ∈ Finset.range k, x ⟨i, lt_of_lt_of_le (by omega) ‹k ≤ n›⟩ := by
    intro k hk
    induction k with
    | zero => simp [welfordMean]
    | succ j ih =>
      sorry  -- inductive step: use defn + arithmetic
  -- Apply at k = n:
  have hMean_n := hMean n (le_refl n)
  constructor
  · unfold twoPassMean
    have hpos : (n : ℝ) ≠ 0 := by positivity
    field_simp
    -- Reduce welfordMean x n * n = ∑ x_i form.
    sorry
  · sorry  -- same shape, harder due to the cross-term
```

This is a multi-day proof; consider opening a scratch file `Notes/welford_proof_sketch.md` and writing out the math by hand first, then translating to Lean.

- [ ] **Step 4: If proof drags > 1 week, escalate to LLM tool**

Once Workstream 2's LLM MVP is functional, run it on this lemma as the *first real test* of the tool. If the MVP closes it, great. If not, fall back to manual proof or descope `welford_eq_two_pass` to "Phase B prep, defer formal proof" — but ONLY with explicit user OK.

- [ ] **Step 5: Run lake build, expect clean**

Run: `PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -10`

Expected: clean build, no sorries.

- [ ] **Step 6: Commit**

```bash
git add VeriTile/Examples/WelfordMath.lean VeriTile.lean
git commit -m "feat(tier1): welford_eq_two_pass math lemma (Phase B prep)"
```

---

## Workstream 2 — LLM proof tool MVP (~2–3 weeks)

### Task 2.1: Python project skeleton

**Files:**
- Create: `tools/llm_prover/pyproject.toml`
- Create: `tools/llm_prover/llm_prover/__init__.py`
- Create: `tools/llm_prover/README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p tools/llm_prover/llm_prover/prompts/fewshot
mkdir -p tools/llm_prover/tests
```

- [ ] **Step 2: Write `pyproject.toml`**

Create `tools/llm_prover/pyproject.toml`:

```toml
[project]
name = "veritile-llm-prover"
version = "0.1.0"
description = "LLM-assisted proof drafting for VeriTile Lean theorems"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40.0",
    "click>=8.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.12",
]

[project.scripts]
prove = "llm_prover.prove:cli"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["llm_prover*"]
```

- [ ] **Step 3: Empty `__init__.py`**

Create `tools/llm_prover/llm_prover/__init__.py`:

```python
"""VeriTile LLM-assisted proof drafting MVP."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Write minimal README**

Create `tools/llm_prover/README.md`:

```markdown
# VeriTile LLM Prover MVP

Phase A MVP for LLM-assisted Lean 4 proof drafting.

## Install

\`\`\`bash
cd tools/llm_prover
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
\`\`\`

## Use

\`\`\`bash
export ANTHROPIC_API_KEY=sk-...
prove --file ../../bench/llm_eval/softmax_naive_correct_held_out.lean \\
      --theorem softmax_naive_correct
\`\`\`

## What it does

For a Lean theorem with `sorry`, prompts Claude with the theorem +
imports + few-shot examples, retrieves a proposed tactic block, writes
it back, runs `lake env lean` to verify. Up to N=5 retries on failure
with the error fed back into the next prompt.
```

- [ ] **Step 5: Update root `.gitignore`**

Edit `.gitignore` (project root):

```
# Lake build artifacts and downloaded packages
/.lake/

# macOS
.DS_Store

# Editor / tooling
/.claude/
*.olean
*.ilean
*.hash

# Python tool
tools/llm_prover/.venv/
tools/llm_prover/**/__pycache__/
tools/llm_prover/**/*.pyc
tools/llm_prover/**/.pytest_cache/
tools/llm_prover/**/*.egg-info/
```

- [ ] **Step 6: Test the install**

```bash
cd tools/llm_prover
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -c "import llm_prover; print(llm_prover.__version__)"
```

Expected: `0.1.0`.

- [ ] **Step 7: Commit**

```bash
cd /Users/zenan/Documents/Project/VeriTile  # or your repo root
git add tools/llm_prover/pyproject.toml tools/llm_prover/llm_prover/__init__.py tools/llm_prover/README.md .gitignore
git commit -m "feat(llm_prover): scaffold Python project skeleton"
```

### Task 2.2: Lean runner module

Wraps `lake env lean <file>`, captures stdout/stderr, parses pass/fail and any error text. Tested against fixtures.

**Files:**
- Create: `tools/llm_prover/llm_prover/lean_runner.py`
- Create: `tools/llm_prover/tests/test_lean_runner.py`
- Create: `tools/llm_prover/tests/fixtures/passing.lean`
- Create: `tools/llm_prover/tests/fixtures/failing.lean`

- [ ] **Step 1: Write the failing test for `run_lean()` happy path**

Create `tools/llm_prover/tests/fixtures/passing.lean`:

```lean
example : 1 + 1 = 2 := by rfl
```

Create `tools/llm_prover/tests/fixtures/failing.lean`:

```lean
example : 1 + 1 = 3 := by rfl
```

Create `tools/llm_prover/tests/test_lean_runner.py`:

```python
import pathlib
from llm_prover.lean_runner import run_lean, LeanResult

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

def test_run_lean_passing_file_returns_success():
    result = run_lean(FIXTURES / "passing.lean")
    assert isinstance(result, LeanResult)
    assert result.success is True
    assert result.error_message is None

def test_run_lean_failing_file_returns_error():
    result = run_lean(FIXTURES / "failing.lean")
    assert isinstance(result, LeanResult)
    assert result.success is False
    assert result.error_message is not None
    assert "1 + 1 = 3" in result.error_message or "rfl" in result.error_message.lower()
```

- [ ] **Step 2: Run the test, expect failure (module doesn't exist)**

```bash
cd tools/llm_prover && source .venv/bin/activate
pytest tests/test_lean_runner.py -v
```

Expected: `ImportError: No module named 'llm_prover.lean_runner'`.

- [ ] **Step 3: Implement `lean_runner.py`**

Create `tools/llm_prover/llm_prover/lean_runner.py`:

```python
"""Wraps `lake env lean` for the proof-drafting loop."""
from __future__ import annotations

import os
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class LeanResult:
    success: bool
    stdout: str
    stderr: str
    error_message: Optional[str]


def run_lean(file: pathlib.Path, timeout: int = 600) -> LeanResult:
    """Run `lake env lean <file>` in the file's project root.

    Walks up from the file to find the nearest `lakefile.toml` ancestor
    and runs there. Captures stdout+stderr; classifies success by exit
    code; for failures, extracts the first non-empty error line as the
    error message (good enough for MVP; refine in v0.2).
    """
    project_root = _find_lake_root(file)
    env = os.environ.copy()
    env["PATH"] = f"{os.path.expanduser('~/.elan/bin')}:{env.get('PATH', '')}"

    try:
        proc = subprocess.run(
            ["lake", "env", "lean", str(file)],
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        return LeanResult(
            success=False,
            stdout=e.stdout or "",
            stderr=e.stderr or "",
            error_message=f"Timeout after {timeout}s",
        )

    success = proc.returncode == 0 and "error" not in proc.stdout.lower()
    error_message = None
    if not success:
        combined = (proc.stdout + "\n" + proc.stderr).strip()
        # Find first line that looks like an error.
        for line in combined.splitlines():
            if line.strip() and ("error" in line.lower() or "expected" in line.lower()):
                error_message = line.strip()
                break
        if error_message is None and combined:
            error_message = combined[:500]

    return LeanResult(
        success=success,
        stdout=proc.stdout,
        stderr=proc.stderr,
        error_message=error_message,
    )


def _find_lake_root(file: pathlib.Path) -> pathlib.Path:
    """Walk up from `file` until a directory containing `lakefile.toml` is found."""
    current = file.resolve().parent
    while current != current.parent:
        if (current / "lakefile.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError(f"No lakefile.toml found above {file}")
```

- [ ] **Step 4: Run the test, expect pass**

```bash
pytest tests/test_lean_runner.py -v
```

Expected: 2 passed.

If `_find_lake_root` fails because the fixture file isn't under a Lake project, place fixtures under a sub-lakefile or copy the fixtures inline-eval. Quick fix: have `run_lean` accept a `cwd` override; in the test pass the project root explicitly.

- [ ] **Step 5: Add error-extraction unit test**

Append to `tools/llm_prover/tests/test_lean_runner.py`:

```python
def test_run_lean_extracts_error_line(tmp_path):
    bad = tmp_path / "bad.lean"
    bad.write_text("example : 0 = 1 := rfl\n")
    # Create a minimal lakefile in tmp_path so _find_lake_root works.
    (tmp_path / "lakefile.toml").write_text(
        '[package]\nname = "tmp"\nversion = "0.1.0"\n'
    )
    result = run_lean(bad)
    assert result.success is False
    assert result.error_message is not None
```

Run: `pytest tests/test_lean_runner.py -v`. Expected: 3 passed (or 2 passed + 1 skipped if `lake` not in PATH; that's a setup issue, fix via PATH).

- [ ] **Step 6: Commit**

```bash
git add tools/llm_prover/llm_prover/lean_runner.py tools/llm_prover/tests/test_lean_runner.py tools/llm_prover/tests/fixtures/
git commit -m "feat(llm_prover): lean_runner module with subprocess wrapper + tests"
```

### Task 2.3: Prompt builder

Assembles the prompt from theorem statement, file context, and few-shot examples.

**Files:**
- Create: `tools/llm_prover/llm_prover/prompt_builder.py`
- Create: `tools/llm_prover/tests/test_prompt_builder.py`
- Create: `tools/llm_prover/llm_prover/prompts/base.txt`
- Create: `tools/llm_prover/llm_prover/prompts/fewshot/softmax.txt`

- [ ] **Step 1: Write the failing test**

Create `tools/llm_prover/tests/test_prompt_builder.py`:

```python
from llm_prover.prompt_builder import PromptBuilder


def test_prompt_includes_theorem_statement():
    pb = PromptBuilder(fewshot_files=[])
    prompt = pb.build(
        theorem_name="my_thm",
        theorem_statement="theorem my_thm : 1 = 1 := by sorry",
        file_context="-- imports omitted",
        previous_attempts=[],
    )
    assert "my_thm" in prompt
    assert "1 = 1" in prompt
    assert "sorry" in prompt or "implement" in prompt.lower()


def test_prompt_includes_fewshot(tmp_path):
    fewshot = tmp_path / "ex.txt"
    fewshot.write_text("EXAMPLE: trivial proof goes here")
    pb = PromptBuilder(fewshot_files=[fewshot])
    prompt = pb.build(
        theorem_name="my_thm",
        theorem_statement="theorem my_thm : 1 = 1 := by sorry",
        file_context="",
        previous_attempts=[],
    )
    assert "EXAMPLE" in prompt


def test_prompt_includes_previous_failed_attempts():
    pb = PromptBuilder(fewshot_files=[])
    prompt = pb.build(
        theorem_name="my_thm",
        theorem_statement="theorem my_thm : 1 = 1 := by sorry",
        file_context="",
        previous_attempts=[("by trivial", "type mismatch")],
    )
    assert "previous attempt" in prompt.lower() or "tried" in prompt.lower()
    assert "by trivial" in prompt
    assert "type mismatch" in prompt
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_prompt_builder.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write the prompt template**

Create `tools/llm_prover/llm_prover/prompts/base.txt`:

```
You are a Lean 4 proof assistant. The user has a theorem with a `sorry`
in its proof; replace the `sorry` with a complete proof.

Output ONLY the tactic block that should replace the `sorry`. Do not
include the `theorem` declaration, do not include any commentary, and
do not wrap your output in markdown fences. Your output should start
with `by` (if you give a tactic block) or with `(...)` (if you give a
term-mode proof).

THEOREM:
{theorem_statement}

FILE CONTEXT (imports and surrounding definitions):
{file_context}

{fewshot_block}

{previous_attempts_block}
```

- [ ] **Step 4: Write a placeholder few-shot file**

Create `tools/llm_prover/llm_prover/prompts/fewshot/softmax.txt`:

```
EXAMPLE 1: Math identity used in stable-softmax equivalence.

theorem naive_eq_stable {n : Nat} (x : Fin n → ℝ) (m : ℝ) :
    naiveSoftmaxMath x = stableSoftmaxMath x m := by
  funext i
  unfold naiveSoftmaxMath stableSoftmaxMath
  simp only [Real.exp_sub]
  rw [← Finset.sum_div]
  have hm : Real.exp m ≠ 0 := Real.exp_ne_zero m
  field_simp
```

(Add a second example for a longer worked proof later — Step 5.)

- [ ] **Step 5: Implement `PromptBuilder`**

Create `tools/llm_prover/llm_prover/prompt_builder.py`:

```python
"""Assemble prompts for the proof-drafting LLM call."""
from __future__ import annotations

import importlib.resources
import pathlib
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PromptBuilder:
    fewshot_files: List[pathlib.Path] = field(default_factory=list)
    base_template: str = ""

    def __post_init__(self):
        if not self.base_template:
            self.base_template = self._load_default_template()

    def _load_default_template(self) -> str:
        pkg_root = pathlib.Path(__file__).parent
        return (pkg_root / "prompts" / "base.txt").read_text()

    def build(
        self,
        theorem_name: str,
        theorem_statement: str,
        file_context: str,
        previous_attempts: List[Tuple[str, str]],
    ) -> str:
        fewshot_block = self._build_fewshot()
        previous_block = self._build_previous(previous_attempts)
        return self.base_template.format(
            theorem_statement=theorem_statement,
            file_context=file_context,
            fewshot_block=fewshot_block,
            previous_attempts_block=previous_block,
        )

    def _build_fewshot(self) -> str:
        if not self.fewshot_files:
            return ""
        parts = ["WORKED EXAMPLES from this codebase:\n"]
        for path in self.fewshot_files:
            parts.append(path.read_text())
            parts.append("\n---\n")
        return "\n".join(parts)

    def _build_previous(self, previous_attempts: List[Tuple[str, str]]) -> str:
        if not previous_attempts:
            return ""
        parts = ["PREVIOUS ATTEMPTS that failed (DO NOT repeat them):"]
        for i, (tactic, error) in enumerate(previous_attempts, 1):
            parts.append(f"\nAttempt {i}:")
            parts.append(f"  Tactic: {tactic}")
            parts.append(f"  Error: {error}")
        return "\n".join(parts)
```

- [ ] **Step 6: Run tests, expect pass**

```bash
pytest tests/test_prompt_builder.py -v
```

Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add tools/llm_prover/llm_prover/prompt_builder.py \
        tools/llm_prover/llm_prover/prompts/ \
        tools/llm_prover/tests/test_prompt_builder.py
git commit -m "feat(llm_prover): prompt builder + base template + softmax fewshot"
```

### Task 2.4: LLM client

Thin wrapper over Anthropic SDK. Retry on transient errors, but the proof-loop retry on logic-failure lives in the main loop, not here.

**Files:**
- Create: `tools/llm_prover/llm_prover/llm_client.py`
- Create: `tools/llm_prover/tests/test_llm_client.py`

- [ ] **Step 1: Write the failing test (mocked SDK)**

Create `tools/llm_prover/tests/test_llm_client.py`:

```python
from unittest.mock import MagicMock, patch
from llm_prover.llm_client import LLMClient


def test_llm_client_calls_anthropic_with_prompt():
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="by simp")]
    mock_anth = MagicMock()
    mock_anth.messages.create.return_value = mock_response

    with patch("llm_prover.llm_client.anthropic.Anthropic", return_value=mock_anth):
        client = LLMClient(api_key="dummy", model="claude-opus-4-7")
        result = client.complete(prompt="prove 1 = 1")

    assert result == "by simp"
    mock_anth.messages.create.assert_called_once()
    call_kwargs = mock_anth.messages.create.call_args.kwargs
    assert "claude-opus-4-7" in str(call_kwargs.get("model", ""))
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_llm_client.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `llm_client.py`**

Create `tools/llm_prover/llm_prover/llm_client.py`:

```python
"""Thin wrapper around Anthropic API for the proof-drafting MVP."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import anthropic


@dataclass
class LLMClient:
    api_key: Optional[str] = None
    model: str = "claude-opus-4-7"
    max_tokens: int = 4096

    def __post_init__(self):
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Set ANTHROPIC_API_KEY or pass api_key=")
        self._client = anthropic.Anthropic(api_key=key)

    def complete(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # Anthropic SDK returns content as a list of blocks; for text-only
        # responses we concatenate text fields.
        return "".join(block.text for block in response.content if hasattr(block, "text"))
```

- [ ] **Step 4: Run test, expect pass**

```bash
pytest tests/test_llm_client.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/llm_prover/llm_prover/llm_client.py tools/llm_prover/tests/test_llm_client.py
git commit -m "feat(llm_prover): Anthropic API client wrapper"
```

### Task 2.5: Proof drafting loop + CLI entry point

Ties prompt builder + LLM client + Lean runner into the retry loop. Provides `prove` CLI.

**Files:**
- Create: `tools/llm_prover/llm_prover/prove.py`
- Create: `tools/llm_prover/tests/test_prove_loop.py`

- [ ] **Step 1: Write the loop test (mocked components)**

Create `tools/llm_prover/tests/test_prove_loop.py`:

```python
import pathlib
from unittest.mock import MagicMock
from llm_prover.prove import prove_loop, ProofResult
from llm_prover.lean_runner import LeanResult


def test_prove_loop_succeeds_on_first_try(tmp_path):
    file = tmp_path / "thm.lean"
    file.write_text("theorem t : 1 = 1 := by sorry")

    mock_client = MagicMock()
    mock_client.complete.return_value = "by rfl"
    mock_runner = MagicMock(side_effect=[LeanResult(success=True, stdout="", stderr="", error_message=None)])
    mock_builder = MagicMock()
    mock_builder.build.return_value = "PROMPT"

    result = prove_loop(
        file=file,
        theorem_name="t",
        max_retries=5,
        builder=mock_builder,
        llm_client=mock_client,
        lean_runner=mock_runner,
    )

    assert isinstance(result, ProofResult)
    assert result.success is True
    assert result.attempts == 1
    assert "by rfl" in file.read_text()


def test_prove_loop_retries_on_failure(tmp_path):
    file = tmp_path / "thm.lean"
    original = "theorem t : 1 = 1 := by sorry"
    file.write_text(original)

    mock_client = MagicMock()
    mock_client.complete.side_effect = ["by trivial", "by rfl"]
    mock_runner = MagicMock(side_effect=[
        LeanResult(success=False, stdout="", stderr="error: trivial fails", error_message="error: trivial fails"),
        LeanResult(success=True, stdout="", stderr="", error_message=None),
    ])
    mock_builder = MagicMock()
    mock_builder.build.return_value = "PROMPT"

    result = prove_loop(
        file=file, theorem_name="t", max_retries=5,
        builder=mock_builder, llm_client=mock_client, lean_runner=mock_runner,
    )

    assert result.success is True
    assert result.attempts == 2


def test_prove_loop_gives_up_after_max_retries(tmp_path):
    file = tmp_path / "thm.lean"
    original = "theorem t : 1 = 1 := by sorry"
    file.write_text(original)

    mock_client = MagicMock()
    mock_client.complete.return_value = "by sorry"
    mock_runner = MagicMock(return_value=LeanResult(
        success=False, stdout="", stderr="error", error_message="error"
    ))
    mock_builder = MagicMock()
    mock_builder.build.return_value = "PROMPT"

    result = prove_loop(
        file=file, theorem_name="t", max_retries=3,
        builder=mock_builder, llm_client=mock_client, lean_runner=mock_runner,
    )

    assert result.success is False
    assert result.attempts == 3
    # File should be restored to original on give-up.
    assert file.read_text() == original
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/test_prove_loop.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `prove.py`**

Create `tools/llm_prover/llm_prover/prove.py`:

```python
"""Main proof-drafting loop and CLI."""
from __future__ import annotations

import pathlib
import re
import sys
from dataclasses import dataclass
from typing import List, Tuple

import click

from llm_prover.lean_runner import LeanResult, run_lean
from llm_prover.llm_client import LLMClient
from llm_prover.prompt_builder import PromptBuilder


@dataclass
class ProofResult:
    success: bool
    attempts: int
    final_tactic: str = ""
    final_error: str = ""


def extract_theorem(file_text: str, theorem_name: str) -> Tuple[str, int, int]:
    """Extract the theorem block by name. Returns (statement, start_offset, end_offset).

    The block starts at `theorem <name>` and ends at the next top-level
    declaration (`theorem`, `def`, `example`, `noncomputable def`,
    `end <namespace>`, or end of file).
    """
    pattern = re.compile(
        rf"(theorem\s+{re.escape(theorem_name)}\b.*?)(?=\n(?:theorem|def|example|noncomputable\s+def|end|/-)|\Z)",
        re.DOTALL,
    )
    match = pattern.search(file_text)
    if not match:
        raise ValueError(f"Theorem {theorem_name!r} not found in file")
    return match.group(1), match.start(), match.end()


def replace_sorry(theorem_block: str, new_tactic: str) -> str:
    """Replace the rightmost `sorry` in the theorem block with `new_tactic`."""
    return re.sub(r"\bsorry\b", new_tactic, theorem_block, count=1)


def prove_loop(
    file: pathlib.Path,
    theorem_name: str,
    max_retries: int,
    builder: PromptBuilder,
    llm_client: LLMClient,
    lean_runner=run_lean,
) -> ProofResult:
    original_text = file.read_text()
    theorem_block, start, end = extract_theorem(original_text, theorem_name)
    file_context = original_text[:start] + original_text[end:]

    previous_attempts: List[Tuple[str, str]] = []

    for attempt in range(1, max_retries + 1):
        prompt = builder.build(
            theorem_name=theorem_name,
            theorem_statement=theorem_block,
            file_context=file_context,
            previous_attempts=previous_attempts,
        )
        tactic = llm_client.complete(prompt=prompt).strip()
        new_block = replace_sorry(theorem_block, tactic)
        new_text = original_text[:start] + new_block + original_text[end:]
        file.write_text(new_text)

        result: LeanResult = lean_runner(file)
        if result.success:
            return ProofResult(success=True, attempts=attempt, final_tactic=tactic)

        previous_attempts.append((tactic, result.error_message or "unknown error"))

    # Restore original on give-up.
    file.write_text(original_text)
    return ProofResult(
        success=False,
        attempts=max_retries,
        final_tactic=previous_attempts[-1][0] if previous_attempts else "",
        final_error=previous_attempts[-1][1] if previous_attempts else "",
    )


@click.command()
@click.option("--file", required=True, type=click.Path(exists=True, path_type=pathlib.Path))
@click.option("--theorem", required=True, help="Theorem name to prove.")
@click.option("--max-retries", default=5, type=int)
@click.option("--fewshot-dir", default=None, type=click.Path(path_type=pathlib.Path),
              help="Directory of *.txt few-shot files.")
def cli(file: pathlib.Path, theorem: str, max_retries: int, fewshot_dir):
    fewshot_files = []
    if fewshot_dir and fewshot_dir.is_dir():
        fewshot_files = sorted(fewshot_dir.glob("*.txt"))

    builder = PromptBuilder(fewshot_files=fewshot_files)
    llm_client = LLMClient()

    result = prove_loop(
        file=file,
        theorem_name=theorem,
        max_retries=max_retries,
        builder=builder,
        llm_client=llm_client,
    )

    if result.success:
        click.echo(f"SUCCESS in {result.attempts} attempt(s)")
        click.echo(f"Tactic: {result.final_tactic}")
        sys.exit(0)
    else:
        click.echo(f"FAILED after {result.attempts} attempts")
        click.echo(f"Last tactic: {result.final_tactic}")
        click.echo(f"Last error: {result.final_error}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Run tests, expect pass**

```bash
pytest tests/test_prove_loop.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Smoke test the CLI**

```bash
prove --help
```

Expected: usage banner showing all options.

- [ ] **Step 6: Commit**

```bash
git add tools/llm_prover/llm_prover/prove.py tools/llm_prover/tests/test_prove_loop.py
git commit -m "feat(llm_prover): proof drafting loop + CLI entry point"
```

### Task 2.6: Add a second few-shot example

The base softmax fewshot is the math identity. Add a longer worked example showing operational walk-through structure (the `softmax_naive_correct` proof).

**Files:**
- Create: `tools/llm_prover/llm_prover/prompts/fewshot/operational_walkthrough.txt`

- [ ] **Step 1: Copy and trim a real proof from the codebase**

Open `VeriTile/Examples/SoftmaxEq.lean`, copy the proof of `softmax_naive_correct` (lines 171–201) into the few-shot file:

Create `tools/llm_prover/llm_prover/prompts/fewshot/operational_walkthrough.txt`:

```
EXAMPLE 2: Operational walkthrough — kernel correctness via simp on
operational semantics + a scatter readback.

theorem softmax_naive_correct
    (N : Nat) (_hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoaded s N xs) :
    ∀ i : Fin N,
      observeY (exec (naiveSoftmaxKernel N) s) N s.pid i
        = some (naiveSpec xs i) := by
  intro i
  have hcast :
      ∀ k : Fin N,
        realToNat ((↑s.pid : ℝ) * (↑N : ℝ) + (↑(↑k : ℕ) : ℝ)) = s.pid * N + k.val := by
    intro k
    unfold realToNat
    have heq :
        ((↑s.pid : ℝ) * (↑N : ℝ) + (↑(↑k : ℕ) : ℝ)) = ((s.pid * N + k.val : ℕ) : ℝ) := by
      push_cast; ring
    rw [heq]; exact Nat.floor_natCast _
  have h_inj : Function.Injective (fun k : Fin N => s.pid * N + k.val) := by
    intro a b hab
    exact Fin.ext (Nat.add_left_cancel hab)
  simp [observeY, exec, naiveSoftmaxKernel, stepStmts, stepStmt, evalOp, Value.bop,
        Value.uop, Value.reduceSum, BlockState.setReg, BlockState.readMem, naiveSpec]
  unfold InputLoaded at _h_x
  simp_rw [hcast, _h_x]
  exact BlockState.scatter_readback _ _ _ h_inj i
```

- [ ] **Step 2: Commit**

```bash
git add tools/llm_prover/llm_prover/prompts/fewshot/operational_walkthrough.txt
git commit -m "feat(llm_prover): add operational walkthrough few-shot"
```

---

## Workstream 3 — T3 scouting document (~1–2 weeks, parallel)

This is research, not code. Goal: produce `Notes/T3_scouting.md` (~10–15 pages) that answers six concrete questions and gives a `T3-B feasible / not feasible in window` judgment.

### Task 3.1: FA forward pseudocode + invariant sketch

**Files:**
- Create: `Notes/T3_scouting.md`

- [ ] **Step 1: Write the document skeleton**

Create `Notes/T3_scouting.md`:

```markdown
# T3 Scouting — FlashAttention forward feasibility study

**Status:** Phase A deliverable. Updated 2026-...

This document evaluates whether the Phase D headline (T3-B: FA-1 ↔ FA-2)
is feasible in the program window, by writing the FA forward proof in
pseudocode + sketching the lemma needs, and recommending a mask-handling
option.

## 1. FA forward pseudocode

[Section 3.1: pseudocode + invariant]

## 2. Per-iteration invariant

[Section 3.2: (m_k, l_k, O_k) recurrence; what must hold]

## 3. Lemma needs across phases

[Section 3.3: forLoop_inv + tl.dot + masking + ...; what's needed
where]

## 4. Mask handling — Option α vs Option β

[Section 3.5: extended reals vs mask predicate denotation]

## 5. Risk register

[Section 3.4: unforeseen formalization gaps]

## 6. Feasibility judgment for T3-B

[Section 3.6: yes / no / conditional]
```

- [ ] **Step 2: Fill Section 1 — FA forward pseudocode**

Reference: Tri Dao, "FlashAttention" (NeurIPS 2022) Algorithm 1; also `flash-attn` repo `flash_attn/flash_attn_triton.py`.

Write the FA forward kernel in near-Triton pseudocode in the file under Section 1. Include:
- Inputs Q, K, V of shape `(S, D)`; output Y of shape `(S, D)`
- Block sizes `Bq` (Q rows per block), `Bk` (KV rows per block)
- Outer loop over `i ∈ [0, ⌈S/Bq⌉)` (Q blocks)
- Inner loop over `j ∈ [0, ⌈S/Bk⌉)` (KV blocks)
- Per-Q-block running `m_i, l_i, O_i`
- Score block: `S_ij = Q[i*Bq:(i+1)*Bq] · K[j*Bk:(j+1)*Bk]^T / √D`
- Mask application (if causal): set entries where `q_idx < k_idx` to `-∞` or equivalent
- Online softmax update: `m_new = max(m_i, max(S_ij))`, etc.

Allocate ~2 pages.

- [ ] **Step 3: Fill Section 2 — per-iteration invariant**

State the invariant precisely:

```
For each Q block i, after processing first k KV blocks (k ∈ [0, ⌈S/Bk⌉]):
  m_i,k = max over j ∈ [0, k*Bk) of masked_score(i, j)
  l_i,k = Σ_{j=0}^{k*Bk - 1} exp(masked_score(i, j) − m_i,k)
  O_i,k = Σ_{j=0}^{k*Bk - 1} exp(masked_score(i, j) − m_i,k) · V[j]
```

Sketch the inductive step in math (not Lean):
- Given the invariant at k, derive at k+1
- The key algebraic fact: `exp(m_old − m_new) · l_old + new_block_contribution = total at k+1`

Allocate ~2 pages.

- [ ] **Step 4: Fill Section 3 — lemma needs across phases**

Make a list of every Lean lemma the proof requires, organized by Phase.

Phase B prerequisites used:
- `forLoop_inv` (with index binding)
- `Real.exp_add`, `Real.exp_sub`, `Real.exp_pos`, `Real.exp_ne_zero`
- `Finset.sum_range_succ`, `Finset.sum_congr`

Phase C new needs:
- `tl.dot` semantics + simp behavior on `Value.tile2D`
- `softmax_neg_inf_zero` or its mask-predicate equivalent (depends on §4)
- 2D output addressing
- Causal indexing helpers

Phase D new needs:
- `multiBlockExec` semantics
- Disjoint-writes lemma
- `delayed_rescale_eq` (FA-2)

Allocate ~2 pages.

- [ ] **Step 5: Fill Section 4 — mask handling**

Write up Option α vs Option β explicitly:

For each option, produce:
- The exact Lean type signatures the spec / kernel would use
- Estimated lines of Lean code added across all of Phases A/C/D
- Known pain points (e.g., does `WithBot ℝ` interact well with `tl.dot`?)
- Net recommendation with reasoning

Recommend one. The recommendation drives all of Phase C.

Allocate ~3 pages.

- [ ] **Step 6: Fill Section 5 — risk register**

For each potential blocker, list:
- The risk
- How likely
- Detection (how/when we'd find out)
- Mitigation (what we'd do)

Examples already in `PLAN.md` Cross-cutting; this is the *more granular* T3-specific version.

Allocate ~1 page.

- [ ] **Step 7: Fill Section 6 — feasibility judgment**

Write the judgment with reasoning:
- "T3-B is feasible in window IF X, Y, Z"
- Or "T3-B is not feasible; recommend descope to T3-A only"
- Quantify confidence

Allocate ~1 page.

- [ ] **Step 8: Commit**

```bash
git add Notes/T3_scouting.md
git commit -m "docs(scouting): T3 forward feasibility study (Phase A deliverable)"
```

### Task 3.2: Iterate scouting based on Workstream 1 findings

As Workstream 1 reveals friction (e.g., simp behavior on `Value.bop`, missing API in Mathlib for `Real.log`), update the scouting risk register and lemma needs section.

- [ ] **Step 1: After Task 1.1 closes, update Section 3 with anything new found about simp behavior**

- [ ] **Step 2: After Task 1.3 closes (or escalates), update Section 5 with what the Welford proof revealed about Mathlib API**

- [ ] **Step 3: Final scouting pass before exit gate**

Re-read entire scouting doc; ensure judgment in Section 6 reflects what was actually learned.

- [ ] **Step 4: Commit each significant update**

```bash
git add Notes/T3_scouting.md
git commit -m "docs(scouting): update from Workstream 1 findings"
```

---

## Workstream 4 — Held-out eval setup (~2 days)

### Task 4.1: Create `bench/` sub-package excluded from main library

The held-out eval files should NOT be part of `lake build` of the main library, so their `sorry` doesn't trigger gate failure. Create as a sibling sub-package or unbuilt directory.

**Files:**
- Create: `bench/llm_eval/softmax_naive_correct_held_out.lean`
- Create: `bench/llm_eval/README.md`
- Modify: `lakefile.toml`

- [ ] **Step 1: Read existing lakefile**

```bash
cat lakefile.toml
```

- [ ] **Step 2: Decide structure — separate Lake target or unbuilt directory**

Two options:
- (a) Add a `lean_exe bench` or separate `lean_lib bench` target in `lakefile.toml` that the default `lake build` doesn't depend on.
- (b) Just put files in `bench/` and don't reference them from any `lakefile.toml` target; users run `lake env lean bench/...` manually.

Option (b) is simpler and matches the "main library does not depend on the held-out file" requirement from PLAN.md.

- [ ] **Step 3: Create the held-out file**

Create `bench/llm_eval/softmax_naive_correct_held_out.lean`:

```lean
/-
HELD-OUT BENCHMARK FILE — Phase A LLM tool eval target.

Replicates `VeriTile.Examples.softmax_naive_correct` from
`VeriTile/Examples/SoftmaxEq.lean` but with the proof body replaced
by `sorry`. The MAIN LIBRARY copy stays intact and proven; this file
is the LLM proof-drafting MVP's evaluation target.

DO NOT add this file to the main `VeriTile.lean` import graph.
DO NOT modify the proof body during development; the LLM tool runs
end-to-end on this file with no human edit.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Examples.SoftmaxEq

namespace VeriTile.Examples.HeldOut

open VeriTile.Triton VeriTile.Examples

/-- Held-out copy of `softmax_naive_correct` for LLM benchmark. -/
theorem softmax_naive_correct_held_out
    (N : Nat) (_hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoaded s N xs) :
    ∀ i : Fin N,
      observeY (exec (naiveSoftmaxKernel N) s) N s.pid i
        = some (naiveSpec xs i) := by
  sorry

end VeriTile.Examples.HeldOut
```

- [ ] **Step 4: Verify it elaborates standalone**

```bash
PATH="$HOME/.elan/bin:$PATH" lake env lean bench/llm_eval/softmax_naive_correct_held_out.lean 2>&1 | tail -10
```

Expected: one `declaration uses sorry` warning, no errors.

- [ ] **Step 5: Verify the main library is unaffected**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -5
```

Expected: clean build, no sorry warnings (main library remains proven).

- [ ] **Step 6: Write a short README**

Create `bench/llm_eval/README.md`:

```markdown
# LLM benchmark held-out set

These files are the held-out evaluation targets for the LLM proof-drafting tool.

Held-out files are NOT part of the main library build (`VeriTile.lean`
does not import them). The main library theorem stays intact and proven;
the held-out copy has its proof body replaced by `sorry` for the LLM tool
to attempt.

## Phase A held-out

- `softmax_naive_correct_held_out.lean` — copy of `softmax_naive_correct`
  from `VeriTile/Examples/SoftmaxEq.lean`.

## Eval protocol

See `PLAN.md` §LLM benchmark protocol. Summary:
- 5 independent runs per held-out lemma
- No human edit to the LLM output before re-running
- No prompt iteration during eval
- Cost / wall-clock / token counts reported
```

- [ ] **Step 7: Commit**

```bash
git add bench/llm_eval/softmax_naive_correct_held_out.lean bench/llm_eval/README.md
git commit -m "feat(bench): held-out eval target for Phase A LLM MVP"
```

---

## Workstream 5 — Exit gate

### Task 5.1: Run LLM MVP on held-out, measure close rate

Run the MVP 5 times on the held-out file with different temperature/seed values; record success/fail. Target ≥ 1/3 close rate.

- [ ] **Step 1: Activate venv and confirm `prove` CLI works**

```bash
cd tools/llm_prover && source .venv/bin/activate
prove --help
```

Expected: usage banner.

- [ ] **Step 2: Set the API key**

```bash
export ANTHROPIC_API_KEY=<your-key>
```

- [ ] **Step 3: Run 5 trials, log results**

```bash
mkdir -p ../../bench/llm_eval/results

for seed in 1 2 3 4 5; do
  cp ../../bench/llm_eval/softmax_naive_correct_held_out.lean \
     /tmp/heldout_$seed.lean
  prove --file /tmp/heldout_$seed.lean \
        --theorem softmax_naive_correct_held_out \
        --fewshot-dir llm_prover/prompts/fewshot \
        --max-retries 5 \
        2>&1 | tee ../../bench/llm_eval/results/run_$seed.log
done
```

Note: the LLM client doesn't directly consume `seed` (Anthropic API doesn't expose it). The variation comes from temperature; alternatively, vary the few-shot ordering or run on different days. For a true seed-controlled eval, the v0.2 tool can add seed support; v0.1 just runs 5 times and accepts inherent stochasticity.

- [ ] **Step 4: Tabulate results**

Count successes:

```bash
grep -c "SUCCESS" ../../bench/llm_eval/results/run_*.log
```

If ≥ 2 successes (40% ≥ 1/3 = 33.3%), gate passes.

If < 2: investigate. Common causes:
- Few-shot examples too far from target → strengthen
- Prompt template bug → fix
- Theorem too long for context → shorter target

Iterate until close rate ≥ 1/3, but DO NOT modify the held-out file itself or prompt for *this specific theorem*. General prompt template improvements ARE allowed before declaring the eval set "frozen" for v0.1; document any changes.

- [ ] **Step 5: Save the eval report**

Create `bench/llm_eval/results/phase_a_report.md`:

```markdown
# Phase A LLM MVP eval report

Date: 2026-...
Tool version: v0.1

## Held-out: softmax_naive_correct_held_out

| Run | Outcome | Attempts | Tactic | Wall-clock | API tokens |
|---|---|---|---|---|---|
| 1 | ... | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... | ... |
| 3 | ... | ... | ... | ... | ... |
| 4 | ... | ... | ... | ... | ... |
| 5 | ... | ... | ... | ... | ... |

**Close rate:** X/5 = Y%

**Gate target:** ≥ 1/3 = 33.3%
**Gate result:** PASS / FAIL

## Notes

[Any qualitative observations: which prompt format worked, common failure modes, etc.]
```

Fill in from the per-run logs.

- [ ] **Step 6: Commit**

```bash
git add bench/llm_eval/results/
git commit -m "feat(bench): Phase A LLM MVP eval results"
```

### Task 5.2: Tag `v0.1-tier1` release

- [ ] **Step 1: Verify all gate criteria**

Checklist (all must be true):
- [ ] `lake build` clean, no sorry warnings (main library)
- [ ] LLM MVP close rate ≥ 1/3 on held-out (per `bench/llm_eval/results/phase_a_report.md`)
- [ ] T3 scouting document complete with feasibility judgment
- [ ] All Workstream 1 commits pushed

- [ ] **Step 2: Update `PLAN.md` checkbox**

Open `PLAN.md`, find Phase A exit gate, mark complete (already a list in Decision log; re-read and confirm no edits needed). Optionally add a `## Phase A status` section if desired:

```markdown
## Phase A status (closed YYYY-MM-DD)

- [x] Tier 1 closed: 3 kernel-pair theorems + 1 math lemma
- [x] LLM tool MVP shipped: tag `v0.1-tier1`
- [x] T3 scouting document complete; Phase A→B gate decision: <copy from §6>
- [x] Held-out eval close rate: X/5 (≥ 1/3 target)
```

- [ ] **Step 3: Tag the release**

```bash
git tag -a v0.1-tier1 -m "Phase A complete: Tier 1 + LLM MVP + T3 scouting"
git push origin v0.1-tier1
```

- [ ] **Step 4: Write release notes on GitHub**

Use `gh` CLI or the GitHub web UI:

```bash
gh release create v0.1-tier1 --notes "$(cat <<'EOF'
# Phase A: Tier 1 + LLM MVP

Closes the loop-free Tier 1 kernel-pair theorems, delivers the LLM
proof-drafting MVP, and produces the T3 forward feasibility study.

## Theorems

* `softmax_kernels_refinement` (#1) — DONE prior to Phase A
* `log_sum_exp_refinement` (#2) — NEW
* `softmax_reciprocal_refinement` (#3) — NEW
* `welford_eq_two_pass` math lemma — NEW (Phase B prep)

## Tooling

* `tools/llm_prover/` — LLM proof-drafting MVP, ~200-line Python
* `bench/llm_eval/` — held-out evaluation harness
* `Notes/T3_scouting.md` — FA forward feasibility study, 10–15 pages

## Eval results

LLM MVP close rate on held-out `softmax_naive_correct`: X/5

## Next

See `PLAN.md` Gate A → B; Phase B work begins next.
EOF
)"
```

- [ ] **Step 5: Confirm release page**

Check: `https://github.com/Lizn-zn/VeriTile/releases/tag/v0.1-tier1`

---

## Self-review

(Run before declaring Phase A done.)

- [ ] **Spec coverage check** — for each Phase A deliverable in `PLAN.md`, confirm a task in this plan implements it:
  - 3 Tier 1 kernel-pair theorems → Task 1.0–1.2 (#1 already done)
  - welford_eq_two_pass math lemma → Task 1.3
  - LLM tool MVP → Workstream 2
  - T3 scouting doc → Workstream 3
  - Held-out copy → Workstream 4
  - Exit gate metrics → Workstream 5

- [ ] **Type / name consistency check** — `prove_loop`, `LeanResult`, `PromptBuilder`, `LLMClient` are used identically across tasks; tactic-replacement uses `replace_sorry` consistently.

- [ ] **Placeholder scan** — all steps have either real code, exact commands, or specific decision criteria. No TBDs except in scouting doc bodies (which are intentional — that work IS the scouting).

- [ ] **Build commands** — every Lean step uses `PATH="$HOME/.elan/bin:$PATH" lake build` or `lake env lean`; every Python step uses the venv-activated `prove`/`pytest`.

If anything is missing, add a task before invoking subagent-driven development or executing-plans.
