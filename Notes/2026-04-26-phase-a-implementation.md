# Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close 2 new Tier 1 kernel-pair theorems and 1 math lemma, write a ~80-line `scripts/prove.sh` wrapping the `lean4` Claude Code plugin's `/lean4:autoprove`, produce a T3 feasibility scouting document, and set up the LLM benchmark held-out infrastructure — all per `PLAN.md` Phase A.

**Architecture:** Four workstreams sharing one exit gate (`v0.1-tier1` tag). Lean work extends the existing `VeriTile` library; the LLM tool is a thin bash wrapper at `scripts/prove.sh` around the existing `lean4` Claude Code plugin (no Python project); scouting is a markdown analysis under `Notes/`; the benchmark held-out lives at `bench/llm_eval/`.

**Tech Stack:** Lean 4 (v4.15.0+, Mathlib), bash, the `lean4` Claude Code plugin (`/lean4:autoprove`, version pinned in `scripts/README.md`), `lake build` (Lean), `git`, `jq` (for JSON output parsing).

---

## Scope reference

This plan implements `PLAN.md` Phase A:
- 2 new Tier 1 theorems: `log_sum_exp_refinement` (#2), `softmax_reciprocal_refinement` (#3) — #1 `softmax_kernels_refinement` is already done
- 1 math lemma: `welford_eq_two_pass`
- `scripts/prove.sh` wrapper around `/lean4:autoprove`
- T3 scouting at `Notes/T3_scouting.md`
- Held-out eval at `bench/llm_eval/softmax_naive_correct_held_out.lean`
- Exit gate: `lake build` clean, `scripts/prove.sh` close rate ≥ 1/3 on the held-out, `v0.1-tier1` tag

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

**LLM tool wrapper:**

| File | Status | Responsibility |
|---|---|---|
| `scripts/prove.sh` | Create | ~80-line bash wrapper around `claude -p "/lean4:autoprove ..."`; captures JSON output to `Logs/`; exits 0 on success / 1 on fail |
| `scripts/README.md` | Create | Usage instructions + pinned `lean4` plugin version |
| `Logs/` | Create | (gitignored) Per-run JSON output from `scripts/prove.sh` |

**Scouting + meta:**

| File | Status | Responsibility |
|---|---|---|
| `Notes/T3_scouting.md` | Create | FA pseudocode + invariants + lemma needs + mask option choice + risk register + T3-B feasibility judgment |
| `.gitignore` | Modify | Add `Logs/` |
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

Once Workstream 2's `scripts/prove.sh` is functional, run it on this lemma as a real test of the wrapper. If `/lean4:autoprove` closes it, great. If not, fall back to manual proof or descope `welford_eq_two_pass` to "Phase B prep, defer formal proof" — but ONLY with explicit user OK.

- [ ] **Step 5: Run lake build, expect clean**

Run: `PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -10`

Expected: clean build, no sorries.

- [ ] **Step 6: Commit**

```bash
git add VeriTile/Examples/WelfordMath.lean VeriTile.lean
git commit -m "feat(tier1): welford_eq_two_pass math lemma (Phase B prep)"
```

---

## Workstream 2 — `scripts/prove.sh` wrapper (~1 day)

Wrap the existing `lean4` Claude Code plugin (`/lean4:autoprove`) in a thin bash script that gives us a uniform interface for benchmark eval. **Massively rescoped** from the earlier 6-task Python project (200 lines, multi-week) — the plugin already provides LSP integration, multi-cycle iteration with stuck detection, deep-mode escalation, tactic cascade, and repair mode. We wrap, we don't rebuild. See `PLAN.md` decision log entry 5.

### Task 2.1: Write `scripts/prove.sh`

**Files:**
- Create: `scripts/prove.sh`
- Create: `scripts/README.md`
- Modify: `.gitignore` (add `Logs/`)

- [ ] **Step 1: Create directories**

```bash
mkdir -p scripts Logs
```

- [ ] **Step 2: Write `scripts/prove.sh`** — see PLAN.md `scripts/prove.sh` template (~80 lines bash; argument parsing for `<lean_file> [--max-cycles N] [--prompt "..."]`; invokes `claude -p "/lean4:autoprove ..."`; captures JSON output to `Logs/<basename>_<timestamp>.json`; parses success/fail; exit 0/1 for benchmark counting).

The full script is documented in PLAN.md (Workstream 2 / Task 2.1 in the program plan). Highlights:
- `claude -p "/lean4:autoprove ${LEAN_FILE_ABS} --max-cycles=${MAX_CYCLES} --commit=never --planning=off --review-source=none"`
- `--dangerously-skip-permissions --max-budget-usd 10.00 --output-format stream-json --include-partial-messages --verbose`
- Result parsing via `jq` on the last `{"type":"result", ...}` line in the stream
- Heuristic success: clean exit AND result subtype isn't error AND result text doesn't say "fail/stuck/unable to close/sorry remains"

- [ ] **Step 3: `chmod +x scripts/prove.sh`**

- [ ] **Step 4: Write `scripts/README.md`** with usage examples, exit codes, and a section "Pinned plugin version" recording `~/.claude/plugins/cache/lean4-skills/lean4/4.4.9/` as the version this was built against.

- [ ] **Step 5: Append `/Logs/` to `.gitignore`**

- [ ] **Step 6: Smoke-test argument parsing**

```bash
scripts/prove.sh --help 2>&1 | head -10
```

Expected: usage banner.

- [ ] **Step 7: Commit**

```bash
git add scripts/prove.sh scripts/README.md .gitignore
git commit -m "feat(scripts): prove.sh — thin wrapper around /lean4:autoprove"
```

(Use heredoc for a longer commit body explaining the rescope.)

### Task 2.2: Smoke-test against held-out file

Single-shot invocation to confirm `/lean4:autoprove` actually fires through the wrapper and produces a usable JSON output. Detailed eval (5 trials, close rate measurement) is Workstream 5.

- [ ] **Step 1: Ensure Workstream 4's held-out file exists**

Workstream 4 creates `bench/llm_eval/softmax_naive_correct_held_out.lean` with a `sorry`d copy. If W4 hasn't run yet, do it first.

- [ ] **Step 2: Run wrapper once with low budget**

```bash
scripts/prove.sh bench/llm_eval/softmax_naive_correct_held_out.lean --max-cycles=2
```

Expected: `[SUCCESS]` or `[FAIL]` line. Either is fine for Task 2.2; we are testing plumbing, not measuring close rate.

- [ ] **Step 3: Inspect log structure**

```bash
ls -lt Logs/ | head -3
jq '. | {type, subtype}' Logs/<latest>.json | head -10
```

Confirms JSON parses. If `jq` complains, adjust `prove.sh`'s `RESULT_LINE` extraction.

- [ ] **Step 4: Commit any tweaks** (skip if no changes)

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
git commit -m "feat(bench): held-out eval target for Phase A prove.sh eval"
```

---

## Workstream 5 — Exit gate

### Task 5.1: Run `scripts/prove.sh` on held-out, measure close rate

Run the wrapper 5 times on the held-out file; record success/fail. Target ≥ 1/3 close rate.

- [ ] **Step 1: Confirm script works and `claude` CLI is available**

```bash
scripts/prove.sh --help 2>&1 | head -10
which claude
```

Expected: usage banner from `prove.sh`; absolute path to `claude`.

- [ ] **Step 2: Run 5 trials, log results**

```bash
mkdir -p bench/llm_eval/results

for trial in 1 2 3 4 5; do
  # Fresh copy each trial: previous trial may have modified the file.
  cp bench/llm_eval/softmax_naive_correct_held_out.lean \
     /tmp/heldout_${trial}.lean
  scripts/prove.sh /tmp/heldout_${trial}.lean --max-cycles=5 \
    > bench/llm_eval/results/run_${trial}.log 2>&1
  echo "trial $trial: exit=$?" >> bench/llm_eval/results/summary.log
done
```

Note: the wrapper inherits `/lean4:autoprove`'s inherent stochasticity (sampling temperature, plugin's internal candidate generation). For a true seed-controlled eval, future protocol versions can add seeding; v0.1 accepts inherent variance across 5 runs.

- [ ] **Step 3: Tabulate**

```bash
grep -c "exit=0" bench/llm_eval/results/summary.log  # successful trials
```

If ≥ 2 successes (40% ≥ 1/3 = 33.3%), gate passes.

If < 2:
- Inspect `Logs/heldout_*_*.json` to see what the plugin actually did
- Common: plugin gets stuck on a specific tactic; try `--max-cycles=10` (more budget) or `--prompt "Try the scatter_readback approach used in SoftmaxEq.lean"` (steer it)
- Adjusting `--max-cycles` is allowed; per `PLAN.md` §LLM benchmark protocol, prompt-iteration on the held-out theorem is NOT allowed during eval. If you change the prompt, document it as a separate "tuned" run and report both numbers

- [ ] **Step 4: Save eval report**

Create `bench/llm_eval/results/phase_a_report.md`:

```markdown
# Phase A `scripts/prove.sh` eval report

Date: 2026-...
Wrapper version: v0.1 (`scripts/prove.sh` initial release)
Plugin: lean4-skills 4.4.9 (commit / cache path: ~/.claude/plugins/cache/lean4-skills/lean4/4.4.9/)

## Held-out: softmax_naive_correct_held_out

| Trial | Outcome | Cycles used | Wall-clock | API cost (USD) | Notes |
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

[Qualitative observations: which sub-goals the plugin closed easily vs got stuck on,
how often deep mode escalated, etc.]
```

Fill in from the per-run logs (cycles + wall-clock from `Logs/heldout_*_*.json`; API cost from claude's `result.total_cost_usd` field).

- [ ] **Step 5: Commit**

```bash
git add bench/llm_eval/results/
git commit -m "feat(bench): Phase A scripts/prove.sh eval results"
```

### Task 5.2: Tag `v0.1-tier1` release

- [ ] **Step 1: Verify all gate criteria**

Checklist (all must be true):
- [ ] `lake build` clean, no sorry warnings (main library)
- [ ] `scripts/prove.sh` close rate ≥ 1/3 on held-out (per `bench/llm_eval/results/phase_a_report.md`)
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
git tag -a v0.1-tier1 -m "Phase A complete: Tier 1 + scripts/prove.sh + T3 scouting"
git push origin v0.1-tier1
```

- [ ] **Step 4: Write release notes on GitHub**

Use `gh` CLI or the GitHub web UI:

```bash
gh release create v0.1-tier1 --notes "$(cat <<'EOF'
# Phase A: Tier 1 + scripts/prove.sh

Closes the loop-free Tier 1 kernel-pair theorems, ships the
`scripts/prove.sh` wrapper around `/lean4:autoprove`, and produces
the T3 forward feasibility study.

## Theorems

* `softmax_kernels_refinement` (#1) — DONE prior to Phase A
* `log_sum_exp_refinement` (#2) — NEW
* `softmax_reciprocal_refinement` (#3) — NEW
* `welford_eq_two_pass` math lemma — NEW (Phase B prep)

## Tooling

* `scripts/prove.sh` — ~80-line bash wrapper around the `lean4` Claude Code
  plugin's `/lean4:autoprove`; captures JSON output to `Logs/`
* `bench/llm_eval/` — held-out evaluation harness
* `Notes/T3_scouting.md` — FA forward feasibility study, 10–15 pages

## Eval results

`scripts/prove.sh` close rate on held-out `softmax_naive_correct`: X/5

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
