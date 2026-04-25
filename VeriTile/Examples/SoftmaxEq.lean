/-
VeriTile.Examples.SoftmaxEq

Worked equivalence example: naive softmax vs numerically stable
(max-subtracted) softmax. Embedded as `triton { ... }` macro kernels and
proved equivalent via the ARM-in-Lean refinement pattern.

Structure:

  (a) Source `.py` Triton (commented, for reference)
  (b) Embedded Triton ASTs via `triton { ... }` macro
  (c) Math denotation (`Fin n → ℝ` functions) and `naive_eq_stable` lemma
  (d) ARM-style kernel refinement:
        softmax_naive_correct       — closed-form output of naiveSoftmaxKernel
        softmax_stable_correct      — closed-form output of stableSoftmaxKernel
        softmax_kernels_refinement  — both kernels write the same `Y` memory

The math step is fully proven in (c). The two kernel-correctness lemmas in (d)
currently carry P2-level mechanical sorries — they reduce `exec kernel s` to
its closed form via simp on the operational semantics. The refinement theorem
itself is fully proven from those two lemmas plus `naive_eq_stable`.
-/

import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Tactic.FieldSimp
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL

namespace VeriTile.Examples

open VeriTile.Triton

/-! ## (a) Source Triton (`.py`) — for reference only

```python
# naive_softmax.py
@triton.jit
def naive_softmax(X, Y, N: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)
    x    = tl.load(X + offs)
    e    = tl.exp(x)
    s    = tl.sum(e, axis=0)
    y    = e / s
    tl.store(Y + offs, y)

# stable_softmax.py
@triton.jit
def stable_softmax(X, Y, N: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)
    x    = tl.load(X + offs)
    m    = tl.max(x, axis=0)
    e    = tl.exp(x - m)
    s    = tl.sum(e, axis=0)
    y    = e / s
    tl.store(Y + offs, y)
```
-/

/-! ## (b) Embedded Triton ASTs (using the `triton { ... }` macro) -/

/-- Naive softmax: `y = exp(x) / sum(exp(x))`. -/
def naiveSoftmaxKernel (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange($(N))
  x    := tl.load(X, offs)
  e    := tl.exp(x)
  s    := tl.sum(e)
  y    := e / s
  tl.store(Y, offs, y)
}

/-- Stable softmax: subtract the max before exponentiating. -/
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

/-! ## (c) Math denotation and equivalence -/

/-- Naive softmax over a tile of length `n` in ℝ. -/
noncomputable def naiveSoftmaxMath {n : Nat} (x : Fin n → ℝ) : Fin n → ℝ :=
  fun i => Real.exp (x i) / ∑ j, Real.exp (x j)

/-- Max-subtracted softmax over a tile of length `n` in ℝ. -/
noncomputable def stableSoftmaxMath {n : Nat} (x : Fin n → ℝ) (m : ℝ) :
    Fin n → ℝ :=
  fun i => Real.exp (x i - m) / ∑ j, Real.exp (x j - m)

/-- Naive and max-subtracted softmax produce the same output for any shift `m`.
    The load-bearing mathematical fact behind the numerical-stability trick. -/
theorem naive_eq_stable {n : Nat} (x : Fin n → ℝ) (m : ℝ) :
    naiveSoftmaxMath x = stableSoftmaxMath x m := by
  funext i
  unfold naiveSoftmaxMath stableSoftmaxMath
  simp only [Real.exp_sub]
  rw [← Finset.sum_div]
  have hm : Real.exp m ≠ 0 := Real.exp_ne_zero m
  field_simp

/-! ## (d) ARM-style kernel refinement

The refinement pattern from arm-in-lean's `tnum_const_refinement`:

```lean
theorem tnum_const_O3_correct (s : ArmState) ... :
    let sf := exec 2 tnum_const_O3 s
    readReg sf x0 = readReg s x0 ∧ readReg sf x1 = 0 ∧ ...

theorem tnum_const_refinement ... :
    let r3 := exec 2 tnum_const_O3 s3
    let r0 := exec 9 tnum_const_O0 s0
    readReg r3 x0 = readReg r0 x0 ∧ readReg r3 x1 = readReg r0 x1 := by
  obtain ⟨h3x0, h3x1, _⟩ := tnum_const_O3_correct s3 ...
  obtain ⟨h0x0, h0x1, _⟩ := tnum_const_O0_correct s0 ...
  simp only [...]
  trivial
```

We mirror it for Triton:
* `exec` runs the kernel,
* `readMem ... "Y" addr` plays the role of `readReg ... x0`,
* the two `softmax_*_correct` lemmas play the role of `tnum_const_O*_correct`,
* the refinement theorem composes them via `naive_eq_stable`.
-/

/-- The `Y` region at offset `pid*N + i.val` after running a kernel from `s`. -/
noncomputable def observeY
    (sf : Option BlockState) (N : Nat) (basePid : Nat) (i : Fin N) : Option ℝ :=
  sf.map (·.readMem "Y" (basePid * N + i.val))

/-- What `naiveSoftmaxKernel` writes at `Y[pid*N + i]`. -/
noncomputable def naiveSpec {N : Nat} (xs : Fin N → ℝ) (i : Fin N) : ℝ :=
  Real.exp (xs i) / ∑ j, Real.exp (xs j)

/-- What `stableSoftmaxKernel` writes at `Y[pid*N + i]`. -/
noncomputable def stableSpec {N : Nat} (xs : Fin N → ℝ) (m : ℝ) (i : Fin N) : ℝ :=
  Real.exp (xs i - m) / ∑ j, Real.exp (xs j - m)

/-- Max of a non-empty tile, in `Value.reduceMax`'s form. -/
noncomputable def tileMax {N : Nat} (h : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  match N, h, xs with
  | _ + 1, _, xs => Finset.univ.sup' Finset.univ_nonempty xs

/-- Pre-condition: the `X` region holds the input tile `xs` at the offsets
    `[pid*N, pid*N + N - 1]` (which the kernel will gather). -/
def InputLoaded (s : BlockState) (N : Nat) (xs : Fin N → ℝ) : Prop :=
  ∀ i : Fin N, s.mem "X" (s.pid * N + i.val) = xs i

/-- **Naive softmax kernel correctness.**

After running `naiveSoftmaxKernel N` on a state with input properly loaded
in region `"X"`, the `Y` region at observable offsets equals the naive softmax
of the input.

Proof outline (P2 work to discharge the `sorry`): the kernel has 7 statements
(`assign × 6` then `store × 1`); each `assign` updates one register, the final
`store` does a tile-tile scatter to `"Y"`. Walking through them with `simp` on
`exec / stepStmts / stepStmt / evalOp / writeMem` yields the closed form.
**Math content here: zero** — that's all isolated in `naive_eq_stable`. -/
theorem softmax_naive_correct
    (N : Nat) (_hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoaded s N xs) :
    ∀ i : Fin N,
      observeY (exec (naiveSoftmaxKernel N) s) N s.pid i
        = some (naiveSpec xs i) := by
  sorry

/-- **Stable softmax kernel correctness.** Same scheme as above with the
    max-shift formula as the closed form. P2 polish. -/
theorem softmax_stable_correct
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoaded s N xs) :
    ∀ i : Fin N,
      observeY (exec (stableSoftmaxKernel N) s) N s.pid i
        = some (stableSpec xs (tileMax hN xs) i) := by
  sorry

/-- **Refinement: `naiveSoftmaxKernel` and `stableSoftmaxKernel` produce the
    same `Y`-region memory.**

This is the ARM-in-Lean-style theorem: take the optional final state from each
kernel's `exec`, read off the relevant `Y` cells, and assert pointwise equality.
The proof composes the two correctness lemmas with `naive_eq_stable`, exactly
mirroring `tnum_const_refinement`'s `obtain ... + simp + trivial`. -/
theorem softmax_kernels_refinement
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoaded s N xs) :
    ∀ i : Fin N,
      observeY (exec (naiveSoftmaxKernel N) s) N s.pid i =
      observeY (exec (stableSoftmaxKernel N) s) N s.pid i := by
  intro i
  -- Discharge each side via its correctness lemma (the ARM-style obtain).
  rw [softmax_naive_correct  N hN s xs h_x i,
      softmax_stable_correct N hN s xs h_x i]
  -- Now: `some (naiveSpec xs i) = some (stableSpec xs (tileMax hN xs) i)`.
  -- Reduce to the math identity and apply `naive_eq_stable`.
  congr 1
  unfold naiveSpec stableSpec
  have h := naive_eq_stable xs (tileMax hN xs)
  simp only [naiveSoftmaxMath, stableSoftmaxMath] at h
  exact congrFun h i

end VeriTile.Examples
