/-
VeriTile.Examples.ApproxGeLU

Embedded Triton example for the tanh-style approximate GeLU commonly used
in transformer MLP blocks, together with the exact erf-based mathematical
GeLU specification it approximates:

    gelu_exact(x)  = 0.5 * x * (1 + erf(x / sqrt(2)))
    gelu_approx(x) = 0.5 * x *
      (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))

The reference Triton snippet often computes `tanh` through sigmoid:

```python
@triton.jit
def gelu_approx_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x3 = x * x * x
    u = 0.7978845608028654 * (x + 0.044715 * x3)
    tanh_u = 2.0 * tl.sigmoid(2.0 * u) - 1.0
    y = 0.5 * x * (1.0 + tanh_u)

    tl.store(out_ptr + offsets, y, mask=mask)
```

This file embeds the aligned single-block, unmasked approximate Triton
implementation in VeriTile's DSL. The exact erf-based GeLU appears as the
Lean mathematical spec, not as an embedded kernel: the current DSL has
`tl.sigmoid` but not `tl.erf` or masks.

Constants are written as rational expressions because the DSL currently
supports natural numeric literals plus arithmetic, not decimal literals.
For example, `7978845608028654 / 10000000000000000` represents the usual
`sqrt(2 / pi)` decimal approximation used by the Triton reference.
-/

import Mathlib.Analysis.SpecialFunctions.Sigmoid
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import VeriTile.Math.RealErf
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile.Triton VeriTile.Math

/-! ## Embedded Triton AST -/

/-- Approximate GeLU, aligned single-block version.

Reads `blockSize` values from `xReg`, computes the tanh-style approximate
GeLU using `2 * sigmoid(2u) - 1` for `tanh(u)`, and writes the result to
`outReg`. -/
def approxGeLUKernel (xReg outReg : RegionName) (blockSize : Nat) : Kernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x       := tl.load($(xReg) + offsets)
  x3      := x * x * x
  c       := 7978845608028654 / 10000000000000000
  k       := 44715 / 1000000
  u       := c * (x + k * x3)
  tanh_u  := 2 * tl.sigmoid(2 * u) - 1
  y       := (1 / 2) * x * (1 + tanh_u)
  tl.store($(outReg) + offsets, y)
}

/-! ## Math Denotations -/

/-- Exact mathematical GeLU scalar function. -/
noncomputable def exactGeLUScalar (x : ℝ) : ℝ :=
  (1 / 2) * x * (1 + realErf (x / Real.sqrt 2))

/-- Tanh-style approximate GeLU scalar function implemented by the Triton kernel.

The `tanh` is represented as `2 * sigmoid(2u) - 1`, matching the embedded
DSL implementation. -/
noncomputable def approxGeLUScalar (x : ℝ) : ℝ :=
  let x3 := x * x * x
  let c : ℝ := 7978845608028654 / 10000000000000000
  let k : ℝ := 44715 / 1000000
  let u := c * (x + k * x3)
  (1 / 2) * x * (1 + (2 * Real.sigmoid (2 * u) - 1))

/-- Math-level denotation of the embedded approximate GeLU expression. -/
noncomputable def approxGeLUSpec {N : Nat} (xs : Fin N → ℝ) (i : Fin N) : ℝ :=
  approxGeLUScalar (xs i)

/-- Exact erf-based GeLU spec used as the mathematical reference. -/
noncomputable def exactGeLUSpec {N : Nat} (xs : Fin N → ℝ) (i : Fin N) : ℝ :=
  exactGeLUScalar (xs i)

/-! ## Algebraic Decomposition and Trivial Bound

Two helper lemmas that factor the approximation error into the gap between
`tanh` and `realErf`, and a uniform `|x|` bound that follows from `|tanh| ≤ 1`
and `|realErf| ≤ 1`. -/

private lemma two_sigmoid_two_sub_one_eq_tanh (u : ℝ) :
    2 * Real.sigmoid (2 * u) - 1 = Real.tanh u := by
  rw [Real.sigmoid_def, Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
  set a := Real.exp u
  set b := Real.exp (-u)
  have hab : a * b = 1 := by
    simp [a, b, ← Real.exp_add]
  have hsum : a + b ≠ 0 :=
    ne_of_gt (add_pos (Real.exp_pos _) (Real.exp_pos _))
  have hbb : 1 + b * b ≠ 0 := by positivity
  have hex : Real.exp (-(2 * u)) = b * b := by
    rw [show -(2 * u) = -u + -u from by ring, Real.exp_add]
  rw [hex,
      show (a - b) / 2 / ((a + b) / 2) = (a - b) / (a + b) from by field_simp,
      show 2 * (1 + b * b)⁻¹ - 1 = (1 - b * b) / (1 + b * b) from by
        field_simp; ring,
      div_eq_div_iff hbb hsum]
  linear_combination (-2 * b) * hab

/-- `tanh` tail bound: `1 − tanh u ≤ 2 · exp(−2u)` for any `u`. The bound is
nontrivial only for `u ≫ 0`; for `u ≤ 0` both sides are ≥ 1. -/
private lemma one_sub_tanh_le_two_exp_neg_two_mul (u : ℝ) :
    1 - Real.tanh u ≤ 2 * Real.exp (-(2 * u)) := by
  rw [Real.tanh_eq]
  have heu : 0 < Real.exp u := Real.exp_pos _
  have henu : 0 < Real.exp (-u) := Real.exp_pos _
  have hsum : 0 < Real.exp u + Real.exp (-u) := add_pos heu henu
  have hek : Real.exp u * Real.exp (-u) = 1 := by
    rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
  have h_eq :
      1 - (Real.exp u - Real.exp (-u)) / (Real.exp u + Real.exp (-u))
        = 2 * Real.exp (-u) / (Real.exp u + Real.exp (-u)) := by
    field_simp; ring
  rw [h_eq,
      show (2 * Real.exp (-(2 * u)))
            = 2 * Real.exp (-u) * Real.exp (-u) from by
        rw [show -(2 * u) = -u + -u from by ring, Real.exp_add]; ring,
      div_le_iff₀ hsum]
  have hen2 : 0 < Real.exp (-u) * Real.exp (-u) := mul_pos henu henu
  nlinarith [hek, hen2, henu]

/-- Symmetric `tanh` tail bound: `tanh u + 1 ≤ 2 · exp(2u)` for any `u`.
Combined with the previous lemma this controls the gap `|tanh u − ε|` for
both signs of `u` against the appropriate exponential decay. -/
private lemma tanh_add_one_le_two_exp_two_mul (u : ℝ) :
    Real.tanh u + 1 ≤ 2 * Real.exp (2 * u) := by
  have h := one_sub_tanh_le_two_exp_neg_two_mul (-u)
  rw [Real.tanh_neg] at h
  rw [show -(2 * -u) = 2 * u from by ring] at h
  linarith

private lemma abs_tanh_le_one (u : ℝ) : |Real.tanh u| ≤ 1 := by
  rw [Real.tanh_eq_sinh_div_cosh, abs_div, abs_of_pos (Real.cosh_pos u),
      div_le_one (Real.cosh_pos u), Real.sinh_eq, Real.cosh_eq]
  rcases abs_choice (Real.exp u - Real.exp (-u)) with h | h
  · rw [show |(Real.exp u - Real.exp (-u)) / 2|
          = (Real.exp u - Real.exp (-u)) / 2 from by
        rw [abs_div, abs_of_pos (by norm_num : (0:ℝ) < 2), h]]
    have : 0 < Real.exp (-u) := Real.exp_pos _
    linarith
  · rw [show |(Real.exp u - Real.exp (-u)) / 2|
          = -(Real.exp u - Real.exp (-u)) / 2 from by
        rw [abs_div, abs_of_pos (by norm_num : (0:ℝ) < 2), h]]
    have : 0 < Real.exp u := Real.exp_pos _
    linarith

/-- Algebraic factorization of the approximation error:
`approxGeLUScalar x − exactGeLUScalar x = (x / 2) · (tanh u − realErf (x / √2))`,
where `u = c · (x + k · x³)` is the inner argument of the tanh approximation. -/
lemma approxGeLUScalar_sub_exactGeLUScalar (x : ℝ) :
    approxGeLUScalar x - exactGeLUScalar x =
      (x / 2) *
        (Real.tanh
            ((7978845608028654 / 10000000000000000 : ℝ) *
              (x + (44715 / 1000000) * (x * x * x)))
          - realErf (x / Real.sqrt 2)) := by
  unfold approxGeLUScalar exactGeLUScalar
  simp only [two_sigmoid_two_sub_one_eq_tanh]
  ring

/-- Uniform pointwise bound: the gelu approximation differs from the exact
erf-based gelu by at most `|x|`. This is a strictly stronger bound than the
trivial one and follows from `|tanh| ≤ 1`, `|realErf| ≤ 1`, and the
decomposition above. -/
lemma abs_approxGeLUScalar_sub_exactGeLUScalar_le_abs (x : ℝ) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ |x| := by
  rw [approxGeLUScalar_sub_exactGeLUScalar x]
  set u : ℝ := (7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))
  rw [abs_mul, abs_div, abs_two]
  have hgap : |Real.tanh u - realErf (x / Real.sqrt 2)| ≤ 2 :=
    calc |Real.tanh u - realErf (x / Real.sqrt 2)|
        ≤ |Real.tanh u| + |realErf (x / Real.sqrt 2)| := abs_sub _ _
      _ ≤ 1 + 1 := add_le_add (abs_tanh_le_one u) (abs_realErf_le_one _)
      _ = 2 := by norm_num
  have hx2 : 0 ≤ |x| / 2 := by positivity
  calc |x| / 2 * |Real.tanh u - realErf (x / Real.sqrt 2)|
      ≤ |x| / 2 * 2 := by exact mul_le_mul_of_nonneg_left hgap hx2
    _ = |x| := by ring

/-! ## Explicit asymptotic tail bound

For `x ≥ √2`, the approximation error is dominated by a quantitative
exponential tail. This is the structural step that, combined with concrete
numerics on a single (large) value of `x`, would close the
`approx_gelu_error_bound_large` sorry on the half-line `[T, ∞)` for any
`T ≥ √2` chosen large enough that the RHS is `≤ 1/1000`. -/

private lemma c_pos : (7978845608028654 / 10000000000000000 : ℝ) > 0 := by norm_num

private lemma k_pos : (44715 / 1000000 : ℝ) > 0 := by norm_num

/-- For `x ≥ √2 > 0`, the gelu approximation error is bounded by an explicit
sum of two exponential tails. The first tail comes from
`1 − tanh(c · (x + k · x³)) ≤ 2 · exp(−2c·x)` (using `x + k·x³ ≥ x` for
`x ≥ 0`); the second comes from `1 − realErf(x/√2) ≤ (2/√π) · exp(−x/√2)`. -/
lemma abs_approxGeLUScalar_sub_exactGeLUScalar_tail_pos
    {x : ℝ} (hx : Real.sqrt 2 ≤ x) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤
      x * Real.exp (-(2 * (7978845608028654 / 10000000000000000) * x)) +
        (x / Real.sqrt Real.pi) * Real.exp (-(x / Real.sqrt 2)) := by
  set c : ℝ := 7978845608028654 / 10000000000000000 with hc_def
  set k : ℝ := 44715 / 1000000 with hk_def
  have hc_pos : 0 < c := c_pos
  have hk_pos : 0 < k := k_pos
  have hx_pos : 0 < x := by
    have : 0 < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
    linarith
  have hx_ge_one : 1 ≤ x / Real.sqrt 2 := by
    rw [le_div_iff₀ (Real.sqrt_pos.mpr (by norm_num : (0:ℝ) < 2))]; linarith
  -- Set u := c · (x + k · x³)
  set u : ℝ := c * (x + k * (x * x * x)) with hu_def
  have hx3_pos : 0 ≤ x * x * x := by positivity
  have hu_ge_cx : c * x ≤ u := by
    rw [hu_def]
    have : 0 ≤ k * (x * x * x) := by positivity
    nlinarith
  -- 1 − tanh u ≤ 2 · exp(−2u) ≤ 2 · exp(−2cx)
  have h_tanh_tail : 1 - Real.tanh u ≤ 2 * Real.exp (-(2 * c * x)) := by
    have h1 := one_sub_tanh_le_two_exp_neg_two_mul u
    have h2 : Real.exp (-(2 * u)) ≤ Real.exp (-(2 * c * x)) :=
      Real.exp_le_exp.mpr (by nlinarith)
    linarith
  -- 1 − realErf(x/√2) ≤ (2/√π) · exp(−x/√2)
  have h_erf_tail : 1 - realErf (x / Real.sqrt 2)
        ≤ (2 / Real.sqrt Real.pi) * Real.exp (-(x / Real.sqrt 2)) :=
    VeriTile.Math.one_sub_realErf_le_exp_neg hx_ge_one
  -- Decomposition: |approx − exact| = (x/2)·|tanh u − realErf(x/√2)|
  rw [approxGeLUScalar_sub_exactGeLUScalar x]
  -- Now use the algebraic shape, hu_def matches exactly the inner term
  have hu_match :
      Real.tanh ((7978845608028654 / 10000000000000000 : ℝ) *
              (x + (44715 / 1000000 : ℝ) * (x * x * x)))
        = Real.tanh u := by
    rw [hu_def]
  rw [hu_match]
  rw [abs_mul, abs_div, abs_two, abs_of_pos hx_pos]
  -- Goal: (x / 2) · |tanh u − realErf(x/√2)| ≤ x · exp(-2cx) + (x/√π)·exp(-x/√2)
  have h_tanh_le : Real.tanh u ≤ 1 := (abs_le.mp (abs_tanh_le_one u)).2
  have h_erf_le : realErf (x / Real.sqrt 2) ≤ 1 :=
    (abs_le.mp (abs_realErf_le_one _)).2
  have h_gap : |Real.tanh u - realErf (x / Real.sqrt 2)| ≤
        (1 - Real.tanh u) + (1 - realErf (x / Real.sqrt 2)) :=
    calc |Real.tanh u - realErf (x / Real.sqrt 2)|
        = |(1 - realErf (x / Real.sqrt 2)) - (1 - Real.tanh u)| := by
            congr 1; ring
      _ ≤ |1 - realErf (x / Real.sqrt 2)| + |1 - Real.tanh u| := abs_sub _ _
      _ = (1 - realErf (x / Real.sqrt 2)) + (1 - Real.tanh u) := by
            rw [abs_of_nonneg (by linarith), abs_of_nonneg (by linarith)]
      _ = (1 - Real.tanh u) + (1 - realErf (x / Real.sqrt 2)) := by ring
  have h_gap_bound : |Real.tanh u - realErf (x / Real.sqrt 2)| ≤
        2 * Real.exp (-(2 * c * x)) +
          (2 / Real.sqrt Real.pi) * Real.exp (-(x / Real.sqrt 2)) := by
    calc |Real.tanh u - realErf (x / Real.sqrt 2)|
        ≤ (1 - Real.tanh u) + (1 - realErf (x / Real.sqrt 2)) := h_gap
      _ ≤ 2 * Real.exp (-(2 * c * x)) +
          (2 / Real.sqrt Real.pi) * Real.exp (-(x / Real.sqrt 2)) :=
            add_le_add h_tanh_tail h_erf_tail
  have hx2 : 0 ≤ x / 2 := by positivity
  calc (x / 2) * |Real.tanh u - realErf (x / Real.sqrt 2)|
      ≤ (x / 2) *
          (2 * Real.exp (-(2 * c * x)) +
            (2 / Real.sqrt Real.pi) * Real.exp (-(x / Real.sqrt 2))) := by
            exact mul_le_mul_of_nonneg_left h_gap_bound hx2
    _ = x * Real.exp (-(2 * c * x)) +
          (x / Real.sqrt Real.pi) * Real.exp (-(x / Real.sqrt 2)) := by
            field_simp

/-! ## Correctness and Approximation Statements -/

/-- **`approxGeLUKernel` correctness against the approximate GeLU expression.**

This is the operational statement: the embedded Triton kernel computes exactly
the tanh/sigmoid approximation represented by `approxGeLUSpec`. -/
theorem approx_gelu_kernel_correct
    (xReg outReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState)
    (xs : Fin blockSize → ℝ)
    (_h_x : InputLoadedAt s xReg blockSize xs) :
    ∀ i : Fin blockSize,
      observeAt (exec (approxGeLUKernel xReg outReg blockSize) s)
          outReg blockSize s.pid i
        = some (approxGeLUSpec xs i) := by
  intro i
  have h_inj :
      Function.Injective (fun k : Fin blockSize => s.pid * blockSize + k.val) := by
    intro a b hab
    exact Fin.ext (Nat.add_left_cancel hab)
  simp [observeAt, exec, approxGeLUKernel, stepStmts, stepStmt, evalOp,
        Value.bop, Value.uop, BlockState.setReg, BlockState.readMem,
        approxGeLUSpec, approxGeLUScalar]
  unfold InputLoadedAt at _h_x
  simp_rw [_h_x]
  exact BlockState.scatter_readback _ _ _ h_inj i

/-- Target error tolerance for the standard tanh/sigmoid GeLU approximation. -/
noncomputable def approxGeLUEps : ℝ := 1 / 1000

/-- Quantitative gap remaining for the global `approxGeLUEps = 1e-3` bound.

For inputs with `|xs i| > approxGeLUEps`, the algebraic decomposition reduces
the global error bound to a quantitative bound on the gap between
`tanh(c·(x + k·x³))` and `realErf(x / √2)`: specifically,

    |x| / 2 · |tanh u − realErf(x / √2)|  ≤  1 / 1000.

This is the unsolved analytic content of `approx_gelu_error_bound`. The bound
is known to hold globally (the empirical max of the gelu approximation error
is ≈ 4·10⁻⁴, well within `1e-3`), but a Lean-checked proof requires either
certified interval arithmetic across a finite window plus tail bounds via
`realErf_tendsto_atTop`/`atBot`, or a Taylor-with-remainder analysis using
`realErf_hasDerivAt`. Both are out of scope for the current snapshot. -/
theorem approx_gelu_error_bound_large {x : ℝ} (_hx : approxGeLUEps < |x|) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ approxGeLUEps := by
  -- Open analytic content: the tanh-vs-erf gap times |x|/2 is bounded by 1e-3
  -- whenever |x| > 1e-3. See module docstring above.
  sorry

/-- The analytic/numerical approximation claim we need:
    on this input tile, the tanh/sigmoid approximate GeLU differs from the
    exact erf-based GeLU by at most `1e-3`.

The proof case-splits on `|xs i|` against `approxGeLUEps = 1/1000`. The small
case `|xs i| ≤ 1/1000` is closed unconditionally by the trivial linear bound
`|approx − exact| ≤ |x|`. The large case is reduced to the still-pending
quantitative gap statement `approx_gelu_error_bound_large`. -/
theorem approx_gelu_error_bound {N : Nat} (xs : Fin N → ℝ) :
    ∀ i : Fin N, |approxGeLUSpec xs i - exactGeLUSpec xs i| ≤ approxGeLUEps := by
  intro i
  unfold approxGeLUSpec exactGeLUSpec
  rcases le_or_gt (|xs i|) approxGeLUEps with hsmall | hlarge
  · exact (abs_approxGeLUScalar_sub_exactGeLUScalar_le_abs (xs i)).trans hsmall
  · exact approx_gelu_error_bound_large hlarge

/-- Every value written by the Triton approximate GeLU kernel is within `1e-3`
    of the exact erf-based GeLU spec, assuming the scalar approximation theorem
    above. -/
theorem approx_gelu_kernel_approximates_exact
    (xReg outReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState)
    (xs : Fin blockSize → ℝ)
    (h_x : InputLoadedAt s xReg blockSize xs) :
    ∀ i : Fin blockSize,
      (observeAt (exec (approxGeLUKernel xReg outReg blockSize) s)
          outReg blockSize s.pid i).map (fun y => |y - exactGeLUSpec xs i|)
        = some (|approxGeLUSpec xs i - exactGeLUSpec xs i|)
      ∧ |approxGeLUSpec xs i - exactGeLUSpec xs i| ≤ approxGeLUEps := by
  intro i
  constructor
  · rw [approx_gelu_kernel_correct xReg outReg blockSize hN s xs h_x i]
    rfl
  · exact approx_gelu_error_bound xs i

/-- Sorry-free linear approximation: every value written by the Triton
    approximate GeLU kernel is within `|xs i|` of the exact erf-based GeLU
    spec. This is a strictly weaker bound than `approxGeLUEps = 1e-3` — the
    constant is `|xs i|` rather than `1/1000` — but it depends only on
    `|tanh| ≤ 1` and `|realErf| ≤ 1` and is therefore established without the
    quantitative `tanh`-vs-`realErf` analytic content still pending in
    `approx_gelu_error_bound`. -/
theorem approx_gelu_kernel_approximates_linearly
    (xReg outReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState)
    (xs : Fin blockSize → ℝ)
    (h_x : InputLoadedAt s xReg blockSize xs) :
    ∀ i : Fin blockSize,
      (observeAt (exec (approxGeLUKernel xReg outReg blockSize) s)
          outReg blockSize s.pid i).map (fun y => |y - exactGeLUSpec xs i|)
        = some (|approxGeLUSpec xs i - exactGeLUSpec xs i|)
      ∧ |approxGeLUSpec xs i - exactGeLUSpec xs i| ≤ |xs i| := by
  intro i
  refine ⟨?_, ?_⟩
  · rw [approx_gelu_kernel_correct xReg outReg blockSize hN s xs h_x i]; rfl
  · exact abs_approxGeLUScalar_sub_exactGeLUScalar_le_abs (xs i)

end VeriTile.Examples
