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
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import VeriTile.Math.GeluTaylor20Cert
import VeriTile.Math.RealErf
import VeriTile.Math.Tanh
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

/-- Sharper positive tail bound used to close the approximation error from
`x ≥ 4`. Compared with `abs_approxGeLUScalar_sub_exactGeLUScalar_tail_pos`,
this keeps the cubic growth in the tanh argument and uses the Gaussian
`erf` tail `exp(-z²)/z`. -/
private lemma abs_approxGeLUScalar_sub_exactGeLUScalar_tail_pos_four
    {x : ℝ} (hx : 4 ≤ x) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤
      x * Real.exp (-(2 * (7978845608028654 / 10000000000000000 : ℝ) *
        (1 + 16 * (44715 / 1000000 : ℝ)) * x)) +
        Real.exp (-(x * x / 2)) := by
  set c : ℝ := 7978845608028654 / 10000000000000000 with hc_def
  set k : ℝ := 44715 / 1000000 with hk_def
  have hc_pos : 0 < c := c_pos
  have hk_pos : 0 < k := k_pos
  have hx_pos : 0 < x := by linarith
  have hx_ge_one : 1 ≤ x / Real.sqrt 2 := by
    have hsqrt2_le_four : Real.sqrt 2 ≤ 4 := by
      rw [show (4:ℝ) = Real.sqrt (4^2) from (Real.sqrt_sq (by norm_num)).symm]
      exact Real.sqrt_le_sqrt (by norm_num)
    have hsqrt2_pos : 0 < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
    rw [le_div_iff₀ hsqrt2_pos]
    linarith
  set u : ℝ := c * (x + k * (x * x * x)) with hu_def
  have hx_sq_ge : (16 : ℝ) ≤ x * x := by nlinarith
  have h_inner_ge : (1 + 16 * k) * x ≤ x + k * (x * x * x) := by
    have hx_nonneg : 0 ≤ x := hx_pos.le
    have hmul : 16 * x ≤ x * x * x := by nlinarith
    nlinarith
  have hu_ge : c * ((1 + 16 * k) * x) ≤ u := by
    rw [hu_def]
    exact mul_le_mul_of_nonneg_left h_inner_ge hc_pos.le
  have h_tanh_tail : 1 - Real.tanh u
      ≤ 2 * Real.exp (-(2 * c * (1 + 16 * k) * x)) := by
    have h1 := one_sub_tanh_le_two_exp_neg_two_mul u
    have h2 : Real.exp (-(2 * u))
        ≤ Real.exp (-(2 * c * (1 + 16 * k) * x)) :=
      Real.exp_le_exp.mpr (by nlinarith)
    linarith
  have h_erf_tail : 1 - realErf (x / Real.sqrt 2)
      ≤ (2 / Real.sqrt Real.pi) *
          (Real.exp (-((x / Real.sqrt 2) * (x / Real.sqrt 2))) / (x / Real.sqrt 2)) :=
    VeriTile.Math.one_sub_realErf_le_exp_neg_sq_div hx_ge_one
  rw [approxGeLUScalar_sub_exactGeLUScalar x]
  have hu_match :
      Real.tanh ((7978845608028654 / 10000000000000000 : ℝ) *
              (x + (44715 / 1000000 : ℝ) * (x * x * x)))
        = Real.tanh u := by
    rw [hu_def, hc_def, hk_def]
  rw [hu_match]
  rw [abs_mul, abs_div, abs_two, abs_of_pos hx_pos]
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
        2 * Real.exp (-(2 * c * (1 + 16 * k) * x)) +
          (2 / Real.sqrt Real.pi) *
            (Real.exp (-((x / Real.sqrt 2) * (x / Real.sqrt 2))) / (x / Real.sqrt 2)) := by
    exact h_gap.trans (add_le_add h_tanh_tail h_erf_tail)
  have hx2 : 0 ≤ x / 2 := by positivity
  have h_pre :
      (x / 2) * |Real.tanh u - realErf (x / Real.sqrt 2)| ≤
        x * Real.exp (-(2 * c * (1 + 16 * k) * x)) +
          (x / Real.sqrt Real.pi) *
            (Real.exp (-((x / Real.sqrt 2) * (x / Real.sqrt 2))) / (x / Real.sqrt 2)) := by
    calc (x / 2) * |Real.tanh u - realErf (x / Real.sqrt 2)|
        ≤ (x / 2) *
            (2 * Real.exp (-(2 * c * (1 + 16 * k) * x)) +
              (2 / Real.sqrt Real.pi) *
                (Real.exp (-((x / Real.sqrt 2) * (x / Real.sqrt 2))) / (x / Real.sqrt 2))) := by
              exact mul_le_mul_of_nonneg_left h_gap_bound hx2
      _ = x * Real.exp (-(2 * c * (1 + 16 * k) * x)) +
          (x / Real.sqrt Real.pi) *
            (Real.exp (-((x / Real.sqrt 2) * (x / Real.sqrt 2))) / (x / Real.sqrt 2)) := by
            field_simp [hx_pos.ne']
  have hsqrt2_pos : 0 < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  have hpi_pos : 0 < Real.sqrt Real.pi := Real.sqrt_pos.mpr Real.pi_pos
  have hsqrt2_le_sqrtpi : Real.sqrt 2 ≤ Real.sqrt Real.pi := by
    exact Real.sqrt_le_sqrt (by linarith [Real.pi_gt_three])
  have h_erf_term :
      (x / Real.sqrt Real.pi) *
          (Real.exp (-((x / Real.sqrt 2) * (x / Real.sqrt 2))) / (x / Real.sqrt 2))
        ≤ Real.exp (-(x * x / 2)) := by
    have h_exp_arg :
        -((x / Real.sqrt 2) * (x / Real.sqrt 2)) = -(x * x / 2) := by
      have h2 : (Real.sqrt 2)^2 = 2 := Real.sq_sqrt (by norm_num)
      field_simp [h2]
      nlinarith [h2]
    rw [h_exp_arg]
    have h_coeff : (x / Real.sqrt Real.pi) / (x / Real.sqrt 2) ≤ 1 := by
      field_simp [hx_pos.ne', hpi_pos.ne', hsqrt2_pos.ne']
      exact hsqrt2_le_sqrtpi
    have h_coeff_nonneg : 0 ≤ (x / Real.sqrt Real.pi) / (x / Real.sqrt 2) := by positivity
    calc (x / Real.sqrt Real.pi) *
          (Real.exp (-(x * x / 2)) / (x / Real.sqrt 2))
        = ((x / Real.sqrt Real.pi) / (x / Real.sqrt 2)) *
            Real.exp (-(x * x / 2)) := by ring
      _ ≤ 1 * Real.exp (-(x * x / 2)) := by
            exact mul_le_mul_of_nonneg_right h_coeff (Real.exp_pos _).le
      _ = Real.exp (-(x * x / 2)) := by ring
  exact h_pre.trans (add_le_add le_rfl h_erf_term)

/-- Sharper positive tail bound from `x ≥ 19/5`. This is the same argument as
`abs_approxGeLUScalar_sub_exactGeLUScalar_tail_pos_four`, with the cubic lower
bound specialized at `(19/5)^2 = 361/25`. -/
private lemma abs_approxGeLUScalar_sub_exactGeLUScalar_tail_pos_19_5
    {x : ℝ} (hx : (19/5 : ℝ) ≤ x) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤
      x * Real.exp (-(2 * (7978845608028654 / 10000000000000000 : ℝ) *
        (1 + (361/25 : ℝ) * (44715 / 1000000 : ℝ)) * x)) +
        Real.exp (-(x * x / 2)) := by
  set c : ℝ := 7978845608028654 / 10000000000000000 with hc_def
  set k : ℝ := 44715 / 1000000 with hk_def
  have hc_pos : 0 < c := c_pos
  have hk_pos : 0 < k := k_pos
  have hx_pos : 0 < x := by nlinarith
  have hx_ge_one : 1 ≤ x / Real.sqrt 2 := by
    have hsqrt2_le : Real.sqrt 2 ≤ (19/5 : ℝ) := by
      rw [show (19/5:ℝ) = Real.sqrt ((19/5)^2) from
        (Real.sqrt_sq (by norm_num)).symm]
      exact Real.sqrt_le_sqrt (by norm_num)
    have hsqrt2_pos : 0 < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
    rw [le_div_iff₀ hsqrt2_pos]
    linarith
  set u : ℝ := c * (x + k * (x * x * x)) with hu_def
  have hx_sq_ge : (361/25 : ℝ) ≤ x * x := by nlinarith
  have h_inner_ge : (1 + (361/25 : ℝ) * k) * x ≤ x + k * (x * x * x) := by
    have hmul : (361/25 : ℝ) * x ≤ x * x * x := by nlinarith
    nlinarith
  have hu_ge : c * ((1 + (361/25 : ℝ) * k) * x) ≤ u := by
    rw [hu_def]
    exact mul_le_mul_of_nonneg_left h_inner_ge hc_pos.le
  have h_tanh_tail : 1 - Real.tanh u
      ≤ 2 * Real.exp (-(2 * c * (1 + (361/25 : ℝ) * k) * x)) := by
    have h1 := one_sub_tanh_le_two_exp_neg_two_mul u
    have h2 : Real.exp (-(2 * u))
        ≤ Real.exp (-(2 * c * (1 + (361/25 : ℝ) * k) * x)) :=
      Real.exp_le_exp.mpr (by nlinarith)
    linarith
  have h_erf_tail : 1 - realErf (x / Real.sqrt 2)
      ≤ (2 / Real.sqrt Real.pi) *
          (Real.exp (-((x / Real.sqrt 2) * (x / Real.sqrt 2))) / (x / Real.sqrt 2)) :=
    VeriTile.Math.one_sub_realErf_le_exp_neg_sq_div hx_ge_one
  rw [approxGeLUScalar_sub_exactGeLUScalar x]
  have hu_match :
      Real.tanh ((7978845608028654 / 10000000000000000 : ℝ) *
              (x + (44715 / 1000000 : ℝ) * (x * x * x)))
        = Real.tanh u := by
    rw [hu_def, hc_def, hk_def]
  rw [hu_match]
  rw [abs_mul, abs_div, abs_two, abs_of_pos hx_pos]
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
        2 * Real.exp (-(2 * c * (1 + (361/25 : ℝ) * k) * x)) +
          (2 / Real.sqrt Real.pi) *
            (Real.exp (-((x / Real.sqrt 2) * (x / Real.sqrt 2))) / (x / Real.sqrt 2)) := by
    exact h_gap.trans (add_le_add h_tanh_tail h_erf_tail)
  have hx2 : 0 ≤ x / 2 := by positivity
  have h_pre :
      (x / 2) * |Real.tanh u - realErf (x / Real.sqrt 2)| ≤
        x * Real.exp (-(2 * c * (1 + (361/25 : ℝ) * k) * x)) +
          (x / Real.sqrt Real.pi) *
            (Real.exp (-((x / Real.sqrt 2) * (x / Real.sqrt 2))) / (x / Real.sqrt 2)) := by
    calc (x / 2) * |Real.tanh u - realErf (x / Real.sqrt 2)|
        ≤ (x / 2) *
            (2 * Real.exp (-(2 * c * (1 + (361/25 : ℝ) * k) * x)) +
              (2 / Real.sqrt Real.pi) *
                (Real.exp (-((x / Real.sqrt 2) * (x / Real.sqrt 2))) / (x / Real.sqrt 2))) := by
              exact mul_le_mul_of_nonneg_left h_gap_bound hx2
      _ = x * Real.exp (-(2 * c * (1 + (361/25 : ℝ) * k) * x)) +
          (x / Real.sqrt Real.pi) *
            (Real.exp (-((x / Real.sqrt 2) * (x / Real.sqrt 2))) / (x / Real.sqrt 2)) := by
            field_simp [hx_pos.ne']
  have hsqrt2_pos : 0 < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  have hpi_pos : 0 < Real.sqrt Real.pi := Real.sqrt_pos.mpr Real.pi_pos
  have hsqrt2_le_sqrtpi : Real.sqrt 2 ≤ Real.sqrt Real.pi := by
    exact Real.sqrt_le_sqrt (by linarith [Real.pi_gt_three])
  have h_erf_term :
      (x / Real.sqrt Real.pi) *
          (Real.exp (-((x / Real.sqrt 2) * (x / Real.sqrt 2))) / (x / Real.sqrt 2))
        ≤ Real.exp (-(x * x / 2)) := by
    have h_exp_arg :
        -((x / Real.sqrt 2) * (x / Real.sqrt 2)) = -(x * x / 2) := by
      have h2 : (Real.sqrt 2)^2 = 2 := Real.sq_sqrt (by norm_num)
      field_simp [h2]
      nlinarith [h2]
    rw [h_exp_arg]
    have h_coeff : (x / Real.sqrt Real.pi) / (x / Real.sqrt 2) ≤ 1 := by
      field_simp [hx_pos.ne', hpi_pos.ne', hsqrt2_pos.ne']
      exact hsqrt2_le_sqrtpi
    calc (x / Real.sqrt Real.pi) *
          (Real.exp (-(x * x / 2)) / (x / Real.sqrt 2))
        = ((x / Real.sqrt Real.pi) / (x / Real.sqrt 2)) *
            Real.exp (-(x * x / 2)) := by ring
      _ ≤ 1 * Real.exp (-(x * x / 2)) := by
            exact mul_le_mul_of_nonneg_right h_coeff (Real.exp_pos _).le
      _ = Real.exp (-(x * x / 2)) := by ring
  exact h_pre.trans (add_le_add le_rfl h_erf_term)

/-! ## Taylor expansions of `tanh u` and `realErf z` at the gelu arguments

For the medium range `1/1000 < |x| < 20`, the asymptotic tail bounds of
`abs_approxGeLUScalar_sub_exactGeLUScalar_tail_pos` are too loose. The
tighter route is via Taylor expansions of `tanh u` and `realErf z` with the
specific substitutions `u = c·(x + k·x³)` and `z = x/√2`. These two lemmas
discharge the prerequisite (`|u| ≤ 1`, `|z| ≤ 1`) for `|x| ≤ 1` and apply the
septenary Taylor bounds from `VeriTile.Math.Tanh` and `VeriTile.Math.RealErf`. -/

/-- For `|x| ≤ 1`, `|u| = |c·(x + k·x³)| ≤ c·(1 + k) < 1` (numerically
≈ 0.834), so `abs_tanh_sub_taylor5_le` applies and gives the seventh-order
Taylor remainder bound on `tanh u`. -/
lemma abs_tanh_at_gelu_arg_sub_taylor5_le {x : ℝ} (hx : |x| ≤ 1) :
    |Real.tanh ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))) -
        ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x)) -
         ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))) ^ 3 / 3 +
         2 * ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))) ^ 5 / 15)|
      ≤ |(7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))| ^ 7 / 7 := by
  set u : ℝ := (7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x)) with hu_def
  have hu_bound : |u| ≤ 1 := by
    rw [hu_def, abs_mul]
    have h_c_pos : 0 < (7978845608028654 / 10000000000000000 : ℝ) := by norm_num
    rw [abs_of_pos h_c_pos]
    have h_inner : |x + (44715 / 1000000 : ℝ) * (x * x * x)|
        ≤ |x| + (44715 / 1000000) * |x| ^ 3 := by
      have hb := abs_add_le x ((44715 / 1000000 : ℝ) * (x * x * x))
      have h_eq : |(44715 / 1000000 : ℝ) * (x * x * x)|
            = (44715 / 1000000) * |x| ^ 3 := by
        rw [abs_mul, show |(44715 / 1000000 : ℝ)| = 44715 / 1000000 from by norm_num]
        congr 1
        rw [show x * x * x = x ^ 3 from by ring, abs_pow]
      linarith
    have h_x3_le : |x| ^ 3 ≤ |x| := by
      nlinarith [sq_nonneg (|x| - 1), sq_nonneg (|x|), abs_nonneg x]
    have h_inner_le : |x + (44715 / 1000000 : ℝ) * (x * x * x)|
          ≤ |x| * (1 + 44715 / 1000000) := by
      have h1 : |x| + (44715 / 1000000 : ℝ) * |x| ^ 3
            ≤ |x| + (44715 / 1000000) * |x| := by
        nlinarith
      have h_factor : |x| + (44715 / 1000000 : ℝ) * |x| = |x| * (1 + 44715 / 1000000) := by ring
      linarith
    calc (7978845608028654 / 10000000000000000 : ℝ) *
            |x + (44715 / 1000000) * (x * x * x)|
        ≤ (7978845608028654 / 10000000000000000) * (|x| * (1 + 44715 / 1000000)) := by
            apply mul_le_mul_of_nonneg_left h_inner_le (by norm_num)
      _ ≤ (7978845608028654 / 10000000000000000) * (1 * (1 + 44715 / 1000000)) := by
            apply mul_le_mul_of_nonneg_left
            · exact mul_le_mul_of_nonneg_right hx (by norm_num)
            · norm_num
      _ ≤ 1 := by norm_num
  exact abs_tanh_sub_taylor5_le hu_bound

/-- For `|x| ≤ 1`, `|x/√2| ≤ 1/√2 ≤ 1`, so `abs_realErf_sub_taylor7_le`
applies and gives the seventh-order Taylor remainder bound on `realErf z`. -/
lemma abs_realErf_at_gelu_arg_sub_taylor7_le {x : ℝ} (hx : |x| ≤ 1) :
    |realErf (x / Real.sqrt 2) -
        (2 / Real.sqrt Real.pi) *
          ((x / Real.sqrt 2) - (x / Real.sqrt 2) ^ 3 / 3 +
           (x / Real.sqrt 2) ^ 5 / 10 - (x / Real.sqrt 2) ^ 7 / 42)|
      ≤ (2 / Real.sqrt Real.pi) * (5 * |x / Real.sqrt 2| ^ 9 / 864) := by
  have hsqrt2_pos : 0 < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  have hsqrt2_ge_1 : 1 ≤ Real.sqrt 2 := by
    rw [show (1:ℝ) = Real.sqrt 1 from Real.sqrt_one.symm]
    exact Real.sqrt_le_sqrt (by norm_num)
  have hz_bound : |x / Real.sqrt 2| ≤ 1 := by
    rw [abs_div, abs_of_pos hsqrt2_pos, div_le_iff₀ hsqrt2_pos]
    linarith
  exact abs_realErf_sub_taylor7_le hz_bound

/-! ## Numeric closure for very-large `|x|`

Combines `abs_approxGeLUScalar_sub_exactGeLUScalar_tail_pos` with a generic
monotonicity helper for `x · exp(−a·x)` and a concrete numeric verification
at `T = 20` to close `|approx − exact| ≤ 1/1000` whenever `|x| ≥ 20`. -/

/-- For `a · T ≥ 1` and `T > 0`, the function `x ↦ x · exp(−a·x)` is bounded
above by its value at `T` on the half-line `[T, ∞)`. The proof uses
`Real.add_one_le_exp`. -/
private lemma x_exp_neg_ax_mono {a T x : ℝ}
    (haT : 1 ≤ a * T) (hT : 0 < T) (hx : T ≤ x) :
    x * Real.exp (-(a * x)) ≤ T * Real.exp (-(a * T)) := by
  have hxmT : 0 ≤ x - T := by linarith
  have h_exp_lin : 1 + a * (x - T) ≤ Real.exp (a * (x - T)) := by
    have := Real.add_one_le_exp (a * (x - T)); linarith
  have h_aT_x : x - T ≤ a * T * (x - T) := by nlinarith
  have key : x ≤ T * Real.exp (a * (x - T)) := by
    have h1 : T * (1 + a * (x - T)) = T + a * T * (x - T) := by ring
    have h2 : T + (x - T) ≤ T * (1 + a * (x - T)) := by
      rw [h1]; linarith
    have h3 : T + (x - T) = x := by ring
    have h4 : T * (1 + a * (x - T)) ≤ T * Real.exp (a * (x - T)) :=
      mul_le_mul_of_nonneg_left h_exp_lin hT.le
    linarith
  have h_exp_pos : 0 < Real.exp (-(a * x)) := Real.exp_pos _
  have h_combine : Real.exp (a * (x - T)) * Real.exp (-(a * x)) =
        Real.exp (-(a * T)) := by
    rw [← Real.exp_add]; congr 1; ring
  calc x * Real.exp (-(a * x))
      ≤ T * Real.exp (a * (x - T)) * Real.exp (-(a * x)) :=
        mul_le_mul_of_nonneg_right key h_exp_pos.le
    _ = T * (Real.exp (a * (x - T)) * Real.exp (-(a * x))) := by ring
    _ = T * Real.exp (-(a * T)) := by rw [h_combine]

/-- Concrete verification at `T = 20`: the asymptotic tail bound evaluated at
`20` is ≤ `1/1000`. The numeric chain uses `√π ≥ 3/2`, `√2 ≤ 10/7`,
`c ≥ 7/10` (so `2c·20 ≥ 28`), and `exp(N) ≥ (27/10)^N` from
`Real.exp_one_gt_d9`. -/
private lemma gelu_tail_at_twenty :
    20 * Real.exp (-(2 * (7978845608028654 / 10000000000000000 : ℝ) * 20)) +
      (20 / Real.sqrt Real.pi) * Real.exp (-(20 / Real.sqrt 2)) ≤ 1 / 1000 := by
  have hpi : (3/2 : ℝ) ≤ Real.sqrt Real.pi := by
    rw [show (3/2:ℝ) = Real.sqrt ((3/2)^2) from (Real.sqrt_sq (by norm_num)).symm]
    apply Real.sqrt_le_sqrt
    have : ((3/2:ℝ))^2 = 9/4 := by norm_num
    rw [this]; linarith [Real.pi_gt_three]
  have hpi_pos : 0 < Real.sqrt Real.pi := Real.sqrt_pos.mpr Real.pi_pos
  have hsqrt2 : Real.sqrt 2 ≤ 10/7 := by
    rw [show (10/7:ℝ) = Real.sqrt ((10/7)^2) from (Real.sqrt_sq (by norm_num)).symm]
    exact Real.sqrt_le_sqrt (by norm_num)
  have hsqrt2_pos : 0 < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  have h_ratio_pi : 20 / Real.sqrt Real.pi ≤ 40 / 3 := by
    rw [div_le_div_iff₀ hpi_pos (by norm_num : (0:ℝ) < 3)]
    nlinarith [hpi]
  have h_ratio_2 : (14 : ℝ) ≤ 20 / Real.sqrt 2 := by
    rw [le_div_iff₀ hsqrt2_pos]; nlinarith [hsqrt2]
  have h_2c20 : (28 : ℝ) ≤
        2 * (7978845608028654 / 10000000000000000 : ℝ) * 20 := by norm_num
  have h_exp_2 : Real.exp (-(20 / Real.sqrt 2)) ≤ Real.exp (-(14 : ℝ)) :=
    Real.exp_le_exp.mpr (by linarith)
  have h_exp_c : Real.exp (-(2 * (7978845608028654 / 10000000000000000 : ℝ) * 20))
        ≤ Real.exp (-(28 : ℝ)) := Real.exp_le_exp.mpr (by linarith)
  have h27 : (2.7 : ℝ) ≤ Real.exp 1 := by linarith [Real.exp_one_gt_d9]
  have h_exp_14 : Real.exp (-(14 : ℝ)) ≤ 1/1000000 := by
    have h_e14 : Real.exp 14 = Real.exp 1 ^ 14 := by
      rw [show (14 : ℝ) = ((14:ℕ) : ℝ) * 1 from by norm_num, Real.exp_nat_mul]
    have h_e14_lower : (1000000 : ℝ) ≤ Real.exp 14 := by
      rw [h_e14]
      calc (1000000:ℝ) ≤ (2.7:ℝ)^14 := by norm_num
        _ ≤ Real.exp 1 ^ 14 := pow_le_pow_left₀ (by norm_num) h27 14
    rw [Real.exp_neg, le_div_iff₀ (by norm_num : (0:ℝ) < 1000000),
        inv_mul_le_iff₀ (Real.exp_pos _)]; linarith
  have h_exp_28 : Real.exp (-(28 : ℝ)) ≤ 1/1000000000000 := by
    have h_e28 : Real.exp 28 = Real.exp 1 ^ 28 := by
      rw [show (28 : ℝ) = ((28:ℕ) : ℝ) * 1 from by norm_num, Real.exp_nat_mul]
    have h_e28_lower : (1000000000000 : ℝ) ≤ Real.exp 28 := by
      rw [h_e28]
      calc (1000000000000:ℝ) ≤ (2.7:ℝ)^28 := by norm_num
        _ ≤ Real.exp 1 ^ 28 := pow_le_pow_left₀ (by norm_num) h27 28
    rw [Real.exp_neg, le_div_iff₀ (by norm_num : (0:ℝ) < 1000000000000),
        inv_mul_le_iff₀ (Real.exp_pos _)]; linarith
  have h_term1 :
      20 * Real.exp (-(2 * (7978845608028654 / 10000000000000000 : ℝ) * 20))
        ≤ 20 * (1/1000000000000) := by
    have : Real.exp (-(2 * (7978845608028654 / 10000000000000000 : ℝ) * 20))
            ≤ 1/1000000000000 := h_exp_c.trans h_exp_28
    linarith
  have h_term2 : (20 / Real.sqrt Real.pi) * Real.exp (-(20 / Real.sqrt 2))
        ≤ (40/3) * (1/1000000) :=
    mul_le_mul h_ratio_pi (h_exp_2.trans h_exp_14)
      (Real.exp_pos _).le (by norm_num)
  have h_sum_bound :
      20 * (1/1000000000000 : ℝ) + (40/3) * (1/1000000) ≤ 1/1000 := by norm_num
  linarith

/-- Concrete verification at `T = 4` for the sharper tail bound:
`4·exp(-2c(1+16k)·4) + exp(-8) ≤ 1/1000`. -/
private lemma gelu_sharp_tail_at_four :
    4 * Real.exp (-(2 * (7978845608028654 / 10000000000000000 : ℝ) *
      (1 + 16 * (44715 / 1000000 : ℝ)) * 4)) +
        Real.exp (-(4 * 4 / 2 : ℝ)) ≤ 1 / 1000 := by
  have h_arg1 : (10 : ℝ) ≤
      2 * (7978845608028654 / 10000000000000000 : ℝ) *
        (1 + 16 * (44715 / 1000000 : ℝ)) * 4 := by norm_num
  have h_exp_1 : Real.exp
      (-(2 * (7978845608028654 / 10000000000000000 : ℝ) *
        (1 + 16 * (44715 / 1000000 : ℝ)) * 4))
        ≤ Real.exp (-(10 : ℝ)) :=
    Real.exp_le_exp.mpr (by linarith)
  have h27 : (2.7 : ℝ) ≤ Real.exp 1 := by linarith [Real.exp_one_gt_d9]
  have h_exp_10 : Real.exp (-(10 : ℝ)) ≤ 1/20000 := by
    have h_e10 : Real.exp 10 = Real.exp 1 ^ 10 := by
      rw [show (10 : ℝ) = ((10:ℕ) : ℝ) * 1 from by norm_num, Real.exp_nat_mul]
    have h_e10_lower : (20000 : ℝ) ≤ Real.exp 10 := by
      rw [h_e10]
      calc (20000:ℝ) ≤ (2.7:ℝ)^10 := by norm_num
        _ ≤ Real.exp 1 ^ 10 := pow_le_pow_left₀ (by norm_num) h27 10
    rw [Real.exp_neg, le_div_iff₀ (by norm_num : (0:ℝ) < 20000),
        inv_mul_le_iff₀ (Real.exp_pos _)]; linarith
  have h_exp_8 : Real.exp (-(8 : ℝ)) ≤ 1/2700 := by
    have h_e8 : Real.exp 8 = Real.exp 1 ^ 8 := by
      rw [show (8 : ℝ) = ((8:ℕ) : ℝ) * 1 from by norm_num, Real.exp_nat_mul]
    have h_e8_lower : (2700 : ℝ) ≤ Real.exp 8 := by
      rw [h_e8]
      calc (2700:ℝ) ≤ (2.7:ℝ)^8 := by norm_num
        _ ≤ Real.exp 1 ^ 8 := pow_le_pow_left₀ (by norm_num) h27 8
    rw [Real.exp_neg, le_div_iff₀ (by norm_num : (0:ℝ) < 2700),
        inv_mul_le_iff₀ (Real.exp_pos _)]; linarith
  have h_term1 :
      4 * Real.exp (-(2 * (7978845608028654 / 10000000000000000 : ℝ) *
        (1 + 16 * (44715 / 1000000 : ℝ)) * 4))
        ≤ 4 * (1/20000) := by
    linarith
  have h_arg2 : -(4 * 4 / 2 : ℝ) = -(8 : ℝ) := by norm_num
  have h_term2 : Real.exp (-(4 * 4 / 2 : ℝ)) ≤ 1/2700 := by
    rw [h_arg2]
    exact h_exp_8
  have h_sum : 4 * (1/20000 : ℝ) + 1/2700 ≤ 1/1000 := by norm_num
  linarith

/-- Concrete verification at `T = 19/5` for the sharper tail bound:
`(19/5)·exp(-99/10) + exp(-36/5) ≤ 1/1000`. -/
private lemma gelu_sharp_tail_at_19_5 :
    (19/5 : ℝ) * Real.exp (-(99/10 : ℝ)) + Real.exp (-(36/5 : ℝ))
      ≤ 1 / 1000 := by
  have h27 : (2.71 : ℝ) ≤ Real.exp 1 := by linarith [Real.exp_one_gt_d9]
  have h_exp_9_10 : (461/200 : ℝ) ≤ Real.exp (9/10) := by
    have hsum := Real.sum_le_exp_of_nonneg (by norm_num : (0:ℝ) ≤ 9/10) 3
    norm_num [Finset.sum_range_succ] at hsum ⊢
    linarith
  have h_exp_99_10 : Real.exp (-(99/10 : ℝ)) ≤ 1/17000 := by
    have h_e99 : Real.exp (99/10 : ℝ) = Real.exp 1 ^ 9 * Real.exp (9/10) := by
      rw [show (99/10 : ℝ) = (9:ℝ) * 1 + 9/10 by norm_num, Real.exp_add]
      rw [show (9:ℝ) * 1 = ((9:ℕ) : ℝ) * 1 by norm_num, Real.exp_nat_mul]
    have h_e99_lower : (17000 : ℝ) ≤ Real.exp (99/10 : ℝ) := by
      rw [h_e99]
      calc (17000:ℝ) ≤ (2.71:ℝ)^9 * (461/200:ℝ) := by norm_num
        _ ≤ Real.exp 1 ^ 9 * Real.exp (9/10) := by
            exact mul_le_mul
              (pow_le_pow_left₀ (by norm_num) h27 9)
              h_exp_9_10
              (by positivity)
              (by positivity)
    rw [Real.exp_neg, le_div_iff₀ (by norm_num : (0:ℝ) < 17000),
        inv_mul_le_iff₀ (Real.exp_pos _)]; linarith
  have h_exp_one_fifth : (61/50 : ℝ) ≤ Real.exp (1/5) := by
    have hsum := Real.sum_le_exp_of_nonneg (by norm_num : (0:ℝ) ≤ 1/5) 3
    norm_num [Finset.sum_range_succ] at hsum ⊢
    linarith
  have h_exp_36_5 : Real.exp (36/5 : ℝ) = Real.exp 1 ^ 7 * Real.exp (1/5) := by
    rw [show (36/5 : ℝ) = (7:ℝ) * 1 + 1/5 by norm_num, Real.exp_add]
    rw [show (7:ℝ) * 1 = ((7:ℕ) : ℝ) * 1 by norm_num, Real.exp_nat_mul]
  have h_exp_36_5_lower : (1300 : ℝ) ≤ Real.exp (36/5) := by
    rw [h_exp_36_5]
    calc (1300:ℝ) ≤ (2.71:ℝ)^7 * (61/50:ℝ) := by norm_num
      _ ≤ Real.exp 1 ^ 7 * Real.exp (1/5) := by
          exact mul_le_mul
            (pow_le_pow_left₀ (by norm_num) h27 7)
            h_exp_one_fifth
            (by positivity)
            (by positivity)
  have h_exp_36_5_neg : Real.exp (-(36/5 : ℝ)) ≤ 1/1300 := by
    rw [Real.exp_neg, le_div_iff₀ (by norm_num : (0:ℝ) < 1300),
        inv_mul_le_iff₀ (Real.exp_pos _)]; linarith
  have h_term1 : (19/5 : ℝ) * Real.exp (-(99/10 : ℝ)) ≤ (19/5) * (1/17000) := by
    exact mul_le_mul_of_nonneg_left h_exp_99_10 (by norm_num)
  have h_sum : (19/5 : ℝ) * (1/17000) + 1/1300 ≤ 1/1000 := by norm_num
  linarith

/-- Closure of the very-large positive case: for `x ≥ 20` the gelu
approximation error is at most `1/1000`. -/
private theorem approx_gelu_error_bound_very_large_pos {x : ℝ} (hx : 20 ≤ x) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ 1 / 1000 := by
  have hsqrt2_le_20 : Real.sqrt 2 ≤ 20 := by
    have : Real.sqrt 2 ≤ 10/7 := by
      rw [show (10/7:ℝ) = Real.sqrt ((10/7)^2) from
            (Real.sqrt_sq (by norm_num)).symm]
      exact Real.sqrt_le_sqrt (by norm_num)
    linarith
  have hsqrt2_le_x : Real.sqrt 2 ≤ x := le_trans hsqrt2_le_20 hx
  have h_tail := abs_approxGeLUScalar_sub_exactGeLUScalar_tail_pos hsqrt2_le_x
  set c : ℝ := 7978845608028654 / 10000000000000000
  -- Monotonicity for the c-tail: x·exp(-2c·x) ≤ 20·exp(-2c·20).
  have h_2c20 : (1 : ℝ) ≤ (2 * c) * 20 := by
    show (1 : ℝ) ≤ 2 * (7978845608028654 / 10000000000000000) * 20
    norm_num
  have h_mono_c : x * Real.exp (-(2 * c * x)) ≤ 20 * Real.exp (-(2 * c * 20)) :=
    x_exp_neg_ax_mono h_2c20 (by norm_num) hx
  -- Monotonicity for the erf-tail: write (x/√π)·exp(-x/√2) = (1/√π)·(x·exp(-(1/√2)·x))
  -- and apply the helper with a = 1/√2.
  have hsqrt2_pos : 0 < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  have hpi_pos : 0 < Real.sqrt Real.pi := Real.sqrt_pos.mpr Real.pi_pos
  have h_ratio_2 : (14 : ℝ) ≤ 20 / Real.sqrt 2 := by
    rw [le_div_iff₀ hsqrt2_pos]
    have hsqrt2_le : Real.sqrt 2 ≤ 10/7 := by
      rw [show (10/7:ℝ) = Real.sqrt ((10/7)^2) from
            (Real.sqrt_sq (by norm_num)).symm]
      exact Real.sqrt_le_sqrt (by norm_num)
    nlinarith
  have h_inv_sqrt2_20 : (1 : ℝ) ≤ (1 / Real.sqrt 2) * 20 := by
    rw [one_div]; rw [show (1:ℝ) = (Real.sqrt 2)⁻¹ * Real.sqrt 2 from by
      rw [inv_mul_cancel₀ hsqrt2_pos.ne']]
    exact mul_le_mul_of_nonneg_left
      (by linarith : Real.sqrt 2 ≤ 20) (inv_nonneg.mpr hsqrt2_pos.le)
  have h_mono_erf_unscaled :
      x * Real.exp (-((1 / Real.sqrt 2) * x))
        ≤ 20 * Real.exp (-((1 / Real.sqrt 2) * 20)) :=
    x_exp_neg_ax_mono h_inv_sqrt2_20 (by norm_num) hx
  have h_eq_form_x : x * Real.exp (-(x / Real.sqrt 2))
        = x * Real.exp (-((1 / Real.sqrt 2) * x)) := by
    congr 2; rw [div_eq_inv_mul, one_div]
  have h_eq_form_20 : (20 : ℝ) * Real.exp (-(20 / Real.sqrt 2))
        = 20 * Real.exp (-((1 / Real.sqrt 2) * 20)) := by
    congr 2; rw [div_eq_inv_mul, one_div]
  have h_mono_erf_xform : x * Real.exp (-(x / Real.sqrt 2))
        ≤ 20 * Real.exp (-(20 / Real.sqrt 2)) := by
    rw [h_eq_form_x, h_eq_form_20]; exact h_mono_erf_unscaled
  -- Scale by 1/√π:
  have h_mono_erf : (x / Real.sqrt Real.pi) * Real.exp (-(x / Real.sqrt 2))
        ≤ (20 / Real.sqrt Real.pi) * Real.exp (-(20 / Real.sqrt 2)) := by
    rw [div_eq_inv_mul (a := x), div_eq_inv_mul (a := 20)]
    rw [mul_assoc, mul_assoc]
    exact mul_le_mul_of_nonneg_left h_mono_erf_xform (inv_nonneg.mpr hpi_pos.le)
  -- Combine
  have h_combined :
      x * Real.exp (-(2 * c * x)) +
        (x / Real.sqrt Real.pi) * Real.exp (-(x / Real.sqrt 2))
      ≤ 20 * Real.exp (-(2 * c * 20)) +
          (20 / Real.sqrt Real.pi) * Real.exp (-(20 / Real.sqrt 2)) :=
    add_le_add h_mono_c h_mono_erf
  exact le_trans h_tail (le_trans h_combined gelu_tail_at_twenty)

/-- Closure of the positive tail from `x ≥ 4` using the sharper Gaussian
tail bound. -/
private theorem approx_gelu_error_bound_tail_pos_four {x : ℝ} (hx : 4 ≤ x) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ 1 / 1000 := by
  have h_tail := abs_approxGeLUScalar_sub_exactGeLUScalar_tail_pos_four hx
  set c : ℝ := 7978845608028654 / 10000000000000000
  set k : ℝ := 44715 / 1000000
  have h_mono_c : x * Real.exp (-(2 * c * (1 + 16 * k) * x))
      ≤ 4 * Real.exp (-(2 * c * (1 + 16 * k) * 4)) := by
    apply x_exp_neg_ax_mono
    · show (1 : ℝ) ≤ (2 * (7978845608028654 / 10000000000000000) *
          (1 + 16 * (44715 / 1000000))) * 4
      norm_num
    · norm_num
    · exact hx
  have h_exp_arg : -(x * x / 2) ≤ -(4 * 4 / 2 : ℝ) := by
    have hx_sq : (4 : ℝ) * 4 ≤ x * x := by nlinarith
    nlinarith
  have h_mono_erf : Real.exp (-(x * x / 2)) ≤ Real.exp (-(4 * 4 / 2 : ℝ)) :=
    Real.exp_le_exp.mpr h_exp_arg
  have h_combined :
      x * Real.exp (-(2 * c * (1 + 16 * k) * x)) + Real.exp (-(x * x / 2))
        ≤ 4 * Real.exp (-(2 * c * (1 + 16 * k) * 4)) +
            Real.exp (-(4 * 4 / 2 : ℝ)) :=
    add_le_add h_mono_c h_mono_erf
  exact h_tail.trans (h_combined.trans gelu_sharp_tail_at_four)

/-- Closure of the positive tail from `x ≥ 19/5`. -/
private theorem approx_gelu_error_bound_tail_pos_19_5 {x : ℝ} (hx : (19/5 : ℝ) ≤ x) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ 1 / 1000 := by
  have h_tail := abs_approxGeLUScalar_sub_exactGeLUScalar_tail_pos_19_5 hx
  set c : ℝ := 7978845608028654 / 10000000000000000
  set k : ℝ := 44715 / 1000000
  have h_mono_c : x * Real.exp (-(2 * c * (1 + (361/25 : ℝ) * k) * x))
      ≤ (19/5 : ℝ) * Real.exp (-(2 * c * (1 + (361/25 : ℝ) * k) * (19/5))) := by
    apply x_exp_neg_ax_mono
    · show (1 : ℝ) ≤ (2 * (7978845608028654 / 10000000000000000) *
          (1 + (361/25 : ℝ) * (44715 / 1000000))) * (19/5)
      norm_num
    · norm_num
    · exact hx
  have h_exp_arg : -(x * x / 2) ≤ -(36/5 : ℝ) := by
    have hx_sq : (361/25 : ℝ) ≤ x * x := by nlinarith
    nlinarith
  have h_mono_erf : Real.exp (-(x * x / 2)) ≤ Real.exp (-(36/5 : ℝ)) :=
    Real.exp_le_exp.mpr h_exp_arg
  have h_arg1 : (99/10 : ℝ) ≤
      2 * (7978845608028654 / 10000000000000000 : ℝ) *
        (1 + (361/25 : ℝ) * (44715 / 1000000 : ℝ)) * (19/5) := by norm_num
  have h_exp1 :
      Real.exp (-(2 * c * (1 + (361/25 : ℝ) * k) * (19/5)))
        ≤ Real.exp (-(99/10 : ℝ)) := by
    rw [show c = (7978845608028654 / 10000000000000000 : ℝ) by rfl,
      show k = (44715 / 1000000 : ℝ) by rfl]
    exact Real.exp_le_exp.mpr (by linarith)
  have h_term1 :
      (19/5 : ℝ) * Real.exp (-(2 * c * (1 + (361/25 : ℝ) * k) * (19/5)))
        ≤ (19/5 : ℝ) * Real.exp (-(99/10 : ℝ)) := by
    exact mul_le_mul_of_nonneg_left h_exp1 (by norm_num)
  have h_combined :
      x * Real.exp (-(2 * c * (1 + (361/25 : ℝ) * k) * x)) + Real.exp (-(x * x / 2))
        ≤ (19/5 : ℝ) * Real.exp (-(99/10 : ℝ)) + Real.exp (-(36/5 : ℝ)) := by
    exact (add_le_add h_mono_c h_mono_erf).trans (add_le_add h_term1 le_rfl)
  exact h_tail.trans (h_combined.trans gelu_sharp_tail_at_19_5)

/-- Evenness of the gelu approximation error in the input variable. The
gelu approximation error has the symmetry `f(−x) = f(x)` even though neither
the approximate nor the exact function is itself even or odd; the evenness
falls out of the algebraic decomposition once `tanh` and `realErf` are odd. -/
private lemma approxSubExact_neg (x : ℝ) :
    approxGeLUScalar (-x) - exactGeLUScalar (-x)
      = approxGeLUScalar x - exactGeLUScalar x := by
  rw [approxGeLUScalar_sub_exactGeLUScalar, approxGeLUScalar_sub_exactGeLUScalar]
  have h_arg :
      (7978845608028654 / 10000000000000000 : ℝ) *
          (-x + (44715 / 1000000) * (-x * -x * -x))
        = -((7978845608028654 / 10000000000000000) *
              (x + (44715 / 1000000) * (x * x * x))) := by ring
  rw [h_arg, Real.tanh_neg]
  rw [show ((-x) / Real.sqrt 2 : ℝ) = -(x / Real.sqrt 2) from by ring]
  rw [VeriTile.Math.realErf_neg]
  ring

/-- Legacy closure for the very-large case (both signs): for `|x| ≥ 20` the gelu
approximation error is at most `1/1000`. -/
private theorem approx_gelu_error_bound_very_large {x : ℝ} (hx : 20 ≤ |x|) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ 1 / 1000 := by
  rcases le_or_gt 0 x with hxn | hxn
  · -- Positive case: |x| = x, reduce to _pos.
    rw [abs_of_nonneg hxn] at hx
    exact approx_gelu_error_bound_very_large_pos hx
  · -- Negative case: use evenness `f(−x) = f(x)` to flip.
    rw [abs_of_neg hxn] at hx
    have hxneg : 20 ≤ -x := hx
    have h_pos := approx_gelu_error_bound_very_large_pos hxneg
    -- |approx(-x) - exact(-x)| ≤ 1/1000, and `_neg` says it equals |approx x - exact x|.
    rwa [approxSubExact_neg] at h_pos

/-- Closure for the sharper tail case (both signs): for `|x| ≥ 4` the gelu
approximation error is at most `1/1000`. -/
private theorem approx_gelu_error_bound_tail_four {x : ℝ} (hx : 4 ≤ |x|) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ 1 / 1000 := by
  rcases le_or_gt 0 x with hxn | hxn
  · rw [abs_of_nonneg hxn] at hx
    exact approx_gelu_error_bound_tail_pos_four hx
  · rw [abs_of_neg hxn] at hx
    have hxneg : 4 ≤ -x := hx
    have h_pos := approx_gelu_error_bound_tail_pos_four hxneg
    rwa [approxSubExact_neg] at h_pos

/-- Closure for the sharper tail case (both signs): for `|x| ≥ 19/5` the gelu
approximation error is at most `1/1000`. -/
private theorem approx_gelu_error_bound_tail_19_5 {x : ℝ} (hx : (19/5 : ℝ) ≤ |x|) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ 1 / 1000 := by
  rcases le_or_gt 0 x with hxn | hxn
  · rw [abs_of_nonneg hxn] at hx
    exact approx_gelu_error_bound_tail_pos_19_5 hx
  · rw [abs_of_neg hxn] at hx
    have hxneg : (19/5 : ℝ) ≤ -x := hx
    have h_pos := approx_gelu_error_bound_tail_pos_19_5 hxneg
    rwa [approxSubExact_neg] at h_pos

/-! ### Interval-certificate skeleton for the remaining compact range

The remaining interval proof will be made of reusable certificates: on each
small interval, a bound at the midpoint plus a derivative bound gives the
whole-interval bound by the mean value theorem. -/

/-- Generic one-dimensional interval certificate.

If `f` has derivative `f'` on `[a,b]`, the midpoint value is bounded by `E`,
the derivative is bounded by `L`, and every point in the interval lies within
radius `r` of the midpoint `m`, then `|f x| ≤ eps` follows from the budget
`E + L*r ≤ eps`. -/
private lemma interval_abs_bound_of_center_deriv
    {f f' : ℝ → ℝ} {a b m x E L r eps : ℝ}
    (hm_mem : m ∈ Set.Icc a b) (hx_mem : x ∈ Set.Icc a b)
    (h_deriv : ∀ y ∈ Set.Icc a b, HasDerivAt f (f' y) y)
    (h_deriv_bound : ∀ y ∈ Set.Icc a b, |f' y| ≤ L)
    (h_center : |f m| ≤ E)
    (h_radius : |x - m| ≤ r)
    (h_budget : E + L * r ≤ eps) :
    |f x| ≤ eps := by
  have h_lip :
      ‖f x - f m‖ ≤ L * ‖x - m‖ :=
    (convex_Icc a b).norm_image_sub_le_of_norm_deriv_le
      (fun y hy => (h_deriv y hy).differentiableAt)
      (fun y hy => by
        have hdy : deriv f y = f' y := (h_deriv y hy).deriv
        rw [hdy]
        simpa [Real.norm_eq_abs] using h_deriv_bound y hy)
      hm_mem hx_mem
  have h_dist : ‖x - m‖ ≤ r := by
    simpa [Real.norm_eq_abs] using h_radius
  have h_delta : |f x - f m| ≤ L * r := by
    have hL_nonneg : 0 ≤ L := by
      have h := h_deriv_bound m hm_mem
      exact (abs_nonneg (f' m)).trans h
    calc |f x - f m|
        = ‖f x - f m‖ := by rw [Real.norm_eq_abs]
      _ ≤ L * ‖x - m‖ := h_lip
      _ ≤ L * r := mul_le_mul_of_nonneg_left h_dist hL_nonneg
  calc |f x|
      ≤ |f m| + |f x - f m| := by
          have h := abs_add_le (f m) (f x - f m)
          rw [show f m + (f x - f m) = f x by ring] at h
          exact h
    _ ≤ E + L * r := add_le_add h_center h_delta
    _ ≤ eps := h_budget

/-- Scalar error function used by the interval-certificate layer. -/
private noncomputable def geluError (x : ℝ) : ℝ :=
  approxGeLUScalar x - exactGeLUScalar x

private lemma geluError_neg (x : ℝ) : geluError (-x) = geluError x := by
  unfold geluError
  exact approxSubExact_neg x

private lemma geluError_abs_arg (x : ℝ) : geluError |x| = geluError x := by
  rcases le_or_gt 0 x with hx | hx
  · rw [abs_of_nonneg hx]
  · rw [abs_of_neg hx, geluError_neg]

/-- Explicit derivative of `geluError`. -/
private noncomputable def geluErrorDeriv (x : ℝ) : ℝ :=
  let c : ℝ := 7978845608028654 / 10000000000000000
  let k : ℝ := 44715 / 1000000
  let u := c * (x + k * (x * x * x))
  let z := x / Real.sqrt 2
  (1 / 2) * (Real.tanh u - realErf z) +
    (x / 2) *
      ((1 - Real.tanh u ^ 2) * (c * (1 + 3 * k * x ^ 2)) -
        (2 / Real.sqrt Real.pi) * gaussianKernel z / Real.sqrt 2)

/-- Derivative formula for the scalar GeLU approximation error. -/
private lemma geluError_hasDerivAt (x : ℝ) :
    HasDerivAt geluError (geluErrorDeriv x) x := by
  set c : ℝ := 7978845608028654 / 10000000000000000 with hc_def
  set k : ℝ := 44715 / 1000000 with hk_def
  set u : ℝ := c * (x + k * (x * x * x)) with hu_def
  set z : ℝ := x / Real.sqrt 2 with hz_def
  have hu_deriv : HasDerivAt (fun y : ℝ => c * (y + k * (y * y * y)))
      (c * (1 + 3 * k * x ^ 2)) x := by
    have h_id : HasDerivAt (fun y : ℝ => y) 1 x := hasDerivAt_id x
    have h_cube : HasDerivAt (fun y : ℝ => y * y * y) (3 * x ^ 2) x := by
      have hpow := hasDerivAt_pow 3 x
      convert hpow using 1
      ring_nf
    have h_inner := h_id.add (h_cube.const_mul k)
    have h_all := h_inner.const_mul c
    convert h_all using 1
    ring
  have hz_deriv : HasDerivAt (fun y : ℝ => y / Real.sqrt 2) (1 / Real.sqrt 2) x := by
    simpa using (hasDerivAt_id x).div_const (Real.sqrt 2)
  have h_tanh : HasDerivAt
      (fun y : ℝ => Real.tanh (c * (y + k * (y * y * y))))
      ((1 - Real.tanh u ^ 2) * (c * (1 + 3 * k * x ^ 2))) x := by
    have h := (tanh_hasDerivAt u).comp x hu_deriv
    exact h
  have h_erf : HasDerivAt (fun y : ℝ => realErf (y / Real.sqrt 2))
      ((2 / Real.sqrt Real.pi) * gaussianKernel z / Real.sqrt 2) x := by
    have h := (realErf_hasDerivAt z).comp x hz_deriv
    convert h using 1
    rw [hz_def, div_eq_mul_inv, one_div]
  have h_gap : HasDerivAt
      (fun y : ℝ => Real.tanh (c * (y + k * (y * y * y))) - realErf (y / Real.sqrt 2))
      ((1 - Real.tanh u ^ 2) * (c * (1 + 3 * k * x ^ 2)) -
        (2 / Real.sqrt Real.pi) * gaussianKernel z / Real.sqrt 2) x :=
    h_tanh.sub h_erf
  have h_xhalf : HasDerivAt (fun y : ℝ => y / 2) (1 / 2) x := by
    simpa using (hasDerivAt_id x).div_const (2 : ℝ)
  have h_prod := h_xhalf.mul h_gap
  have h_eq :
      geluError =
        (fun y : ℝ => (y / 2) *
          (Real.tanh (c * (y + k * (y * y * y))) - realErf (y / Real.sqrt 2))) := by
    funext y
    rw [geluError, approxGeLUScalar_sub_exactGeLUScalar]
  rw [h_eq]
  rw [geluErrorDeriv, hc_def, hk_def, ← hu_def, ← hz_def]
  exact h_prod

/-- GeLU-specialized interval certificate. This packages the generic mean-value
certificate with the explicit `geluError` derivative formula. -/
private lemma gelu_interval_abs_bound
    {a b m x E L r eps : ℝ}
    (hm_mem : m ∈ Set.Icc a b) (hx_mem : x ∈ Set.Icc a b)
    (h_deriv_bound : ∀ y ∈ Set.Icc a b, |geluErrorDeriv y| ≤ L)
    (h_center : |geluError m| ≤ E)
    (h_radius : |x - m| ≤ r)
    (h_budget : E + L * r ≤ eps) :
    |geluError x| ≤ eps :=
  interval_abs_bound_of_center_deriv hm_mem hx_mem
    (fun y _ => geluError_hasDerivAt y) h_deriv_bound h_center h_radius h_budget

/-! ## Structural Taylor decomposition of the gelu gap

For `|x| ≤ 1`, the gap `|tanh u - realErf z|` (with `u = c·(x + k·x³)` and
`z = x/√2`) splits via triangle inequality into three pieces:
* the tanh Taylor remainder `|u|^7/7`,
* the polynomial coefficient difference `|P_t(u) - (2/√π)·P_e(z)|`, and
* the realErf Taylor remainder `(2/√π) · 5|z|^9/864`.

The first and third pieces are already discharged by the wiring lemmas
`abs_tanh_at_gelu_arg_sub_taylor5_le` and `abs_realErf_at_gelu_arg_sub_taylor7_le`.
The middle piece is purely algebraic — it is a polynomial in `x` whose
coefficients involve `c`, `k`, and the irrational constant `c_exact = √(2/π)`. -/

/-! ### `|c − √(2/π)|` is tiny

The DSL's rational constant `c = 7978845608028654 / 10000000000000000` is
the 16-decimal rounding of `√(2/π)`. Using `Real.pi_gt_d20` and
`Real.pi_lt_d20` (20 decimals each), we show `c² ≥ 2/π` and
`|c − √(2/π)| ≤ 10⁻¹⁵`. The leading coefficient gap of the polynomial
difference `P_t(u) − (2/√π)·P_e(z)` is exactly this quantity. -/

private lemma c_pos_real : (0 : ℝ) < 7978845608028654 / 10000000000000000 := by
  norm_num

/-- The rational constant `c² ≥ 2/π`, certified by `Real.pi_gt_d20`. -/
private lemma c_sq_ge_two_div_pi :
    (2 : ℝ) / Real.pi
      ≤ (7978845608028654 / 10000000000000000 : ℝ)^2 := by
  have hpi_lb : (3.14159265358979323846 : ℝ) < Real.pi := Real.pi_gt_d20
  rw [div_le_iff₀ Real.pi_pos]
  calc (2 : ℝ)
      ≤ (7978845608028654 / 10000000000000000 : ℝ)^2 *
            3.14159265358979323846 := by norm_num
    _ ≤ (7978845608028654 / 10000000000000000 : ℝ)^2 * Real.pi := by
          have h := hpi_lb.le
          have h_sq_nn :
              (0 : ℝ) ≤ (7978845608028654 / 10000000000000000 : ℝ)^2 := sq_nonneg _
          exact mul_le_mul_of_nonneg_left h h_sq_nn

/-- `c ≥ √(2/π)`, the rational rounding sits above the irrational true value. -/
private lemma c_ge_sqrt_two_div_pi :
    Real.sqrt (2 / Real.pi)
      ≤ (7978845608028654 / 10000000000000000 : ℝ) := by
  have hsq := c_sq_ge_two_div_pi
  have h_c_nn : (0 : ℝ) ≤ (7978845608028654 / 10000000000000000 : ℝ) :=
    c_pos_real.le
  have h_2pi_nn : (0 : ℝ) ≤ 2 / Real.pi := div_nonneg (by norm_num) Real.pi_pos.le
  have h := Real.sqrt_le_sqrt hsq
  rwa [Real.sqrt_sq h_c_nn] at h

/-- Tight upper bound on `c² − 2/π`. Numerically the true value is `≈ 7·10⁻¹⁷`;
this proof uses the bound `≤ 10⁻¹⁶` certified by `Real.pi_lt_d20`. -/
private lemma c_sq_sub_two_div_pi_le :
    (7978845608028654 / 10000000000000000 : ℝ)^2 - 2 / Real.pi
      ≤ (1 : ℝ) / 10^16 := by
  have hpi_ub : Real.pi < 3.14159265358979323847 := Real.pi_lt_d20
  have h_two_div_le : (2 : ℝ) / 3.14159265358979323847 ≤ 2 / Real.pi := by
    apply div_le_div_of_nonneg_left (by norm_num) Real.pi_pos hpi_ub.le
  have h_step : (7978845608028654 / 10000000000000000 : ℝ)^2 -
        2 / 3.14159265358979323847 ≤ 1 / 10^16 := by norm_num
  linarith

/-- Tight upper bound: `c − √(2/π) ≤ 10⁻¹⁵`. The numerical truth is ~10⁻¹⁶,
so this leaves an order-of-magnitude margin. -/
private lemma c_sub_sqrt_two_div_pi_le :
    (7978845608028654 / 10000000000000000 : ℝ) - Real.sqrt (2 / Real.pi)
      ≤ (1 : ℝ) / 10^15 := by
  set c : ℝ := 7978845608028654 / 10000000000000000 with hc_def
  set s : ℝ := Real.sqrt (2 / Real.pi) with hs_def
  have hc_pos : 0 < c := c_pos_real
  have h_2pi_nn : (0 : ℝ) ≤ 2 / Real.pi := div_nonneg (by norm_num) Real.pi_pos.le
  have hs_nn : 0 ≤ s := Real.sqrt_nonneg _
  have hs2 : s^2 = 2 / Real.pi := Real.sq_sqrt h_2pi_nn
  have hs_le_c : s ≤ c := c_ge_sqrt_two_div_pi
  have hc_sub_s_nn : 0 ≤ c - s := sub_nonneg.mpr hs_le_c
  have h_factor : (c - s) * (c + s) = c^2 - 2 / Real.pi := by
    have h_diff : c^2 - s^2 = c^2 - 2 / Real.pi := by rw [hs2]
    linarith [sq_sub_sq c s]
  have hc_add_s_ge_c : c ≤ c + s := by linarith
  have hc_sub_s_le : c - s ≤ (c^2 - 2 / Real.pi) / c := by
    rw [le_div_iff₀ hc_pos]
    calc (c - s) * c
        ≤ (c - s) * (c + s) :=
            mul_le_mul_of_nonneg_left hc_add_s_ge_c hc_sub_s_nn
      _ = c^2 - 2 / Real.pi := h_factor
  have h_sq_le : c^2 - 2 / Real.pi ≤ 1 / 10^16 := c_sq_sub_two_div_pi_le
  have hc_ge_half : (1 : ℝ) / 2 ≤ c := by
    show (1 : ℝ) / 2 ≤ 7978845608028654 / 10000000000000000
    norm_num
  have h_div_le : (c^2 - 2 / Real.pi) / c ≤ 1 / 10^16 / c :=
    div_le_div_of_nonneg_right h_sq_le hc_pos.le
  have h_div_bound : (1 : ℝ) / 10^16 / c ≤ 1 / 10^16 / (1/2) := by
    apply div_le_div_of_nonneg_left (by norm_num) (by norm_num) hc_ge_half
  calc c - s
      ≤ (c^2 - 2 / Real.pi) / c := hc_sub_s_le
    _ ≤ 1 / 10^16 / c := h_div_le
    _ ≤ 1 / 10^16 / (1/2) := h_div_bound
    _ = 2 / 10^16 := by norm_num
    _ ≤ 1 / 10^15 := by norm_num

/-- Combined: `|c − √(2/π)| ≤ 10⁻¹⁵`. Since `c ≥ √(2/π)`, the absolute value
is just `c − √(2/π)`. -/
private lemma abs_c_sub_sqrt_two_div_pi_le :
    |(7978845608028654 / 10000000000000000 : ℝ) - Real.sqrt (2 / Real.pi)|
      ≤ (1 : ℝ) / 10^15 := by
  have h_le : Real.sqrt (2 / Real.pi)
        ≤ (7978845608028654 / 10000000000000000 : ℝ) := c_ge_sqrt_two_div_pi
  rw [abs_of_nonneg (by linarith)]
  exact c_sub_sqrt_two_div_pi_le

/-! ### Reduction of `(2/√π)·P_e(x/√2)` to a clean monic polynomial in `x`

Using `2/√π = √(2/π)·√2` we eliminate one `√2` from each rescaled `(x/√2)^n`,
leaving the explicit polynomial form `c_exact · (x − x³/6 + x⁵/40 − x⁷/336)`. -/

/-- Algebraic identity: `2/√π = √(2/π) · √2`. -/
private lemma two_div_sqrt_pi_eq :
    (2 : ℝ) / Real.sqrt Real.pi = Real.sqrt (2 / Real.pi) * Real.sqrt 2 := by
  have h_2_eq : (2 : ℝ) = Real.sqrt 4 := by
    rw [show (4:ℝ) = (2:ℝ)^2 from by norm_num, Real.sqrt_sq (by norm_num : (0:ℝ) ≤ 2)]
  conv_lhs => rw [h_2_eq, ← Real.sqrt_div (by norm_num : (0:ℝ) ≤ 4) Real.pi]
  rw [show (4 : ℝ) / Real.pi = (2 / Real.pi) * 2 from by ring]
  rw [Real.sqrt_mul (by positivity : (0 : ℝ) ≤ 2 / Real.pi)]

/-- Algebraic identity: `(2/√π)·P_e(x/√2) = √(2/π) · (x − x³/6 + x⁵/40 − x⁷/336)`.
The `1/√2` factors absorb into the irrational coefficient `2/√π`, leaving a
polynomial in `x` with the single irrational `√(2/π)` prefactor. -/
private lemma two_div_sqrt_pi_realErfTaylor7_eq (x : ℝ) :
    (2 / Real.sqrt Real.pi) *
        ((x / Real.sqrt 2) - (x / Real.sqrt 2)^3/3
          + (x / Real.sqrt 2)^5/10 - (x / Real.sqrt 2)^7/42)
      = Real.sqrt (2 / Real.pi) * (x - x^3/6 + x^5/40 - x^7/336) := by
  set s : ℝ := Real.sqrt 2 with hs_def
  have hs_pos : 0 < s := Real.sqrt_pos.mpr (by norm_num)
  have hs_ne : s ≠ 0 := hs_pos.ne'
  have hs2 : s^2 = 2 := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 2)
  rw [two_div_sqrt_pi_eq]
  -- Reduces to: √(2/π) · s · ((x/s) - (x/s)³/3 + (x/s)⁵/10 - (x/s)⁷/42)
  --        = √(2/π) · (x - x³/6 + x⁵/40 - x⁷/336)
  rw [mul_assoc]
  congr 1
  -- Inner identity, using s² = 2.
  have h_inner : s * ((x / s) - (x / s)^3/3 + (x / s)^5/10 - (x / s)^7/42)
              = x - x^3/6 + x^5/40 - x^7/336 := by
    field_simp
    linear_combination
      60480 * (((280 * x^3 - 42 * x^5 + 5 * x^7) * s^4
        + (-84 * x^5 + 10 * x^7) * s^2 + 20 * x^7) * hs2)
  exact h_inner

/-- The two tanh Taylor coefficients sit naturally as `u - u³/3 + 2u⁵/15`. -/
private noncomputable def tanhTaylor5 (u : ℝ) : ℝ := u - u^3/3 + 2 * u^5/15

/-- The four realErf Taylor coefficients sit naturally as the polynomial
`z - z³/3 + z⁵/10 - z⁷/42` (still to be multiplied by the `2/√π` prefactor). -/
private noncomputable def realErfTaylor7 (z : ℝ) : ℝ :=
  z - z^3/3 + z^5/10 - z^7/42

/-! ### Polynomial coefficient bound for the medium-low subcase `|x| ≤ 1/2`

Expanding `tanhTaylor5(c·(x + k·x³)) − c·(x − x³/6 + x⁵/40 − x⁷/336)` as a
polynomial in `x` (rational coefficients, since `c` and `k` are rational) and
applying triangle inequality gives a `≤ 1/5000` bound for `|x| ≤ 1/2`. The
remaining `(c − √(2/π))·(...)` correction is `O(10⁻¹⁵)` and folded in below. -/

/-- Polynomial expansion identity: the difference of `tanhTaylor5` at the gelu
argument and the rational version `c·(x − x³/6 + x⁵/40 − x⁷/336)` factors as
an odd polynomial in `x³…x¹⁵` whose coefficients are explicit rationals in
`c, k`. -/
private lemma tanhTaylor5_sub_rationalErf_eq (c k x : ℝ) :
    tanhTaylor5 (c * (x + k * (x*x*x))) -
        c * (x - x^3/6 + x^5/40 - x^7/336)
      = (c*k - c^3/3 + c/6) * x^3
          + (-c^3*k + 2*c^5/15 - c/40) * x^5
          + (-c^3*k^2 + 2*c^5*k/3 + c/336) * x^7
          + (-c^3*k^3/3 + 4*c^5*k^2/3) * x^9
          + (4*c^5*k^3/3) * x^11
          + (2*c^5*k^4/3) * x^13
          + (2*c^5*k^5/15) * x^15 := by
  unfold tanhTaylor5
  ring

/-- Polynomial difference bound for `|x| ≤ 1/2`: the rational-version gap
`|tanhTaylor5(c·(x + k·x³)) − c·(x − x³/6 + x⁵/40 − x⁷/336)| ≤ 1/4000`. The
true sup is ~1.84·10⁻⁴ so the constant is just over the truth. -/
private lemma poly_diff_bound {x : ℝ} (hx : |x| ≤ 1/2) :
    |tanhTaylor5 ((7978845608028654 / 10000000000000000 : ℝ) *
                    (x + (44715 / 1000000) * (x * x * x))) -
        (7978845608028654 / 10000000000000000 : ℝ) *
            (x - x^3/6 + x^5/40 - x^7/336)|
      ≤ 1 / 4000 := by
  set c : ℝ := 7978845608028654 / 10000000000000000 with hc_def
  set k : ℝ := 44715 / 1000000 with hk_def
  rw [tanhTaylor5_sub_rationalErf_eq c k x]
  -- Bound each `|coef|·|x|^n` ≤ |coef|·(1/2)^n.
  have hx_nn : 0 ≤ |x| := abs_nonneg _
  have hx3 : |x|^3 ≤ (1/2)^3 := pow_le_pow_left₀ hx_nn hx 3
  have hx5 : |x|^5 ≤ (1/2)^5 := pow_le_pow_left₀ hx_nn hx 5
  have hx7 : |x|^7 ≤ (1/2)^7 := pow_le_pow_left₀ hx_nn hx 7
  have hx9 : |x|^9 ≤ (1/2)^9 := pow_le_pow_left₀ hx_nn hx 9
  have hx11 : |x|^11 ≤ (1/2)^11 := pow_le_pow_left₀ hx_nn hx 11
  have hx13 : |x|^13 ≤ (1/2)^13 := pow_le_pow_left₀ hx_nn hx 13
  have hx15 : |x|^15 ≤ (1/2)^15 := pow_le_pow_left₀ hx_nn hx 15
  -- Shape: |∑ᵢ cᵢ·x^iⁿ| ≤ ∑ᵢ |cᵢ|·|x|^n ≤ ∑ᵢ |cᵢ|·(1/2)^n
  set a3 := c*k - c^3/3 + c/6 with ha3
  set a5 := -c^3*k + 2*c^5/15 - c/40 with ha5
  set a7 := -c^3*k^2 + 2*c^5*k/3 + c/336 with ha7
  set a9 := -c^3*k^3/3 + 4*c^5*k^2/3 with ha9
  set a11 := 4*c^5*k^3/3 with ha11
  set a13 := 2*c^5*k^4/3 with ha13
  set a15 := 2*c^5*k^5/15 with ha15
  -- Bound each term `|aᵢ · x^i|` by `|aᵢ| · (1/2)^i`.
  have hb3 : |a3 * x^3| ≤ |a3| * (1/2)^3 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx3 (abs_nonneg _)
  have hb5 : |a5 * x^5| ≤ |a5| * (1/2)^5 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx5 (abs_nonneg _)
  have hb7 : |a7 * x^7| ≤ |a7| * (1/2)^7 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx7 (abs_nonneg _)
  have hb9 : |a9 * x^9| ≤ |a9| * (1/2)^9 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx9 (abs_nonneg _)
  have hb11 : |a11 * x^11| ≤ |a11| * (1/2)^11 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx11 (abs_nonneg _)
  have hb13 : |a13 * x^13| ≤ |a13| * (1/2)^13 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx13 (abs_nonneg _)
  have hb15 : |a15 * x^15| ≤ |a15| * (1/2)^15 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx15 (abs_nonneg _)
  -- Numeric bounds on each `|aᵢ| · (1/2)^i`.
  have hn3 : |a3| * (1/2)^3 ≤ 9 / 100000 := by
    rw [ha3, hc_def, hk_def]
    rw [abs_of_neg (by norm_num)]; norm_num
  have hn5 : |a5| * (1/2)^5 ≤ 2 / 100000 := by
    rw [ha5, hc_def, hk_def]
    rw [abs_of_pos (by norm_num)]; norm_num
  have hn7 : |a7| * (1/2)^7 ≤ 9 / 100000 := by
    rw [ha7, hc_def, hk_def]
    rw [abs_of_pos (by norm_num)]; norm_num
  have hn9 : |a9| * (1/2)^9 ≤ 1 / 500000 := by
    rw [ha9, hc_def, hk_def]
    rw [abs_of_pos (by norm_num)]; norm_num
  have hn11 : |a11| * (1/2)^11 ≤ 1 / 50000000 := by
    rw [ha11, hc_def, hk_def]
    rw [abs_of_pos (by norm_num)]; norm_num
  have hn13 : |a13| * (1/2)^13 ≤ 1 / 5000000000 := by
    rw [ha13, hc_def, hk_def]
    rw [abs_of_pos (by norm_num)]; norm_num
  have hn15 : |a15| * (1/2)^15 ≤ 1 / 1000000000000 := by
    rw [ha15, hc_def, hk_def]
    rw [abs_of_pos (by norm_num)]; norm_num
  -- Sum the bounds.
  calc |a3 * x^3 + a5 * x^5 + a7 * x^7 + a9 * x^9 + a11 * x^11 + a13 * x^13 + a15 * x^15|
      ≤ |a3 * x^3| + |a5 * x^5| + |a7 * x^7| + |a9 * x^9| + |a11 * x^11|
          + |a13 * x^13| + |a15 * x^15| := by
        have h1 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11 + a13*x^13)
                              (a15*x^15)
        have h2 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11) (a13*x^13)
        have h3 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9) (a11*x^11)
        have h4 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7) (a9*x^9)
        have h5 := abs_add_le (a3*x^3 + a5*x^5) (a7*x^7)
        have h6 := abs_add_le (a3*x^3) (a5*x^5)
        linarith
    _ ≤ |a3| * (1/2)^3 + |a5| * (1/2)^5 + |a7| * (1/2)^7 + |a9| * (1/2)^9
          + |a11| * (1/2)^11 + |a13| * (1/2)^13 + |a15| * (1/2)^15 := by
        linarith [hb3, hb5, hb7, hb9, hb11, hb13, hb15]
    _ ≤ 9/100000 + 2/100000 + 9/100000 + 1/500000 + 1/50000000
          + 1/5000000000 + 1/1000000000000 := by
        linarith [hn3, hn5, hn7, hn9, hn11, hn13, hn15]
    _ ≤ 1/4000 := by norm_num

/-- Three-piece triangle decomposition of the inner gelu gap. For `|x| ≤ 1`,
the difference `|tanh u - realErf z|` is dominated by the sum of the two
Taylor residual bounds plus the explicit polynomial coefficient difference
`|tanhTaylor5(u) - (2/√π) · realErfTaylor7(z)|`. -/
lemma gelu_gap_taylor_decomposition {x : ℝ} (hx : |x| ≤ 1) :
    |Real.tanh ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))) -
        realErf (x / Real.sqrt 2)|
      ≤ |(7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))| ^ 7 / 7
        + |tanhTaylor5
              ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x)))
            - (2 / Real.sqrt Real.pi) *
                realErfTaylor7 (x / Real.sqrt 2)|
        + (2 / Real.sqrt Real.pi) * (5 * |x / Real.sqrt 2| ^ 9 / 864) := by
  set u : ℝ := (7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x)) with hu_def
  set z : ℝ := x / Real.sqrt 2 with hz_def
  have h_tanh_res : |Real.tanh u - tanhTaylor5 u| ≤ |u| ^ 7 / 7 := by
    have h := abs_tanh_at_gelu_arg_sub_taylor5_le hx
    show |Real.tanh u - (u - u^3/3 + 2 * u^5/15)| ≤ |u| ^ 7 / 7
    convert h using 2
  have h_erf_res :
      |realErf z - (2 / Real.sqrt Real.pi) * realErfTaylor7 z|
        ≤ (2 / Real.sqrt Real.pi) * (5 * |z| ^ 9 / 864) := by
    have h := abs_realErf_at_gelu_arg_sub_taylor7_le hx
    show |realErf z - (2 / Real.sqrt Real.pi) *
            (z - z^3/3 + z^5/10 - z^7/42)|
          ≤ (2 / Real.sqrt Real.pi) * (5 * |z| ^ 9 / 864)
    convert h using 2
  -- Triangle:  |tanh u − realErf z|
  --         = |(tanh u − P_t(u)) + (P_t(u) − (2/√π)·P_e(z))
  --                              + ((2/√π)·P_e(z) − realErf z)|
  --        ≤ |tanh u − P_t(u)| + |P_t(u) − (2/√π)·P_e(z)|
  --                            + |realErf z − (2/√π)·P_e(z)|
  have h_split :
      Real.tanh u - realErf z
        = (Real.tanh u - tanhTaylor5 u)
            + (tanhTaylor5 u - (2 / Real.sqrt Real.pi) * realErfTaylor7 z)
            - (realErf z - (2 / Real.sqrt Real.pi) * realErfTaylor7 z) := by
    ring
  rw [h_split]
  calc |(Real.tanh u - tanhTaylor5 u)
          + (tanhTaylor5 u - (2 / Real.sqrt Real.pi) * realErfTaylor7 z)
          - (realErf z - (2 / Real.sqrt Real.pi) * realErfTaylor7 z)|
      ≤ |(Real.tanh u - tanhTaylor5 u)
            + (tanhTaylor5 u - (2 / Real.sqrt Real.pi) * realErfTaylor7 z)|
        + |realErf z - (2 / Real.sqrt Real.pi) * realErfTaylor7 z| :=
            abs_sub _ _
    _ ≤ (|Real.tanh u - tanhTaylor5 u|
            + |tanhTaylor5 u - (2 / Real.sqrt Real.pi) * realErfTaylor7 z|)
        + |realErf z - (2 / Real.sqrt Real.pi) * realErfTaylor7 z| := by
            gcongr
            exact abs_add_le _ _
    _ ≤ (|u| ^ 7 / 7
            + |tanhTaylor5 u - (2 / Real.sqrt Real.pi) * realErfTaylor7 z|)
        + (2 / Real.sqrt Real.pi) * (5 * |z| ^ 9 / 864) := by
            gcongr

/-! ### Combined polynomial bound including the `c − √(2/π)` correction

For `|x| ≤ 1/2`, the polynomial gap `|tanhTaylor5(c·v(x)) − (2/√π)·realErfTaylor7(x/√2)|`
absorbs the `(c − √(2/π))·R(x)` correction (which is `O(10⁻¹⁵)`) into a clean
`≤ 1/3000` bound. -/

/-- Bound on `|R(x)| = |x − x³/6 + x⁵/40 − x⁷/336|` for `|x| ≤ 1`. The exact sup
is ≈1.195; the looser bound `≤ 2` simplifies downstream arithmetic. -/
private lemma R_bound_at_one {x : ℝ} (hx : |x| ≤ 1) :
    |x - x^3/6 + x^5/40 - x^7/336| ≤ 2 := by
  have hx3 : |x|^3 ≤ 1 := by
    have := pow_le_pow_left₀ (abs_nonneg x) hx 3; simpa using this
  have hx5 : |x|^5 ≤ 1 := by
    have := pow_le_pow_left₀ (abs_nonneg x) hx 5; simpa using this
  have hx7 : |x|^7 ≤ 1 := by
    have := pow_le_pow_left₀ (abs_nonneg x) hx 7; simpa using this
  have h_eq : x - x^3/6 + x^5/40 - x^7/336
            = x + (-(x^3/6)) + x^5/40 + (-(x^7/336)) := by ring
  rw [h_eq]
  have h1 := abs_add_le (x + (-(x^3/6)) + x^5/40) (-(x^7/336))
  have h2 := abs_add_le (x + (-(x^3/6))) (x^5/40)
  have h3 := abs_add_le x (-(x^3/6))
  have hxabs : |x| ≤ 1 := hx
  have habs3 : |(-(x^3/6))| = |x|^3/6 := by
    rw [abs_neg, abs_div, abs_pow,
        show |(6:ℝ)| = 6 from by norm_num]
  have habs5 : |x^5/40| = |x|^5/40 := by
    rw [abs_div, abs_pow, show |(40:ℝ)| = 40 from by norm_num]
  have habs7 : |(-(x^7/336))| = |x|^7/336 := by
    rw [abs_neg, abs_div, abs_pow, show |(336:ℝ)| = 336 from by norm_num]
  rw [habs3] at h3
  rw [habs5] at h2
  rw [habs7] at h1
  have hb3 : |x|^3/6 ≤ 1/6 := by linarith
  have hb5 : |x|^5/40 ≤ 1/40 := by linarith
  have hb7 : |x|^7/336 ≤ 1/336 := by linarith
  linarith

/-- Combined polynomial bound including the `c − √(2/π)` correction.
For `|x| ≤ 1/2`, `|tanhTaylor5(c·(x+k·x³)) − (2/√π)·realErfTaylor7(x/√2)| ≤ 1/3000`. -/
private lemma poly_diff_with_correction_bound {x : ℝ} (hx : |x| ≤ 1/2) :
    |tanhTaylor5 ((7978845608028654 / 10000000000000000 : ℝ) *
                    (x + (44715 / 1000000) * (x * x * x))) -
        (2 / Real.sqrt Real.pi) *
            realErfTaylor7 (x / Real.sqrt 2)|
      ≤ 1 / 3000 := by
  -- Reshape the second term to √(2/π)·R(x) via the algebraic identity.
  have h_rewrite : (2 / Real.sqrt Real.pi) *
                       realErfTaylor7 (x / Real.sqrt 2)
                = Real.sqrt (2 / Real.pi) * (x - x^3/6 + x^5/40 - x^7/336) := by
    show (2 / Real.sqrt Real.pi) *
            ((x / Real.sqrt 2) - (x / Real.sqrt 2)^3/3
              + (x / Real.sqrt 2)^5/10 - (x / Real.sqrt 2)^7/42)
          = Real.sqrt (2 / Real.pi) * (x - x^3/6 + x^5/40 - x^7/336)
    exact two_div_sqrt_pi_realErfTaylor7_eq x
  rw [h_rewrite]
  -- Add ± c·R(x) to split.
  have h_split :
      tanhTaylor5 ((7978845608028654 / 10000000000000000 : ℝ) *
                      (x + (44715 / 1000000) * (x * x * x))) -
          Real.sqrt (2 / Real.pi) * (x - x^3/6 + x^5/40 - x^7/336)
        = (tanhTaylor5 ((7978845608028654 / 10000000000000000 : ℝ) *
                          (x + (44715 / 1000000) * (x * x * x))) -
            (7978845608028654 / 10000000000000000 : ℝ) *
                (x - x^3/6 + x^5/40 - x^7/336))
          + ((7978845608028654 / 10000000000000000 : ℝ) -
              Real.sqrt (2 / Real.pi)) *
              (x - x^3/6 + x^5/40 - x^7/336) := by
    ring
  rw [h_split]
  -- Triangle.
  have hpoly := poly_diff_bound hx
  have hcorr_sqrt := abs_c_sub_sqrt_two_div_pi_le
  have hx_le_one : |x| ≤ 1 := by linarith
  have hR := R_bound_at_one hx_le_one
  calc |(tanhTaylor5 _ - _ * (x - x^3/6 + x^5/40 - x^7/336))
            + ((7978845608028654 / 10000000000000000 : ℝ)
                - Real.sqrt (2 / Real.pi)) *
                (x - x^3/6 + x^5/40 - x^7/336)|
      ≤ |tanhTaylor5 _ - _ * (x - x^3/6 + x^5/40 - x^7/336)|
        + |((7978845608028654 / 10000000000000000 : ℝ)
              - Real.sqrt (2 / Real.pi)) *
                (x - x^3/6 + x^5/40 - x^7/336)| := abs_add_le _ _
    _ ≤ 1/4000 + (1/10^15) * 2 := by
        gcongr
        rw [abs_mul]
        gcongr
    _ ≤ 1/3000 := by norm_num

/-! ### Closure for `|x| ≤ 1/2` via the combined Taylor bound

For `|x| ≤ 1/2`, all three pieces of `gelu_gap_taylor_decomposition` are
explicitly bounded:
* `|u|^7/7 ≤ 1/3500`  (since `|u| ≤ 41/100`),
* the polynomial coefficient gap `≤ 1/3000`,
* the realErf Taylor remainder `≤ 1/60000`.

Then `|approx − exact| = (|x|/2)·|tanh u − realErf z| ≤ (1/4)·(1/3500 + 1/3000 + 1/60000) < 1/1000`. -/

/-- For `|x| ≤ 1/2`, the inner argument `u = c·(x + k·x³)` of the tanh
approximation satisfies `|u| ≤ 41/100`. -/
private lemma u_bound_at_half {x : ℝ} (hx : |x| ≤ 1/2) :
    |(7978845608028654 / 10000000000000000 : ℝ) *
        (x + (44715 / 1000000) * (x * x * x))| ≤ 41 / 100 := by
  rw [abs_mul]
  have hc_pos : (0 : ℝ) < 7978845608028654 / 10000000000000000 := by norm_num
  rw [abs_of_pos hc_pos]
  have h_x3_le : |x|^3 ≤ (1/2)^3 := pow_le_pow_left₀ (abs_nonneg x) hx 3
  have h_inner : |x + (44715 / 1000000 : ℝ) * (x * x * x)|
        ≤ |x| + (44715 / 1000000) * |x|^3 := by
    have hb := abs_add_le x ((44715 / 1000000 : ℝ) * (x * x * x))
    have h_eq : |(44715 / 1000000 : ℝ) * (x * x * x)|
          = (44715 / 1000000) * |x|^3 := by
      rw [abs_mul, show |(44715 / 1000000 : ℝ)| = 44715 / 1000000 from by norm_num]
      congr 1
      rw [show x * x * x = x^3 from by ring, abs_pow]
    linarith
  calc (7978845608028654 / 10000000000000000 : ℝ) *
            |x + (44715 / 1000000) * (x * x * x)|
      ≤ (7978845608028654 / 10000000000000000) *
          (|x| + (44715 / 1000000) * |x|^3) := by
            apply mul_le_mul_of_nonneg_left h_inner hc_pos.le
    _ ≤ (7978845608028654 / 10000000000000000) *
          ((1/2) + (44715 / 1000000) * (1/2)^3) := by
            apply mul_le_mul_of_nonneg_left _ hc_pos.le
            have hk_pos : (0 : ℝ) ≤ 44715 / 1000000 := by norm_num
            have h := mul_le_mul_of_nonneg_left h_x3_le hk_pos
            linarith
    _ ≤ 41/100 := by norm_num

/-- For `|x| ≤ 1/2`, the tanh Taylor remainder satisfies `|u|^7/7 ≤ 1/3500`. -/
private lemma tanh_residual_at_half {x : ℝ} (hx : |x| ≤ 1/2) :
    |(7978845608028654 / 10000000000000000 : ℝ) *
        (x + (44715 / 1000000) * (x * x * x))|^7 / 7
      ≤ 1 / 3500 := by
  have hu := u_bound_at_half hx
  have hu_nn : (0 : ℝ) ≤ |(7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))| := abs_nonneg _
  have hu7 : |(7978845608028654 / 10000000000000000 : ℝ) *
                (x + (44715 / 1000000) * (x * x * x))|^7
              ≤ (41/100)^7 :=
    pow_le_pow_left₀ hu_nn hu 7
  calc |(7978845608028654 / 10000000000000000 : ℝ) *
            (x + (44715 / 1000000) * (x * x * x))|^7 / 7
      ≤ (41/100)^7 / 7 := by
        exact div_le_div_of_nonneg_right hu7 (by norm_num)
    _ ≤ 1/3500 := by norm_num

/-- For `|x| ≤ 1/2`, `|x/√2| ≤ 1/2`. -/
private lemma z_bound_at_half {x : ℝ} (hx : |x| ≤ 1/2) :
    |x / Real.sqrt 2| ≤ 1/2 := by
  have hsqrt2_pos : (0 : ℝ) < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  have hsqrt2_ge_1 : (1 : ℝ) ≤ Real.sqrt 2 := by
    rw [show (1:ℝ) = Real.sqrt 1 from Real.sqrt_one.symm]
    exact Real.sqrt_le_sqrt (by norm_num)
  rw [abs_div, abs_of_pos hsqrt2_pos]
  rw [div_le_iff₀ hsqrt2_pos]
  nlinarith [hx, hsqrt2_ge_1]

/-- For `|x| ≤ 1/2`, the realErf Taylor remainder satisfies
`(2/√π)·5·|z|^9/864 ≤ 1/60000`. -/
private lemma erf_residual_at_half {x : ℝ} (hx : |x| ≤ 1/2) :
    (2 / Real.sqrt Real.pi) * (5 * |x / Real.sqrt 2|^9 / 864) ≤ 1 / 60000 := by
  have hpi_pos : (0 : ℝ) < Real.pi := Real.pi_pos
  have hsqrt_pi_pos : (0 : ℝ) < Real.sqrt Real.pi := Real.sqrt_pos.mpr hpi_pos
  -- 2/√π ≤ 4/3 since √π ≥ 3/2, i.e., π ≥ 9/4.
  have hpi_ge : (3/2 : ℝ) ≤ Real.sqrt Real.pi := by
    rw [show (3/2:ℝ) = Real.sqrt ((3/2)^2) from (Real.sqrt_sq (by norm_num)).symm]
    apply Real.sqrt_le_sqrt
    have : ((3/2:ℝ))^2 = 9/4 := by norm_num
    rw [this]; linarith [Real.pi_gt_three]
  have h_inv_pi : (2 : ℝ) / Real.sqrt Real.pi ≤ 4/3 := by
    rw [div_le_div_iff₀ hsqrt_pi_pos (by norm_num : (0:ℝ) < 3)]
    nlinarith [hpi_ge]
  have hz := z_bound_at_half hx
  have hz_nn : (0 : ℝ) ≤ |x / Real.sqrt 2| := abs_nonneg _
  have hz9 : |x / Real.sqrt 2|^9 ≤ (1/2)^9 := pow_le_pow_left₀ hz_nn hz 9
  calc (2 / Real.sqrt Real.pi) * (5 * |x / Real.sqrt 2|^9 / 864)
      ≤ (4/3 : ℝ) * (5 * (1/2)^9 / 864) := by
        apply mul_le_mul h_inv_pi
        · apply div_le_div_of_nonneg_right
          · exact mul_le_mul_of_nonneg_left hz9 (by norm_num)
          · norm_num
        · positivity
        · norm_num
    _ ≤ 1/60000 := by norm_num

/-- Closure for `|x| ≤ 1/2`: gelu approximation error is at most `1/1000`. -/
private theorem approx_gelu_error_bound_small {x : ℝ} (hx : |x| ≤ 1/2) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ 1 / 1000 := by
  rw [approxGeLUScalar_sub_exactGeLUScalar x]
  have hx_le_one : |x| ≤ 1 := by linarith
  have h_taylor := gelu_gap_taylor_decomposition hx_le_one
  have h_tanh := tanh_residual_at_half hx
  have h_poly := poly_diff_with_correction_bound hx
  have h_erf := erf_residual_at_half hx
  have h_gap_bound :
      |Real.tanh ((7978845608028654 / 10000000000000000 : ℝ) *
                    (x + (44715 / 1000000) * (x * x * x))) -
          realErf (x / Real.sqrt 2)|
        ≤ 1/3500 + 1/3000 + 1/60000 := by
    calc |Real.tanh _ - realErf _|
        ≤ |(7978845608028654 / 10000000000000000 : ℝ) *
                (x + (44715 / 1000000) * (x * x * x))|^7 / 7
          + |tanhTaylor5 _ - (2 / Real.sqrt Real.pi) * realErfTaylor7 _|
          + (2 / Real.sqrt Real.pi) * (5 * |x / Real.sqrt 2|^9 / 864) :=
            h_taylor
      _ ≤ 1/3500 + 1/3000 + 1/60000 := by linarith
  -- Now: |approx − exact| = |x/2| · |tanh u − realErf z| ≤ (1/4) · (1/3500+1/3000+1/60000) < 1/1000.
  rw [abs_mul, abs_div, abs_two]
  have hx2_le : |x| / 2 ≤ 1/4 := by linarith
  have hx2_nn : 0 ≤ |x| / 2 := by positivity
  have h_combined : 1/3500 + 1/3000 + (1/60000 : ℝ) ≤ 1/700 := by norm_num
  calc |x| / 2 *
          |Real.tanh _ - realErf _|
      ≤ |x| / 2 * (1/3500 + 1/3000 + 1/60000) :=
          mul_le_mul_of_nonneg_left h_gap_bound hx2_nn
    _ ≤ |x| / 2 * (1/700) := by
          exact mul_le_mul_of_nonneg_left h_combined hx2_nn
    _ ≤ 1/4 * (1/700) := by
          exact mul_le_mul_of_nonneg_right hx2_le (by norm_num)
    _ ≤ 1/1000 := by norm_num

/-! ### Closure for `|x| ≤ 3/5` (extended)

The same Taylor decomposition closes a wider window once the per-coefficient
bounds are re-tuned for `(3/5)^n` instead of `(1/2)^n`. The bottleneck shifts
from the polynomial gap to the tanh remainder `|u|⁷/7`, but the budget still
clears `1/1000`. -/

/-- Polynomial difference bound for `|x| ≤ 3/5`. Sum of seven per-coefficient
bounds is `≈ 5.09·10⁻⁴`; the lemma asserts `≤ 1/1900`. -/
private lemma poly_diff_bound_06 {x : ℝ} (hx : |x| ≤ 3/5) :
    |tanhTaylor5 ((7978845608028654 / 10000000000000000 : ℝ) *
                    (x + (44715 / 1000000) * (x * x * x))) -
        (7978845608028654 / 10000000000000000 : ℝ) *
            (x - x^3/6 + x^5/40 - x^7/336)|
      ≤ 1 / 1900 := by
  set c : ℝ := 7978845608028654 / 10000000000000000 with hc_def
  set k : ℝ := 44715 / 1000000 with hk_def
  rw [tanhTaylor5_sub_rationalErf_eq c k x]
  have hx_nn : 0 ≤ |x| := abs_nonneg _
  have hx3 : |x|^3 ≤ (3/5)^3 := pow_le_pow_left₀ hx_nn hx 3
  have hx5 : |x|^5 ≤ (3/5)^5 := pow_le_pow_left₀ hx_nn hx 5
  have hx7 : |x|^7 ≤ (3/5)^7 := pow_le_pow_left₀ hx_nn hx 7
  have hx9 : |x|^9 ≤ (3/5)^9 := pow_le_pow_left₀ hx_nn hx 9
  have hx11 : |x|^11 ≤ (3/5)^11 := pow_le_pow_left₀ hx_nn hx 11
  have hx13 : |x|^13 ≤ (3/5)^13 := pow_le_pow_left₀ hx_nn hx 13
  have hx15 : |x|^15 ≤ (3/5)^15 := pow_le_pow_left₀ hx_nn hx 15
  set a3 := c*k - c^3/3 + c/6 with ha3
  set a5 := -c^3*k + 2*c^5/15 - c/40 with ha5
  set a7 := -c^3*k^2 + 2*c^5*k/3 + c/336 with ha7
  set a9 := -c^3*k^3/3 + 4*c^5*k^2/3 with ha9
  set a11 := 4*c^5*k^3/3 with ha11
  set a13 := 2*c^5*k^4/3 with ha13
  set a15 := 2*c^5*k^5/15 with ha15
  have hb3 : |a3 * x^3| ≤ |a3| * (3/5)^3 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx3 (abs_nonneg _)
  have hb5 : |a5 * x^5| ≤ |a5| * (3/5)^5 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx5 (abs_nonneg _)
  have hb7 : |a7 * x^7| ≤ |a7| * (3/5)^7 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx7 (abs_nonneg _)
  have hb9 : |a9 * x^9| ≤ |a9| * (3/5)^9 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx9 (abs_nonneg _)
  have hb11 : |a11 * x^11| ≤ |a11| * (3/5)^11 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx11 (abs_nonneg _)
  have hb13 : |a13 * x^13| ≤ |a13| * (3/5)^13 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx13 (abs_nonneg _)
  have hb15 : |a15 * x^15| ≤ |a15| * (3/5)^15 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx15 (abs_nonneg _)
  have hn3 : |a3| * (3/5)^3 ≤ 15 / 100000 := by
    rw [ha3, hc_def, hk_def]; rw [abs_of_neg (by norm_num)]; norm_num
  have hn5 : |a5| * (3/5)^5 ≤ 4 / 100000 := by
    rw [ha5, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn7 : |a7| * (3/5)^7 ≤ 31 / 100000 := by
    rw [ha7, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn9 : |a9| * (3/5)^9 ≤ 9 / 1000000 := by
    rw [ha9, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn11 : |a11| * (3/5)^11 ≤ 2 / 10000000 := by
    rw [ha11, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn13 : |a13| * (3/5)^13 ≤ 2 / 1000000000 := by
    rw [ha13, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn15 : |a15| * (3/5)^15 ≤ 4 / 1000000000000 := by
    rw [ha15, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  calc |a3 * x^3 + a5 * x^5 + a7 * x^7 + a9 * x^9 + a11 * x^11 + a13 * x^13 + a15 * x^15|
      ≤ |a3 * x^3| + |a5 * x^5| + |a7 * x^7| + |a9 * x^9| + |a11 * x^11|
          + |a13 * x^13| + |a15 * x^15| := by
        have h1 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11 + a13*x^13)
                              (a15*x^15)
        have h2 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11) (a13*x^13)
        have h3 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9) (a11*x^11)
        have h4 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7) (a9*x^9)
        have h5 := abs_add_le (a3*x^3 + a5*x^5) (a7*x^7)
        have h6 := abs_add_le (a3*x^3) (a5*x^5)
        linarith
    _ ≤ |a3| * (3/5)^3 + |a5| * (3/5)^5 + |a7| * (3/5)^7 + |a9| * (3/5)^9
          + |a11| * (3/5)^11 + |a13| * (3/5)^13 + |a15| * (3/5)^15 := by
        linarith [hb3, hb5, hb7, hb9, hb11, hb13, hb15]
    _ ≤ 15/100000 + 4/100000 + 31/100000 + 9/1000000 + 2/10000000
          + 2/1000000000 + 4/1000000000000 := by
        linarith [hn3, hn5, hn7, hn9, hn11, hn13, hn15]
    _ ≤ 1/1900 := by norm_num

/-- Combined polynomial bound including the `c − √(2/π)` correction for `|x| ≤ 3/5`. -/
private lemma poly_diff_with_correction_bound_06 {x : ℝ} (hx : |x| ≤ 3/5) :
    |tanhTaylor5 ((7978845608028654 / 10000000000000000 : ℝ) *
                    (x + (44715 / 1000000) * (x * x * x))) -
        (2 / Real.sqrt Real.pi) *
            realErfTaylor7 (x / Real.sqrt 2)|
      ≤ 1 / 1500 := by
  have h_rewrite : (2 / Real.sqrt Real.pi) *
                       realErfTaylor7 (x / Real.sqrt 2)
                = Real.sqrt (2 / Real.pi) * (x - x^3/6 + x^5/40 - x^7/336) := by
    show (2 / Real.sqrt Real.pi) *
            ((x / Real.sqrt 2) - (x / Real.sqrt 2)^3/3
              + (x / Real.sqrt 2)^5/10 - (x / Real.sqrt 2)^7/42)
          = Real.sqrt (2 / Real.pi) * (x - x^3/6 + x^5/40 - x^7/336)
    exact two_div_sqrt_pi_realErfTaylor7_eq x
  rw [h_rewrite]
  have h_split :
      tanhTaylor5 ((7978845608028654 / 10000000000000000 : ℝ) *
                      (x + (44715 / 1000000) * (x * x * x))) -
          Real.sqrt (2 / Real.pi) * (x - x^3/6 + x^5/40 - x^7/336)
        = (tanhTaylor5 ((7978845608028654 / 10000000000000000 : ℝ) *
                          (x + (44715 / 1000000) * (x * x * x))) -
            (7978845608028654 / 10000000000000000 : ℝ) *
                (x - x^3/6 + x^5/40 - x^7/336))
          + ((7978845608028654 / 10000000000000000 : ℝ) -
              Real.sqrt (2 / Real.pi)) *
              (x - x^3/6 + x^5/40 - x^7/336) := by
    ring
  rw [h_split]
  have hpoly := poly_diff_bound_06 hx
  have hcorr_sqrt := abs_c_sub_sqrt_two_div_pi_le
  have hx_le_one : |x| ≤ 1 := by linarith
  have hR := R_bound_at_one hx_le_one
  calc |(tanhTaylor5 _ - _ * (x - x^3/6 + x^5/40 - x^7/336))
            + ((7978845608028654 / 10000000000000000 : ℝ)
                - Real.sqrt (2 / Real.pi)) *
                (x - x^3/6 + x^5/40 - x^7/336)|
      ≤ |tanhTaylor5 _ - _ * (x - x^3/6 + x^5/40 - x^7/336)|
        + |((7978845608028654 / 10000000000000000 : ℝ)
              - Real.sqrt (2 / Real.pi)) *
                (x - x^3/6 + x^5/40 - x^7/336)| := abs_add_le _ _
    _ ≤ 1/1900 + (1/10^15) * 2 := by
        gcongr
        rw [abs_mul]
        gcongr
    _ ≤ 1/1500 := by norm_num

/-- For `|x| ≤ 3/5`, `|u| ≤ 49/100`. -/
private lemma u_bound_at_06 {x : ℝ} (hx : |x| ≤ 3/5) :
    |(7978845608028654 / 10000000000000000 : ℝ) *
        (x + (44715 / 1000000) * (x * x * x))| ≤ 49 / 100 := by
  rw [abs_mul]
  have hc_pos : (0 : ℝ) < 7978845608028654 / 10000000000000000 := by norm_num
  rw [abs_of_pos hc_pos]
  have h_x3_le : |x|^3 ≤ (3/5)^3 := pow_le_pow_left₀ (abs_nonneg x) hx 3
  have h_inner : |x + (44715 / 1000000 : ℝ) * (x * x * x)|
        ≤ |x| + (44715 / 1000000) * |x|^3 := by
    have hb := abs_add_le x ((44715 / 1000000 : ℝ) * (x * x * x))
    have h_eq : |(44715 / 1000000 : ℝ) * (x * x * x)|
          = (44715 / 1000000) * |x|^3 := by
      rw [abs_mul, show |(44715 / 1000000 : ℝ)| = 44715 / 1000000 from by norm_num]
      congr 1
      rw [show x * x * x = x^3 from by ring, abs_pow]
    linarith
  calc (7978845608028654 / 10000000000000000 : ℝ) *
            |x + (44715 / 1000000) * (x * x * x)|
      ≤ (7978845608028654 / 10000000000000000) *
          (|x| + (44715 / 1000000) * |x|^3) := by
            apply mul_le_mul_of_nonneg_left h_inner hc_pos.le
    _ ≤ (7978845608028654 / 10000000000000000) *
          ((3/5) + (44715 / 1000000) * (3/5)^3) := by
            apply mul_le_mul_of_nonneg_left _ hc_pos.le
            have hk_pos : (0 : ℝ) ≤ 44715 / 1000000 := by norm_num
            have h := mul_le_mul_of_nonneg_left h_x3_le hk_pos
            linarith
    _ ≤ 49/100 := by norm_num

/-- For `|x| ≤ 3/5`, `|u|⁷/7 ≤ 1/1000`. -/
private lemma tanh_residual_at_06 {x : ℝ} (hx : |x| ≤ 3/5) :
    |(7978845608028654 / 10000000000000000 : ℝ) *
        (x + (44715 / 1000000) * (x * x * x))|^7 / 7
      ≤ 1 / 1000 := by
  have hu := u_bound_at_06 hx
  have hu_nn : (0 : ℝ) ≤ |(7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))| := abs_nonneg _
  have hu7 : |(7978845608028654 / 10000000000000000 : ℝ) *
                (x + (44715 / 1000000) * (x * x * x))|^7
              ≤ (49/100)^7 :=
    pow_le_pow_left₀ hu_nn hu 7
  calc |(7978845608028654 / 10000000000000000 : ℝ) *
            (x + (44715 / 1000000) * (x * x * x))|^7 / 7
      ≤ (49/100)^7 / 7 :=
        div_le_div_of_nonneg_right hu7 (by norm_num)
    _ ≤ 1/1000 := by norm_num

/-- For `|x| ≤ 3/5`, `|x/√2| ≤ 3/5`. -/
private lemma z_bound_at_06 {x : ℝ} (hx : |x| ≤ 3/5) :
    |x / Real.sqrt 2| ≤ 3/5 := by
  have hsqrt2_pos : (0 : ℝ) < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  have hsqrt2_ge_1 : (1 : ℝ) ≤ Real.sqrt 2 := by
    rw [show (1:ℝ) = Real.sqrt 1 from Real.sqrt_one.symm]
    exact Real.sqrt_le_sqrt (by norm_num)
  rw [abs_div, abs_of_pos hsqrt2_pos, div_le_iff₀ hsqrt2_pos]
  nlinarith [hx, hsqrt2_ge_1]

/-- For `|x| ≤ 3/5`, the realErf Taylor remainder satisfies
`(2/√π)·5·|z|^9/864 ≤ 1/12000`. -/
private lemma erf_residual_at_06 {x : ℝ} (hx : |x| ≤ 3/5) :
    (2 / Real.sqrt Real.pi) * (5 * |x / Real.sqrt 2|^9 / 864) ≤ 1 / 12000 := by
  have hpi_pos : (0 : ℝ) < Real.pi := Real.pi_pos
  have hsqrt_pi_pos : (0 : ℝ) < Real.sqrt Real.pi := Real.sqrt_pos.mpr hpi_pos
  have hpi_ge : (3/2 : ℝ) ≤ Real.sqrt Real.pi := by
    rw [show (3/2:ℝ) = Real.sqrt ((3/2)^2) from (Real.sqrt_sq (by norm_num)).symm]
    apply Real.sqrt_le_sqrt
    have : ((3/2:ℝ))^2 = 9/4 := by norm_num
    rw [this]; linarith [Real.pi_gt_three]
  have h_inv_pi : (2 : ℝ) / Real.sqrt Real.pi ≤ 4/3 := by
    rw [div_le_div_iff₀ hsqrt_pi_pos (by norm_num : (0:ℝ) < 3)]
    nlinarith [hpi_ge]
  have hz := z_bound_at_06 hx
  have hz_nn : (0 : ℝ) ≤ |x / Real.sqrt 2| := abs_nonneg _
  have hz9 : |x / Real.sqrt 2|^9 ≤ (3/5)^9 := pow_le_pow_left₀ hz_nn hz 9
  calc (2 / Real.sqrt Real.pi) * (5 * |x / Real.sqrt 2|^9 / 864)
      ≤ (4/3 : ℝ) * (5 * (3/5)^9 / 864) := by
        apply mul_le_mul h_inv_pi
        · apply div_le_div_of_nonneg_right
          · exact mul_le_mul_of_nonneg_left hz9 (by norm_num)
          · norm_num
        · positivity
        · norm_num
    _ ≤ 1/12000 := by norm_num

/-- Closure for `|x| ≤ 3/5`: gelu approximation error is at most `1/1000`. -/
private theorem approx_gelu_error_bound_06 {x : ℝ} (hx : |x| ≤ 3/5) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ 1 / 1000 := by
  rw [approxGeLUScalar_sub_exactGeLUScalar x]
  have hx_le_one : |x| ≤ 1 := by linarith
  have h_taylor := gelu_gap_taylor_decomposition hx_le_one
  have h_tanh := tanh_residual_at_06 hx
  have h_poly := poly_diff_with_correction_bound_06 hx
  have h_erf := erf_residual_at_06 hx
  have h_gap_bound :
      |Real.tanh ((7978845608028654 / 10000000000000000 : ℝ) *
                    (x + (44715 / 1000000) * (x * x * x))) -
          realErf (x / Real.sqrt 2)|
        ≤ 1/1000 + 1/1500 + 1/12000 := by
    calc |Real.tanh _ - realErf _|
        ≤ |(7978845608028654 / 10000000000000000 : ℝ) *
                (x + (44715 / 1000000) * (x * x * x))|^7 / 7
          + |tanhTaylor5 _ - (2 / Real.sqrt Real.pi) * realErfTaylor7 _|
          + (2 / Real.sqrt Real.pi) * (5 * |x / Real.sqrt 2|^9 / 864) :=
            h_taylor
      _ ≤ 1/1000 + 1/1500 + 1/12000 := by linarith
  rw [abs_mul, abs_div, abs_two]
  have hx2_le : |x| / 2 ≤ 3/10 := by linarith
  have hx2_nn : 0 ≤ |x| / 2 := by positivity
  have h_combined : 1/1000 + 1/1500 + (1/12000 : ℝ) ≤ 1/300 := by norm_num
  calc |x| / 2 *
          |Real.tanh _ - realErf _|
      ≤ |x| / 2 * (1/1000 + 1/1500 + 1/12000) :=
          mul_le_mul_of_nonneg_left h_gap_bound hx2_nn
    _ ≤ |x| / 2 * (1/300) :=
          mul_le_mul_of_nonneg_left h_combined hx2_nn
    _ ≤ 3/10 * (1/300) :=
          mul_le_mul_of_nonneg_right hx2_le (by norm_num)
    _ ≤ 1/1000 := by norm_num

/-! ### Closure for `|x| ≤ 3/4` via 7th-order tanh Taylor

The 7th-order tanh Taylor expansion `tanhTaylor7(u) := u − u³/3 + 2u⁵/15 − 17u⁷/315`
gives a tighter remainder `|tanh u − tanhTaylor7(u)| ≤ |u|⁹/9` (vs `|u|⁷/7`).
This pushes the closed range from `|x| ≤ 3/5` up to `|x| ≤ 3/4`. -/

/-- 7th-order tanh Taylor polynomial: `u − u³/3 + 2u⁵/15 − 17u⁷/315`. -/
private noncomputable def tanhTaylor7 (u : ℝ) : ℝ :=
  u - u^3/3 + 2 * u^5/15 - 17 * u^7/315

/-- Tanh 7th-order Taylor at the gelu argument: for `|x| ≤ 1`, `|u| ≤ 1`, so
`abs_tanh_sub_taylor7_le` applies. -/
lemma abs_tanh_at_gelu_arg_sub_taylor7_le {x : ℝ} (hx : |x| ≤ 1) :
    |Real.tanh ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))) -
        ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x)) -
         ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))) ^ 3 / 3 +
         2 * ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))) ^ 5 / 15 -
         17 * ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))) ^ 7 / 315)|
      ≤ |(7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))| ^ 9 / 9 := by
  set u : ℝ := (7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x)) with hu_def
  have hu_bound : |u| ≤ 1 := by
    rw [hu_def, abs_mul]
    have h_c_pos : 0 < (7978845608028654 / 10000000000000000 : ℝ) := by norm_num
    rw [abs_of_pos h_c_pos]
    have h_inner : |x + (44715 / 1000000 : ℝ) * (x * x * x)|
        ≤ |x| + (44715 / 1000000) * |x| ^ 3 := by
      have hb := abs_add_le x ((44715 / 1000000 : ℝ) * (x * x * x))
      have h_eq : |(44715 / 1000000 : ℝ) * (x * x * x)|
            = (44715 / 1000000) * |x| ^ 3 := by
        rw [abs_mul, show |(44715 / 1000000 : ℝ)| = 44715 / 1000000 from by norm_num]
        congr 1
        rw [show x * x * x = x ^ 3 from by ring, abs_pow]
      linarith
    have h_x3_le : |x| ^ 3 ≤ |x| := by
      nlinarith [sq_nonneg (|x| - 1), sq_nonneg (|x|), abs_nonneg x]
    have h_inner_le : |x + (44715 / 1000000 : ℝ) * (x * x * x)|
          ≤ |x| * (1 + 44715 / 1000000) := by
      have h1 : |x| + (44715 / 1000000 : ℝ) * |x| ^ 3
            ≤ |x| + (44715 / 1000000) * |x| := by
        nlinarith
      have h_factor : |x| + (44715 / 1000000 : ℝ) * |x|
            = |x| * (1 + 44715 / 1000000) := by ring
      linarith
    calc (7978845608028654 / 10000000000000000 : ℝ) *
            |x + (44715 / 1000000) * (x * x * x)|
        ≤ (7978845608028654 / 10000000000000000) *
            (|x| * (1 + 44715 / 1000000)) := by
              apply mul_le_mul_of_nonneg_left h_inner_le (by norm_num)
      _ ≤ (7978845608028654 / 10000000000000000) *
            (1 * (1 + 44715 / 1000000)) := by
              apply mul_le_mul_of_nonneg_left
              · exact mul_le_mul_of_nonneg_right hx (by norm_num)
              · norm_num
      _ ≤ 1 := by norm_num
  exact abs_tanh_sub_taylor7_le hu_bound

/-- Three-piece triangle decomposition with 7th-order tanh. -/
lemma gelu_gap_taylor7_decomposition {x : ℝ} (hx : |x| ≤ 1) :
    |Real.tanh ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))) -
        realErf (x / Real.sqrt 2)|
      ≤ |(7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))| ^ 9 / 9
        + |tanhTaylor7
              ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x)))
            - (2 / Real.sqrt Real.pi) *
                realErfTaylor7 (x / Real.sqrt 2)|
        + (2 / Real.sqrt Real.pi) * (5 * |x / Real.sqrt 2| ^ 9 / 864) := by
  set u : ℝ := (7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x)) with hu_def
  set z : ℝ := x / Real.sqrt 2 with hz_def
  have h_tanh_res : |Real.tanh u - tanhTaylor7 u| ≤ |u| ^ 9 / 9 := by
    have h := abs_tanh_at_gelu_arg_sub_taylor7_le hx
    show |Real.tanh u - (u - u^3/3 + 2 * u^5/15 - 17 * u^7/315)| ≤ |u| ^ 9 / 9
    convert h using 2
  have h_erf_res :
      |realErf z - (2 / Real.sqrt Real.pi) * realErfTaylor7 z|
        ≤ (2 / Real.sqrt Real.pi) * (5 * |z| ^ 9 / 864) := by
    have h := abs_realErf_at_gelu_arg_sub_taylor7_le hx
    show |realErf z - (2 / Real.sqrt Real.pi) *
            (z - z^3/3 + z^5/10 - z^7/42)|
          ≤ (2 / Real.sqrt Real.pi) * (5 * |z| ^ 9 / 864)
    convert h using 2
  have h_split :
      Real.tanh u - realErf z
        = (Real.tanh u - tanhTaylor7 u)
            + (tanhTaylor7 u - (2 / Real.sqrt Real.pi) * realErfTaylor7 z)
            - (realErf z - (2 / Real.sqrt Real.pi) * realErfTaylor7 z) := by
    ring
  rw [h_split]
  calc |(Real.tanh u - tanhTaylor7 u)
          + (tanhTaylor7 u - (2 / Real.sqrt Real.pi) * realErfTaylor7 z)
          - (realErf z - (2 / Real.sqrt Real.pi) * realErfTaylor7 z)|
      ≤ |(Real.tanh u - tanhTaylor7 u)
            + (tanhTaylor7 u - (2 / Real.sqrt Real.pi) * realErfTaylor7 z)|
        + |realErf z - (2 / Real.sqrt Real.pi) * realErfTaylor7 z| :=
            abs_sub _ _
    _ ≤ (|Real.tanh u - tanhTaylor7 u|
            + |tanhTaylor7 u - (2 / Real.sqrt Real.pi) * realErfTaylor7 z|)
        + |realErf z - (2 / Real.sqrt Real.pi) * realErfTaylor7 z| := by
            gcongr
            exact abs_add_le _ _
    _ ≤ (|u| ^ 9 / 9
            + |tanhTaylor7 u - (2 / Real.sqrt Real.pi) * realErfTaylor7 z|)
        + (2 / Real.sqrt Real.pi) * (5 * |z| ^ 9 / 864) := by
            gcongr

/-- Polynomial expansion identity for the 7th-order Taylor tanh. -/
private lemma tanhTaylor7_sub_rationalErf_eq (c k x : ℝ) :
    tanhTaylor7 (c * (x + k * (x*x*x))) -
        c * (x - x^3/6 + x^5/40 - x^7/336)
      = (c*k - c^3/3 + c/6) * x^3
          + (-c^3*k + 2*c^5/15 - c/40) * x^5
          + (-c^3*k^2 + 2*c^5*k/3 + c/336 - 17*c^7/315) * x^7
          + (-c^3*k^3/3 + 4*c^5*k^2/3 - 17*c^7*k/45) * x^9
          + (4*c^5*k^3/3 - 17*c^7*k^2/15) * x^11
          + (2*c^5*k^4/3 - 17*c^7*k^3/9) * x^13
          + (2*c^5*k^5/15 - 17*c^7*k^4/9) * x^15
          + (-17*c^7*k^5/15) * x^17
          + (-17*c^7*k^6/45) * x^19
          + (-17*c^7*k^7/315) * x^21 := by
  unfold tanhTaylor7
  ring

/-- Polynomial difference bound for `|x| ≤ 3/4` (10 terms). Sum is `≈ 6.17e-4`. -/
private lemma poly_diff_bound_075 {x : ℝ} (hx : |x| ≤ 3/4) :
    |tanhTaylor7 ((7978845608028654 / 10000000000000000 : ℝ) *
                    (x + (44715 / 1000000) * (x * x * x))) -
        (7978845608028654 / 10000000000000000 : ℝ) *
            (x - x^3/6 + x^5/40 - x^7/336)|
      ≤ 65 / 100000 := by
  set c : ℝ := 7978845608028654 / 10000000000000000 with hc_def
  set k : ℝ := 44715 / 1000000 with hk_def
  rw [tanhTaylor7_sub_rationalErf_eq c k x]
  have hx_nn : 0 ≤ |x| := abs_nonneg _
  have hx3 : |x|^3 ≤ (3/4)^3 := pow_le_pow_left₀ hx_nn hx 3
  have hx5 : |x|^5 ≤ (3/4)^5 := pow_le_pow_left₀ hx_nn hx 5
  have hx7 : |x|^7 ≤ (3/4)^7 := pow_le_pow_left₀ hx_nn hx 7
  have hx9 : |x|^9 ≤ (3/4)^9 := pow_le_pow_left₀ hx_nn hx 9
  have hx11 : |x|^11 ≤ (3/4)^11 := pow_le_pow_left₀ hx_nn hx 11
  have hx13 : |x|^13 ≤ (3/4)^13 := pow_le_pow_left₀ hx_nn hx 13
  have hx15 : |x|^15 ≤ (3/4)^15 := pow_le_pow_left₀ hx_nn hx 15
  have hx17 : |x|^17 ≤ (3/4)^17 := pow_le_pow_left₀ hx_nn hx 17
  have hx19 : |x|^19 ≤ (3/4)^19 := pow_le_pow_left₀ hx_nn hx 19
  have hx21 : |x|^21 ≤ (3/4)^21 := pow_le_pow_left₀ hx_nn hx 21
  set a3 := c*k - c^3/3 + c/6 with ha3
  set a5 := -c^3*k + 2*c^5/15 - c/40 with ha5
  set a7 := -c^3*k^2 + 2*c^5*k/3 + c/336 - 17*c^7/315 with ha7
  set a9 := -c^3*k^3/3 + 4*c^5*k^2/3 - 17*c^7*k/45 with ha9
  set a11 := 4*c^5*k^3/3 - 17*c^7*k^2/15 with ha11
  set a13 := 2*c^5*k^4/3 - 17*c^7*k^3/9 with ha13
  set a15 := 2*c^5*k^5/15 - 17*c^7*k^4/9 with ha15
  set a17 := -17*c^7*k^5/15 with ha17
  set a19 := -17*c^7*k^6/45 with ha19
  set a21 := -17*c^7*k^7/315 with ha21
  have hb3 : |a3 * x^3| ≤ |a3| * (3/4)^3 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx3 (abs_nonneg _)
  have hb5 : |a5 * x^5| ≤ |a5| * (3/4)^5 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx5 (abs_nonneg _)
  have hb7 : |a7 * x^7| ≤ |a7| * (3/4)^7 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx7 (abs_nonneg _)
  have hb9 : |a9 * x^9| ≤ |a9| * (3/4)^9 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx9 (abs_nonneg _)
  have hb11 : |a11 * x^11| ≤ |a11| * (3/4)^11 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx11 (abs_nonneg _)
  have hb13 : |a13 * x^13| ≤ |a13| * (3/4)^13 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx13 (abs_nonneg _)
  have hb15 : |a15 * x^15| ≤ |a15| * (3/4)^15 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx15 (abs_nonneg _)
  have hb17 : |a17 * x^17| ≤ |a17| * (3/4)^17 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx17 (abs_nonneg _)
  have hb19 : |a19 * x^19| ≤ |a19| * (3/4)^19 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx19 (abs_nonneg _)
  have hb21 : |a21 * x^21| ≤ |a21| * (3/4)^21 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx21 (abs_nonneg _)
  have hn3 : |a3| * (3/4)^3 ≤ 28 / 100000 := by
    rw [ha3, hc_def, hk_def]; rw [abs_of_neg (by norm_num)]; norm_num
  have hn5 : |a5| * (3/4)^5 ≤ 11 / 100000 := by
    rw [ha5, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn7 : |a7| * (3/4)^7 ≤ 15 / 1000000 := by
    rw [ha7, hc_def, hk_def]; rw [abs_of_neg (by norm_num)]; norm_num
  have hn9 : |a9| * (3/4)^9 ≤ 20 / 100000 := by
    rw [ha9, hc_def, hk_def]; rw [abs_of_neg (by norm_num)]; norm_num
  have hn11 : |a11| * (3/4)^11 ≤ 19 / 1000000 := by
    rw [ha11, hc_def, hk_def]; rw [abs_of_neg (by norm_num)]; norm_num
  have hn13 : |a13| * (3/4)^13 ≤ 9 / 10000000 := by
    rw [ha13, hc_def, hk_def]; rw [abs_of_neg (by norm_num)]; norm_num
  have hn15 : |a15| * (3/4)^15 ≤ 3 / 100000000 := by
    rw [ha15, hc_def, hk_def]; rw [abs_of_neg (by norm_num)]; norm_num
  have hn17 : |a17| * (3/4)^17 ≤ 4 / 10000000000 := by
    rw [ha17, hc_def, hk_def]; rw [abs_of_neg (by norm_num)]; norm_num
  have hn19 : |a19| * (3/4)^19 ≤ 3 / 1000000000000 := by
    rw [ha19, hc_def, hk_def]; rw [abs_of_neg (by norm_num)]; norm_num
  have hn21 : |a21| * (3/4)^21 ≤ 1 / 100000000000000 := by
    rw [ha21, hc_def, hk_def]; rw [abs_of_neg (by norm_num)]; norm_num
  calc |a3 * x^3 + a5 * x^5 + a7 * x^7 + a9 * x^9 + a11 * x^11 + a13 * x^13
          + a15 * x^15 + a17 * x^17 + a19 * x^19 + a21 * x^21|
      ≤ |a3 * x^3| + |a5 * x^5| + |a7 * x^7| + |a9 * x^9| + |a11 * x^11|
          + |a13 * x^13| + |a15 * x^15| + |a17 * x^17| + |a19 * x^19|
          + |a21 * x^21| := by
        have h1 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11
                               + a13*x^13 + a15*x^15 + a17*x^17 + a19*x^19) (a21*x^21)
        have h2 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11
                               + a13*x^13 + a15*x^15 + a17*x^17) (a19*x^19)
        have h3 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11
                               + a13*x^13 + a15*x^15) (a17*x^17)
        have h4 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11
                               + a13*x^13) (a15*x^15)
        have h5 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11)
                               (a13*x^13)
        have h6 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9) (a11*x^11)
        have h7 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7) (a9*x^9)
        have h8 := abs_add_le (a3*x^3 + a5*x^5) (a7*x^7)
        have h9 := abs_add_le (a3*x^3) (a5*x^5)
        linarith
    _ ≤ |a3| * (3/4)^3 + |a5| * (3/4)^5 + |a7| * (3/4)^7 + |a9| * (3/4)^9
          + |a11| * (3/4)^11 + |a13| * (3/4)^13 + |a15| * (3/4)^15
          + |a17| * (3/4)^17 + |a19| * (3/4)^19 + |a21| * (3/4)^21 := by
        linarith [hb3, hb5, hb7, hb9, hb11, hb13, hb15, hb17, hb19, hb21]
    _ ≤ 28/100000 + 11/100000 + 15/1000000 + 20/100000 + 19/1000000
          + 9/10000000 + 3/100000000 + 4/10000000000 + 3/1000000000000
          + 1/100000000000000 := by
        linarith [hn3, hn5, hn7, hn9, hn11, hn13, hn15, hn17, hn19, hn21]
    _ ≤ 65/100000 := by norm_num

/-- Combined polynomial bound + `c − √(2/π)` correction for `|x| ≤ 3/4`. -/
private lemma poly_diff_with_correction_bound_075 {x : ℝ} (hx : |x| ≤ 3/4) :
    |tanhTaylor7 ((7978845608028654 / 10000000000000000 : ℝ) *
                    (x + (44715 / 1000000) * (x * x * x))) -
        (2 / Real.sqrt Real.pi) *
            realErfTaylor7 (x / Real.sqrt 2)|
      ≤ 7 / 10000 := by
  have h_rewrite : (2 / Real.sqrt Real.pi) *
                       realErfTaylor7 (x / Real.sqrt 2)
                = Real.sqrt (2 / Real.pi) * (x - x^3/6 + x^5/40 - x^7/336) := by
    show (2 / Real.sqrt Real.pi) *
            ((x / Real.sqrt 2) - (x / Real.sqrt 2)^3/3
              + (x / Real.sqrt 2)^5/10 - (x / Real.sqrt 2)^7/42)
          = Real.sqrt (2 / Real.pi) * (x - x^3/6 + x^5/40 - x^7/336)
    exact two_div_sqrt_pi_realErfTaylor7_eq x
  rw [h_rewrite]
  have h_split :
      tanhTaylor7 ((7978845608028654 / 10000000000000000 : ℝ) *
                      (x + (44715 / 1000000) * (x * x * x))) -
          Real.sqrt (2 / Real.pi) * (x - x^3/6 + x^5/40 - x^7/336)
        = (tanhTaylor7 ((7978845608028654 / 10000000000000000 : ℝ) *
                          (x + (44715 / 1000000) * (x * x * x))) -
            (7978845608028654 / 10000000000000000 : ℝ) *
                (x - x^3/6 + x^5/40 - x^7/336))
          + ((7978845608028654 / 10000000000000000 : ℝ) -
              Real.sqrt (2 / Real.pi)) *
              (x - x^3/6 + x^5/40 - x^7/336) := by
    ring
  rw [h_split]
  have hpoly := poly_diff_bound_075 hx
  have hcorr_sqrt := abs_c_sub_sqrt_two_div_pi_le
  have hx_le_one : |x| ≤ 1 := by linarith
  have hR := R_bound_at_one hx_le_one
  calc |(tanhTaylor7 _ - _ * (x - x^3/6 + x^5/40 - x^7/336))
            + ((7978845608028654 / 10000000000000000 : ℝ)
                - Real.sqrt (2 / Real.pi)) *
                (x - x^3/6 + x^5/40 - x^7/336)|
      ≤ |tanhTaylor7 _ - _ * (x - x^3/6 + x^5/40 - x^7/336)|
        + |((7978845608028654 / 10000000000000000 : ℝ)
              - Real.sqrt (2 / Real.pi)) *
                (x - x^3/6 + x^5/40 - x^7/336)| := abs_add_le _ _
    _ ≤ 65/100000 + (1/10^15) * 2 := by
        gcongr
        rw [abs_mul]
        gcongr
    _ ≤ 7/10000 := by norm_num

/-- For `|x| ≤ 3/4`, `|u| ≤ 62/100`. (Numerically `|u| ≈ 0.614`.) -/
private lemma u_bound_at_075 {x : ℝ} (hx : |x| ≤ 3/4) :
    |(7978845608028654 / 10000000000000000 : ℝ) *
        (x + (44715 / 1000000) * (x * x * x))| ≤ 62 / 100 := by
  rw [abs_mul]
  have hc_pos : (0 : ℝ) < 7978845608028654 / 10000000000000000 := by norm_num
  rw [abs_of_pos hc_pos]
  have h_x3_le : |x|^3 ≤ (3/4)^3 := pow_le_pow_left₀ (abs_nonneg x) hx 3
  have h_inner : |x + (44715 / 1000000 : ℝ) * (x * x * x)|
        ≤ |x| + (44715 / 1000000) * |x|^3 := by
    have hb := abs_add_le x ((44715 / 1000000 : ℝ) * (x * x * x))
    have h_eq : |(44715 / 1000000 : ℝ) * (x * x * x)|
          = (44715 / 1000000) * |x|^3 := by
      rw [abs_mul, show |(44715 / 1000000 : ℝ)| = 44715 / 1000000 from by norm_num]
      congr 1
      rw [show x * x * x = x^3 from by ring, abs_pow]
    linarith
  calc (7978845608028654 / 10000000000000000 : ℝ) *
            |x + (44715 / 1000000) * (x * x * x)|
      ≤ (7978845608028654 / 10000000000000000) *
          (|x| + (44715 / 1000000) * |x|^3) := by
            apply mul_le_mul_of_nonneg_left h_inner hc_pos.le
    _ ≤ (7978845608028654 / 10000000000000000) *
          ((3/4) + (44715 / 1000000) * (3/4)^3) := by
            apply mul_le_mul_of_nonneg_left _ hc_pos.le
            have hk_pos : (0 : ℝ) ≤ 44715 / 1000000 := by norm_num
            have h := mul_le_mul_of_nonneg_left h_x3_le hk_pos
            linarith
    _ ≤ 62/100 := by norm_num

/-- For `|x| ≤ 3/4`, `|u|⁹/9 ≤ 16/10000`. (True value `≈ 1.51·10⁻³` using
the loose u-bound `|u| ≤ 62/100`.) -/
private lemma tanh_residual_at_075 {x : ℝ} (hx : |x| ≤ 3/4) :
    |(7978845608028654 / 10000000000000000 : ℝ) *
        (x + (44715 / 1000000) * (x * x * x))|^9 / 9
      ≤ 16 / 10000 := by
  have hu := u_bound_at_075 hx
  have hu_nn : (0 : ℝ) ≤ |(7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))| := abs_nonneg _
  have hu9 : |(7978845608028654 / 10000000000000000 : ℝ) *
                (x + (44715 / 1000000) * (x * x * x))|^9
              ≤ (62/100)^9 :=
    pow_le_pow_left₀ hu_nn hu 9
  calc |(7978845608028654 / 10000000000000000 : ℝ) *
            (x + (44715 / 1000000) * (x * x * x))|^9 / 9
      ≤ (62/100)^9 / 9 :=
        div_le_div_of_nonneg_right hu9 (by norm_num)
    _ ≤ 16/10000 := by norm_num

/-- For `|x| ≤ 3/4`, `(2/√π)·5·|z|⁹/864 ≤ 4/100000`. The bound uses
`|z|² = x²/2 ≤ M²/2` (so `|z|⁸ ≤ (M²/2)⁴ = M⁸/16`) plus `|z| ≤ M` to get
`|z|⁹ ≤ M⁹/16`, much tighter than `|z|⁹ ≤ M⁹` directly. -/
private lemma erf_residual_at_075 {x : ℝ} (hx : |x| ≤ 3/4) :
    (2 / Real.sqrt Real.pi) * (5 * |x / Real.sqrt 2|^9 / 864) ≤ 4 / 100000 := by
  have hpi_pos : (0 : ℝ) < Real.pi := Real.pi_pos
  have hsqrt_pi_pos : (0 : ℝ) < Real.sqrt Real.pi := Real.sqrt_pos.mpr hpi_pos
  have hpi_ge : (3/2 : ℝ) ≤ Real.sqrt Real.pi := by
    rw [show (3/2:ℝ) = Real.sqrt ((3/2)^2) from (Real.sqrt_sq (by norm_num)).symm]
    apply Real.sqrt_le_sqrt
    have : ((3/2:ℝ))^2 = 9/4 := by norm_num
    rw [this]; linarith [Real.pi_gt_three]
  have h_inv_pi : (2 : ℝ) / Real.sqrt Real.pi ≤ 4/3 := by
    rw [div_le_div_iff₀ hsqrt_pi_pos (by norm_num : (0:ℝ) < 3)]
    nlinarith [hpi_ge]
  have hsqrt2_pos : (0 : ℝ) < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  have hsqrt2_sq : (Real.sqrt 2)^2 = 2 := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 2)
  have hsqrt2_ge_1 : (1 : ℝ) ≤ Real.sqrt 2 := by
    rw [show (1:ℝ) = Real.sqrt 1 from Real.sqrt_one.symm]
    exact Real.sqrt_le_sqrt (by norm_num)
  have hz_le : |x / Real.sqrt 2| ≤ 3/4 := by
    rw [abs_div, abs_of_pos hsqrt2_pos, div_le_iff₀ hsqrt2_pos]
    nlinarith [hx, hsqrt2_ge_1]
  have hz_nn : (0 : ℝ) ≤ |x / Real.sqrt 2| := abs_nonneg _
  -- |z|² = x²/2 (using (√2)² = 2):
  have hz_sq : |x / Real.sqrt 2|^2 = x^2 / 2 := by
    rw [show |x / Real.sqrt 2|^2 = (x / Real.sqrt 2)^2 from (sq_abs _),
        div_pow, hsqrt2_sq]
  -- |z|² ≤ (3/4)²/2 = 9/32:
  have hz_sq_le : |x / Real.sqrt 2|^2 ≤ 9/32 := by
    rw [hz_sq]
    have h_x_sq : x^2 ≤ (3/4)^2 := by
      rw [show x^2 = |x|^2 from (sq_abs x).symm]
      exact pow_le_pow_left₀ (abs_nonneg x) hx 2
    have h_target : (3/4 : ℝ)^2 / 2 = 9/32 := by norm_num
    linarith
  -- |z|⁸ ≤ (9/32)⁴:
  have hz_sq_nn : 0 ≤ |x / Real.sqrt 2|^2 := sq_nonneg _
  have hz8 : |x / Real.sqrt 2|^8 ≤ (9/32)^4 := by
    rw [show |x / Real.sqrt 2|^8 = (|x / Real.sqrt 2|^2)^4 from by ring]
    exact pow_le_pow_left₀ hz_sq_nn hz_sq_le 4
  -- |z|⁹ = |z|·|z|⁸ ≤ (3/4)·(9/32)⁴ = M⁹/16 where M = 3/4:
  have hz9 : |x / Real.sqrt 2|^9 ≤ (3/4) * (9/32)^4 := by
    rw [show |x / Real.sqrt 2|^9 = |x / Real.sqrt 2| * |x / Real.sqrt 2|^8 from by ring]
    have h_pos8 : (0 : ℝ) ≤ (9/32 : ℝ)^4 := by positivity
    exact mul_le_mul hz_le hz8 (pow_nonneg hz_nn 8) (by norm_num)
  calc (2 / Real.sqrt Real.pi) * (5 * |x / Real.sqrt 2|^9 / 864)
      ≤ (4/3 : ℝ) * (5 * ((3/4) * (9/32)^4) / 864) := by
        apply mul_le_mul h_inv_pi
        · apply div_le_div_of_nonneg_right
          · exact mul_le_mul_of_nonneg_left hz9 (by norm_num)
          · norm_num
        · positivity
        · norm_num
    _ ≤ 4/100000 := by norm_num

/-- Closure for `|x| ≤ 3/4`: gelu approximation error is at most `1/1000`. -/
private theorem approx_gelu_error_bound_075 {x : ℝ} (hx : |x| ≤ 3/4) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ 1 / 1000 := by
  rw [approxGeLUScalar_sub_exactGeLUScalar x]
  have hx_le_one : |x| ≤ 1 := by linarith
  have h_taylor := gelu_gap_taylor7_decomposition hx_le_one
  have h_tanh := tanh_residual_at_075 hx
  have h_poly := poly_diff_with_correction_bound_075 hx
  have h_erf := erf_residual_at_075 hx
  have h_gap_bound :
      |Real.tanh ((7978845608028654 / 10000000000000000 : ℝ) *
                    (x + (44715 / 1000000) * (x * x * x))) -
          realErf (x / Real.sqrt 2)|
        ≤ 16/10000 + 7/10000 + 4/100000 := by
    calc |Real.tanh _ - realErf _|
        ≤ |(7978845608028654 / 10000000000000000 : ℝ) *
                (x + (44715 / 1000000) * (x * x * x))|^9 / 9
          + |tanhTaylor7 _ - (2 / Real.sqrt Real.pi) * realErfTaylor7 _|
          + (2 / Real.sqrt Real.pi) * (5 * |x / Real.sqrt 2|^9 / 864) :=
            h_taylor
      _ ≤ 16/10000 + 7/10000 + 4/100000 := by linarith
  rw [abs_mul, abs_div, abs_two]
  have hx2_le : |x| / 2 ≤ 3/8 := by linarith
  have hx2_nn : 0 ≤ |x| / 2 := by positivity
  have h_combined : 16/10000 + 7/10000 + (4/100000 : ℝ) ≤ 24/10000 := by norm_num
  calc |x| / 2 *
          |Real.tanh _ - realErf _|
      ≤ |x| / 2 * (16/10000 + 7/10000 + 4/100000) :=
          mul_le_mul_of_nonneg_left h_gap_bound hx2_nn
    _ ≤ |x| / 2 * (24/10000) :=
          mul_le_mul_of_nonneg_left h_combined hx2_nn
    _ ≤ 3/8 * (24/10000) :=
          mul_le_mul_of_nonneg_right hx2_le (by norm_num)
    _ ≤ 1/1000 := by norm_num

/-! ### Closure for `|x| ≤ 4/5` via 9th-order tanh Taylor

The 9th-order tanh Taylor `tanhTaylor9(u) := u − u³/3 + 2u⁵/15 − 17u⁷/315 + 62u⁹/2835`
gives remainder `|tanh u − tanhTaylor9(u)| ≤ |u|¹¹/11`. At `|x| = 4/5`, the
tanh residual drops by an order of magnitude vs the 7th-order remainder,
opening up the closure for `|x| ≤ 4/5`. -/

/-- 9th-order tanh Taylor polynomial. -/
private noncomputable def tanhTaylor9 (u : ℝ) : ℝ :=
  u - u^3/3 + 2 * u^5/15 - 17 * u^7/315 + 62 * u^9/2835

/-- 9th-order tanh wiring at the gelu argument. -/
lemma abs_tanh_at_gelu_arg_sub_taylor9_le {x : ℝ} (hx : |x| ≤ 1) :
    |Real.tanh ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))) -
        ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x)) -
         ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))) ^ 3 / 3 +
         2 * ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))) ^ 5 / 15 -
         17 * ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))) ^ 7 / 315 +
         62 * ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))) ^ 9 / 2835)|
      ≤ |(7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))| ^ 11 / 11 := by
  set u : ℝ := (7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x)) with hu_def
  have hu_bound : |u| ≤ 1 := by
    rw [hu_def, abs_mul]
    have h_c_pos : 0 < (7978845608028654 / 10000000000000000 : ℝ) := by norm_num
    rw [abs_of_pos h_c_pos]
    have h_inner : |x + (44715 / 1000000 : ℝ) * (x * x * x)|
        ≤ |x| + (44715 / 1000000) * |x| ^ 3 := by
      have hb := abs_add_le x ((44715 / 1000000 : ℝ) * (x * x * x))
      have h_eq : |(44715 / 1000000 : ℝ) * (x * x * x)|
            = (44715 / 1000000) * |x| ^ 3 := by
        rw [abs_mul, show |(44715 / 1000000 : ℝ)| = 44715 / 1000000 from by norm_num]
        congr 1
        rw [show x * x * x = x ^ 3 from by ring, abs_pow]
      linarith
    have h_x3_le : |x| ^ 3 ≤ |x| := by
      nlinarith [sq_nonneg (|x| - 1), sq_nonneg (|x|), abs_nonneg x]
    have h_inner_le : |x + (44715 / 1000000 : ℝ) * (x * x * x)|
          ≤ |x| * (1 + 44715 / 1000000) := by
      have h1 : |x| + (44715 / 1000000 : ℝ) * |x| ^ 3
            ≤ |x| + (44715 / 1000000) * |x| := by
        nlinarith
      have h_factor : |x| + (44715 / 1000000 : ℝ) * |x|
            = |x| * (1 + 44715 / 1000000) := by ring
      linarith
    calc (7978845608028654 / 10000000000000000 : ℝ) *
            |x + (44715 / 1000000) * (x * x * x)|
        ≤ (7978845608028654 / 10000000000000000) *
            (|x| * (1 + 44715 / 1000000)) := by
              apply mul_le_mul_of_nonneg_left h_inner_le (by norm_num)
      _ ≤ (7978845608028654 / 10000000000000000) *
            (1 * (1 + 44715 / 1000000)) := by
              apply mul_le_mul_of_nonneg_left
              · exact mul_le_mul_of_nonneg_right hx (by norm_num)
              · norm_num
      _ ≤ 1 := by norm_num
  exact abs_tanh_sub_taylor9_le hu_bound

/-- 3-piece triangle decomposition with 9th-order tanh. -/
lemma gelu_gap_taylor9_decomposition {x : ℝ} (hx : |x| ≤ 1) :
    |Real.tanh ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))) -
        realErf (x / Real.sqrt 2)|
      ≤ |(7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))| ^ 11 / 11
        + |tanhTaylor9
              ((7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x)))
            - (2 / Real.sqrt Real.pi) *
                realErfTaylor7 (x / Real.sqrt 2)|
        + (2 / Real.sqrt Real.pi) * (5 * |x / Real.sqrt 2| ^ 9 / 864) := by
  set u : ℝ := (7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x)) with hu_def
  set z : ℝ := x / Real.sqrt 2 with hz_def
  have h_tanh_res : |Real.tanh u - tanhTaylor9 u| ≤ |u| ^ 11 / 11 := by
    have h := abs_tanh_at_gelu_arg_sub_taylor9_le hx
    show |Real.tanh u - (u - u^3/3 + 2 * u^5/15 - 17 * u^7/315
                          + 62 * u^9/2835)| ≤ |u| ^ 11 / 11
    convert h using 2
  have h_erf_res :
      |realErf z - (2 / Real.sqrt Real.pi) * realErfTaylor7 z|
        ≤ (2 / Real.sqrt Real.pi) * (5 * |z| ^ 9 / 864) := by
    have h := abs_realErf_at_gelu_arg_sub_taylor7_le hx
    show |realErf z - (2 / Real.sqrt Real.pi) *
            (z - z^3/3 + z^5/10 - z^7/42)|
          ≤ (2 / Real.sqrt Real.pi) * (5 * |z| ^ 9 / 864)
    convert h using 2
  have h_split :
      Real.tanh u - realErf z
        = (Real.tanh u - tanhTaylor9 u)
            + (tanhTaylor9 u - (2 / Real.sqrt Real.pi) * realErfTaylor7 z)
            - (realErf z - (2 / Real.sqrt Real.pi) * realErfTaylor7 z) := by
    ring
  rw [h_split]
  calc |(Real.tanh u - tanhTaylor9 u)
          + (tanhTaylor9 u - (2 / Real.sqrt Real.pi) * realErfTaylor7 z)
          - (realErf z - (2 / Real.sqrt Real.pi) * realErfTaylor7 z)|
      ≤ |(Real.tanh u - tanhTaylor9 u)
            + (tanhTaylor9 u - (2 / Real.sqrt Real.pi) * realErfTaylor7 z)|
        + |realErf z - (2 / Real.sqrt Real.pi) * realErfTaylor7 z| :=
            abs_sub _ _
    _ ≤ (|Real.tanh u - tanhTaylor9 u|
            + |tanhTaylor9 u - (2 / Real.sqrt Real.pi) * realErfTaylor7 z|)
        + |realErf z - (2 / Real.sqrt Real.pi) * realErfTaylor7 z| := by
            gcongr
            exact abs_add_le _ _
    _ ≤ (|u| ^ 11 / 11
            + |tanhTaylor9 u - (2 / Real.sqrt Real.pi) * realErfTaylor7 z|)
        + (2 / Real.sqrt Real.pi) * (5 * |z| ^ 9 / 864) := by
            gcongr

/-- Polynomial expansion identity for `tanhTaylor9` (13 nonzero terms x³-x²⁷). -/
private lemma tanhTaylor9_sub_rationalErf_eq (c k x : ℝ) :
    tanhTaylor9 (c * (x + k * (x*x*x))) -
        c * (x - x^3/6 + x^5/40 - x^7/336)
      = (c*k - c^3/3 + c/6) * x^3
          + (-c^3*k + 2*c^5/15 - c/40) * x^5
          + (-c^3*k^2 + 2*c^5*k/3 + c/336 - 17*c^7/315) * x^7
          + (-c^3*k^3/3 + 4*c^5*k^2/3 - 17*c^7*k/45 + 62*c^9/2835) * x^9
          + (4*c^5*k^3/3 - 17*c^7*k^2/15 + 62*c^9*k/315) * x^11
          + (2*c^5*k^4/3 - 17*c^7*k^3/9 + 248*c^9*k^2/315) * x^13
          + (2*c^5*k^5/15 - 17*c^7*k^4/9 + 248*c^9*k^3/135) * x^15
          + (-17*c^7*k^5/15 + 124*c^9*k^4/45) * x^17
          + (-17*c^7*k^6/45 + 124*c^9*k^5/45) * x^19
          + (-17*c^7*k^7/315 + 248*c^9*k^6/135) * x^21
          + (248*c^9*k^7/315) * x^23
          + (62*c^9*k^8/315) * x^25
          + (62*c^9*k^9/2835) * x^27 := by
  unfold tanhTaylor9
  ring

/-- Polynomial difference bound for `|x| ≤ 4/5`. Sum is `≈ 6.14·10⁻⁴`. -/
private lemma poly_diff_bound_08 {x : ℝ} (hx : |x| ≤ 4/5) :
    |tanhTaylor9 ((7978845608028654 / 10000000000000000 : ℝ) *
                    (x + (44715 / 1000000) * (x * x * x))) -
        (7978845608028654 / 10000000000000000 : ℝ) *
            (x - x^3/6 + x^5/40 - x^7/336)|
      ≤ 1 / 1500 := by
  set c : ℝ := 7978845608028654 / 10000000000000000 with hc_def
  set k : ℝ := 44715 / 1000000 with hk_def
  rw [tanhTaylor9_sub_rationalErf_eq c k x]
  have hx_nn : 0 ≤ |x| := abs_nonneg _
  have hx3 : |x|^3 ≤ (4/5)^3 := pow_le_pow_left₀ hx_nn hx 3
  have hx5 : |x|^5 ≤ (4/5)^5 := pow_le_pow_left₀ hx_nn hx 5
  have hx7 : |x|^7 ≤ (4/5)^7 := pow_le_pow_left₀ hx_nn hx 7
  have hx9 : |x|^9 ≤ (4/5)^9 := pow_le_pow_left₀ hx_nn hx 9
  have hx11 : |x|^11 ≤ (4/5)^11 := pow_le_pow_left₀ hx_nn hx 11
  have hx13 : |x|^13 ≤ (4/5)^13 := pow_le_pow_left₀ hx_nn hx 13
  have hx15 : |x|^15 ≤ (4/5)^15 := pow_le_pow_left₀ hx_nn hx 15
  have hx17 : |x|^17 ≤ (4/5)^17 := pow_le_pow_left₀ hx_nn hx 17
  have hx19 : |x|^19 ≤ (4/5)^19 := pow_le_pow_left₀ hx_nn hx 19
  have hx21 : |x|^21 ≤ (4/5)^21 := pow_le_pow_left₀ hx_nn hx 21
  have hx23 : |x|^23 ≤ (4/5)^23 := pow_le_pow_left₀ hx_nn hx 23
  have hx25 : |x|^25 ≤ (4/5)^25 := pow_le_pow_left₀ hx_nn hx 25
  have hx27 : |x|^27 ≤ (4/5)^27 := pow_le_pow_left₀ hx_nn hx 27
  set a3 := c*k - c^3/3 + c/6 with ha3
  set a5 := -c^3*k + 2*c^5/15 - c/40 with ha5
  set a7 := -c^3*k^2 + 2*c^5*k/3 + c/336 - 17*c^7/315 with ha7
  set a9 := -c^3*k^3/3 + 4*c^5*k^2/3 - 17*c^7*k/45 + 62*c^9/2835 with ha9
  set a11 := 4*c^5*k^3/3 - 17*c^7*k^2/15 + 62*c^9*k/315 with ha11
  set a13 := 2*c^5*k^4/3 - 17*c^7*k^3/9 + 248*c^9*k^2/315 with ha13
  set a15 := 2*c^5*k^5/15 - 17*c^7*k^4/9 + 248*c^9*k^3/135 with ha15
  set a17 := -17*c^7*k^5/15 + 124*c^9*k^4/45 with ha17
  set a19 := -17*c^7*k^6/45 + 124*c^9*k^5/45 with ha19
  set a21 := -17*c^7*k^7/315 + 248*c^9*k^6/135 with ha21
  set a23 := 248*c^9*k^7/315 with ha23
  set a25 := 62*c^9*k^8/315 with ha25
  set a27 := 62*c^9*k^9/2835 with ha27
  have hb3 : |a3 * x^3| ≤ |a3| * (4/5)^3 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx3 (abs_nonneg _)
  have hb5 : |a5 * x^5| ≤ |a5| * (4/5)^5 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx5 (abs_nonneg _)
  have hb7 : |a7 * x^7| ≤ |a7| * (4/5)^7 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx7 (abs_nonneg _)
  have hb9 : |a9 * x^9| ≤ |a9| * (4/5)^9 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx9 (abs_nonneg _)
  have hb11 : |a11 * x^11| ≤ |a11| * (4/5)^11 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx11 (abs_nonneg _)
  have hb13 : |a13 * x^13| ≤ |a13| * (4/5)^13 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx13 (abs_nonneg _)
  have hb15 : |a15 * x^15| ≤ |a15| * (4/5)^15 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx15 (abs_nonneg _)
  have hb17 : |a17 * x^17| ≤ |a17| * (4/5)^17 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx17 (abs_nonneg _)
  have hb19 : |a19 * x^19| ≤ |a19| * (4/5)^19 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx19 (abs_nonneg _)
  have hb21 : |a21 * x^21| ≤ |a21| * (4/5)^21 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx21 (abs_nonneg _)
  have hb23 : |a23 * x^23| ≤ |a23| * (4/5)^23 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx23 (abs_nonneg _)
  have hb25 : |a25 * x^25| ≤ |a25| * (4/5)^25 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx25 (abs_nonneg _)
  have hb27 : |a27 * x^27| ≤ |a27| * (4/5)^27 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx27 (abs_nonneg _)
  have hn3 : |a3| * (4/5)^3 ≤ 34 / 100000 := by
    rw [ha3, hc_def, hk_def]; rw [abs_of_neg (by norm_num)]; norm_num
  have hn5 : |a5| * (4/5)^5 ≤ 15 / 100000 := by
    rw [ha5, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn7 : |a7| * (4/5)^7 ≤ 24 / 1000000 := by
    rw [ha7, hc_def, hk_def]; rw [abs_of_neg (by norm_num)]; norm_num
  have hn9 : |a9| * (4/5)^9 ≤ 32 / 1000000 := by
    rw [ha9, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn11 : |a11| * (4/5)^11 ≤ 63 / 1000000 := by
    rw [ha11, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn13 : |a13| * (4/5)^13 ≤ 1 / 100000 := by
    rw [ha13, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn15 : |a15| * (4/5)^15 ≤ 8 / 10000000 := by
    rw [ha15, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn17 : |a17| * (4/5)^17 ≤ 4 / 100000000 := by
    rw [ha17, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn19 : |a19| * (4/5)^19 ≤ 1 / 1000000000 := by
    rw [ha19, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn21 : |a21| * (4/5)^21 ≤ 2 / 100000000000 := by
    rw [ha21, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn23 : |a23| * (4/5)^23 ≤ 3 / 10000000000000 := by
    rw [ha23, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn25 : |a25| * (4/5)^25 ≤ 2 / 1000000000000000 := by
    rw [ha25, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn27 : |a27| * (4/5)^27 ≤ 5 / 1000000000000000000 := by
    rw [ha27, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  calc |a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11 + a13*x^13 + a15*x^15
          + a17*x^17 + a19*x^19 + a21*x^21 + a23*x^23 + a25*x^25 + a27*x^27|
      ≤ |a3*x^3| + |a5*x^5| + |a7*x^7| + |a9*x^9| + |a11*x^11| + |a13*x^13|
          + |a15*x^15| + |a17*x^17| + |a19*x^19| + |a21*x^21| + |a23*x^23|
          + |a25*x^25| + |a27*x^27| := by
        have h1 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11
                               + a13*x^13 + a15*x^15 + a17*x^17 + a19*x^19
                               + a21*x^21 + a23*x^23 + a25*x^25) (a27*x^27)
        have h2 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11
                               + a13*x^13 + a15*x^15 + a17*x^17 + a19*x^19
                               + a21*x^21 + a23*x^23) (a25*x^25)
        have h3 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11
                               + a13*x^13 + a15*x^15 + a17*x^17 + a19*x^19
                               + a21*x^21) (a23*x^23)
        have h4 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11
                               + a13*x^13 + a15*x^15 + a17*x^17 + a19*x^19) (a21*x^21)
        have h5 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11
                               + a13*x^13 + a15*x^15 + a17*x^17) (a19*x^19)
        have h6 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11
                               + a13*x^13 + a15*x^15) (a17*x^17)
        have h7 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11
                               + a13*x^13) (a15*x^15)
        have h8 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11) (a13*x^13)
        have h9 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9) (a11*x^11)
        have h10 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7) (a9*x^9)
        have h11 := abs_add_le (a3*x^3 + a5*x^5) (a7*x^7)
        have h12 := abs_add_le (a3*x^3) (a5*x^5)
        linarith
    _ ≤ |a3| * (4/5)^3 + |a5| * (4/5)^5 + |a7| * (4/5)^7 + |a9| * (4/5)^9
          + |a11| * (4/5)^11 + |a13| * (4/5)^13 + |a15| * (4/5)^15
          + |a17| * (4/5)^17 + |a19| * (4/5)^19 + |a21| * (4/5)^21
          + |a23| * (4/5)^23 + |a25| * (4/5)^25 + |a27| * (4/5)^27 := by
        linarith [hb3, hb5, hb7, hb9, hb11, hb13, hb15, hb17, hb19, hb21, hb23,
                  hb25, hb27]
    _ ≤ 34/100000 + 15/100000 + 24/1000000 + 32/1000000 + 63/1000000 + 1/100000
          + 8/10000000 + 4/100000000 + 1/1000000000 + 2/100000000000
          + 3/10000000000000 + 2/1000000000000000 + 5/1000000000000000000 := by
        linarith [hn3, hn5, hn7, hn9, hn11, hn13, hn15, hn17, hn19, hn21, hn23,
                  hn25, hn27]
    _ ≤ 1/1500 := by norm_num

/-- Combined polynomial bound + `c − √(2/π)` correction for `|x| ≤ 4/5`. -/
private lemma poly_diff_with_correction_bound_08 {x : ℝ} (hx : |x| ≤ 4/5) :
    |tanhTaylor9 ((7978845608028654 / 10000000000000000 : ℝ) *
                    (x + (44715 / 1000000) * (x * x * x))) -
        (2 / Real.sqrt Real.pi) *
            realErfTaylor7 (x / Real.sqrt 2)|
      ≤ 7 / 10000 := by
  have h_rewrite : (2 / Real.sqrt Real.pi) *
                       realErfTaylor7 (x / Real.sqrt 2)
                = Real.sqrt (2 / Real.pi) * (x - x^3/6 + x^5/40 - x^7/336) := by
    show (2 / Real.sqrt Real.pi) *
            ((x / Real.sqrt 2) - (x / Real.sqrt 2)^3/3
              + (x / Real.sqrt 2)^5/10 - (x / Real.sqrt 2)^7/42)
          = Real.sqrt (2 / Real.pi) * (x - x^3/6 + x^5/40 - x^7/336)
    exact two_div_sqrt_pi_realErfTaylor7_eq x
  rw [h_rewrite]
  have h_split :
      tanhTaylor9 ((7978845608028654 / 10000000000000000 : ℝ) *
                      (x + (44715 / 1000000) * (x * x * x))) -
          Real.sqrt (2 / Real.pi) * (x - x^3/6 + x^5/40 - x^7/336)
        = (tanhTaylor9 ((7978845608028654 / 10000000000000000 : ℝ) *
                          (x + (44715 / 1000000) * (x * x * x))) -
            (7978845608028654 / 10000000000000000 : ℝ) *
                (x - x^3/6 + x^5/40 - x^7/336))
          + ((7978845608028654 / 10000000000000000 : ℝ) -
              Real.sqrt (2 / Real.pi)) *
              (x - x^3/6 + x^5/40 - x^7/336) := by
    ring
  rw [h_split]
  have hpoly := poly_diff_bound_08 hx
  have hcorr_sqrt := abs_c_sub_sqrt_two_div_pi_le
  have hx_le_one : |x| ≤ 1 := by linarith
  have hR := R_bound_at_one hx_le_one
  calc |(tanhTaylor9 _ - _ * (x - x^3/6 + x^5/40 - x^7/336))
            + ((7978845608028654 / 10000000000000000 : ℝ)
                - Real.sqrt (2 / Real.pi)) *
                (x - x^3/6 + x^5/40 - x^7/336)|
      ≤ |tanhTaylor9 _ - _ * (x - x^3/6 + x^5/40 - x^7/336)|
        + |((7978845608028654 / 10000000000000000 : ℝ)
              - Real.sqrt (2 / Real.pi)) *
                (x - x^3/6 + x^5/40 - x^7/336)| := abs_add_le _ _
    _ ≤ 1/1500 + (1/10^15) * 2 := by
        gcongr
        rw [abs_mul]
        gcongr
    _ ≤ 7/10000 := by norm_num

/-- For `|x| ≤ 4/5`, `|u| ≤ 67/100`. (True value `≈ 0.657`.) -/
private lemma u_bound_at_08 {x : ℝ} (hx : |x| ≤ 4/5) :
    |(7978845608028654 / 10000000000000000 : ℝ) *
        (x + (44715 / 1000000) * (x * x * x))| ≤ 67 / 100 := by
  rw [abs_mul]
  have hc_pos : (0 : ℝ) < 7978845608028654 / 10000000000000000 := by norm_num
  rw [abs_of_pos hc_pos]
  have h_x3_le : |x|^3 ≤ (4/5)^3 := pow_le_pow_left₀ (abs_nonneg x) hx 3
  have h_inner : |x + (44715 / 1000000 : ℝ) * (x * x * x)|
        ≤ |x| + (44715 / 1000000) * |x|^3 := by
    have hb := abs_add_le x ((44715 / 1000000 : ℝ) * (x * x * x))
    have h_eq : |(44715 / 1000000 : ℝ) * (x * x * x)|
          = (44715 / 1000000) * |x|^3 := by
      rw [abs_mul, show |(44715 / 1000000 : ℝ)| = 44715 / 1000000 from by norm_num]
      congr 1
      rw [show x * x * x = x^3 from by ring, abs_pow]
    linarith
  calc (7978845608028654 / 10000000000000000 : ℝ) *
            |x + (44715 / 1000000) * (x * x * x)|
      ≤ (7978845608028654 / 10000000000000000) *
          (|x| + (44715 / 1000000) * |x|^3) := by
            apply mul_le_mul_of_nonneg_left h_inner hc_pos.le
    _ ≤ (7978845608028654 / 10000000000000000) *
          ((4/5) + (44715 / 1000000) * (4/5)^3) := by
            apply mul_le_mul_of_nonneg_left _ hc_pos.le
            have hk_pos : (0 : ℝ) ≤ 44715 / 1000000 := by norm_num
            have h := mul_le_mul_of_nonneg_left h_x3_le hk_pos
            linarith
    _ ≤ 67/100 := by norm_num

/-- For `|x| ≤ 4/5`, `|u|¹¹/11 ≤ 12/10000`. (True `(67/100)¹¹/11 ≈ 1.11·10⁻³`.) -/
private lemma tanh_residual_at_08 {x : ℝ} (hx : |x| ≤ 4/5) :
    |(7978845608028654 / 10000000000000000 : ℝ) *
        (x + (44715 / 1000000) * (x * x * x))|^11 / 11
      ≤ 12 / 10000 := by
  have hu := u_bound_at_08 hx
  have hu_nn : (0 : ℝ) ≤ |(7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))| := abs_nonneg _
  have hu11 : |(7978845608028654 / 10000000000000000 : ℝ) *
                (x + (44715 / 1000000) * (x * x * x))|^11
              ≤ (67/100)^11 :=
    pow_le_pow_left₀ hu_nn hu 11
  calc |(7978845608028654 / 10000000000000000 : ℝ) *
            (x + (44715 / 1000000) * (x * x * x))|^11 / 11
      ≤ (67/100)^11 / 11 :=
        div_le_div_of_nonneg_right hu11 (by norm_num)
    _ ≤ 12/10000 := by norm_num

/-- For `|x| ≤ 4/5`, `(2/√π)·5·|z|⁹/864 ≤ 7/100000` via the `|z|² ≤ x²/2` trick. -/
private lemma erf_residual_at_08 {x : ℝ} (hx : |x| ≤ 4/5) :
    (2 / Real.sqrt Real.pi) * (5 * |x / Real.sqrt 2|^9 / 864) ≤ 7 / 100000 := by
  have hpi_pos : (0 : ℝ) < Real.pi := Real.pi_pos
  have hsqrt_pi_pos : (0 : ℝ) < Real.sqrt Real.pi := Real.sqrt_pos.mpr hpi_pos
  have hpi_ge : (3/2 : ℝ) ≤ Real.sqrt Real.pi := by
    rw [show (3/2:ℝ) = Real.sqrt ((3/2)^2) from (Real.sqrt_sq (by norm_num)).symm]
    apply Real.sqrt_le_sqrt
    have : ((3/2:ℝ))^2 = 9/4 := by norm_num
    rw [this]; linarith [Real.pi_gt_three]
  have h_inv_pi : (2 : ℝ) / Real.sqrt Real.pi ≤ 4/3 := by
    rw [div_le_div_iff₀ hsqrt_pi_pos (by norm_num : (0:ℝ) < 3)]
    nlinarith [hpi_ge]
  have hsqrt2_pos : (0 : ℝ) < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  have hsqrt2_sq : (Real.sqrt 2)^2 = 2 := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 2)
  have hsqrt2_ge_1 : (1 : ℝ) ≤ Real.sqrt 2 := by
    rw [show (1:ℝ) = Real.sqrt 1 from Real.sqrt_one.symm]
    exact Real.sqrt_le_sqrt (by norm_num)
  have hz_le : |x / Real.sqrt 2| ≤ 4/5 := by
    rw [abs_div, abs_of_pos hsqrt2_pos, div_le_iff₀ hsqrt2_pos]
    nlinarith [hx, hsqrt2_ge_1]
  have hz_nn : (0 : ℝ) ≤ |x / Real.sqrt 2| := abs_nonneg _
  have hz_sq : |x / Real.sqrt 2|^2 = x^2 / 2 := by
    rw [show |x / Real.sqrt 2|^2 = (x / Real.sqrt 2)^2 from (sq_abs _),
        div_pow, hsqrt2_sq]
  have hz_sq_le : |x / Real.sqrt 2|^2 ≤ (4/5)^2 / 2 := by
    rw [hz_sq]
    have h_x_sq : x^2 ≤ (4/5)^2 := by
      rw [show x^2 = |x|^2 from (sq_abs x).symm]
      exact pow_le_pow_left₀ (abs_nonneg x) hx 2
    linarith
  have hz_sq_nn : 0 ≤ |x / Real.sqrt 2|^2 := sq_nonneg _
  have hz8 : |x / Real.sqrt 2|^8 ≤ ((4/5)^2 / 2)^4 := by
    rw [show |x / Real.sqrt 2|^8 = (|x / Real.sqrt 2|^2)^4 from by ring]
    exact pow_le_pow_left₀ hz_sq_nn hz_sq_le 4
  have hz9 : |x / Real.sqrt 2|^9 ≤ (4/5) * ((4/5)^2 / 2)^4 := by
    rw [show |x / Real.sqrt 2|^9 = |x / Real.sqrt 2| * |x / Real.sqrt 2|^8 from by ring]
    exact mul_le_mul hz_le hz8 (pow_nonneg hz_nn 8) (by norm_num)
  calc (2 / Real.sqrt Real.pi) * (5 * |x / Real.sqrt 2|^9 / 864)
      ≤ (4/3 : ℝ) * (5 * ((4/5) * ((4/5)^2 / 2)^4) / 864) := by
        apply mul_le_mul h_inv_pi
        · apply div_le_div_of_nonneg_right
          · exact mul_le_mul_of_nonneg_left hz9 (by norm_num)
          · norm_num
        · positivity
        · norm_num
    _ ≤ 7/100000 := by norm_num

/-- Closure for `|x| ≤ 4/5`: gelu approximation error is at most `1/1000`. -/
private theorem approx_gelu_error_bound_08 {x : ℝ} (hx : |x| ≤ 4/5) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ 1 / 1000 := by
  rw [approxGeLUScalar_sub_exactGeLUScalar x]
  have hx_le_one : |x| ≤ 1 := by linarith
  have h_taylor := gelu_gap_taylor9_decomposition hx_le_one
  have h_tanh := tanh_residual_at_08 hx
  have h_poly := poly_diff_with_correction_bound_08 hx
  have h_erf := erf_residual_at_08 hx
  have h_gap_bound :
      |Real.tanh ((7978845608028654 / 10000000000000000 : ℝ) *
                    (x + (44715 / 1000000) * (x * x * x))) -
          realErf (x / Real.sqrt 2)|
        ≤ 12/10000 + 7/10000 + 7/100000 := by
    calc |Real.tanh _ - realErf _|
        ≤ |(7978845608028654 / 10000000000000000 : ℝ) *
                (x + (44715 / 1000000) * (x * x * x))|^11 / 11
          + |tanhTaylor9 _ - (2 / Real.sqrt Real.pi) * realErfTaylor7 _|
          + (2 / Real.sqrt Real.pi) * (5 * |x / Real.sqrt 2|^9 / 864) :=
            h_taylor
      _ ≤ 12/10000 + 7/10000 + 7/100000 := by linarith
  rw [abs_mul, abs_div, abs_two]
  have hx2_le : |x| / 2 ≤ 2/5 := by linarith
  have hx2_nn : 0 ≤ |x| / 2 := by positivity
  have h_combined : 12/10000 + 7/10000 + (7/100000 : ℝ) ≤ 197/100000 := by norm_num
  calc |x| / 2 *
          |Real.tanh _ - realErf _|
      ≤ |x| / 2 * (12/10000 + 7/10000 + 7/100000) :=
          mul_le_mul_of_nonneg_left h_gap_bound hx2_nn
    _ ≤ |x| / 2 * (197/100000) :=
          mul_le_mul_of_nonneg_left h_combined hx2_nn
    _ ≤ 2/5 * (197/100000) :=
          mul_le_mul_of_nonneg_right hx2_le (by norm_num)
    _ ≤ 1/1000 := by norm_num

/-! ### One more Taylor-certified segment: `|x| ≤ 83/100`

This reuses the 9th-order Taylor decomposition, but retunes every numerical
certificate at `M = 83/100`. It is intentionally a short extension past `4/5`:
at `17/20` the same triangle decomposition no longer has enough budget. -/

/-- Polynomial difference bound for `|x| ≤ 83/100`. Sum is `≈ 7.41·10⁻⁴`. -/
private lemma poly_diff_bound_083 {x : ℝ} (hx : |x| ≤ 83/100) :
    |tanhTaylor9 ((7978845608028654 / 10000000000000000 : ℝ) *
                    (x + (44715 / 1000000) * (x * x * x))) -
        (7978845608028654 / 10000000000000000 : ℝ) *
            (x - x^3/6 + x^5/40 - x^7/336)|
      ≤ 187 / 250000 := by
  set c : ℝ := 7978845608028654 / 10000000000000000 with hc_def
  set k : ℝ := 44715 / 1000000 with hk_def
  rw [tanhTaylor9_sub_rationalErf_eq c k x]
  have hx_nn : 0 ≤ |x| := abs_nonneg _
  have hx3 : |x|^3 ≤ (83/100)^3 := pow_le_pow_left₀ hx_nn hx 3
  have hx5 : |x|^5 ≤ (83/100)^5 := pow_le_pow_left₀ hx_nn hx 5
  have hx7 : |x|^7 ≤ (83/100)^7 := pow_le_pow_left₀ hx_nn hx 7
  have hx9 : |x|^9 ≤ (83/100)^9 := pow_le_pow_left₀ hx_nn hx 9
  have hx11 : |x|^11 ≤ (83/100)^11 := pow_le_pow_left₀ hx_nn hx 11
  have hx13 : |x|^13 ≤ (83/100)^13 := pow_le_pow_left₀ hx_nn hx 13
  have hx15 : |x|^15 ≤ (83/100)^15 := pow_le_pow_left₀ hx_nn hx 15
  have hx17 : |x|^17 ≤ (83/100)^17 := pow_le_pow_left₀ hx_nn hx 17
  have hx19 : |x|^19 ≤ (83/100)^19 := pow_le_pow_left₀ hx_nn hx 19
  have hx21 : |x|^21 ≤ (83/100)^21 := pow_le_pow_left₀ hx_nn hx 21
  have hx23 : |x|^23 ≤ (83/100)^23 := pow_le_pow_left₀ hx_nn hx 23
  have hx25 : |x|^25 ≤ (83/100)^25 := pow_le_pow_left₀ hx_nn hx 25
  have hx27 : |x|^27 ≤ (83/100)^27 := pow_le_pow_left₀ hx_nn hx 27
  set a3 := c*k - c^3/3 + c/6 with ha3
  set a5 := -c^3*k + 2*c^5/15 - c/40 with ha5
  set a7 := -c^3*k^2 + 2*c^5*k/3 + c/336 - 17*c^7/315 with ha7
  set a9 := -c^3*k^3/3 + 4*c^5*k^2/3 - 17*c^7*k/45 + 62*c^9/2835 with ha9
  set a11 := 4*c^5*k^3/3 - 17*c^7*k^2/15 + 62*c^9*k/315 with ha11
  set a13 := 2*c^5*k^4/3 - 17*c^7*k^3/9 + 248*c^9*k^2/315 with ha13
  set a15 := 2*c^5*k^5/15 - 17*c^7*k^4/9 + 248*c^9*k^3/135 with ha15
  set a17 := -17*c^7*k^5/15 + 124*c^9*k^4/45 with ha17
  set a19 := -17*c^7*k^6/45 + 124*c^9*k^5/45 with ha19
  set a21 := -17*c^7*k^7/315 + 248*c^9*k^6/135 with ha21
  set a23 := 248*c^9*k^7/315 with ha23
  set a25 := 62*c^9*k^8/315 with ha25
  set a27 := 62*c^9*k^9/2835 with ha27
  have hb3 : |a3 * x^3| ≤ |a3| * (83/100)^3 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx3 (abs_nonneg _)
  have hb5 : |a5 * x^5| ≤ |a5| * (83/100)^5 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx5 (abs_nonneg _)
  have hb7 : |a7 * x^7| ≤ |a7| * (83/100)^7 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx7 (abs_nonneg _)
  have hb9 : |a9 * x^9| ≤ |a9| * (83/100)^9 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx9 (abs_nonneg _)
  have hb11 : |a11 * x^11| ≤ |a11| * (83/100)^11 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx11 (abs_nonneg _)
  have hb13 : |a13 * x^13| ≤ |a13| * (83/100)^13 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx13 (abs_nonneg _)
  have hb15 : |a15 * x^15| ≤ |a15| * (83/100)^15 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx15 (abs_nonneg _)
  have hb17 : |a17 * x^17| ≤ |a17| * (83/100)^17 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx17 (abs_nonneg _)
  have hb19 : |a19 * x^19| ≤ |a19| * (83/100)^19 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx19 (abs_nonneg _)
  have hb21 : |a21 * x^21| ≤ |a21| * (83/100)^21 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx21 (abs_nonneg _)
  have hb23 : |a23 * x^23| ≤ |a23| * (83/100)^23 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx23 (abs_nonneg _)
  have hb25 : |a25 * x^25| ≤ |a25| * (83/100)^25 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx25 (abs_nonneg _)
  have hb27 : |a27 * x^27| ≤ |a27| * (83/100)^27 := by
    rw [abs_mul, abs_pow]; exact mul_le_mul_of_nonneg_left hx27 (abs_nonneg _)
  have hn3 : |a3| * (83/100)^3 ≤ 38 / 100000 := by
    rw [ha3, hc_def, hk_def]; rw [abs_of_neg (by norm_num)]; norm_num
  have hn5 : |a5| * (83/100)^5 ≤ 18 / 100000 := by
    rw [ha5, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn7 : |a7| * (83/100)^7 ≤ 31 / 1000000 := by
    rw [ha7, hc_def, hk_def]; rw [abs_of_neg (by norm_num)]; norm_num
  have hn9 : |a9| * (83/100)^9 ≤ 45 / 1000000 := by
    rw [ha9, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn11 : |a11| * (83/100)^11 ≤ 94 / 1000000 := by
    rw [ha11, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn13 : |a13| * (83/100)^13 ≤ 16 / 1000000 := by
    rw [ha13, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn15 : |a15| * (83/100)^15 ≤ 13 / 10000000 := by
    rw [ha15, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn17 : |a17| * (83/100)^17 ≤ 6 / 100000000 := by
    rw [ha17, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn19 : |a19| * (83/100)^19 ≤ 2 / 1000000000 := by
    rw [ha19, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn21 : |a21| * (83/100)^21 ≤ 4 / 100000000000 := by
    rw [ha21, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn23 : |a23| * (83/100)^23 ≤ 6 / 10000000000000 := by
    rw [ha23, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn25 : |a25| * (83/100)^25 ≤ 4 / 1000000000000000 := by
    rw [ha25, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  have hn27 : |a27| * (83/100)^27 ≤ 2 / 100000000000000000 := by
    rw [ha27, hc_def, hk_def]; rw [abs_of_pos (by norm_num)]; norm_num
  calc |a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11 + a13*x^13 + a15*x^15
          + a17*x^17 + a19*x^19 + a21*x^21 + a23*x^23 + a25*x^25 + a27*x^27|
      ≤ |a3*x^3| + |a5*x^5| + |a7*x^7| + |a9*x^9| + |a11*x^11| + |a13*x^13|
          + |a15*x^15| + |a17*x^17| + |a19*x^19| + |a21*x^21| + |a23*x^23|
          + |a25*x^25| + |a27*x^27| := by
        have h1 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11
                               + a13*x^13 + a15*x^15 + a17*x^17 + a19*x^19
                               + a21*x^21 + a23*x^23 + a25*x^25) (a27*x^27)
        have h2 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11
                               + a13*x^13 + a15*x^15 + a17*x^17 + a19*x^19
                               + a21*x^21 + a23*x^23) (a25*x^25)
        have h3 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11
                               + a13*x^13 + a15*x^15 + a17*x^17 + a19*x^19
                               + a21*x^21) (a23*x^23)
        have h4 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11
                               + a13*x^13 + a15*x^15 + a17*x^17 + a19*x^19) (a21*x^21)
        have h5 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11
                               + a13*x^13 + a15*x^15 + a17*x^17) (a19*x^19)
        have h6 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11
                               + a13*x^13 + a15*x^15) (a17*x^17)
        have h7 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11
                               + a13*x^13) (a15*x^15)
        have h8 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9 + a11*x^11) (a13*x^13)
        have h9 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7 + a9*x^9) (a11*x^11)
        have h10 := abs_add_le (a3*x^3 + a5*x^5 + a7*x^7) (a9*x^9)
        have h11 := abs_add_le (a3*x^3 + a5*x^5) (a7*x^7)
        have h12 := abs_add_le (a3*x^3) (a5*x^5)
        linarith
    _ ≤ |a3| * (83/100)^3 + |a5| * (83/100)^5 + |a7| * (83/100)^7
          + |a9| * (83/100)^9 + |a11| * (83/100)^11 + |a13| * (83/100)^13
          + |a15| * (83/100)^15 + |a17| * (83/100)^17 + |a19| * (83/100)^19
          + |a21| * (83/100)^21 + |a23| * (83/100)^23 + |a25| * (83/100)^25
          + |a27| * (83/100)^27 := by
        linarith [hb3, hb5, hb7, hb9, hb11, hb13, hb15, hb17, hb19, hb21, hb23,
                  hb25, hb27]
    _ ≤ 38/100000 + 18/100000 + 31/1000000 + 45/1000000 + 94/1000000
          + 16/1000000 + 13/10000000 + 6/100000000 + 2/1000000000
          + 4/100000000000 + 6/10000000000000 + 4/1000000000000000
          + 2/100000000000000000 := by
        linarith [hn3, hn5, hn7, hn9, hn11, hn13, hn15, hn17, hn19, hn21, hn23,
                  hn25, hn27]
    _ ≤ 187/250000 := by norm_num

/-- Combined polynomial bound plus the `c − √(2/π)` correction for `|x| ≤ 83/100`. -/
private lemma poly_diff_with_correction_bound_083 {x : ℝ} (hx : |x| ≤ 83/100) :
    |tanhTaylor9 ((7978845608028654 / 10000000000000000 : ℝ) *
                    (x + (44715 / 1000000) * (x * x * x))) -
        (2 / Real.sqrt Real.pi) *
            realErfTaylor7 (x / Real.sqrt 2)|
      ≤ 3 / 4000 := by
  have h_rewrite : (2 / Real.sqrt Real.pi) *
                       realErfTaylor7 (x / Real.sqrt 2)
                = Real.sqrt (2 / Real.pi) * (x - x^3/6 + x^5/40 - x^7/336) := by
    show (2 / Real.sqrt Real.pi) *
            ((x / Real.sqrt 2) - (x / Real.sqrt 2)^3/3
              + (x / Real.sqrt 2)^5/10 - (x / Real.sqrt 2)^7/42)
          = Real.sqrt (2 / Real.pi) * (x - x^3/6 + x^5/40 - x^7/336)
    exact two_div_sqrt_pi_realErfTaylor7_eq x
  rw [h_rewrite]
  have h_split :
      tanhTaylor9 ((7978845608028654 / 10000000000000000 : ℝ) *
                      (x + (44715 / 1000000) * (x * x * x))) -
          Real.sqrt (2 / Real.pi) * (x - x^3/6 + x^5/40 - x^7/336)
        = (tanhTaylor9 ((7978845608028654 / 10000000000000000 : ℝ) *
                          (x + (44715 / 1000000) * (x * x * x))) -
            (7978845608028654 / 10000000000000000 : ℝ) *
                (x - x^3/6 + x^5/40 - x^7/336))
          + ((7978845608028654 / 10000000000000000 : ℝ) -
              Real.sqrt (2 / Real.pi)) *
              (x - x^3/6 + x^5/40 - x^7/336) := by
    ring
  rw [h_split]
  have hpoly := poly_diff_bound_083 hx
  have hcorr_sqrt := abs_c_sub_sqrt_two_div_pi_le
  have hx_le_one : |x| ≤ 1 := by linarith
  have hR := R_bound_at_one hx_le_one
  calc |(tanhTaylor9 _ - _ * (x - x^3/6 + x^5/40 - x^7/336))
            + ((7978845608028654 / 10000000000000000 : ℝ)
                - Real.sqrt (2 / Real.pi)) *
                (x - x^3/6 + x^5/40 - x^7/336)|
      ≤ |tanhTaylor9 _ - _ * (x - x^3/6 + x^5/40 - x^7/336)|
        + |((7978845608028654 / 10000000000000000 : ℝ)
              - Real.sqrt (2 / Real.pi)) *
                (x - x^3/6 + x^5/40 - x^7/336)| := abs_add_le _ _
    _ ≤ 187/250000 + (1/10^15) * 2 := by
        gcongr
        rw [abs_mul]
        gcongr
    _ ≤ 3/4000 := by norm_num

/-- For `|x| ≤ 83/100`, the gelu tanh argument has `|u| ≤ 683/1000`. -/
private lemma u_bound_at_083 {x : ℝ} (hx : |x| ≤ 83/100) :
    |(7978845608028654 / 10000000000000000 : ℝ) *
        (x + (44715 / 1000000) * (x * x * x))| ≤ 683 / 1000 := by
  rw [abs_mul]
  have hc_pos : (0 : ℝ) < 7978845608028654 / 10000000000000000 := by norm_num
  rw [abs_of_pos hc_pos]
  have h_x3_le : |x|^3 ≤ (83/100)^3 := pow_le_pow_left₀ (abs_nonneg x) hx 3
  have h_inner : |x + (44715 / 1000000 : ℝ) * (x * x * x)|
        ≤ |x| + (44715 / 1000000) * |x|^3 := by
    have hb := abs_add_le x ((44715 / 1000000 : ℝ) * (x * x * x))
    have h_eq : |(44715 / 1000000 : ℝ) * (x * x * x)|
          = (44715 / 1000000) * |x|^3 := by
      rw [abs_mul, show |(44715 / 1000000 : ℝ)| = 44715 / 1000000 from by norm_num]
      congr 1
      rw [show x * x * x = x^3 from by ring, abs_pow]
    linarith
  calc (7978845608028654 / 10000000000000000 : ℝ) *
            |x + (44715 / 1000000) * (x * x * x)|
      ≤ (7978845608028654 / 10000000000000000) *
          (|x| + (44715 / 1000000) * |x|^3) := by
            apply mul_le_mul_of_nonneg_left h_inner hc_pos.le
    _ ≤ (7978845608028654 / 10000000000000000) *
          ((83/100) + (44715 / 1000000) * (83/100)^3) := by
            apply mul_le_mul_of_nonneg_left _ hc_pos.le
            have hk_pos : (0 : ℝ) ≤ 44715 / 1000000 := by norm_num
            have h := mul_le_mul_of_nonneg_left h_x3_le hk_pos
            linarith
    _ ≤ 683/1000 := by norm_num

/-- For `|x| ≤ 83/100`, the 9th-order tanh residual is bounded by `14/10000`. -/
private lemma tanh_residual_at_083 {x : ℝ} (hx : |x| ≤ 83/100) :
    |(7978845608028654 / 10000000000000000 : ℝ) *
        (x + (44715 / 1000000) * (x * x * x))|^11 / 11
      ≤ 14 / 10000 := by
  have hu := u_bound_at_083 hx
  have hu_nn : (0 : ℝ) ≤ |(7978845608028654 / 10000000000000000 : ℝ) *
                  (x + (44715 / 1000000) * (x * x * x))| := abs_nonneg _
  have hu11 : |(7978845608028654 / 10000000000000000 : ℝ) *
                (x + (44715 / 1000000) * (x * x * x))|^11
              ≤ (683/1000)^11 :=
    pow_le_pow_left₀ hu_nn hu 11
  calc |(7978845608028654 / 10000000000000000 : ℝ) *
            (x + (44715 / 1000000) * (x * x * x))|^11 / 11
      ≤ (683/1000)^11 / 11 :=
        div_le_div_of_nonneg_right hu11 (by norm_num)
    _ ≤ 14/10000 := by norm_num

/-- For `|x| ≤ 83/100`, `(2/√π)·5·|z|⁹/864 ≤ 91/1000000`. -/
private lemma erf_residual_at_083 {x : ℝ} (hx : |x| ≤ 83/100) :
    (2 / Real.sqrt Real.pi) * (5 * |x / Real.sqrt 2|^9 / 864) ≤ 91 / 1000000 := by
  have hpi_pos : (0 : ℝ) < Real.pi := Real.pi_pos
  have hsqrt_pi_pos : (0 : ℝ) < Real.sqrt Real.pi := Real.sqrt_pos.mpr hpi_pos
  have hpi_ge : (3/2 : ℝ) ≤ Real.sqrt Real.pi := by
    rw [show (3/2:ℝ) = Real.sqrt ((3/2)^2) from (Real.sqrt_sq (by norm_num)).symm]
    apply Real.sqrt_le_sqrt
    have : ((3/2:ℝ))^2 = 9/4 := by norm_num
    rw [this]; linarith [Real.pi_gt_three]
  have h_inv_pi : (2 : ℝ) / Real.sqrt Real.pi ≤ 4/3 := by
    rw [div_le_div_iff₀ hsqrt_pi_pos (by norm_num : (0:ℝ) < 3)]
    nlinarith [hpi_ge]
  have hsqrt2_pos : (0 : ℝ) < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  have hsqrt2_sq : (Real.sqrt 2)^2 = 2 := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 2)
  have hsqrt2_ge_1 : (1 : ℝ) ≤ Real.sqrt 2 := by
    rw [show (1:ℝ) = Real.sqrt 1 from Real.sqrt_one.symm]
    exact Real.sqrt_le_sqrt (by norm_num)
  have hz_le : |x / Real.sqrt 2| ≤ 83/100 := by
    rw [abs_div, abs_of_pos hsqrt2_pos, div_le_iff₀ hsqrt2_pos]
    nlinarith [hx, hsqrt2_ge_1]
  have hz_nn : (0 : ℝ) ≤ |x / Real.sqrt 2| := abs_nonneg _
  have hz_sq : |x / Real.sqrt 2|^2 = x^2 / 2 := by
    rw [show |x / Real.sqrt 2|^2 = (x / Real.sqrt 2)^2 from (sq_abs _),
        div_pow, hsqrt2_sq]
  have hz_sq_le : |x / Real.sqrt 2|^2 ≤ (83/100)^2 / 2 := by
    rw [hz_sq]
    have h_x_sq : x^2 ≤ (83/100)^2 := by
      rw [show x^2 = |x|^2 from (sq_abs x).symm]
      exact pow_le_pow_left₀ (abs_nonneg x) hx 2
    linarith
  have hz_sq_nn : 0 ≤ |x / Real.sqrt 2|^2 := sq_nonneg _
  have hz8 : |x / Real.sqrt 2|^8 ≤ ((83/100)^2 / 2)^4 := by
    rw [show |x / Real.sqrt 2|^8 = (|x / Real.sqrt 2|^2)^4 from by ring]
    exact pow_le_pow_left₀ hz_sq_nn hz_sq_le 4
  have hz9 : |x / Real.sqrt 2|^9 ≤ (83/100) * ((83/100)^2 / 2)^4 := by
    rw [show |x / Real.sqrt 2|^9 = |x / Real.sqrt 2| * |x / Real.sqrt 2|^8 from by ring]
    exact mul_le_mul hz_le hz8 (pow_nonneg hz_nn 8) (by norm_num)
  calc (2 / Real.sqrt Real.pi) * (5 * |x / Real.sqrt 2|^9 / 864)
      ≤ (4/3 : ℝ) * (5 * ((83/100) * ((83/100)^2 / 2)^4) / 864) := by
        apply mul_le_mul h_inv_pi
        · apply div_le_div_of_nonneg_right
          · exact mul_le_mul_of_nonneg_left hz9 (by norm_num)
          · norm_num
        · positivity
        · norm_num
    _ ≤ 91/1000000 := by norm_num

/-- Closure for `|x| ≤ 83/100`: gelu approximation error is at most `1/1000`. -/
private theorem approx_gelu_error_bound_083 {x : ℝ} (hx : |x| ≤ 83/100) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ 1 / 1000 := by
  rw [approxGeLUScalar_sub_exactGeLUScalar x]
  have hx_le_one : |x| ≤ 1 := by linarith
  have h_taylor := gelu_gap_taylor9_decomposition hx_le_one
  have h_tanh := tanh_residual_at_083 hx
  have h_poly := poly_diff_with_correction_bound_083 hx
  have h_erf := erf_residual_at_083 hx
  have h_gap_bound :
      |Real.tanh ((7978845608028654 / 10000000000000000 : ℝ) *
                    (x + (44715 / 1000000) * (x * x * x))) -
          realErf (x / Real.sqrt 2)|
        ≤ 14/10000 + 3/4000 + 91/1000000 := by
    calc |Real.tanh _ - realErf _|
        ≤ |(7978845608028654 / 10000000000000000 : ℝ) *
                (x + (44715 / 1000000) * (x * x * x))|^11 / 11
          + |tanhTaylor9 _ - (2 / Real.sqrt Real.pi) * realErfTaylor7 _|
          + (2 / Real.sqrt Real.pi) * (5 * |x / Real.sqrt 2|^9 / 864) :=
            h_taylor
      _ ≤ 14/10000 + 3/4000 + 91/1000000 := by linarith
  rw [abs_mul, abs_div, abs_two]
  have hx2_le : |x| / 2 ≤ 83/200 := by linarith
  have hx2_nn : 0 ≤ |x| / 2 := by positivity
  have h_combined :
      14/10000 + 3/4000 + (91/1000000 : ℝ) ≤ 2241/1000000 := by norm_num
  calc |x| / 2 *
          |Real.tanh _ - realErf _|
      ≤ |x| / 2 * (14/10000 + 3/4000 + 91/1000000) :=
          mul_le_mul_of_nonneg_left h_gap_bound hx2_nn
    _ ≤ |x| / 2 * (2241/1000000) :=
          mul_le_mul_of_nonneg_left h_combined hx2_nn
    _ ≤ 83/200 * (2241/1000000) :=
          mul_le_mul_of_nonneg_right hx2_le (by norm_num)
    _ ≤ 1/1000 := by norm_num

/-! ### External Taylor-20 midrange certificate

This experimental certificate expands the combined error
`geluError = approxGeLUScalar - exactGeLUScalar` at the midpoint
`463/200 = 2.315` to degree 20. The standalone certificate theorem is
deliberately limited to the two numeric facts an external CAS/Taylor checker
is expected to certify on
`[83/100, 19/5]`: a remainder bound and a polynomial bound.

External numerical verification (mpmath, 60-digit precision) results:
* `|geluError(x) − poly20(x)|`  worst case ≈ `1.78·10⁻⁷` at `x = 0.83`
  (certificate claims ≤ `1·10⁻⁵`, slack ≈ 98%).
* `|poly20(x)|`                 worst case ≈ `4.7324·10⁻⁴` at
  `x* ≈ 2.6989413864` (a stationary point), certificate claims ≤ `5·10⁻⁴`,
  slack ≈ 5.4%.

Combined: `|geluError(x)| ≤ 1·10⁻⁵ + 5·10⁻⁴ = 51/100000 < 1/1000`. ✓

A standalone Python script reproducing the verification is embedded below
for archival reuse.
```python
# Run with mpmath ≥ 1.3.0; use mp.mp.dps = 50 or higher.
from fractions import Fraction
import mpmath as mp
mp.mp.dps = 50

c_rat = Fraction(7978845608028654, 10**16)
k_rat = Fraction(44715, 10**6)
center = Fraction(463, 200)
sqrt2 = mp.sqrt(2)

coeffs = [
    Fraction(58799578250044, 168873512033139177),
    Fraction(490511376410848, 765861081911736471),
    -Fraction(69595973957346, 98732046708052927),
    -Fraction(318812859576406, 603001279934315985),
    Fraction(95323914281662, 166041057487226167),
    Fraction(83409703730901, 541893669526039966),
    -Fraction(131670005603011, 523709508683584689),
    -Fraction(3688290726053, 972548317670397000),
    Fraction(45774012660025, 681762334304929013),
    -Fraction(7348721185503, 706588070591977423),
    -Fraction(5731745181647, 501999458364499667),
    Fraction(12817416488, 3546237897524503),
    Fraction(1104703829666, 959634314348063215),
    -Fraction(638985140807, 953651943452319460),
    -Fraction(20677130095, 708673697371547448),
    Fraction(23268802103, 302557557620128678),
    -Fraction(3607948043, 331131972419363373),
    -Fraction(2089268122, 454157679889909063),
    Fraction(577867222, 307044885316340023),
    -Fraction(15956207, 139052182026179153),
    -Fraction(30752279, 251067520891536832),
]
coeffs_mp = [mp.mpf(c.numerator) / mp.mpf(c.denominator) for c in coeffs]

def gelu_error(x):
    c = mp.mpf(c_rat.numerator) / mp.mpf(c_rat.denominator)
    k = mp.mpf(k_rat.numerator) / mp.mpf(k_rat.denominator)
    u = c * (x + k * x**3)
    z = x / sqrt2
    return mp.mpf(1)/2 * x * (mp.tanh(u) - mp.erf(z))

def poly20(x):
    t = x - mp.mpf(center.numerator) / mp.mpf(center.denominator)
    return sum(c * t**i for i, c in enumerate(coeffs_mp))

def dpoly20(x):
    t = x - mp.mpf(center.numerator) / mp.mpf(center.denominator)
    return sum(i * coeffs_mp[i] * t**(i-1) for i in range(1, len(coeffs_mp)))

# Locate extrema by sign changes of dpoly20 + endpoints
a, b = mp.mpf("0.83"), mp.mpf("3.8")
N = 100001
xs = [a + (b - a) * i / (N - 1) for i in range(N)]
extrema = [a, b]
prev_d = dpoly20(xs[0])
for i in range(1, N):
    d = dpoly20(xs[i])
    if (prev_d > 0 and d < 0) or (prev_d < 0 and d > 0):
        try:
            extrema.append(mp.findroot(dpoly20, (xs[i-1] + xs[i]) / 2))
        except Exception:
            pass
    prev_d = d

worst_p = max(abs(poly20(x)) for x in extrema)
worst_r = max(abs(gelu_error(x) - poly20(x)) for x in extrema)
assert worst_p <= mp.mpf(50) / 100000, f"poly bound failed: {worst_p}"
assert worst_r <= mp.mpf(1) / 100000, f"residual bound failed: {worst_r}"
print(f"poly20 sup ≈ {float(worst_p):.6e}  (bound 5e-4, slack {float((mp.mpf(50)/100000 - worst_p)/(mp.mpf(50)/100000)*100):.1f}%)")
print(f"residual sup ≈ {float(worst_r):.6e}  (bound 1e-5, slack {float((mp.mpf(1)/100000 - worst_r)/(mp.mpf(1)/100000)*100):.1f}%)")
```
-/

private noncomputable def geluError_mid_taylor20 (x : ℝ) : ℝ :=
  let t := x - 463/200
  58799578250044/168873512033139177
    + (490511376410848/765861081911736471) * t
    + (-(69595973957346/98732046708052927)) * t^2
    + (-(318812859576406/603001279934315985)) * t^3
    + (95323914281662/166041057487226167) * t^4
    + (83409703730901/541893669526039966) * t^5
    + (-(131670005603011/523709508683584689)) * t^6
    + (-(3688290726053/972548317670397000)) * t^7
    + (45774012660025/681762334304929013) * t^8
    + (-(7348721185503/706588070591977423)) * t^9
    + (-(5731745181647/501999458364499667)) * t^10
    + (12817416488/3546237897524503) * t^11
    + (1104703829666/959634314348063215) * t^12
    + (-(638985140807/953651943452319460)) * t^13
    + (-(20677130095/708673697371547448)) * t^14
    + (23268802103/302557557620128678) * t^15
    + (-(3607948043/331131972419363373)) * t^16
    + (-(2089268122/454157679889909063)) * t^17
    + (577867222/307044885316340023) * t^18
    + (-(15956207/139052182026179153)) * t^19
    + (-(30752279/251067520891536832)) * t^20

private theorem geluError_mid_taylor20_cert :
    ∀ x ∈ Set.Icc (83/100 : ℝ) (19/5),
      |geluError x - geluError_mid_taylor20 x| ≤ 1/100000 ∧
      |geluError_mid_taylor20 x| ≤ 50/100000 := by
  intro x hx
  simpa [geluError, geluError_mid_taylor20,
    VeriTile.Math.geluErrorForCert,
    VeriTile.Math.approxGeLUScalarForCert,
    VeriTile.Math.exactGeLUScalarForCert,
    VeriTile.Math.geluError_mid_taylor20_forCert,
    approxGeLUScalar, exactGeLUScalar]
    using VeriTile.Math.geluError_mid_taylor20_cert_standalone x hx

private theorem approx_gelu_error_bound_midrange_taylor20 {x : ℝ}
    (hx : x ∈ Set.Icc (83/100 : ℝ) (19/5)) :
    |geluError x| ≤ 1/1000 := by
  have hcert := geluError_mid_taylor20_cert x hx
  calc |geluError x|
      = |(geluError x - geluError_mid_taylor20 x) + geluError_mid_taylor20 x| := by
          ring_nf
    _ ≤ |geluError x - geluError_mid_taylor20 x| + |geluError_mid_taylor20 x| :=
          abs_add_le _ _
    _ ≤ 1/100000 + 50/100000 := add_le_add hcert.1 hcert.2
    _ ≤ 1/1000 := by norm_num

private theorem approx_gelu_error_bound_medium_taylor20 {x : ℝ}
    (_hxlow : (1/1000 : ℝ) < |x|) (hxhigh : |x| < 19/5) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ 1/1000 := by
  rcases le_or_gt (|x|) (83/100) with h083 | hmid
  · exact approx_gelu_error_bound_083 h083
  · have hx_abs_mem : |x| ∈ Set.Icc (83/100 : ℝ) (19/5) :=
      ⟨le_of_lt hmid, le_of_lt hxhigh⟩
    have hpos := approx_gelu_error_bound_midrange_taylor20 hx_abs_mem
    change |geluError x| ≤ 1 / 1000
    rwa [geluError_abs_arg x] at hpos

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
  have h_inj : Function.Injective
      (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) :=
    injective_offset_singleton (s.pid * blockSize)
  simp [observeAt, exec, approxGeLUKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
        NumericDType.sub, NumericDType.div, BlockState.setReg,
        BlockState.readMem, approxGeLUSpec, approxGeLUScalar]
  unfold InputLoadedAt at _h_x
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_x]

/-- View-level surface for `approx_gelu_kernel_correct`. -/
theorem approx_gelu_kernel_correct_view
    (xReg outReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState)
    (xs : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1)) :
    ∀ idx : TileIndex [blockSize],
      TensorView.observe (exec (approxGeLUKernel xReg outReg blockSize) s)
          (programTileView s outReg blockSize) idx
        = some (approxGeLUSpec xs idx.1) := by
  intro idx
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := blockSize) (xs := xs) h_x
  simpa [TensorView.observe, observeTileAt, programTileView,
         TensorView.offset, Offset.strided, observeAt]
    using approx_gelu_kernel_correct xReg outReg blockSize hN s xs hx idx.1

/-- Target error tolerance for the standard tanh/sigmoid GeLU approximation. -/
noncomputable def approxGeLUEps : ℝ := 1 / 1000

/-- Quantitative gap for the global `approxGeLUEps = 1e-3` bound, restricted
to the medium range `1/1000 < |x| < 19/5`. The proof dispatches on whether
`|x| ≤ 83/100`:

* **`|x| ≤ 83/100`** — closed by `approx_gelu_error_bound_083`, which applies
  the 9th-order tanh Taylor decomposition `gelu_gap_taylor9_decomposition`
  with explicit numerical bounds on each piece.
* **Mid `83/100 < |x| < 19/5`** — closed by the standalone degree-20 Taylor
  certificate from `VeriTile.Math.GeluTaylor20Cert` at midpoint `463/200`. -/
theorem approx_gelu_error_bound_medium {x : ℝ}
    (_hxlow : approxGeLUEps < |x|) (hxhigh : |x| < 19/5) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ approxGeLUEps := by
  exact approx_gelu_error_bound_medium_taylor20 _hxlow hxhigh

/-- The `|x| > 1/1000` half of `approx_gelu_error_bound`. Dispatches via
`approx_gelu_error_bound_tail_19_5` for `|x| ≥ 19/5` and
`approx_gelu_error_bound_medium` for `1/1000 < |x| < 19/5`. -/
theorem approx_gelu_error_bound_large {x : ℝ} (hx : approxGeLUEps < |x|) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ approxGeLUEps := by
  rcases lt_or_ge (|x|) (19/5) with hmedium | hlarge
  · exact approx_gelu_error_bound_medium hx hmedium
  · -- `|x| ≥ 19/5`: discharge from the sharpened asymptotic tail bound.
    have h := approx_gelu_error_bound_tail_19_5 hlarge
    show |approxGeLUScalar x - exactGeLUScalar x| ≤ 1 / 1000
    exact h

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
