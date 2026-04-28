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

/-- Closure for the very-large case (both signs): for `|x| ≥ 20` the gelu
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

/-- Quantitative gap remaining for the global `approxGeLUEps = 1e-3` bound,
restricted to the medium range `1/1000 < |x| < 20`.

The global `|x| > 1/1000` case splits into two pieces:

* **Very-large `|x| ≥ 20`** — closed by `approx_gelu_error_bound_very_large`,
  which combines `abs_approxGeLUScalar_sub_exactGeLUScalar_tail_pos` with the
  monotonicity helper `x_exp_neg_ax_mono` and the concrete numeric bound
  `gelu_tail_at_twenty`.
* **Medium `1/1000 < |x| < 20`** — the remaining open analytic content,
  carried by this theorem. Closing it requires either certified interval
  arithmetic across `[1/1000, 20]` or a Taylor-with-remainder analysis using
  `realErf_hasDerivAt`. The asymptotic tail bounds are too loose on this
  bounded window to give `≤ 1/1000`. Out of scope for the current snapshot. -/
theorem approx_gelu_error_bound_medium {x : ℝ}
    (_hxlow : approxGeLUEps < |x|) (_hxhigh : |x| < 20) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ approxGeLUEps := by
  -- Medium range: the gap is at its empirical max here; tail bounds
  -- are insufficient. Requires certified numerics or Taylor analysis.
  sorry

/-- The `|x| > 1/1000` half of `approx_gelu_error_bound`. Dispatches via
`approx_gelu_error_bound_very_large` for `|x| ≥ 20` and
`approx_gelu_error_bound_medium` for `1/1000 < |x| < 20`. -/
theorem approx_gelu_error_bound_large {x : ℝ} (hx : approxGeLUEps < |x|) :
    |approxGeLUScalar x - exactGeLUScalar x| ≤ approxGeLUEps := by
  rcases lt_or_ge (|x|) 20 with hmedium | hlarge
  · exact approx_gelu_error_bound_medium hx hmedium
  · -- `|x| ≥ 20`: discharge from the asymptotic tail bound.
    have h := approx_gelu_error_bound_very_large hlarge
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
