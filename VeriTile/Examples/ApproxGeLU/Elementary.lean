/-
VeriTile.Examples.ApproxGeLU.Elementary

Split-out support for the ApproxGeLU example.
-/

import VeriTile.Examples.ApproxGeLU.Kernel

namespace VeriTile.Examples

open VeriTile.Triton VeriTile.Math

lemma two_sigmoid_two_sub_one_eq_tanh (u : ℝ) :
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
lemma one_sub_tanh_le_two_exp_neg_two_mul (u : ℝ) :
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
lemma tanh_add_one_le_two_exp_two_mul (u : ℝ) :
    Real.tanh u + 1 ≤ 2 * Real.exp (2 * u) := by
  have h := one_sub_tanh_le_two_exp_neg_two_mul (-u)
  rw [Real.tanh_neg] at h
  rw [show -(2 * -u) = 2 * u from by ring] at h
  linarith

lemma abs_tanh_le_one (u : ℝ) : |Real.tanh u| ≤ 1 := by
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
numerics on a single (large) value of `x`, would close
`approx_gelu_error_bound_large` on the half-line `[T, ∞)` for any
`T ≥ √2` chosen large enough that the RHS is `≤ 1/1000`. -/

lemma c_pos : (7978845608028654 / 10000000000000000 : ℝ) > 0 := by norm_num

lemma k_pos : (44715 / 1000000 : ℝ) > 0 := by norm_num

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
lemma abs_approxGeLUScalar_sub_exactGeLUScalar_tail_pos_four
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
lemma abs_approxGeLUScalar_sub_exactGeLUScalar_tail_pos_19_5
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


end VeriTile.Examples
