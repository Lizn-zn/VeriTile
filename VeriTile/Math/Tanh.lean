/-
VeriTile.Math.Tanh

Foundational quantitative bounds for `Real.tanh` that this snapshot of mathlib
(v4.29.0) does not provide directly: derivative of `tanh`, continuity, the
linear bound `|tanh z| ≤ |z|`, and the cubic bound `|tanh z − z| ≤ |z|³/3`.

These are the entry points for the (still in-progress) Taylor-style argument
required to close `approx_gelu_error_bound_medium` over `1/1000 < |x| < 20`.
Higher-order bounds (5th, 7th order) are TODO — each one requires another
integral-by-parts or derivative-iteration step.
-/

import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic

namespace VeriTile.Math

open Real

/-- The derivative of `tanh` is `1 − tanh²`. Direct from the quotient rule on
`tanh = sinh / cosh` plus `cosh² − sinh² = 1`. -/
theorem tanh_hasDerivAt (z : ℝ) :
    HasDerivAt Real.tanh (1 - Real.tanh z ^ 2) z := by
  have h_quot :=
    (Real.hasDerivAt_sinh z).div (Real.hasDerivAt_cosh z) (Real.cosh_pos z).ne'
  have h_eq : (Real.sinh / Real.cosh) = Real.tanh := by
    funext z; exact (Real.tanh_eq_sinh_div_cosh z).symm
  rw [h_eq] at h_quot
  have h_val :
      (Real.cosh z * Real.cosh z - Real.sinh z * Real.sinh z) / Real.cosh z ^ 2
        = 1 - Real.tanh z ^ 2 := by
    rw [Real.tanh_eq_sinh_div_cosh, div_pow]
    have hcsq : Real.cosh z ^ 2 ≠ 0 := by positivity
    have hkey : Real.cosh z ^ 2 - Real.sinh z ^ 2 = 1 := Real.cosh_sq_sub_sinh_sq z
    field_simp
  rw [h_val] at h_quot
  exact h_quot

theorem continuous_tanh : Continuous Real.tanh := by
  rw [show Real.tanh = (fun z => Real.sinh z / Real.cosh z) from
        funext fun z => Real.tanh_eq_sinh_div_cosh z]
  exact Real.continuous_sinh.div Real.continuous_cosh
    (fun x => (Real.cosh_pos x).ne')

/-- Linear bound: `|tanh z| ≤ |z|` for all real `z`. Proof via the mean value
inequality applied to `tanh` with derivative `1 − tanh² ∈ [0, 1]`. -/
theorem abs_tanh_le_abs (z : ℝ) : |Real.tanh z| ≤ |z| := by
  have h_diff : ∀ x ∈ (Set.univ : Set ℝ), DifferentiableAt ℝ Real.tanh x :=
    fun x _ => (tanh_hasDerivAt x).differentiableAt
  have h_bound : ∀ x ∈ (Set.univ : Set ℝ), ‖deriv Real.tanh x‖ ≤ 1 := by
    intro x _
    rw [(tanh_hasDerivAt x).deriv]
    have h1 : Real.tanh x ^ 2 ≤ 1 := by
      have := Real.tanh_lt_one x
      have := Real.neg_one_lt_tanh x
      nlinarith [sq_nonneg (Real.tanh x - 1), sq_nonneg (Real.tanh x + 1)]
    rw [Real.norm_eq_abs, abs_le]
    constructor <;> nlinarith [sq_nonneg (Real.tanh x)]
  have h := Convex.norm_image_sub_le_of_norm_deriv_le h_diff h_bound
    convex_univ (Set.mem_univ 0) (Set.mem_univ z)
  rw [Real.tanh_zero] at h
  simpa using h

/-- Cubic bound: `|tanh z − z| ≤ |z|³ / 3`. Proved by writing
`tanh z − z = −∫_0^z tanh²(t) dt` (FTC + `tanh' = 1 − tanh²`) and bounding
the integrand pointwise via `tanh²(t) ≤ t²` (corollary of `abs_tanh_le_abs`). -/
theorem abs_tanh_sub_self_le (z : ℝ) :
    |Real.tanh z - z| ≤ |z| ^ 3 / 3 := by
  have h_int_repr : Real.tanh z - z = - ∫ t in (0)..z, Real.tanh t ^ 2 := by
    have h_FTC :
        ∫ t in (0)..z, (1 - Real.tanh t ^ 2) = Real.tanh z - Real.tanh 0 := by
      apply intervalIntegral.integral_eq_sub_of_hasDerivAt
      · intro x _; exact tanh_hasDerivAt x
      · exact (continuous_const.sub (continuous_tanh.pow 2)).intervalIntegrable _ _
    have h_split : ∫ t in (0)..z, (1 - Real.tanh t ^ 2)
          = (∫ t in (0)..z, (1:ℝ)) - ∫ t in (0)..z, Real.tanh t ^ 2 := by
      rw [intervalIntegral.integral_sub]
      · exact intervalIntegrable_const
      · exact (continuous_tanh.pow 2).intervalIntegrable _ _
    have h_ones : ∫ t in (0)..z, (1:ℝ) = z := by simp
    rw [h_split, h_ones, Real.tanh_zero] at h_FTC
    linarith
  rw [h_int_repr, abs_neg]
  have h_pointwise : ∀ t : ℝ, Real.tanh t ^ 2 ≤ t ^ 2 := by
    intro t
    have h_abs := abs_tanh_le_abs t
    rw [show Real.tanh t ^ 2 = |Real.tanh t| ^ 2 from (sq_abs _).symm,
        show t ^ 2 = |t| ^ 2 from (sq_abs _).symm]
    exact pow_le_pow_left₀ (abs_nonneg _) h_abs 2
  have h_compare :
      |∫ t in (0)..z, Real.tanh t ^ 2| ≤ |∫ t in (0)..z, t ^ 2| := by
    rcases le_or_gt 0 z with hz | hz
    · rw [intervalIntegral.integral_of_le hz, intervalIntegral.integral_of_le hz]
      have h_lower_a : (0:ℝ) ≤ ∫ t in Set.Ioc 0 z, Real.tanh t ^ 2 :=
        MeasureTheory.setIntegral_nonneg measurableSet_Ioc (fun _ _ => sq_nonneg _)
      have h_lower_b : (0:ℝ) ≤ ∫ t in Set.Ioc 0 z, t ^ 2 :=
        MeasureTheory.setIntegral_nonneg measurableSet_Ioc (fun _ _ => sq_nonneg _)
      rw [abs_of_nonneg h_lower_a, abs_of_nonneg h_lower_b]
      exact MeasureTheory.setIntegral_mono_on
        (continuous_tanh.pow 2).integrableOn_Ioc (continuous_id.pow 2).integrableOn_Ioc
        measurableSet_Ioc (fun t _ => h_pointwise t)
    · rw [intervalIntegral.integral_of_ge hz.le, intervalIntegral.integral_of_ge hz.le,
          abs_neg, abs_neg]
      have h_lower_a : (0:ℝ) ≤ ∫ t in Set.Ioc z 0, Real.tanh t ^ 2 :=
        MeasureTheory.setIntegral_nonneg measurableSet_Ioc (fun _ _ => sq_nonneg _)
      have h_lower_b : (0:ℝ) ≤ ∫ t in Set.Ioc z 0, t ^ 2 :=
        MeasureTheory.setIntegral_nonneg measurableSet_Ioc (fun _ _ => sq_nonneg _)
      rw [abs_of_nonneg h_lower_a, abs_of_nonneg h_lower_b]
      exact MeasureTheory.setIntegral_mono_on
        (continuous_tanh.pow 2).integrableOn_Ioc (continuous_id.pow 2).integrableOn_Ioc
        measurableSet_Ioc (fun t _ => h_pointwise t)
  have h_sq_int : ∫ t in (0)..z, t ^ 2 = z ^ 3 / 3 := by
    rw [integral_pow]; norm_num
  have h_abs_z3 : |∫ t in (0)..z, t ^ 2| = |z| ^ 3 / 3 := by
    rw [h_sq_int, abs_div, show |(3:ℝ)| = 3 from by norm_num, abs_pow]
  linarith [h_compare]

end VeriTile.Math
