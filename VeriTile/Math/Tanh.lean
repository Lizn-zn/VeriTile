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

/-- Quintic bound: `|tanh z − (z − z³/3)| ≤ 2|z|⁵ / 15`. Proved by writing
`tanh z − z + z³/3 = ∫_0^z (t² − tanh²(t)) dt` (FTC + `tanh' = 1 − tanh²`)
and bounding the integrand pointwise as
`0 ≤ t² − tanh²(t) ≤ 2 · t⁴ / 3` using `|tanh t| ≤ |t|` and the cubic bound
`|tanh t − t| ≤ |t|³/3`. -/
theorem abs_tanh_sub_taylor3_le (z : ℝ) :
    |Real.tanh z - (z - z ^ 3 / 3)| ≤ 2 * |z| ^ 5 / 15 := by
  have h_int_repr :
      Real.tanh z - z + z ^ 3 / 3
        = ∫ t in (0)..z, (t ^ 2 - Real.tanh t ^ 2) := by
    have h_g_deriv : ∀ t : ℝ,
        HasDerivAt (fun u => Real.tanh u - u + u ^ 3 / 3)
          (t ^ 2 - Real.tanh t ^ 2) t := by
      intro t
      have h1 := tanh_hasDerivAt t
      have h2 : HasDerivAt (fun u : ℝ => u) 1 t := hasDerivAt_id t
      have h3 : HasDerivAt (fun u : ℝ => u ^ 3 / 3) (t ^ 2) t := by
        have := (hasDerivAt_pow 3 t).div_const 3
        convert this using 1; push_cast; ring
      have h_combined := (h1.sub h2).add h3
      convert h_combined using 1; ring
    have h_FTC : ∫ t in (0)..z, (t ^ 2 - Real.tanh t ^ 2)
        = (Real.tanh z - z + z ^ 3 / 3) - (Real.tanh 0 - 0 + 0 ^ 3 / 3) :=
      intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => h_g_deriv x)
        (((continuous_id.pow 2).sub (continuous_tanh.pow 2)).intervalIntegrable _ _)
    rw [Real.tanh_zero] at h_FTC
    linarith
  have h_lower : ∀ t : ℝ, 0 ≤ t ^ 2 - Real.tanh t ^ 2 := by
    intro t
    have h_abs := abs_tanh_le_abs t
    rw [show Real.tanh t ^ 2 = |Real.tanh t| ^ 2 from (sq_abs _).symm,
        show t ^ 2 = |t| ^ 2 from (sq_abs _).symm]
    have := pow_le_pow_left₀ (abs_nonneg _) h_abs 2
    linarith
  have h_upper : ∀ t : ℝ, t ^ 2 - Real.tanh t ^ 2 ≤ 2 * t ^ 4 / 3 := by
    intro t
    have h_abs := abs_tanh_le_abs t
    have h_cubic := abs_tanh_sub_self_le t
    have h_factor :
        t ^ 2 - Real.tanh t ^ 2 = (t - Real.tanh t) * (t + Real.tanh t) := by ring
    rw [h_factor]
    have h1 : |t - Real.tanh t| ≤ |t| ^ 3 / 3 := by
      rw [show t - Real.tanh t = -(Real.tanh t - t) from by ring, abs_neg]
      exact h_cubic
    have h2 : |t + Real.tanh t| ≤ 2 * |t| := by
      have := abs_add_le t (Real.tanh t); linarith
    have h_prod : |(t - Real.tanh t) * (t + Real.tanh t)| ≤ 2 * |t| ^ 4 / 3 := by
      rw [abs_mul]
      have := mul_le_mul h1 h2 (abs_nonneg _)
        (by positivity : (0 : ℝ) ≤ |t| ^ 3 / 3)
      have h_eq : |t| ^ 3 / 3 * (2 * |t|) = 2 * |t| ^ 4 / 3 := by ring
      linarith
    have h_t4 : |t| ^ 4 = t ^ 4 := by
      rw [show (4 : ℕ) = 2 * 2 from rfl, pow_mul, pow_mul, sq_abs]
    rw [h_t4] at h_prod
    exact (le_abs_self _).trans h_prod
  rw [show Real.tanh z - (z - z ^ 3 / 3) = Real.tanh z - z + z ^ 3 / 3 from by ring,
      h_int_repr]
  rcases le_or_gt 0 z with hz | hz
  · rw [intervalIntegral.integral_of_le hz]
    have h_abs_eq : |∫ t in Set.Ioc 0 z, t ^ 2 - Real.tanh t ^ 2|
        = ∫ t in Set.Ioc 0 z, t ^ 2 - Real.tanh t ^ 2 :=
      abs_of_nonneg (MeasureTheory.setIntegral_nonneg measurableSet_Ioc
        (fun t _ => h_lower t))
    rw [h_abs_eq]
    have h_compare :
        ∫ t in Set.Ioc 0 z, (t ^ 2 - Real.tanh t ^ 2)
          ≤ ∫ t in Set.Ioc 0 z, 2 * t ^ 4 / 3 :=
      MeasureTheory.setIntegral_mono_on
        ((continuous_id.pow 2).sub (continuous_tanh.pow 2)).integrableOn_Ioc
        ((continuous_const.mul (continuous_id.pow 4)).div_const _).integrableOn_Ioc
        measurableSet_Ioc (fun t _ => h_upper t)
    have h_compute : ∫ t in Set.Ioc 0 z, 2 * t ^ 4 / 3 = 2 * z ^ 5 / 15 := by
      have hrw : ∫ t in Set.Ioc 0 z, 2 * t ^ 4 / 3 = ∫ t in (0)..z, 2 * t ^ 4 / 3 := by
        rw [intervalIntegral.integral_of_le hz]
      rw [hrw]
      have h_split :
          ∫ t in (0)..z, 2 * t ^ 4 / 3 = (2/3) * ∫ t in (0)..z, t ^ 4 := by
        have : ∫ t in (0)..z, 2 * t ^ 4 / 3 = ∫ t in (0)..z, (2/3) * t ^ 4 := by
          apply intervalIntegral.integral_congr
          intro t _; ring
        rw [this, intervalIntegral.integral_const_mul]
      rw [h_split, integral_pow]; ring
    have h_z5 : |z| ^ 5 = z ^ 5 := by
      rw [show |z| ^ 5 = |z ^ 5| from by rw [abs_pow]]
      exact abs_of_nonneg (by positivity)
    rw [h_z5]; linarith
  · rw [intervalIntegral.integral_of_ge hz.le, abs_neg]
    have h_abs_eq : |∫ t in Set.Ioc z 0, t ^ 2 - Real.tanh t ^ 2|
        = ∫ t in Set.Ioc z 0, t ^ 2 - Real.tanh t ^ 2 :=
      abs_of_nonneg (MeasureTheory.setIntegral_nonneg measurableSet_Ioc
        (fun t _ => h_lower t))
    rw [h_abs_eq]
    have h_compare :
        ∫ t in Set.Ioc z 0, (t ^ 2 - Real.tanh t ^ 2)
          ≤ ∫ t in Set.Ioc z 0, 2 * t ^ 4 / 3 :=
      MeasureTheory.setIntegral_mono_on
        ((continuous_id.pow 2).sub (continuous_tanh.pow 2)).integrableOn_Ioc
        ((continuous_const.mul (continuous_id.pow 4)).div_const _).integrableOn_Ioc
        measurableSet_Ioc (fun t _ => h_upper t)
    have h_compute : ∫ t in Set.Ioc z 0, 2 * t ^ 4 / 3 = -2 * z ^ 5 / 15 := by
      have hrw : ∫ t in Set.Ioc z 0, 2 * t ^ 4 / 3 = ∫ t in z..0, 2 * t ^ 4 / 3 := by
        rw [intervalIntegral.integral_of_le hz.le]
      rw [hrw]
      have h_split :
          ∫ t in z..(0:ℝ), 2 * t ^ 4 / 3 = (2/3) * ∫ t in z..(0:ℝ), t ^ 4 := by
        have : ∫ t in z..(0:ℝ), 2 * t ^ 4 / 3
            = ∫ t in z..(0:ℝ), (2/3) * t ^ 4 := by
          apply intervalIntegral.integral_congr
          intro t _; ring
        rw [this, intervalIntegral.integral_const_mul]
      rw [h_split, integral_pow]; ring
    have h_z5 : |z| ^ 5 = -z ^ 5 := by
      rw [show |z| ^ 5 = |z ^ 5| from by rw [abs_pow]]
      apply abs_of_neg
      have : z ^ 5 = z * z ^ 4 := by ring
      rw [this]
      have h_z4_pos : 0 < z ^ 4 := by
        have hz_ne : z ≠ 0 := ne_of_lt hz
        positivity
      exact mul_neg_of_neg_of_pos hz h_z4_pos
    rw [h_z5]; linarith

end VeriTile.Math
