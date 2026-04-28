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

/-- Pointwise bound on the seventh-order Taylor remainder integrand:
`|t² − tanh²(t) − 2t⁴/3| ≤ t⁶` for `|t| ≤ 1`. Derived by writing
`tanh t = (t − t³/3) + ε(t)` with `|ε(t)| ≤ 2|t|⁵/15` (the quintic bound),
expanding `tanh²(t)`, and using `|1 − t²/3| ≤ 1` plus `|t|¹⁰ ≤ |t|⁶` on the
unit ball. The numeric constant `89/225 < 1` provides the `≤ t⁶` envelope. -/
private lemma tanh_seventh_integrand_bound {t : ℝ} (ht : |t| ≤ 1) :
    |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3| ≤ t ^ 6 := by
  have h_quintic := abs_tanh_sub_taylor3_le t
  set ε : ℝ := Real.tanh t - (t - t ^ 3 / 3) with hε_def
  have h_ε_bound : |ε| ≤ 2 * |t| ^ 5 / 15 := by rw [hε_def]; exact h_quintic
  have h_tanh : Real.tanh t = (t - t ^ 3 / 3) + ε := by rw [hε_def]; ring
  have h_eq : t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3
        = -(t ^ 6 / 9) - 2 * (t - t ^ 3 / 3) * ε - ε ^ 2 := by rw [h_tanh]; ring
  rw [h_eq]
  have h_t_bound : |t - t ^ 3 / 3| ≤ |t| := by
    rw [show t - t ^ 3 / 3 = t * (1 - t ^ 2 / 3) from by ring, abs_mul]
    have h_sub : |1 - t ^ 2 / 3| ≤ 1 := by
      rw [abs_le]
      have h_t2_le : t ^ 2 ≤ 1 := by
        rw [show t ^ 2 = |t| ^ 2 from (sq_abs _).symm]
        exact pow_le_one₀ (abs_nonneg _) ht
      constructor <;> nlinarith [sq_nonneg t]
    nlinarith [abs_nonneg t, h_sub]
  have h_t_eq_t6 : t ^ 6 = |t| ^ 6 := by
    rw [show (6 : ℕ) = 2 * 3 from rfl, pow_mul, pow_mul, sq_abs]
  have h_t10_le : |t| ^ 10 ≤ |t| ^ 6 := by
    have h_t2_le : |t| ^ 2 ≤ 1 := pow_le_one₀ (abs_nonneg _) ht
    nlinarith [sq_nonneg (|t| ^ 4 - |t| ^ 2), sq_nonneg (|t| ^ 2),
               sq_nonneg (|t| ^ 3), abs_nonneg t]
  calc |-(t ^ 6 / 9) - 2 * (t - t ^ 3 / 3) * ε - ε ^ 2|
      ≤ |t ^ 6 / 9| + |2 * (t - t ^ 3 / 3) * ε| + |ε ^ 2| := by
        have hr : -(t ^ 6 / 9) - 2 * (t - t ^ 3 / 3) * ε - ε ^ 2
                = (-(t ^ 6 / 9) + (-(2 * (t - t ^ 3 / 3) * ε))) + (-(ε ^ 2)) := by ring
        rw [hr]
        calc _ ≤ |(-(t ^ 6 / 9) + (-(2 * (t - t ^ 3 / 3) * ε)))| + |(-(ε ^ 2))| :=
                abs_add_le _ _
          _ ≤ (|(-(t ^ 6 / 9))| + |(-(2 * (t - t ^ 3 / 3) * ε))|) + |(-(ε ^ 2))| := by
              gcongr; exact abs_add_le _ _
          _ = |t ^ 6 / 9| + |2 * (t - t ^ 3 / 3) * ε| + |ε ^ 2| := by
              rw [abs_neg, abs_neg, abs_neg]
    _ = |t| ^ 6 / 9 + 2 * |t - t ^ 3 / 3| * |ε| + ε ^ 2 := by
        rw [abs_div, abs_mul, abs_mul]
        rw [show |t ^ 6| = t ^ 6 from abs_of_nonneg (by positivity), h_t_eq_t6]
        rw [show |(9 : ℝ)| = 9 from by norm_num,
            show |(2 : ℝ)| = 2 from by norm_num,
            show |ε ^ 2| = ε ^ 2 from abs_of_nonneg (sq_nonneg _)]
    _ ≤ |t| ^ 6 / 9 + 2 * |t| * (2 * |t| ^ 5 / 15) + (2 * |t| ^ 5 / 15) ^ 2 := by
        have h_eps_sq : ε ^ 2 ≤ (2 * |t| ^ 5 / 15) ^ 2 := by
          rw [show ε ^ 2 = |ε| ^ 2 from (sq_abs _).symm]
          exact pow_le_pow_left₀ (abs_nonneg _) h_ε_bound 2
        have h_mid : 2 * |t - t ^ 3 / 3| * |ε|
              ≤ 2 * |t| * (2 * |t| ^ 5 / 15) :=
          mul_le_mul (mul_le_mul_of_nonneg_left h_t_bound (by norm_num))
            h_ε_bound (abs_nonneg _) (by positivity)
        linarith
    _ = |t| ^ 6 * (1 / 9 + 4 / 15) + 4 * |t| ^ 10 / 225 := by ring
    _ ≤ |t| ^ 6 * (1 / 9 + 4 / 15) + 4 * |t| ^ 6 / 225 := by
        have h_diff : 4 * |t| ^ 10 / 225 ≤ 4 * |t| ^ 6 / 225 := by
          nlinarith [h_t10_le]
        linarith
    _ = |t| ^ 6 * (89 / 225) := by ring
    _ ≤ |t| ^ 6 := by
        have : 0 ≤ |t| ^ 6 := by positivity
        linarith
    _ = t ^ 6 := h_t_eq_t6.symm

/-- Septenary bound: `|tanh z − (z − z³/3 + 2z⁵/15)| ≤ |z|⁷ / 7` for `|z| ≤ 1`.

This is the third iterated-integral step in the Taylor tower for `tanh`.
The proof: define `g(z) = tanh z − z + z³/3 − 2z⁵/15`; then
`g'(z) = z² − tanh²(z) − 2z⁴/3` (FTC + previous tanh derivatives), and via
the quintic bound `|g'(t)| ≤ t⁶` on `|t| ≤ 1`. Integration gives
`|g(z)| ≤ ∫_0^|z| t⁶ dt = |z|⁷/7`. -/
theorem abs_tanh_sub_taylor5_le {z : ℝ} (hz : |z| ≤ 1) :
    |Real.tanh z - (z - z ^ 3 / 3 + 2 * z ^ 5 / 15)| ≤ |z| ^ 7 / 7 := by
  have h_int_repr :
      Real.tanh z - z + z ^ 3 / 3 - 2 * z ^ 5 / 15
        = ∫ t in (0)..z, (t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3) := by
    have h_g_deriv : ∀ t : ℝ,
        HasDerivAt (fun u => Real.tanh u - u + u ^ 3 / 3 - 2 * u ^ 5 / 15)
          (t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3) t := by
      intro t
      have h1 := tanh_hasDerivAt t
      have h2 : HasDerivAt (fun u : ℝ => u) 1 t := hasDerivAt_id t
      have h3 : HasDerivAt (fun u : ℝ => u ^ 3 / 3) (t ^ 2) t := by
        have := (hasDerivAt_pow 3 t).div_const 3
        convert this using 1; push_cast; ring
      have h4 : HasDerivAt (fun u : ℝ => 2 * u ^ 5 / 15) (2 * t ^ 4 / 3) t := by
        have h_div := ((hasDerivAt_pow 5 t).const_mul 2).div_const 15
        convert h_div using 1; push_cast; ring
      have h_combined := ((h1.sub h2).add h3).sub h4
      convert h_combined using 1; ring
    have h_FTC :
        ∫ t in (0)..z, (t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3)
          = (Real.tanh z - z + z ^ 3 / 3 - 2 * z ^ 5 / 15)
              - (Real.tanh 0 - 0 + 0 ^ 3 / 3 - 2 * 0 ^ 5 / 15) :=
      intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => h_g_deriv x)
        ((((continuous_id.pow 2).sub (continuous_tanh.pow 2)).sub
          ((continuous_const.mul (continuous_id.pow 4)).div_const _)).intervalIntegrable _ _)
    rw [Real.tanh_zero] at h_FTC
    linarith
  rw [show Real.tanh z - (z - z ^ 3 / 3 + 2 * z ^ 5 / 15)
        = Real.tanh z - z + z ^ 3 / 3 - 2 * z ^ 5 / 15 from by ring, h_int_repr]
  rcases le_or_gt 0 z with hz_nn | hz_neg
  · rw [intervalIntegral.integral_of_le hz_nn]
    have hz_eq : |z| = z := abs_of_nonneg hz_nn
    have h_pointwise :
        ∀ t ∈ Set.Ioc 0 z, |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3| ≤ t ^ 6 := by
      intro t ht
      apply tanh_seventh_integrand_bound
      rw [abs_of_pos ht.1]
      have : z ≤ 1 := by rw [← hz_eq]; exact hz
      linarith [ht.2]
    have h_step1 :
        |∫ t in Set.Ioc 0 z, (t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3)|
          ≤ ∫ t in Set.Ioc 0 z, |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3| :=
      MeasureTheory.abs_integral_le_integral_abs
    have h_step2 : ∫ t in Set.Ioc 0 z, |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3|
          ≤ ∫ t in Set.Ioc 0 z, t ^ 6 := by
      apply MeasureTheory.setIntegral_mono_on
      · exact (((continuous_id.pow 2).sub (continuous_tanh.pow 2)).sub
          ((continuous_const.mul (continuous_id.pow 4)).div_const _)).abs.integrableOn_Ioc
      · exact (continuous_id.pow 6).integrableOn_Ioc
      · exact measurableSet_Ioc
      · exact h_pointwise
    have h_step3 : ∫ t in Set.Ioc 0 z, t ^ 6 = z ^ 7 / 7 := by
      rw [show ∫ t in Set.Ioc 0 z, t ^ 6 = ∫ t in (0)..z, t ^ 6 from
            (intervalIntegral.integral_of_le hz_nn).symm]
      rw [integral_pow]; ring
    have h_z7 : |z| ^ 7 = z ^ 7 := by
      rw [show |z| ^ 7 = |z ^ 7| from by rw [abs_pow]]
      exact abs_of_nonneg (by positivity)
    rw [h_z7]; linarith
  · rw [intervalIntegral.integral_of_ge hz_neg.le, abs_neg]
    have hz_eq : |z| = -z := abs_of_neg hz_neg
    have h_pointwise :
        ∀ t ∈ Set.Ioc z 0, |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3| ≤ t ^ 6 := by
      intro t ht
      apply tanh_seventh_integrand_bound
      have h_neg_z : -z ≤ 1 := by rw [← hz_eq]; exact hz
      rcases lt_or_eq_of_le ht.2 with h_t_neg | h_t_zero
      · rw [abs_of_neg h_t_neg]; linarith [ht.1]
      · rw [h_t_zero]; simp
    have h_step1 :
        |∫ t in Set.Ioc z 0, (t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3)|
          ≤ ∫ t in Set.Ioc z 0, |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3| :=
      MeasureTheory.abs_integral_le_integral_abs
    have h_step2 : ∫ t in Set.Ioc z 0, |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3|
          ≤ ∫ t in Set.Ioc z 0, t ^ 6 := by
      apply MeasureTheory.setIntegral_mono_on
      · exact (((continuous_id.pow 2).sub (continuous_tanh.pow 2)).sub
          ((continuous_const.mul (continuous_id.pow 4)).div_const _)).abs.integrableOn_Ioc
      · exact (continuous_id.pow 6).integrableOn_Ioc
      · exact measurableSet_Ioc
      · exact h_pointwise
    have h_step3 : ∫ t in Set.Ioc z 0, t ^ 6 = -z ^ 7 / 7 := by
      rw [show ∫ t in Set.Ioc z 0, t ^ 6 = ∫ t in z..(0:ℝ), t ^ 6 from
            (intervalIntegral.integral_of_le hz_neg.le).symm]
      rw [integral_pow]; ring
    have h_z7 : |z| ^ 7 = -z ^ 7 := by
      rw [show |z| ^ 7 = |z ^ 7| from by rw [abs_pow]]
      apply abs_of_neg
      have : z ^ 7 = z * z ^ 6 := by ring
      rw [this]
      have h_z6_pos : 0 < z ^ 6 := by
        have : z ≠ 0 := ne_of_lt hz_neg
        positivity
      exact mul_neg_of_neg_of_pos hz_neg h_z6_pos
    rw [h_z7]; linarith

/-- Pointwise bound on the ninth-order Taylor remainder integrand:
`|t² − tanh²(t) − 2t⁴/3 + 17t⁶/45| ≤ t⁸` for `|t| ≤ 1`. Derived by writing
`tanh t = (t − t³/3 + 2t⁵/15) + ε(t)` with `|ε(t)| ≤ |t|⁷/7` (the septenary
bound), expanding `tanh²(t)`, and bounding each error term using `|t| ≤ 1`.
The numeric constant `1517/3675 < 1` gives the `≤ t⁸` envelope. -/
private lemma tanh_ninth_integrand_bound {t : ℝ} (ht : |t| ≤ 1) :
    |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3 + 17 * t ^ 6 / 45| ≤ t ^ 8 := by
  have h_septenary := abs_tanh_sub_taylor5_le ht
  set ε : ℝ := Real.tanh t - (t - t ^ 3 / 3 + 2 * t ^ 5 / 15) with hε_def
  have h_ε_bound : |ε| ≤ |t| ^ 7 / 7 := by rw [hε_def]; exact h_septenary
  have h_tanh : Real.tanh t = (t - t ^ 3 / 3 + 2 * t ^ 5 / 15) + ε := by
    rw [hε_def]; ring
  have h_eq : t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3 + 17 * t ^ 6 / 45
        = (4 * t ^ 8 / 45 - 4 * t ^ 10 / 225)
            - 2 * (t - t ^ 3 / 3 + 2 * t ^ 5 / 15) * ε - ε ^ 2 := by
    rw [h_tanh]; ring
  rw [h_eq]
  -- Bound |poly5(t)| ≤ |t| on |t| ≤ 1.
  have h_poly5_bound : |t - t ^ 3 / 3 + 2 * t ^ 5 / 15| ≤ |t| := by
    rw [show t - t ^ 3 / 3 + 2 * t ^ 5 / 15 = t * (1 - t ^ 2 / 3 + 2 * t ^ 4 / 15)
          from by ring, abs_mul]
    have h_t2_le : t ^ 2 ≤ 1 := by
      rw [show t ^ 2 = |t| ^ 2 from (sq_abs _).symm]
      exact pow_le_one₀ (abs_nonneg _) ht
    have h_t4_le : t ^ 4 ≤ 1 := by
      rw [show t ^ 4 = (t ^ 2) ^ 2 from by ring]
      exact pow_le_one₀ (sq_nonneg _) h_t2_le
    have h_factor : |1 - t ^ 2 / 3 + 2 * t ^ 4 / 15| ≤ 1 := by
      rw [abs_le]
      constructor <;> nlinarith [sq_nonneg t, sq_nonneg (t ^ 2)]
    nlinarith [abs_nonneg t, h_factor]
  have h_t_eq_t8 : t ^ 8 = |t| ^ 8 := by
    rw [show (8 : ℕ) = 2 * 4 from rfl, pow_mul, pow_mul, sq_abs]
  have h_t10_le : |t| ^ 10 ≤ |t| ^ 8 := by
    have h_t2_le : |t| ^ 2 ≤ 1 := pow_le_one₀ (abs_nonneg _) ht
    nlinarith [sq_nonneg (|t| ^ 4), abs_nonneg t]
  have h_t14_le : |t| ^ 14 ≤ |t| ^ 8 :=
    pow_le_pow_of_le_one (abs_nonneg t) ht (by norm_num)
  -- Now triangle inequality on the three-piece sum.
  calc |(4 * t ^ 8 / 45 - 4 * t ^ 10 / 225)
          - 2 * (t - t ^ 3 / 3 + 2 * t ^ 5 / 15) * ε - ε ^ 2|
      ≤ |4 * t ^ 8 / 45 - 4 * t ^ 10 / 225|
          + |2 * (t - t ^ 3 / 3 + 2 * t ^ 5 / 15) * ε| + |ε ^ 2| := by
        have hr : (4 * t ^ 8 / 45 - 4 * t ^ 10 / 225)
                  - 2 * (t - t ^ 3 / 3 + 2 * t ^ 5 / 15) * ε - ε ^ 2
                = ((4 * t ^ 8 / 45 - 4 * t ^ 10 / 225)
                    + (-(2 * (t - t ^ 3 / 3 + 2 * t ^ 5 / 15) * ε))) + (-(ε ^ 2)) := by
          ring
        rw [hr]
        calc _ ≤ |((4 * t ^ 8 / 45 - 4 * t ^ 10 / 225)
                    + (-(2 * (t - t ^ 3 / 3 + 2 * t ^ 5 / 15) * ε)))| + |(-(ε ^ 2))| :=
                abs_add_le _ _
          _ ≤ (|(4 * t ^ 8 / 45 - 4 * t ^ 10 / 225)|
                + |(-(2 * (t - t ^ 3 / 3 + 2 * t ^ 5 / 15) * ε))|) + |(-(ε ^ 2))| := by
              gcongr; exact abs_add_le _ _
          _ = |4 * t ^ 8 / 45 - 4 * t ^ 10 / 225|
              + |2 * (t - t ^ 3 / 3 + 2 * t ^ 5 / 15) * ε| + |ε ^ 2| := by
              rw [abs_neg, abs_neg]
    _ ≤ 4 * |t| ^ 8 / 45 + 4 * |t| ^ 10 / 225
          + 2 * |t| * (|t| ^ 7 / 7) + (|t| ^ 7 / 7) ^ 2 := by
        have h_first : |4 * t ^ 8 / 45 - 4 * t ^ 10 / 225|
              ≤ 4 * |t| ^ 8 / 45 + 4 * |t| ^ 10 / 225 := by
          have hb := abs_sub (4 * t ^ 8 / 45) (4 * t ^ 10 / 225)
          rw [abs_div, abs_div, abs_mul, abs_mul,
              show |(4:ℝ)| = 4 from by norm_num,
              show |(45:ℝ)| = 45 from by norm_num,
              show |(225:ℝ)| = 225 from by norm_num,
              show |t ^ 8| = |t| ^ 8 from abs_pow t 8,
              show |t ^ 10| = |t| ^ 10 from abs_pow t 10] at hb
          linarith
        have h_mid : |2 * (t - t ^ 3 / 3 + 2 * t ^ 5 / 15) * ε|
              ≤ 2 * |t| * (|t| ^ 7 / 7) := by
          rw [show 2 * (t - t ^ 3 / 3 + 2 * t ^ 5 / 15) * ε
                = 2 * ((t - t ^ 3 / 3 + 2 * t ^ 5 / 15) * ε) from by ring,
              abs_mul, abs_mul,
              show |(2 : ℝ)| = 2 from by norm_num]
          have h := mul_le_mul h_poly5_bound h_ε_bound (abs_nonneg _) (abs_nonneg _)
          linarith
        have h_last : |ε ^ 2| ≤ (|t| ^ 7 / 7) ^ 2 := by
          rw [show ε ^ 2 = |ε| ^ 2 from (sq_abs _).symm]
          rw [show |(|ε| ^ 2)| = |ε| ^ 2 from abs_of_nonneg (sq_nonneg _)]
          exact pow_le_pow_left₀ (abs_nonneg _) h_ε_bound 2
        linarith
    _ = |t| ^ 8 * (4 / 45 + 2 / 7) + 4 * |t| ^ 10 / 225 + |t| ^ 14 / 49 := by ring
    _ ≤ |t| ^ 8 * (4 / 45 + 2 / 7) + 4 * |t| ^ 8 / 225 + |t| ^ 8 / 49 := by
        have h1 : 4 * |t| ^ 10 / 225 ≤ 4 * |t| ^ 8 / 225 := by
          have := h_t10_le
          linarith
        have h2 : |t| ^ 14 / 49 ≤ |t| ^ 8 / 49 := by
          have := h_t14_le
          linarith
        linarith
    _ = |t| ^ 8 * (1517 / 3675) := by ring
    _ ≤ |t| ^ 8 := by
        have hnn : 0 ≤ |t| ^ 8 := by positivity
        nlinarith
    _ = t ^ 8 := h_t_eq_t8.symm

/-- Ninth-order bound: `|tanh z − (z − z³/3 + 2z⁵/15 − 17z⁷/315)| ≤ |z|⁹ / 9`
for `|z| ≤ 1`.

The fourth iterated-integral step in the Taylor tower for `tanh`. The proof
mirrors `abs_tanh_sub_taylor5_le`: define
`g(z) = tanh z − z + z³/3 − 2z⁵/15 + 17z⁷/315`; then
`g'(z) = z² − tanh²(z) − 2z⁴/3 + 17z⁶/45` and `|g'(t)| ≤ t⁸` on `|t| ≤ 1` via
`tanh_ninth_integrand_bound`. Integration gives
`|g(z)| ≤ ∫_0^|z| t⁸ dt = |z|⁹/9`. -/
theorem abs_tanh_sub_taylor7_le {z : ℝ} (hz : |z| ≤ 1) :
    |Real.tanh z - (z - z ^ 3 / 3 + 2 * z ^ 5 / 15 - 17 * z ^ 7 / 315)|
      ≤ |z| ^ 9 / 9 := by
  have h_int_repr :
      Real.tanh z - z + z ^ 3 / 3 - 2 * z ^ 5 / 15 + 17 * z ^ 7 / 315
        = ∫ t in (0)..z,
              (t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3 + 17 * t ^ 6 / 45) := by
    have h_g_deriv : ∀ t : ℝ,
        HasDerivAt
          (fun u => Real.tanh u - u + u ^ 3 / 3 - 2 * u ^ 5 / 15 + 17 * u ^ 7 / 315)
          (t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3 + 17 * t ^ 6 / 45) t := by
      intro t
      have h1 := tanh_hasDerivAt t
      have h2 : HasDerivAt (fun u : ℝ => u) 1 t := hasDerivAt_id t
      have h3 : HasDerivAt (fun u : ℝ => u ^ 3 / 3) (t ^ 2) t := by
        have := (hasDerivAt_pow 3 t).div_const 3
        convert this using 1; push_cast; ring
      have h4 : HasDerivAt (fun u : ℝ => 2 * u ^ 5 / 15) (2 * t ^ 4 / 3) t := by
        have h_div := ((hasDerivAt_pow 5 t).const_mul 2).div_const 15
        convert h_div using 1; push_cast; ring
      have h5 : HasDerivAt (fun u : ℝ => 17 * u ^ 7 / 315) (17 * t ^ 6 / 45) t := by
        have h_div := ((hasDerivAt_pow 7 t).const_mul 17).div_const 315
        convert h_div using 1; push_cast; ring
      have h_combined := (((h1.sub h2).add h3).sub h4).add h5
      convert h_combined using 1; ring
    have h_FTC :
        ∫ t in (0)..z,
            (t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3 + 17 * t ^ 6 / 45)
          = (Real.tanh z - z + z ^ 3 / 3 - 2 * z ^ 5 / 15 + 17 * z ^ 7 / 315)
              - (Real.tanh 0 - 0 + 0 ^ 3 / 3 - 2 * 0 ^ 5 / 15 + 17 * 0 ^ 7 / 315) :=
      intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => h_g_deriv x)
        (((((continuous_id.pow 2).sub (continuous_tanh.pow 2)).sub
            ((continuous_const.mul (continuous_id.pow 4)).div_const _)).add
            ((continuous_const.mul (continuous_id.pow 6)).div_const _)).intervalIntegrable _ _)
    rw [Real.tanh_zero] at h_FTC
    linarith
  rw [show Real.tanh z - (z - z ^ 3 / 3 + 2 * z ^ 5 / 15 - 17 * z ^ 7 / 315)
        = Real.tanh z - z + z ^ 3 / 3 - 2 * z ^ 5 / 15 + 17 * z ^ 7 / 315 from by
        ring, h_int_repr]
  rcases le_or_gt 0 z with hz_nn | hz_neg
  · rw [intervalIntegral.integral_of_le hz_nn]
    have hz_eq : |z| = z := abs_of_nonneg hz_nn
    have h_pointwise :
        ∀ t ∈ Set.Ioc 0 z,
          |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3 + 17 * t ^ 6 / 45| ≤ t ^ 8 := by
      intro t ht
      apply tanh_ninth_integrand_bound
      rw [abs_of_pos ht.1]
      have : z ≤ 1 := by rw [← hz_eq]; exact hz
      linarith [ht.2]
    have h_step1 :
        |∫ t in Set.Ioc 0 z,
            (t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3 + 17 * t ^ 6 / 45)|
          ≤ ∫ t in Set.Ioc 0 z,
              |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3 + 17 * t ^ 6 / 45| :=
      MeasureTheory.abs_integral_le_integral_abs
    have h_step2 :
        ∫ t in Set.Ioc 0 z,
            |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3 + 17 * t ^ 6 / 45|
          ≤ ∫ t in Set.Ioc 0 z, t ^ 8 := by
      apply MeasureTheory.setIntegral_mono_on
      · exact ((((continuous_id.pow 2).sub (continuous_tanh.pow 2)).sub
            ((continuous_const.mul (continuous_id.pow 4)).div_const _)).add
            ((continuous_const.mul (continuous_id.pow 6)).div_const _)).abs.integrableOn_Ioc
      · exact (continuous_id.pow 8).integrableOn_Ioc
      · exact measurableSet_Ioc
      · exact h_pointwise
    have h_step3 : ∫ t in Set.Ioc 0 z, t ^ 8 = z ^ 9 / 9 := by
      rw [show ∫ t in Set.Ioc 0 z, t ^ 8 = ∫ t in (0)..z, t ^ 8 from
            (intervalIntegral.integral_of_le hz_nn).symm]
      rw [integral_pow]; ring
    have h_z9 : |z| ^ 9 = z ^ 9 := by
      rw [show |z| ^ 9 = |z ^ 9| from by rw [abs_pow]]
      exact abs_of_nonneg (by positivity)
    rw [h_z9]; linarith
  · rw [intervalIntegral.integral_of_ge hz_neg.le, abs_neg]
    have hz_eq : |z| = -z := abs_of_neg hz_neg
    have h_pointwise :
        ∀ t ∈ Set.Ioc z 0,
          |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3 + 17 * t ^ 6 / 45| ≤ t ^ 8 := by
      intro t ht
      apply tanh_ninth_integrand_bound
      have h_neg_z : -z ≤ 1 := by rw [← hz_eq]; exact hz
      rcases lt_or_eq_of_le ht.2 with h_t_neg | h_t_zero
      · rw [abs_of_neg h_t_neg]; linarith [ht.1]
      · rw [h_t_zero]; simp
    have h_step1 :
        |∫ t in Set.Ioc z 0,
            (t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3 + 17 * t ^ 6 / 45)|
          ≤ ∫ t in Set.Ioc z 0,
              |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3 + 17 * t ^ 6 / 45| :=
      MeasureTheory.abs_integral_le_integral_abs
    have h_step2 :
        ∫ t in Set.Ioc z 0,
            |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3 + 17 * t ^ 6 / 45|
          ≤ ∫ t in Set.Ioc z 0, t ^ 8 := by
      apply MeasureTheory.setIntegral_mono_on
      · exact ((((continuous_id.pow 2).sub (continuous_tanh.pow 2)).sub
            ((continuous_const.mul (continuous_id.pow 4)).div_const _)).add
            ((continuous_const.mul (continuous_id.pow 6)).div_const _)).abs.integrableOn_Ioc
      · exact (continuous_id.pow 8).integrableOn_Ioc
      · exact measurableSet_Ioc
      · exact h_pointwise
    have h_step3 : ∫ t in Set.Ioc z 0, t ^ 8 = -z ^ 9 / 9 := by
      rw [show ∫ t in Set.Ioc z 0, t ^ 8 = ∫ t in z..(0:ℝ), t ^ 8 from
            (intervalIntegral.integral_of_le hz_neg.le).symm]
      rw [integral_pow]; ring
    have h_z9 : |z| ^ 9 = -z ^ 9 := by
      rw [show |z| ^ 9 = |z ^ 9| from by rw [abs_pow]]
      apply abs_of_neg
      have : z ^ 9 = z * z ^ 8 := by ring
      rw [this]
      have h_z8_pos : 0 < z ^ 8 := by
        have : z ≠ 0 := ne_of_lt hz_neg
        positivity
      exact mul_neg_of_neg_of_pos hz_neg h_z8_pos
    rw [h_z9]; linarith

/-- Pointwise bound on the eleventh-order Taylor remainder integrand:
`|t² − tanh²(t) − 2t⁴/3 + 17t⁶/45 − 62t⁸/315| ≤ t¹⁰` for `|t| ≤ 1`. Derived
by writing `tanh t = poly7(t) + ε(t)` with `|ε(t)| ≤ |t|⁹/9` (the 7th-order
Taylor bound), expanding `tanh²(t)`, and bounding the polynomial residual
`254t¹⁰/4725 − 68t¹²/4725 + 289t¹⁴/99225` plus the `2·poly7·ε` and `ε²` cross
terms. Total is `30326/99225 < 1` of `|t|¹⁰`. -/
private lemma tanh_eleventh_integrand_bound {t : ℝ} (ht : |t| ≤ 1) :
    |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3 + 17 * t ^ 6 / 45
       - 62 * t ^ 8 / 315| ≤ t ^ 10 := by
  have h_septor := abs_tanh_sub_taylor7_le ht
  set ε : ℝ := Real.tanh t - (t - t ^ 3 / 3 + 2 * t ^ 5 / 15 - 17 * t ^ 7 / 315)
    with hε_def
  have h_ε_bound : |ε| ≤ |t| ^ 9 / 9 := by rw [hε_def]; exact h_septor
  have h_tanh : Real.tanh t
        = (t - t ^ 3 / 3 + 2 * t ^ 5 / 15 - 17 * t ^ 7 / 315) + ε := by
    rw [hε_def]; ring
  have h_eq : t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3 + 17 * t ^ 6 / 45
                - 62 * t ^ 8 / 315
        = -(254 * t ^ 10 / 4725 - 68 * t ^ 12 / 4725 + 289 * t ^ 14 / 99225)
            - 2 * (t - t ^ 3 / 3 + 2 * t ^ 5 / 15 - 17 * t ^ 7 / 315) * ε
            - ε ^ 2 := by
    rw [h_tanh]; ring
  rw [h_eq]
  -- Bound |poly7(t)| ≤ |t| on |t| ≤ 1.
  have h_poly7_bound : |t - t ^ 3 / 3 + 2 * t ^ 5 / 15 - 17 * t ^ 7 / 315|
        ≤ |t| := by
    rw [show t - t ^ 3 / 3 + 2 * t ^ 5 / 15 - 17 * t ^ 7 / 315
          = t * (1 - t ^ 2 / 3 + 2 * t ^ 4 / 15 - 17 * t ^ 6 / 315)
          from by ring, abs_mul]
    have h_t2_le : t ^ 2 ≤ 1 := by
      rw [show t ^ 2 = |t| ^ 2 from (sq_abs _).symm]
      exact pow_le_one₀ (abs_nonneg _) ht
    have h_t4_le : t ^ 4 ≤ 1 := by
      rw [show t ^ 4 = (t ^ 2) ^ 2 from by ring]
      exact pow_le_one₀ (sq_nonneg _) h_t2_le
    have h_t6_le : t ^ 6 ≤ 1 := by
      rw [show t ^ 6 = (t ^ 2) ^ 3 from by ring]
      exact pow_le_one₀ (sq_nonneg _) h_t2_le
    have h_factor : |1 - t ^ 2 / 3 + 2 * t ^ 4 / 15 - 17 * t ^ 6 / 315| ≤ 1 := by
      rw [abs_le]
      constructor <;> nlinarith [sq_nonneg t, sq_nonneg (t ^ 2), sq_nonneg (t ^ 3)]
    nlinarith [abs_nonneg t, h_factor]
  have h_t_eq_t10 : t ^ 10 = |t| ^ 10 := by
    rw [show (10 : ℕ) = 2 * 5 from rfl, pow_mul, pow_mul, sq_abs]
  have h_t12_le : |t| ^ 12 ≤ |t| ^ 10 :=
    pow_le_pow_of_le_one (abs_nonneg t) ht (by norm_num)
  have h_t14_le : |t| ^ 14 ≤ |t| ^ 10 :=
    pow_le_pow_of_le_one (abs_nonneg t) ht (by norm_num)
  have h_t18_le : |t| ^ 18 ≤ |t| ^ 10 :=
    pow_le_pow_of_le_one (abs_nonneg t) ht (by norm_num)
  calc |-(254 * t ^ 10 / 4725 - 68 * t ^ 12 / 4725 + 289 * t ^ 14 / 99225)
          - 2 * (t - t ^ 3 / 3 + 2 * t ^ 5 / 15 - 17 * t ^ 7 / 315) * ε - ε ^ 2|
      ≤ |254 * t ^ 10 / 4725 - 68 * t ^ 12 / 4725 + 289 * t ^ 14 / 99225|
          + |2 * (t - t ^ 3 / 3 + 2 * t ^ 5 / 15 - 17 * t ^ 7 / 315) * ε|
          + |ε ^ 2| := by
        have hr : -(254 * t ^ 10 / 4725 - 68 * t ^ 12 / 4725
                   + 289 * t ^ 14 / 99225)
                  - 2 * (t - t ^ 3 / 3 + 2 * t ^ 5 / 15 - 17 * t ^ 7 / 315) * ε
                  - ε ^ 2
                = (-(254 * t ^ 10 / 4725 - 68 * t ^ 12 / 4725
                    + 289 * t ^ 14 / 99225)
                   + (-(2 * (t - t ^ 3 / 3 + 2 * t ^ 5 / 15 - 17 * t ^ 7 / 315) * ε)))
                  + (-(ε ^ 2)) := by ring
        rw [hr]
        calc _ ≤ |(-(254 * t ^ 10 / 4725 - 68 * t ^ 12 / 4725
                    + 289 * t ^ 14 / 99225)
                  + (-(2 * (t - t ^ 3 / 3 + 2 * t ^ 5 / 15 - 17 * t ^ 7 / 315) * ε)))|
                + |(-(ε ^ 2))| := abs_add_le _ _
          _ ≤ (|(-(254 * t ^ 10 / 4725 - 68 * t ^ 12 / 4725
                + 289 * t ^ 14 / 99225))|
                + |(-(2 * (t - t ^ 3 / 3 + 2 * t ^ 5 / 15 - 17 * t ^ 7 / 315) * ε))|)
              + |(-(ε ^ 2))| := by
              gcongr; exact abs_add_le _ _
          _ = |254 * t ^ 10 / 4725 - 68 * t ^ 12 / 4725 + 289 * t ^ 14 / 99225|
              + |2 * (t - t ^ 3 / 3 + 2 * t ^ 5 / 15 - 17 * t ^ 7 / 315) * ε|
              + |ε ^ 2| := by rw [abs_neg, abs_neg, abs_neg]
    _ ≤ 254 * |t| ^ 10 / 4725 + 68 * |t| ^ 12 / 4725 + 289 * |t| ^ 14 / 99225
          + 2 * |t| * (|t| ^ 9 / 9) + (|t| ^ 9 / 9) ^ 2 := by
        have h_first : |254 * t ^ 10 / 4725 - 68 * t ^ 12 / 4725
                          + 289 * t ^ 14 / 99225|
              ≤ 254 * |t| ^ 10 / 4725 + 68 * |t| ^ 12 / 4725
                + 289 * |t| ^ 14 / 99225 := by
          have hb1 := abs_add_le (254 * t ^ 10 / 4725 - 68 * t ^ 12 / 4725)
                                 (289 * t ^ 14 / 99225)
          have hb2 := abs_sub (254 * t ^ 10 / 4725) (68 * t ^ 12 / 4725)
          have e1 : |254 * t ^ 10 / 4725| = 254 * |t| ^ 10 / 4725 := by
            rw [abs_div, abs_mul, show |(254 : ℝ)| = 254 from by norm_num,
                show |(4725 : ℝ)| = 4725 from by norm_num,
                show |t ^ 10| = |t| ^ 10 from abs_pow t 10]
          have e2 : |68 * t ^ 12 / 4725| = 68 * |t| ^ 12 / 4725 := by
            rw [abs_div, abs_mul, show |(68 : ℝ)| = 68 from by norm_num,
                show |(4725 : ℝ)| = 4725 from by norm_num,
                show |t ^ 12| = |t| ^ 12 from abs_pow t 12]
          have e3 : |289 * t ^ 14 / 99225| = 289 * |t| ^ 14 / 99225 := by
            rw [abs_div, abs_mul, show |(289 : ℝ)| = 289 from by norm_num,
                show |(99225 : ℝ)| = 99225 from by norm_num,
                show |t ^ 14| = |t| ^ 14 from abs_pow t 14]
          rw [e1, e2] at hb2
          rw [e3] at hb1
          linarith
        have h_mid : |2 * (t - t ^ 3 / 3 + 2 * t ^ 5 / 15 - 17 * t ^ 7 / 315) * ε|
              ≤ 2 * |t| * (|t| ^ 9 / 9) := by
          rw [show 2 * (t - t ^ 3 / 3 + 2 * t ^ 5 / 15 - 17 * t ^ 7 / 315) * ε
                = 2 * ((t - t ^ 3 / 3 + 2 * t ^ 5 / 15 - 17 * t ^ 7 / 315) * ε)
                from by ring,
              abs_mul, abs_mul, show |(2 : ℝ)| = 2 from by norm_num]
          have h := mul_le_mul h_poly7_bound h_ε_bound (abs_nonneg _) (abs_nonneg _)
          linarith
        have h_last : |ε ^ 2| ≤ (|t| ^ 9 / 9) ^ 2 := by
          rw [show ε ^ 2 = |ε| ^ 2 from (sq_abs _).symm]
          rw [show |(|ε| ^ 2)| = |ε| ^ 2 from abs_of_nonneg (sq_nonneg _)]
          exact pow_le_pow_left₀ (abs_nonneg _) h_ε_bound 2
        linarith
    _ = |t| ^ 10 * (254 / 4725 + 2 / 9) + 68 * |t| ^ 12 / 4725
          + 289 * |t| ^ 14 / 99225 + |t| ^ 18 / 81 := by ring
    _ ≤ |t| ^ 10 * (254 / 4725 + 2 / 9) + 68 * |t| ^ 10 / 4725
          + 289 * |t| ^ 10 / 99225 + |t| ^ 10 / 81 := by
        have h1 : 68 * |t| ^ 12 / 4725 ≤ 68 * |t| ^ 10 / 4725 := by
          have := h_t12_le; linarith
        have h2 : 289 * |t| ^ 14 / 99225 ≤ 289 * |t| ^ 10 / 99225 := by
          have := h_t14_le; linarith
        have h3 : |t| ^ 18 / 81 ≤ |t| ^ 10 / 81 := by
          have := h_t18_le; linarith
        linarith
    _ = |t| ^ 10 * (30326 / 99225) := by ring
    _ ≤ |t| ^ 10 := by
        have hnn : 0 ≤ |t| ^ 10 := by positivity
        nlinarith
    _ = t ^ 10 := h_t_eq_t10.symm

/-- Eleventh-order bound: `|tanh z − (z − z³/3 + 2z⁵/15 − 17z⁷/315 + 62z⁹/2835)|
≤ |z|¹¹ / 11` for `|z| ≤ 1`. The fifth iterated-integral step. -/
theorem abs_tanh_sub_taylor9_le {z : ℝ} (hz : |z| ≤ 1) :
    |Real.tanh z - (z - z ^ 3 / 3 + 2 * z ^ 5 / 15 - 17 * z ^ 7 / 315
                      + 62 * z ^ 9 / 2835)| ≤ |z| ^ 11 / 11 := by
  have h_int_repr :
      Real.tanh z - z + z ^ 3 / 3 - 2 * z ^ 5 / 15 + 17 * z ^ 7 / 315
          - 62 * z ^ 9 / 2835
        = ∫ t in (0)..z, (t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3
              + 17 * t ^ 6 / 45 - 62 * t ^ 8 / 315) := by
    have h_g_deriv : ∀ t : ℝ,
        HasDerivAt
          (fun u => Real.tanh u - u + u ^ 3 / 3 - 2 * u ^ 5 / 15 + 17 * u ^ 7 / 315
                      - 62 * u ^ 9 / 2835)
          (t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3 + 17 * t ^ 6 / 45
            - 62 * t ^ 8 / 315) t := by
      intro t
      have h1 := tanh_hasDerivAt t
      have h2 : HasDerivAt (fun u : ℝ => u) 1 t := hasDerivAt_id t
      have h3 : HasDerivAt (fun u : ℝ => u ^ 3 / 3) (t ^ 2) t := by
        have := (hasDerivAt_pow 3 t).div_const 3
        convert this using 1; push_cast; ring
      have h4 : HasDerivAt (fun u : ℝ => 2 * u ^ 5 / 15) (2 * t ^ 4 / 3) t := by
        have h_div := ((hasDerivAt_pow 5 t).const_mul 2).div_const 15
        convert h_div using 1; push_cast; ring
      have h5 : HasDerivAt (fun u : ℝ => 17 * u ^ 7 / 315) (17 * t ^ 6 / 45) t := by
        have h_div := ((hasDerivAt_pow 7 t).const_mul 17).div_const 315
        convert h_div using 1; push_cast; ring
      have h6 : HasDerivAt (fun u : ℝ => 62 * u ^ 9 / 2835) (62 * t ^ 8 / 315) t := by
        have h_div := ((hasDerivAt_pow 9 t).const_mul 62).div_const 2835
        convert h_div using 1; push_cast; ring
      have h_combined := ((((h1.sub h2).add h3).sub h4).add h5).sub h6
      convert h_combined using 1; ring
    have h_FTC :
        ∫ t in (0)..z, (t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3
              + 17 * t ^ 6 / 45 - 62 * t ^ 8 / 315)
          = (Real.tanh z - z + z ^ 3 / 3 - 2 * z ^ 5 / 15 + 17 * z ^ 7 / 315
                - 62 * z ^ 9 / 2835)
              - (Real.tanh 0 - 0 + 0 ^ 3 / 3 - 2 * 0 ^ 5 / 15 + 17 * 0 ^ 7 / 315
                  - 62 * 0 ^ 9 / 2835) :=
      intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => h_g_deriv x)
        ((((((continuous_id.pow 2).sub (continuous_tanh.pow 2)).sub
            ((continuous_const.mul (continuous_id.pow 4)).div_const _)).add
            ((continuous_const.mul (continuous_id.pow 6)).div_const _)).sub
            ((continuous_const.mul (continuous_id.pow 8)).div_const _)).intervalIntegrable _ _)
    rw [Real.tanh_zero] at h_FTC
    linarith
  rw [show Real.tanh z - (z - z ^ 3 / 3 + 2 * z ^ 5 / 15 - 17 * z ^ 7 / 315
                            + 62 * z ^ 9 / 2835)
        = Real.tanh z - z + z ^ 3 / 3 - 2 * z ^ 5 / 15 + 17 * z ^ 7 / 315
              - 62 * z ^ 9 / 2835 from by ring, h_int_repr]
  rcases le_or_gt 0 z with hz_nn | hz_neg
  · rw [intervalIntegral.integral_of_le hz_nn]
    have hz_eq : |z| = z := abs_of_nonneg hz_nn
    have h_pointwise :
        ∀ t ∈ Set.Ioc 0 z,
          |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3 + 17 * t ^ 6 / 45
            - 62 * t ^ 8 / 315| ≤ t ^ 10 := by
      intro t ht
      apply tanh_eleventh_integrand_bound
      rw [abs_of_pos ht.1]
      have : z ≤ 1 := by rw [← hz_eq]; exact hz
      linarith [ht.2]
    have h_step1 :
        |∫ t in Set.Ioc 0 z, (t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3
              + 17 * t ^ 6 / 45 - 62 * t ^ 8 / 315)|
          ≤ ∫ t in Set.Ioc 0 z, |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3
              + 17 * t ^ 6 / 45 - 62 * t ^ 8 / 315| :=
      MeasureTheory.abs_integral_le_integral_abs
    have h_step2 :
        ∫ t in Set.Ioc 0 z, |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3
              + 17 * t ^ 6 / 45 - 62 * t ^ 8 / 315|
          ≤ ∫ t in Set.Ioc 0 z, t ^ 10 := by
      apply MeasureTheory.setIntegral_mono_on
      · exact (((((continuous_id.pow 2).sub (continuous_tanh.pow 2)).sub
            ((continuous_const.mul (continuous_id.pow 4)).div_const _)).add
            ((continuous_const.mul (continuous_id.pow 6)).div_const _)).sub
            ((continuous_const.mul (continuous_id.pow 8)).div_const _)).abs.integrableOn_Ioc
      · exact (continuous_id.pow 10).integrableOn_Ioc
      · exact measurableSet_Ioc
      · exact h_pointwise
    have h_step3 : ∫ t in Set.Ioc 0 z, t ^ 10 = z ^ 11 / 11 := by
      rw [show ∫ t in Set.Ioc 0 z, t ^ 10 = ∫ t in (0)..z, t ^ 10 from
            (intervalIntegral.integral_of_le hz_nn).symm]
      rw [integral_pow]; ring
    have h_z11 : |z| ^ 11 = z ^ 11 := by
      rw [show |z| ^ 11 = |z ^ 11| from by rw [abs_pow]]
      exact abs_of_nonneg (by positivity)
    rw [h_z11]; linarith
  · rw [intervalIntegral.integral_of_ge hz_neg.le, abs_neg]
    have hz_eq : |z| = -z := abs_of_neg hz_neg
    have h_pointwise :
        ∀ t ∈ Set.Ioc z 0,
          |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3 + 17 * t ^ 6 / 45
            - 62 * t ^ 8 / 315| ≤ t ^ 10 := by
      intro t ht
      apply tanh_eleventh_integrand_bound
      have h_neg_z : -z ≤ 1 := by rw [← hz_eq]; exact hz
      rcases lt_or_eq_of_le ht.2 with h_t_neg | h_t_zero
      · rw [abs_of_neg h_t_neg]; linarith [ht.1]
      · rw [h_t_zero]; simp
    have h_step1 :
        |∫ t in Set.Ioc z 0, (t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3
              + 17 * t ^ 6 / 45 - 62 * t ^ 8 / 315)|
          ≤ ∫ t in Set.Ioc z 0, |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3
              + 17 * t ^ 6 / 45 - 62 * t ^ 8 / 315| :=
      MeasureTheory.abs_integral_le_integral_abs
    have h_step2 :
        ∫ t in Set.Ioc z 0, |t ^ 2 - Real.tanh t ^ 2 - 2 * t ^ 4 / 3
              + 17 * t ^ 6 / 45 - 62 * t ^ 8 / 315|
          ≤ ∫ t in Set.Ioc z 0, t ^ 10 := by
      apply MeasureTheory.setIntegral_mono_on
      · exact (((((continuous_id.pow 2).sub (continuous_tanh.pow 2)).sub
            ((continuous_const.mul (continuous_id.pow 4)).div_const _)).add
            ((continuous_const.mul (continuous_id.pow 6)).div_const _)).sub
            ((continuous_const.mul (continuous_id.pow 8)).div_const _)).abs.integrableOn_Ioc
      · exact (continuous_id.pow 10).integrableOn_Ioc
      · exact measurableSet_Ioc
      · exact h_pointwise
    have h_step3 : ∫ t in Set.Ioc z 0, t ^ 10 = -z ^ 11 / 11 := by
      rw [show ∫ t in Set.Ioc z 0, t ^ 10 = ∫ t in z..(0:ℝ), t ^ 10 from
            (intervalIntegral.integral_of_le hz_neg.le).symm]
      rw [integral_pow]; ring
    have h_z11 : |z| ^ 11 = -z ^ 11 := by
      rw [show |z| ^ 11 = |z ^ 11| from by rw [abs_pow]]
      apply abs_of_neg
      have : z ^ 11 = z * z ^ 10 := by ring
      rw [this]
      have h_z10_pos : 0 < z ^ 10 := by
        have : z ≠ 0 := ne_of_lt hz_neg
        positivity
      exact mul_neg_of_neg_of_pos hz_neg h_z10_pos
    rw [h_z11]; linarith

/-- Public corollary of the eleventh-order integrand bound, expressed as a
direct bound on the derivative `d/du (tanh u − tanhTaylor9(u))`. For `|u| ≤ 1`,
`|(1 − tanh²u) − (1 − u² + 2u⁴/3 − 17u⁶/45 + 62u⁸/315)| ≤ |u|¹⁰`. -/
theorem abs_tanh_minus_taylor9_deriv_le {u : ℝ} (hu : |u| ≤ 1) :
    |(1 - Real.tanh u ^ 2)
      - (1 - u ^ 2 + 2 * u ^ 4 / 3 - 17 * u ^ 6 / 45 + 62 * u ^ 8 / 315)|
      ≤ |u| ^ 10 := by
  have h := tanh_eleventh_integrand_bound hu
  have h_eq :
      (1 - Real.tanh u ^ 2)
        - (1 - u ^ 2 + 2 * u ^ 4 / 3 - 17 * u ^ 6 / 45 + 62 * u ^ 8 / 315)
      = u ^ 2 - Real.tanh u ^ 2 - 2 * u ^ 4 / 3 + 17 * u ^ 6 / 45
          - 62 * u ^ 8 / 315 := by ring
  rw [h_eq]
  rw [show u ^ 10 = |u| ^ 10 from by
    rw [show (10 : ℕ) = 2 * 5 from rfl, pow_mul, pow_mul, sq_abs]] at h
  exact h

end VeriTile.Math
