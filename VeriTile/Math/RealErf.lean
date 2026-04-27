/-
VeriTile.Math.RealErf

A self-contained development of the Gauss error function used by
`VeriTile.Examples.ApproxGeLU`. The mathlib snapshot pinned by VeriTile
(currently v4.29.0) does not expose `Real.erf`, so we build the API we need
on top of `integral_gaussian_Ioi` and the fundamental theorem of calculus.

The error function is defined exactly as in the standard references:

    realErf x = (2 / √π) * ∫_0^x exp(-(t*t)) dt.

Tier 1 results provided here:
  * algebraic identities: `realErf_zero`, `realErf_neg`
  * regularity: `realErf_continuous`, `realErf_hasDerivAt`, `realErf_deriv`
  * tendsto at infinity: `realErf_tendsto_atTop`, `realErf_tendsto_atBot`
  * the global pointwise bound `|realErf x| ≤ 1`.

Tier 2/3 results (quantitative `tanh`-vs-`erf` approximation bounds) are
deferred and not part of this file.
-/

import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.MeasureTheory.Integral.IntegralEqImproper

namespace VeriTile.Math

open Real MeasureTheory Set Filter Topology
open scoped Topology

/-! ## Gaussian kernel `t ↦ exp(-(t*t))` -/

/-- Pointwise integrand of the Gauss error function. We use `t * t` rather
than `t ^ 2` so the algebraic shape lines up with the kernel that appears in
`VeriTile.Examples.ApproxGeLU`. -/
noncomputable def gaussianKernel (t : ℝ) : ℝ := Real.exp (-(t * t))

lemma gaussianKernel_def (t : ℝ) :
    gaussianKernel t = Real.exp (-(t * t)) := rfl

lemma gaussianKernel_eq_pow (t : ℝ) :
    gaussianKernel t = Real.exp (-1 * t ^ 2) := by
  simp [gaussianKernel, sq, mul_comm]

lemma gaussianKernel_neg (t : ℝ) : gaussianKernel (-t) = gaussianKernel t := by
  simp [gaussianKernel]

lemma continuous_gaussianKernel : Continuous gaussianKernel := by
  unfold gaussianKernel
  fun_prop

lemma gaussianKernel_pos (t : ℝ) : 0 < gaussianKernel t :=
  Real.exp_pos _

lemma gaussianKernel_nonneg (t : ℝ) : 0 ≤ gaussianKernel t :=
  (gaussianKernel_pos t).le

lemma gaussianKernel_le_one (t : ℝ) : gaussianKernel t ≤ 1 := by
  unfold gaussianKernel
  exact Real.exp_le_one_iff.mpr (neg_nonpos.mpr (mul_self_nonneg _))

lemma integrable_gaussianKernel : Integrable gaussianKernel := by
  have h : Integrable (fun t => Real.exp (-1 * t ^ 2)) :=
    integrable_exp_neg_mul_sq one_pos
  refine h.congr ?_
  refine Filter.Eventually.of_forall (fun t => ?_)
  exact (gaussianKernel_eq_pow t).symm

lemma integrableOn_gaussianKernel_Ioi (a : ℝ) :
    IntegrableOn gaussianKernel (Ioi a) :=
  integrable_gaussianKernel.integrableOn

lemma intervalIntegrable_gaussianKernel (a b : ℝ) :
    IntervalIntegrable gaussianKernel volume a b :=
  continuous_gaussianKernel.intervalIntegrable a b

/-- The half-line Gaussian integral, in the `t * t` shape.
Direct consequence of `integral_gaussian_Ioi 1`. -/
lemma integral_gaussianKernel_Ioi_zero :
    ∫ t in Ioi (0 : ℝ), gaussianKernel t = Real.sqrt Real.pi / 2 := by
  have h := integral_gaussian_Ioi (1 : ℝ)
  have hcongr : ∫ t in Ioi (0 : ℝ), gaussianKernel t
      = ∫ t in Ioi (0 : ℝ), Real.exp (-1 * t ^ 2) := by
    refine setIntegral_congr_ae measurableSet_Ioi ?_
    refine Filter.Eventually.of_forall (fun t _ => ?_)
    exact gaussianKernel_eq_pow t
  rw [hcongr, h, div_one]

/-! ## The error function -/

/-- The Gauss error function:
`realErf x = (2 / √π) * ∫_0^x exp(-(t*t)) dt`. -/
noncomputable def realErf (x : ℝ) : ℝ :=
  (2 / Real.sqrt Real.pi) * ∫ t in (0)..x, gaussianKernel t

lemma realErf_def (x : ℝ) :
    realErf x = (2 / Real.sqrt Real.pi) *
      ∫ t in (0)..x, Real.exp (-(t * t)) := rfl

@[simp] lemma realErf_zero : realErf 0 = 0 := by
  simp [realErf, intervalIntegral.integral_same]

lemma sqrt_pi_pos : 0 < Real.sqrt Real.pi :=
  Real.sqrt_pos.mpr Real.pi_pos

lemma sqrt_pi_ne_zero : Real.sqrt Real.pi ≠ 0 :=
  ne_of_gt sqrt_pi_pos

lemma two_div_sqrt_pi_pos : 0 < 2 / Real.sqrt Real.pi :=
  div_pos two_pos sqrt_pi_pos

/-- Oddness of `realErf`. The Gaussian kernel is even, so reflecting the
upper integration limit flips the sign of the interval integral. -/
lemma realErf_neg (x : ℝ) : realErf (-x) = -realErf x := by
  unfold realErf
  rw [neg_mul_eq_mul_neg]
  congr 1
  -- ∫ t in 0..(-x), gaussianKernel t = -∫ t in 0..x, gaussianKernel t
  have h_even : ∀ t : ℝ, gaussianKernel t = gaussianKernel (-t) :=
    fun t => (gaussianKernel_neg t).symm
  calc
    ∫ t in (0)..(-x), gaussianKernel t
        = ∫ t in (0)..(-x), gaussianKernel (-t) := by
              refine intervalIntegral.integral_congr ?_
              intro t _; exact h_even t
      _ = ∫ t in -(-x)..(-0), gaussianKernel t := by
              rw [intervalIntegral.integral_comp_neg]
      _ = ∫ t in x..0, gaussianKernel t := by simp
      _ = -∫ t in (0)..x, gaussianKernel t := by
              rw [intervalIntegral.integral_symm]

/-! ## Regularity -/

lemma realErf_hasDerivAt (x : ℝ) :
    HasDerivAt realErf ((2 / Real.sqrt Real.pi) * gaussianKernel x) x := by
  have h_int : IntervalIntegrable gaussianKernel volume 0 x :=
    intervalIntegrable_gaussianKernel 0 x
  have h_meas : StronglyMeasurableAtFilter gaussianKernel (𝓝 x) :=
    continuous_gaussianKernel.stronglyMeasurableAtFilter _ _
  have h_cts : ContinuousAt gaussianKernel x :=
    continuous_gaussianKernel.continuousAt
  have hF : HasDerivAt (fun u => ∫ t in (0)..u, gaussianKernel t)
      (gaussianKernel x) x :=
    intervalIntegral.integral_hasDerivAt_right h_int h_meas h_cts
  exact hF.const_mul _

lemma realErf_differentiable : Differentiable ℝ realErf :=
  fun x => (realErf_hasDerivAt x).differentiableAt

lemma realErf_continuous : Continuous realErf :=
  realErf_differentiable.continuous

lemma realErf_deriv (x : ℝ) :
    deriv realErf x = (2 / Real.sqrt Real.pi) * gaussianKernel x :=
  (realErf_hasDerivAt x).deriv

/-! ## Pointwise bound `|realErf x| ≤ 1` -/

/-- For non-negative `x`, the inner integral lies between `0` and the full
half-line Gaussian integral `√π / 2`. -/
lemma integral_gaussianKernel_zero_to_le_sqrt_pi_div_two
    {x : ℝ} (hx : 0 ≤ x) :
    ∫ t in (0)..x, gaussianKernel t ≤ Real.sqrt Real.pi / 2 := by
  rw [intervalIntegral.integral_of_le hx, ← integral_gaussianKernel_Ioi_zero]
  refine setIntegral_mono_set (integrableOn_gaussianKernel_Ioi 0) ?_ ?_
  · exact Filter.Eventually.of_forall (fun t => gaussianKernel_nonneg t)
  · exact Set.Ioc_subset_Ioi_self.eventuallyLE

lemma integral_gaussianKernel_zero_to_nonneg {x : ℝ} (hx : 0 ≤ x) :
    0 ≤ ∫ t in (0)..x, gaussianKernel t := by
  rw [intervalIntegral.integral_of_le hx]
  exact setIntegral_nonneg measurableSet_Ioc
    (fun t _ => gaussianKernel_nonneg t)

/-- For `x ≥ 0`, `0 ≤ realErf x ≤ 1`. -/
lemma realErf_mem_unitInterval_of_nonneg {x : ℝ} (hx : 0 ≤ x) :
    0 ≤ realErf x ∧ realErf x ≤ 1 := by
  refine ⟨?_, ?_⟩
  · unfold realErf
    exact mul_nonneg two_div_sqrt_pi_pos.le
      (integral_gaussianKernel_zero_to_nonneg hx)
  · unfold realErf
    have hub :=
      integral_gaussianKernel_zero_to_le_sqrt_pi_div_two hx
    have h2 : (2 / Real.sqrt Real.pi) * (Real.sqrt Real.pi / 2) = 1 := by
      field_simp
    calc
      (2 / Real.sqrt Real.pi) * ∫ t in (0)..x, gaussianKernel t
          ≤ (2 / Real.sqrt Real.pi) * (Real.sqrt Real.pi / 2) := by
                exact mul_le_mul_of_nonneg_left hub two_div_sqrt_pi_pos.le
        _ = 1 := h2

/-- The global pointwise bound `|realErf x| ≤ 1`. -/
theorem abs_realErf_le_one (x : ℝ) : |realErf x| ≤ 1 := by
  rcases le_or_gt 0 x with hx | hx
  · obtain ⟨h0, h1⟩ := realErf_mem_unitInterval_of_nonneg hx
    exact abs_le.mpr ⟨by linarith, h1⟩
  · -- Reduce to the non-negative case via oddness.
    have hx' : 0 ≤ -x := by linarith
    obtain ⟨h0, h1⟩ := realErf_mem_unitInterval_of_nonneg hx'
    rw [show realErf x = -realErf (-x) by rw [realErf_neg, neg_neg]]
    rw [abs_neg]
    exact abs_le.mpr ⟨by linarith, h1⟩

/-! ## Tendsto at ±∞ -/

/-- `realErf x → 1` as `x → ∞`. The factor `2/√π` cancels with the half-line
Gaussian integral `√π / 2`. -/
theorem realErf_tendsto_atTop :
    Tendsto realErf atTop (𝓝 1) := by
  have h_int : Tendsto (fun u => ∫ t in (0)..u, gaussianKernel t) atTop
      (𝓝 (∫ t in Ioi (0 : ℝ), gaussianKernel t)) :=
    intervalIntegral_tendsto_integral_Ioi 0
      (integrableOn_gaussianKernel_Ioi 0) tendsto_id
  have h_const :
      Tendsto realErf atTop
        (𝓝 ((2 / Real.sqrt Real.pi) *
              ∫ t in Ioi (0 : ℝ), gaussianKernel t)) :=
    h_int.const_mul (2 / Real.sqrt Real.pi)
  have h_eq : (2 / Real.sqrt Real.pi) *
      ∫ t in Ioi (0 : ℝ), gaussianKernel t = 1 := by
    rw [integral_gaussianKernel_Ioi_zero]
    field_simp
  rw [h_eq] at h_const
  exact h_const

/-- `realErf x → -1` as `x → -∞`. Follows from oddness and the limit at `+∞`. -/
theorem realErf_tendsto_atBot :
    Tendsto realErf atBot (𝓝 (-1)) := by
  have h := realErf_tendsto_atTop
  have hcomp : Tendsto (fun x => realErf (-x)) atBot (𝓝 1) := by
    have hneg : Tendsto (fun x : ℝ => -x) atBot atTop := tendsto_neg_atBot_atTop
    exact h.comp hneg
  have hodd : ∀ x, realErf (-x) = -realErf x := realErf_neg
  have : Tendsto (fun x => -realErf x) atBot (𝓝 1) := by
    refine hcomp.congr ?_
    intro x; exact hodd x
  -- Tendsto (-realErf) atBot (𝓝 1) ↔ Tendsto realErf atBot (𝓝 (-1))
  have := this.neg
  simpa using this

/-! ## Tail bound for `realErf` -/

/-- Splitting `Ioi 0 = Ioc 0 z ∪ Ioi z` for `z ≥ 0`. -/
private lemma Ioi_zero_eq_union_Ioc_Ioi {z : ℝ} (hz : 0 ≤ z) :
    Ioi (0 : ℝ) = Ioc 0 z ∪ Ioi z := by
  ext t
  simp only [mem_Ioc, mem_Ioi, mem_union]
  refine ⟨?_, ?_⟩
  · intro h
    rcases le_or_gt t z with h' | h'
    · exact Or.inl ⟨h, h'⟩
    · exact Or.inr h'
  · rintro (⟨h, _⟩ | h)
    · exact h
    · linarith

/-- For `z ≥ 0`, the half-line Gaussian integral splits as the inner-interval
piece plus the right tail. -/
private lemma integral_gaussianKernel_Ioi_split {z : ℝ} (hz : 0 ≤ z) :
    ∫ t in Ioi (0 : ℝ), gaussianKernel t
      = (∫ t in Ioc 0 z, gaussianKernel t) +
          ∫ t in Ioi z, gaussianKernel t := by
  have hsub1 : Ioc (0 : ℝ) z ⊆ Ioi 0 := fun t ht => ht.1
  have hsub2 : Ioi z ⊆ Ioi (0 : ℝ) := fun t ht => lt_of_le_of_lt hz ht
  have hu : Ioi (0 : ℝ) = Ioc 0 z ∪ Ioi z := Ioi_zero_eq_union_Ioc_Ioi hz
  conv_lhs => rw [hu]
  exact MeasureTheory.setIntegral_union Set.Ioc_disjoint_Ioi_same measurableSet_Ioi
    ((integrableOn_gaussianKernel_Ioi 0).mono_set hsub1)
    ((integrableOn_gaussianKernel_Ioi 0).mono_set hsub2)

/-- For `z ≥ 0`, `realErf z = 1 − (2/√π) · ∫_(Ioi z) exp(-(t·t)) dt`. This is
the integral form of the right tail of `realErf`. -/
lemma one_sub_realErf_eq {z : ℝ} (hz : 0 ≤ z) :
    1 - realErf z =
      (2 / Real.sqrt Real.pi) * ∫ t in Ioi z, gaussianKernel t := by
  have hsplit := integral_gaussianKernel_Ioi_split hz
  have hpos : Real.sqrt Real.pi ≠ 0 := sqrt_pi_ne_zero
  have hint_ioc :
      ∫ t in (0)..z, gaussianKernel t = ∫ t in Ioc 0 z, gaussianKernel t :=
    intervalIntegral.integral_of_le hz
  have hone :
      (2 / Real.sqrt Real.pi) * ∫ t in Ioi (0 : ℝ), gaussianKernel t = 1 := by
    rw [integral_gaussianKernel_Ioi_zero]
    field_simp
  unfold realErf
  rw [hint_ioc]
  have : (2 / Real.sqrt Real.pi) * (∫ t in Ioc (0:ℝ) z, gaussianKernel t)
        + (2 / Real.sqrt Real.pi) * (∫ t in Ioi z, gaussianKernel t)
        = 1 := by
    rw [← mul_add, ← hsplit, hone]
  linarith

/-- The right Gaussian tail is bounded by the exponential tail for `z ≥ 1`.
On `Ioi z ⊆ Ioi 1`, `t² ≥ t`, so `exp(-(t·t)) ≤ exp(-t)`, and integrating
gives `∫_(Ioi z) exp(-(t·t)) dt ≤ ∫_(Ioi z) exp(-t) dt = exp(-z)`. -/
private lemma integral_gaussianKernel_Ioi_le_exp_neg {z : ℝ} (hz : 1 ≤ z) :
    ∫ t in Ioi z, gaussianKernel t ≤ Real.exp (-z) := by
  have h_pointwise : ∀ t ∈ Ioi z, gaussianKernel t ≤ Real.exp (-t) := by
    intro t ht
    have htge : 1 ≤ t := le_trans hz (le_of_lt ht)
    have htnn : 0 ≤ t := by linarith
    have htsq : t ≤ t * t := by nlinarith
    unfold gaussianKernel
    exact Real.exp_le_exp.mpr (by linarith [htsq])
  have h_int_exp_neg : IntegrableOn (fun t => Real.exp (-t)) (Ioi z) :=
    integrableOn_exp_neg_Ioi z
  have h_int_gauss : IntegrableOn gaussianKernel (Ioi z) :=
    integrable_gaussianKernel.integrableOn
  calc ∫ t in Ioi z, gaussianKernel t
      ≤ ∫ t in Ioi z, Real.exp (-t) := by
          refine setIntegral_mono_on h_int_gauss h_int_exp_neg measurableSet_Ioi ?_
          exact h_pointwise
    _ = Real.exp (-z) := integral_exp_neg_Ioi z

/-- Tail bound for `realErf`: for `z ≥ 1`,
`1 − realErf z ≤ (2 / √π) · exp(−z)`. -/
theorem one_sub_realErf_le_exp_neg {z : ℝ} (hz : 1 ≤ z) :
    1 - realErf z ≤ (2 / Real.sqrt Real.pi) * Real.exp (-z) := by
  rw [one_sub_realErf_eq (by linarith : (0:ℝ) ≤ z)]
  have hcoeff : 0 ≤ 2 / Real.sqrt Real.pi := two_div_sqrt_pi_pos.le
  have h_tail := integral_gaussianKernel_Ioi_le_exp_neg hz
  exact mul_le_mul_of_nonneg_left h_tail hcoeff

/-- Symmetric tail bound: for `z ≤ -1`,
`realErf z + 1 ≤ (2 / √π) · exp(z)`. Follows from oddness. -/
theorem realErf_add_one_le_exp {z : ℝ} (hz : z ≤ -1) :
    realErf z + 1 ≤ (2 / Real.sqrt Real.pi) * Real.exp z := by
  have hz' : 1 ≤ -z := by linarith
  have h := one_sub_realErf_le_exp_neg hz'
  rw [realErf_neg] at h
  have hexp : Real.exp (-(-z)) = Real.exp z := by rw [neg_neg]
  rw [hexp] at h
  linarith

end VeriTile.Math
