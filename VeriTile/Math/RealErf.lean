/-
VeriTile.Math.RealErf

A self-contained development of the Gauss error function used by
`VeriTile.Examples.ApproxGeLU`. The mathlib snapshot pinned by VeriTile
(currently v4.29.0) does not expose `Real.erf`, so we build the API we need
on top of `integral_gaussian_Ioi` and the fundamental theorem of calculus.

The definitions themselves (`gaussianKernel`, `realErf`) live in the
lightweight module `VeriTile.Triton.Math.Erf` (same namespace, same
constants), so the core Triton semantics can use them without pulling in the
heavy analysis imports below. This file proves the theorems about them.

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
import VeriTile.Triton.Math.Erf

namespace VeriTile.Math

open Real MeasureTheory Set Filter Topology
open scoped Topology

/-! ## Gaussian kernel `t ↦ exp(-(t*t))`

`gaussianKernel` itself is defined in `VeriTile.Triton.Math.Erf`. -/

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

/-! ## The error function

`realErf` itself is defined in `VeriTile.Triton.Math.Erf`. -/

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

/-- Sharper Gaussian-tail bound for `realErf`: for `z ≥ 1`,
`1 − realErf z ≤ (2 / √π) · exp (−z²) / z`.

The proof compares `exp (−t²)` on `t ∈ (z, ∞)` with `exp (−z·t)` and then
uses the closed form for the exponential half-line integral. -/
theorem one_sub_realErf_le_exp_neg_sq_div {z : ℝ} (hz : 1 ≤ z) :
    1 - realErf z ≤ (2 / Real.sqrt Real.pi) * (Real.exp (-(z * z)) / z) := by
  rw [one_sub_realErf_eq (by linarith : (0:ℝ) ≤ z)]
  have hz_pos : 0 < z := by linarith
  have h_pointwise : ∀ t ∈ Ioi z, gaussianKernel t ≤ Real.exp ((-z) * t) := by
    intro t ht
    have hzt : z ≤ t := le_of_lt ht
    have hz_nonneg : 0 ≤ z := hz_pos.le
    have ht_nonneg : 0 ≤ t := by linarith
    have hsq : z * t ≤ t * t := by nlinarith
    unfold gaussianKernel
    exact Real.exp_le_exp.mpr (by nlinarith)
  have h_int_exp : IntegrableOn (fun t : ℝ => Real.exp ((-z) * t)) (Ioi z) :=
    integrableOn_exp_mul_Ioi (by linarith : (-z : ℝ) < 0) z
  have h_int_gauss : IntegrableOn gaussianKernel (Ioi z) :=
    integrable_gaussianKernel.integrableOn
  have h_tail :
      ∫ t in Ioi z, gaussianKernel t ≤ Real.exp (-(z * z)) / z := by
    calc ∫ t in Ioi z, gaussianKernel t
        ≤ ∫ t in Ioi z, Real.exp ((-z) * t) := by
            refine setIntegral_mono_on h_int_gauss h_int_exp measurableSet_Ioi ?_
            exact h_pointwise
      _ = -Real.exp ((-z) * z) / (-z) := by
            exact integral_exp_mul_Ioi (by linarith : (-z : ℝ) < 0) z
      _ = Real.exp (-(z * z)) / z := by ring_nf
  exact mul_le_mul_of_nonneg_left h_tail two_div_sqrt_pi_pos.le

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

/-! ## Taylor remainder for `realErf` on the unit ball

For `|z| ≤ 1`, the integrand `exp(-(t·t))` admits the Taylor expansion
`1 − t² + t⁴/2 − t⁶/6` with explicit remainder `5·t⁸/96` (from
mathlib's `Real.exp_bound` applied at `x = -(t·t)`). Integrating
term-by-term gives a 7th-order Taylor bound for `realErf z` with a
9th-order remainder. -/

/-- Pointwise Taylor remainder for the Gaussian kernel on `|t| ≤ 1`:
`|exp(-(t·t)) − (1 − t² + t⁴/2 − t⁶/6)| ≤ 5 · t⁸ / 96`. Direct
application of `Real.exp_bound` with `x = -(t·t)` and `n = 4`. -/
private lemma gaussianKernel_taylor3_bound {t : ℝ} (ht : |t| ≤ 1) :
    |gaussianKernel t - (1 - t ^ 2 + t ^ 4 / 2 - t ^ 6 / 6)| ≤ 5 * t ^ 8 / 96 := by
  unfold gaussianKernel
  have h_x_bound : |(-(t * t))| ≤ 1 := by
    rw [abs_neg, abs_mul]
    nlinarith [abs_nonneg t]
  have h := Real.exp_bound h_x_bound (by norm_num : 0 < 4)
  have h_sum : ∑ m ∈ Finset.range 4, (-(t * t))^m / (m.factorial : ℝ)
        = 1 - t ^ 2 + t ^ 4 / 2 - t ^ 6 / 6 := by
    simp [Finset.sum_range_succ, Nat.factorial]; ring
  rw [h_sum] at h
  have h_x_pow : |(-(t * t))| ^ 4 = t ^ 8 := by
    rw [abs_neg, show t * t = t ^ 2 from by ring, abs_pow, sq_abs]; ring
  rw [h_x_pow] at h
  have h_const : ((Nat.succ 4 : ℕ) : ℝ) / ((Nat.factorial 4 : ℕ) * (4 : ℕ) : ℝ) = 5 / 96 := by
    norm_num [Nat.factorial]
  rw [h_const] at h
  linarith

/-- Antiderivative computation via FTC: the polynomial
`F(z) = z − z³/3 + z⁵/10 − z⁷/42` is an antiderivative of
`1 − z² + z⁴/2 − z⁶/6`, so the integral on `[0, z]` evaluates to `F(z)`. -/
private lemma integral_gaussianKernel_taylor_poly (z : ℝ) :
    ∫ t in (0)..z, (1 - t ^ 2 + t ^ 4 / 2 - t ^ 6 / 6)
      = z - z ^ 3 / 3 + z ^ 5 / 10 - z ^ 7 / 42 := by
  have h_deriv : ∀ t : ℝ, HasDerivAt
      (fun u => u - u ^ 3 / 3 + u ^ 5 / 10 - u ^ 7 / 42)
      (1 - t ^ 2 + t ^ 4 / 2 - t ^ 6 / 6) t := by
    intro t
    have h1 : HasDerivAt (fun u : ℝ => u) 1 t := hasDerivAt_id t
    have h2 : HasDerivAt (fun u : ℝ => u ^ 3 / 3) (t ^ 2) t := by
      have := (hasDerivAt_pow 3 t).div_const 3
      convert this using 1; push_cast; ring
    have h3 : HasDerivAt (fun u : ℝ => u ^ 5 / 10) (t ^ 4 / 2) t := by
      have := (hasDerivAt_pow 5 t).div_const 10
      convert this using 1; push_cast; ring
    have h4 : HasDerivAt (fun u : ℝ => u ^ 7 / 42) (t ^ 6 / 6) t := by
      have := (hasDerivAt_pow 7 t).div_const 42
      convert this using 1; push_cast; ring
    have h_combined := ((h1.sub h2).add h3).sub h4
    convert h_combined using 1
  have h_FTC :
      ∫ t in (0)..z, (1 - t ^ 2 + t ^ 4 / 2 - t ^ 6 / 6)
        = (z - z ^ 3 / 3 + z ^ 5 / 10 - z ^ 7 / 42)
            - (0 - 0 ^ 3 / 3 + 0 ^ 5 / 10 - 0 ^ 7 / 42) := by
    apply intervalIntegral.integral_eq_sub_of_hasDerivAt (fun x _ => h_deriv x)
    have h_cont : Continuous (fun t : ℝ => 1 - t ^ 2 + t ^ 4 / 2 - t ^ 6 / 6) := by
      fun_prop
    exact h_cont.intervalIntegrable _ _
  linarith [h_FTC]

/-- Integral form of the Taylor bound: for `|z| ≤ 1`, the inner integral
satisfies `|∫_0^z gaussianKernel − (z − z³/3 + z⁵/10 − z⁷/42)| ≤ 5·|z|⁹/864`.
Follows from `gaussianKernel_taylor3_bound` integrated pointwise. -/
private lemma integral_gaussianKernel_taylor_bound {z : ℝ} (hz : |z| ≤ 1) :
    |(∫ t in (0)..z, gaussianKernel t) - (z - z ^ 3 / 3 + z ^ 5 / 10 - z ^ 7 / 42)|
      ≤ 5 * |z| ^ 9 / 864 := by
  have h_split :
      (∫ t in (0)..z, gaussianKernel t) - (z - z ^ 3 / 3 + z ^ 5 / 10 - z ^ 7 / 42)
        = ∫ t in (0)..z, (gaussianKernel t - (1 - t ^ 2 + t ^ 4 / 2 - t ^ 6 / 6)) := by
    rw [intervalIntegral.integral_sub
        (continuous_gaussianKernel.intervalIntegrable _ _)
        (Continuous.intervalIntegrable (by fun_prop) _ _)]
    rw [integral_gaussianKernel_taylor_poly]
  rw [h_split]
  -- Pointwise bound: for t ∈ uIoc 0 z, |gaussianKernel t - poly3(t)| ≤ 5*t⁸/96
  rcases le_or_gt 0 z with hz_nn | hz_neg
  · rw [intervalIntegral.integral_of_le hz_nn]
    have hz_eq : |z| = z := abs_of_nonneg hz_nn
    have h_pointwise :
        ∀ t ∈ Set.Ioc 0 z,
          |gaussianKernel t - (1 - t ^ 2 + t ^ 4 / 2 - t ^ 6 / 6)| ≤ 5 * t ^ 8 / 96 := by
      intro t ht
      apply gaussianKernel_taylor3_bound
      rw [abs_of_pos ht.1]; linarith [ht.2, hz, hz_eq.symm.le]
    have h_step1 :
        |∫ t in Set.Ioc 0 z,
          gaussianKernel t - (1 - t ^ 2 + t ^ 4 / 2 - t ^ 6 / 6)|
          ≤ ∫ t in Set.Ioc 0 z,
              |gaussianKernel t - (1 - t ^ 2 + t ^ 4 / 2 - t ^ 6 / 6)| :=
      MeasureTheory.abs_integral_le_integral_abs
    have h_step2 : ∫ t in Set.Ioc 0 z,
            |gaussianKernel t - (1 - t ^ 2 + t ^ 4 / 2 - t ^ 6 / 6)|
          ≤ ∫ t in Set.Ioc 0 z, 5 * t ^ 8 / 96 := by
      apply MeasureTheory.setIntegral_mono_on
      · exact (continuous_gaussianKernel.sub (by fun_prop)).abs.integrableOn_Ioc
      · exact (continuous_const.mul (continuous_id.pow 8) |>.div_const _).integrableOn_Ioc
      · exact measurableSet_Ioc
      · exact h_pointwise
    have h_step3 : ∫ t in Set.Ioc 0 z, 5 * t ^ 8 / 96 = 5 * z ^ 9 / 864 := by
      rw [show ∫ t in Set.Ioc 0 z, 5 * t ^ 8 / 96 = ∫ t in (0)..z, 5 * t ^ 8 / 96 from
            (intervalIntegral.integral_of_le hz_nn).symm]
      have : ∫ t in (0)..z, 5 * t ^ 8 / 96 = ∫ t in (0)..z, (5/96) * t ^ 8 := by
        apply intervalIntegral.integral_congr; intro t _; ring
      rw [this, intervalIntegral.integral_const_mul, integral_pow]; ring
    have h_z9 : |z| ^ 9 = z ^ 9 := by
      rw [show |z| ^ 9 = |z ^ 9| from by rw [abs_pow]]
      exact abs_of_nonneg (by positivity)
    rw [h_z9]; linarith
  · rw [intervalIntegral.integral_of_ge hz_neg.le, abs_neg]
    have hz_eq : |z| = -z := abs_of_neg hz_neg
    have h_pointwise :
        ∀ t ∈ Set.Ioc z 0,
          |gaussianKernel t - (1 - t ^ 2 + t ^ 4 / 2 - t ^ 6 / 6)| ≤ 5 * t ^ 8 / 96 := by
      intro t ht
      apply gaussianKernel_taylor3_bound
      have h_neg_z_le : -z ≤ 1 := by rw [← hz_eq]; exact hz
      rcases lt_or_eq_of_le ht.2 with h_t_neg | h_t_zero
      · rw [abs_of_neg h_t_neg]; linarith [ht.1]
      · rw [h_t_zero]; simp
    have h_step1 :
        |∫ t in Set.Ioc z 0,
          gaussianKernel t - (1 - t ^ 2 + t ^ 4 / 2 - t ^ 6 / 6)|
          ≤ ∫ t in Set.Ioc z 0,
              |gaussianKernel t - (1 - t ^ 2 + t ^ 4 / 2 - t ^ 6 / 6)| :=
      MeasureTheory.abs_integral_le_integral_abs
    have h_step2 : ∫ t in Set.Ioc z 0,
            |gaussianKernel t - (1 - t ^ 2 + t ^ 4 / 2 - t ^ 6 / 6)|
          ≤ ∫ t in Set.Ioc z 0, 5 * t ^ 8 / 96 := by
      apply MeasureTheory.setIntegral_mono_on
      · exact (continuous_gaussianKernel.sub (by fun_prop)).abs.integrableOn_Ioc
      · exact (continuous_const.mul (continuous_id.pow 8) |>.div_const _).integrableOn_Ioc
      · exact measurableSet_Ioc
      · exact h_pointwise
    have h_step3 : ∫ t in Set.Ioc z 0, 5 * t ^ 8 / 96 = -5 * z ^ 9 / 864 := by
      rw [show ∫ t in Set.Ioc z 0, 5 * t ^ 8 / 96 = ∫ t in z..(0:ℝ), 5 * t ^ 8 / 96 from
            (intervalIntegral.integral_of_le hz_neg.le).symm]
      have : ∫ t in z..(0:ℝ), 5 * t ^ 8 / 96 = ∫ t in z..(0:ℝ), (5/96) * t ^ 8 := by
        apply intervalIntegral.integral_congr; intro t _; ring
      rw [this, intervalIntegral.integral_const_mul, integral_pow]; ring
    have h_z9 : |z| ^ 9 = -z ^ 9 := by
      rw [show |z| ^ 9 = |z ^ 9| from by rw [abs_pow]]
      apply abs_of_neg
      have h_z8_pos : 0 < z ^ 8 := by
        have hne : z ≠ 0 := ne_of_lt hz_neg
        positivity
      have : z ^ 9 = z * z ^ 8 := by ring
      rw [this]; exact mul_neg_of_neg_of_pos hz_neg h_z8_pos
    rw [h_z9]; linarith

/-- Septenary Taylor bound for `realErf` on the unit ball:
`|realErf z − (2/√π)·(z − z³/3 + z⁵/10 − z⁷/42)| ≤ (2/√π)·5·|z|⁹/864`
for `|z| ≤ 1`. -/
theorem abs_realErf_sub_taylor7_le {z : ℝ} (hz : |z| ≤ 1) :
    |realErf z - (2 / Real.sqrt Real.pi) *
        (z - z ^ 3 / 3 + z ^ 5 / 10 - z ^ 7 / 42)|
      ≤ (2 / Real.sqrt Real.pi) * (5 * |z| ^ 9 / 864) := by
  have h_factor :
      realErf z - (2 / Real.sqrt Real.pi) *
          (z - z ^ 3 / 3 + z ^ 5 / 10 - z ^ 7 / 42)
        = (2 / Real.sqrt Real.pi) *
            ((∫ t in (0)..z, gaussianKernel t)
              - (z - z ^ 3 / 3 + z ^ 5 / 10 - z ^ 7 / 42)) := by
    unfold realErf; ring
  rw [h_factor, abs_mul,
      show |2 / Real.sqrt Real.pi| = 2 / Real.sqrt Real.pi from
        abs_of_pos two_div_sqrt_pi_pos]
  exact mul_le_mul_of_nonneg_left
    (integral_gaussianKernel_taylor_bound hz) two_div_sqrt_pi_pos.le

end VeriTile.Math
