/-
VeriTile.Math.GeluTaylor20PolyLowerCases

Case lemmas for the lower half of the Taylor-20 polynomial bound.
-/

import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import VeriTile.Math.GeluTaylor20PolyDef

namespace VeriTile.Math

set_option maxHeartbeats 500000 in
theorem geluError_mid_taylor20_lower_left
    (x : ℝ) (hx : x ∈ Set.Icc (83 / 100 : ℝ) (19 / 5))
    (h2 : x ≤ 463 / 200) :
    -(50 / 100000) ≤ geluError_mid_taylor20_forCert x := by
  norm_num [geluError_mid_taylor20_forCert] at *
  have hmul0 := mul_le_mul_of_nonneg_left h2 (sub_nonneg_of_le hx.1)
  have hmul1 := mul_le_mul_of_nonneg_left hmul0 (sq_nonneg (x - 463 / 200))
  have hmul2 := mul_le_mul_of_nonneg_left hmul1 (sq_nonneg (x - 463 / 200))
  have hmul3 := mul_le_mul_of_nonneg_left hmul2 (sq_nonneg (x - 463 / 200))
  have hmul4 := mul_le_mul_of_nonneg_left hmul3 (sq_nonneg (x - 463 / 200))
  have _hmul5 := mul_le_mul_of_nonneg_left hmul4 (sq_nonneg (x - 463 / 200))
  nlinarith [pow_nonneg (sub_nonneg.mpr h2) 3,
    pow_nonneg (sub_nonneg.mpr h2) 4,
    pow_nonneg (sub_nonneg.mpr h2) 5,
    pow_nonneg (sub_nonneg.mpr h2) 6,
    pow_nonneg (sub_nonneg.mpr h2) 7,
    pow_nonneg (sub_nonneg.mpr h2) 8,
    pow_nonneg (sub_nonneg.mpr h2) 9,
    pow_nonneg (sub_nonneg.mpr h2) 10]

set_option maxHeartbeats 500000 in
theorem geluError_mid_taylor20_lower_right
    (x : ℝ) (hx : x ∈ Set.Icc (83 / 100 : ℝ) (19 / 5))
    (h2 : ¬ x ≤ 463 / 200) :
    -(50 / 100000) ≤ geluError_mid_taylor20_forCert x := by
  norm_num [geluError_mid_taylor20_forCert] at *
  obtain ⟨t, ht_pos, _ht_eq⟩ : ∃ t > 0, x = 463 / 200 + t := by
    exact ⟨x - 463 / 200, by linarith, by ring⟩
  subst x
  norm_num at *
  have _hmul := mul_le_mul_of_nonneg_left hx.2 (sq_nonneg t)
  nlinarith [pow_pos ht_pos 3, pow_pos ht_pos 4, pow_pos ht_pos 5,
    pow_pos ht_pos 6, pow_pos ht_pos 7, pow_pos ht_pos 8,
    pow_pos ht_pos 9, pow_pos ht_pos 10, pow_pos ht_pos 11,
    pow_pos ht_pos 12, pow_pos ht_pos 13, pow_pos ht_pos 14,
    pow_pos ht_pos 15, pow_pos ht_pos 16, pow_pos ht_pos 17,
    pow_pos ht_pos 18, pow_pos ht_pos 19, pow_pos ht_pos 20]

end VeriTile.Math
