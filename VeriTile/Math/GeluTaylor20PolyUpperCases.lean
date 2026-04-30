/-
VeriTile.Math.GeluTaylor20PolyUpperCases

Case lemmas for the upper half of the Taylor-20 polynomial bound.
-/

import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import VeriTile.Math.GeluTaylor20PolyDef

namespace VeriTile.Math

set_option maxHeartbeats 700000 in
theorem geluError_mid_taylor20_upper_left
    (x : ℝ) (_hx : x ∈ Set.Icc (83 / 100 : ℝ) (19 / 5))
    (h2 : x ≤ 463 / 200) :
    geluError_mid_taylor20_forCert x ≤ 50 / 100000 := by
  norm_num [geluError_mid_taylor20_forCert] at *
  by_contra h_contra
  by_cases h3 : x - 463 / 200 ≤ 0
  · by_cases h4 : x - 463 / 200 ≥ -1
    · nlinarith only [h4, h3, h_contra,
        pow_nonneg (sub_nonneg_of_le h3) 2,
        pow_nonneg (sub_nonneg_of_le h3) 3,
        pow_nonneg (sub_nonneg_of_le h3) 4,
        pow_nonneg (sub_nonneg_of_le h3) 5,
        pow_nonneg (sub_nonneg_of_le h3) 6,
        pow_nonneg (sub_nonneg_of_le h3) 7,
        pow_nonneg (sub_nonneg_of_le h3) 8,
        pow_nonneg (sub_nonneg_of_le h3) 9,
        pow_nonneg (sub_nonneg_of_le h3) 10,
        pow_nonneg (sub_nonneg_of_le h3) 11,
        pow_nonneg (sub_nonneg_of_le h3) 12,
        pow_nonneg (sub_nonneg_of_le h3) 13,
        pow_nonneg (sub_nonneg_of_le h3) 14,
        pow_nonneg (sub_nonneg_of_le h3) 15,
        pow_nonneg (sub_nonneg_of_le h3) 16,
        pow_nonneg (sub_nonneg_of_le h3) 17,
        pow_nonneg (sub_nonneg_of_le h3) 18,
        pow_nonneg (sub_nonneg_of_le h3) 19]
    · by_cases h5 : x - 463 / 200 ≤ -1.485
      · norm_num [show x = 83 / 100 by linarith] at *
      · norm_num at *
        nlinarith only [h5, h4, h3, h_contra,
          pow_pos (sub_pos.mpr h4) 2,
          pow_pos (sub_pos.mpr h4) 3,
          pow_pos (sub_pos.mpr h4) 4,
          pow_pos (sub_pos.mpr h4) 5,
          pow_pos (sub_pos.mpr h4) 6,
          pow_pos (sub_pos.mpr h4) 7,
          pow_pos (sub_pos.mpr h4) 8,
          pow_pos (sub_pos.mpr h4) 9,
          pow_pos (sub_pos.mpr h4) 10,
          pow_pos (sub_pos.mpr h4) 11,
          pow_pos (sub_pos.mpr h4) 12,
          pow_pos (sub_pos.mpr h4) 13,
          pow_pos (sub_pos.mpr h4) 14,
          pow_pos (sub_pos.mpr h4) 15,
          pow_pos (sub_pos.mpr h4) 16,
          pow_pos (sub_pos.mpr h4) 17,
          pow_pos (sub_pos.mpr h4) 18]
  · linarith

set_option maxHeartbeats 700000 in
theorem geluError_mid_taylor20_upper_right
    (x : ℝ) (hx : x ∈ Set.Icc (83 / 100 : ℝ) (19 / 5))
    (h2 : ¬ x ≤ 463 / 200) :
    geluError_mid_taylor20_forCert x ≤ 50 / 100000 := by
  norm_num [geluError_mid_taylor20_forCert] at *
  by_cases h5 : x ≤ 463 / 200 + 1 / 2
  · by_contra h_contra
    norm_num +zetaDelta at *
    nlinarith only [h2, h5, h_contra,
      pow_nonneg (sub_nonneg_of_le h2.le) 2,
      pow_nonneg (sub_nonneg_of_le h2.le) 3,
      pow_nonneg (sub_nonneg_of_le h2.le) 4,
      pow_nonneg (sub_nonneg_of_le h2.le) 5,
      pow_nonneg (sub_nonneg_of_le h2.le) 6,
      pow_nonneg (sub_nonneg_of_le h2.le) 7,
      pow_nonneg (sub_nonneg_of_le h2.le) 8,
      pow_nonneg (sub_nonneg_of_le h2.le) 9,
      pow_nonneg (sub_nonneg_of_le h2.le) 10,
      pow_nonneg (sub_nonneg_of_le h2.le) 11,
      pow_nonneg (sub_nonneg_of_le h2.le) 12,
      pow_nonneg (sub_nonneg_of_le h2.le) 13,
      pow_nonneg (sub_nonneg_of_le h2.le) 14,
      pow_nonneg (sub_nonneg_of_le h2.le) 15,
      pow_nonneg (sub_nonneg_of_le h2.le) 16,
      pow_nonneg (sub_nonneg_of_le h2.le) 17,
      pow_nonneg (sub_nonneg_of_le h2.le) 18,
      pow_nonneg (sub_nonneg_of_le h2.le) 19]
  · by_cases h6 : x ≤ 463 / 200 + 1
    · by_contra h_contra
      norm_num at *
      nlinarith only [h5, h6,
        pow_pos (sub_pos.mpr h5) 2,
        pow_pos (sub_pos.mpr h5) 3,
        pow_pos (sub_pos.mpr h5) 4,
        pow_pos (sub_pos.mpr h5) 5,
        pow_pos (sub_pos.mpr h5) 6,
        pow_pos (sub_pos.mpr h5) 7,
        pow_pos (sub_pos.mpr h5) 8,
        pow_pos (sub_pos.mpr h5) 9,
        pow_pos (sub_pos.mpr h5) 10,
        pow_pos (sub_pos.mpr h5) 11,
        pow_pos (sub_pos.mpr h5) 12,
        pow_pos (sub_pos.mpr h5) 13,
        pow_pos (sub_pos.mpr h5) 14,
        pow_pos (sub_pos.mpr h5) 15,
        pow_pos (sub_pos.mpr h5) 16,
        pow_pos (sub_pos.mpr h5) 17,
        pow_pos (sub_pos.mpr h5) 18,
        h_contra]
    · nlinarith only [hx.2, h2, h5, h6,
        pow_nonneg (sub_nonneg.2 <| le_of_not_ge h6) 2,
        pow_nonneg (sub_nonneg.2 <| le_of_not_ge h6) 3,
        pow_nonneg (sub_nonneg.2 <| le_of_not_ge h6) 4,
        pow_nonneg (sub_nonneg.2 <| le_of_not_ge h6) 5,
        pow_nonneg (sub_nonneg.2 <| le_of_not_ge h6) 6,
        pow_nonneg (sub_nonneg.2 <| le_of_not_ge h6) 7,
        pow_nonneg (sub_nonneg.2 <| le_of_not_ge h6) 8,
        pow_nonneg (sub_nonneg.2 <| le_of_not_ge h6) 9,
        pow_nonneg (sub_nonneg.2 <| le_of_not_ge h6) 10]

end VeriTile.Math
