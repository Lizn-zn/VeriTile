/-
VeriTile.Examples.ApproxGeLU.Taylor9

Split-out support for the ApproxGeLU example.
-/

import VeriTile.Examples.ApproxGeLU.Taylor5

namespace VeriTile.Examples

open VeriTile VeriTile.Math

/-- 7th-order tanh Taylor polynomial: `u − u³/3 + 2u⁵/15 − 17u⁷/315`. -/
noncomputable def tanhTaylor7 (u : ℝ) : ℝ :=
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
lemma tanhTaylor7_sub_rationalErf_eq (c k x : ℝ) :
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
lemma poly_diff_bound_075 {x : ℝ} (hx : |x| ≤ 3/4) :
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
lemma poly_diff_with_correction_bound_075 {x : ℝ} (hx : |x| ≤ 3/4) :
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
lemma u_bound_at_075 {x : ℝ} (hx : |x| ≤ 3/4) :
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
lemma tanh_residual_at_075 {x : ℝ} (hx : |x| ≤ 3/4) :
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
lemma erf_residual_at_075 {x : ℝ} (hx : |x| ≤ 3/4) :
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
theorem approx_gelu_error_bound_075 {x : ℝ} (hx : |x| ≤ 3/4) :
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
noncomputable def tanhTaylor9 (u : ℝ) : ℝ :=
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
lemma tanhTaylor9_sub_rationalErf_eq (c k x : ℝ) :
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
lemma poly_diff_bound_08 {x : ℝ} (hx : |x| ≤ 4/5) :
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
lemma poly_diff_with_correction_bound_08 {x : ℝ} (hx : |x| ≤ 4/5) :
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
lemma u_bound_at_08 {x : ℝ} (hx : |x| ≤ 4/5) :
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
lemma tanh_residual_at_08 {x : ℝ} (hx : |x| ≤ 4/5) :
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
lemma erf_residual_at_08 {x : ℝ} (hx : |x| ≤ 4/5) :
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
theorem approx_gelu_error_bound_08 {x : ℝ} (hx : |x| ≤ 4/5) :
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
lemma poly_diff_bound_083 {x : ℝ} (hx : |x| ≤ 83/100) :
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
lemma poly_diff_with_correction_bound_083 {x : ℝ} (hx : |x| ≤ 83/100) :
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
lemma u_bound_at_083 {x : ℝ} (hx : |x| ≤ 83/100) :
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
lemma tanh_residual_at_083 {x : ℝ} (hx : |x| ≤ 83/100) :
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
lemma erf_residual_at_083 {x : ℝ} (hx : |x| ≤ 83/100) :
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
theorem approx_gelu_error_bound_083 {x : ℝ} (hx : |x| ≤ 83/100) :
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


end VeriTile.Examples
