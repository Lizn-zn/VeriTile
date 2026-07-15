/-
VeriTile.Triton.Math.Reduction

Reusable reduction-shaped mathematical specifications: sum and max over a
non-empty tile, two-pass arithmetic mean and population variance, layer
normalization.

This is the pure-math layer. Definitions here only depend on Mathlib's
big-operator infrastructure and `Real.sqrt`; they do not touch `BlockState`,
`RegionName`, or memory layout. The running Welford recurrence and its
equivalence to the two-pass closed form live in the `WelfordRec` sub-namespace
below — pure ℝ math, shared (via `open TiledReduction.WelfordRec`) by the
`bench/examples` Welford/LayerNorm showcases.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Finset.Lattice.Fold

namespace VeriTile.Triton

namespace TiledReduction

/-! ## Sum and max -/

/-- Sum over a tile of length `N`. -/
noncomputable def tileSum {N : Nat} (xs : Fin N → ℝ) : ℝ := ∑ i, xs i

/-- Maximum over a non-empty tile. The non-emptiness hypothesis mirrors
`Tile.reduceMax`, which is only defined on positive-length axes since
`Finset.sup'` requires a non-empty index set. -/
noncomputable def tileMax {N : Nat} (h : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  match N, h, xs with
  | _ + 1, _, xs => Finset.univ.sup' Finset.univ_nonempty xs

/-! ## Two-pass / Welford mean and variance -/

/-- Two-pass arithmetic mean: `μ = (∑ xᵢ) / N`. Coerces `N : Nat` to `ℝ`
through Lean's natural numeric coercion. -/
noncomputable def welfordMean {N : Nat} (xs : Fin N → ℝ) : ℝ := tileSum xs / N

/-- Sum of squared deviations from the mean: `S = ∑ (xᵢ − μ)²`. -/
noncomputable def welfordSumSq {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  ∑ i, (xs i - welfordMean xs) ^ 2

/-- Population variance: `σ² = S / N`. -/
noncomputable def welfordVar {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  welfordSumSq xs / N

/-! ## LayerNorm -/

/-- LayerNorm: `(xᵢ − μ) / √(var + ε) · γᵢ + βᵢ`, where `μ`, `var` are the
two-pass mean and population variance over the input tile. -/
noncomputable def layerNorm {N : Nat}
    (xs γs βs : Fin N → ℝ) (ε : ℝ) (i : Fin N) : ℝ :=
  (xs i - welfordMean xs) / Real.sqrt (welfordVar xs + ε) * γs i + βs i

/-! ## Running Welford recurrence and its two-pass equivalence

The kernel-internal running Welford recurrence, extracted as reusable pure-ℝ
math (it does not interleave with kernel step semantics). The `bench/examples`
Welford/LayerNorm showcases share this
cluster via `open TiledReduction.WelfordRec`; the names shadow the closed-form
`welfordMean`/`welfordSumSq` above only inside this sub-namespace. -/
namespace WelfordRec

/-- Two-pass mean: μ = (∑ xᵢ) / n. -/
noncomputable def twoPassMean {n : Nat} (x : Fin n → ℝ) : ℝ :=
  (∑ i, x i) / n

/-- Two-pass sum-of-squared-deviations: S = ∑ (xᵢ − μ)². -/
noncomputable def twoPassS {n : Nat} (x : Fin n → ℝ) : ℝ :=
  ∑ i, (x i - twoPassMean x) ^ 2

/-- Welford recurrence: running mean M_k after processing x[0..k-1].
    M_0 = 0, M_{k+1} = M_k + (x_k − M_k) / (k+1).
    Returns 0 if k > n (out-of-range). -/
noncomputable def welfordMean {n : Nat} (x : Fin n → ℝ) : Nat → ℝ
  | 0     => 0
  | k + 1 =>
      if h : k < n then
        let prev := welfordMean x k
        prev + (x ⟨k, h⟩ - prev) / (k + 1)
      else welfordMean x k

/-- Welford recurrence: running sum-of-squared-deviations S_k.
    S_0 = 0, S_{k+1} = S_k + (x_k − M_k) · (x_k − M_{k+1}). -/
noncomputable def welfordS {n : Nat} (x : Fin n → ℝ) : Nat → ℝ
  | 0     => 0
  | k + 1 =>
      if h : k < n then
        let prevM := welfordMean x k
        let curM  := welfordMean x (k + 1)
        welfordS x k + (x ⟨k, h⟩ - prevM) * (x ⟨k, h⟩ - curM)
      else welfordS x k

/-- Helper: for any prefix length k ≤ n, the running Welford mean times k
    equals the sum of the first k inputs. Used to derive the final
    `welfordMean x n = twoPassMean x` claim. -/
theorem welford_mean_mul_eq_sum {n : Nat} (x : Fin n → ℝ) :
    ∀ k : Nat, ∀ (h : k ≤ n),
      welfordMean x k * k = ∑ i : Fin k, x (Fin.castLE h i) := by
  intro k
  induction k with
  | zero =>
    intro _
    simp [welfordMean]
  | succ j ih =>
    intro hk
    have hj : j ≤ n := Nat.le_of_succ_le hk
    have hj_lt : j < n := hk
    have ih' := ih hj
    have hwm : welfordMean x (j + 1) =
        welfordMean x j + (x ⟨j, hj_lt⟩ - welfordMean x j) / (j + 1) := by
      simp [welfordMean, hj_lt]
    rw [hwm]
    have hjp1_ne : ((j : ℝ) + 1) ≠ 0 := by
      have : (0 : ℝ) ≤ j := Nat.cast_nonneg j
      linarith
    have hcast : ((j + 1 : Nat) : ℝ) = (j : ℝ) + 1 := by push_cast; ring
    rw [hcast]
    have hlhs : (welfordMean x j + (x ⟨j, hj_lt⟩ - welfordMean x j) / ((j : ℝ) + 1))
                  * ((j : ℝ) + 1) = welfordMean x j * j + x ⟨j, hj_lt⟩ := by
      field_simp
      ring
    rw [hlhs]
    rw [Fin.sum_univ_castSucc]
    have h_last : x (Fin.castLE hk (Fin.last j)) = x ⟨j, hj_lt⟩ := by
      rfl
    have h_cs : ∀ i : Fin j, x (Fin.castLE hk i.castSucc) = x (Fin.castLE hj i) := by
      intro i; rfl
    simp only [h_cs, h_last]
    rw [ih']

/-- Welford's variance identity. For any prefix length k ≤ n,
    the running Welford S_k equals the sum-of-squared-deviations from M_k. -/
theorem welford_S_eq_sum_sq_dev {n : Nat} (x : Fin n → ℝ) :
    ∀ k : Nat, ∀ (hk : k ≤ n),
      welfordS x k = ∑ i : Fin k, (x (Fin.castLE hk i) - welfordMean x k) ^ 2 := by
  intro k
  induction k with
  | zero =>
    intro _
    simp [welfordS]
  | succ j ih =>
    intro hk
    have hj : j ≤ n := Nat.le_of_succ_le hk
    have hj_lt : j < n := hk
    have ih' := ih hj
    set M := welfordMean x j with hMdef
    set M' := welfordMean x (j + 1) with hM'def
    set xj := x ⟨j, hj_lt⟩ with hxjdef
    have hM' : M' = M + (xj - M) / ((j : ℝ) + 1) := by
      simp [hM'def, welfordMean, hj_lt, hMdef, hxjdef]
    have hS' : welfordS x (j + 1) = welfordS x j + (xj - M) * (xj - M') := by
      simp [welfordS, hj_lt, hMdef, hM'def, hxjdef]
    have hjp1_pos : (0 : ℝ) < (j : ℝ) + 1 := by
      have : (0 : ℝ) ≤ j := Nat.cast_nonneg j
      linarith
    have hjp1_ne : ((j : ℝ) + 1) ≠ 0 := ne_of_gt hjp1_pos
    have hMean := welford_mean_mul_eq_sum x j hj
    have hMM' : ((j : ℝ) + 1) * (M' - M) = xj - M := by
      rw [hM']; field_simp; ring
    have hxj_M' : xj - M' = (j : ℝ) * (M' - M) := by
      have : xj - M' = (xj - M) - (M' - M) := by ring
      rw [this, ← hMM']; ring
    have hxj_M : xj - M = ((j : ℝ) + 1) * (M' - M) := hMM'.symm
    rw [hS', ih']
    rw [Fin.sum_univ_castSucc]
    have h_last : x (Fin.castLE hk (Fin.last j)) = xj := rfl
    have h_cs : ∀ i : Fin j, x (Fin.castLE hk i.castSucc) = x (Fin.castLE hj i) := by
      intro i; rfl
    simp only [h_cs, h_last]
    have key :
        (∑ i : Fin j, (x (Fin.castLE hj i) - M') ^ 2) -
          (∑ i : Fin j, (x (Fin.castLE hj i) - M) ^ 2)
        = (j : ℝ) * (M - M') ^ 2 := by
      rw [← Finset.sum_sub_distrib]
      have h_per : ∀ i : Fin j,
          (x (Fin.castLE hj i) - M') ^ 2 - (x (Fin.castLE hj i) - M) ^ 2 =
          (M - M') * (2 * x (Fin.castLE hj i) - M - M') := by
        intro i; ring
      simp_rw [h_per]
      rw [← Finset.mul_sum]
      have h_sum_split : ∀ i : Fin j,
          2 * x (Fin.castLE hj i) - M - M' =
            2 * x (Fin.castLE hj i) + (- M - M') := by
        intro i; ring
      simp_rw [h_sum_split]
      rw [Finset.sum_add_distrib, ← Finset.mul_sum]
      rw [Finset.sum_const]
      simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      have hSumX : ∑ i : Fin j, x (Fin.castLE hj i) = M * j := hMean.symm
      rw [hSumX]
      ring
    have lhs_alg : (xj - M) * (xj - M') - (xj - M') ^ 2
                 = (j : ℝ) * (M - M') ^ 2 := by
      rw [hxj_M, hxj_M']
      ring
    linarith [key, lhs_alg]

/-- The load-bearing identity for Welford kernel refinement: after processing
all `n` inputs, Welford's running `(M, S)` equals the two-pass `(μ, S)`. -/
theorem welford_eq_two_pass {n : Nat} (hn : 0 < n) (x : Fin n → ℝ) :
    welfordMean x n = twoPassMean x ∧ welfordS x n = twoPassS x := by
  refine ⟨?_, ?_⟩
  · have hMul := welford_mean_mul_eq_sum x n (le_refl n)
    have hn_pos : (0 : ℝ) < n := by exact_mod_cast hn
    have hn_ne : (n : ℝ) ≠ 0 := ne_of_gt hn_pos
    unfold twoPassMean
    rw [eq_div_iff hn_ne, hMul]
    apply Finset.sum_congr rfl
    intro i _
    rfl
  · have hS := welford_S_eq_sum_sq_dev x n (le_refl n)
    have hM : welfordMean x n = twoPassMean x := by
      have hMul := welford_mean_mul_eq_sum x n (le_refl n)
      have hn_pos : (0 : ℝ) < n := by exact_mod_cast hn
      have hn_ne : (n : ℝ) ≠ 0 := ne_of_gt hn_pos
      unfold twoPassMean
      rw [eq_div_iff hn_ne, hMul]
      apply Finset.sum_congr rfl
      intro i _; rfl
    rw [hS, hM]
    unfold twoPassS
    apply Finset.sum_congr rfl
    intro i _
    rfl

end WelfordRec

end TiledReduction

end VeriTile.Triton
