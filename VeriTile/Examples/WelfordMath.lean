/-
VeriTile.Examples.WelfordMath

Math-only lemma: Welford's online variance recurrence equals the two-pass
formula. This is preparation for Phase B's `welford_kernels_refinement`
theorem (#4); the kernel-level lift will use `forLoop_inv` once forLoop
semantics is in place.

No Triton kernels, no operational semantics — pure ℝ-arithmetic.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

namespace VeriTile.Examples

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

/-- Cast `Fin k` to `Fin n` when `k ≤ n`. -/
def castFin {n k : Nat} (h : k ≤ n) (i : Fin k) : Fin n :=
  ⟨i.val, lt_of_lt_of_le i.isLt h⟩

/-- Helper: for any prefix length k ≤ n, the running Welford mean times k
    equals the sum of the first k inputs. Used to derive the final
    `welfordMean x n = twoPassMean x` claim. -/
private theorem welford_mean_mul_eq_sum {n : Nat} (x : Fin n → ℝ) :
    ∀ k : Nat, ∀ (h : k ≤ n),
      welfordMean x k * k = ∑ i : Fin k, x (castFin h i) := by
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
    -- Unfold welfordMean at successor
    have hwm : welfordMean x (j + 1) =
        welfordMean x j + (x ⟨j, hj_lt⟩ - welfordMean x j) / (j + 1) := by
      simp [welfordMean, hj_lt]
    rw [hwm]
    -- LHS: (welfordMean x j + (x_j - welfordMean x j) / (j+1)) * (j+1)
    --    = welfordMean x j * (j+1) + (x_j - welfordMean x j)
    --    = welfordMean x j * j + welfordMean x j + x_j - welfordMean x j
    --    = welfordMean x j * j + x_j
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
    -- RHS: ∑ i : Fin (j+1), x (castFin hk i) = (∑ i : Fin j, x (castFin hj i)) + x ⟨j, hj_lt⟩
    rw [Fin.sum_univ_castSucc]
    -- Now: ∑ i : Fin j, x (castFin hk i.castSucc) + x (castFin hk (Fin.last j))
    --     = (welfordMean x j * j) + x_j   from IH and definition of castFin
    have h_last : x (castFin hk (Fin.last j)) = x ⟨j, hj_lt⟩ := by
      rfl
    have h_cs : ∀ i : Fin j, x (castFin hk i.castSucc) = x (castFin hj i) := by
      intro i; rfl
    simp only [h_cs, h_last]
    rw [ih']

/-- Welford's variance identity. For any prefix length k ≤ n,
    the running Welford S_k equals the sum-of-squared-deviations from M_k. -/
private theorem welford_S_eq_sum_sq_dev {n : Nat} (x : Fin n → ℝ) :
    ∀ k : Nat, ∀ (hk : k ≤ n),
      welfordS x k = ∑ i : Fin k, (x (castFin hk i) - welfordMean x k) ^ 2 := by
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
    -- Notation
    set M := welfordMean x j with hMdef
    set M' := welfordMean x (j + 1) with hM'def
    set xj := x ⟨j, hj_lt⟩ with hxjdef
    -- Welford recurrences
    have hM' : M' = M + (xj - M) / ((j : ℝ) + 1) := by
      simp [hM'def, welfordMean, hj_lt, hMdef, hxjdef]
    have hS' : welfordS x (j + 1) = welfordS x j + (xj - M) * (xj - M') := by
      simp [welfordS, hj_lt, hMdef, hM'def, hxjdef]
    -- (j+1) ≠ 0 in ℝ
    have hjp1_pos : (0 : ℝ) < (j : ℝ) + 1 := by
      have : (0 : ℝ) ≤ j := Nat.cast_nonneg j
      linarith
    have hjp1_ne : ((j : ℝ) + 1) ≠ 0 := ne_of_gt hjp1_pos
    -- Mean prefix: M * j = ∑ i : Fin j, x (castFin hj i)
    have hMean := welford_mean_mul_eq_sum x j hj
    -- Key algebraic relations between M and M':
    -- (j+1)*(M' - M) = xj - M
    have hMM' : ((j : ℝ) + 1) * (M' - M) = xj - M := by
      rw [hM']; field_simp; ring
    -- xj - M' = j * (M' - M)
    have hxj_M' : xj - M' = (j : ℝ) * (M' - M) := by
      have : xj - M' = (xj - M) - (M' - M) := by ring
      rw [this, ← hMM']; ring
    -- xj - M = (j+1) * (M' - M)
    have hxj_M : xj - M = ((j : ℝ) + 1) * (M' - M) := hMM'.symm
    -- Now expand the goal
    rw [hS', ih']
    -- Goal: (∑ i : Fin j, (x (castFin hj i) - M)^2) + (xj - M)*(xj - M')
    --     = ∑ i : Fin (j+1), (x (castFin hk i) - M')^2
    rw [Fin.sum_univ_castSucc]
    -- last summand on RHS:
    have h_last : x (castFin hk (Fin.last j)) = xj := rfl
    have h_cs : ∀ i : Fin j, x (castFin hk i.castSucc) = x (castFin hj i) := by
      intro i; rfl
    simp only [h_cs, h_last]
    -- Goal: (∑ i, (x (castFin hj i) - M)^2) + (xj - M) * (xj - M')
    --     = (∑ i, (x (castFin hj i) - M')^2) + (xj - M')^2
    --
    -- Move (xj - M')^2 to LHS: equivalently
    --   (∑ i, (x (castFin hj i) - M)^2) + (xj - M) * (xj - M') - (xj - M')^2
    --     = ∑ i, (x (castFin hj i) - M')^2
    -- The diff per term: (x_i - M')^2 - (x_i - M)^2
    --   = ((x_i - M') + (x_i - M)) * ((x_i - M') - (x_i - M))
    --   = (2 x_i - M - M') * (M - M')
    -- Sum: (M - M') * (2 ∑ x_i - j(M + M')) = (M - M') * (2 M j - j(M + M'))
    --                                      = (M - M') * j * (M - M')
    --                                      = j * (M - M')^2
    -- And on the LHS: (xj - M)(xj - M') - (xj - M')^2
    --   = (xj - M')[(xj - M) - (xj - M')] = (xj - M')(M' - M)
    --   = j(M' - M) * (M' - M) = j (M' - M)^2 = j (M - M')^2.
    -- So both sides match.
    have key :
        (∑ i : Fin j, (x (castFin hj i) - M') ^ 2) -
          (∑ i : Fin j, (x (castFin hj i) - M) ^ 2)
        = (j : ℝ) * (M - M') ^ 2 := by
      rw [← Finset.sum_sub_distrib]
      have h_per : ∀ i : Fin j,
          (x (castFin hj i) - M') ^ 2 - (x (castFin hj i) - M) ^ 2 =
          (M - M') * (2 * x (castFin hj i) - M - M') := by
        intro i; ring
      simp_rw [h_per]
      rw [← Finset.mul_sum]
      -- ∑ i, (2 x_i - M - M') = 2 (∑ x_i) - j*(M + M')
      have h_sum_split : ∀ i : Fin j,
          2 * x (castFin hj i) - M - M' =
            2 * x (castFin hj i) + (- M - M') := by
        intro i; ring
      simp_rw [h_sum_split]
      rw [Finset.sum_add_distrib, ← Finset.mul_sum]
      rw [Finset.sum_const]
      simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      -- Now: (M - M') * (2 * (∑ x_i) + j * (-M - M')) = j * (M - M')^2
      -- ∑ x_i = M * j (from hMean)
      have hSumX : ∑ i : Fin j, x (castFin hj i) = M * j := hMean.symm
      rw [hSumX]
      ring
    -- Now use `key` plus the LHS algebra identity
    have lhs_alg : (xj - M) * (xj - M') - (xj - M') ^ 2
                 = (j : ℝ) * (M - M') ^ 2 := by
      rw [hxj_M, hxj_M']
      ring
    linarith [key, lhs_alg]

/-- The load-bearing identity for Phase B's Welford kernel theorem.
    `welford_eq_two_pass` says: after processing all n inputs,
    Welford's running (M, S) equals the two-pass (μ, S). -/
theorem welford_eq_two_pass {n : Nat} (hn : 0 < n) (x : Fin n → ℝ) :
    welfordMean x n = twoPassMean x ∧ welfordS x n = twoPassS x := by
  refine ⟨?_, ?_⟩
  · -- Mean part: from welford_mean_mul_eq_sum at k = n
    have hMul := welford_mean_mul_eq_sum x n (le_refl n)
    have hn_pos : (0 : ℝ) < n := by exact_mod_cast hn
    have hn_ne : (n : ℝ) ≠ 0 := ne_of_gt hn_pos
    unfold twoPassMean
    rw [eq_div_iff hn_ne, hMul]
    -- Goal: ∑ i : Fin n, x (castFin (le_refl n) i) = ∑ i, x i
    -- castFin (le_refl n) i = i (up to subtype mk)
    apply Finset.sum_congr rfl
    intro i _
    rfl
  · -- Variance part: from welford_S_eq_sum_sq_dev at k = n
    -- TODO Phase B: depends on welford_S_eq_sum_sq_dev (sorry'd above);
    -- Welford's identity proof requires non-trivial algebra over the
    -- running mean shift. Mean part composed below.
    have hS := welford_S_eq_sum_sq_dev x n (le_refl n)
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

end VeriTile.Examples
