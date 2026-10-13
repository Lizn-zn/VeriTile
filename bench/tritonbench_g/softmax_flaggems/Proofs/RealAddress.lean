import Mathlib.Data.Nat.Basic
import Mathlib.Tactic.Common

/-! Discrete coverage and injectivity for the actual FlagGems reverse writeback.
All dimensions are unbounded naturals; hardware address overflow is separate. -/
namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealAddress

def rounds (N B : Nat) : Nat := (N + B - 1) / B
def previous (N B : Nat) : Nat := rounds N B * B - B

theorem previous_eq (N B : Nat) : previous N B = (rounds N B - 1) * B := by
  simp [previous, Nat.sub_mul]

theorem rounds_positive {N B : Nat} (hN : 0 < N) (hB : 0 < B) :
    0 < rounds N B := by
  unfold rounds
  apply Nat.div_pos <;> omega

theorem coverage {N B : Nat} (hB : 0 < B) : N ≤ rounds N B * B := by
  have hrem := Nat.mod_lt (N + B - 1) hB
  have heq := Nat.mod_add_div (N + B - 1) B
  rw [Nat.mul_comm B] at heq
  unfold rounds
  omega

theorem previous_lt {N B : Nat} (hN : 0 < N) (hB : 0 < B) : previous N B < N := by
  have hdiv := Nat.div_mul_le_self (N + B - 1) B
  have hq := rounds_positive hN hB
  have hprod : B ≤ rounds N B * B := by
    simpa using Nat.mul_le_mul_right B hq
  unfold previous rounds at *
  omega

/-- The dynamic source loop takes exactly the ceiling number of tile steps. -/
theorem tile_start_lt_iff {N B c : Nat} (hN : 0 < N) (hB : 0 < B) :
    c * B < N ↔ c < rounds N B := by
  constructor
  · intro hc
    by_contra h
    have hmul := Nat.mul_le_mul_right B (Nat.le_of_not_gt h)
    have hcover := coverage (N := N) hB
    omega
  · intro hc
    have hle : c ≤ rounds N B - 1 := by omega
    have hmul := Nat.mul_le_mul_right B hle
    have hlast := previous_lt hN hB
    rw [previous_eq] at hlast
    omega

/-- The source's subtraction enumerates tile bases in reverse order. -/
theorem reverse_base (N B tile : Nat) :
    previous N B - tile * B = (rounds N B - 1 - tile) * B := by
  rw [previous_eq]
  exact (Nat.sub_mul (rounds N B - 1) tile B).symm

/-- The reverse loop visits every tile exactly once. -/
def reverseTile {q : Nat} (t : Fin q) : Fin q :=
  ⟨q - 1 - t.val, by have := t.isLt; omega⟩

theorem reverseTile_involution {q : Nat} (t : Fin q) :
    reverseTile (reverseTile t) = t := by
  apply Fin.ext
  have := t.isLt
  simp only [reverseTile]
  omega

theorem reverseTile_bijective (q : Nat) : Function.Bijective (@reverseTile q) :=
  Function.Involutive.bijective reverseTile_involution

/-- Distinct active row/column coordinates never alias inside a contiguous matrix. -/
theorem row_column_injective {K n₁ n₂ k₁ k₂ : Nat} (hk₁ : k₁ < K) (hk₂ : k₂ < K)
    (h : n₁ * K + k₁ = n₂ * K + k₂) : n₁ = n₂ ∧ k₁ = k₂ := by
  have hk := congrArg (fun a ↦ a % K) h
  dsimp only at hk
  rw [Nat.mul_add_mod_of_lt hk₁, Nat.mul_add_mod_of_lt hk₂] at hk
  refine ⟨?_, hk⟩
  rw [hk] at h
  have hm := Nat.add_right_cancel h
  exact Nat.eq_of_mul_eq_mul_right (by omega) hm

/-- Bounds for the actual non-inner address, including the program row. -/
theorem non_inner_bound {M N K row n k : Nat}
    (hr : row < M) (hn : n < N) (hk : k < K) :
    row * N * K + n * K + k < M * N * K := by
  have hrow : row * N + n < M * N := by
    have hstep := Nat.mul_le_mul_right N hr
    have : row * N + n < (row + 1) * N := by
      simp only [Nat.add_mul, Nat.one_mul]; omega
    exact Nat.lt_of_lt_of_le this hstep
  have hstep := Nat.mul_le_mul_right K hrow
  have : (row * N + n) * K + k < (row * N + n + 1) * K := by
    simp only [Nat.add_mul, Nat.one_mul]; omega
  simpa [Nat.add_mul] using Nat.lt_of_lt_of_le this hstep

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealAddress
