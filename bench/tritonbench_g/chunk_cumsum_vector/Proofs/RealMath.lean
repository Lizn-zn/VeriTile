import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Independent prefix-sum partition, usable separately for each column.
The carry includes every supplied value, including unspecified tail lanes. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealMath
open VeriTile.Triton
open scoped BigOperators
set_option maxHeartbeats 1000000

noncomputable def carry (f : Nat → ℝ) (BT c : Nat) : ℝ :=
  ∑ flat ∈ Finset.range (c * BT), f flat

/-- Global cumulative sum at index `flat` of an independent input function. -/
noncomputable def prefixValue (f : Nat → ℝ) (T flat : Nat) : ℝ :=
  ∑ m ∈ (Finset.range T).filter (fun m => m ≤ flat), f m

/-- A valid output prefix never includes terms from later padding lanes. -/
theorem prefix_from_chunk (f : Nat → ℝ) (T BT c : Nat) (i : Fin BT)
    (hi : c * BT + i.val < T) :
    (∑ k ∈ Finset.univ.filter (fun k : Fin BT => k.val ≤ i.val), f (c * BT + k.val))
      + carry f BT c = prefixValue f T (c * BT + i.val) := by
  have hchunk :
      (∑ k ∈ Finset.univ.filter (fun k : Fin BT => k.val ≤ i.val), f (c * BT + k.val)) =
        ∑ k ∈ Finset.range (i.val + 1), f (c * BT + k) := by
    apply Finset.sum_bij (fun k _ => k.val)
    · intro k hk
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk
      exact Finset.mem_range.mpr (by omega)
    · intro a _ b _ he; exact Fin.ext he
    · intro k hk
      have hk' := Finset.mem_range.mp hk
      refine ⟨⟨k, by omega⟩, ?_, rfl⟩
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      omega
    · intro _ _; rfl
  have hset : (Finset.range T).filter (fun m => m ≤ c * BT + i.val) =
      Finset.range (c * BT + (i.val + 1)) := by
    ext m
    simp only [Finset.mem_filter, Finset.mem_range]
    omega
  rw [hchunk, carry, add_comm, ← Finset.sum_range_add, prefixValue, hset]

/-- The exact carry recurrence also covers the partial final chunk. -/
theorem carry_succ (f : Nat → ℝ) (BT c : Nat) :
    carry f BT (c + 1) = carry f BT c + ∑ k : Fin BT, f (c * BT + k.val) := by
  simp only [carry, Nat.add_mul, Nat.one_mul, Finset.sum_range_add]
  congr 1
  exact (Fin.sum_univ_eq_sum_range (fun k => f (c * BT + k)) BT).symm

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealMath
