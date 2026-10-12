import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealMath

/-! Independent input arrays for one BMM output tile. The contraction formula
uses the actual chunk length and includes the optional initial residual.
Neither the formula nor its input arrays refer to an interpreter or final state. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealInputs
open scoped BigOperators
noncomputable section

structure Inputs (BM BN chunk : Nat) where
  dout : Fin BM → Fin chunk → ℝ
  a : Fin chunk → Fin BN → ℝ
  residual : Fin BM → Fin BN → ℝ

def formula {BM BN chunk : Nat} (length : Nat) (hasResidual : Bool)
    (xs : Inputs BM BN chunk) (i : Fin BM) (j : Fin BN) : ℝ :=
  (∑ k : Fin chunk, if k.val < length then xs.dout i k * xs.a k j else 0) +
    if hasResidual then xs.residual i j else 0

/-- The array formula agrees with a range sum whenever its live entries are
pinned to the initial reads. Padding outside the actual length is arbitrary. -/
theorem sum_of_pins (chunk length : Nat) (h : length ≤ chunk)
    (xs : Fin chunk → ℝ) (values : Nat → ℝ)
    (hp : ∀ k : Fin chunk, k.val < length → values k.val = xs k) :
    (∑ k ∈ Finset.range length, values k) =
      ∑ k : Fin chunk, if k.val < length then xs k else 0 := by
  have hsum := Finset.sum_subset (f := fun k => if k < length then values k else 0)
    (Finset.range_mono h) (by intro k _ hk; simp only [Finset.mem_range] at hk; simp [hk])
  have hleft : (∑ k ∈ Finset.range length, if k < length then values k else 0) =
      ∑ k ∈ Finset.range length, values k := by
    apply Finset.sum_congr rfl
    intro k hk
    simp [Finset.mem_range.mp hk]
  rw [hleft] at hsum
  rw [hsum, Finset.sum_range]
  apply Finset.sum_congr rfl
  intro k _
  by_cases hk : k.val < length <;> simp [hk, hp k]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealInputs
