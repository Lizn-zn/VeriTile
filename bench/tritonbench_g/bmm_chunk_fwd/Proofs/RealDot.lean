import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealLoads
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealDot

/-! The real value of a masked dot block, derived from the current pointer
and memory tiles. The partial contraction tail remains in the sum. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealDot
open VeriTile.Triton RealLoads
noncomputable section
set_option linter.unusedSimpArgs false

open BmmChunkBwd.RealDot (finite_dot)

def value {BM BN BK : Nat} (s : BlockState)
    (dp : Tile .ptr [BM, BK]) (ap : Tile .ptr [BK, BN])
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (length K c : Nat)
    (i : Fin BM) (j : Fin BN) : ℝ :=
  if rows i < length ∧ columns j < length then
    ∑ k : Fin BK, if k.val < K - c * BK then
      s.readMem (dp.data (i,k,PUnit.unit)).1 (dp.data (i,k,PUnit.unit)).2 *
      s.readMem (ap.data (k,j,PUnit.unit)).1 (ap.data (k,j,PUnit.unit)).2 else 0
  else 0

theorem masked_dot (BM BN BK : Nat) (s : BlockState)
    (dp : Tile .ptr [BM, BK]) (ap : Tile .ptr [BK, BN])
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (length K c : Nat) :
    Tile.dot [] (aValues s dp rows length K c) (bValues s ap columns length K c) =
      Tile.mat (fun i j => some (value s dp ap rows columns length K c i j)) := by
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  rw [finite_dot BM BK BN _ _ i j _ _ (by intro e; rfl) (by intro e; rfl)]
  apply congrArg some
  by_cases hr : rows i < length <;> by_cases hc : columns j < length
  · simp only [value, hr, hc, true_and, ↓reduceIte]
    apply Finset.sum_congr rfl
    intro k _
    by_cases hk : k.val < K - c * BK <;>
      simp [aValues, bValues, Tile.mat, hr, hc, hk]
  all_goals simp [value, aValues, bValues, Tile.mat, hr, hc]

/-- The source's accumulation op consumes the memory-derived tiles. -/
theorem accumulation_eval (swapped : Bool) (BM BN BK : Nat) (s : BlockState)
    (dp : Tile .ptr [BM, BK]) (ap : Tile .ptr [BK, BN])
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (length K c : Nat)
    (carry : Fin BM → Fin BN → ℝ)
    (hz : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j)))
    (hd : s.regs .real [BM, BK] "a" = some (aValues s dp rows length K c))
    (ha : s.regs .real [BK, BN] "b" = some (bValues s ap columns length K c)) :
    evalOp (RealLoads.update swapped BM BN BK) s =
      some (Tile.mat fun i j => some (carry i j + value s dp ap rows columns length K c i j)) := by
  have hdot : evalOp (.dot (batch := []) (.ref .real [BM, BK] "a")
      (.ref .real [BK, BN] "b")) s =
      some (Tile.mat fun i j => some (value s dp ap rows columns length K c i j)) := by
    rw [evalOp_dot]
    simp [hd, ha, masked_dot]
  cases swapped
  · simp only [RealLoads.update, Bool.false_eq_true, ↓reduceIte, evalOp_add,
      evalOp_ref, hz, bind, Option.bind_some]
    erw [hdot]
    rfl
  · simp only [RealLoads.update, ↓reduceIte, evalOp_add]
    erw [hdot]
    simp only [evalOp_ref, hz, bind, Option.bind_some]
    apply congrArg some
    apply Tile.ext
    rintro ⟨i,j,⟨⟩⟩
    simp [Tile.bop, Tile.mat, NumericDType.add, Broadcast.leftIndex,
      Broadcast.rightIndex, add_comm]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealDot
