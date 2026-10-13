import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerSourceSafety
import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerIO

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 1200000

private theorem indexPrefix_safe (bounds : RegionBounds) (kind : RealCandidate.PrefixKind)
    (swapped : Bool) (B : Nat) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealCandidate.indexPrefix kind swapped B) s := by
  cases kind <;> cases swapped <;>
    simp [RealCandidate.indexPrefix, Stmt.TraceSafeList, Stmt.TraceSafe,
      Op.SafeAt.eq_def, stepStmts, stepStmt, evalOp.eq_def]

private theorem indexPrefix_exec (kind : RealCandidate.PrefixKind) (B : Nat) (s : BlockState) :
    stepStmts (RealCandidate.indexPrefix kind Bool.false B) s =
      stepStmts (RealCandidate.indexPrefix kind Bool.true B) s := by
  cases kind <;> simp [RealCandidate.indexPrefix, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, NumericDType.mul, NumericDType.add, Nat.mul_comm]

theorem original_safe (storage : Storage) (bounds : RegionBounds) (s : BlockState)
    (P G Y : RegionName) (M N K B C : Nat) (hN : 0 < N) (hB : 0 < B) (hPY : P ≠ Y) (hGY : G ≠ Y)
    (hr : s.pids 0 < M) (hP : M * N * K ≤ bounds P) (hG : M * N * K ≤ bounds G) (hY : M * N * K ≤ bounds Y) :
    (softmax_backward_kernel_non_inner storage P G Y M N K B C Bool.false).eraseDType.toAlgKernel.TraceSafe bounds s := by
  rw [RealProjection.softmax_backward_kernel_non_inner_erased]
  exact source_safe bounds s P G Y M N K B C hN hB hPY hGY hr hP hG hY

theorem candidate_safe (storage : Storage) (bounds : RegionBounds) (s : BlockState)
    (P G Y : RegionName) (M N K B C : Nat) (hN : 0 < N) (hB : 0 < B) (hPY : P ≠ Y) (hGY : G ≠ Y)
    (hr : s.pids 0 < M) (hP : M * N * K ≤ bounds P) (hG : M * N * K ≤ bounds G) (hY : M * N * K ≤ bounds Y) :
    (softmax_backward_kernel_non_inner_candidate storage P G Y M N K B C Bool.false).eraseDType.toAlgKernel.TraceSafe bounds s := by
  have h := original_safe storage bounds s P G Y M N K B C hN hB hPY hGY hr hP hG hY
  unfold Kernel.TraceSafe at h ⊢
  rw [RealCandidate.backward_non_inner_original_body, RealTrace.append_iff] at h
  rw [RealCandidate.backward_non_inner_body, RealTrace.append_iff]
  refine ⟨indexPrefix_safe bounds _ _ _ _, ?_⟩
  intro next hnext
  apply h.2 next
  rw [indexPrefix_exec]
  exact hnext

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
