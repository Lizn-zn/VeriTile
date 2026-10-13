import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerSourceSafety
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerIO

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
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

theorem original_safe (input output : Storage) (bounds : RegionBounds) (s : BlockState)
    (Y X : RegionName) (M N K B C : Nat) (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y)
    (hr : s.pids 0 < M) (hX : M * N * K ≤ bounds X) (hY : M * N * K ≤ bounds Y) :
    (softmax_kernel_non_inner input output Y X M N K B C Bool.false).eraseDType.toAlgKernel.TraceSafe bounds s := by
  rw [RealProjection.softmax_kernel_non_inner_erased]
  exact source_safe bounds s Y X M N K B C hN hB hXY hr hX hY

theorem candidate_safe (input output : Storage) (bounds : RegionBounds) (s : BlockState)
    (Y X : RegionName) (M N K B C : Nat) (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y)
    (hr : s.pids 0 < M) (hX : M * N * K ≤ bounds X) (hY : M * N * K ≤ bounds Y) :
    (softmax_kernel_non_inner_candidate input output Y X M N K B C Bool.false).eraseDType.toAlgKernel.TraceSafe bounds s := by
  have h := original_safe input output bounds s Y X M N K B C hN hB hXY hr hX hY
  unfold Kernel.TraceSafe at h ⊢
  rw [RealCandidate.forward_non_inner_original_body, RealTrace.append_iff] at h
  rw [RealCandidate.forward_non_inner_body, RealTrace.append_iff]
  refine ⟨indexPrefix_safe bounds _ _ _ _, ?_⟩
  intro next hnext
  apply h.2 next
  rw [indexPrefix_exec]
  exact hnext

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
