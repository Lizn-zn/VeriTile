import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerSafetySource
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerIO

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton FullSource
set_option maxHeartbeats 1200000

private theorem innerPrefix_safe (bounds : RegionBounds) (swapped : Bool)
    (Y X : RegionName) (N B : Nat) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealCandidate.innerPrefix swapped Y X N B) s := by
  apply RealTrace.of_all
  intro statement hstatement st
  cases swapped <;> simp only [RealCandidate.innerPrefix, Bool.false_eq_true,
    ↓reduceIte, List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at hstatement
  all_goals rcases hstatement with rfl | rfl | rfl | rfl
  all_goals simp [Stmt.TraceSafe, Op.SafeAt.eq_def]

private theorem innerPrefix_exec (Y X : RegionName) (N B : Nat) (s : BlockState) :
    stepStmts (RealCandidate.innerPrefix Bool.false Y X N B) s =
      stepStmts (RealCandidate.innerPrefix Bool.true Y X N B) s := by
  simp [RealCandidate.innerPrefix, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, NumericDType.mul, NumericDType.add, Nat.mul_comm]

theorem original_safe (input output : Storage) (bounds : RegionBounds) (s : BlockState)
    (Y X : RegionName) (M N B : Nat) (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y)
    (hr : s.pids 0 < M) (hX : M * N ≤ bounds X) (hY : M * N ≤ bounds Y) :
    (softmax_kernel_inner input output Y X M N B Bool.false).eraseDType.toAlgKernel.TraceSafe bounds s := by
  rw [RealProjection.softmax_kernel_inner_erased]
  exact source_safe bounds s Y X M N B hN hB hXY hr hX hY

theorem candidate_safe (input output : Storage) (bounds : RegionBounds) (s : BlockState)
    (Y X : RegionName) (M N B : Nat) (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y)
    (hr : s.pids 0 < M) (hX : M * N ≤ bounds X) (hY : M * N ≤ bounds Y) :
    (softmax_kernel_inner_candidate input output Y X M N B Bool.false).eraseDType.toAlgKernel.TraceSafe bounds s := by
  have h := original_safe input output bounds s Y X M N B hN hB hXY hr hX hY
  unfold Kernel.TraceSafe at h ⊢
  rw [RealCandidate.forward_inner_original_body] at h
  rw [RealCandidate.forward_inner_body]
  simp only [Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    stepStmt, evalOp.eq_def, Option.bind_eq_bind, Option.bind,
    Tile.scalar_data, Bool.false_eq_true, ↓reduceIte, true_and] at h ⊢
  rw [RealTrace.append_iff] at h
  rw [RealTrace.append_iff]
  refine ⟨⟨innerPrefix_safe bounds _ _ _ _ _ _, ?_⟩, ?_⟩
  · intro next hnext
    apply h.1.2 next
    rw [innerPrefix_exec]
    exact hnext
  · cases stepStmts (RealCandidate.innerPrefix Bool.true Y X N B ++
      (RealCandidate.elseBody (softmax_kernel_inner input output Y X M N B Bool.false).eraseDType.toAlgKernel).drop 4)
      (s.setReg "pid_m" .nat [] (Tile.scalar (s.pids 0))) <;> trivial

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner

#guard_msgs (drop info) in
#auditModuleAxioms
