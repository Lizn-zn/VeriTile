import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerFinish
import bench.tritonbench_g.softmax_flaggems.Proofs.RealProjection
import bench.tritonbench_g.softmax_flaggems.Proofs.RealCandidate

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton FullSource

theorem inner_prefix_exec (Y X : RegionName) (N B : Nat) (s : BlockState) :
    stepStmts (RealCandidate.innerPrefix Bool.false Y X N B) s =
      stepStmts (RealCandidate.innerPrefix Bool.true Y X N B) s := by
  simp [RealCandidate.innerPrefix, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.ptrAdd, NumericDType.mul, Nat.mul_comm]

theorem candidate_exec (input output : Storage) (Y X : RegionName)
    (M N B : Nat) (oneTile : Bool) (s : BlockState) :
    exec (softmax_kernel_inner input output Y X M N B oneTile).eraseDType.toAlgKernel s =
      exec (softmax_kernel_inner_candidate input output Y X M N B oneTile).eraseDType.toAlgKernel s := by
  unfold exec
  conv_lhs => rw [RealCandidate.forward_inner_original_body]
  conv_rhs => rw [RealCandidate.forward_inner_body]
  cases oneTile <;> simp [stepStmts, stepStmt, evalOp.eq_def, ProgramRewrite.append, inner_prefix_exec]

/-- Independent full-column softmax outputs and full-cell frame for the complete
two-pass branch. No execution or output property is an input to this theorem.
Storage-independent statements concern the real projection only. -/
theorem original_io (input output : Storage) (s : BlockState) (Y X : RegionName)
    (M N B : Nat) (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y) :
    ∃ final,
      exec (softmax_kernel_inner input output Y X M N B Bool.false).eraseDType.toAlgKernel s = some final ∧
      BlockState.WriteWithin (outputFootprint Y (s.pids 0) N) s final ∧
      (∀ n, n < N →
        final.readMem Y (s.pids 0 * N + n) =
          expected s X (s.pids 0) N n) ∧
      final.pids = s.pids ∧ final.numPids = s.numPids ∧ final.undef = s.undef := by
  rw [RealProjection.softmax_kernel_inner_erased]
  obtain ⟨final, hexec, h⟩ := source_run s Y X M N B hN hB hXY
  refine ⟨final, hexec, h.frame, ?_, h.pids, h.numPids, h.undef⟩
  intro n hn
  exact h.values n hn (by simp)

theorem candidate_io (input output : Storage) (s : BlockState) (Y X : RegionName)
    (M N B : Nat) (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y) :
    ∃ final,
      exec (softmax_kernel_inner_candidate input output Y X M N B Bool.false).eraseDType.toAlgKernel s = some final ∧
      BlockState.WriteWithin (outputFootprint Y (s.pids 0) N) s final ∧
      (∀ n, n < N →
        final.readMem Y (s.pids 0 * N + n) =
          expected s X (s.pids 0) N n) ∧
      final.pids = s.pids ∧ final.numPids = s.numPids ∧ final.undef = s.undef := by
  rw [← candidate_exec]
  exact original_io input output s Y X M N B hN hB hXY

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
