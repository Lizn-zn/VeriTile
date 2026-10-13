import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerInitialize
import bench.tritonbench_g.softmax_flaggems.Proofs.RealProjection
import bench.tritonbench_g.softmax_flaggems.Proofs.RealCandidate

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton FullSource

private theorem prefix_exec (kind : RealCandidate.PrefixKind) (B : Nat) (s : BlockState) :
    stepStmts (RealCandidate.indexPrefix kind Bool.false B) s =
      stepStmts (RealCandidate.indexPrefix kind Bool.true B) s := by
  cases kind <;> simp [RealCandidate.indexPrefix, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, NumericDType.mul, NumericDType.add, Nat.mul_comm]

theorem candidate_exec (input output : Storage) (Y X : RegionName)
    (M N K B C : Nat) (oneTile : Bool) (s : BlockState) :
    exec (softmax_kernel_non_inner input output Y X M N K B C oneTile).eraseDType.toAlgKernel s =
      exec (softmax_kernel_non_inner_candidate input output Y X M N K B C oneTile).eraseDType.toAlgKernel s := by
  unfold exec
  conv_lhs => rw [RealCandidate.forward_non_inner_original_body]
  conv_rhs => rw [RealCandidate.forward_non_inner_body]
  simp only [ProgramRewrite.append, prefix_exec]

/-- Independent full-column softmax outputs and full-cell frame for the complete
two-pass branch. No execution or output property is an input to this theorem.
Storage-independent statements concern the real projection only. -/
theorem original_io (input output : Storage) (s : BlockState) (Y X : RegionName)
    (M N K B C : Nat) (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y) :
    ∃ final,
      exec (softmax_kernel_non_inner input output Y X M N K B C Bool.false).eraseDType.toAlgKernel s = some final ∧
      BlockState.WriteWithin (outputFootprint Y (s.pids 0) N K (s.pids 1 * C) C) s final ∧
      (∀ n k, n < N → k < C → s.pids 1 * C + k < K →
        final.readMem Y (s.pids 0 * N * K + n * K + (s.pids 1 * C + k)) =
          expected s X (s.pids 0) N K n (s.pids 1 * C + k)) ∧
      final.pids = s.pids ∧ final.numPids = s.numPids ∧ final.undef = s.undef := by
  rw [RealProjection.softmax_kernel_non_inner_erased]
  obtain ⟨final, hexec, h⟩ := source_run s Y X M N K B C hN hB hXY
  refine ⟨final, hexec, h.frame, ?_, h.pids, h.numPids, h.undef⟩
  intro n k hn hk hbound
  exact h.values n k hn hk hbound (by simp)

theorem candidate_io (input output : Storage) (s : BlockState) (Y X : RegionName)
    (M N K B C : Nat) (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y) :
    ∃ final,
      exec (softmax_kernel_non_inner_candidate input output Y X M N K B C Bool.false).eraseDType.toAlgKernel s = some final ∧
      BlockState.WriteWithin (outputFootprint Y (s.pids 0) N K (s.pids 1 * C) C) s final ∧
      (∀ n k, n < N → k < C → s.pids 1 * C + k < K →
        final.readMem Y (s.pids 0 * N * K + n * K + (s.pids 1 * C + k)) =
          expected s X (s.pids 0) N K n (s.pids 1 * C + k)) ∧
      final.pids = s.pids ∧ final.numPids = s.numPids ∧ final.undef = s.undef := by
  rw [← candidate_exec]
  exact original_io input output s Y X M N K B C hN hB hXY

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
