import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerInitialize
import bench.tritonbench_g.softmax_flaggems.Proofs.RealProjection
import bench.tritonbench_g.softmax_flaggems.Proofs.RealCandidate

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton FullSource

private theorem prefix_exec (kind : RealCandidate.PrefixKind) (B : Nat) (s : BlockState) :
    stepStmts (RealCandidate.indexPrefix kind Bool.false B) s =
      stepStmts (RealCandidate.indexPrefix kind Bool.true B) s := by
  cases kind <;> simp [RealCandidate.indexPrefix, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, NumericDType.mul, NumericDType.add, Nat.mul_comm]

theorem candidate_exec (storage : Storage) (P G Y : RegionName)
    (M N K B C : Nat) (oneTile : Bool) (s : BlockState) :
    exec (softmax_backward_kernel_non_inner storage P G Y M N K B C oneTile).eraseDType.toAlgKernel s =
      exec (softmax_backward_kernel_non_inner_candidate storage P G Y M N K B C oneTile).eraseDType.toAlgKernel s := by
  unfold exec
  conv_lhs => rw [RealCandidate.backward_non_inner_original_body]
  conv_rhs => rw [RealCandidate.backward_non_inner_body]
  simp only [ProgramRewrite.append, prefix_exec]

/-- Complete source backward outputs with exact omitted-other padding and full memory frame. -/
theorem original_io (storage : Storage) (s : BlockState) (P G Y : RegionName)
    (M N K B C : Nat) (hN : 0 < N) (hB : 0 < B) (hPY : P ≠ Y) (hGY : G ≠ Y) :
    ∃ final,
      exec (softmax_backward_kernel_non_inner storage P G Y M N K B C Bool.false).eraseDType.toAlgKernel s = some final ∧
      BlockState.WriteWithin (outputFootprint Y (s.pids 0) N K (s.pids 1 * C) C) s final ∧
      (∀ n k, n < N → k < C → s.pids 1 * C + k < K →
        final.readMem Y (s.pids 0 * N * K + n * K + (s.pids 1 * C + k)) =
          expected s P G (s.pids 0) N K (RealAddress.rounds N B * B) n (s.pids 1 * C + k)) ∧
      final.pids = s.pids ∧ final.numPids = s.numPids ∧ final.undef = s.undef := by
  rw [RealProjection.softmax_backward_kernel_non_inner_erased]
  obtain ⟨final, hexec, h⟩ := source_run s P G Y M N K B C hN hB hPY hGY
  refine ⟨final, hexec, h.frame, ?_, h.pids, h.numPids, h.undef⟩
  intro n k hn hk hbound
  exact h.values n k hn hk hbound (RealAddress.active_tile_lt hB hn)

theorem candidate_io (storage : Storage) (s : BlockState) (P G Y : RegionName)
    (M N K B C : Nat) (hN : 0 < N) (hB : 0 < B) (hPY : P ≠ Y) (hGY : G ≠ Y) :
    ∃ final,
      exec (softmax_backward_kernel_non_inner_candidate storage P G Y M N K B C Bool.false).eraseDType.toAlgKernel s = some final ∧
      BlockState.WriteWithin (outputFootprint Y (s.pids 0) N K (s.pids 1 * C) C) s final ∧
      (∀ n k, n < N → k < C → s.pids 1 * C + k < K →
        final.readMem Y (s.pids 0 * N * K + n * K + (s.pids 1 * C + k)) =
          expected s P G (s.pids 0) N K (RealAddress.rounds N B * B) n (s.pids 1 * C + k)) ∧
      final.pids = s.pids ∧ final.numPids = s.numPids ∧ final.undef = s.undef := by
  rw [← candidate_exec]
  exact original_io storage s P G Y M N K B C hN hB hPY hGY

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
