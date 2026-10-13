import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardInnerInitialize
import bench.tritonbench_g.softmax_flaggems.Proofs.RealProjection
import bench.tritonbench_g.softmax_flaggems.Proofs.RealCandidate

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
open VeriTile Triton FullSource

private theorem prefix_exec (kind : RealCandidate.PrefixKind) (B : Nat) (s : BlockState) :
    stepStmts (RealCandidate.indexPrefix kind Bool.false B) s =
      stepStmts (RealCandidate.indexPrefix kind Bool.true B) s := by
  cases kind <;> simp [RealCandidate.indexPrefix, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, NumericDType.mul, NumericDType.add, Nat.mul_comm]

theorem candidate_exec (storage : Storage) (P G Y : RegionName)
    (M N B C : Nat) (one : Bool) (s : BlockState) :
    exec (softmax_backward_kernel_inner storage P G Y M N C B one).eraseDType.toAlgKernel s =
      exec (softmax_backward_kernel_inner_candidate storage P G Y M N C B one).eraseDType.toAlgKernel s := by
  unfold exec
  conv_lhs => rw [RealCandidate.backward_inner_original_body]
  conv_rhs => rw [RealCandidate.backward_inner_body]
  simp only [ProgramRewrite.append, prefix_exec]

theorem original_io (storage : Storage) (s : BlockState) (P G Y : RegionName)
    (M N B C : Nat) (hN : 0<N) (hB : 0<B) (hPY : P≠Y) (hGY : G≠Y) :
    ∃ final, exec (softmax_backward_kernel_inner storage P G Y M N C B Bool.false).eraseDType.toAlgKernel s = some final ∧
      BlockState.WriteWithin (outputFootprint Y M N (s.pids 0*C) C) s final ∧
      (∀ r n, r<C → s.pids 0*C+r<M → n<N →
        final.readMem Y (offset N (s.pids 0*C+r) n) =
          expected s P G M N (RealAddress.rounds N B*B) (s.pids 0*C+r) n) ∧
      final.pids=s.pids ∧ final.numPids=s.numPids ∧ final.undef=s.undef := by
  rw [RealProjection.softmax_backward_kernel_inner_erased]
  obtain ⟨final, he, h⟩ := source_run s P G Y M N B C hN hB hPY hGY
  refine ⟨final, he, h.frame, ?_, h.pids, h.numPids, h.undef⟩
  intro r n hrC hrM hn
  exact h.values r n hrC hrM hn (RealAddress.active_tile_lt hB hn)

theorem candidate_io (storage : Storage) (s : BlockState) (P G Y : RegionName)
    (M N B C : Nat) (hN : 0<N) (hB : 0<B) (hPY : P≠Y) (hGY : G≠Y) :
    ∃ final, exec (softmax_backward_kernel_inner_candidate storage P G Y M N C B Bool.false).eraseDType.toAlgKernel s = some final ∧
      BlockState.WriteWithin (outputFootprint Y M N (s.pids 0*C) C) s final ∧
      (∀ r n, r<C → s.pids 0*C+r<M → n<N →
        final.readMem Y (offset N (s.pids 0*C+r) n) =
          expected s P G M N (RealAddress.rounds N B*B) (s.pids 0*C+r) n) ∧
      final.pids=s.pids ∧ final.numPids=s.numPids ∧ final.undef=s.undef := by
  rw [←candidate_exec]
  exact original_io storage s P G Y M N B C hN hB hPY hGY

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
