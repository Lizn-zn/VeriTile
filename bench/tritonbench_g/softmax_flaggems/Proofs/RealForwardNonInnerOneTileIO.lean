import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerOneTileRun

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton FullSource

theorem oneTile_original_io (input output : Storage) (s : BlockState) (Y X : RegionName)
    (M N K B C : Nat) (hN : 0 < N) (hB : 0 < B) (hcover : N ≤ B) :
    ∃ final,
      exec (softmax_kernel_non_inner input output Y X M N K B C Bool.true).eraseDType.toAlgKernel s = some final ∧
      BlockState.WriteWithin (outputFootprint Y (s.pids 0) N K (s.pids 1 * C) C) s final ∧
      (∀ n k, n < N → k < C → s.pids 1 * C + k < K →
        final.readMem Y (s.pids 0 * N * K + n * K + (s.pids 1 * C + k)) =
          expected s X (s.pids 0) N K n (s.pids 1 * C + k)) ∧
      final.pids = s.pids ∧ final.numPids = s.numPids ∧ final.undef = s.undef := by
  rw [RealProjection.softmax_kernel_non_inner_erased]
  obtain ⟨st, hpre, hmem, hpids, hnum, hundef, hrow, hk⟩ := oneTile_prefix_run s C
  obtain ⟨final, hbody, hframe, hvalues, hpids', hnum', hundef'⟩ :=
    oneTile_body_io st Y X M N K B C (s.pids 0) (s.pids 1 * C) hN hB hcover hrow hk
  refine ⟨final, ?_, ?_, ?_, hpids'.trans hpids, hnum'.trans hnum, hundef'.trans hundef⟩
  · obtain ⟨no, hsource⟩ := oneTile_decomposition Y X M N K B C
    unfold exec
    rw [hsource, ProgramRewrite.append, hpre]
    simp only [Option.bind_some, stepStmts, stepStmt, evalOp.eq_def, Tile.scalar_data,
      Bool.true_eq, ↓reduceIte, Option.bind_eq_bind, Option.bind, Option.pure_def, hbody]
  · intro r o ho
    rw [← hmem]
    exact hframe r o ho
  · intro n k hn hkC hkK
    rw [hvalues n k hn hkC hkK]
    simp only [expected, denominator, BlockState.readMem, hmem]

theorem oneTile_candidate_io (input output : Storage) (s : BlockState) (Y X : RegionName)
    (M N K B C : Nat) (hN : 0 < N) (hB : 0 < B) (hcover : N ≤ B) :
    ∃ final,
      exec (softmax_kernel_non_inner_candidate input output Y X M N K B C Bool.true).eraseDType.toAlgKernel s = some final ∧
      BlockState.WriteWithin (outputFootprint Y (s.pids 0) N K (s.pids 1 * C) C) s final ∧
      (∀ n k, n < N → k < C → s.pids 1 * C + k < K →
        final.readMem Y (s.pids 0 * N * K + n * K + (s.pids 1 * C + k)) =
          expected s X (s.pids 0) N K n (s.pids 1 * C + k)) ∧
      final.pids = s.pids ∧ final.numPids = s.numPids ∧ final.undef = s.undef := by
  rw [← candidate_exec]
  exact oneTile_original_io input output s Y X M N K B C hN hB hcover

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
