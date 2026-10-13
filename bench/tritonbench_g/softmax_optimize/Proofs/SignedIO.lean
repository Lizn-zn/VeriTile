import bench.tritonbench_g.softmax_optimize.Proofs.SignedProgram
import bench.tritonbench_g.softmax_optimize.Proofs.RealFullIO
import bench.tritonbench_g.softmax_optimize.Proofs.SignedCandidateEquality

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton FullSource

/-- The exact signed four-loop source executes successfully and computes the
independent full-row formula. Memory outside this row is preserved cell-for-cell. -/
theorem original_io (input : Storage) (output : Accumulation)
    (Y X : RegionName) (M N B : Nat) (hN : 0<N) (hB : 0<B) (hYX : Y≠X)
    (s : BlockState) : ∃ t,
      exec (kernel input output Bool.false Y X M N B).eraseDType.toAlgKernel s = some t ∧
      (∀ j : Fin N, t.readMem Y (linearOffset s N j) = softmaxOptimizeFullSpec s X N j) ∧
      RowFrame Y N s t := by
  obtain ⟨u, hu, hval, hframe⟩ := full_surface_io Y X M N B hN hB hYX s
  have hr := source_positive_related Y X M N B hN hB s
  rw [hu] at hr
  cases ht : exec (mathematical Bool.false Y X M N B) s with
  | none => simp [ht, ResultsRelated] at hr
  | some t =>
    have hrel : LiveRelated t u := by simpa only [ht, ResultsRelated] using hr
    refine ⟨t, ?_, ?_, ?_⟩
    · rw [projection]
      exact ht
    · intro j
      rw [hrel.readMem]
      exact hval j
    · intro r o ho
      rw [hrel.mem_eq]
      exact hframe r o ho


theorem candidate_io (input : Storage) (output : Accumulation)
    (Y X : RegionName) (M N B : Nat) (hN : 0<N) (hB : 0<B) (hYX : Y≠X)
    (s : BlockState) : ∃ t,
      exec (kernel input output Bool.true Y X M N B).eraseDType.toAlgKernel s = some t ∧
      (∀ j : Fin N, t.readMem Y (linearOffset s N j) = softmaxOptimizeFullSpec s X N j) ∧
      RowFrame Y N s t := by
  rw [candidate_exec_eq]
  exact original_io input output Y X M N B hN hB hYX s

/-- A genuinely multi-tile source instance with a partial second tile. The
masked lane immediately following the three-column row remains untouched. -/
theorem three_columns_two_lanes (input : Storage) (output : Accumulation)
    (Y X : RegionName) (M : Nat) (hYX : Y≠X) (s : BlockState) : ∃ t,
      exec (kernel input output Bool.false Y X M 3 2).eraseDType.toAlgKernel s = some t ∧
      (∀ j : Fin 3, t.readMem Y (linearOffset s 3 j) = softmaxOptimizeFullSpec s X 3 j) ∧
      t.mem Y (s.pid*3+3) = s.mem Y (s.pid*3+3) := by
  obtain ⟨t, ht, hv, hf⟩ := original_io input output Y X M 3 2 (by omega) (by omega) hYX s
  exact ⟨t, ht, hv, hf Y (s.pid*3+3) (Or.inr (Or.inr (Nat.le_refl _)))⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
