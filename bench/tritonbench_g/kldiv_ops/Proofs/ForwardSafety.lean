import bench.tritonbench_g.kldiv_ops.Proofs.ForwardSafetyLoop

namespace VeriTile.Bench.TritonBenchG.KldivOps.FullForward
open VeriTile.Triton FullAlgorithm
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- Complete elementwise trace safety, including the initial and final source branches. -/
theorem element_source_safe (bounds : RegionBounds) (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (eps : ℝ) (logTarget : Bool) (hYO : Y ≠ O) (hTO : T ≠ O)
    (hby : ∀ k : Fin N, s.pids 0 * sy + k.val < bounds Y)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < bounds T)
    (hbo : ∀ k : Fin N, s.pids 0 * so + k.val < bounds O) :
    (forward Y T O sy st so N B eps logTarget 0).TraceSafeR .triv bounds s := by
  unfold Kernel.TraceSafeR
  rw [element_source_decomposition, List.append_assoc]
  refine Stmt.TraceSafeListR.append_intro _ _
    (prefix_safe bounds s Y T O sy st so N B eps logTarget 0) ?_
  intro initial hi
  rw [stepStmtsR_triv, forward_initialize] at hi
  have heq := Option.some.inj hi
  subst initial
  simp only [List.cons_append, List.nil_append]
  refine Stmt.TraceSafeListR.cons_intro
    (element_loop_safe bounds s _ Y T O sy st so (s.pids 0 * sy) (s.pids 0 * st)
      (s.pids 0 * so) N B eps logTarget hYO hTO
      (element_initial_inv s Y T O sy st so N B eps logTarget) hby hbt hbo) ?_
  intro final _
  exact element_finish_safe bounds final Y T O sy st so N B eps logTarget

/-- Nonzero modes read the whole source row before the scalar store, permitting every alias.
The scalar output bound is required even for N=0; input bounds are then vacuous. -/
theorem reduction_source_safe (bounds : RegionBounds) (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (eps : ℝ) (logTarget : Bool) (reduction : Nat)
    (hB : 0 < B) (hR : reduction ≠ 0)
    (hby : ∀ k : Fin N, s.pids 0 * sy + k.val < bounds Y)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < bounds T)
    (hbo : s.pids 0 * so < bounds O) :
    (forward Y T O sy st so N B eps logTarget reduction).TraceSafeR .triv bounds s := by
  let initial := forwardInitial s Y T O sy st so B
  have hinv := reduction_initial_inv s Y T O sy st so N B eps logTarget
  obtain ⟨covered, middle, _, he, hf⟩ :=
    reduction_loop_run s Y T O sy st so (s.pids 0 * sy) (s.pids 0 * st)
      (s.pids 0 * so) N B eps logTarget reduction hB hR 0 initial hinv
  have hloop : stepStmt
      (.forRangeInt "i" (.constInt 0) (.constInt N) (.constNat B)
        (body Y T O sy st so N B eps logTarget reduction)) initial = some middle := by
    simp only [Nat.zero_mul, Nat.cast_zero] at he
    simpa only [stepStmt, evalOp.eq_def, Option.bind_eq_bind, Option.bind,
      Tile.scalar_data] using he
  unfold Kernel.TraceSafeR
  rw [source_decomposition, List.append_assoc]
  refine Stmt.TraceSafeListR.append_intro _ _
    (prefix_safe bounds s Y T O sy st so N B eps logTarget reduction) ?_
  intro state hs
  rw [stepStmtsR_triv, forward_initialize] at hs
  have hstate : state = initial := (Option.some.inj hs).symm
  subst state
  simp only [List.cons_append, List.nil_append]
  refine Stmt.TraceSafeListR.cons_intro
    (reduction_loop_safe bounds s initial Y T O sy st so (s.pids 0 * sy) (s.pids 0 * st)
      (s.pids 0 * so) N B eps logTarget reduction hR hinv hby hbt) ?_
  intro final hfinal
  rw [stepStmtR_triv, hloop] at hfinal
  have hstate := Option.some.inj hfinal
  subst final
  exact reduction_finish_safe bounds middle Y T O sy st so N B (s.pids 0 * so)
    eps logTarget reduction hR hf.optr hbo

end VeriTile.Bench.TritonBenchG.KldivOps.FullForward

#axiomsClean VeriTile.Bench.TritonBenchG.KldivOps.FullForward.element_source_safe
#axiomsClean VeriTile.Bench.TritonBenchG.KldivOps.FullForward.reduction_source_safe
#guard_msgs (drop info) in
#auditModuleAxioms
