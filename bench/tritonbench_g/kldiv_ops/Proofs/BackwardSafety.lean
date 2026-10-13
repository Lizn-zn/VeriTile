import bench.tritonbench_g.kldiv_ops.Proofs.BackwardExecution

namespace VeriTile.Bench.TritonBenchG.KldivOps.FullBackward
open VeriTile.Triton FullAlgorithm
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- Safety follows the actual states through the masked load, arithmetic and masked store. -/
theorem block_safe (bounds : RegionBounds) (s : BlockState) (T O : RegionName)
    (st so targetBase outputBase N B current : Nat) (flag : Bool)
    (hT : s.regs .ptr [] "T" = some (Tile.scalar (T, targetBase)))
    (hO : s.regs .ptr [] "O" = some (Tile.scalar (O, outputBase)))
    (hbt : ∀ k : Fin N, targetBase + k.val < bounds T)
    (hbo : ∀ k : Fin N, outputBase + k.val < bounds O) :
    Stmt.TraceSafeListR .triv bounds (body T O st so N B flag)
      (s.setReg "i" .int [] (Tile.scalar (current : Int))) := by
  cases flag <;>
    simp [body, backward, backwardSurface, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
      MemAccess.SafeAtR, MaskOpt.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR, stepStmtR_triv, evalOpR_triv, stepStmts, stepStmt,
      evalOp.eq_def, Option.bind_eq_bind, Option.bind,
      hT, hO, Tile.ptrAdd, Tile.bop, Tile.uop, Tile.cop, Tile.scalar, -Tile.scalar_eta,
      NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.lt,
      Region.cast, Except.bind, bind]
  all_goals
    constructor
    · intro j hj
      have hcast : (current : Int) + (j.val : Int) < (N : Int) := hj
      have hn : current + j.val < N := by exact_mod_cast hcast
      simpa only [← Int.natCast_add, Int.toNat_natCast, Nat.add_assoc] using
        hbt ⟨current + j.val, hn⟩
    · intro j hj
      have hcast : (current : Int) + (j.val : Int) < (N : Int) := hj
      have hn : current + j.val < N := by exact_mod_cast hcast
      simpa only [← Int.natCast_add, Int.toNat_natCast, Nat.add_assoc] using
        hbo ⟨current + j.val, hn⟩

/-- The signed-loop invariant supplies both safety and successful body execution at every step. -/
theorem loop_safe (bounds : RegionBounds) (initial s : BlockState) (T O : RegionName)
    (st so targetBase outputBase N B : Nat) (flag : Bool) (hTO : T ≠ O)
    (h : Inv initial T O targetBase outputBase N B 0 flag s)
    (hbt : ∀ k : Fin N, targetBase + k.val < bounds T)
    (hbo : ∀ k : Fin N, outputBase + k.val < bounds O) :
    Stmt.TraceSafeR .triv bounds
      (.forRangeInt "i" (.constInt 0) (.constInt N) (.constNat B)
        (body T O st so N B flag)) s := by
  simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, evalOpR_triv, evalOp.eq_def,
    Tile.scalar_data, Option.some.injEq, forall_eq', true_and]
  apply Stmt.forRangeIntTraceSafeR_inv .triv bounds "i" N B (body T O st so N B flag)
    (fun current st ↦ ∃ c, current = (c * B : Nat) ∧
      Inv initial T O targetBase outputBase N B c flag st) ?_ 0 s
    ⟨0, by simp, h⟩
  intro current state hc hs
  rcases hs with ⟨c, rfl, hs⟩
  refine ⟨block_safe bounds state T O st so targetBase outputBase N B (c * B) flag
    hs.targetReg hs.outputReg hbt hbo, ?_⟩
  obtain ⟨next, he, hn⟩ := invariant_step initial state T O st so targetBase outputBase N B c flag hTO hs
  refine ⟨next, ?_, c+1, ?_, hn⟩
  · simpa only [stepStmtsR_triv] using he
  · simp [Nat.add_mul]

/-- Initialization only reads program metadata and constructs pointers and index tiles. -/
theorem prefix_safe (bounds : RegionBounds) (s : BlockState) (T O : RegionName)
    (st so N B : Nat) (flag : Bool) :
    Stmt.TraceSafeListR .triv bounds ((backward T O st so N B flag).body.take 5) s := by
  cases flag <;>
    simp [backward, backwardSurface, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
      stepStmtR_triv, evalOpR_triv, stepStmt, evalOp.eq_def, Tile.scalar,
      -Tile.scalar_eta, Except.bind, bind]

/-- Every actual backward access is in bounds throughout the complete signed loop. -/
theorem source_safe (bounds : RegionBounds) (s : BlockState) (T O : RegionName)
    (st so N B : Nat) (flag : Bool) (hTO : T ≠ O)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < bounds T)
    (hbo : ∀ k : Fin N, s.pids 0 * so + k.val < bounds O) :
    (backward T O st so N B flag).TraceSafeR .triv bounds s := by
  unfold Kernel.TraceSafeR
  rw [source_decomposition]
  refine Stmt.TraceSafeListR.append_intro _ _ (prefix_safe bounds s T O st so N B flag) ?_
  intro initial hi
  rw [stepStmtsR_triv, backward_initialize] at hi
  have heq := Option.some.inj hi
  subst initial
  refine Stmt.TraceSafeListR.cons_intro
    (loop_safe bounds s _ T O st so (s.pids 0 * st) (s.pids 0 * so) N B flag hTO
      (initial_inv s T O st so N B flag) hbt hbo) ?_
  intro final _
  exact Stmt.TraceSafeListR.nil_intro

end VeriTile.Bench.TritonBenchG.KldivOps.FullBackward
#guard_msgs (drop info) in
#auditModuleAxioms
