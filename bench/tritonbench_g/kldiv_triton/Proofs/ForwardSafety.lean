import bench.tritonbench_g.kldiv_triton.Proofs.ForwardExecution

namespace VeriTile.Bench.TritonBenchG.KldivTriton.FullForward
open VeriTile.Triton FullAlgorithm FullBackward
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

/-- Access safety for one actual source block, including its scalar reduction store. -/
theorem block_safe (bounds : RegionBounds) (s : BlockState) (Y T O : RegionName)
    (sy st so yBase tBase oBase N B current reduction : Nat) (flag : Bool)
    (hY : s.regs .ptr [] "Y" = some (Tile.scalar (Y, yBase)))
    (hT : s.regs .ptr [] "T" = some (Tile.scalar (T, tBase)))
    (hO : s.regs .ptr [] "O" = some (Tile.scalar (O, oBase)))
    (hBase : s.regs .nat [B] "base_offsets" = some (Tile.vec fun j ↦ j.val))
    (hby : ∀ k : Fin N, yBase + k.val < bounds Y)
    (hbt : ∀ k : Fin N, tBase + k.val < bounds T)
    (hbo : ∀ r k, blockFootprint O oBase N B current reduction (r, k) → k < bounds r) :
    Stmt.TraceSafeListR .triv bounds (body Y T O sy st so N B flag reduction)
      (s.setReg "i" .int [] (Tile.scalar (current : Int))) := by
  have hout (hr : reduction = 0) (j : Fin B) (hj : (current : Int) + j.val < N) :
      oBase + ((current : Int) + j.val).toNat < bounds O := by
    have hn : current + j.val < N := by exact_mod_cast hj
    have hh := hbo O (oBase + (current + j.val))
    have he : ((current : Int) + j.val).toNat = current + j.val := by
      rw [← Int.natCast_add, Int.toNat_natCast]
    rw [he]
    apply hh
    simpa only [blockFootprint, hr, if_true, Nat.add_assoc] using
      (show O = O ∧ ∃ lane, lane < B ∧ current + lane < N ∧
        oBase + (current + j.val) = oBase + (current + lane) from
        ⟨rfl, j.val, j.isLt, hn, rfl⟩)
  cases flag <;> by_cases hr : reduction = 0 <;>
    simp [body, forward, forwardSurface, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
      MemAccess.SafeAtR, MaskOpt.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR, stepStmtR_triv, evalOpR_triv, stepStmts, stepStmt, evalOp.eq_def,
      Option.bind_eq_bind, Option.bind, hY, hT, hO, hBase, hr, Tile.ptrAdd, Tile.bop, Tile.uop, Tile.cop, Tile.scalar,
      Tile.vec, -Tile.scalar_eta, NumericDType.add, NumericDType.mul, NumericDType.sub,
      ComparableDType.lt, ComparableDType.eq, Region.cast, Except.bind, bind]
  all_goals
    try constructor
    · intro j hj
      have hcast : (current : Int) + (j.val : Int) < (N : Int) := hj
      have hn : current + j.val < N := by exact_mod_cast hcast
      simpa only [← Int.natCast_add, Int.toNat_natCast, Nat.add_assoc] using
        hby ⟨current + j.val, hn⟩
    · constructor
      · intro j hj
        have hcast : (current : Int) + (j.val : Int) < (N : Int) := hj
        have hn : current + j.val < N := by exact_mod_cast hcast
        simpa only [← Int.natCast_add, Int.toNat_natCast, Nat.add_assoc] using
          hbt ⟨current + j.val, hn⟩
      · first
        | exact hout hr
        | exact hbo O oBase (by simp [blockFootprint, hr])

/-- The loop invariant validates the changing scalar output pointer at every chunk. -/
theorem loop_safe (bounds : RegionBounds) (initial s : BlockState) (Y T O : RegionName)
    (sy st so yBase tBase base N B reduction : Nat) (flag : Bool)
    (hB : 0 < B) (hYO : Y ≠ O) (hTO : T ≠ O)
    (h : Inv initial Y T O yBase tBase base N B 0 reduction flag s)
    (hby : ∀ k : Fin N, yBase + k.val < bounds Y)
    (hbt : ∀ k : Fin N, tBase + k.val < bounds T)
    (hbo : ∀ k : Fin (FullMath.outputCount N B reduction), base + k.val < bounds O) :
    Stmt.TraceSafeR .triv bounds
      (.forRangeInt "i" (.constInt 0) (.constInt N) (.constNat B)
        (body Y T O sy st so N B flag reduction)) s := by
  simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, evalOpR_triv, evalOp.eq_def,
    Tile.scalar_data, Option.some.injEq, forall_eq', true_and]
  apply Stmt.forRangeIntTraceSafeR_inv .triv bounds "i" N B
    (body Y T O sy st so N B flag reduction)
    (fun current state ↦ ∃ c, current = (c * B : Nat) ∧
      Inv initial Y T O yBase tBase base N B c reduction flag state) ?_ 0 s
    ⟨0, by simp, h⟩
  intro current state hc hs
  rcases hs with ⟨c, rfl, hs⟩
  have hcn : c * B < N := by exact_mod_cast hc
  have hout : ∀ r k,
      blockFootprint O (outputBase base c reduction) N B (c * B) reduction (r, k) →
        k < bounds r := by
    intro r k hk
    obtain ⟨hr, j, hj, he⟩ := block_within O base N B c reduction hB hcn (r, k) hk
    change r = O at hr
    change k = base + j at he
    subst r
    simpa only [he] using hbo ⟨j, hj⟩
  refine ⟨block_safe bounds state Y T O sy st so yBase tBase
    (outputBase base c reduction) N B (c * B) reduction flag
    hs.context.yReg hs.context.tReg hs.context.oReg hs.context.baseReg hby hbt hout, ?_⟩
  obtain ⟨next, he, hn⟩ := invariant_step initial state Y T O sy st so
    yBase tBase base N B c reduction flag hB hcn hYO hTO hs
  refine ⟨next, ?_, c + 1, ?_, hn⟩
  · simpa only [stepStmtsR_triv] using he
  · simp [Nat.add_mul]

theorem prefix_safe (bounds : RegionBounds) (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (flag : Bool) (reduction : Nat) :
    Stmt.TraceSafeListR .triv bounds
      ((forward Y T O sy st so N B flag reduction).body.take 5) s := by
  simp [forward, forwardSurface, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
    stepStmtR_triv, evalOpR_triv, stepStmt, evalOp.eq_def, Tile.scalar,
    -Tile.scalar_eta, Except.bind, bind]

/-- Trace safety uses the actual chunk footprint; no whole-row scalar-allocation assumption. -/
theorem source_safe (bounds : RegionBounds) (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (flag : Bool) (reduction : Nat)
    (hB : 0 < B) (hYO : Y ≠ O) (hTO : T ≠ O)
    (hby : ∀ k : Fin N, s.pids 0 * sy + k.val < bounds Y)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < bounds T)
    (hbo : ∀ k : Fin (FullMath.outputCount N B reduction),
      s.pids 0 * so + k.val < bounds O) :
    (forward Y T O sy st so N B flag reduction).TraceSafeR .triv bounds s := by
  unfold Kernel.TraceSafeR
  rw [source_decomposition]
  refine Stmt.TraceSafeListR.append_intro _ _
    (prefix_safe bounds s Y T O sy st so N B flag reduction) ?_
  intro initial hi
  rw [stepStmtsR_triv, forward_initialize] at hi
  have heq := Option.some.inj hi
  subst initial
  refine Stmt.TraceSafeListR.cons_intro
    (loop_safe bounds s _ Y T O sy st so (s.pids 0 * sy) (s.pids 0 * st)
      (s.pids 0 * so) N B reduction flag hB hYO hTO
      (initial_inv s Y T O sy st so N B flag reduction) hby hbt hbo) ?_
  intro final _
  exact Stmt.TraceSafeListR.nil_intro

end VeriTile.Bench.TritonBenchG.KldivTriton.FullForward
#guard_msgs (drop info) in
#auditModuleAxioms
