import bench.tritonbench_g.kldiv_ops.Proofs.ForwardReductionExecution
import bench.tritonbench_g.kldiv_ops.Proofs.ForwardElementExecution

namespace VeriTile.Bench.TritonBenchG.KldivOps.FullForward
open VeriTile.Triton FullAlgorithm
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- The actual block reads only active prediction/target lanes and writes only in mode zero. -/
theorem block_safe (bounds : RegionBounds) (s : BlockState) (Y T O : RegionName)
    (sy st so yBase tBase outputBase N B current : Nat) (eps : ℝ)
    (logTarget : Bool) (reduction : Nat)
    (hY : s.regs .ptr [] "Y" = some (Tile.scalar (Y, yBase)))
    (hT : s.regs .ptr [] "T" = some (Tile.scalar (T, tBase)))
    (hO : s.regs .ptr [] "O" = some (Tile.scalar (O, outputBase)))
    (hBase : s.regs .nat [B] "base_offsets" = some (Tile.vec fun j ↦ j.val))
    (hby : ∀ k : Fin N, yBase + k.val < bounds Y)
    (hbt : ∀ k : Fin N, tBase + k.val < bounds T)
    (hbo : reduction = 0 → ∀ k : Fin N, outputBase + k.val < bounds O) :
    Stmt.TraceSafeListR .triv bounds (body Y T O sy st so N B eps logTarget reduction)
      (s.setReg "i" .int [] (Tile.scalar (current : Int))) := by
  by_cases hR : reduction = 0
  all_goals cases logTarget <;>
    simp [body, forward, forwardSurface, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
      MemAccess.SafeAtR, MaskOpt.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR, stepStmtR_triv, evalOpR_triv, stepStmts, stepStmt,
      evalOp.eq_def, Option.bind_eq_bind, Option.bind,
      hY, hT, hO, hBase, hR, Tile.ptrAdd, Tile.bop, Tile.uop, Tile.cop,
      Tile.scalar, Tile.vec, -Tile.scalar_eta, NumericDType.add, NumericDType.mul,
      NumericDType.sub, ComparableDType.lt, ComparableDType.eq,
      Region.cast, Except.bind, bind]
  all_goals repeat' constructor
  all_goals try (split <;> trivial)
  all_goals
    intro j hj
    have hcast : (current : Int) + (j.val : Int) < (N : Int) := hj
    have hn : current + j.val < N := by exact_mod_cast hcast
    first
    | simpa only [← Int.natCast_add, Int.toNat_natCast, Nat.add_assoc] using
        hby ⟨current + j.val, hn⟩
    | simpa only [← Int.natCast_add, Int.toNat_natCast, Nat.add_assoc] using
        hbt ⟨current + j.val, hn⟩
    | simpa only [← Int.natCast_add, Int.toNat_natCast, Nat.add_assoc] using
        hbo hR ⟨current + j.val, hn⟩

/-- Initialization constructs pointers and registers without touching memory. -/
theorem prefix_safe (bounds : RegionBounds) (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (eps : ℝ) (logTarget : Bool) (reduction : Nat) :
    Stmt.TraceSafeListR .triv bounds
      ((forward Y T O sy st so N B eps logTarget reduction).body.take 6) s := by
  cases logTarget <;>
    simp [forward, forwardSurface, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
      stepStmtR_triv, evalOpR_triv, stepStmt, evalOp.eq_def, Tile.scalar,
      -Tile.scalar_eta, Except.bind, bind]

/-- Mode zero has no final memory access, even when the loop was empty. -/
theorem element_finish_safe (bounds : RegionBounds) (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (eps : ℝ) (logTarget : Bool) :
    Stmt.TraceSafeListR .triv bounds
      ((forward Y T O sy st so N B eps logTarget 0).body.drop 7) s := by
  simp [forward, forwardSurface, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
    evalOpR_triv, evalOp.eq_def, Tile.cop, ComparableDType.ne,
    Tile.scalar, -Tile.scalar_eta, Except.bind, bind]
  split <;> trivial

/-- Every nonzero mode performs the single final scalar store, including for N=0. -/
theorem reduction_finish_safe (bounds : RegionBounds) (s : BlockState) (Y T O : RegionName)
    (sy st so N B outputBase : Nat) (eps : ℝ) (logTarget : Bool) (reduction : Nat)
    (hR : reduction ≠ 0)
    (hO : s.regs .ptr [] "O" = some (Tile.scalar (O, outputBase)))
    (hbo : outputBase < bounds O) :
    Stmt.TraceSafeListR .triv bounds
      ((forward Y T O sy st so N B eps logTarget reduction).body.drop 7) s := by
  simp [forward, forwardSurface, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
    MemAccess.SafeAtR, MaskOpt.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
    memAccessActiveAddressSafeR, evalOpR_triv, evalOp.eq_def,
    Tile.cop, ComparableDType.ne, hR, hO, hbo,
    Tile.scalar, -Tile.scalar_eta, Region.cast, Except.bind, bind]
  constructor <;> split <;> trivial

end VeriTile.Bench.TritonBenchG.KldivOps.FullForward
#guard_msgs (drop info) in
#auditModuleAxioms
