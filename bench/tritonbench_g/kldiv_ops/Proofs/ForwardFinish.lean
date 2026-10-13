import bench.tritonbench_g.kldiv_ops.Proofs.ForwardStep

namespace VeriTile.Bench.TritonBenchG.KldivOps.FullForward
open VeriTile.Triton FullAlgorithm FullBackward
open scoped BigOperators

/-- Nonzero reduction modes update the accumulator once and perform no block store. -/
theorem accumulate (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (eps acc : ℝ) (logTarget : Bool) (reduction : Nat)
    (f : Fin B → ℝ) (hR : reduction ≠ 0)
    (hAcc : s.regs .real [] "loss_sum" = some (Tile.scalar (some acc)))
    (hLoss : s.regs .real [B] "loss" = some ⟨fun j ↦ some (f j.1)⟩) :
    stepStmts ((body Y T O sy st so N B eps logTarget reduction).drop 5) s =
      some (s.setReg "loss_sum" .real [] (Tile.scalar (some (acc + ∑ j : Fin B, f j)))) := by
  simp [body, forward, forwardSurface, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    stepStmts, stepStmt, evalOp_add, evalOp.eq_def, hR, hAcc, hLoss,
    Tile.cop, Tile.bop, ComparableDType.eq, NumericDType.add,
    Tile.scalar, -Tile.scalar_eta, Region.cast, Except.bind, bind,
    Option.bind_eq_bind, Option.bind]
  rfl

/-- Elementwise mode scatters the cached loss values under the actual source mask. -/
theorem elementwise_store (s : BlockState) (Y T O : RegionName)
    (sy st so N B outputBase : Nat) (current : Int) (eps : ℝ) (logTarget : Bool)
    (values : TileIndex [B] → ℝ)
    (hO : s.regs .ptr [] "O" = some (Tile.scalar (O, outputBase)))
    (hOffsets : s.regs .int [B] "offsets" = some ⟨offset current⟩)
    (hMask : s.regs .bool [B] "mask" = some ⟨active N current⟩)
    (hLoss : s.regs .real [B] "loss" = some ⟨fun j ↦ some (values j)⟩) :
    stepStmts ((body Y T O sy st so N B eps logTarget 0).drop 5) s =
      some (scatter s O outputBase N current values) := by
  simp [body, forward, forwardSurface, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    stepStmts, stepStmt, evalOp.eq_def, hO, hOffsets, hMask, hLoss,
    scatter, Tile.ptrAdd, Tile.cop, Tile.bop, Tile.uop,
    ComparableDType.eq, NumericDType.add, Tile.scalar, -Tile.scalar_eta,
    BlockState.writeMemTyped_real, Region.cast, Except.bind, bind]

end VeriTile.Bench.TritonBenchG.KldivOps.FullForward

#guard_msgs (drop info) in
#auditModuleAxioms
