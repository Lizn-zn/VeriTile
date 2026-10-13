import bench.tritonbench_g.kldiv_ops.Proofs.Algorithm
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KldivOps.FullAlgorithm
open VeriTile.Triton

/-- Nonzero reduction modes store the accumulated scalar once after the loop;
elementwise mode performs no final store. -/
theorem forward_finish (s : BlockState) (Y T O : RegionName)
    (sy st so N B outputAddress : Nat) (eps acc : ℝ)
    (logTarget : Bool) (reduction : Nat)
    (hptr : s.regs .ptr [] "O" = some (Tile.scalar (O, outputAddress)))
    (hacc : s.regs .real [] "loss_sum" = some (Tile.scalar (some acc))) :
    stepStmts ((forward Y T O sy st so N B eps logTarget reduction).body.drop 7) s =
      some (if reduction = 0 then s else s.writeMem O outputAddress acc) := by
  by_cases h : reduction = 0
  all_goals simp [forward, forwardSurface, ComputeKernel.surfaceBody,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    stepStmts, stepStmt, evalOp.eq_def, h, hptr, hacc, Tile.cop,
    ComparableDType.eq, ComparableDType.ne, Tile.scalar, -Tile.scalar_eta, TileShape.allIndices,
    BlockState.writeMemTyped_real, Region.cast, Except.bind, bind]

end VeriTile.Bench.TritonBenchG.KldivOps.FullAlgorithm

#guard_msgs (drop info) in
#auditModuleAxioms
