import bench.tritonbench_g.attn_fwd_causal.Proofs.RealAccessSafety

/-! Both complete erased sources belong to the existing flat-memory bridge's
supported syntax. No alternate interpreter or replacement kernel is used. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealFlatten
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem supported (Q K V QS KS Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage : Nat) (hs : stage = 2 ∨ stage = 3) (candidate : Bool) :
    ((if candidate then TypedSource.reciprocal else TypedSource.original)
      Q K V QS KS Out sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage hs).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk, RealSource.body]
  cases candidate
  all_goals
    simp [RealPrologue.code, RealCoordinates.code, RealCoordinates.ceilExpr,
      RealPointers.code, RealPointers.rowExpr, RealPointers.rowPointerExpr, RealPointers.keyExpr,
      RealPointers.queryScaleExpr, RealPointers.keyScaleExpr, RealInit.code, RealInit.queryMask,
      RealMiddle.code, RealFirstSetup.code, RealFirstSetup.copies,
      RealTwoLoops.code, RealSetupSource.statement, RealLoop.sourceBody,
      RealSetup.code, RealSetup.lowerExpr, RealSetup.upperExpr,
      RealBody.first, RealBody.second, RealBody.loadPrefix, RealBody.keyMask, RealBody.valueMask,
      RealBody.score, RealBody.scoreExpr, RealBody.diagonal, RealBody.center, RealBody.centerExpr,
      RealBody.suffix, RealBody.probability, RealBody.rescale, RealBody.loadValue, RealBody.accumulate,
      RealBody.advance, RealExpressions.maskExpr, RealExpressions.penaltyExpr,
      RealSource.epilogue, RealSource.normalization, RealSource.store,
      StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealFlatten
