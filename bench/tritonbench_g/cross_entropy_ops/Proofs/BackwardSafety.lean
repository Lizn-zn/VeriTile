import bench.tritonbench_g.cross_entropy_ops.Proofs.BackwardProgram
import VeriTile.Triton.Memory.TraceProofs
namespace VeriTile.Bench.TritonBenchG.CrossEntropyOps.BackwardSafety
open VeriTile.Triton BackwardFormula BackwardProgram
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

/-- Bounds follow actual source accesses. An ignored label suppresses only
the upstream-loss load; label, LSE and active logits are still read. -/
structure Bounds (p : Layout) (r : Regions) (l : Launch) (a : Inputs) (extent : RegionBounds) : Prop where
  label : l.row < extent (Region.cast r.label)
  lse : l.row < extent r.lse
  dloss : a.label l.row ≠ p.ignored → l.row*p.dlossStride < extent r.dloss
  logits : ∀ i : Fin p.B, active p l i.val → logitAddress p l i.val < extent r.logits
  gradient : ∀ i : Fin p.B, active p l i.val → gradientAddress p l i.val < extent r.gradient

theorem traceSafe (p : Layout) (r : Regions) (extent : RegionBounds) (u : BlockState)
    (hb : Bounds p r (launch u) (inputs u r) extent) :
    Kernel.TraceSafe extent (mathematical p r).toAlgKernel u := by
  obtain ⟨hl,hs,hd,hx,ho⟩ := hb
  simp only [launch,inputs,active,column,logitAddress,gradientAddress,Region.cast] at hl hs hd hx ho
  cases hm : p.hasSmoothing <;>
    by_cases hi : u.readMemValue .int r.label (u.pids 0) = p.ignored
  all_goals simp only [Region.cast] at hi
  all_goals
    simp [Kernel.TraceSafe,mathematical,realKernel,ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,
      ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Stmt.TraceSafeList,Stmt.TraceSafe,
      Op.SafeAt.eq_def,MaskOpt.SafeAt,stepStmts,stepStmt,evalOp.eq_def,
      Option.bind,Tile.bop,Tile.uop,Tile.cop,Tile.ptrAdd,NumericDType.add,
      NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt,ComparableDType.eq,
      ComparableDType.ne,Region.cast,hm,hi,MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,
      MemAccess.SafeAt,MaskOpt.Active,Op.PointerAddressesSafeOn,Op.MemorySafe,hl,hs,hx,ho]
  all_goals first | exact hd hi | exact ⟨hx,ho⟩ | exact ⟨hd hi,hx,ho⟩

theorem flattenOk (p : Layout) (r : Regions) : (mathematical p r).toAlgKernel.FlattenOk := by
  cases hs : p.hasSmoothing
  all_goals simp [Kernel.FlattenOk,mathematical,realKernel,hs,ComputeKernel.toAlgKernel,
    ComputeKernel.surfaceBody,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CrossEntropyOps.BackwardSafety
