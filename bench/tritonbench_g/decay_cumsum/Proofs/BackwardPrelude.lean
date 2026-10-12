import bench.tritonbench_g.decay_cumsum.Proofs.BackwardProjection
import bench.tritonbench_g.decay_cumsum.Proofs.BackwardRegisterFrame
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.BackwardPrelude
open VeriTile.Triton TypedSource
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def referencePrelude (dq_inner dq_inter dk_inner dk_inter q k g dg : RegionName) (sH DK BT BK : Nat) (scale : ℝ) : List Stmt :=
  ((RealReference.bwd_decay_global_cumsum_surface dq_inner dq_inter dk_inner dk_inter q k g dg sH DK BT BK).toAlgKernel.body).take 15

/-- The old cached arange register is the only additional final-state binding.
Both preludes initialize the same eight pointers and two fp32 accumulators. -/
theorem equivalent (dq_inner dq_inter dk_inner dk_inter q k g dg : RegionName) (sH DK BT BK : Nat)
    (scale : ℝ) (u : BlockState) :
    stepStmts (referencePrelude dq_inner dq_inter dk_inner dk_inter q k g dg sH DK BT BK scale) u =
      (stepStmts (BackwardProjection.prelude dq_inner dq_inter dk_inner dk_inter q k g dg sH DK BT BK scale) u).map
        (BackwardRegisterFrame.setOffs BK) := by
  simp only [referencePrelude,BackwardProjection.prelude,backward,RealReference.bwd_decay_global_cumsum_surface,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,Except.bind,bind,eraseDType_float]
  simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType]
  simp only [List.take,List.cons_append,List.nil_append,stepStmts,stepStmt,evalOp,
    Option.bind,Option.map,Tile.scalar_data]
  simp [-Tile.scalar_eta,BackwardRegisterFrame.setOffs,RealReference.prepOffsTile,BlockState.setReg,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,Tile.vec,NumericDType.add,
    NumericDType.mul,NumericDType.sub,ComparableDType.lt]
  funext d sh name
  by_cases h : name="offs" <;> simp [h,BlockState.readMem]

/-- The old cached arange register is the only additional final-state binding.
Both preludes initialize the same eight pointers and two fp32 accumulators. -/
theorem flat_equivalent (A : FlatAlloc) (dq_inner dq_inter dk_inner dk_inter q k g dg : RegionName) (sH DK BT BK : Nat)
    (scale : ℝ) (u : BlockState) :
    stepStmts (A.flattenStmts (referencePrelude dq_inner dq_inter dk_inner dk_inter q k g dg sH DK BT BK scale)) u =
      (stepStmts (A.flattenStmts (BackwardProjection.prelude dq_inner dq_inter dk_inner dk_inter q k g dg sH DK BT BK scale)) u).map
        (BackwardRegisterFrame.setOffs BK) := by
  simp only [referencePrelude,BackwardProjection.prelude,backward,RealReference.bwd_decay_global_cumsum_surface,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,Except.bind,bind,eraseDType_float]
  simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType]
  simp only [List.take,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
    List.cons_append,List.nil_append,stepStmts,stepStmt,evalOp,
    Option.bind,Option.map,Tile.scalar_data]
  simp [-Tile.scalar_eta,BackwardRegisterFrame.setOffs,RealReference.prepOffsTile,BlockState.setReg,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,Tile.vec,NumericDType.add,
    NumericDType.mul,NumericDType.sub,ComparableDType.lt]
  funext d sh name
  by_cases h : name="offs" <;> simp [h,BlockState.readMem]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DecayCumsum.BackwardPrelude
