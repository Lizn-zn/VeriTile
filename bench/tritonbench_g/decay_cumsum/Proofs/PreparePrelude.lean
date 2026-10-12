import bench.tritonbench_g.decay_cumsum.Proofs.PrepareProjection
import bench.tritonbench_g.decay_cumsum.Proofs.PrepareRegisterFrame
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.PreparePrelude
open VeriTile.Triton TypedSource
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def referencePrelude (q k g qg kg : RegionName) (sH DK BT BK : Nat) (scale : ℝ) : List Stmt :=
  ((RealReference.prepare_qg_kg_surface q k g qg kg sH DK BT BK scale).toAlgKernel.body).take 11

/-- The old cached arange register is the only additional final-state binding.
Both preludes include the same unmasked last-gate load. -/
theorem equivalent (q k g qg kg : RegionName) (sH DK BT BK : Nat)
    (scale : ℝ) (u : BlockState) :
    stepStmts (referencePrelude q k g qg kg sH DK BT BK scale) u =
      (stepStmts (PrepareProjection.prelude q k g qg kg sH DK BT BK scale) u).map
        (PrepareRegisterFrame.setOffs BK) := by
  simp only [referencePrelude,PrepareProjection.prelude,prepare,RealReference.prepare_qg_kg_surface,
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
  simp [-Tile.scalar_eta,PrepareRegisterFrame.setOffs,RealReference.prepOffsTile,BlockState.setReg,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,Tile.vec,NumericDType.add,
    NumericDType.mul,NumericDType.sub,ComparableDType.lt]
  funext d sh name
  by_cases h : name="offs" <;> simp [h,BlockState.readMem]

/-- The old cached arange register is the only additional final-state binding.
Both preludes include the same unmasked last-gate load. -/
theorem flat_equivalent (A : FlatAlloc) (q k g qg kg : RegionName) (sH DK BT BK : Nat)
    (scale : ℝ) (u : BlockState) :
    stepStmts (A.flattenStmts (referencePrelude q k g qg kg sH DK BT BK scale)) u =
      (stepStmts (A.flattenStmts (PrepareProjection.prelude q k g qg kg sH DK BT BK scale)) u).map
        (PrepareRegisterFrame.setOffs BK) := by
  simp only [referencePrelude,PrepareProjection.prelude,prepare,RealReference.prepare_qg_kg_surface,
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
  simp [-Tile.scalar_eta,PrepareRegisterFrame.setOffs,RealReference.prepOffsTile,BlockState.setReg,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,Tile.vec,NumericDType.add,
    NumericDType.mul,NumericDType.sub,ComparableDType.lt]
  funext d sh name
  by_cases h : name="offs" <;> simp [h,BlockState.readMem]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DecayCumsum.PreparePrelude
