import bench.tritonbench_g.nested_loops_processing.Proofs.RealCandidate
import VeriTile.Triton.Memory.FlattenTolerant

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.RealCandidate
open VeriTile.Triton TypedSource
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem pointer_flat_step (A : FlatAlloc) (src dst : RegionName) (sm sn : Int) (s : BlockState) :
    stepStmt (A.flattenStmt (initialPointer Bool.false src dst sm sn)) s =
      stepStmt (A.flattenStmt (initialPointer Bool.true src dst sm sn)) s := by
  simp [initialPointer, algorithm, kernel, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  cases ha : s.regs .nat [2] "offs_am" <;> cases hb : s.regs .nat [2] "offs_an"
  all_goals simp [FlatAlloc.flattenStmt, FlatAlloc.flattenOp.eq_def,
    FlatAlloc.flattenAccess, FlatAlloc.flattenMask, stepStmt, evalOp.eq_def, ha, hb, Tile.bop, Tile.uop,
    Tile.ptrAdd, NumericDType.mul, NumericDType.add, Int.mul_comm]
theorem flat_execution (A : FlatAlloc) (storage : Storage)
    (src dst : RegionName) (sm sn : Int) (s : BlockState) :
    exec (A.flattenKernel (algorithm storage Bool.false src dst sm sn)) s =
      exec (A.flattenKernel (algorithm storage Bool.true src dst sm sn)) s := by
  change stepStmts (A.flattenStmts (algorithm storage Bool.false src dst sm sn).body) s =
    stepStmts (A.flattenStmts (algorithm storage Bool.true src dst sm sn).body) s
  conv_lhs => rw [decomposition storage Bool.false]
  conv_rhs => rw [decomposition storage Bool.true]
  simp only [FlatAlloc.flattenStmts_append]
  apply ProgramRewrite.context
  intro t
  simp only [FlatAlloc.flattenStmts, stepStmts, pointer_flat_step]

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.RealCandidate

#guard_msgs (drop info) in
#auditModuleAxioms
