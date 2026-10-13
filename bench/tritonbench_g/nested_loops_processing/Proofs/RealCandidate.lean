import bench.tritonbench_g.nested_loops_processing.Kernels
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.RealCandidate
open VeriTile.Triton TypedSource
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def algorithm (storage : Storage) (v : Bool) (src dst : RegionName) (sm sn : Int) : Kernel :=
  (kernel storage v src dst sm sn).eraseDType.toAlgKernel

def initialPointer (v : Bool) (src dst : RegionName) (sm sn : Int) : Stmt :=
  ((algorithm .int32 v src dst sm sn).body[2]?).getD (.forRange "" 0 0 1 [])

theorem pointer_step (src dst : RegionName) (sm sn : Int) (s : BlockState) :
    stepStmt (initialPointer Bool.false src dst sm sn) s =
      stepStmt (initialPointer Bool.true src dst sm sn) s := by
  simp [initialPointer, algorithm, kernel, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  cases ha : s.regs .nat [2] "offs_am" <;> cases hb : s.regs .nat [2] "offs_an"
  all_goals simp [stepStmt, evalOp.eq_def, ha, hb, Tile.bop, Tile.uop,
    Tile.ptrAdd, NumericDType.mul, NumericDType.add, Int.mul_comm]

theorem decomposition (storage : Storage) (v : Bool)
    (src dst : RegionName) (sm sn : Int) :
    (algorithm storage v src dst sm sn).body =
      (algorithm .int32 Bool.false src dst sm sn).body.take 2 ++
      [initialPointer v src dst sm sn] ++
      (algorithm storage Bool.false src dst sm sn).body.drop 3 := by
  cases storage <;> cases v
  all_goals simp [algorithm, initialPointer, kernel, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

theorem execution (storage : Storage) (src dst : RegionName) (sm sn : Int) (s : BlockState) :
    stepStmts (algorithm storage Bool.false src dst sm sn).body s =
      stepStmts (algorithm storage Bool.true src dst sm sn).body s := by
  conv_lhs => rw [decomposition storage Bool.false]
  conv_rhs => rw [decomposition storage Bool.true]
  apply ProgramRewrite.context
  intro t
  simp only [stepStmts, pointer_step]

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.RealCandidate

#guard_msgs (drop info) in
#auditModuleAxioms
