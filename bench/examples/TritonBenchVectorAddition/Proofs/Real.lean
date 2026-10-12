import bench.examples.TritonBenchVectorAddition.Kernels
import bench.tritonbench_g.vector_addition.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Real execution, arithmetic and memory lemmas for TritonBenchVectorAddition.
The public IO contracts and mathematical specifications are in ../Correct.lean. -/

namespace VeriTile.Bench.Examples.TritonBenchVectorAdditionCorrect
open VeriTile.Bench.Examples.TritonBenchVectorAddition.Kernels
open VeriTile Triton
open scoped VeriTile.Triton.MaskedKernelIO₂

/-- Erasing precision recovers the existing TritonBench mathematical kernel. -/
theorem real_projection (nElements blockSize : Nat) :
    (originalKernel nElements blockSize).toAlgorithm? =
      (VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel
        "x" "y" "output" nElements blockSize).toAlgorithm? := rfl

section Optimized

set_option maxHeartbeats 5000000

attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

theorem optional_add_comm (a b : Option ℝ) :
    Option.map₂ (fun x y => x + y) a b = Option.map₂ (fun x y => x + y) b a :=
  Option.map₂_comm add_comm

theorem optimized_exec (n B : Nat) (s : BlockState) :
    exec (optimizedKernel n B).toAlgKernel s =
      exec (VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel "x" "y" "output" n B).toAlgKernel s := by
  simp [optimizedKernel, VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Region.cast, Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
      ComparableDType.lt, add_comm, optional_add_comm]

end Optimized

end VeriTile.Bench.Examples.TritonBenchVectorAdditionCorrect
