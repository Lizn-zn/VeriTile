import bench.examples.TritonBenchVectorAddition.Kernels
/-
Real-valued correctness of the typed TritonBench vector-addition kernel.
The specification states pointwise addition on active lanes, with termination
and preservation of memory outside the output window. No numerical assumptions.
-/
import bench.tritonbench_g.vector_addition.VectorAddition
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.TritonBenchVectorAdditionCorrect
open VeriTile.Bench.Examples.TritonBenchVectorAddition.Kernels
open VeriTile Triton
open scoped VeriTile.Triton.MaskedKernelIO₂


/-- Erasing precision recovers the existing TritonBench mathematical kernel. -/
theorem real_projection (nElements blockSize : Nat) :
    (originalKernel nElements blockSize).toAlgorithm? =
      (VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel
        "x" "y" "output" nElements blockSize).toAlgorithm? := rfl

/-- Independent IO wiring, program window and active-lane predicate. -/
def addIO (nElements blockSize : Nat) : MaskedKernelIO₂ where
  kernel := originalKernel nElements blockSize
  in1 := "x"
  in2 := "y"
  out := "output"
  B := blockSize
  read1 := fun pid => pid * blockSize
  read2 := fun pid => pid * blockSize
  write := fun pid => pid * blockSize
  mask := fun pid i => pid * blockSize + i.val < nElements

/-- The kernel implements the mathematical formula x[i] + y[i]. -/
specification vector_addition_correct (nElements blockSize : Nat) :
    Spec.Real (addIO nElements blockSize ⊨ fun xs ys i => xs i + ys i) := by
  exact VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel_correctness
    "x" "y" "output" nElements blockSize

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.TritonBenchVectorAdditionCorrect
