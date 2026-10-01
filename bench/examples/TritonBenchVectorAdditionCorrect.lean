/-
Real-valued correctness of the typed TritonBench vector-addition kernel.
The specification states pointwise addition on active lanes, with termination
and preservation of memory outside the output window. No numerical assumptions.
-/
import bench.tritonbench_g.vector_addition.VectorAddition
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.TritonBenchVectorAdditionCorrect

open VeriTile Triton
open scoped VeriTile.Triton.MaskedKernelIO₂

/-- The fp32 kernel; correctness interprets its mathematical projection. -/
def originalKernel (nElements blockSize : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := ⟨"x"⟩
  let y_ptr : Region .fp32 := ⟨"y"⟩
  let output_ptr : Region .fp32 := ⟨"output"⟩
  triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(blockSize)
  offsets = block_start + tl.arange(0, $(blockSize))
  mask = offsets < $(nElements)
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  tl.store(output_ptr + offsets, output, mask=mask)
}

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
