import bench.examples.TritonBenchVectorAddition.Kernels
import bench.examples.TritonBenchVectorAddition.Proofs.Real
import bench.tritonbench_g.vector_addition.Correct
import VeriTile.Meta.StatementAudit

/-! Real correctness of the sources in Kernels.lean.
IO contracts and mathematical formulas are stated below; execution and memory
lemmas are in Proofs/Real.lean. Numerical casts are interpreted over the reals. -/

namespace VeriTile.Bench.Examples.TritonBenchVectorAdditionCorrect
open VeriTile.Bench.Examples.TritonBenchVectorAddition.Kernels
open VeriTile Triton
open scoped VeriTile.Triton.MaskedKernelIO₂

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

section Optimized

set_option maxHeartbeats 5000000

attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

/-- The optimized source retains the original IO windows and memory contract. -/
def optimizedIO (n B : Nat) : MaskedKernelIO₂ :=
  { addIO n B with kernel := optimizedKernel n B, projection := by rfl }

/-- The optimized implementation computes the same independent real formula. -/
specification vector_addition_optimized_correct (n B : Nat) :
    Spec.Real (optimizedIO n B ⊨ fun xs ys i => xs i + ys i) := by
  refine MaskedKernelIO₂.Implements.intro _
    ?_ ?_ ?_
  · simp [optimizedIO, optimizedKernel, Kernel.FlattenOk,
      StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]
  · intro bounds s h1 h2 h3 _
    simpa [optimizedIO, addIO, Kernel.TraceSafe, optimizedKernel, VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt,
      MemAccess.SafeAt, stepStmt, evalOp.eq_def,
      Region.cast, Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
      ComparableDType.lt, add_comm, optional_add_comm]
      using VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel_traceSafe "x" "y" "output" n B bounds s h1 h2 h3
  · intro s xs ys hx hy
    change ∃ s1, exec (optimizedKernel n B).toAlgKernel s = some s1 ∧ _
    rw [optimized_exec]
    obtain ⟨s1, he, hv, hf⟩ := VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel_region_run "x" "y" "output" n B s xs ys hx hy
    exact ⟨s1, he, hv, fun r o hmiss _ => hf r o hmiss⟩

#axiomsClean vector_addition_optimized_correct
#stmtSurfaceSubset vector_addition_optimized_correct ⊆
  [Spec.Real, optimizedIO, VeriTile.Triton.MaskedKernelIO₂.Implements, VeriTile.Triton.MaskedKernelIO₂.B]

end Optimized

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.TritonBenchVectorAdditionCorrect
