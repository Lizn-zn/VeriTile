import bench.examples.TritonBenchVectorAddition.Kernels
/-
Real-valued correctness of the typed TritonBench vector-addition kernel.
The specification states pointwise addition on active lanes, with termination
and preservation of memory outside the output window. No numerical assumptions.

The optimized implementation in Kernels.lean has its own real specification
below, with the same mathematical formula and IO contract. Its proof uses
real arithmetic and execution semantics independently of FPEquiv.lean.
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

/-! ## Optimized implementation: real correctness -/

section Optimized
set_option maxHeartbeats 5000000
attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

private theorem optional_add_comm (a b : Option ℝ) :
    Option.map₂ (fun x y => x + y) a b = Option.map₂ (fun x y => x + y) b a :=
  Option.map₂_comm add_comm

-- This equality is derived in the real interpreter; it uses no FP assumptions.
private theorem optimized_exec (n B : Nat) (s : BlockState) :
    exec (optimizedKernel n B).toAlgKernel s =
      exec (VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel "x" "y" "output" n B).toAlgKernel s := by
  simp [optimizedKernel, VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Region.cast, Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
      ComparableDType.lt, add_comm, optional_add_comm]

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
