import bench.examples.VectorAdd.Kernels
import bench.examples.VectorAdd.Proofs.Real
import VeriTile.Triton
import VeriTile.Triton.Memory.Flatten
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-! Real correctness of the sources in Kernels.lean.
IO contracts and mathematical formulas are stated below; execution and memory
lemmas are in Proofs/Real.lean. Numerical casts are interpreted over the reals. -/

namespace VeriTile.Bench.Examples.VectorAdd
open VeriTile.Bench.Examples.VectorAdd.Kernels

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?
  VeriTile.Triton.ComputeOp.toAlgorithm? VeriTile.Triton.ComputeDType.eraseDType

open VeriTile.Triton
open VeriTile.Triton.KernelIO₂ (Implements)
open scoped VeriTile.Triton.KernelIO₂
open VeriTile.Examples

/-- Input/output windows for one aligned tile; allocation bounds remain quantified by the specification. -/
def addIO (B : Nat) : KernelIO₂ where
  kernel := addKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B
  in1 := ⟨"x"⟩
  in2 := ⟨"y"⟩
  out := ⟨"out"⟩
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  write := fun pid => pid * B

/-- The original kernel computes pointwise addition and preserves all other memory. -/
specification add_kernel_correctness (B : Nat) (hB : 0 < B) :
    Spec.Real (addIO B ⊨ fun xs ys i => xs i + ys i) := by
  refine KernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact addKernel_flattenOk ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B
  · intro bounds s h1 h2 h3
    exact addKernel_traceSafe ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B bounds s h1 h2 h3
  · intro s₀ xs ys hx hy
    exact addKernel_region_run B hB s₀ xs ys hx hy

section Optimized

set_option maxHeartbeats 5000000

attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

/-- The optimized source retains the original IO windows and memory contract. -/
def optimizedIO (B : Nat) : KernelIO₂ :=
  { addIO B with kernel := optimizedKernel B, projection := by rfl }

/-- The optimized implementation computes the same independent real formula. -/
specification add_kernel_optimized_correctness (B : Nat) :
    Spec.Real (optimizedIO B ⊨ fun xs ys i => xs i + ys i) := by
  refine KernelIO₂.Implements.intro _
    ?_ ?_ ?_
  · simp [optimizedIO, optimizedKernel, Kernel.FlattenOk,
      StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]
  · intro bounds s h1 h2 h3
    simpa [optimizedIO, addIO, Kernel.TraceSafe, optimizedKernel, addKernel,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt,
      MemAccess.SafeAt, stepStmt, evalOp.eq_def,
      Region.cast, Tile.bop, NumericDType.add, NumericDType.mul, add_comm]
      using addKernel_traceSafe "x" "y" "out" B bounds s h1 h2 h3
  · intro s xs ys hx hy
    change ∃ s1, exec (optimizedKernel B).toAlgKernel s = some s1 ∧ _
    rw [optimized_exec]
    rcases Nat.eq_zero_or_pos B with rfl | hB
    · simp [optimizedIO, addIO, addKernel, exec, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, NumericDType.add, NumericDType.mul, TileShape.allIndices]
    · exact addKernel_region_run B hB s xs ys hx hy

#axiomsClean add_kernel_optimized_correctness
#stmtSurfaceSubset add_kernel_optimized_correctness ⊆
  [Spec.Real, optimizedIO, VeriTile.Triton.KernelIO₂.Implements, VeriTile.Triton.KernelIO₂.B]

end Optimized

#axiomsClean add_kernel_correctness
#stmtSurfaceSubset add_kernel_correctness ⊆
  [Spec.Real, addIO, VeriTile.Triton.KernelIO₂.Implements, VeriTile.Triton.KernelIO₂.B]

end VeriTile.Bench.Examples.VectorAdd
