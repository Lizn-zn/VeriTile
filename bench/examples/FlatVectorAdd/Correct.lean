import bench.examples.FlatVectorAdd.Kernels
import bench.examples.FlatVectorAdd.Proofs.Real
import VeriTile.Triton
import VeriTile.Triton.Memory.Flatten
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-! Real correctness of the sources in Kernels.lean.
IO contracts and mathematical formulas are stated below; execution and memory
lemmas are in Proofs/Real.lean. Numerical casts are interpreted over the reals. -/

namespace VeriTile.Bench.Examples.FlatVectorAdd
open VeriTile.Bench.Examples.FlatVectorAdd.Kernels

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?
  VeriTile.Triton.ComputeOp.toAlgorithm? VeriTile.Triton.ComputeDType.eraseDType

open VeriTile.Triton
open VeriTile.Triton.MaskedKernelIO₂ (Implements)
open scoped VeriTile.Triton.MaskedKernelIO₂
open VeriTile.Examples (observeAt)

/-- Masked tile windows: only lanes with pid * B + i < n are active. -/
def addMaskedIO (B n : Nat) : MaskedKernelIO₂ where
  kernel := addKernelMasked ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n
  in1 := ⟨"x"⟩
  in2 := ⟨"y"⟩
  out := ⟨"out"⟩
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  write := fun pid => pid * B
  mask := fun pid j => pid * B + j.val < n

/-- Pointwise addition on active lanes, with successful execution and memory framing. -/
specification add_kernel_masked_correctness (B n : Nat) (hB : 0 < B) :
    Spec.Real (addMaskedIO B n ⊨ fun xs ys i => xs i + ys i) := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact addKernelMasked_flattenOk ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n
  · intro bounds s h1 h2 h3 _
    exact addKernelMasked_traceSafe ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B n bounds s h1 h2 h3
  · intro s₀ xs ys hx hy
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      addKernelMasked_region_run B n hB s₀ xs ys hx hy
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

section Optimized

set_option maxHeartbeats 5000000

attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

/-- The optimized source retains the original IO windows and memory contract. -/
def optimizedIO (B n : Nat) : MaskedKernelIO₂ :=
  { addMaskedIO B n with kernel := optimizedKernel n B, projection := by rfl }

/-- The optimized implementation computes the same independent real formula. -/
specification add_kernel_masked_optimized_correctness (B n : Nat) (hB : 0 < B) :
    Spec.Real (optimizedIO B n ⊨ fun xs ys i => xs i + ys i) := by
  refine MaskedKernelIO₂.Implements.intro _
    ?_ ?_ ?_
  · simp [optimizedIO, optimizedKernel, Kernel.FlattenOk,
      StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]
  · intro bounds s h1 h2 h3 _
    simpa [optimizedIO, addMaskedIO, Kernel.TraceSafe, optimizedKernel, addKernelMasked,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt,
      MemAccess.SafeAt, stepStmt, evalOp.eq_def,
      Region.cast, Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
      ComparableDType.lt, add_comm, optional_add_comm]
      using addKernelMasked_traceSafe "x" "y" "out" B n bounds s h1 h2 h3
  · intro s xs ys hx hy
    change ∃ s1, exec (optimizedKernel n B).toAlgKernel s = some s1 ∧ _
    rw [optimized_exec]
    obtain ⟨s1, he, hv, hf⟩ := addKernelMasked_region_run B n hB s xs ys hx hy
    exact ⟨s1, he, hv, fun r o hmiss _ => hf r o hmiss⟩

#axiomsClean add_kernel_masked_optimized_correctness
#stmtSurfaceSubset add_kernel_masked_optimized_correctness ⊆
  [Spec.Real, optimizedIO, VeriTile.Triton.MaskedKernelIO₂.Implements,
   VeriTile.Triton.MaskedKernelIO₂.B]

end Optimized

#axiomsClean add_kernel_masked_correct
#axiomsClean add_kernel_masked_correctness
#stmtSurfaceSubset add_kernel_masked_correctness ⊆
  [Spec.Real, addMaskedIO, VeriTile.Triton.MaskedKernelIO₂.Implements,
   VeriTile.Triton.MaskedKernelIO₂.B]

end VeriTile.Bench.Examples.FlatVectorAdd
