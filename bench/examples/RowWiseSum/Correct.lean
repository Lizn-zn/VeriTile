import bench.examples.RowWiseSum.Kernels
import bench.examples.RowWiseSum.Proofs.Real
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Fin.Rev
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Reduction
import VeriTile.Triton.Memory.KernelSpec
import VeriTile.Examples.Common
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit

/-! Real correctness of the sources in Kernels.lean.
IO contracts and mathematical formulas are stated below; execution and memory
lemmas are in Proofs/Real.lean. Numerical casts are interpreted over the reals. -/

namespace VeriTile.Bench.Examples.RowWiseSum
open VeriTile.Bench.Examples.RowWiseSum.Kernels

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?
  VeriTile.Triton.ComputeOp.toAlgorithm? VeriTile.Triton.ComputeDType.eraseDType

open VeriTile.Triton
open VeriTile.Triton.KernelIO₁ (Implements)
open scoped VeriTile.Triton.KernelIO₁
open VeriTile.Examples

/-- Read B elements from row offset pid * nCol and write one sum at pid. -/
def rowWiseSumIO (nCol B : Nat) : KernelIO₁ where
  kernel := rowWiseSumKernel ⟨"x"⟩ ⟨"y"⟩ nCol B
  inp := ⟨"x"⟩
  out := ⟨"y"⟩
  Bin := B
  Bout := 1
  read := fun pid => pid * nCol
  write := fun pid => pid

/-- The original kernel computes the mathematical sum of the input row. -/
specification rowWiseSum_correctness (nCol B : Nat) :
    Spec.Real (rowWiseSumIO nCol B ⊨ fun xs _ => ∑ k, xs k) := by
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact rowWiseSum_flattenOk ⟨"x"⟩ ⟨"y"⟩ nCol B
  · intro bounds s h1 h2 _
    exact rowWiseSum_traceSafe ⟨"x"⟩ ⟨"y"⟩ nCol B bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := rowWiseSum_region_run nCol B s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

section Optimized

set_option maxHeartbeats 1600000

def reversedIO (nCol B : Nat) : KernelIO₁ :=
  { rowWiseSumIO nCol B with kernel := reversedKernel "x" "y" nCol B, projection := by rfl }

/-- The optimized source computes the same mathematical row reduction. -/
specification rowWiseSum_reversed_correctness (nCol B : Nat) :
    Spec.Real (reversedIO nCol B ⊨ fun xs _ => ∑ k, xs k) := by
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · simp [reversedIO, reversedKernel, Kernel.FlattenOk,
      StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]
  · intro bounds s hx hy _
    simp [reversedIO, Kernel.TraceSafe, reversedKernel,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt,
      MemAccess.SafeAt, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
      MaskOpt.Active, BlockState.setReg, stepStmt, evalOp.eq_def,
      Region.cast, Tile.bop, NumericDType.add, NumericDType.mul, NumericDType.sub,
      Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
    refine ⟨?_, ?_⟩
    · intro i
      have hi : B - 1 - i.val < B := by omega
      exact lt_of_lt_of_le (Nat.add_lt_add_left hi _) hx
    · exact Nat.lt_of_add_one_le hy
  · intro s xs hx
    obtain ⟨t, he, hv, hf⟩ := reversed_region_run nCol B s xs hx
    exact ⟨t, he, hv, fun r o hmiss _ => hf r o hmiss⟩

#axiomsClean rowWiseSum_reversed_correctness
#stmtSurfaceSubset rowWiseSum_reversed_correctness ⊆
  [Spec.Real, reversedIO, VeriTile.Triton.KernelIO₁.Implements,
   VeriTile.Triton.KernelIO₁.Bin, VeriTile.Triton.KernelIO₁.Bout,
   Finset.sum, Finset.univ]

end Optimized

#axiomsClean rowWiseSum_correctness
#axiomsClean rowWiseSum_correct
#stmtSurfaceSubset rowWiseSum_correctness ⊆
  [Spec.Real, rowWiseSumIO, VeriTile.Triton.KernelIO₁.Implements,
   VeriTile.Triton.KernelIO₁.Bin, VeriTile.Triton.KernelIO₁.Bout,
   Finset.sum, Finset.univ]

end VeriTile.Bench.Examples.RowWiseSum
