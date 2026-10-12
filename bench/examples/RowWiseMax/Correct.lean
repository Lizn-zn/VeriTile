import bench.examples.RowWiseMax.Kernels
import bench.examples.RowWiseMax.Proofs.Real
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
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

namespace VeriTile.Bench.Examples.RowWiseMax
open VeriTile.Bench.Examples.RowWiseMax.Kernels
open VeriTile.Triton
open VeriTile.Triton.KernelIO₁ (Implements)
open scoped VeriTile.Triton.KernelIO₁
open VeriTile.Examples

/-- Read B elements from row offset pid * nCol and write one maximum at pid. -/
def rowWiseMaxIO (nCol B : Nat) : KernelIO₁ where
  kernel := rowWiseMaxKernel ⟨"x"⟩ ⟨"y"⟩ nCol B
  inp := ⟨"x"⟩
  out := ⟨"y"⟩
  Bin := B
  Bout := 1
  read := fun pid => pid * nCol
  write := fun pid => pid

/-- The original kernel computes the maximum of a nonempty input row. -/
specification rowWiseMax_correctness (nCol B : Nat) (hB : 0 < B) :
    Spec.Real (rowWiseMaxIO nCol B ⊨ fun xs _ => Triton.TiledReduction.tileMax hB xs) := by
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact rowWiseMax_flattenOk ⟨"x"⟩ ⟨"y"⟩ nCol B
  · intro bounds s h1 h2 _
    exact rowWiseMax_traceSafe ⟨"x"⟩ ⟨"y"⟩ nCol B bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := rowWiseMax_region_run nCol B hB s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

section Optimized

set_option maxHeartbeats 1600000

def inlinedIO (nCol B : Nat) : KernelIO₁ :=
  { rowWiseMaxIO nCol B with kernel := inlinedKernel "x" "y" nCol B, projection := by rfl }

/-- The optimized source computes the same mathematical row reduction. -/
specification rowWiseMax_inlined_correctness (nCol B : Nat) (hB : 0 < B) :
    Spec.Real (inlinedIO nCol B ⊨ fun xs _ => Triton.TiledReduction.tileMax hB xs) := by
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · simp [inlinedIO, inlinedKernel, Kernel.FlattenOk,
      StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]
  · intro bounds s hx hy _
    obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
    simpa [inlinedIO, rowWiseMaxIO, Kernel.TraceSafe, inlinedKernel, rowWiseMaxKernel,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt,
      MemAccess.SafeAt, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
      MaskOpt.Active, BlockState.setReg, stepStmt, evalOp.eq_def,
      Tile.bop, NumericDType.add, NumericDType.mul, Tile.reduceMaxDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex, and_comm]
      using rowWiseMax_traceSafe "x" "y" nCol (n+1) bounds s hx hy
  · intro s xs hx
    obtain ⟨t, he, hv, hf⟩ := inlined_region_run nCol B hB s xs hx
    exact ⟨t, he, hv, fun r o hmiss _ => hf r o hmiss⟩

#axiomsClean rowWiseMax_inlined_correctness
#stmtSurfaceSubset rowWiseMax_inlined_correctness ⊆
  [Spec.Real, inlinedIO, VeriTile.Triton.KernelIO₁.Implements,
   VeriTile.Triton.KernelIO₁.Bin, VeriTile.Triton.KernelIO₁.Bout,
   VeriTile.Triton.TiledReduction.tileMax]

end Optimized

#axiomsClean rowWiseMax_correctness
#axiomsClean rowWiseMax_correct
#stmtSurfaceSubset rowWiseMax_correctness ⊆
  [Spec.Real, rowWiseMaxIO, VeriTile.Triton.KernelIO₁.Implements,
   VeriTile.Triton.KernelIO₁.Bin, VeriTile.Triton.KernelIO₁.Bout,
   VeriTile.Triton.TiledReduction.tileMax]

end VeriTile.Bench.Examples.RowWiseMax
