import VeriTile.Examples.SoftmaxEq

/-!
Refinement theorem (`ComputeRefine.Refines`, writes-equality) for the naive
vs stable softmax kernel pair. The kernels, specs, correctness lemmas, and the
exec-level refinement live in `VeriTile.Examples.SoftmaxEq`.

Headline: `softmax_kernels_refinement_view`. For the rounding-model (∀R) variant of the
writes-equality surface see `bench/examples/Swiglu.lean`.
-/

namespace VeriTile.Examples

open VeriTile.Triton VeriTile.Triton.TiledSoftmax

/-- Writes-equality refinement surface for `softmax_kernels_refinement`:
from the same initial state, the naive and stable softmax kernels perform
the same writes (no scratch regions). -/
theorem softmax_kernels_refinement_view
    (xReg yReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1)) :
    ComputeRefine.Refines
      (naiveSoftmaxKernel xReg yReg blockSize)
      (stableSoftmaxKernel xReg yReg blockSize) s [] := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  apply ComputeKernel.computeRefine_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r hr o
  simp [exec, naiveSoftmaxKernel, stepStmts, stepStmt, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul, NumericDType.div] at hL
  simp [exec, stableSoftmaxKernel, stepStmts, stepStmt, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div] at hR
  repeat unfold evalOp at hL
  repeat unfold evalOp at hR
  simp [Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hL
  simp [Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hR
  subst lhs'
  subst rhs'
  refine BlockState.foldl_writeMem_mem_congr _ _ _ _ ?_ r o _ _ rfl
  intro k _
  exact congrFun (naive_eq_stable
    (fun i : Fin (n + 1) => s.readMem xReg (s.pids 0 * (n + 1) + i.val)) _) k.1

end VeriTile.Examples
