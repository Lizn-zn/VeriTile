import VeriTile.Examples.SoftmaxReciprocal

/-!
Refinement theorem (`ComputeRefine.Refines`, writes-equality) for the
reciprocal-softmax rewrite. The kernels, specs, correctness lemmas, and the
exec-level refinement live in `VeriTile.Examples.SoftmaxReciprocal`.

Headline: `softmax_reciprocal_refinement_view`. For the rounding-model (∀R) variant of the
writes-equality surface see `bench/examples/SwigluRoundingInvariance.lean`.
-/

namespace VeriTile.Examples

open VeriTile.Triton VeriTile.Triton.TiledSoftmax

/-- Writes-equality refinement surface for the reciprocal softmax rewrite:
from the same initial state, the per-element-divide and precomputed-reciprocal
stable softmax kernels perform the same writes (no scratch regions). -/
theorem softmax_reciprocal_refinement_view
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    ComputeRefine.Refines
      (stableSoftmaxKernel xReg yReg N)
      (softmaxRecipKernel xReg yReg N) s [] := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  apply ComputeKernel.computeRefine_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r hr o
  simp [exec, stableSoftmaxKernel, stepStmts, stepStmt, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div] at hL
  simp [exec, softmaxRecipKernel, stepStmts, stepStmt, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div] at hR
  repeat unfold evalOp at hL
  repeat unfold evalOp at hR
  simp [Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hL
  simp [Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hR
  subst lhs'
  subst rhs'
  refine BlockState.foldl_writeMem_mem_congr _ _ _ _ ?_ r o _ _ rfl
  intro k _
  -- Per-lane value equality: `e / S = e * S⁻¹` (simp normalized `1 / S` to
  -- `S⁻¹`), the reciprocal rewrite `div_eq_mul_inv_real` in inverse form.
  exact div_eq_mul_inv _ _

end VeriTile.Examples
