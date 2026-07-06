import VeriTile.Examples.SoftmaxEq

/-!
Refinement theorem (`ComputeRefine` pair surface) for the naive vs stable
softmax kernel pair. The kernels, specs, correctness lemmas, and the
exec-level refinement live in `VeriTile.Examples.SoftmaxEq`.
-/

namespace VeriTile.Examples

open VeriTile.Triton VeriTile.Triton.TiledSoftmax

/-- Compute-facing view-level surface for `softmax_kernels_refinement`. -/
theorem softmax_kernels_refinement_view
    (xReg yReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1)) :
    ComputeRefine.Realizes
      (lhs := naiveSoftmaxKernel xReg yReg blockSize)
      (rhs := stableSoftmaxKernel xReg yReg blockSize)
      (initialState := s)
      (lhsWrite := ComputeCorrect.WriteMap.ofTensorView
        (programTileView s yReg blockSize))
      (rhsWrite := ComputeCorrect.WriteMap.ofTensorView
        (programTileView s yReg blockSize))
      (relation := fun (_ : TileIndex [blockSize]) (lhs rhs : ℝ) => lhs = rhs) := by
  apply ComputeKernel.computeRefine_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro idx
  have hview := softmax_kernels_refinement_exec_view xReg yReg blockSize hN s xs h_x idx
  rw [hL, hR] at hview
  simpa [ComputeCorrect.WriteMap.ofTensorView, TensorView.observe,
    observeTileAt] using hview

end VeriTile.Examples
