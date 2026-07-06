import VeriTile.Examples.SoftmaxReciprocal

/-!
Refinement theorem (`ComputeRefine` pair surface) for the reciprocal-softmax
rewrite. The kernels, specs, correctness lemmas, and the exec-level
refinement live in `VeriTile.Examples.SoftmaxReciprocal`.
-/

namespace VeriTile.Examples

open VeriTile.Triton VeriTile.Triton.TiledSoftmax

/-- Compute-facing view-level refinement surface for the reciprocal softmax
rewrite. -/
theorem softmax_reciprocal_refinement_view
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    ComputeRefine.Realizes
      (lhs := stableSoftmaxKernel xReg yReg N)
      (rhs := softmaxRecipKernel xReg yReg N)
      (initialState := s)
      (lhsWrite := ComputeCorrect.WriteMap.ofTensorView (programTileView s yReg N))
      (rhsWrite := ComputeCorrect.WriteMap.ofTensorView (programTileView s yReg N))
      (relation := fun (_ : TileIndex [N]) (lhs rhs : ℝ) => lhs = rhs) := by
  apply ComputeKernel.computeRefine_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro idx
  have hview := softmax_reciprocal_refinement_exec_view xReg yReg N hN s xs h_x idx
  rw [hL, hR] at hview
  simpa [ComputeCorrect.WriteMap.ofTensorView, TensorView.observe,
    observeTileAt] using hview

end VeriTile.Examples
