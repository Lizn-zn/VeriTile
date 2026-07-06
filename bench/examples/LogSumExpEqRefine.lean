import VeriTile.Examples.LogSumExpEq

/-!
Refinement theorem (`ComputeRefine` pair surface) for the direct vs
shift-trick log-sum-exp kernel pair. The kernels, specs, correctness lemmas,
and the exec-level refinement live in `VeriTile.Examples.LogSumExpEq`.
-/

namespace VeriTile.Examples

open VeriTile.Triton VeriTile.Triton.TiledLogSumExp VeriTile.Triton.TiledSoftmax

/-- Compute-facing view-level surface for `log_sum_exp_refinement`. -/
theorem log_sum_exp_refinement_view
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    ComputeRefine.Realizes
      (lhs := directLSEKernel xReg yReg N)
      (rhs := stableLSEKernel xReg yReg N)
      (initialState := s)
      (lhsWrite := ComputeCorrect.WriteMap.scalar yReg s.pid)
      (rhsWrite := ComputeCorrect.WriteMap.scalar yReg s.pid)
      (relation := fun (_ : PUnit) (lhs rhs : ℝ) => lhs = rhs) := by
  apply ComputeKernel.computeRefine_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  have hview := log_sum_exp_refinement_exec_view xReg yReg N hN s xs h_x
  rw [hL, hR] at hview
  intro _
  simpa [TensorView.observe, observeTileAt, scalarCellView, TensorView.offset,
    Offset.strided] using hview

end VeriTile.Examples
