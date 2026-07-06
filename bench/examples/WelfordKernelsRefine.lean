import VeriTile.Examples.WelfordKernels

/-!
Refinement theorem (`ComputeRefine` pair surface) for the two-pass vs online
Welford kernel pair. The kernels, specs, correctness lemmas, and the
exec-level refinement live in `VeriTile.Examples.WelfordKernels`.
-/

namespace VeriTile.Examples

open VeriTile.Triton

/-- Compute-facing view-level surface for `welford_kernels_refinement`. -/
theorem welford_kernels_refinement_view
    (xReg meanReg varReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1))
    (h_mv : meanReg ≠ varReg) :
    ComputeRefine.Realizes
      (lhs := twopassWelfordKernel xReg meanReg varReg blockSize)
      (rhs := onlineWelfordKernel xReg meanReg varReg blockSize)
      (initialState := s)
      (lhsWrite := fun i : Fin 2 =>
        some (if i.val = 0 then (meanReg, 0) else (varReg, 0)))
      (rhsWrite := fun i : Fin 2 =>
        some (if i.val = 0 then (meanReg, 0) else (varReg, 0)))
      (relation := fun (_ : Fin 2) (lhs rhs : ℝ) => lhs = rhs) := by
  apply ComputeKernel.computeRefine_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  have hview := welford_kernels_refinement_exec_view xReg meanReg varReg blockSize
    hN s xs h_x h_mv
  rw [hL, hR] at hview
  intro i
  fin_cases i
  · simpa [TensorView.observe, observeTileAt, scalarCellView, TensorView.offset,
      Offset.strided] using hview.1
  · simpa [TensorView.observe, observeTileAt, scalarCellView, TensorView.offset,
      Offset.strided] using hview.2

end VeriTile.Examples
