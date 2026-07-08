import VeriTile.Examples.LogSumExpEq

/-!
Refinement theorem (`ComputeRefine.Refines`, writes-equality) for the direct
vs shift-trick log-sum-exp kernel pair. The kernels, specs, correctness
lemmas, and the exec-level refinement live in `VeriTile.Examples.LogSumExpEq`.

Headline: `log_sum_exp_refinement_view`. For the rounding-model (∀R) variant of the
writes-equality surface see `bench/examples/Swiglu.lean`.
-/

namespace VeriTile.Examples

open VeriTile.Triton VeriTile.Triton.TiledLogSumExp VeriTile.Triton.TiledSoftmax

/-- Writes-equality refinement surface for `log_sum_exp_refinement`: from the
same initial state, the direct and shift-trick LSE kernels perform the same
writes (no scratch regions). -/
theorem log_sum_exp_refinement_view
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    ComputeRefine.Refines
      (directLSEKernel xReg yReg N)
      (stableLSEKernel xReg yReg N) s [] := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  apply ComputeKernel.computeRefine_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r hr o
  simp [exec, directLSEKernel, stepStmts, stepStmt, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul] at hL
  simp [exec, stableLSEKernel, stepStmts, stepStmt, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul, NumericDType.sub] at hR
  repeat unfold evalOp at hL
  repeat unfold evalOp at hR
  simp [Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hL
  simp [Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hR
  subst lhs'
  subst rhs'
  -- Both sides are a single scalar write `writeMem yReg s.pid v` on top of a
  -- register-only (mem-preserving) setReg chain.
  rw [BlockState.writeMem_mem, BlockState.writeMem_mem]
  by_cases hc : r = yReg ∧ o = s.pids 0
  · rw [if_pos hc, if_pos hc]
    -- Written-value equality: the shift-invariance of log-sum-exp.
    exact congrArg MemCell.real (log_sum_exp_shift_invariant hN _ _)
  · rw [if_neg hc, if_neg hc]
    rfl

end VeriTile.Examples
