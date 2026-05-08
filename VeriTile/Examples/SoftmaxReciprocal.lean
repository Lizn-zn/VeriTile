/-
VeriTile.Examples.SoftmaxReciprocal

Real Triton optimization: replace `y = e/s` (per-element division) with
`inv_s = 1/s; y = e * inv_s` (one division total + one multiply per element).
Algorithmically equivalent in ℝ since e/s = e * (1/s) when s ≠ 0.

The "div" side of the comparison is `stableSoftmaxKernel` (already defined
in `SoftmaxEq.lean` and proven correct via `softmax_stable_correct`).
This file adds only the reciprocal-form kernel and the equivalence theorem.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Softmax
import VeriTile.Examples.Common
import VeriTile.Examples.SoftmaxEq

namespace VeriTile.Examples

open VeriTile.Triton VeriTile.Triton.TiledSoftmax

/-- Stable softmax with precomputed reciprocal. Saves N-1 divisions vs
    the per-element-divide form (`stableSoftmaxKernel`). -/
def softmaxRecipKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x      := tl.load($(xReg) + offs)
  m      := tl.max(x, axis=0)
  e      := tl.exp(x - m)
  s      := tl.sum(e, axis=0)
  inv_s  := 1 / s
  y      := e * inv_s
  tl.store($(yReg) + offs, y)
}

/-- The load-bearing math identity: division equals multiplication by
    reciprocal (for non-zero divisor). -/
theorem div_eq_mul_inv_real (a s : ℝ) (hs : s ≠ 0) : a / s = a * (1 / s) := by
  field_simp

/-- Closed-form spec for `softmaxRecipKernel`'s `Y[pid*N+i]` cell. -/
noncomputable def stableRecipSpec {N : Nat} (xs : Fin N → ℝ) (m : ℝ) (i : Fin N) : ℝ :=
  Real.exp (xs i - m) * (1 / ∑ j, Real.exp (xs j - m))

theorem softmax_recip_correct
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs) :
    ∀ i : Fin N,
      observeAt (exec (softmaxRecipKernel xReg yReg N) s) yReg N s.pid i
        = some (stableRecipSpec xs (tileMax hN xs) i) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [n + 1] => s.pid * (n + 1) + idx.1.val) :=
    injective_offset_singleton (s.pid * (n + 1))
  simp [observeAt, exec, softmaxRecipKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div, stableRecipSpec, tileMax]
  unfold InputLoadedAt at _h_x
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_x]
  rfl

/-- **Refinement: stable softmax with per-element division ≡ stable softmax
    with precomputed reciprocal.** Composes via `div_eq_mul_inv_real`. -/
theorem softmax_reciprocal_refinement
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoadedAt s xReg N xs) :
    ∀ i : Fin N,
      observeAt (exec (stableSoftmaxKernel xReg yReg N) s) yReg N s.pid i =
      observeAt (exec (softmaxRecipKernel  xReg yReg N) s) yReg N s.pid i := by
  intro i
  rw [softmax_stable_correct xReg yReg N hN s xs h_x i,
      softmax_recip_correct  xReg yReg N hN s xs h_x i]
  congr 1
  -- Goal: stableSpec xs (tileMax hN xs) i = stableRecipSpec xs (tileMax hN xs) i
  -- These differ only by a / b vs a * (1/b).
  unfold stableSpec stableRecipSpec
  have h_sum_pos : 0 < ∑ j, Real.exp (xs j - tileMax hN xs) := by
    apply Finset.sum_pos
    · intro j _; exact Real.exp_pos _
    · obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
      exact ⟨⟨0, hN⟩, Finset.mem_univ _⟩
  exact div_eq_mul_inv_real _ _ (ne_of_gt h_sum_pos)

/-- View-level surface for `softmax_reciprocal_refinement`. -/
theorem softmax_reciprocal_refinement_exec_view
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    ∀ idx : TileIndex [N],
      TensorView.observe (exec (stableSoftmaxKernel xReg yReg N) s)
          (programTileView s yReg N) idx =
      TensorView.observe (exec (softmaxRecipKernel  xReg yReg N) s)
          (programTileView s yReg N) idx := by
  intro idx
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := N) (xs := xs) h_x
  simpa [TensorView.observe, observeTileAt, programTileView,
         TensorView.offset, Offset.strided, observeAt]
    using softmax_reciprocal_refinement xReg yReg N hN s xs hx idx.1

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
