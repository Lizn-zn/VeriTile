/-
VeriTile.Examples.FlashAttention1.Backward

FA-1 backward math surface.

This file is intentionally proof-facing and Real-only.  Compute/IEEE behavior
stays on the ComputeCorrect/testing track; the theorem surface here is the
algorithm contract consumed by FA-1 backward kernels.
-/

import VeriTile.Examples.FlashAttention1.Common
import VeriTile.Triton.Concurrency.Atomic

namespace VeriTile.Examples

open VeriTile.Triton
open BigOperators

namespace FA1Backward

/-- Backward outputs for one non-causal FA-1 slice. -/
structure Grads (M S D : Nat) where
  dQ : TileIndex [M, D] → ℝ
  dK : TileIndex [S, D] → ℝ
  dV : TileIndex [S, D] → ℝ

/-- Forward-side log-sum-exp reconstructed from Q/K. -/
noncomputable def lseReal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (scale : ℝ) (i : Fin M) : ℝ :=
  Real.log (Finset.univ.sum fun j : Fin S =>
    Real.exp (FA1Math.scaledScore Q K scale i j))

/-- The value computed by the online-softmax registers at forward readout:
`m_i + log(l_i)`. -/
noncomputable def streamingLSE {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) (i : Fin M) : ℝ :=
  (FA1Math.mPartial Bk Q numKVBlocks K scale numKVBlocks i).unbotD 0 +
    Real.log (FA1Math.lPartial Q numKVBlocks K scale numKVBlocks i)

/-- The forward LSE store is the usual unshifted log-sum-exp.

This is the math bridge needed by a forward kernel that stores
`m_i + tl.log(l_i)` for backward. -/
theorem streamingLSE_eq_lseReal {M D Bk : Nat} (hBk : 0 < Bk)
    (hN : 0 < numKVBlocks) (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) (i : Fin M) :
    streamingLSE Q numKVBlocks K scale i = lseReal Q K scale i := by
  unfold streamingLSE lseReal
  let m : ℝ := (FA1Math.mPartial Bk Q numKVBlocks K scale numKVBlocks i).unbotD 0
  have hlf_pos : 0 < FA1Math.lFree Q K scale numKVBlocks (le_refl numKVBlocks) i :=
    FA1Math.lFree_final_pos hBk hN Q K scale i
  have hlf_ne :
      FA1Math.lFree Q K scale numKVBlocks (le_refl numKVBlocks) i ≠ 0 :=
    ne_of_gt hlf_pos
  rw [FA1Math.lPartial_eq_mShifted hBk Q numKVBlocks K scale
      numKVBlocks (le_refl _) i]
  rw [Real.log_mul (Real.exp_ne_zero _) hlf_ne]
  rw [Real.log_exp]
  rw [FA1Math.lFree_eq_flat]
  ring

/-- Softmax probabilities reconstructed from a stored LSE row. -/
noncomputable def probability {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (LSE : Fin M → ℝ) (scale : ℝ) (i : Fin M) (j : Fin S) : ℝ :=
  Real.exp (FA1Math.scaledScore Q K scale i j - LSE i)

/-- `dP = dO · Vᵀ`. -/
noncomputable def dP {M S D : Nat}
    (V : TileIndex [S, D] → ℝ) (dO : TileIndex [M, D] → ℝ)
    (i : Fin M) (j : Fin S) : ℝ :=
  Finset.univ.sum fun d : Fin D =>
    dO (i, d, PUnit.unit) * V (j, d, PUnit.unit)

/-- Row correction `D_i = Σ_j P_ij · dP_ij`. -/
noncomputable def rowCorrection {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (V : TileIndex [S, D] → ℝ) (dO : TileIndex [M, D] → ℝ)
    (LSE : Fin M → ℝ) (scale : ℝ) (i : Fin M) : ℝ :=
  Finset.univ.sum fun j : Fin S =>
    probability Q K LSE scale i j * dP V dO i j

/-- Softmax JVP in the form used by FA backward:
`dS_ij = P_ij · (dP_ij - D_i)`. -/
noncomputable def dS {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (V : TileIndex [S, D] → ℝ) (dO : TileIndex [M, D] → ℝ)
    (LSE : Fin M → ℝ) (scale : ℝ) (i : Fin M) (j : Fin S) : ℝ :=
  probability Q K LSE scale i j *
    (dP V dO i j - rowCorrection Q K V dO LSE scale i)

/-- Closed-form reverse-mode FA-1 backward over Real tensors. -/
noncomputable def attentionBackwardReal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ) :
    Grads M S D :=
  { dQ := fun (i, d, _) =>
      scale * Finset.univ.sum fun j : Fin S =>
        dS Q K V dO LSE scale i j * K (j, d, PUnit.unit)
    dK := fun (j, d, _) =>
      scale * Finset.univ.sum fun i : Fin M =>
        dS Q K V dO LSE scale i j * Q (i, d, PUnit.unit)
    dV := fun (j, d, _) =>
      Finset.univ.sum fun i : Fin M =>
        probability Q K LSE scale i j * dO (i, d, PUnit.unit) }

/-- Named reverse-mode spec.  Keeping this as a separate name makes the public
theorem surface state the intended equivalence explicitly while leaving the
closed form above as the executable definition. -/
noncomputable def reverseModeAttentionReal := @attentionBackwardReal

theorem attentionBackwardReal_eq_reverseMode {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ) :
    attentionBackwardReal Q K V dO LSE scale =
      reverseModeAttentionReal Q K V dO LSE scale := rfl

@[simp] theorem attentionBackwardReal_dQ {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    (attentionBackwardReal Q K V dO LSE scale).dQ idx =
      scale * Finset.univ.sum (fun j : Fin S =>
        dS Q K V dO LSE scale idx.1 j * K (j, idx.2.1, PUnit.unit)) := by
  cases idx
  rfl

@[simp] theorem attentionBackwardReal_dK {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [S, D]) :
    (attentionBackwardReal Q K V dO LSE scale).dK idx =
      scale * Finset.univ.sum (fun i : Fin M =>
        dS Q K V dO LSE scale i idx.1 * Q (i, idx.2.1, PUnit.unit)) := by
  cases idx
  rfl

@[simp] theorem attentionBackwardReal_dV {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [S, D]) :
    (attentionBackwardReal Q K V dO LSE scale).dV idx =
      Finset.univ.sum (fun i : Fin M =>
        probability Q K LSE scale i idx.1 * dO (i, idx.2.1, PUnit.unit)) := by
  cases idx
  rfl

/-- Contribution to `dQ` from one KV block in the block-partitioned atomic
backward kernel.  Summing this over all blocks recovers the closed-form
`attentionBackwardReal.dQ`. -/
noncomputable def dQBlockContribution {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [M, D]) : ℝ :=
  scale * Finset.univ.sum fun jLocal : Fin Bk =>
    let j : Fin (Bk * numKVBlocks) :=
      FA1Math.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) jLocal
    dS Q K V dO LSE scale idx.1 j * K (j, idx.2.1, PUnit.unit)

@[simp] theorem dQBlockContribution_eq {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [M, D]) :
    dQBlockContribution Q K V dO LSE scale block idx =
      scale * Finset.univ.sum (fun jLocal : Fin Bk =>
        let j : Fin (Bk * numKVBlocks) :=
          FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) jLocal
        dS Q K V dO LSE scale idx.1 j * K (j, idx.2.1, PUnit.unit)) := rfl

theorem dQBlockContribution_sum_eq_attentionBackwardReal {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    (Finset.univ.sum fun block : Fin numKVBlocks =>
      dQBlockContribution Q K V dO LSE scale block idx) =
      (attentionBackwardReal Q K V dO LSE scale).dQ idx := by
  simp only [dQBlockContribution, attentionBackwardReal_dQ]
  rw [← Finset.mul_sum]
  congr 1
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (FA1Math.blockIndexEquiv Bk numKVBlocks) ?_ ?_).symm
  · intro _; simp
  · intro j _
    rw [FA1Math.blockIndex_blockIndexEquiv]

theorem exp_sum_mul_scale_eq {ι : Type} [Fintype ι]
    (f : ι → ℝ) (scale lse : ℝ) :
    Real.exp (((Finset.univ.sum f) * scale) - lse) =
      Real.exp (scale * (Finset.univ.sum f) - lse) := by
  congr 1
  ring

/-- Tile-level bridge for one block-local `dQ_part = dS_block · K_block · scale`. -/
theorem dQ_block_tile_some_eq_dQBlockContribution {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [M, D]) :
    Option.map (fun a : ℝ => a * scale)
      ((Tile.dot []
        (Tile.ofReal fun idx : TileIndex [M, Bk] =>
          let j : Fin (Bk * numKVBlocks) :=
            FA1Math.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) idx.2.1
          dS Q K V dO LSE scale idx.1 j)
        (Tile.ofReal fun idx : TileIndex [Bk, D] =>
          let j : Fin (Bk * numKVBlocks) :=
            FA1Math.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) idx.1
          K (j, idx.2.1, PUnit.unit))).data idx) =
      some (dQBlockContribution Q K V dO LSE scale block idx) := by
  rw [Tile.dot_nil_data]
  simp [Tile.ofReal, dQBlockContribution]
  ring

/-- Tile-level bridge for one block-local `dK_block = dS_blockᵀ · Q · scale`. -/
theorem dK_block_tile_some_eq_attentionBackwardReal {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [Bk, D]) :
    Option.map (fun a : ℝ => a * scale)
      ((Tile.dot []
        (Tile.transpose []
          (Tile.ofReal fun idx : TileIndex [M, Bk] =>
            let j : Fin (Bk * numKVBlocks) :=
              FA1Math.blockIndex Bk numKVBlocks block.val
                (by have := block.isLt; omega) idx.2.1
            dS Q K V dO LSE scale idx.1 j))
        (Tile.ofReal Q)).data idx) =
      some ((attentionBackwardReal Q K V dO LSE scale).dK
        (FA1Math.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) := by
  rw [Tile.dot_nil_data]
  simp [Tile.transpose, Tile.ofReal, attentionBackwardReal_dK]
  ring

/-- Tile-level bridge for one block-local `dV_block = P_blockᵀ · dO`. -/
theorem dV_block_tile_some_eq_attentionBackwardReal {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [Bk, D]) :
    (Tile.dot []
      (Tile.transpose []
        (Tile.ofReal fun idx : TileIndex [M, Bk] =>
          let j : Fin (Bk * numKVBlocks) :=
            FA1Math.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) idx.2.1
          probability Q K LSE scale idx.1 j))
      (Tile.ofReal dO)).data idx =
      some ((attentionBackwardReal Q K V dO LSE scale).dV
        (FA1Math.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) := by
  rw [Tile.dot_nil_data]
  simp [Tile.transpose, Tile.ofReal, attentionBackwardReal_dV]

/-- Tile-level bridge for the stripped kernel's `dV = Pᵀ · dO`
register computation. -/
theorem dV_tile_eq_attentionBackwardReal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [S, D]) :
    ((Tile.dot []
        (Tile.transpose []
          (Tile.ofReal fun idx : TileIndex [M, S] =>
            probability Q K LSE scale idx.1 idx.2.1))
        (Tile.ofReal dO)).data idx).unbotD 0 =
      (attentionBackwardReal Q K V dO LSE scale).dV idx := by
  rw [Tile.dot_nil_data]
  simp [Tile.transpose, Tile.ofReal, attentionBackwardReal_dV]

/-- Tile-level bridge for the stripped kernel's `dQ = dS · K · scale`
register computation. -/
theorem dQ_tile_eq_attentionBackwardReal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    Option.getD
      (Option.map (fun a : ℝ => a * scale)
        ((Tile.dot []
          (Tile.ofReal fun idx : TileIndex [M, S] =>
            dS Q K V dO LSE scale idx.1 idx.2.1)
          (Tile.ofReal K)).data idx)) 0 =
      (attentionBackwardReal Q K V dO LSE scale).dQ idx := by
  rw [Tile.dot_nil_data]
  simp [Tile.ofReal, attentionBackwardReal_dQ]
  ring

theorem dQ_tile_some_eq_attentionBackwardReal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    Option.map (fun a : ℝ => a * scale)
      ((Tile.dot []
        (Tile.ofReal fun idx : TileIndex [M, S] =>
          dS Q K V dO LSE scale idx.1 idx.2.1)
        (Tile.ofReal K)).data idx) =
      some ((attentionBackwardReal Q K V dO LSE scale).dQ idx) := by
  rw [Tile.dot_nil_data]
  simp [Tile.ofReal, attentionBackwardReal_dQ]
  ring

/-- Tile-level bridge for the stripped kernel's `dK = dSᵀ · Q · scale`
register computation. -/
theorem dK_tile_eq_attentionBackwardReal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [S, D]) :
    Option.getD
      (Option.map (fun a : ℝ => a * scale)
        ((Tile.dot []
          (Tile.transpose []
            (Tile.ofReal fun idx : TileIndex [M, S] =>
              dS Q K V dO LSE scale idx.1 idx.2.1))
          (Tile.ofReal Q)).data idx)) 0 =
      (attentionBackwardReal Q K V dO LSE scale).dK idx := by
  rw [Tile.dot_nil_data]
  simp [Tile.transpose, Tile.ofReal, attentionBackwardReal_dK]
  ring

theorem dK_tile_some_eq_attentionBackwardReal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [S, D]) :
    Option.map (fun a : ℝ => a * scale)
      ((Tile.dot []
        (Tile.transpose []
          (Tile.ofReal fun idx : TileIndex [M, S] =>
            dS Q K V dO LSE scale idx.1 idx.2.1))
        (Tile.ofReal Q)).data idx) =
      some ((attentionBackwardReal Q K V dO LSE scale).dK idx) := by
  rw [Tile.dot_nil_data]
  simp [Tile.transpose, Tile.ofReal, attentionBackwardReal_dK]
  ring

theorem dV_tile_some_eq_attentionBackwardReal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [S, D]) :
    (Tile.dot []
      (Tile.transpose []
        (Tile.ofReal fun idx : TileIndex [M, S] =>
          probability Q K LSE scale idx.1 idx.2.1))
      (Tile.ofReal dO)).data idx =
      some ((attentionBackwardReal Q K V dO LSE scale).dV idx) := by
  rw [Tile.dot_nil_data]
  simp [Tile.transpose, Tile.ofReal, attentionBackwardReal_dV]

/-- Tile-level bridge for the stripped kernel's probability tile
`P = exp(scores - LSE[:, None])`. -/
theorem probability_tile_eq {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (LSE : Fin M → ℝ) (scale : ℝ) (idx : TileIndex [M, S]) :
    WithBot.realExp
      (Option.map₂ (fun x1 x2 : ℝ => x1 - x2)
        (Option.map (fun a : ℝ => a * scale)
          ((Tile.dot [] (Tile.ofReal Q) (Tile.transpose [] (Tile.ofReal K))).data idx))
        ((Tile.ofReal fun idx : TileIndex [M] => LSE idx.1).data (idx.1, PUnit.unit))) =
      some (probability Q K LSE scale idx.1 idx.2.1) := by
  change WithBot.realExp
      (Option.map₂ (fun x1 x2 : ℝ => x1 - x2)
        (Option.map (fun a : ℝ => a * scale)
          ((Tile.dot [] (Tile.ofReal Q) (Tile.transpose [] (Tile.ofReal K))).data
            (idx.1, idx.2.1, PUnit.unit)))
        ((Tile.ofReal fun idx : TileIndex [M] => LSE idx.1).data (idx.1, PUnit.unit))) = _
  rw [FA1Math.scaled_data_eq' Q K scale idx.1 idx.2.1]
  simp [Tile.ofReal, probability]

/-- Tile-level bridge for `dP = dO · Vᵀ`. -/
theorem dP_tile_eq {M S D : Nat}
    (V : TileIndex [S, D] → ℝ) (dO : TileIndex [M, D] → ℝ)
    (idx : TileIndex [M, S]) :
    ((Tile.dot [] (Tile.ofReal dO) (Tile.transpose [] (Tile.ofReal V))).data idx).unbotD 0 =
      dP V dO idx.1 idx.2.1 := by
  rw [Tile.dot_nil_data]
  simp [Tile.transpose, Tile.ofReal, dP]

/-- Tile-level bridge for `corr = sum(P * dP, axis=1)`. -/
theorem rowCorrection_tile_eq {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (V : TileIndex [S, D] → ℝ) (dO : TileIndex [M, D] → ℝ)
    (LSE : Fin M → ℝ) (scale : ℝ) (idx : TileIndex [M]) :
    (Tile.reduceSumDrop (shape := [M, S]) 1
      (Tile.ofReal fun idx : TileIndex [M, S] =>
        probability Q K LSE scale idx.1 idx.2.1 * dP V dO idx.1 idx.2.1)).data idx =
      some (rowCorrection Q K V dO LSE scale idx.1) := by
  simp [Tile.reduceSumDrop, Tile.ofReal, rowCorrection]
  congr 1

/-- Tile-level bridge for `dS = P * (dP - corr[:, None])`. -/
theorem dS_tile_eq {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (V : TileIndex [S, D] → ℝ) (dO : TileIndex [M, D] → ℝ)
    (LSE : Fin M → ℝ) (scale : ℝ) (idx : TileIndex [M, S]) :
    Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
      ((Tile.ofReal fun idx : TileIndex [M, S] =>
        probability Q K LSE scale idx.1 idx.2.1).data idx)
      (Option.map₂ (fun x1 x2 : ℝ => x1 - x2)
        ((Tile.ofReal fun idx : TileIndex [M, S] => dP V dO idx.1 idx.2.1).data idx)
        ((Tile.ofReal fun idx : TileIndex [M] =>
          rowCorrection Q K V dO LSE scale idx.1).data (idx.1, PUnit.unit))) =
      some (dS Q K V dO LSE scale idx.1 idx.2.1) := by
  simp [Tile.ofReal, dS]

/-- Bundled theorem surface for the stripped backward pure-tile computation:
the three gradient tiles computed from the Real intermediates are exactly the
three components of `attentionBackwardReal`. -/
theorem strippedBackward_tile_bridges_complete {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ) :
    (∀ idx : TileIndex [M, D],
      Option.getD
        (Option.map (fun a : ℝ => a * scale)
          ((Tile.dot []
            (Tile.ofReal fun idx : TileIndex [M, S] =>
              dS Q K V dO LSE scale idx.1 idx.2.1)
            (Tile.ofReal K)).data idx)) 0 =
        (attentionBackwardReal Q K V dO LSE scale).dQ idx) ∧
    (∀ idx : TileIndex [S, D],
      Option.getD
        (Option.map (fun a : ℝ => a * scale)
          ((Tile.dot []
            (Tile.transpose []
              (Tile.ofReal fun idx : TileIndex [M, S] =>
                dS Q K V dO LSE scale idx.1 idx.2.1))
            (Tile.ofReal Q)).data idx)) 0 =
        (attentionBackwardReal Q K V dO LSE scale).dK idx) ∧
    (∀ idx : TileIndex [S, D],
      ((Tile.dot []
          (Tile.transpose []
            (Tile.ofReal fun idx : TileIndex [M, S] =>
              probability Q K LSE scale idx.1 idx.2.1))
          (Tile.ofReal dO)).data idx).unbotD 0 =
        (attentionBackwardReal Q K V dO LSE scale).dV idx) := by
  exact ⟨
    dQ_tile_eq_attentionBackwardReal Q K V dO LSE scale,
    dK_tile_eq_attentionBackwardReal Q K V dO LSE scale,
    dV_tile_eq_attentionBackwardReal Q K V dO LSE scale⟩

@[simp] theorem dropInsertedIndex_single_axis1 {n : Nat} (i : Fin n) :
    TileShape.dropInsertedIndex [n] ⟨1, by simp⟩ 1 (i, (0 : Fin 1), PUnit.unit) =
      (i, PUnit.unit) := rfl

@[simp] theorem dropInsertedIndex_single_axis1_ofNat {n : Nat} (i : Fin n) :
    TileShape.dropInsertedIndex [n] (1 : Fin (([n] : TileShape).length + 1))
        1 (i, (0 : Fin 1), PUnit.unit) =
      (i, PUnit.unit) := rfl

@[simp] theorem dropInsertedIndex_single_axis1_any {n : Nat}
    (h : 1 < ([n] : TileShape).length + 1) (i : Fin n) :
    TileShape.dropInsertedIndex [n] ⟨1, h⟩ 1 (i, (0 : Fin 1), PUnit.unit) =
      (i, PUnit.unit) := by
  rfl

@[simp] theorem dropInsertedIndex_single_axis0 {n : Nat} (i : Fin n) :
    TileShape.dropInsertedIndex [n] ⟨0, by simp⟩ 1 ((0 : Fin 1), i, PUnit.unit) =
      (i, PUnit.unit) := rfl

@[simp] theorem dropInsertedIndex_single_axis0_ofNat {n : Nat} (i : Fin n) :
    TileShape.dropInsertedIndex [n] (0 : Fin (([n] : TileShape).length + 1))
        1 ((0 : Fin 1), i, PUnit.unit) =
      (i, PUnit.unit) := rfl

@[simp] theorem dropInsertedIndex_single_axis0_any {n : Nat}
    (h : 0 < ([n] : TileShape).length + 1) (i : Fin n) :
    TileShape.dropInsertedIndex [n] ⟨0, h⟩ 1 ((0 : Fin 1), i, PUnit.unit) =
      (i, PUnit.unit) := by
  rfl

@[simp] theorem insertAxisIndex_two_axis1_drop {m n : Nat} (i : Fin m) (j : Fin n) :
    TileShape.insertAxisIndex [m, n] ⟨1, by simp⟩
        (TileShape.dropInsertedIndex [m] ⟨1, by simp⟩ 1 (i, (0 : Fin 1), PUnit.unit)) j =
      (i, j, PUnit.unit) := rfl

@[simp] theorem insertAxisIndex_two_axis1_drop_ofNat {m n : Nat} (i : Fin m) (j : Fin n) :
    TileShape.insertAxisIndex [m, n] (1 : Fin (([m, n] : TileShape).length))
        (TileShape.dropInsertedIndex [m] (1 : Fin (([m] : TileShape).length + 1))
          1 (i, (0 : Fin 1), PUnit.unit)) j =
      (i, j, PUnit.unit) := rfl

@[simp] theorem insertAxisIndex_two_axis1_drop_any {m n : Nat}
    (h : 1 < ([m, n] : TileShape).length)
    (hdrop : 1 < ([m] : TileShape).length + 1) (i : Fin m) (j : Fin n) :
    TileShape.insertAxisIndex [m, n] ⟨1, h⟩
        (TileShape.dropInsertedIndex [m] ⟨1, hdrop⟩ 1 (i, (0 : Fin 1), PUnit.unit)) j =
      (i, j, PUnit.unit) := by
  rfl

/-- Jacobian-vector product of row softmax after simplifying
`(diag(P) - P·Pᵀ) dP`. -/
noncomputable def softmaxJacobianJVP {S : Nat}
    (P dPRow : Fin S → ℝ) (j : Fin S) : ℝ :=
  P j * (dPRow j - Finset.univ.sum fun k : Fin S => P k * dPRow k)

/-- FA-backward spelling of the same softmax JVP formula. -/
noncomputable def softmaxJVP {S : Nat}
    (P dPRow : Fin S → ℝ) (j : Fin S) : ℝ :=
  P j * (dPRow j - Finset.univ.sum fun k : Fin S => P k * dPRow k)

theorem softmax_jvp_identity {S : Nat}
    (P dPRow : Fin S → ℝ) (j : Fin S) :
    softmaxJacobianJVP P dPRow j = softmaxJVP P dPRow j := rfl

/-! ## Kernel surfaces

The current verified forward kernels remain available under their existing
names.  The LSE-emitting partner below is the forward-side contract needed by
backward: it is the same full-tile FA-1 loop with one additional store of
`m_i + log(l_i)` after the output store.
-/

def fa1ForwardKernelWithLSE
    (qReg kReg vReg outReg lseReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ) : ComputeKernel := triton {
  pid    := tl.program_id(0)

  offs_m := pid * $(M) + tl.arange(0, $(M))
  offs_d := tl.arange(0, $(D))

  q_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]
  q      := tl.load($(qReg) + q_ptrs)

  m_i    := tl.full([$(M)], -inf)
  l_i    := tl.zeros([$(M)])
  o_acc  := tl.zeros([$(M), $(D)])

  tl.for n in $(numKVBlocks) {
    offs_n  := n * $(Bk) + tl.arange(0, $(Bk))
    k_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
    v_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
    k       := tl.load($(kReg) + k_ptrs)
    v       := tl.load($(vReg) + v_ptrs)

    scores  := tl.dot(q, tl.trans(k)) * $ℝ(scale)
    m_block := tl.max(scores, axis = 1)
    m_new   := tl.max(m_i, m_block)
    alpha   := tl.exp(m_i - m_new)
    p       := tl.exp(scores - m_new[:, None])
    l_new   := alpha * l_i + tl.sum(p, axis = 1)
    o_acc   := alpha[:, None] * o_acc + tl.dot(p, v)
    m_i     := m_new
    l_i     := l_new
  }

  out    := o_acc / l_i[:, None]
  lse    := m_i + tl.log(l_i)
  o_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]
  tl.store($(outReg) + o_ptrs, out)
  tl.store($(lseReg) + offs_m, lse)
}

/-- Stripped FA-1 backward kernel: one program computes the full non-causal
Real backward slice without atomics.

This is the no-atomic issue #43 path.  It recomputes `P` from `Q`, `K`, and
stored `LSE`, forms the softmax JVP correction, then writes `dQ`, `dK`, and
`dV`.  Production FA-1 uses a different work partition and atomic `dQ`; that
path should consume the generic atomic/concurrency layer.
-/
def fa1BackwardStrippedKernel
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M S D : Nat) (scale : ℝ) : ComputeKernel := triton {
  offs_m := tl.arange(0, $(M))
  offs_n := tl.arange(0, $(S))
  offs_d := tl.arange(0, $(D))

  q_ptrs  := offs_m[:, None] * $(D) + offs_d[None, :]
  k_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
  v_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
  do_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]

  q   := tl.load($(qReg) + q_ptrs)
  k   := tl.load($(kReg) + k_ptrs)
  v   := tl.load($(vReg) + v_ptrs)
  dO  := tl.load($(dOReg) + do_ptrs)
  lse := tl.load($(lseReg) + offs_m)

  scores := tl.dot(q, tl.trans(k)) * $ℝ(scale)
  p      := tl.exp(scores - lse[:, None])
  dV     := tl.dot(tl.trans(p), dO)
  dP     := tl.dot(dO, tl.trans(v))
  corr   := tl.sum(p * dP, axis = 1)
  dS     := p * (dP - corr[:, None])
  dQ     := tl.dot(dS, k) * $ℝ(scale)
  dK     := tl.dot(tl.trans(dS), q) * $ℝ(scale)

  tl.store($(dQReg) + q_ptrs, dQ)
  tl.store($(dKReg) + k_ptrs, dK)
  tl.store($(dVReg) + v_ptrs, dV)
}

/-- Block-partitioned FA-1 backward kernel with atomic `dQ` accumulation.

Each program owns one KV block of size `Bk`.  It stores that block's `dK` and
`dV` with ordinary stores, while contributing the block-local `dQ` term through
`tl.atomic_add`.  The row correction is recomputed against the full KV range
`Bk * numKVBlocks` so that each block contribution uses the same closed-form
softmax JVP correction as `attentionBackwardReal`.

This is intentionally proof-oriented rather than performance-oriented: the
full-KV recomputation keeps the first atomic correctness surface close to the
already-proved stripped backward math. -/
def fa1BackwardAtomicDQKernel
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ) : ComputeKernel := triton {
  block_n := tl.program_id(0)
  offs_m  := tl.arange(0, $(M))
  offs_b  := block_n * $(Bk) + tl.arange(0, $(Bk))
  offs_n  := tl.arange(0, $(Bk * numKVBlocks))
  offs_d  := tl.arange(0, $(D))

  q_ptrs       := offs_m[:, None] * $(D) + offs_d[None, :]
  do_ptrs      := offs_m[:, None] * $(D) + offs_d[None, :]
  k_block_ptrs := offs_b[:, None] * $(D) + offs_d[None, :]
  v_block_ptrs := offs_b[:, None] * $(D) + offs_d[None, :]
  k_all_ptrs   := offs_n[:, None] * $(D) + offs_d[None, :]
  v_all_ptrs   := offs_n[:, None] * $(D) + offs_d[None, :]

  q       := tl.load($(qReg) + q_ptrs)
  dO      := tl.load($(dOReg) + do_ptrs)
  lse     := tl.load($(lseReg) + offs_m)
  k_block := tl.load($(kReg) + k_block_ptrs)
  v_block := tl.load($(vReg) + v_block_ptrs)
  k_all   := tl.load($(kReg) + k_all_ptrs)
  v_all   := tl.load($(vReg) + v_all_ptrs)

  scores_all := tl.dot(q, tl.trans(k_all)) * $ℝ(scale)
  p_all      := tl.exp(scores_all - lse[:, None])
  dP_all     := tl.dot(dO, tl.trans(v_all))
  corr       := tl.sum(p_all * dP_all, axis = 1)

  scores_block := tl.dot(q, tl.trans(k_block)) * $ℝ(scale)
  p_block      := tl.exp(scores_block - lse[:, None])
  dV_block     := tl.dot(tl.trans(p_block), dO)
  dP_block     := tl.dot(dO, tl.trans(v_block))
  dS_block     := p_block * (dP_block - corr[:, None])
  dQ_part      := tl.dot(dS_block, k_block) * $ℝ(scale)
  dK_block     := tl.dot(tl.trans(dS_block), q) * $ℝ(scale)

  tl.atomic_add($(dQReg) + q_ptrs, dQ_part)
  tl.store($(dKReg) + k_block_ptrs, dK_block)
  tl.store($(dVReg) + v_block_ptrs, dV_block)
}

/-- State immediately before the `tl.atomic_add(dQ, dQ_part)` in the
block-partitioned backward kernel.

This is the stateful trace boundary needed for the full atomic proof: the
atomic payload is not syntactically available from the initial state, because
`dQ_part` is computed by the preceding register program. -/
noncomputable def fa1BackwardAtomicDQPreAtomicState
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ) (s : BlockState) : BlockState :=
  (stepStmts
    ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale).toAlgKernel.body.take 29) s).getD s

/- #97: these FA-1 backward readback simp blocks still need elevated
heartbeats, but the Tile accessor simp pass reduced them from 8M to 2M. -/
set_option maxHeartbeats 2000000 in
set_option linter.unusedSimpArgs false in
theorem fa1BackwardAtomicDQPreAtomic_step
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (s : BlockState)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1)) :
    stepStmts
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.take 29) s =
      some (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        M D Bk numKVBlocks scale s) := by
  simp [InputAt, Offset.rowMajor2D, Offset.strided] at hQ hK hV hdO hLSE
  have hQ0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem qReg (i.val * D + d.val) = Q (i, d, PUnit.unit) :=
    fun i d => hQ i d PUnit.unit
  have hK0 : ∀ (j : Fin (Bk * numKVBlocks)) (d : Fin D),
      s.readMem kReg (j.val * D + d.val) = K (j, d, PUnit.unit) :=
    fun j d => hK j d PUnit.unit
  have hV0 : ∀ (j : Fin (Bk * numKVBlocks)) (d : Fin D),
      s.readMem vReg (j.val * D + d.val) = V (j, d, PUnit.unit) :=
    fun j d => hV j d PUnit.unit
  have hdO0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem dOReg (i.val * D + d.val) = dO (i, d, PUnit.unit) :=
    fun i d => hdO i d PUnit.unit
  have hKBlock : ∀ (jLocal : Fin Bk) (d : Fin D),
      s.readMem kReg ((block.val * Bk + jLocal.val) * D + d.val) =
        K (FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) jLocal, d, PUnit.unit) := by
    intro jLocal d
    simpa [FA1Math.blockIndex] using
      hK0 (FA1Math.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) jLocal) d
  have hVBlock : ∀ (jLocal : Fin Bk) (d : Fin D),
      s.readMem vReg ((block.val * Bk + jLocal.val) * D + d.val) =
        V (FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) jLocal, d, PUnit.unit) := by
    intro jLocal d
    simpa [FA1Math.blockIndex] using
      hV0 (FA1Math.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) jLocal) d
  simp [fa1BackwardAtomicDQPreAtomicState, fa1BackwardAtomicDQKernel,
      stepStmts, stepStmt, evalOp, Option.bind, hPid, hQ0, hK0, hV0, hdO0, hLSE,
      hKBlock, hVBlock, Offset.rowMajor2D, Offset.strided,
      TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
      Tile.ofReal, probability_tile_eq, dP_tile_eq, rowCorrection_tile_eq, dS_tile_eq,
      dQ_block_tile_some_eq_dQBlockContribution, dK_block_tile_some_eq_attentionBackwardReal,
      dV_block_tile_some_eq_attentionBackwardReal,
      WithBot.sum_someTerm_eq_some, exp_sum_mul_scale_eq,
      dS, rowCorrection, dP, probability, FA1Math.scaledScore]

set_option maxHeartbeats 2000000 in
set_option linter.unusedSimpArgs false in
theorem fa1BackwardAtomicDQPreAtomic_qPtrs
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (s : BlockState)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1)) :
    (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale s).regs .nat [M, D] "q_ptrs" =
      some (⟨Offset.rowMajor2D (rows := M) (cols := D) 0 D⟩ : Tile .nat [M, D]) := by
  simp [InputAt, Offset.rowMajor2D, Offset.strided] at hQ hK hV hdO hLSE
  have hQ0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem qReg (i.val * D + d.val) = Q (i, d, PUnit.unit) :=
    fun i d => hQ i d PUnit.unit
  have hK0 : ∀ (j : Fin (Bk * numKVBlocks)) (d : Fin D),
      s.readMem kReg (j.val * D + d.val) = K (j, d, PUnit.unit) :=
    fun j d => hK j d PUnit.unit
  have hV0 : ∀ (j : Fin (Bk * numKVBlocks)) (d : Fin D),
      s.readMem vReg (j.val * D + d.val) = V (j, d, PUnit.unit) :=
    fun j d => hV j d PUnit.unit
  have hdO0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem dOReg (i.val * D + d.val) = dO (i, d, PUnit.unit) :=
    fun i d => hdO i d PUnit.unit
  simp [fa1BackwardAtomicDQPreAtomicState, fa1BackwardAtomicDQKernel,
      stepStmts, stepStmt, evalOp, Option.bind, hPid, hQ0, hK0, hV0, hdO0, hLSE,
      Offset.rowMajor2D, Offset.strided,
      TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
      Tile.ofReal, probability_tile_eq, dP_tile_eq, rowCorrection_tile_eq, dS_tile_eq,
      dQ_block_tile_some_eq_dQBlockContribution, dK_block_tile_some_eq_attentionBackwardReal,
      dV_block_tile_some_eq_attentionBackwardReal,
      WithBot.sum_someTerm_eq_some,
      dS, rowCorrection, dP, probability, FA1Math.scaledScore]

set_option maxHeartbeats 2000000 in
set_option linter.unusedSimpArgs false in
theorem fa1BackwardAtomicDQPreAtomic_kBlockPtrs
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (s : BlockState)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1)) :
    (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale s).regs .nat [Bk, D] "k_block_ptrs" =
      some (⟨Offset.rowMajor2D (rows := Bk) (cols := D) (block.val * Bk * D) D⟩ :
        Tile .nat [Bk, D]) := by
  simp [InputAt, Offset.rowMajor2D, Offset.strided] at hQ hK hV hdO hLSE
  have hQ0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem qReg (i.val * D + d.val) = Q (i, d, PUnit.unit) :=
    fun i d => hQ i d PUnit.unit
  have hK0 : ∀ (j : Fin (Bk * numKVBlocks)) (d : Fin D),
      s.readMem kReg (j.val * D + d.val) = K (j, d, PUnit.unit) :=
    fun j d => hK j d PUnit.unit
  have hV0 : ∀ (j : Fin (Bk * numKVBlocks)) (d : Fin D),
      s.readMem vReg (j.val * D + d.val) = V (j, d, PUnit.unit) :=
    fun j d => hV j d PUnit.unit
  have hdO0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem dOReg (i.val * D + d.val) = dO (i, d, PUnit.unit) :=
    fun i d => hdO i d PUnit.unit
  simp [fa1BackwardAtomicDQPreAtomicState, fa1BackwardAtomicDQKernel,
      stepStmts, stepStmt, evalOp, Option.bind, hPid, hQ0, hK0, hV0, hdO0, hLSE,
      Offset.rowMajor2D, Offset.strided,
      TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
      Tile.ofReal, probability_tile_eq, dP_tile_eq, rowCorrection_tile_eq, dS_tile_eq,
      dQ_block_tile_some_eq_dQBlockContribution, dK_block_tile_some_eq_attentionBackwardReal,
      dV_block_tile_some_eq_attentionBackwardReal,
      WithBot.sum_someTerm_eq_some, exp_sum_mul_scale_eq,
      dS, rowCorrection, dP, probability, FA1Math.scaledScore]
  funext idx
  cases idx with
  | mk j rest =>
      cases rest with
      | mk d tail =>
          cases tail
          simp [Tile.bop, Tile.expandDim, Tile.vec, Tile.scalar,
            Offset.rowMajor2D, Offset.strided]
          ring

set_option maxHeartbeats 2000000 in
set_option linter.unusedSimpArgs false in
theorem fa1BackwardAtomicDQPreAtomic_vBlockPtrs
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (s : BlockState)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1)) :
    (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale s).regs .nat [Bk, D] "v_block_ptrs" =
      some (⟨Offset.rowMajor2D (rows := Bk) (cols := D) (block.val * Bk * D) D⟩ :
        Tile .nat [Bk, D]) := by
  simp [InputAt, Offset.rowMajor2D, Offset.strided] at hQ hK hV hdO hLSE
  have hQ0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem qReg (i.val * D + d.val) = Q (i, d, PUnit.unit) :=
    fun i d => hQ i d PUnit.unit
  have hK0 : ∀ (j : Fin (Bk * numKVBlocks)) (d : Fin D),
      s.readMem kReg (j.val * D + d.val) = K (j, d, PUnit.unit) :=
    fun j d => hK j d PUnit.unit
  have hV0 : ∀ (j : Fin (Bk * numKVBlocks)) (d : Fin D),
      s.readMem vReg (j.val * D + d.val) = V (j, d, PUnit.unit) :=
    fun j d => hV j d PUnit.unit
  have hdO0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem dOReg (i.val * D + d.val) = dO (i, d, PUnit.unit) :=
    fun i d => hdO i d PUnit.unit
  simp [fa1BackwardAtomicDQPreAtomicState, fa1BackwardAtomicDQKernel,
      stepStmts, stepStmt, evalOp, Option.bind, hPid, hQ0, hK0, hV0, hdO0, hLSE,
      Offset.rowMajor2D, Offset.strided,
      TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
      Tile.ofReal, probability_tile_eq, dP_tile_eq, rowCorrection_tile_eq, dS_tile_eq,
      dQ_block_tile_some_eq_dQBlockContribution, dK_block_tile_some_eq_attentionBackwardReal,
      dV_block_tile_some_eq_attentionBackwardReal,
      WithBot.sum_someTerm_eq_some, exp_sum_mul_scale_eq,
      dS, rowCorrection, dP, probability, FA1Math.scaledScore]
  funext idx
  cases idx with
  | mk j rest =>
      cases rest with
      | mk d tail =>
          cases tail
          simp [Tile.bop, Tile.expandDim, Tile.vec, Tile.scalar,
            Offset.rowMajor2D, Offset.strided]
          ring

set_option maxHeartbeats 2000000 in
set_option linter.unusedSimpArgs false in
theorem fa1BackwardAtomicDQPreAtomic_dQPart
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (s : BlockState)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1)) :
    (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale s).regs .real [M, D] "dQ_part" =
      some (Tile.ofReal (dQBlockContribution Q K V dO LSE scale block)) := by
  simp [InputAt, Offset.rowMajor2D, Offset.strided] at hQ hK hV hdO hLSE
  have hQ0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem qReg (i.val * D + d.val) = Q (i, d, PUnit.unit) :=
    fun i d => hQ i d PUnit.unit
  have hK0 : ∀ (j : Fin (Bk * numKVBlocks)) (d : Fin D),
      s.readMem kReg (j.val * D + d.val) = K (j, d, PUnit.unit) :=
    fun j d => hK j d PUnit.unit
  have hV0 : ∀ (j : Fin (Bk * numKVBlocks)) (d : Fin D),
      s.readMem vReg (j.val * D + d.val) = V (j, d, PUnit.unit) :=
    fun j d => hV j d PUnit.unit
  have hdO0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem dOReg (i.val * D + d.val) = dO (i, d, PUnit.unit) :=
    fun i d => hdO i d PUnit.unit
  have hKBlock : ∀ (jLocal : Fin Bk) (d : Fin D),
      s.readMem kReg ((block.val * Bk + jLocal.val) * D + d.val) =
        K (FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) jLocal, d, PUnit.unit) := by
    intro jLocal d
    simpa [FA1Math.blockIndex] using
      hK0 (FA1Math.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) jLocal) d
  have hVBlock : ∀ (jLocal : Fin Bk) (d : Fin D),
      s.readMem vReg ((block.val * Bk + jLocal.val) * D + d.val) =
        V (FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) jLocal, d, PUnit.unit) := by
    intro jLocal d
    simpa [FA1Math.blockIndex] using
      hV0 (FA1Math.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) jLocal) d
  simp [fa1BackwardAtomicDQPreAtomicState, fa1BackwardAtomicDQKernel,
      stepStmts, stepStmt, evalOp, Option.bind, hPid, hQ0, hK0, hV0, hdO0, hLSE,
      hKBlock, hVBlock, Offset.rowMajor2D, Offset.strided,
      TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
      Tile.ofReal, probability_tile_eq, dP_tile_eq, rowCorrection_tile_eq, dS_tile_eq,
      dQ_block_tile_some_eq_dQBlockContribution, dK_block_tile_some_eq_attentionBackwardReal,
      dV_block_tile_some_eq_attentionBackwardReal,
      WithBot.sum_someTerm_eq_some, exp_sum_mul_scale_eq,
      dS, rowCorrection, dP, probability, FA1Math.scaledScore]
  funext i
  congr 1
  rw [mul_comm]
  rfl

set_option maxHeartbeats 2000000 in
set_option linter.unusedSimpArgs false in
theorem fa1BackwardAtomicDQPreAtomic_dKBlock
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (s : BlockState)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1)) :
    (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale s).regs .real [Bk, D] "dK_block" =
      some (Tile.ofReal fun idx : TileIndex [Bk, D] =>
        (attentionBackwardReal Q K V dO LSE scale).dK
          (FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) := by
  simp [InputAt, Offset.rowMajor2D, Offset.strided] at hQ hK hV hdO hLSE
  have hQ0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem qReg (i.val * D + d.val) = Q (i, d, PUnit.unit) :=
    fun i d => hQ i d PUnit.unit
  have hK0 : ∀ (j : Fin (Bk * numKVBlocks)) (d : Fin D),
      s.readMem kReg (j.val * D + d.val) = K (j, d, PUnit.unit) :=
    fun j d => hK j d PUnit.unit
  have hV0 : ∀ (j : Fin (Bk * numKVBlocks)) (d : Fin D),
      s.readMem vReg (j.val * D + d.val) = V (j, d, PUnit.unit) :=
    fun j d => hV j d PUnit.unit
  have hdO0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem dOReg (i.val * D + d.val) = dO (i, d, PUnit.unit) :=
    fun i d => hdO i d PUnit.unit
  have hKBlock : ∀ (jLocal : Fin Bk) (d : Fin D),
      s.readMem kReg ((block.val * Bk + jLocal.val) * D + d.val) =
        K (FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) jLocal, d, PUnit.unit) := by
    intro jLocal d
    simpa [FA1Math.blockIndex] using
      hK0 (FA1Math.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) jLocal) d
  have hVBlock : ∀ (jLocal : Fin Bk) (d : Fin D),
      s.readMem vReg ((block.val * Bk + jLocal.val) * D + d.val) =
        V (FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) jLocal, d, PUnit.unit) := by
    intro jLocal d
    simpa [FA1Math.blockIndex] using
      hV0 (FA1Math.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) jLocal) d
  simp [fa1BackwardAtomicDQPreAtomicState, fa1BackwardAtomicDQKernel,
      stepStmts, stepStmt, evalOp, Option.bind, hPid, hQ0, hK0, hV0, hdO0, hLSE,
      hKBlock, hVBlock, Offset.rowMajor2D, Offset.strided,
      TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
      Tile.ofReal, probability_tile_eq, dP_tile_eq, rowCorrection_tile_eq, dS_tile_eq,
      dQ_block_tile_some_eq_dQBlockContribution, dK_block_tile_some_eq_attentionBackwardReal,
      dV_block_tile_some_eq_attentionBackwardReal,
      WithBot.sum_someTerm_eq_some, exp_sum_mul_scale_eq,
      dS, rowCorrection, dP, probability, FA1Math.scaledScore]
  funext i
  congr 1
  rw [mul_comm]
  rfl

set_option maxHeartbeats 2000000 in
set_option linter.unusedSimpArgs false in
theorem fa1BackwardAtomicDQPreAtomic_dVBlock
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (s : BlockState)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1)) :
    (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale s).regs .real [Bk, D] "dV_block" =
      some (Tile.ofReal fun idx : TileIndex [Bk, D] =>
        (attentionBackwardReal Q K V dO LSE scale).dV
          (FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) := by
  simp [InputAt, Offset.rowMajor2D, Offset.strided] at hQ hK hV hdO hLSE
  have hQ0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem qReg (i.val * D + d.val) = Q (i, d, PUnit.unit) :=
    fun i d => hQ i d PUnit.unit
  have hK0 : ∀ (j : Fin (Bk * numKVBlocks)) (d : Fin D),
      s.readMem kReg (j.val * D + d.val) = K (j, d, PUnit.unit) :=
    fun j d => hK j d PUnit.unit
  have hV0 : ∀ (j : Fin (Bk * numKVBlocks)) (d : Fin D),
      s.readMem vReg (j.val * D + d.val) = V (j, d, PUnit.unit) :=
    fun j d => hV j d PUnit.unit
  have hdO0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem dOReg (i.val * D + d.val) = dO (i, d, PUnit.unit) :=
    fun i d => hdO i d PUnit.unit
  have hKBlock : ∀ (jLocal : Fin Bk) (d : Fin D),
      s.readMem kReg ((block.val * Bk + jLocal.val) * D + d.val) =
        K (FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) jLocal, d, PUnit.unit) := by
    intro jLocal d
    simpa [FA1Math.blockIndex] using
      hK0 (FA1Math.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) jLocal) d
  have hVBlock : ∀ (jLocal : Fin Bk) (d : Fin D),
      s.readMem vReg ((block.val * Bk + jLocal.val) * D + d.val) =
        V (FA1Math.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) jLocal, d, PUnit.unit) := by
    intro jLocal d
    simpa [FA1Math.blockIndex] using
      hV0 (FA1Math.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) jLocal) d
  simp [fa1BackwardAtomicDQPreAtomicState, fa1BackwardAtomicDQKernel,
      stepStmts, stepStmt, evalOp, Option.bind, hPid, hQ0, hK0, hV0, hdO0, hLSE,
      hKBlock, hVBlock, Offset.rowMajor2D, Offset.strided,
      TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
      Tile.ofReal, probability_tile_eq, dP_tile_eq, rowCorrection_tile_eq, dS_tile_eq,
      dQ_block_tile_some_eq_dQBlockContribution, dK_block_tile_some_eq_attentionBackwardReal,
      dV_block_tile_some_eq_attentionBackwardReal,
      WithBot.sum_someTerm_eq_some, exp_sum_mul_scale_eq,
      dS, rowCorrection, dP, probability, FA1Math.scaledScore]

/-- State immediately before the final `dQ`/`dK`/`dV` stores of the stripped
backward kernel.  The first 20 statements are the register-computation prefix;
the remaining three statements are the output stores. -/
noncomputable def fa1BackwardStrippedPreStoreState
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M S D : Nat) (scale : ℝ) (s : BlockState) : BlockState :=
  (stepStmts
    ((fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M S D scale).toAlgKernel.body.take 20) s).getD s

/-- The stripped backward kernel is fully algorithm-projectable: it contains no
compute-only effect such as unsupported bit-level operations or async markers. -/
theorem fa1BackwardStrippedKernel_projectable
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M S D : Nat) (scale : ℝ) :
    ∃ ak, (fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M S D scale).toAlgorithm? = Except.ok ak := by
  simp [fa1BackwardStrippedKernel]

theorem fa1BackwardStrippedKernel_toAlgorithm_eq_toAlgKernel
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M S D : Nat) (scale : ℝ) :
    (fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M S D scale).toAlgorithm? =
      Except.ok (fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        M S D scale).toAlgKernel := by
  simp [fa1BackwardStrippedKernel, ComputeKernel.toAlgKernel]

/-- The block-partitioned atomic-`dQ` backward kernel is algorithm-projectable:
`tl.atomic_add` is part of the AlgKernel surface, unlike async/barrier
effect markers. -/
theorem fa1BackwardAtomicDQKernel_toAlgorithm_eq_toAlgKernel
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ) :
    (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale).toAlgorithm? =
      Except.ok (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        M D Bk numKVBlocks scale).toAlgKernel := by
  simp [fa1BackwardAtomicDQKernel, ComputeKernel.toAlgKernel]

/-- Stateful atomic trace extraction agrees with execution for the full
atomic-`dQ` kernel.  This is the generic bridge needed before proving the
per-program contribution formula consumed by the grid atomic-add theorem. -/
theorem fa1BackwardAtomicDQKernel_statefulTrace_exec
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ)
    (tid : ThreadId) (s final : BlockState) (trace : Trace)
    (hTrace :
      Kernel.AtomicTraceStateful
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel
        tid s trace final) :
    exec
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel
        s =
      some final :=
  Kernel.AtomicTraceStateful_final_eq_exec hTrace

/-- The pre-atomic prefix of the block-partitioned backward kernel emits no
atomic trace events.  It only computes registers; the first trace-producing
statement is the subsequent `tl.atomic_add`. -/
theorem fa1BackwardAtomicDQPreAtomic_trace_empty
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ)
    (tid : ThreadId) (s sPre : BlockState)
    (hPre :
      stepStmts
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.take 29) s =
        some sPre) :
    Kernel.AtomicTraceStatefulList tid
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.take 29) s =
      some ([], sPre) := by
  apply Kernel.AtomicTraceStatefulList_empty_of_stepStmts
  · intro st hmem s0
    simp [fa1BackwardAtomicDQKernel] at hmem
    rcases hmem with hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem |
      hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem |
      hmem | hmem | hmem | hmem | hmem | hmem | hmem | hmem
    all_goals subst st; rfl
  · exact hPre

/-- The post-prefix tail of the block-partitioned backward kernel consists of
the atomic `dQ` contribution followed by ordinary `dK` and `dV` stores. -/
theorem fa1BackwardAtomicDQKernel_drop29
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ) :
    (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale).toAlgKernel.body.drop 29 =
      [
        Stmt.atomicAdd NumericDType.real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ_part") MaskOpt.none,
        Stmt.store .real [Bk, D]
          (MemAccess.region dKReg (Op.ref .nat [Bk, D] "k_block_ptrs"))
          (Op.ref .real [Bk, D] "dK_block") MaskOpt.none,
        Stmt.store .real [Bk, D]
          (MemAccess.region dVReg (Op.ref .nat [Bk, D] "v_block_ptrs"))
          (Op.ref .real [Bk, D] "dV_block") MaskOpt.none
      ] := by
  simp [fa1BackwardAtomicDQKernel]

/-- Recompose the full stateful trace from the no-atomic prefix and the
post-prefix tail. -/
theorem fa1BackwardAtomicDQKernel_statefulTrace_of_tail
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ)
    (tid : ThreadId) (s sPre final : BlockState) (trace : Trace)
    (hPre :
      stepStmts
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.take 29) s =
        some sPre)
    (hTail :
      Kernel.AtomicTraceStatefulList tid
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.drop 29) sPre =
        some (trace, final)) :
    Kernel.AtomicTraceStateful
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel
        tid s trace final := by
  rw [Kernel.AtomicTraceStateful]
  rw [show
      (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        M D Bk numKVBlocks scale).toAlgKernel.body =
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.take 29) ++
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.drop 29) by
      exact (List.take_append_drop 29 _).symm]
  apply Kernel.AtomicTraceStatefulList_append_of_stepPrefix
  · exact fa1BackwardAtomicDQPreAtomic_trace_empty
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale tid s sPre hPre
  · exact hTail

/-- Trace emitted by the atomic `dQ` statement once the pre-atomic registers
are known. -/
theorem fa1BackwardAtomicDQ_atomicTraceEvents_of_regs
    (dQReg : RegionName) (M D : Nat) (tid : ThreadId) (sPre : BlockState)
    (qPtrs : Tile .nat [M, D]) (dQPart : Tile .real [M, D])
    (hPtrs : sPre.regs .nat [M, D] "q_ptrs" = some qPtrs)
    (hVal : sPre.regs .real [M, D] "dQ_part" = some dQPart) :
    Stmt.atomicTraceEvents tid sPre
        (Stmt.atomicAdd NumericDType.real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ_part") MaskOpt.none) =
      some ((TileShape.allIndices [M, D]).filterMap fun i =>
        some (Stmt.atomicTraceEvent tid dQReg (qPtrs.data i) .real (dQPart.data i))) := by
  apply Stmt.atomicTraceEvents_atomicAdd_region_none
    (offsets := qPtrs) (values := dQPart)
  · simp [evalOp, hPtrs]
  · simp [evalOp, hVal]

/-- Specialized trace surface for one block's atomic `dQ` contribution. -/
theorem fa1BackwardAtomicDQ_atomicTraceEvents_blockContribution
    {M D Bk numKVBlocks : Nat}
    (dQReg : RegionName) (tid : ThreadId) (sPre : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks)
    (hPtrs : sPre.regs .nat [M, D] "q_ptrs" =
      some (⟨Offset.rowMajor2D (rows := M) (cols := D) 0 D⟩ : Tile .nat [M, D]))
    (hVal : sPre.regs .real [M, D] "dQ_part" =
      some (Tile.ofReal (dQBlockContribution Q K V dO LSE scale block))) :
    Stmt.atomicTraceEvents tid sPre
        (Stmt.atomicAdd NumericDType.real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ_part") MaskOpt.none) =
      some ((TileShape.allIndices [M, D]).filterMap fun i =>
        some (Stmt.atomicTraceEvent tid dQReg
          (Offset.rowMajor2D (rows := M) (cols := D) 0 D i) .real
          (some (dQBlockContribution Q K V dO LSE scale block i)))) := by
  simpa [Tile.ofReal] using
    fa1BackwardAtomicDQ_atomicTraceEvents_of_regs
      dQReg M D tid sPre
      (⟨Offset.rowMajor2D (rows := M) (cols := D) 0 D⟩ : Tile .nat [M, D])
      (Tile.ofReal (dQBlockContribution Q K V dO LSE scale block))
      hPtrs hVal

/-- Tail trace once the pre-atomic registers are known.  The two ordinary
stores after the atomic statement emit no atomic events, so the tail trace is
exactly the block contribution trace. -/
theorem fa1BackwardAtomicDQKernel_tail_trace_blockContribution
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (scale : ℝ) (tid : ThreadId) (sPre final : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (block : Fin numKVBlocks)
    (hPtrs : sPre.regs .nat [M, D] "q_ptrs" =
      some (⟨Offset.rowMajor2D (rows := M) (cols := D) 0 D⟩ : Tile .nat [M, D]))
    (hVal : sPre.regs .real [M, D] "dQ_part" =
      some (Tile.ofReal (dQBlockContribution Q K V dO LSE scale block)))
    (hTailStep :
      stepStmts
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.drop 29) sPre =
        some final) :
    Kernel.AtomicTraceStatefulList tid
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.drop 29) sPre =
      some
        ((TileShape.allIndices [M, D]).filterMap fun i =>
          some (Stmt.atomicTraceEvent tid dQReg
            (Offset.rowMajor2D (rows := M) (cols := D) 0 D i) .real
            (some (dQBlockContribution Q K V dO LSE scale block i))),
          final) := by
  rw [fa1BackwardAtomicDQKernel_drop29] at hTailStep ⊢
  have hTrace := fa1BackwardAtomicDQ_atomicTraceEvents_blockContribution
    (dQReg := dQReg) (tid := tid) (sPre := sPre)
    (Q := Q) (K := K) (V := V) (dO := dO) (LSE := LSE)
    (scale := scale) (block := block) hPtrs hVal
  conv at hTailStep => lhs; unfold stepStmts
  cases hAtomicStep :
      stepStmt
        (Stmt.atomicAdd NumericDType.real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ_part") MaskOpt.none) sPre with
  | none =>
      simp [hAtomicStep] at hTailStep
  | some sAfterAtomic =>
      simp [hAtomicStep] at hTailStep
      have hStoresTrace :
          Kernel.AtomicTraceStatefulList tid
              [
                Stmt.store .real [Bk, D]
                  (MemAccess.region dKReg (Op.ref .nat [Bk, D] "k_block_ptrs"))
                  (Op.ref .real [Bk, D] "dK_block") MaskOpt.none,
                Stmt.store .real [Bk, D]
                  (MemAccess.region dVReg (Op.ref .nat [Bk, D] "v_block_ptrs"))
                  (Op.ref .real [Bk, D] "dV_block") MaskOpt.none
              ] sAfterAtomic =
            some ([], final) := by
        apply Kernel.AtomicTraceStatefulList_empty_of_stepStmts
        · intro st hmem s0
          simp at hmem
          rcases hmem with hmem | hmem <;> subst st <;> rfl
        · exact hTailStep
      unfold Kernel.AtomicTraceStatefulList
      simp [hTrace, hAtomicStep, hStoresTrace]

/-- Full-kernel stateful trace for one program once the pre-atomic register
facts and tail execution are available. -/
theorem fa1BackwardAtomicDQKernel_statefulTrace_blockContribution
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (scale : ℝ) (tid : ThreadId) (s sPre final : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (block : Fin numKVBlocks)
    (hPre :
      stepStmts
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.take 29) s =
        some sPre)
    (hPtrs : sPre.regs .nat [M, D] "q_ptrs" =
      some (⟨Offset.rowMajor2D (rows := M) (cols := D) 0 D⟩ : Tile .nat [M, D]))
    (hVal : sPre.regs .real [M, D] "dQ_part" =
      some (Tile.ofReal (dQBlockContribution Q K V dO LSE scale block)))
    (hTailStep :
      stepStmts
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.drop 29) sPre =
        some final) :
    Kernel.AtomicTraceStateful
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel
        tid s
        ((TileShape.allIndices [M, D]).filterMap fun i =>
          some (Stmt.atomicTraceEvent tid dQReg
            (Offset.rowMajor2D (rows := M) (cols := D) 0 D i) .real
            (some (dQBlockContribution Q K V dO LSE scale block i))))
        final := by
  apply fa1BackwardAtomicDQKernel_statefulTrace_of_tail
    (qReg := qReg) (kReg := kReg) (vReg := vReg) (dOReg := dOReg)
    (lseReg := lseReg) (dQReg := dQReg) (dKReg := dKReg) (dVReg := dVReg)
    (M := M) (D := D) (Bk := Bk) (numKVBlocks := numKVBlocks)
    (scale := scale) (tid := tid) (sPre := sPre)
  · exact hPre
  · exact fa1BackwardAtomicDQKernel_tail_trace_blockContribution
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      scale tid sPre final Q K V dO LSE block hPtrs hVal hTailStep

/-- Input-level stateful trace theorem for one block-program, modulo the
ordinary tail execution.  The pre-atomic prefix is discharged from the tensor
input assumptions; the remaining tail execution hypothesis is what later
store/readback composition consumes. -/
theorem fa1BackwardAtomicDQKernel_statefulTrace_blockContribution_from_inputs
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (scale : ℝ) (tid : ThreadId) (s final : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (block : Fin numKVBlocks)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1))
    (hTailStep :
      stepStmts
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.drop 29)
        (fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale s) =
        some final) :
    Kernel.AtomicTraceStateful
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel
        tid s
        ((TileShape.allIndices [M, D]).filterMap fun i =>
          some (Stmt.atomicTraceEvent tid dQReg
            (Offset.rowMajor2D (rows := M) (cols := D) 0 D i) .real
            (some (dQBlockContribution Q K V dO LSE scale block i))))
        final := by
  apply fa1BackwardAtomicDQKernel_statefulTrace_blockContribution
    (qReg := qReg) (kReg := kReg) (vReg := vReg) (dOReg := dOReg)
    (lseReg := lseReg) (dQReg := dQReg) (dKReg := dKReg) (dVReg := dVReg)
    (scale := scale) (tid := tid) (sPre :=
      fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        M D Bk numKVBlocks scale s)
    (Q := Q) (K := K) (V := V) (dO := dO) (LSE := LSE) (block := block)
  · exact fa1BackwardAtomicDQPreAtomic_step
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE
  · exact fa1BackwardAtomicDQPreAtomic_qPtrs
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE
  · exact fa1BackwardAtomicDQPreAtomic_dQPart
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE
  · exact hTailStep

/-- Input-level stateful trace theorem for one block-program using the ordinary
`exec = some final` surface instead of an explicit tail-step hypothesis. -/
theorem fa1BackwardAtomicDQKernel_statefulTrace_blockContribution_from_inputs_of_exec
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (scale : ℝ) (tid : ThreadId) (s final : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (block : Fin numKVBlocks)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1))
    (hExec :
      exec
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel s =
        some final) :
    Kernel.AtomicTraceStateful
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel
        tid s
        ((TileShape.allIndices [M, D]).filterMap fun i =>
          some (Stmt.atomicTraceEvent tid dQReg
            (Offset.rowMajor2D (rows := M) (cols := D) 0 D i) .real
            (some (dQBlockContribution Q K V dO LSE scale block i))))
        final := by
  let sPre : BlockState :=
    fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale s
  have hPre :
      stepStmts
          ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
              M D Bk numKVBlocks scale).toAlgKernel.body.take 29) s =
        some sPre := by
    exact fa1BackwardAtomicDQPreAtomic_step
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE
  have hTailStep :
      stepStmts
        ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body.drop 29)
        sPre =
        some final := by
    have hExec' := hExec
    rw [show
        exec
            (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
              M D Bk numKVBlocks scale).toAlgKernel s =
          stepStmts
            ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
              M D Bk numKVBlocks scale).toAlgKernel.body) s by
      rfl] at hExec'
    rw [show
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body =
          ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
            M D Bk numKVBlocks scale).toAlgKernel.body.take 29) ++
          ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
            M D Bk numKVBlocks scale).toAlgKernel.body.drop 29) by
      exact (List.take_append_drop 29 _).symm] at hExec'
    rw [stepStmts.append_some hPre] at hExec'
    exact hExec'
  exact fa1BackwardAtomicDQKernel_statefulTrace_blockContribution_from_inputs
    qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
    scale tid s final Q K V dO LSE block
    hPid hQ hK hV hdO hLSE hTailStep

/-- Store-stage readback helper for the row-major 2D stores used by the
stripped backward kernel.  It factors out the final proof obligation for each
of `dQ`, `dK`, and `dV`: once the address tile and value tile are in
registers, the corresponding unmasked store reads back exactly that value. -/
theorem store_stage_readback_rowMajor2D
    (outReg : RegionName) (ptrName valueName : RegName)
    (rows cols : Nat) (s : BlockState)
    (valueFn : TileIndex [rows, cols] → ℝ)
    (hPtrs : s.regs .nat [rows, cols] ptrName =
      some (⟨Offset.rowMajor2D (rows := rows) (cols := cols) 0 cols⟩ :
        Tile .nat [rows, cols]))
    (hVal : s.regs .real [rows, cols] valueName = some (Tile.ofReal valueFn)) :
    ∀ idx : TileIndex [rows, cols],
      observeTileAt
        (stepStmt (Stmt.store .real [rows, cols]
          (MemAccess.region outReg (Op.ref .nat [rows, cols] ptrName))
          (Op.ref .real [rows, cols] valueName) MaskOpt.none) s)
        outReg (Offset.rowMajor2D (rows := rows) (cols := cols) 0 cols) idx =
      some (valueFn idx) := by
  intro idx
  have h_inj : Function.Injective
      (fun i : TileIndex [rows, cols] => i.1.val * cols + i.2.1.val) := by
    intro a b h
    have hrow : Offset.rowMajor2D (rows := rows) (cols := cols) 0 cols a =
        Offset.rowMajor2D (rows := rows) (cols := cols) 0 cols b := by
      simpa [Offset.rowMajor2D, Offset.strided] using h
    exact Offset.rowMajor2D_inj (base := 0) (rowStride := cols) (le_refl cols) hrow
  simp [observeTileAt, stepStmt, evalOp, hPtrs, hVal, Tile.ofReal,
    BlockState.writeMemTyped_real, Offset.rowMajor2D, Offset.strided]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj idx]

/-- Row-major store-stage readback with a nonzero base offset. -/
theorem store_stage_readback_rowMajor2D_base
    (outReg : RegionName) (ptrName valueName : RegName)
    (rows cols base : Nat) (s : BlockState)
    (valueFn : TileIndex [rows, cols] → ℝ)
    (hPtrs : s.regs .nat [rows, cols] ptrName =
      some (⟨Offset.rowMajor2D (rows := rows) (cols := cols) base cols⟩ :
        Tile .nat [rows, cols]))
    (hVal : s.regs .real [rows, cols] valueName = some (Tile.ofReal valueFn)) :
    ∀ idx : TileIndex [rows, cols],
      observeTileAt
        (stepStmt (Stmt.store .real [rows, cols]
          (MemAccess.region outReg (Op.ref .nat [rows, cols] ptrName))
          (Op.ref .real [rows, cols] valueName) MaskOpt.none) s)
        outReg (Offset.rowMajor2D (rows := rows) (cols := cols) base cols) idx =
      some (valueFn idx) := by
  intro idx
  have h_inj : Function.Injective
      (fun i : TileIndex [rows, cols] => base + i.1.val * cols + i.2.1.val) := by
    intro a b h
    have hrow : Offset.rowMajor2D (rows := rows) (cols := cols) base cols a =
        Offset.rowMajor2D (rows := rows) (cols := cols) base cols b := by
      simpa [Offset.rowMajor2D, Offset.strided] using h
    exact Offset.rowMajor2D_inj (base := base) (rowStride := cols) (le_refl cols) hrow
  simp [observeTileAt, stepStmt, evalOp, hPtrs, hVal, Tile.ofReal,
    BlockState.writeMemTyped_real, Offset.rowMajor2D, Offset.strided]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj idx]

/-- Readback for the ordinary `dK`/`dV` stores in the atomic-`dQ` backward
tail.  The leading atomic `dQ` update only changes memory and preserves the
registers consumed by the subsequent ordinary stores. -/
theorem atomicBackward_tailStores_readback_rowMajor2D_base
    (dQReg dKReg dVReg : RegionName)
    (M D Bk base : Nat) (s : BlockState)
    (dQPart : TileIndex [M, D] → ℝ)
    (dKFn dVFn : TileIndex [Bk, D] → ℝ)
    (hPtrsQ : s.regs .nat [M, D] "q_ptrs" =
      some (⟨Offset.rowMajor2D (rows := M) (cols := D) 0 D⟩ :
        Tile .nat [M, D]))
    (hPtrsK : s.regs .nat [Bk, D] "k_block_ptrs" =
      some (⟨Offset.rowMajor2D (rows := Bk) (cols := D) base D⟩ :
        Tile .nat [Bk, D]))
    (hPtrsV : s.regs .nat [Bk, D] "v_block_ptrs" =
      some (⟨Offset.rowMajor2D (rows := Bk) (cols := D) base D⟩ :
        Tile .nat [Bk, D]))
    (hValQ : s.regs .real [M, D] "dQ_part" = some (Tile.ofReal dQPart))
    (hValK : s.regs .real [Bk, D] "dK_block" = some (Tile.ofReal dKFn))
    (hValV : s.regs .real [Bk, D] "dV_block" = some (Tile.ofReal dVFn))
    (hdKdV : dKReg ≠ dVReg) :
    (∀ idx : TileIndex [Bk, D],
      observeTileAt
        ((stepStmt (Stmt.atomicAdd NumericDType.real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ_part") MaskOpt.none) s).bind fun s1 =>
          (stepStmt (Stmt.store .real [Bk, D]
            (MemAccess.region dKReg (Op.ref .nat [Bk, D] "k_block_ptrs"))
            (Op.ref .real [Bk, D] "dK_block") MaskOpt.none) s1).bind fun s2 =>
            stepStmt (Stmt.store .real [Bk, D]
              (MemAccess.region dVReg (Op.ref .nat [Bk, D] "v_block_ptrs"))
              (Op.ref .real [Bk, D] "dV_block") MaskOpt.none) s2)
        dKReg (Offset.rowMajor2D (rows := Bk) (cols := D) base D) idx =
      some (dKFn idx)) ∧
    (∀ idx : TileIndex [Bk, D],
      observeTileAt
        ((stepStmt (Stmt.atomicAdd NumericDType.real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ_part") MaskOpt.none) s).bind fun s1 =>
          (stepStmt (Stmt.store .real [Bk, D]
            (MemAccess.region dKReg (Op.ref .nat [Bk, D] "k_block_ptrs"))
            (Op.ref .real [Bk, D] "dK_block") MaskOpt.none) s1).bind fun s2 =>
            stepStmt (Stmt.store .real [Bk, D]
              (MemAccess.region dVReg (Op.ref .nat [Bk, D] "v_block_ptrs"))
              (Op.ref .real [Bk, D] "dV_block") MaskOpt.none) s2)
        dVReg (Offset.rowMajor2D (rows := Bk) (cols := D) base D) idx =
      some (dVFn idx)) := by
  have hInj : Function.Injective
      (Offset.rowMajor2D (rows := Bk) (cols := D) base D) :=
    Offset.rowMajor2D_inj (base := base) (rowStride := D) (le_refl D)
  constructor
  · intro idx
    simp [observeTileAt, stepStmt, evalOp, hPtrsQ, hPtrsK, hPtrsV,
      hValQ, hValK, hValV, Tile.ofReal, BlockState.writeMemTyped_real,
      BlockState.foldl_writeMem_regs]
    rw [BlockState.scatter_preserves_other_region dVReg
      (Offset.rowMajor2D (rows := Bk) (cols := D) base D) dVFn dKReg hdKdV]
    rw [BlockState.scatter_readback_nd _ _ _ hInj idx]
  · intro idx
    simp [observeTileAt, stepStmt, evalOp, hPtrsQ, hPtrsK, hPtrsV,
      hValQ, hValK, hValV, Tile.ofReal, BlockState.writeMemTyped_real,
      BlockState.foldl_writeMem_regs]
    rw [BlockState.scatter_readback_nd _ _ _ hInj idx]

set_option maxHeartbeats 2000000 in
set_option linter.unusedSimpArgs false in
/-- Input-level readback for the ordinary `dK`/`dV` stores of one atomic-`dQ`
backward block program.  The `dQ` result is handled by the atomic trace/merge
theorems; this lemma closes the ordinary tail stores for the same full kernel
execution. -/
theorem fa1BackwardAtomicDQKernel_tailStores_readback_from_inputs
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (scale : ℝ) (s : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (block : Fin numKVBlocks)
    (hPid : s.pids 0 = block.val)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1))
    (hdKdV : dKReg ≠ dVReg) :
    let bw := attentionBackwardReal Q K V dO LSE scale
    let base := block.val * Bk * D
    (∀ idx : TileIndex [Bk, D],
      observeTileAt
        (exec
          (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
            M D Bk numKVBlocks scale).toAlgKernel s)
        dKReg (Offset.rowMajor2D (rows := Bk) (cols := D) base D) idx =
      some (bw.dK
        (FA1Math.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) ∧
    (∀ idx : TileIndex [Bk, D],
      observeTileAt
        (exec
          (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
            M D Bk numKVBlocks scale).toAlgKernel s)
        dVReg (Offset.rowMajor2D (rows := Bk) (cols := D) base D) idx =
      some (bw.dV
        (FA1Math.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) := by
  let sPre : BlockState :=
    fa1BackwardAtomicDQPreAtomicState qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      M D Bk numKVBlocks scale s
  let dKFn : TileIndex [Bk, D] → ℝ := fun idx =>
    (attentionBackwardReal Q K V dO LSE scale).dK
      (FA1Math.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)
  let dVFn : TileIndex [Bk, D] → ℝ := fun idx =>
    (attentionBackwardReal Q K V dO LSE scale).dV
      (FA1Math.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)
  have hPre :
      stepStmts
          ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
              M D Bk numKVBlocks scale).toAlgKernel.body.take 29) s =
        some sPre := by
    exact fa1BackwardAtomicDQPreAtomic_step
      qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
      Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE
  have hTail := atomicBackward_tailStores_readback_rowMajor2D_base
    (dQReg := dQReg) (dKReg := dKReg) (dVReg := dVReg)
    (M := M) (D := D) (Bk := Bk) (base := block.val * Bk * D)
    (s := sPre)
    (dQPart := dQBlockContribution Q K V dO LSE scale block)
    (dKFn := dKFn) (dVFn := dVFn)
    (hPtrsQ := by
      exact fa1BackwardAtomicDQPreAtomic_qPtrs
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE)
    (hPtrsK := by
      exact fa1BackwardAtomicDQPreAtomic_kBlockPtrs
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE)
    (hPtrsV := by
      exact fa1BackwardAtomicDQPreAtomic_vBlockPtrs
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE)
    (hValQ := by
      exact fa1BackwardAtomicDQPreAtomic_dQPart
        qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
        Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE)
    (hValK := by
      simpa [dKFn] using
        fa1BackwardAtomicDQPreAtomic_dKBlock
          qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE)
    (hValV := by
      simpa [dVFn] using
        fa1BackwardAtomicDQPreAtomic_dVBlock
          qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          Q K V dO LSE scale block s hPid hQ hK hV hdO hLSE)
    hdKdV
  have hExecTail :
      exec
          (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
            M D Bk numKVBlocks scale).toAlgKernel s =
        ((stepStmt (Stmt.atomicAdd NumericDType.real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ_part") MaskOpt.none) sPre).bind fun s1 =>
          (stepStmt (Stmt.store .real [Bk, D]
            (MemAccess.region dKReg (Op.ref .nat [Bk, D] "k_block_ptrs"))
            (Op.ref .real [Bk, D] "dK_block") MaskOpt.none) s1).bind fun s2 =>
            stepStmt (Stmt.store .real [Bk, D]
              (MemAccess.region dVReg (Op.ref .nat [Bk, D] "v_block_ptrs"))
              (Op.ref .real [Bk, D] "dV_block") MaskOpt.none) s2) := by
    rw [show
        exec
            (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
              M D Bk numKVBlocks scale).toAlgKernel s =
          stepStmts
            ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
              M D Bk numKVBlocks scale).toAlgKernel.body) s by
      rfl]
    rw [show
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel.body =
          ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
            M D Bk numKVBlocks scale).toAlgKernel.body.take 29) ++
          ((fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
            M D Bk numKVBlocks scale).toAlgKernel.body.drop 29) by
      exact (List.take_append_drop 29 _).symm]
    rw [stepStmts.append_some hPre]
    rw [fa1BackwardAtomicDQKernel_drop29]
    simp [stepStmts, stepStmt, evalOp]
    cases hAtomic :
        ((sPre.regs TileDType.real [M, D] "dQ_part").bind fun values =>
          (sPre.regs TileDType.nat [M, D] "q_ptrs").bind fun offsets =>
            some
              (List.foldl
                (fun acc i =>
                  acc.writeMem dQReg (offsets.data i)
                    (WithBot.unbotD 0
                      (NumericDType.real.add
                        (some (acc.readMem dQReg (offsets.data i)))
                        (values.data i))))
                sPre (TileShape.allIndices [M, D]))) with
    | none =>
        simp
    | some s1 =>
        cases hkStore :
            ((s1.regs TileDType.real [Bk, D] "dK_block").bind fun values =>
              (s1.regs TileDType.nat [Bk, D] "k_block_ptrs").bind fun offsets =>
                some
                  (List.foldl
                    (fun acc i => acc.writeMem dKReg (offsets.data i)
                      (WithBot.unbotD 0 (values.data i)))
                    s1 (TileShape.allIndices [Bk, D]))) with
        | none =>
            simp [hkStore]
        | some s2 =>
            cases hvStore :
                ((s2.regs TileDType.real [Bk, D] "dV_block").bind fun values =>
                  (s2.regs TileDType.nat [Bk, D] "v_block_ptrs").bind fun offsets =>
                    some
                      (List.foldl
                        (fun acc i => acc.writeMem dVReg (offsets.data i)
                          (WithBot.unbotD 0 (values.data i)))
                        s2 (TileShape.allIndices [Bk, D]))) with
            | none =>
                simp [hkStore, hvStore]
            | some s3 =>
                simp [hkStore, hvStore]
  constructor
  · intro idx
    rw [hExecTail]
    exact hTail.1 idx
  · intro idx
    rw [hExecTail]
    exact hTail.2 idx

/-- Grid-merge correctness for the atomic `dQ` output cells.

This packages the generic #66 Real atomic-add merge theorem with the FA-1
block-contribution algebra: if the selected per-program traces contribute
exactly one full partition of KV blocks and the initial `dQ` buffer is zero,
then the merged `dQ` tile reads back `attentionBackwardReal.dQ`.

The per-program trace theorem above supplies the concrete trace shape for a
block program; this theorem is the final grid-level composition point. -/
theorem fa1BackwardAtomicDQKernel_gridMerged_dQ_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (scale : ℝ) (s : BlockState)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ)
    (g : Grid)
    (frames :
      Kernel.GridFrames
        (fa1BackwardAtomicDQKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
          M D Bk numKVBlocks scale).toAlgKernel g s)
    (contributors : Finset (GridIndex g))
    (atomicTrace : GridIndex g → Trace)
    (hInitialDQ :
      ∀ idx : TileIndex [M, D],
        s.readMem dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D idx) = 0)
    (hNoOrdinaryDQ :
      ∀ idx : TileIndex [M, D],
        ¬ Kernel.GridWriteFootprint frames
          (dQReg, Offset.rowMajor2D (rows := M) (cols := D) 0 D idx))
    (hAtomicContrib :
      ∀ idx : TileIndex [M, D],
        contributors.sum
            (fun gridIdx =>
              (atomicTrace gridIdx).atomicAddRealSum
                (dQReg, Offset.rowMajor2D (rows := M) (cols := D) 0 D idx)) =
          Finset.univ.sum
            (fun block : Fin numKVBlocks =>
              dQBlockContribution Q K V dO LSE scale block idx)) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
        (some (Kernel.mergeFramesWithAtomic g s frames contributors atomicTrace))
        dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D) idx =
      some ((attentionBackwardReal Q K V dO LSE scale).dQ idx) := by
  intro idx
  simp [observeTileAt]
  rw [Kernel.mergeFramesWithAtomic_atomicAdd_eq_finsetSum
    (frames := frames) (contributors := contributors) (atomicTrace := atomicTrace)
    (region := dQReg) (offset := Offset.rowMajor2D (rows := M) (cols := D) 0 D idx)
    (hNoOrdinaryWrite := hNoOrdinaryDQ idx)]
  rw [hInitialDQ idx, hAtomicContrib idx]
  rw [dQBlockContribution_sum_eq_attentionBackwardReal Q K V dO LSE scale idx]
  simp [attentionBackwardReal_dQ]

/-- Readback for the final three stores of the stripped backward kernel.

The computational prefix establishes the pointer/value registers; this lemma
then handles the memory-only tail and the fact that later stores to `dK`/`dV`
do not clobber `dQ`, and the later store to `dV` does not clobber `dK`. -/
theorem strippedBackward_finalStores_readback
    (dQReg dKReg dVReg : RegionName)
    (M S D : Nat) (s : BlockState)
    (dQFn : TileIndex [M, D] → ℝ)
    (dKFn dVFn : TileIndex [S, D] → ℝ)
    (hPtrsQ : s.regs .nat [M, D] "q_ptrs" =
      some (⟨Offset.rowMajor2D (rows := M) (cols := D) 0 D⟩ :
        Tile .nat [M, D]))
    (hPtrsK : s.regs .nat [S, D] "k_ptrs" =
      some (⟨Offset.rowMajor2D (rows := S) (cols := D) 0 D⟩ :
        Tile .nat [S, D]))
    (hPtrsV : s.regs .nat [S, D] "v_ptrs" =
      some (⟨Offset.rowMajor2D (rows := S) (cols := D) 0 D⟩ :
        Tile .nat [S, D]))
    (hValQ : s.regs .real [M, D] "dQ" = some (Tile.ofReal dQFn))
    (hValK : s.regs .real [S, D] "dK" = some (Tile.ofReal dKFn))
    (hValV : s.regs .real [S, D] "dV" = some (Tile.ofReal dVFn))
    (hdQdK : dQReg ≠ dKReg) (hdQdV : dQReg ≠ dVReg) (hdKdV : dKReg ≠ dVReg) :
    (∀ idx : TileIndex [M, D],
      observeTileAt
        ((stepStmt (Stmt.store .real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ") MaskOpt.none) s).bind fun s1 =>
          (stepStmt (Stmt.store .real [S, D]
            (MemAccess.region dKReg (Op.ref .nat [S, D] "k_ptrs"))
            (Op.ref .real [S, D] "dK") MaskOpt.none) s1).bind fun s2 =>
            stepStmt (Stmt.store .real [S, D]
              (MemAccess.region dVReg (Op.ref .nat [S, D] "v_ptrs"))
              (Op.ref .real [S, D] "dV") MaskOpt.none) s2)
        dQReg (Offset.rowMajor2D (rows := M) (cols := D) 0 D) idx =
      some (dQFn idx)) ∧
    (∀ idx : TileIndex [S, D],
      observeTileAt
        ((stepStmt (Stmt.store .real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ") MaskOpt.none) s).bind fun s1 =>
          (stepStmt (Stmt.store .real [S, D]
            (MemAccess.region dKReg (Op.ref .nat [S, D] "k_ptrs"))
            (Op.ref .real [S, D] "dK") MaskOpt.none) s1).bind fun s2 =>
            stepStmt (Stmt.store .real [S, D]
              (MemAccess.region dVReg (Op.ref .nat [S, D] "v_ptrs"))
              (Op.ref .real [S, D] "dV") MaskOpt.none) s2)
        dKReg (Offset.rowMajor2D (rows := S) (cols := D) 0 D) idx =
      some (dKFn idx)) ∧
    (∀ idx : TileIndex [S, D],
      observeTileAt
        ((stepStmt (Stmt.store .real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ") MaskOpt.none) s).bind fun s1 =>
          (stepStmt (Stmt.store .real [S, D]
            (MemAccess.region dKReg (Op.ref .nat [S, D] "k_ptrs"))
            (Op.ref .real [S, D] "dK") MaskOpt.none) s1).bind fun s2 =>
            stepStmt (Stmt.store .real [S, D]
              (MemAccess.region dVReg (Op.ref .nat [S, D] "v_ptrs"))
              (Op.ref .real [S, D] "dV") MaskOpt.none) s2)
        dVReg (Offset.rowMajor2D (rows := S) (cols := D) 0 D) idx =
      some (dVFn idx)) := by
  have hInjQ : Function.Injective
      (Offset.rowMajor2D (rows := M) (cols := D) 0 D) :=
    Offset.rowMajor2D_inj (base := 0) (rowStride := D) (le_refl D)
  have hInjK : Function.Injective
      (Offset.rowMajor2D (rows := S) (cols := D) 0 D) :=
    Offset.rowMajor2D_inj (base := 0) (rowStride := D) (le_refl D)
  have hInjV : Function.Injective
      (Offset.rowMajor2D (rows := S) (cols := D) 0 D) :=
    Offset.rowMajor2D_inj (base := 0) (rowStride := D) (le_refl D)
  constructor
  · intro idx
    simp [observeTileAt, stepStmt, evalOp, hPtrsQ, hPtrsK, hPtrsV,
      hValQ, hValK, hValV, Tile.ofReal, BlockState.writeMemTyped_real]
    rw [BlockState.scatter_preserves_other_region dVReg
      (Offset.rowMajor2D (rows := S) (cols := D) 0 D) dVFn dQReg hdQdV]
    rw [BlockState.scatter_preserves_other_region dKReg
      (Offset.rowMajor2D (rows := S) (cols := D) 0 D) dKFn dQReg hdQdK]
    rw [BlockState.scatter_readback_nd _ _ _ hInjQ idx]
  · constructor
    · intro idx
      simp [observeTileAt, stepStmt, evalOp, hPtrsQ, hPtrsK, hPtrsV,
        hValQ, hValK, hValV, Tile.ofReal, BlockState.writeMemTyped_real]
      rw [BlockState.scatter_preserves_other_region dVReg
        (Offset.rowMajor2D (rows := S) (cols := D) 0 D) dVFn dKReg hdKdV]
      rw [BlockState.scatter_readback_nd _ _ _ hInjK idx]
    · intro idx
      simp [observeTileAt, stepStmt, evalOp, hPtrsQ, hPtrsK, hPtrsV,
        hValQ, hValK, hValV, Tile.ofReal, BlockState.writeMemTyped_real]
      rw [BlockState.scatter_readback_nd _ _ _ hInjV idx]

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false in

/-- Sub-2b execution wiring: the stripped FA-1 backward kernel produces final
memory whose `dQ` / `dK` / `dV` slices match `attentionBackwardReal`.

Connects:
- the existing tile-level bridges (`dV_tile_eq_attentionBackwardReal` /
  `dQ_tile_eq_attentionBackwardReal` / `dK_tile_eq_attentionBackwardReal`);
- the store-stage readback helper (`store_stage_readback_rowMajor2D`);
- through the `fa1BackwardStrippedKernel` execution path. -/
theorem fa1BackwardStrippedKernel_correct
    {M S D : Nat}
    (qReg kReg vReg dOReg lseReg dQReg dKReg dVReg : RegionName)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := S) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := S) (cols := D) 0 D) V)
    (hdO : InputAt s dOReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) dO)
    (hLSE : InputAt (shape := [M]) s lseReg
        (fun idx : TileIndex [M] => idx.1.val)
        (fun idx : TileIndex [M] => LSE idx.1))
    (hOutDisjoint :
      dQReg ≠ dKReg ∧ dQReg ≠ dVReg ∧ dKReg ≠ dVReg) :
    let bw := attentionBackwardReal Q K V dO LSE scale
    (∀ idx : TileIndex [M, D],
      observeTileAt
        (exec (fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg
            dQReg dKReg dVReg M S D scale) s)
        dQReg
        (Offset.rowMajor2D (rows := M) (cols := D) 0 D) idx
      = some (bw.dQ idx)) ∧
    (∀ idx : TileIndex [S, D],
      observeTileAt
        (exec (fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg
            dQReg dKReg dVReg M S D scale) s)
        dKReg
        (Offset.rowMajor2D (rows := S) (cols := D) 0 D) idx
      = some (bw.dK idx)) ∧
    (∀ idx : TileIndex [S, D],
      observeTileAt
        (exec (fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg
            dQReg dKReg dVReg M S D scale) s)
        dVReg
        (Offset.rowMajor2D (rows := S) (cols := D) 0 D) idx
      = some (bw.dV idx)) := by
  rcases hOutDisjoint with ⟨hdQdK, hdQdV, hdKdV⟩
  simp [InputAt, Offset.rowMajor2D, Offset.strided] at hQ hK hV hdO hLSE
  have hQ0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem qReg (i.val * D + d.val) = Q (i, d, PUnit.unit) :=
    fun i d => hQ i d PUnit.unit
  have hK0 : ∀ (j : Fin S) (d : Fin D),
      s.readMem kReg (j.val * D + d.val) = K (j, d, PUnit.unit) :=
    fun j d => hK j d PUnit.unit
  have hV0 : ∀ (j : Fin S) (d : Fin D),
      s.readMem vReg (j.val * D + d.val) = V (j, d, PUnit.unit) :=
    fun j d => hV j d PUnit.unit
  have hdO0 : ∀ (i : Fin M) (d : Fin D),
      s.readMem dOReg (i.val * D + d.val) = dO (i, d, PUnit.unit) :=
    fun i d => hdO i d PUnit.unit
  let sPre : BlockState :=
    fa1BackwardStrippedPreStoreState qReg kReg vReg dOReg lseReg
      dQReg dKReg dVReg M S D scale s
  have hPre :
      stepStmts
          ((fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg dQReg dKReg dVReg
              M S D scale).toAlgKernel.body.take 20) s =
        some sPre := by
    simp [sPre, fa1BackwardStrippedPreStoreState, fa1BackwardStrippedKernel,
      stepStmts, stepStmt, evalOp, Option.bind, hQ0, hK0, hV0, hdO0, hLSE,
      Offset.rowMajor2D, Offset.strided,
      TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
      Tile.ofReal, dQ_tile_some_eq_attentionBackwardReal, dK_tile_some_eq_attentionBackwardReal,
      dV_tile_some_eq_attentionBackwardReal, dQ_tile_eq_attentionBackwardReal,
      dK_tile_eq_attentionBackwardReal, dV_tile_eq_attentionBackwardReal,
      dS_tile_eq, rowCorrection_tile_eq, dP_tile_eq, probability_tile_eq,
      WithBot.sum_someTerm_eq_some, dS, rowCorrection, dP, probability, FA1Math.scaledScore]
  have hTail := strippedBackward_finalStores_readback
    dQReg dKReg dVReg M S D
    (s := sPre)
    (dQFn := (attentionBackwardReal Q K V dO LSE scale).dQ)
    (dKFn := (attentionBackwardReal Q K V dO LSE scale).dK)
    (dVFn := (attentionBackwardReal Q K V dO LSE scale).dV)
    (hPtrsQ := by
      simp [sPre, fa1BackwardStrippedPreStoreState, fa1BackwardStrippedKernel,
        stepStmts, stepStmt, evalOp, Option.bind, hQ0, hK0, hV0, hdO0, hLSE,
        Offset.rowMajor2D, Offset.strided,
        TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
        Tile.ofReal, probability_tile_eq, dP_tile_eq, rowCorrection_tile_eq, dS_tile_eq,
        dQ_tile_eq_attentionBackwardReal, dK_tile_eq_attentionBackwardReal,
        dV_tile_eq_attentionBackwardReal]
      )
    (hPtrsK := by
      simp [sPre, fa1BackwardStrippedPreStoreState, fa1BackwardStrippedKernel,
        stepStmts, stepStmt, evalOp, Option.bind, hQ0, hK0, hV0, hdO0, hLSE,
        Offset.rowMajor2D, Offset.strided,
        Tile.bop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
        Tile.ofReal, probability_tile_eq, dP_tile_eq, rowCorrection_tile_eq, dS_tile_eq,
        dQ_tile_eq_attentionBackwardReal, dK_tile_eq_attentionBackwardReal,
        dV_tile_eq_attentionBackwardReal]
      funext idx
      cases idx with
      | mk i rest =>
        cases rest with
        | mk d tail =>
          cases tail
          rfl)
    (hPtrsV := by
      simp [sPre, fa1BackwardStrippedPreStoreState, fa1BackwardStrippedKernel,
        stepStmts, stepStmt, evalOp, Option.bind, hQ0, hK0, hV0, hdO0, hLSE,
        Offset.rowMajor2D, Offset.strided,
        Tile.bop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
        Tile.ofReal, probability_tile_eq, dP_tile_eq, rowCorrection_tile_eq, dS_tile_eq,
        dQ_tile_eq_attentionBackwardReal, dK_tile_eq_attentionBackwardReal,
        dV_tile_eq_attentionBackwardReal]
      funext idx
      cases idx with
      | mk i rest =>
        cases rest with
        | mk d tail =>
          cases tail
          rfl)
    (hValQ := by
      simp [sPre, fa1BackwardStrippedPreStoreState, fa1BackwardStrippedKernel,
        stepStmts, stepStmt, evalOp, Option.bind, hQ0, hK0, hV0, hdO0, hLSE,
        Offset.rowMajor2D, Offset.strided,
        Tile.bop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
        Tile.ofReal, dQ_tile_some_eq_attentionBackwardReal, dQ_tile_eq_attentionBackwardReal,
        dS_tile_eq, rowCorrection_tile_eq, dP_tile_eq, probability_tile_eq]
      funext idx
      cases idx with
      | mk i rest =>
        cases rest with
        | mk d tail =>
          cases tail
          simp [TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
            NumericDType.add, NumericDType.mul, NumericDType.sub,
            hQ0, hK0, hV0, hdO0, hLSE,
            Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim,
            Tile.reduceSumDrop, Tile.ofReal,
            dQ_tile_eq_attentionBackwardReal, dS_tile_eq, rowCorrection_tile_eq,
            dP_tile_eq, probability_tile_eq, WithBot.sum_someTerm_eq_some,
            dS, rowCorrection, dP, probability, FA1Math.scaledScore]
          simp [mul_comm, mul_left_comm, mul_assoc]
          rfl)
    (hValK := by
      simp [sPre, fa1BackwardStrippedPreStoreState, fa1BackwardStrippedKernel,
        stepStmts, stepStmt, evalOp, Option.bind, hQ0, hK0, hV0, hdO0, hLSE,
        Offset.rowMajor2D, Offset.strided,
        TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
        Tile.ofReal, dK_tile_some_eq_attentionBackwardReal, dK_tile_eq_attentionBackwardReal,
        dS_tile_eq, rowCorrection_tile_eq, dP_tile_eq, probability_tile_eq]
      funext idx
      cases idx with
      | mk i rest =>
        cases rest with
        | mk d tail =>
          cases tail
          simp [TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
            NumericDType.add, NumericDType.mul, NumericDType.sub,
            hQ0, hK0, hV0, hdO0, hLSE,
            Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim,
            Tile.reduceSumDrop, Tile.ofReal,
            dK_tile_eq_attentionBackwardReal, dS_tile_eq, rowCorrection_tile_eq,
            dP_tile_eq, probability_tile_eq, WithBot.sum_someTerm_eq_some,
            dS, rowCorrection, dP, probability, FA1Math.scaledScore]
          rw [mul_comm]
          apply congrArg (fun z : ℝ => scale * z)
          apply Finset.sum_congr rfl
          intro x hx
          have hExpI :
              Real.exp
                    ((∑ x_1, Q (x, x_1, PUnit.unit) * K (i, x_1, PUnit.unit))
                      * scale - LSE x) =
                Real.exp
                    (scale *
                        ∑ x_1, Q (x, x_1, PUnit.unit) * K (i, x_1, PUnit.unit)
                      - LSE x) := by
            congr 1
            ring
          rw [hExpI]
          set a : ℝ :=
            Real.exp
              (scale * ∑ x_1, Q (x, x_1, PUnit.unit) * K (i, x_1, PUnit.unit)
                - LSE x)
          set b : ℝ := ∑ x_1, dO (x, x_1, PUnit.unit) * V (i, x_1, PUnit.unit)
          set c : ℝ :=
            ∑ x_1,
              Real.exp (scale *
                    ∑ x_2, Q (x, x_2, PUnit.unit) * K (x_1, x_2, PUnit.unit)
                  - LSE x) *
                ∑ x_2, dO (x, x_2, PUnit.unit) * V (x_1, x_2, PUnit.unit)
          set c0 : ℝ :=
            ∑ x_1,
              Real.exp
                  ((∑ x_2, Q (x, x_2, PUnit.unit) * K (x_1, x_2, PUnit.unit))
                    * scale - LSE x) *
                ∑ x_2, dO (x, x_2, PUnit.unit) * V (x_1, x_2, PUnit.unit)
          have hc0 : c0 = c := by
            simp [c0, c]
            apply Finset.sum_congr rfl
            intro x_1 hx_1
            congr 1
            congr 1
            ring
          rw [← hc0]
          set q : ℝ := Q (x, d, PUnit.unit)
          change a * (b - c0) * q = a * (b - c0) * q
          rfl)
    (hValV := by
      simp [sPre, fa1BackwardStrippedPreStoreState, fa1BackwardStrippedKernel,
        stepStmts, stepStmt, evalOp, Option.bind, hQ0, hK0, hV0, hdO0, hLSE,
        Offset.rowMajor2D, Offset.strided,
        TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim, Tile.reduceSumDrop,
        Tile.ofReal, dV_tile_some_eq_attentionBackwardReal, dV_tile_eq_attentionBackwardReal,
        probability_tile_eq]
      funext idx
      cases idx with
      | mk i rest =>
        cases rest with
        | mk d tail =>
          cases tail
          simp [TileShape.dropInsertedIndex, TileShape.insertAxisIndex,
            NumericDType.add, NumericDType.mul, NumericDType.sub,
            hQ0, hK0, hV0, hdO0, hLSE,
            Tile.bop, Tile.uop, Tile.dot, Tile.transpose, Tile.expandDim,
            Tile.reduceSumDrop, Tile.ofReal,
            dV_tile_eq_attentionBackwardReal, probability_tile_eq,
            WithBot.sum_someTerm_eq_some, probability, FA1Math.scaledScore]
          apply Finset.sum_congr rfl
          intro x hx
          congr 1
          ring_nf)
    hdQdK hdQdV hdKdV
  have hExecTail :
      exec (fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg
          dQReg dKReg dVReg M S D scale) s =
        ((stepStmt (Stmt.store .real [M, D]
          (MemAccess.region dQReg (Op.ref .nat [M, D] "q_ptrs"))
          (Op.ref .real [M, D] "dQ") MaskOpt.none) sPre).bind fun s1 =>
          (stepStmt (Stmt.store .real [S, D]
            (MemAccess.region dKReg (Op.ref .nat [S, D] "k_ptrs"))
            (Op.ref .real [S, D] "dK") MaskOpt.none) s1).bind fun s2 =>
            stepStmt (Stmt.store .real [S, D]
              (MemAccess.region dVReg (Op.ref .nat [S, D] "v_ptrs"))
              (Op.ref .real [S, D] "dV") MaskOpt.none) s2) := by
    rw [show
        exec (fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg
          dQReg dKReg dVReg M S D scale) s =
          stepStmts
            ((fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg
              dQReg dKReg dVReg M S D scale).toAlgKernel.body) s by
      rfl]
    rw [show
        (fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg
          dQReg dKReg dVReg M S D scale).toAlgKernel.body =
          ((fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg
              dQReg dKReg dVReg M S D scale).toAlgKernel.body.take 20) ++
            ((fa1BackwardStrippedKernel qReg kReg vReg dOReg lseReg
              dQReg dKReg dVReg M S D scale).toAlgKernel.body.drop 20) by
      exact (List.take_append_drop 20 _).symm]
    rw [stepStmts.append_some hPre]
    simp [fa1BackwardStrippedKernel, stepStmts, stepStmt, evalOp]
    cases hqStore :
        ((sPre.regs TileDType.real [M, D] "dQ").bind fun values =>
          (sPre.regs TileDType.nat [M, D] "q_ptrs").bind fun offsets =>
            some
              (List.foldl
                (fun acc i => acc.writeMem dQReg (offsets.data i)
                  (WithBot.unbotD 0 (values.data i)))
                sPre (TileShape.allIndices [M, D]))) with
    | none =>
        simp
    | some s1 =>
        cases hkStore :
            ((s1.regs TileDType.real [S, D] "dK").bind fun values =>
              (s1.regs TileDType.nat [S, D] "k_ptrs").bind fun offsets =>
                some
                  (List.foldl
                    (fun acc i => acc.writeMem dKReg (offsets.data i)
                      (WithBot.unbotD 0 (values.data i)))
                    s1 (TileShape.allIndices [S, D]))) with
        | none =>
            simp [hkStore]
        | some s2 =>
            cases hvStore :
                ((s2.regs TileDType.real [S, D] "dV").bind fun values =>
                  (s2.regs TileDType.nat [S, D] "v_ptrs").bind fun offsets =>
                    some
                      (List.foldl
                        (fun acc i => acc.writeMem dVReg (offsets.data i)
                          (WithBot.unbotD 0 (values.data i)))
                        s2 (TileShape.allIndices [S, D]))) with
            | none =>
                simp [hkStore, hvStore]
            | some s3 =>
                simp [hkStore, hvStore]
  constructor
  · intro idx
    cases idx with
    | mk i rest =>
      cases rest with
      | mk d tail =>
        rw [hExecTail]
        exact hTail.1 (i, d, tail)
  · constructor
    · intro idx
      cases idx with
      | mk j rest =>
        cases rest with
        | mk d tail =>
          rw [hExecTail]
          exact hTail.2.1 (j, d, tail)
    · intro idx
      cases idx with
      | mk j rest =>
        cases rest with
        | mk d tail =>
          rw [hExecTail]
          exact hTail.2.2 (j, d, tail)

end FA1Backward

end VeriTile.Examples
