/-
VeriTile.Examples.FlashAttention1.Backward.Math

FA-1 backward Real math surface and tile bridge lemmas.
-/

import VeriTile.Examples.FlashAttention1.Common

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
theorem streamingLSE_eq_lseReal_impl {M D Bk : Nat} (hBk : 0 < Bk)
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

/-! ### Masked backward math surface

The production backward kernels eventually need causal and sequence-boundary
masks.  The definitions below expose the generic mask-aware Real spec while
keeping the existing non-masked surface as the all-visible specialization.
-/

/-- Mask-aware softmax probability. Invisible entries contribute zero to all
backward sums, matching the forward convention that masked logits behave like
`-inf`. -/
noncomputable def probabilityMasked {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (LSE : Fin M → ℝ) (scale : ℝ) (i : Fin M) (j : Fin S) : ℝ :=
  if visible i j then probability Q K LSE scale i j else 0

/-- Mask-aware row correction `D_i = Σ_j P_ij · dP_ij`, with invisible lanes
contributing zero. -/
noncomputable def rowCorrectionMasked {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (V : TileIndex [S, D] → ℝ) (dO : TileIndex [M, D] → ℝ)
    (LSE : Fin M → ℝ) (scale : ℝ) (i : Fin M) : ℝ :=
  Finset.univ.sum fun j : Fin S =>
    probabilityMasked visible Q K LSE scale i j * dP V dO i j

/-- Mask-aware softmax JVP. -/
noncomputable def dSMasked {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (V : TileIndex [S, D] → ℝ) (dO : TileIndex [M, D] → ℝ)
    (LSE : Fin M → ℝ) (scale : ℝ) (i : Fin M) (j : Fin S) : ℝ :=
  probabilityMasked visible Q K LSE scale i j *
    (dP V dO i j - rowCorrectionMasked visible Q K V dO LSE scale i)

/-- Closed-form reverse-mode FA-1 backward over Real tensors with an explicit
query/key visibility mask. -/
noncomputable def attentionBackwardRealMasked {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ) :
    Grads M S D :=
  { dQ := fun (i, d, _) =>
      scale * Finset.univ.sum fun j : Fin S =>
        dSMasked visible Q K V dO LSE scale i j * K (j, d, PUnit.unit)
    dK := fun (j, d, _) =>
      scale * Finset.univ.sum fun i : Fin M =>
        dSMasked visible Q K V dO LSE scale i j * Q (i, d, PUnit.unit)
    dV := fun (j, d, _) =>
      Finset.univ.sum fun i : Fin M =>
        probabilityMasked visible Q K LSE scale i j * dO (i, d, PUnit.unit) }

/-- Causal specialization of the generic mask-aware backward spec. -/
noncomputable def attentionBackwardRealCausal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ) :
    Grads M S D :=
  attentionBackwardRealMasked
    (fun i j => decide (j.val ≤ i.val)) Q K V dO LSE scale

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

/-- The mask-aware backward spec collapses to the existing non-masked spec
when every query/key pair is visible. -/
theorem attentionBackwardRealMasked_allVisible_impl {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ) :
    attentionBackwardRealMasked (fun _ _ => Bool.true) Q K V dO LSE scale =
      attentionBackwardReal Q K V dO LSE scale := by
  simp [attentionBackwardRealMasked, attentionBackwardReal, dSMasked, dS,
    rowCorrectionMasked, rowCorrection, probabilityMasked]

/-- Named reverse-mode spec.  Keeping this as a separate name makes the public
theorem surface state the intended equivalence explicitly while leaving the
closed form above as the executable definition. -/
noncomputable def reverseModeAttentionReal := @attentionBackwardReal

theorem attentionBackwardReal_eq_reverseMode_impl {M S D : Nat}
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

/-- Mask-aware contribution to `dQ` from one KV block. -/
noncomputable def dQBlockContributionMasked {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [M, D]) : ℝ :=
  scale * Finset.univ.sum fun jLocal : Fin Bk =>
    let j : Fin (Bk * numKVBlocks) :=
      FA1Math.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) jLocal
    dSMasked visible Q K V dO LSE scale idx.1 j * K (j, idx.2.1, PUnit.unit)

/-- Summing masked block-local `dQ` contributions over all KV blocks recovers
the mask-aware closed-form `dQ`. -/
theorem dQBlockContributionMasked_sum_eq_attentionBackwardRealMasked_impl
    {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    (Finset.univ.sum fun block : Fin numKVBlocks =>
      dQBlockContributionMasked visible Q K V dO LSE scale block idx) =
      (attentionBackwardRealMasked visible Q K V dO LSE scale).dQ idx := by
  simp only [dQBlockContributionMasked]
  cases idx with
  | mk i rest =>
      cases rest with
      | mk d tail =>
          cases tail
          simp only [attentionBackwardRealMasked]
          rw [← Finset.mul_sum]
          congr 1
          rw [← Finset.sum_product', Finset.univ_product_univ]
          refine (Finset.sum_equiv (FA1Math.blockIndexEquiv Bk numKVBlocks) ?_ ?_).symm
          · intro _; simp
          · intro j _
            rw [FA1Math.blockIndex_blockIndexEquiv]

/-- Causal specialization of the mask-aware block-local `dQ` contribution. -/
noncomputable def dQBlockContributionCausal {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [M, D]) : ℝ :=
  dQBlockContributionMasked
    (fun i j => decide (j.val ≤ i.val)) Q K V dO LSE scale block idx

/-- Summing causal block-local `dQ` contributions over all KV blocks recovers
the closed-form causal backward `dQ`. -/
theorem dQBlockContributionCausal_sum_eq_attentionBackwardRealCausal_impl
    {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    (Finset.univ.sum fun block : Fin numKVBlocks =>
      dQBlockContributionCausal Q K V dO LSE scale block idx) =
      (attentionBackwardRealCausal Q K V dO LSE scale).dQ idx := by
  simp [dQBlockContributionCausal, attentionBackwardRealCausal,
    dQBlockContributionMasked_sum_eq_attentionBackwardRealMasked_impl]

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

theorem softmax_jvp_identity_impl {S : Nat}
    (P dPRow : Fin S → ℝ) (j : Fin S) :
    softmaxJacobianJVP P dPRow j = softmaxJVP P dPRow j := rfl

end FA1Backward

end VeriTile.Examples
