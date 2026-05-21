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

/-- Backward outputs for 4D `[B, H, S, D]` tensors. -/
structure Grads4D (B H S_q S_k D : Nat) where
  dQ : TileIndex [B, H, S_q, D] → ℝ
  dK : TileIndex [B, H, S_k, D] → ℝ
  dV : TileIndex [B, H, S_k, D] → ℝ

/-- Forward-side log-sum-exp reconstructed from Q/K. -/
noncomputable def lseReal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (scale : ℝ) (i : Fin M) : ℝ :=
  Real.log (Finset.univ.sum fun j : Fin S =>
    Real.exp (StreamingAccumulator.scaledScore Q K scale i j))

/-- The value computed by the online-softmax registers at forward readout:
`m_i + log(l_i)`. -/
noncomputable def streamingLSE {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) (i : Fin M) : ℝ :=
  (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale numKVBlocks i).unbotD 0 +
    Real.log (StreamingAccumulator.lPartial Q numKVBlocks K scale numKVBlocks i)

/-- The forward LSE store is the usual unshifted log-sum-exp.

This is the math bridge needed by a forward kernel that stores
`m_i + tl.log(l_i)` for backward. -/
theorem streamingLSE_eq_lseReal_impl {M D Bk : Nat} (hBk : 0 < Bk)
    (hN : 0 < numKVBlocks) (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) (i : Fin M) :
    streamingLSE Q numKVBlocks K scale i = lseReal Q K scale i := by
  unfold streamingLSE lseReal
  let m : ℝ := (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale numKVBlocks i).unbotD 0
  have hlf_pos : 0 < StreamingAccumulator.lFree Q K scale numKVBlocks (le_refl numKVBlocks) i :=
    StreamingAccumulator.lFree_final_pos hBk hN Q K scale i
  have hlf_ne :
      StreamingAccumulator.lFree Q K scale numKVBlocks (le_refl numKVBlocks) i ≠ 0 :=
    ne_of_gt hlf_pos
  rw [StreamingAccumulator.lPartial_eq_mShifted hBk Q numKVBlocks K scale
      numKVBlocks (le_refl _) i]
  rw [Real.log_mul (Real.exp_ne_zero _) hlf_ne]
  rw [Real.log_exp]
  rw [StreamingAccumulator.lFree_eq_flat]
  ring

/-- Softmax probabilities reconstructed from a stored LSE row. -/
noncomputable def probability {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (LSE : Fin M → ℝ) (scale : ℝ) (i : Fin M) (j : Fin S) : ℝ :=
  Real.exp (StreamingAccumulator.scaledScore Q K scale i j - LSE i)

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

/-- Normal form bridge for masked logits: selecting the scaled score on
visible lanes and `⊥` on invisible lanes, then applying `exp(score - LSE)`,
is exactly the mask-aware probability.  This is the local semantic bridge used
to connect `tl.where(..., -inf)` code paths to the Real masked spec. -/
theorem maskedProbability_exp_eq {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (LSE : Fin M → ℝ) (scale : ℝ) (i : Fin M) (j : Fin S) :
    WithBot.realExp
      (Option.map (fun a : ℝ => a - LSE i)
        (if visible i j then
          some (scale * Finset.univ.sum fun d : Fin D =>
            Q (i, d, PUnit.unit) * K (j, d, PUnit.unit))
        else none)) =
      some (probabilityMasked visible Q K LSE scale i j) := by
  by_cases h : visible i j
  · simp [h, probabilityMasked, probability, StreamingAccumulator.scaledScore]
  · simp [h, probabilityMasked, WithBot.realExp]

theorem maskedProbability_exp_ite_eq {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (LSE : Fin M → ℝ) (scale : ℝ) (i : Fin M) (j : Fin S) :
    WithBot.realExp
      (if visible i j then
        some (scale * (Finset.univ.sum fun d : Fin D =>
          Q (i, d, PUnit.unit) * K (j, d, PUnit.unit)) - LSE i)
      else none) =
      some (probabilityMasked visible Q K LSE scale i j) := by
  by_cases h : visible i j
  · simp [h, probabilityMasked, probability, StreamingAccumulator.scaledScore]
  · simp [h, probabilityMasked, WithBot.realExp]

/-- Mask-aware row correction `D_i = Σ_j P_ij · dP_ij`, with invisible lanes
contributing zero. -/
noncomputable def rowCorrectionMasked {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (V : TileIndex [S, D] → ℝ) (dO : TileIndex [M, D] → ℝ)
    (LSE : Fin M → ℝ) (scale : ℝ) (i : Fin M) : ℝ :=
  Finset.univ.sum fun j : Fin S =>
    probabilityMasked visible Q K LSE scale i j * dP V dO i j

/-- Row-wise masked correction bridge: the WithBot sum produced by
`tl.sum(p * dP, axis=1)` after `tl.where(..., -inf)` equals the Real
`rowCorrectionMasked` spec. -/
theorem maskedRowCorrection_sum_eq {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ) (i : Fin M) :
    @Finset.sum (Fin S) (WithBot ℝ) _ Finset.univ (fun j : Fin S =>
      (Option.map (fun p : ℝ => p * dP V dO i j)
        (WithBot.realExp
          (Option.map (fun a : ℝ => a - LSE i)
            (if visible i j then
              some (scale * (Finset.univ.sum fun d : Fin D =>
                Q (i, d, PUnit.unit) * K (j, d, PUnit.unit)))
            else none))) : WithBot ℝ)) =
      ((rowCorrectionMasked visible Q K V dO LSE scale i : ℝ) : WithBot ℝ) := by
  have hterm :
      (fun j : Fin S =>
        (Option.map (fun p : ℝ => p * dP V dO i j)
          (WithBot.realExp
            (Option.map (fun a : ℝ => a - LSE i)
              (if visible i j then
                some (scale * (Finset.univ.sum fun d : Fin D =>
                  Q (i, d, PUnit.unit) * K (j, d, PUnit.unit)))
              else none))) : WithBot ℝ)) =
      (fun j : Fin S =>
        (some (probabilityMasked visible Q K LSE scale i j * dP V dO i j) :
          WithBot ℝ)) := by
    funext j
    rw [maskedProbability_exp_eq]
    rfl
  rw [hterm]
  simp only [rowCorrectionMasked]
  exact WithBot.sum_someTerm_eq_some (Finset.univ)
    (fun j : Fin S => probabilityMasked visible Q K LSE scale i j * dP V dO i j)

/-- Mask-aware softmax JVP. -/
noncomputable def dSMasked {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (V : TileIndex [S, D] → ℝ) (dO : TileIndex [M, D] → ℝ)
    (LSE : Fin M → ℝ) (scale : ℝ) (i : Fin M) (j : Fin S) : ℝ :=
  probabilityMasked visible Q K LSE scale i j *
    (dP V dO i j - rowCorrectionMasked visible Q K V dO LSE scale i)

/-- Masked softmax-JVP bridge for the kernel expression
`p * (dP - corr[:, None])`. -/
theorem maskedDS_option_eq {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (i : Fin M) (j : Fin S) :
    Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
      (WithBot.realExp
        (Option.map (fun a : ℝ => a - LSE i)
          (if visible i j then
            some (scale * (Finset.univ.sum fun d : Fin D =>
              Q (i, d, PUnit.unit) * K (j, d, PUnit.unit)))
          else none)))
      (Option.map (fun b : ℝ => dP V dO i j - b)
        (@Finset.sum (Fin S) (WithBot ℝ) _ Finset.univ (fun j' : Fin S =>
          (Option.map (fun p : ℝ => p * dP V dO i j')
            (WithBot.realExp
              (Option.map (fun a : ℝ => a - LSE i)
                (if visible i j' then
                  some (scale * (Finset.univ.sum fun d : Fin D =>
                    Q (i, d, PUnit.unit) * K (j', d, PUnit.unit)))
                else none))) : WithBot ℝ)))) =
      some (dSMasked visible Q K V dO LSE scale i j) := by
  rw [maskedProbability_exp_eq, maskedRowCorrection_sum_eq]
  simp [dSMasked]

/-- Block-local causal specialization of `maskedDS_option_eq`.

The causal atomic backward kernel builds the block-local mask as
`block * Bk + jLocal <= i`, while the Real spec uses the global key index
`blockIndex ... jLocal`.  This bridge identifies the two forms and packages the
kernel expression for `dS_block` as the causal `dSMasked` spec. -/
theorem causalBlockDS_option_eq {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (i : Fin M) (jLocal : Fin Bk) :
    Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
      (WithBot.realExp
        (Option.map (fun a : ℝ => a - LSE i)
          (if block.val * Bk + jLocal.val ≤ i.val then
            some (scale * (Finset.univ.sum fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                    (by have := block.isLt; omega) jLocal, d, PUnit.unit)))
          else none)))
      (Option.map
        (fun b : ℝ =>
          dP V dO i
              (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                (by have := block.isLt; omega) jLocal) - b)
        (@Finset.sum (Fin (Bk * numKVBlocks)) (WithBot ℝ) _ Finset.univ
          (fun j' : Fin (Bk * numKVBlocks) =>
            (Option.map (fun p : ℝ => p * dP V dO i j')
              (WithBot.realExp
                (Option.map (fun a : ℝ => a - LSE i)
                  (if j'.val ≤ i.val then
                    some (scale * (Finset.univ.sum fun d : Fin D =>
                      Q (i, d, PUnit.unit) * K (j', d, PUnit.unit)))
                  else none))) : WithBot ℝ)))) =
      some (dSMasked (fun i j => decide (j.val ≤ i.val))
        Q K V dO LSE scale i
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) jLocal)) := by
  simpa [StreamingAccumulator.blockIndex] using
    maskedDS_option_eq (M := M) (S := Bk * numKVBlocks) (D := D)
      (fun (i : Fin M) (j : Fin (Bk * numKVBlocks)) => decide (j.val ≤ i.val))
      Q K V dO LSE scale i
      (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
        (by have := block.isLt; omega) jLocal)

/-- Same bridge as `causalBlockDS_option_eq`, matching the exact comparison
normal form emitted by the kernel DSL after `tl.where(causal, ..., -inf)`.
The DSL stores the scaled dot product as `(sum) * scale`, while the Real spec
uses `scale * (sum)`; this lemma absorbs that syntactic difference. -/
theorem causalBlockDS_kernel_option_eq {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (i : Fin M) (jLocal : Fin Bk) :
    Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
      (WithBot.realExp
        (Option.map (fun a : ℝ => a - LSE i)
          (if ComparableDType.nat.ge i.val (block.val * Bk + jLocal.val) = Bool.true then
            some ((Finset.univ.sum fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                    (by have := block.isLt; omega) jLocal, d, PUnit.unit)) * scale)
          else none)))
      (Option.map
        (fun b : ℝ =>
          dP V dO i
              (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                (by have := block.isLt; omega) jLocal) - b)
        (@Finset.sum (Fin (Bk * numKVBlocks)) (WithBot ℝ) _ Finset.univ
          (fun j' : Fin (Bk * numKVBlocks) =>
            (Option.map (fun p : ℝ => p * dP V dO i j')
              (WithBot.realExp
                (Option.map (fun a : ℝ => a - LSE i)
                  (if ComparableDType.nat.ge i.val j'.val = Bool.true then
                    some ((Finset.univ.sum fun d : Fin D =>
                      Q (i, d, PUnit.unit) * K (j', d, PUnit.unit)) * scale)
                  else none))) : WithBot ℝ)))) =
      some (dSMasked (fun i j => decide (j.val ≤ i.val))
        Q K V dO LSE scale i
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) jLocal)) := by
  simpa [ComparableDType.ge, mul_comm, StreamingAccumulator.blockIndex] using
    causalBlockDS_option_eq Q K V dO LSE scale block i jLocal

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

/-! ### 4D backward math surface -/

/-- Slice a 4D `[B, H, S]` LSE tensor at fixed `(batch, head)`. -/
def sliceBHLSE {B H S : Nat}
    (LSE : TileIndex [B, H, S] → ℝ) (b : Fin B) (h : Fin H) : Fin S → ℝ :=
  fun i => LSE (b, h, i, PUnit.unit)

/-- 4D arbitrary-mask FA-1 backward Real spec. Each `(batch, head)` slice is
independent and uses the corresponding 2D masked backward spec. -/
noncomputable def attentionBackwardReal4DMasked {B H S_q S_k D : Nat}
    (visible : (b : Fin B) → (h : Fin H) → Fin S_q → Fin S_k → Bool)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (dO : TileIndex [B, H, S_q, D] → ℝ)
    (LSE : TileIndex [B, H, S_q] → ℝ) (scale : ℝ) :
    Grads4D B H S_q S_k D :=
  { dQ := fun (b, h, i, d, _) =>
      (attentionBackwardRealMasked
        (visible b h)
        (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
        (sliceBH dO b h) (sliceBHLSE LSE b h) scale).dQ
        (i, d, PUnit.unit)
    dK := fun (b, h, j, d, _) =>
      (attentionBackwardRealMasked
        (visible b h)
        (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
        (sliceBH dO b h) (sliceBHLSE LSE b h) scale).dK
        (j, d, PUnit.unit)
    dV := fun (b, h, j, d, _) =>
      (attentionBackwardRealMasked
        (visible b h)
        (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
        (sliceBH dO b h) (sliceBHLSE LSE b h) scale).dV
        (j, d, PUnit.unit) }

/-- 4D non-causal FA-1 backward Real spec. -/
noncomputable def attentionBackwardReal4D {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (dO : TileIndex [B, H, S_q, D] → ℝ)
    (LSE : TileIndex [B, H, S_q] → ℝ) (scale : ℝ) :
    Grads4D B H S_q S_k D :=
  { dQ := fun (b, h, i, d, _) =>
      (attentionBackwardReal
        (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
        (sliceBH dO b h) (sliceBHLSE LSE b h) scale).dQ
        (i, d, PUnit.unit)
    dK := fun (b, h, j, d, _) =>
      (attentionBackwardReal
        (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
        (sliceBH dO b h) (sliceBHLSE LSE b h) scale).dK
        (j, d, PUnit.unit)
    dV := fun (b, h, j, d, _) =>
      (attentionBackwardReal
        (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
        (sliceBH dO b h) (sliceBHLSE LSE b h) scale).dV
        (j, d, PUnit.unit) }

/-- 4D causal FA-1 backward Real spec. -/
noncomputable def attentionBackwardReal4DCausal {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (dO : TileIndex [B, H, S_q, D] → ℝ)
    (LSE : TileIndex [B, H, S_q] → ℝ) (scale : ℝ) :
    Grads4D B H S_q S_k D :=
  attentionBackwardReal4DMasked
    (fun _ _ i j => decide (j.val ≤ i.val)) Q K V dO LSE scale

/-! ### Boundary backward math surface -/

/-- Boundary query-row visibility for a padded `M`-row query block. -/
def queryBoundaryVisible {S_q S_k M : Nat} (qStart : Nat) :
    Fin M → Fin S_k → Bool :=
  fun i _ => decide (qStart * M + i.val < S_q)

/-- Boundary + causal query/key visibility for a padded `M`-row query block. -/
def queryBoundaryCausalVisible {S_q S_k M : Nat} (qStart : Nat) :
    Fin M → Fin S_k → Bool :=
  fun i j => decide (qStart * M + i.val < S_q ∧ j.val ≤ qStart * M + i.val)

/-- Boundary-padded Q block. Out-of-range query rows are represented as zero;
the boundary visibility predicate makes them contribute zero to all backward
sums. -/
noncomputable def boundaryQBlock {S_q D M : Nat} (qStart : Nat)
    (Q : TileIndex [S_q, D] → ℝ) : TileIndex [M, D] → ℝ :=
  fun idx =>
    if h : qStart * M + idx.1.val < S_q then
      Q (⟨qStart * M + idx.1.val, h⟩, idx.2.1, PUnit.unit)
    else
      0

/-- Boundary-padded upstream gradient block. -/
noncomputable def boundaryDOBlock {S_q D M : Nat} (qStart : Nat)
    (dO : TileIndex [S_q, D] → ℝ) : TileIndex [M, D] → ℝ :=
  fun idx =>
    if h : qStart * M + idx.1.val < S_q then
      dO (⟨qStart * M + idx.1.val, h⟩, idx.2.1, PUnit.unit)
    else
      0

/-- Boundary-padded LSE block. Out-of-range rows are irrelevant because
`queryBoundaryVisible` masks them off. -/
noncomputable def boundaryLSEBlock {S_q M : Nat} (qStart : Nat)
    (LSE : Fin S_q → ℝ) : Fin M → ℝ :=
  fun i =>
    if h : qStart * M + i.val < S_q then
      LSE ⟨qStart * M + i.val, h⟩
    else
      0

/-- Non-causal boundary backward spec for a padded query block. -/
noncomputable def attentionBackwardRealBoundary {S_q S_k D M : Nat}
    (qStart : Nat)
    (Q : TileIndex [S_q, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ)
    (dO : TileIndex [S_q, D] → ℝ) (LSE : Fin S_q → ℝ) (scale : ℝ) :
    Grads M S_k D :=
  attentionBackwardRealMasked
    (queryBoundaryVisible (S_q := S_q) (S_k := S_k) (M := M) qStart)
    (boundaryQBlock qStart Q) K V (boundaryDOBlock qStart dO)
    (boundaryLSEBlock qStart LSE) scale

/-- Causal boundary backward spec for a padded query block. -/
noncomputable def attentionBackwardRealCausalBoundary {S_q S_k D M : Nat}
    (qStart : Nat)
    (Q : TileIndex [S_q, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ)
    (dO : TileIndex [S_q, D] → ℝ) (LSE : Fin S_q → ℝ) (scale : ℝ) :
    Grads M S_k D :=
  attentionBackwardRealMasked
    (queryBoundaryCausalVisible (S_q := S_q) (S_k := S_k) (M := M) qStart)
    (boundaryQBlock qStart Q) K V (boundaryDOBlock qStart dO)
    (boundaryLSEBlock qStart LSE) scale

/-- Boundary backward is the generic masked backward spec with the
query-boundary visibility predicate. -/
theorem attentionBackwardRealBoundary_eq_masked {S_q S_k D M : Nat}
    (qStart : Nat)
    (Q : TileIndex [S_q, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ)
    (dO : TileIndex [S_q, D] → ℝ) (LSE : Fin S_q → ℝ) (scale : ℝ) :
    attentionBackwardRealBoundary qStart Q K V dO LSE scale =
      attentionBackwardRealMasked
        (queryBoundaryVisible (S_q := S_q) (S_k := S_k) (M := M) qStart)
        (boundaryQBlock qStart Q) K V (boundaryDOBlock qStart dO)
        (boundaryLSEBlock qStart LSE) scale := rfl

/-- Causal boundary backward is the generic masked backward spec with the
query-boundary + causal visibility predicate. -/
theorem attentionBackwardRealCausalBoundary_eq_masked {S_q S_k D M : Nat}
    (qStart : Nat)
    (Q : TileIndex [S_q, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ)
    (dO : TileIndex [S_q, D] → ℝ) (LSE : Fin S_q → ℝ) (scale : ℝ) :
    attentionBackwardRealCausalBoundary qStart Q K V dO LSE scale =
      attentionBackwardRealMasked
        (queryBoundaryCausalVisible (S_q := S_q) (S_k := S_k) (M := M) qStart)
        (boundaryQBlock qStart Q) K V (boundaryDOBlock qStart dO)
        (boundaryLSEBlock qStart LSE) scale := rfl

/-- D-tail boundary backward spec: compute the boundary spec at block width
`Bd`, with logical `[*, D]` inputs zero-padded outside `D`. -/
noncomputable def attentionBackwardRealBoundaryD {S_q S_k D Bd M : Nat}
    (qStart : Nat)
    (Q : TileIndex [S_q, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ)
    (dO : TileIndex [S_q, D] → ℝ) (LSE : Fin S_q → ℝ) (scale : ℝ) :
    Grads M S_k Bd :=
  attentionBackwardRealBoundary (D := Bd) qStart
    (padHeadD (Bd := Bd) Q)
    (padHeadD (Bd := Bd) K)
    (padHeadD (Bd := Bd) V)
    (padHeadD (Bd := Bd) dO)
    LSE scale

/-- D-tail causal-boundary backward spec: compute the causal-boundary spec at
block width `Bd`, with logical `[*, D]` inputs zero-padded outside `D`. -/
noncomputable def attentionBackwardRealCausalBoundaryD {S_q S_k D Bd M : Nat}
    (qStart : Nat)
    (Q : TileIndex [S_q, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ)
    (dO : TileIndex [S_q, D] → ℝ) (LSE : Fin S_q → ℝ) (scale : ℝ) :
    Grads M S_k Bd :=
  attentionBackwardRealCausalBoundary (D := Bd) qStart
    (padHeadD (Bd := Bd) Q)
    (padHeadD (Bd := Bd) K)
    (padHeadD (Bd := Bd) V)
    (padHeadD (Bd := Bd) dO)
    LSE scale

theorem attentionBackwardRealBoundaryD_eq_padded {S_q S_k D Bd M : Nat}
    (qStart : Nat)
    (Q : TileIndex [S_q, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ)
    (dO : TileIndex [S_q, D] → ℝ) (LSE : Fin S_q → ℝ) (scale : ℝ) :
    attentionBackwardRealBoundaryD (M := M) (Bd := Bd) qStart Q K V dO LSE scale =
      attentionBackwardRealBoundary (D := Bd) qStart
        (padHeadD (Bd := Bd) Q)
        (padHeadD (Bd := Bd) K)
        (padHeadD (Bd := Bd) V)
        (padHeadD (Bd := Bd) dO)
        LSE scale := rfl

theorem attentionBackwardRealCausalBoundaryD_eq_padded {S_q S_k D Bd M : Nat}
    (qStart : Nat)
    (Q : TileIndex [S_q, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ)
    (dO : TileIndex [S_q, D] → ℝ) (LSE : Fin S_q → ℝ) (scale : ℝ) :
    attentionBackwardRealCausalBoundaryD (M := M) (Bd := Bd) qStart Q K V dO LSE scale =
      attentionBackwardRealCausalBoundary (D := Bd) qStart
        (padHeadD (Bd := Bd) Q)
        (padHeadD (Bd := Bd) K)
        (padHeadD (Bd := Bd) V)
        (padHeadD (Bd := Bd) dO)
        LSE scale := rfl

/-- 4D arbitrary-mask spec, `dQ` slice equation. -/
@[simp] theorem attentionBackwardReal4DMasked_dQ_slice {B H S_q S_k D : Nat}
    (visible : (b : Fin B) → (h : Fin H) → Fin S_q → Fin S_k → Bool)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (dO : TileIndex [B, H, S_q, D] → ℝ)
    (LSE : TileIndex [B, H, S_q] → ℝ) (scale : ℝ)
    (b : Fin B) (h : Fin H) (idx : TileIndex [S_q, D]) :
    (attentionBackwardReal4DMasked visible Q K V dO LSE scale).dQ
        (b, h, idx.1, idx.2.1, PUnit.unit) =
      (attentionBackwardRealMasked
        (visible b h)
        (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
        (sliceBH dO b h) (sliceBHLSE LSE b h) scale).dQ idx := rfl

/-- 4D arbitrary-mask spec, `dK` slice equation. -/
@[simp] theorem attentionBackwardReal4DMasked_dK_slice {B H S_q S_k D : Nat}
    (visible : (b : Fin B) → (h : Fin H) → Fin S_q → Fin S_k → Bool)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (dO : TileIndex [B, H, S_q, D] → ℝ)
    (LSE : TileIndex [B, H, S_q] → ℝ) (scale : ℝ)
    (b : Fin B) (h : Fin H) (idx : TileIndex [S_k, D]) :
    (attentionBackwardReal4DMasked visible Q K V dO LSE scale).dK
        (b, h, idx.1, idx.2.1, PUnit.unit) =
      (attentionBackwardRealMasked
        (visible b h)
        (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
        (sliceBH dO b h) (sliceBHLSE LSE b h) scale).dK idx := rfl

/-- 4D arbitrary-mask spec, `dV` slice equation. -/
@[simp] theorem attentionBackwardReal4DMasked_dV_slice {B H S_q S_k D : Nat}
    (visible : (b : Fin B) → (h : Fin H) → Fin S_q → Fin S_k → Bool)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (dO : TileIndex [B, H, S_q, D] → ℝ)
    (LSE : TileIndex [B, H, S_q] → ℝ) (scale : ℝ)
    (b : Fin B) (h : Fin H) (idx : TileIndex [S_k, D]) :
    (attentionBackwardReal4DMasked visible Q K V dO LSE scale).dV
        (b, h, idx.1, idx.2.1, PUnit.unit) =
      (attentionBackwardRealMasked
        (visible b h)
        (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
        (sliceBH dO b h) (sliceBHLSE LSE b h) scale).dV idx := rfl

/-- Bundled 4D arbitrary-mask slice equation for all three backward outputs. -/
theorem attentionBackwardReal4DMasked_slice {B H S_q S_k D : Nat}
    (visible : (b : Fin B) → (h : Fin H) → Fin S_q → Fin S_k → Bool)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (dO : TileIndex [B, H, S_q, D] → ℝ)
    (LSE : TileIndex [B, H, S_q] → ℝ) (scale : ℝ)
    (b : Fin B) (h : Fin H) :
    (∀ idx : TileIndex [S_q, D],
      (attentionBackwardReal4DMasked visible Q K V dO LSE scale).dQ
          (b, h, idx.1, idx.2.1, PUnit.unit) =
        (attentionBackwardRealMasked
          (visible b h)
          (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
          (sliceBH dO b h) (sliceBHLSE LSE b h) scale).dQ idx) ∧
    (∀ idx : TileIndex [S_k, D],
      (attentionBackwardReal4DMasked visible Q K V dO LSE scale).dK
          (b, h, idx.1, idx.2.1, PUnit.unit) =
        (attentionBackwardRealMasked
          (visible b h)
          (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
          (sliceBH dO b h) (sliceBHLSE LSE b h) scale).dK idx) ∧
    (∀ idx : TileIndex [S_k, D],
      (attentionBackwardReal4DMasked visible Q K V dO LSE scale).dV
          (b, h, idx.1, idx.2.1, PUnit.unit) =
        (attentionBackwardRealMasked
          (visible b h)
          (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
          (sliceBH dO b h) (sliceBHLSE LSE b h) scale).dV idx) := by
  exact ⟨fun _ => rfl, fun _ => rfl, fun _ => rfl⟩

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
      StreamingAccumulator.blockIndex Bk numKVBlocks block.val
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
          StreamingAccumulator.blockIndex Bk numKVBlocks block.val
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
  refine (Finset.sum_equiv (StreamingAccumulator.blockIndexEquiv Bk numKVBlocks) ?_ ?_).symm
  · intro _; simp
  · intro j _
    rw [StreamingAccumulator.blockIndex_blockIndexEquiv]

/-- Mask-aware contribution to `dQ` from one KV block. -/
noncomputable def dQBlockContributionMasked {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [M, D]) : ℝ :=
  scale * Finset.univ.sum fun jLocal : Fin Bk =>
    let j : Fin (Bk * numKVBlocks) :=
      StreamingAccumulator.blockIndex Bk numKVBlocks block.val
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
          refine (Finset.sum_equiv (StreamingAccumulator.blockIndexEquiv Bk numKVBlocks) ?_ ?_).symm
          · intro _; simp
          · intro j _
            rw [StreamingAccumulator.blockIndex_blockIndexEquiv]

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

/-! ### Boundary multi-block `dQ` bridges -/

/-- Non-causal query-boundary specialization of a block-local masked `dQ`
contribution. -/
noncomputable def dQBlockContributionBoundary {S_q D M Bk numKVBlocks : Nat}
    (qStart : Nat)
    (Q : TileIndex [S_q, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [S_q, D] → ℝ) (LSE : Fin S_q → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [M, D]) : ℝ :=
  dQBlockContributionMasked
    (queryBoundaryVisible (S_q := S_q) (S_k := Bk * numKVBlocks) (M := M) qStart)
    (boundaryQBlock qStart Q) K V (boundaryDOBlock qStart dO)
    (boundaryLSEBlock qStart LSE) scale block idx

/-- Causal query-boundary specialization of a block-local masked `dQ`
contribution. -/
noncomputable def dQBlockContributionCausalBoundary {S_q D M Bk numKVBlocks : Nat}
    (qStart : Nat)
    (Q : TileIndex [S_q, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [S_q, D] → ℝ) (LSE : Fin S_q → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [M, D]) : ℝ :=
  dQBlockContributionMasked
    (queryBoundaryCausalVisible (S_q := S_q) (S_k := Bk * numKVBlocks) (M := M) qStart)
    (boundaryQBlock qStart Q) K V (boundaryDOBlock qStart dO)
    (boundaryLSEBlock qStart LSE) scale block idx

/-- Summing boundary block-local `dQ` contributions over all KV blocks recovers
the query-boundary closed-form `dQ`. -/
theorem dQBlockContributionBoundary_sum_eq_attentionBackwardRealBoundary
    {S_q D M Bk numKVBlocks : Nat}
    (qStart : Nat)
    (Q : TileIndex [S_q, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [S_q, D] → ℝ) (LSE : Fin S_q → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    (Finset.univ.sum fun block : Fin numKVBlocks =>
      dQBlockContributionBoundary qStart Q K V dO LSE scale block idx) =
      (attentionBackwardRealBoundary qStart Q K V dO LSE scale).dQ idx := by
  simp [dQBlockContributionBoundary, attentionBackwardRealBoundary,
    dQBlockContributionMasked_sum_eq_attentionBackwardRealMasked_impl]

/-- Summing causal-boundary block-local `dQ` contributions over all KV blocks
recovers the causal query-boundary closed-form `dQ`. -/
theorem dQBlockContributionCausalBoundary_sum_eq_attentionBackwardRealCausalBoundary
    {S_q D M Bk numKVBlocks : Nat}
    (qStart : Nat)
    (Q : TileIndex [S_q, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [S_q, D] → ℝ) (LSE : Fin S_q → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    (Finset.univ.sum fun block : Fin numKVBlocks =>
      dQBlockContributionCausalBoundary qStart Q K V dO LSE scale block idx) =
      (attentionBackwardRealCausalBoundary qStart Q K V dO LSE scale).dQ idx := by
  simp [dQBlockContributionCausalBoundary, attentionBackwardRealCausalBoundary,
    dQBlockContributionMasked_sum_eq_attentionBackwardRealMasked_impl]

/-! ### Boundary D-tail multi-block `dQ` bridges -/

/-- D-tail query-boundary specialization of a block-local `dQ` contribution. -/
noncomputable def dQBlockContributionBoundaryD {S_q D Bd M Bk numKVBlocks : Nat}
    (qStart : Nat)
    (Q : TileIndex [S_q, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [S_q, D] → ℝ) (LSE : Fin S_q → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [M, Bd]) : ℝ :=
  dQBlockContributionBoundary (D := Bd) qStart
    (padHeadD (Bd := Bd) Q)
    (padHeadD (Bd := Bd) K)
    (padHeadD (Bd := Bd) V)
    (padHeadD (Bd := Bd) dO)
    LSE scale block idx

/-- D-tail causal query-boundary specialization of a block-local `dQ`
contribution. -/
noncomputable def dQBlockContributionCausalBoundaryD {S_q D Bd M Bk numKVBlocks : Nat}
    (qStart : Nat)
    (Q : TileIndex [S_q, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [S_q, D] → ℝ) (LSE : Fin S_q → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [M, Bd]) : ℝ :=
  dQBlockContributionCausalBoundary (D := Bd) qStart
    (padHeadD (Bd := Bd) Q)
    (padHeadD (Bd := Bd) K)
    (padHeadD (Bd := Bd) V)
    (padHeadD (Bd := Bd) dO)
    LSE scale block idx

/-- Summing D-tail boundary block-local `dQ` contributions over all KV blocks
recovers the D-tail query-boundary closed-form `dQ`. -/
theorem dQBlockContributionBoundaryD_sum_eq_attentionBackwardRealBoundaryD
    {S_q D Bd M Bk numKVBlocks : Nat}
    (qStart : Nat)
    (Q : TileIndex [S_q, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [S_q, D] → ℝ) (LSE : Fin S_q → ℝ) (scale : ℝ)
    (idx : TileIndex [M, Bd]) :
    (Finset.univ.sum fun block : Fin numKVBlocks =>
      dQBlockContributionBoundaryD qStart Q K V dO LSE scale block idx) =
      (attentionBackwardRealBoundaryD (M := M) (Bd := Bd)
        qStart Q K V dO LSE scale).dQ idx := by
  simp [dQBlockContributionBoundaryD, attentionBackwardRealBoundaryD,
    dQBlockContributionBoundary_sum_eq_attentionBackwardRealBoundary]

/-- Summing D-tail causal-boundary block-local `dQ` contributions over all KV
blocks recovers the D-tail causal query-boundary closed-form `dQ`. -/
theorem dQBlockContributionCausalBoundaryD_sum_eq_attentionBackwardRealCausalBoundaryD
    {S_q D Bd M Bk numKVBlocks : Nat}
    (qStart : Nat)
    (Q : TileIndex [S_q, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [S_q, D] → ℝ) (LSE : Fin S_q → ℝ) (scale : ℝ)
    (idx : TileIndex [M, Bd]) :
    (Finset.univ.sum fun block : Fin numKVBlocks =>
      dQBlockContributionCausalBoundaryD qStart Q K V dO LSE scale block idx) =
      (attentionBackwardRealCausalBoundaryD (M := M) (Bd := Bd)
        qStart Q K V dO LSE scale).dQ idx := by
  simp [dQBlockContributionCausalBoundaryD, attentionBackwardRealCausalBoundaryD,
    dQBlockContributionCausalBoundary_sum_eq_attentionBackwardRealCausalBoundary]

/-- Raw kernel-form bridge for the causal block-local `dQ_part` computation.

Unlike `dQ_block_tile_some_eq_dQBlockContributionCausal`, this theorem starts
from the actual `tl.where`/`tl.exp`/`tl.sum` option expression emitted by the
causal atomic kernel prefix. -/
theorem dQ_block_kernel_tile_some_eq_dQBlockContributionCausal
    {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [M, D]) :
    Option.map (fun a : ℝ => a * scale)
      (@Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ (fun jLocal : Fin Bk =>
        Option.map
          (fun a : ℝ =>
            a * K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                (by have := block.isLt; omega) jLocal, idx.2.1, PUnit.unit))
          (Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (WithBot.realExp
              (Option.map (fun a : ℝ => a - LSE idx.1)
                (if ComparableDType.nat.ge idx.1.val (block.val * Bk + jLocal.val) =
                    Bool.true then
                  some ((Finset.univ.sum fun d : Fin D =>
                    Q (idx.1, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                        (by have := block.isLt; omega) jLocal, d, PUnit.unit)) * scale)
                else none)))
            (Option.map
              (fun b : ℝ =>
                dP V dO idx.1
                    (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                      (by have := block.isLt; omega) jLocal) - b)
              (@Finset.sum (Fin (Bk * numKVBlocks)) (WithBot ℝ) _ Finset.univ
                (fun j' : Fin (Bk * numKVBlocks) =>
                  (Option.map (fun p : ℝ => p * dP V dO idx.1 j')
                    (WithBot.realExp
                      (Option.map (fun a : ℝ => a - LSE idx.1)
                        (if ComparableDType.nat.ge idx.1.val j'.val = Bool.true then
                          some ((Finset.univ.sum fun d : Fin D =>
                            Q (idx.1, d, PUnit.unit) * K (j', d, PUnit.unit)) * scale)
                        else none))) : WithBot ℝ))))))) =
      some (dQBlockContributionCausal Q K V dO LSE scale block idx) := by
  have hterm :
      (fun jLocal : Fin Bk =>
        Option.map
          (fun a : ℝ =>
            a * K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                (by have := block.isLt; omega) jLocal, idx.2.1, PUnit.unit))
          (Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (WithBot.realExp
              (Option.map (fun a : ℝ => a - LSE idx.1)
                (if ComparableDType.nat.ge idx.1.val (block.val * Bk + jLocal.val) =
                    Bool.true then
                  some ((Finset.univ.sum fun d : Fin D =>
                    Q (idx.1, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                        (by have := block.isLt; omega) jLocal, d, PUnit.unit)) * scale)
                else none)))
            (Option.map
              (fun b : ℝ =>
                dP V dO idx.1
                    (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                      (by have := block.isLt; omega) jLocal) - b)
              (@Finset.sum (Fin (Bk * numKVBlocks)) (WithBot ℝ) _ Finset.univ
                (fun j' : Fin (Bk * numKVBlocks) =>
                  (Option.map (fun p : ℝ => p * dP V dO idx.1 j')
                    (WithBot.realExp
                      (Option.map (fun a : ℝ => a - LSE idx.1)
                        (if ComparableDType.nat.ge idx.1.val j'.val = Bool.true then
                          some ((Finset.univ.sum fun d : Fin D =>
                            Q (idx.1, d, PUnit.unit) * K (j', d, PUnit.unit)) * scale)
                        else none))) : WithBot ℝ)))))) =
      (fun jLocal : Fin Bk =>
        (some
          (dSMasked (fun i j => decide (j.val ≤ i.val))
            Q K V dO LSE scale idx.1
            (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) jLocal) *
            K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) jLocal, idx.2.1, PUnit.unit)) :
          WithBot ℝ)) := by
    funext jLocal
    rw [causalBlockDS_kernel_option_eq]
    rfl
  rw [hterm]
  rw [WithBot.sum_someTerm_eq_some]
  simp [dQBlockContributionCausal, dQBlockContributionMasked]
  ring_nf

/-- Prop-if normal form of `dQ_block_kernel_tile_some_eq_dQBlockContributionCausal`.

The concrete causal prefix proof simplifies `ComparableDType.nat.ge ... =
Bool.true` into the proposition `j ≤ i`.  This wrapper keeps the expensive raw
bridge available after that normalization. -/
theorem dQ_block_kernel_prop_tile_some_eq_dQBlockContributionCausal
    {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [M, D]) :
    Option.map (fun a : ℝ => a * scale)
      (@Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ (fun jLocal : Fin Bk =>
        Option.map
          (fun a : ℝ =>
            a * K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                (by have := block.isLt; omega) jLocal, idx.2.1, PUnit.unit))
          (Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (WithBot.realExp
              (Option.map (fun a : ℝ => a - LSE idx.1)
                (if block.val * Bk + jLocal.val ≤ idx.1.val then
                  some ((Finset.univ.sum fun d : Fin D =>
                    Q (idx.1, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                        (by have := block.isLt; omega) jLocal, d, PUnit.unit)) * scale)
                else none)))
            (Option.map
              (fun b : ℝ =>
                dP V dO idx.1
                    (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                      (by have := block.isLt; omega) jLocal) - b)
              (@Finset.sum (Fin (Bk * numKVBlocks)) (WithBot ℝ) _ Finset.univ
                (fun j' : Fin (Bk * numKVBlocks) =>
                  (Option.map (fun p : ℝ => p * dP V dO idx.1 j')
                    (WithBot.realExp
                      (Option.map (fun a : ℝ => a - LSE idx.1)
                        (if j'.val ≤ idx.1.val then
                          some ((Finset.univ.sum fun d : Fin D =>
                            Q (idx.1, d, PUnit.unit) * K (j', d, PUnit.unit)) * scale)
                        else none))) : WithBot ℝ))))))) =
      some (dQBlockContributionCausal Q K V dO LSE scale block idx) := by
  simpa [ComparableDType.ge] using
    dQ_block_kernel_tile_some_eq_dQBlockContributionCausal
      Q K V dO LSE scale block idx

/-- Prop-if unscaled sum bridge for the causal block-local `dQ_part`.

This is the exact residual shape left after the concrete prefix proof simplifies
the outer `Option.map (fun a => a * scale)`: the `WithBot` sum is already known
to be `some`, and the remaining arithmetic only distributes the final scale. -/
theorem dQ_block_kernel_prop_sum_some_eq_dQBlockContributionCausal_unscaled
    {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [M, D]) :
    @Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ (fun jLocal : Fin Bk =>
      Option.map
        (fun a : ℝ =>
          a * K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) jLocal, idx.2.1, PUnit.unit))
        (Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
          (WithBot.realExp
            (Option.map (fun a : ℝ => a - LSE idx.1)
              (if block.val * Bk + jLocal.val ≤ idx.1.val then
                some (scale * (Finset.univ.sum fun d : Fin D =>
                  Q (idx.1, d, PUnit.unit) *
                    K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                      (by have := block.isLt; omega) jLocal, d, PUnit.unit)))
              else none)))
          (Option.map
            (fun b : ℝ =>
              dP V dO idx.1
                  (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                    (by have := block.isLt; omega) jLocal) - b)
            (@Finset.sum (Fin (Bk * numKVBlocks)) (WithBot ℝ) _ Finset.univ
              (fun j' : Fin (Bk * numKVBlocks) =>
                (Option.map (fun p : ℝ => p * dP V dO idx.1 j')
                  (WithBot.realExp
                    (Option.map (fun a : ℝ => a - LSE idx.1)
                      (if j'.val ≤ idx.1.val then
                        some (scale * (Finset.univ.sum fun d : Fin D =>
                          Q (idx.1, d, PUnit.unit) * K (j', d, PUnit.unit)))
                      else none))) : WithBot ℝ)))))) =
      some (Finset.univ.sum fun jLocal : Fin Bk =>
        dSMasked (fun i j => decide (j.val ≤ i.val))
          Q K V dO LSE scale idx.1
          (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) jLocal) *
        K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) jLocal, idx.2.1, PUnit.unit)) := by
  have hterm :
      (fun jLocal : Fin Bk =>
        Option.map
          (fun a : ℝ =>
            a * K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                (by have := block.isLt; omega) jLocal, idx.2.1, PUnit.unit))
          (Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (WithBot.realExp
              (Option.map (fun a : ℝ => a - LSE idx.1)
                (if block.val * Bk + jLocal.val ≤ idx.1.val then
                  some (scale * (Finset.univ.sum fun d : Fin D =>
                    Q (idx.1, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                        (by have := block.isLt; omega) jLocal, d, PUnit.unit)))
                else none)))
            (Option.map
              (fun b : ℝ =>
                dP V dO idx.1
                    (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                      (by have := block.isLt; omega) jLocal) - b)
              (@Finset.sum (Fin (Bk * numKVBlocks)) (WithBot ℝ) _ Finset.univ
                (fun j' : Fin (Bk * numKVBlocks) =>
                  (Option.map (fun p : ℝ => p * dP V dO idx.1 j')
                    (WithBot.realExp
                      (Option.map (fun a : ℝ => a - LSE idx.1)
                        (if j'.val ≤ idx.1.val then
                          some (scale * (Finset.univ.sum fun d : Fin D =>
                            Q (idx.1, d, PUnit.unit) * K (j', d, PUnit.unit)))
                        else none))) : WithBot ℝ)))))) =
      (fun jLocal : Fin Bk =>
        (some
          (dSMasked (fun i j => decide (j.val ≤ i.val))
            Q K V dO LSE scale idx.1
            (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) jLocal) *
            K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) jLocal, idx.2.1, PUnit.unit)) :
          WithBot ℝ)) := by
    funext jLocal
    rw [causalBlockDS_option_eq]
    rfl
  rw [hterm]
  rw [WithBot.sum_someTerm_eq_some]

/-- Raw kernel-form bridge for the causal block-local `dK_block` computation. -/
theorem dK_block_kernel_tile_some_eq_attentionBackwardRealCausal
    {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [Bk, D]) :
    Option.map (fun a : ℝ => a * scale)
      (@Finset.sum (Fin M) (WithBot ℝ) _ Finset.univ (fun i : Fin M =>
        Option.map (fun a : ℝ => a * Q (i, idx.2.1, PUnit.unit))
          (Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (WithBot.realExp
              (Option.map (fun a : ℝ => a - LSE i)
                (if ComparableDType.nat.ge i.val (block.val * Bk + idx.1.val) =
                    Bool.true then
                  some ((Finset.univ.sum fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                        (by have := block.isLt; omega) idx.1, d, PUnit.unit)) * scale)
                else none)))
            (Option.map
              (fun b : ℝ =>
                dP V dO i
                    (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                      (by have := block.isLt; omega) idx.1) - b)
              (@Finset.sum (Fin (Bk * numKVBlocks)) (WithBot ℝ) _ Finset.univ
                (fun j' : Fin (Bk * numKVBlocks) =>
                  (Option.map (fun p : ℝ => p * dP V dO i j')
                    (WithBot.realExp
                      (Option.map (fun a : ℝ => a - LSE i)
                        (if ComparableDType.nat.ge i.val j'.val = Bool.true then
                          some ((Finset.univ.sum fun d : Fin D =>
                            Q (i, d, PUnit.unit) * K (j', d, PUnit.unit)) * scale)
                        else none))) : WithBot ℝ))))))) =
      some ((attentionBackwardRealCausal Q K V dO LSE scale).dK
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) := by
  have hterm :
      (fun i : Fin M =>
        Option.map (fun a : ℝ => a * Q (i, idx.2.1, PUnit.unit))
          (Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (WithBot.realExp
              (Option.map (fun a : ℝ => a - LSE i)
                (if ComparableDType.nat.ge i.val (block.val * Bk + idx.1.val) =
                    Bool.true then
                  some ((Finset.univ.sum fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                        (by have := block.isLt; omega) idx.1, d, PUnit.unit)) * scale)
                else none)))
            (Option.map
              (fun b : ℝ =>
                dP V dO i
                    (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                      (by have := block.isLt; omega) idx.1) - b)
              (@Finset.sum (Fin (Bk * numKVBlocks)) (WithBot ℝ) _ Finset.univ
                (fun j' : Fin (Bk * numKVBlocks) =>
                  (Option.map (fun p : ℝ => p * dP V dO i j')
                    (WithBot.realExp
                      (Option.map (fun a : ℝ => a - LSE i)
                        (if ComparableDType.nat.ge i.val j'.val = Bool.true then
                          some ((Finset.univ.sum fun d : Fin D =>
                            Q (i, d, PUnit.unit) * K (j', d, PUnit.unit)) * scale)
                        else none))) : WithBot ℝ)))))) =
      (fun i : Fin M =>
        (some
          (dSMasked (fun i j => decide (j.val ≤ i.val))
            Q K V dO LSE scale i
            (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) idx.1) *
            Q (i, idx.2.1, PUnit.unit)) : WithBot ℝ)) := by
    funext i
    rw [causalBlockDS_kernel_option_eq]
    rfl
  rw [hterm]
  rw [WithBot.sum_someTerm_eq_some]
  simp [attentionBackwardRealCausal, attentionBackwardRealMasked]
  ring_nf

/-- Prop-if normal form of `dK_block_kernel_tile_some_eq_attentionBackwardRealCausal`. -/
theorem dK_block_kernel_prop_tile_some_eq_attentionBackwardRealCausal
    {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [Bk, D]) :
    Option.map (fun a : ℝ => a * scale)
      (@Finset.sum (Fin M) (WithBot ℝ) _ Finset.univ (fun i : Fin M =>
        Option.map (fun a : ℝ => a * Q (i, idx.2.1, PUnit.unit))
          (Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (WithBot.realExp
              (Option.map (fun a : ℝ => a - LSE i)
                (if block.val * Bk + idx.1.val ≤ i.val then
                  some ((Finset.univ.sum fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                        (by have := block.isLt; omega) idx.1, d, PUnit.unit)) * scale)
                else none)))
            (Option.map
              (fun b : ℝ =>
                dP V dO i
                    (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                      (by have := block.isLt; omega) idx.1) - b)
              (@Finset.sum (Fin (Bk * numKVBlocks)) (WithBot ℝ) _ Finset.univ
                (fun j' : Fin (Bk * numKVBlocks) =>
                  (Option.map (fun p : ℝ => p * dP V dO i j')
                    (WithBot.realExp
                      (Option.map (fun a : ℝ => a - LSE i)
                        (if j'.val ≤ i.val then
                          some ((Finset.univ.sum fun d : Fin D =>
                            Q (i, d, PUnit.unit) * K (j', d, PUnit.unit)) * scale)
                        else none))) : WithBot ℝ))))))) =
      some ((attentionBackwardRealCausal Q K V dO LSE scale).dK
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) := by
  simpa [ComparableDType.ge] using
    dK_block_kernel_tile_some_eq_attentionBackwardRealCausal
      Q K V dO LSE scale block idx

/-- Prop-if unscaled sum bridge for the causal block-local `dK_block`. -/
theorem dK_block_kernel_prop_sum_some_eq_attentionBackwardRealCausal_unscaled
    {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [Bk, D]) :
    @Finset.sum (Fin M) (WithBot ℝ) _ Finset.univ (fun i : Fin M =>
      Option.map (fun a : ℝ => a * Q (i, idx.2.1, PUnit.unit))
        (Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
          (WithBot.realExp
            (Option.map (fun a : ℝ => a - LSE i)
              (if block.val * Bk + idx.1.val ≤ i.val then
                some (scale * (Finset.univ.sum fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                      (by have := block.isLt; omega) idx.1, d, PUnit.unit)))
              else none)))
          (Option.map
            (fun b : ℝ =>
              dP V dO i
                  (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                    (by have := block.isLt; omega) idx.1) - b)
            (@Finset.sum (Fin (Bk * numKVBlocks)) (WithBot ℝ) _ Finset.univ
              (fun j' : Fin (Bk * numKVBlocks) =>
                (Option.map (fun p : ℝ => p * dP V dO i j')
                  (WithBot.realExp
                    (Option.map (fun a : ℝ => a - LSE i)
                      (if j'.val ≤ i.val then
                        some (scale * (Finset.univ.sum fun d : Fin D =>
                          Q (i, d, PUnit.unit) * K (j', d, PUnit.unit)))
                      else none))) : WithBot ℝ)))))) =
      some (Finset.univ.sum fun i : Fin M =>
        dSMasked (fun i j => decide (j.val ≤ i.val))
          Q K V dO LSE scale i
          (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1) *
        Q (i, idx.2.1, PUnit.unit)) := by
  have hterm :
      (fun i : Fin M =>
        Option.map (fun a : ℝ => a * Q (i, idx.2.1, PUnit.unit))
          (Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (WithBot.realExp
              (Option.map (fun a : ℝ => a - LSE i)
                (if block.val * Bk + idx.1.val ≤ i.val then
                  some (scale * (Finset.univ.sum fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                        (by have := block.isLt; omega) idx.1, d, PUnit.unit)))
                else none)))
            (Option.map
              (fun b : ℝ =>
                dP V dO i
                    (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                      (by have := block.isLt; omega) idx.1) - b)
              (@Finset.sum (Fin (Bk * numKVBlocks)) (WithBot ℝ) _ Finset.univ
                (fun j' : Fin (Bk * numKVBlocks) =>
                  (Option.map (fun p : ℝ => p * dP V dO i j')
                    (WithBot.realExp
                      (Option.map (fun a : ℝ => a - LSE i)
                        (if j'.val ≤ i.val then
                          some (scale * (Finset.univ.sum fun d : Fin D =>
                            Q (i, d, PUnit.unit) * K (j', d, PUnit.unit)))
                        else none))) : WithBot ℝ)))))) =
      (fun i : Fin M =>
        (some
          (dSMasked (fun i j => decide (j.val ≤ i.val))
            Q K V dO LSE scale i
            (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) idx.1) *
            Q (i, idx.2.1, PUnit.unit)) : WithBot ℝ)) := by
    funext i
    rw [causalBlockDS_option_eq]
    rfl
  rw [hterm]
  rw [WithBot.sum_someTerm_eq_some]

/-- Raw kernel-form bridge for the causal block-local `dV_block` computation. -/
theorem dV_block_kernel_tile_some_eq_attentionBackwardRealCausal
    {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [Bk, D]) :
    @Finset.sum (Fin M) (WithBot ℝ) _ Finset.univ (fun i : Fin M =>
      (Option.map (fun a : ℝ => a * dO (i, idx.2.1, PUnit.unit))
        (WithBot.realExp
          (Option.map (fun a : ℝ => a - LSE i)
            (if ComparableDType.nat.ge i.val (block.val * Bk + idx.1.val) =
                Bool.true then
              some ((Finset.univ.sum fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                    (by have := block.isLt; omega) idx.1, d, PUnit.unit)) * scale)
            else none))) : WithBot ℝ)) =
      some ((attentionBackwardRealCausal Q K V dO LSE scale).dV
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) := by
  have hterm :
      (fun i : Fin M =>
        (Option.map (fun a : ℝ => a * dO (i, idx.2.1, PUnit.unit))
          (WithBot.realExp
            (Option.map (fun a : ℝ => a - LSE i)
              (if ComparableDType.nat.ge i.val (block.val * Bk + idx.1.val) =
                  Bool.true then
                some ((Finset.univ.sum fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                      (by have := block.isLt; omega) idx.1, d, PUnit.unit)) * scale)
              else none))) : WithBot ℝ)) =
      (fun i : Fin M =>
        (some
          (probabilityMasked (fun i j => decide (j.val ≤ i.val))
            Q K LSE scale i
            (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) idx.1) *
            dO (i, idx.2.1, PUnit.unit)) : WithBot ℝ)) := by
    funext i
    have hprob :
        WithBot.realExp
          (Option.map (fun a : ℝ => a - LSE i)
            (if ComparableDType.nat.ge i.val (block.val * Bk + idx.1.val) =
                Bool.true then
              some ((Finset.univ.sum fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                    (by have := block.isLt; omega) idx.1, d, PUnit.unit)) * scale)
            else none)) =
          some (probabilityMasked (fun i j => decide (j.val ≤ i.val))
            Q K LSE scale i
            (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) idx.1)) := by
      simpa [ComparableDType.ge, mul_comm, StreamingAccumulator.blockIndex] using
        maskedProbability_exp_eq
          (fun (i : Fin M) (j : Fin (Bk * numKVBlocks)) => decide (j.val ≤ i.val))
          Q K LSE scale i
          (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1)
    rw [hprob]
    rfl
  rw [hterm]
  rw [WithBot.sum_someTerm_eq_some]
  simp [attentionBackwardRealCausal, attentionBackwardRealMasked]

/-- Prop-if normal form of `dV_block_kernel_tile_some_eq_attentionBackwardRealCausal`. -/
theorem dV_block_kernel_prop_tile_some_eq_attentionBackwardRealCausal
    {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [Bk, D]) :
    @Finset.sum (Fin M) (WithBot ℝ) _ Finset.univ (fun i : Fin M =>
      (Option.map (fun a : ℝ => a * dO (i, idx.2.1, PUnit.unit))
        (WithBot.realExp
          (Option.map (fun a : ℝ => a - LSE i)
            (if block.val * Bk + idx.1.val ≤ i.val then
              some ((Finset.univ.sum fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                    (by have := block.isLt; omega) idx.1, d, PUnit.unit)) * scale)
            else none))) : WithBot ℝ)) =
      some ((attentionBackwardRealCausal Q K V dO LSE scale).dV
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) := by
  simpa [ComparableDType.ge] using
    dV_block_kernel_tile_some_eq_attentionBackwardRealCausal
      Q K V dO LSE scale block idx

/-- Prop-if unscaled sum bridge for the causal block-local `dV_block`. -/
theorem dV_block_kernel_prop_sum_some_eq_attentionBackwardRealCausal_unscaled
    {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K _V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [Bk, D]) :
    @Finset.sum (Fin M) (WithBot ℝ) _ Finset.univ (fun i : Fin M =>
      (Option.map (fun a : ℝ => a * dO (i, idx.2.1, PUnit.unit))
        (WithBot.realExp
          (Option.map (fun a : ℝ => a - LSE i)
            (if block.val * Bk + idx.1.val ≤ i.val then
              some (scale * (Finset.univ.sum fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                    (by have := block.isLt; omega) idx.1, d, PUnit.unit)))
            else none))) : WithBot ℝ)) =
      some (Finset.univ.sum fun i : Fin M =>
        probabilityMasked (fun i j => decide (j.val ≤ i.val))
          Q K LSE scale i
          (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1) *
        dO (i, idx.2.1, PUnit.unit)) := by
  have hterm :
      (fun i : Fin M =>
        (Option.map (fun a : ℝ => a * dO (i, idx.2.1, PUnit.unit))
          (WithBot.realExp
            (Option.map (fun a : ℝ => a - LSE i)
              (if block.val * Bk + idx.1.val ≤ i.val then
                some (scale * (Finset.univ.sum fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                      (by have := block.isLt; omega) idx.1, d, PUnit.unit)))
              else none))) : WithBot ℝ)) =
      (fun i : Fin M =>
        (some
          (probabilityMasked (fun i j => decide (j.val ≤ i.val))
            Q K LSE scale i
            (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) idx.1) *
            dO (i, idx.2.1, PUnit.unit)) : WithBot ℝ)) := by
    funext i
    have hprob :
        WithBot.realExp
          (Option.map (fun a : ℝ => a - LSE i)
            (if block.val * Bk + idx.1.val ≤ i.val then
              some (scale * (Finset.univ.sum fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                    (by have := block.isLt; omega) idx.1, d, PUnit.unit)))
            else none)) =
          some (probabilityMasked (fun i j => decide (j.val ≤ i.val))
            Q K LSE scale i
            (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) idx.1)) := by
      simpa [StreamingAccumulator.blockIndex] using
        maskedProbability_exp_eq
          (fun (i : Fin M) (j : Fin (Bk * numKVBlocks)) => decide (j.val ≤ i.val))
          Q K LSE scale i
          (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1)
    rw [hprob]
    rfl
  rw [hterm]
  rw [WithBot.sum_someTerm_eq_some]

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
            StreamingAccumulator.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) idx.2.1
          dS Q K V dO LSE scale idx.1 j)
        (Tile.ofReal fun idx : TileIndex [Bk, D] =>
          let j : Fin (Bk * numKVBlocks) :=
            StreamingAccumulator.blockIndex Bk numKVBlocks block.val
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
              StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                (by have := block.isLt; omega) idx.2.1
            dS Q K V dO LSE scale idx.1 j))
        (Tile.ofReal Q)).data idx) =
      some ((attentionBackwardReal Q K V dO LSE scale).dK
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
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
            StreamingAccumulator.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) idx.2.1
          probability Q K LSE scale idx.1 j))
      (Tile.ofReal dO)).data idx =
      some ((attentionBackwardReal Q K V dO LSE scale).dV
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) := by
  rw [Tile.dot_nil_data]
  simp [Tile.transpose, Tile.ofReal, attentionBackwardReal_dV]

/-- Masked tile-level bridge for one block-local
`dQ_part = dS_masked_block · K_block · scale`. -/
theorem dQ_block_tile_some_eq_dQBlockContributionMasked {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [M, D]) :
    Option.map (fun a : ℝ => a * scale)
      ((Tile.dot []
        (Tile.ofReal fun idx : TileIndex [M, Bk] =>
          let j : Fin (Bk * numKVBlocks) :=
            StreamingAccumulator.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) idx.2.1
          dSMasked visible Q K V dO LSE scale idx.1 j)
        (Tile.ofReal fun idx : TileIndex [Bk, D] =>
          let j : Fin (Bk * numKVBlocks) :=
            StreamingAccumulator.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) idx.1
          K (j, idx.2.1, PUnit.unit))).data idx) =
      some (dQBlockContributionMasked visible Q K V dO LSE scale block idx) := by
  rw [Tile.dot_nil_data]
  simp [Tile.ofReal, dQBlockContributionMasked]
  ring

/-- Masked tile-level bridge for one block-local `dK_block`. -/
theorem dK_block_tile_some_eq_attentionBackwardRealMasked {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [Bk, D]) :
    Option.map (fun a : ℝ => a * scale)
      ((Tile.dot []
        (Tile.transpose []
          (Tile.ofReal fun idx : TileIndex [M, Bk] =>
            let j : Fin (Bk * numKVBlocks) :=
              StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                (by have := block.isLt; omega) idx.2.1
            dSMasked visible Q K V dO LSE scale idx.1 j))
        (Tile.ofReal Q)).data idx) =
      some ((attentionBackwardRealMasked visible Q K V dO LSE scale).dK
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) := by
  rw [Tile.dot_nil_data]
  simp [Tile.transpose, Tile.ofReal, attentionBackwardRealMasked]
  ring

/-- Masked tile-level bridge for one block-local `dV_block`. -/
theorem dV_block_tile_some_eq_attentionBackwardRealMasked {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [Bk, D]) :
    (Tile.dot []
      (Tile.transpose []
        (Tile.ofReal fun idx : TileIndex [M, Bk] =>
          let j : Fin (Bk * numKVBlocks) :=
            StreamingAccumulator.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) idx.2.1
          probabilityMasked visible Q K LSE scale idx.1 j))
      (Tile.ofReal dO)).data idx =
      some ((attentionBackwardRealMasked visible Q K V dO LSE scale).dV
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) := by
  rw [Tile.dot_nil_data]
  simp [Tile.transpose, Tile.ofReal, attentionBackwardRealMasked]

/-- Causal tile-level bridge for one block-local
`dQ_part = dS_causal_block · K_block · scale`. -/
theorem dQ_block_tile_some_eq_dQBlockContributionCausal {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [M, D]) :
    Option.map (fun a : ℝ => a * scale)
      ((Tile.dot []
        (Tile.ofReal fun idx : TileIndex [M, Bk] =>
          let j : Fin (Bk * numKVBlocks) :=
            StreamingAccumulator.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) idx.2.1
          dSMasked (fun i j => decide (j.val ≤ i.val)) Q K V dO LSE scale idx.1 j)
        (Tile.ofReal fun idx : TileIndex [Bk, D] =>
          let j : Fin (Bk * numKVBlocks) :=
            StreamingAccumulator.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) idx.1
          K (j, idx.2.1, PUnit.unit))).data idx) =
      some (dQBlockContributionCausal Q K V dO LSE scale block idx) := by
  simpa [dQBlockContributionCausal] using
    dQ_block_tile_some_eq_dQBlockContributionMasked
      (fun i j => decide (j.val ≤ i.val)) Q K V dO LSE scale block idx

/-- Causal tile-level bridge for one block-local `dK_block`. -/
theorem dK_block_tile_some_eq_attentionBackwardRealCausal {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [Bk, D]) :
    Option.map (fun a : ℝ => a * scale)
      ((Tile.dot []
        (Tile.transpose []
          (Tile.ofReal fun idx : TileIndex [M, Bk] =>
            let j : Fin (Bk * numKVBlocks) :=
              StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                (by have := block.isLt; omega) idx.2.1
            dSMasked (fun i j => decide (j.val ≤ i.val)) Q K V dO LSE scale idx.1 j))
        (Tile.ofReal Q)).data idx) =
      some ((attentionBackwardRealCausal Q K V dO LSE scale).dK
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) := by
  simpa [attentionBackwardRealCausal] using
    dK_block_tile_some_eq_attentionBackwardRealMasked
      (fun i j => decide (j.val ≤ i.val)) Q K V dO LSE scale block idx

/-- Causal tile-level bridge for one block-local `dV_block`. -/
theorem dV_block_tile_some_eq_attentionBackwardRealCausal {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) (idx : TileIndex [Bk, D]) :
    (Tile.dot []
      (Tile.transpose []
        (Tile.ofReal fun idx : TileIndex [M, Bk] =>
          let j : Fin (Bk * numKVBlocks) :=
            StreamingAccumulator.blockIndex Bk numKVBlocks block.val
              (by have := block.isLt; omega) idx.2.1
          probabilityMasked (fun i j => decide (j.val ≤ i.val)) Q K LSE scale idx.1 j))
      (Tile.ofReal dO)).data idx =
      some ((attentionBackwardRealCausal Q K V dO LSE scale).dV
        (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
          (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit)) := by
  simpa [attentionBackwardRealCausal] using
    dV_block_tile_some_eq_attentionBackwardRealMasked
      (fun i j => decide (j.val ≤ i.val)) Q K V dO LSE scale block idx

/-- Bundled arbitrary-mask block-local bridge surface for the atomic backward
prefix proof: the per-block `dQ_part`, `dK_block`, and `dV_block` tile
computations match the masked Real backward semantics. -/
theorem maskedBackward_block_tile_bridges_complete {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) :
    (∀ idx : TileIndex [M, D],
      Option.map (fun a : ℝ => a * scale)
        ((Tile.dot []
          (Tile.ofReal fun idx : TileIndex [M, Bk] =>
            let j : Fin (Bk * numKVBlocks) :=
              StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                (by have := block.isLt; omega) idx.2.1
            dSMasked visible Q K V dO LSE scale idx.1 j)
          (Tile.ofReal fun idx : TileIndex [Bk, D] =>
            let j : Fin (Bk * numKVBlocks) :=
              StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                (by have := block.isLt; omega) idx.1
            K (j, idx.2.1, PUnit.unit))).data idx) =
        some (dQBlockContributionMasked visible Q K V dO LSE scale block idx)) ∧
    (∀ idx : TileIndex [Bk, D],
      Option.map (fun a : ℝ => a * scale)
        ((Tile.dot []
          (Tile.transpose []
            (Tile.ofReal fun idx : TileIndex [M, Bk] =>
              let j : Fin (Bk * numKVBlocks) :=
                StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                  (by have := block.isLt; omega) idx.2.1
              dSMasked visible Q K V dO LSE scale idx.1 j))
          (Tile.ofReal Q)).data idx) =
        some ((attentionBackwardRealMasked visible Q K V dO LSE scale).dK
          (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) ∧
    (∀ idx : TileIndex [Bk, D],
      (Tile.dot []
        (Tile.transpose []
          (Tile.ofReal fun idx : TileIndex [M, Bk] =>
            let j : Fin (Bk * numKVBlocks) :=
              StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                (by have := block.isLt; omega) idx.2.1
            probabilityMasked visible Q K LSE scale idx.1 j))
        (Tile.ofReal dO)).data idx =
        some ((attentionBackwardRealMasked visible Q K V dO LSE scale).dV
          (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) := by
  exact ⟨
    dQ_block_tile_some_eq_dQBlockContributionMasked visible Q K V dO LSE scale block,
    dK_block_tile_some_eq_attentionBackwardRealMasked visible Q K V dO LSE scale block,
    dV_block_tile_some_eq_attentionBackwardRealMasked visible Q K V dO LSE scale block⟩

/-- Bundled causal block-local bridge surface for the causal atomic backward
prefix proof: the per-block `dQ_part`, `dK_block`, and `dV_block` tile
computations match the causal masked Real backward semantics. -/
theorem causalBackward_block_tile_bridges_complete {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (block : Fin numKVBlocks) :
    (∀ idx : TileIndex [M, D],
      Option.map (fun a : ℝ => a * scale)
        ((Tile.dot []
          (Tile.ofReal fun idx : TileIndex [M, Bk] =>
            let j : Fin (Bk * numKVBlocks) :=
              StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                (by have := block.isLt; omega) idx.2.1
            dSMasked (fun i j => decide (j.val ≤ i.val))
              Q K V dO LSE scale idx.1 j)
          (Tile.ofReal fun idx : TileIndex [Bk, D] =>
            let j : Fin (Bk * numKVBlocks) :=
              StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                (by have := block.isLt; omega) idx.1
            K (j, idx.2.1, PUnit.unit))).data idx) =
        some (dQBlockContributionCausal Q K V dO LSE scale block idx)) ∧
    (∀ idx : TileIndex [Bk, D],
      Option.map (fun a : ℝ => a * scale)
        ((Tile.dot []
          (Tile.transpose []
            (Tile.ofReal fun idx : TileIndex [M, Bk] =>
              let j : Fin (Bk * numKVBlocks) :=
                StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                  (by have := block.isLt; omega) idx.2.1
              dSMasked (fun i j => decide (j.val ≤ i.val))
                Q K V dO LSE scale idx.1 j))
          (Tile.ofReal Q)).data idx) =
        some ((attentionBackwardRealCausal Q K V dO LSE scale).dK
          (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) ∧
    (∀ idx : TileIndex [Bk, D],
      (Tile.dot []
        (Tile.transpose []
          (Tile.ofReal fun idx : TileIndex [M, Bk] =>
            let j : Fin (Bk * numKVBlocks) :=
              StreamingAccumulator.blockIndex Bk numKVBlocks block.val
                (by have := block.isLt; omega) idx.2.1
            probabilityMasked (fun i j => decide (j.val ≤ i.val))
              Q K LSE scale idx.1 j))
        (Tile.ofReal dO)).data idx =
        some ((attentionBackwardRealCausal Q K V dO LSE scale).dV
          (StreamingAccumulator.blockIndex Bk numKVBlocks block.val
            (by have := block.isLt; omega) idx.1, idx.2.1, PUnit.unit))) := by
  exact ⟨
    dQ_block_tile_some_eq_dQBlockContributionCausal Q K V dO LSE scale block,
    dK_block_tile_some_eq_attentionBackwardRealCausal Q K V dO LSE scale block,
    dV_block_tile_some_eq_attentionBackwardRealCausal Q K V dO LSE scale block⟩

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
  rw [StreamingAccumulator.scaled_data_eq' Q K scale idx.1 idx.2.1]
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

theorem dQ_tile_some_eq_attentionBackwardRealCausal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    Option.map (fun a : ℝ => a * scale)
      ((Tile.dot []
        (Tile.ofReal fun idx : TileIndex [M, S] =>
          dSMasked (fun i j => decide (j.val ≤ i.val)) Q K V dO LSE scale idx.1 idx.2.1)
        (Tile.ofReal K)).data idx) =
      some ((attentionBackwardRealCausal Q K V dO LSE scale).dQ idx) := by
  rw [Tile.dot_nil_data]
  simp [Tile.ofReal, attentionBackwardRealCausal, attentionBackwardRealMasked]
  ring

theorem dK_tile_some_eq_attentionBackwardRealCausal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [S, D]) :
    Option.map (fun a : ℝ => a * scale)
      ((Tile.dot []
        (Tile.transpose []
          (Tile.ofReal fun idx : TileIndex [M, S] =>
            dSMasked (fun i j => decide (j.val ≤ i.val)) Q K V dO LSE scale idx.1 idx.2.1))
        (Tile.ofReal Q)).data idx) =
      some ((attentionBackwardRealCausal Q K V dO LSE scale).dK idx) := by
  rw [Tile.dot_nil_data]
  simp [Tile.transpose, Tile.ofReal, attentionBackwardRealCausal,
    attentionBackwardRealMasked]
  ring

theorem dV_tile_some_eq_attentionBackwardRealCausal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ)
    (idx : TileIndex [S, D]) :
    (Tile.dot []
      (Tile.transpose []
        (Tile.ofReal fun idx : TileIndex [M, S] =>
          probabilityMasked (fun i j => decide (j.val ≤ i.val)) Q K LSE scale idx.1 idx.2.1))
      (Tile.ofReal dO)).data idx =
      some ((attentionBackwardRealCausal Q K V dO LSE scale).dV idx) := by
  rw [Tile.dot_nil_data]
  simp [Tile.transpose, Tile.ofReal, attentionBackwardRealCausal,
    attentionBackwardRealMasked]

/-- Bundled causal tile bridge surface for the causal backward prefix proof. -/
theorem causalBackward_tile_bridges_complete {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (dO : TileIndex [M, D] → ℝ) (LSE : Fin M → ℝ) (scale : ℝ) :
    (∀ idx : TileIndex [M, D],
      Option.map (fun a : ℝ => a * scale)
        ((Tile.dot []
          (Tile.ofReal fun idx : TileIndex [M, S] =>
            dSMasked (fun i j => decide (j.val ≤ i.val))
              Q K V dO LSE scale idx.1 idx.2.1)
          (Tile.ofReal K)).data idx) =
        some ((attentionBackwardRealCausal Q K V dO LSE scale).dQ idx)) ∧
    (∀ idx : TileIndex [S, D],
      Option.map (fun a : ℝ => a * scale)
        ((Tile.dot []
          (Tile.transpose []
            (Tile.ofReal fun idx : TileIndex [M, S] =>
              dSMasked (fun i j => decide (j.val ≤ i.val))
                Q K V dO LSE scale idx.1 idx.2.1))
          (Tile.ofReal Q)).data idx) =
        some ((attentionBackwardRealCausal Q K V dO LSE scale).dK idx)) ∧
    (∀ idx : TileIndex [S, D],
      (Tile.dot []
        (Tile.transpose []
          (Tile.ofReal fun idx : TileIndex [M, S] =>
            probabilityMasked (fun i j => decide (j.val ≤ i.val))
              Q K LSE scale idx.1 idx.2.1))
        (Tile.ofReal dO)).data idx =
        some ((attentionBackwardRealCausal Q K V dO LSE scale).dV idx)) := by
  exact ⟨dQ_tile_some_eq_attentionBackwardRealCausal Q K V dO LSE scale,
    dK_tile_some_eq_attentionBackwardRealCausal Q K V dO LSE scale,
    dV_tile_some_eq_attentionBackwardRealCausal Q K V dO LSE scale⟩

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
