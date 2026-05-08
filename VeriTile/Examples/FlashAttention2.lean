/-
VeriTile.Examples.FlashAttention2

FA-2 forward proof surface.

This file starts Tier 3-B with a separate FA-2-facing module.  The first
surface is the semantically equivalent forward baseline: same 4D view contract
as FA-1 forward, exposed under FA-2 names.  The later FA-2-specific deltas
(work partitioning, delayed rescale, and block-skip control flow) should
refine this baseline rather than changing the user-facing spec.
-/

import VeriTile.Examples.FlashAttention1.Core
import VeriTile.Triton.Math.Softmax
import VeriTile.Triton.Launch.Grid

namespace VeriTile.Examples

open VeriTile.Triton
open VeriTile.Triton.TiledSoftmax
open BigOperators

/-! ## FA-2 delayed-rescale math bridges

FA-2 forward computes block-local softmax fragments and rescales them when a
larger running max is discovered.  These lemmas isolate the algebraic step:
normalizing by a block-local max and then rescaling to the merged max is the
same as normalizing by the merged max directly.
-/

/-- Scalar delayed-rescale identity for the softmax denominator. -/
theorem fa2_delayed_rescale_sum_eq {ι : Type} [Fintype ι]
    (scores : ι → ℝ) (mBlock mNew : ℝ) :
    Real.exp (mBlock - mNew) *
        (Finset.univ.sum fun j : ι => Real.exp (scores j - mBlock)) =
      Finset.univ.sum fun j : ι => Real.exp (scores j - mNew) := by
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl ?_
  intro j _
  rw [← Real.exp_add]
  congr 1
  ring_nf

/-- Weighted delayed-rescale identity for the softmax numerator. -/
theorem fa2_delayed_rescale_weighted_sum_eq {ι : Type} [Fintype ι]
    (scores values : ι → ℝ) (mBlock mNew : ℝ) :
    Real.exp (mBlock - mNew) *
        (Finset.univ.sum fun j : ι => Real.exp (scores j - mBlock) * values j) =
      Finset.univ.sum fun j : ι => Real.exp (scores j - mNew) * values j := by
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl ?_
  intro j _
  rw [← mul_assoc, ← Real.exp_add]
  congr 1
  ring_nf

/-- Two-fragment denominator merge: FA-2 can combine two block-local
denominators by rescaling each to the merged max and adding them. -/
theorem fa2_two_fragment_denominator_merge_eq_flat
    {ι κ : Type} [Fintype ι] [Fintype κ]
    (scoresLeft : ι → ℝ) (scoresRight : κ → ℝ)
    (mLeft mRight mMerged : ℝ) :
    Real.exp (mLeft - mMerged) *
        (Finset.univ.sum fun j : ι => Real.exp (scoresLeft j - mLeft)) +
      Real.exp (mRight - mMerged) *
        (Finset.univ.sum fun j : κ => Real.exp (scoresRight j - mRight)) =
      (Finset.univ.sum fun j : ι => Real.exp (scoresLeft j - mMerged)) +
        (Finset.univ.sum fun j : κ => Real.exp (scoresRight j - mMerged)) := by
  rw [fa2_delayed_rescale_sum_eq scoresLeft mLeft mMerged,
    fa2_delayed_rescale_sum_eq scoresRight mRight mMerged]

/-- Two-fragment numerator merge: FA-2 can combine two block-local weighted
numerators by rescaling each to the merged max and adding them. -/
theorem fa2_two_fragment_numerator_merge_eq_flat
    {ι κ : Type} [Fintype ι] [Fintype κ]
    (scoresLeft valuesLeft : ι → ℝ) (scoresRight valuesRight : κ → ℝ)
    (mLeft mRight mMerged : ℝ) :
    Real.exp (mLeft - mMerged) *
        (Finset.univ.sum fun j : ι => Real.exp (scoresLeft j - mLeft) * valuesLeft j) +
      Real.exp (mRight - mMerged) *
        (Finset.univ.sum fun j : κ => Real.exp (scoresRight j - mRight) * valuesRight j) =
      (Finset.univ.sum fun j : ι => Real.exp (scoresLeft j - mMerged) * valuesLeft j) +
        (Finset.univ.sum fun j : κ => Real.exp (scoresRight j - mMerged) * valuesRight j) := by
  rw [fa2_delayed_rescale_weighted_sum_eq scoresLeft valuesLeft mLeft mMerged,
    fa2_delayed_rescale_weighted_sum_eq scoresRight valuesRight mRight mMerged]

/-- Two-fragment attention-output merge: the FA-2 rescale-and-merge ratio is
the same as the flat attention ratio over both fragments.  This is the scalar
per-output-coordinate form; vector outputs use this pointwise for each `D`
coordinate. -/
theorem fa2_two_fragment_attention_ratio_eq_flat
    {ι κ : Type} [Fintype ι] [Fintype κ]
    (scoresLeft valuesLeft : ι → ℝ) (scoresRight valuesRight : κ → ℝ)
    (mLeft mRight mMerged : ℝ) :
    (Real.exp (mLeft - mMerged) *
          (Finset.univ.sum fun j : ι => Real.exp (scoresLeft j - mLeft) * valuesLeft j) +
        Real.exp (mRight - mMerged) *
          (Finset.univ.sum fun j : κ =>
            Real.exp (scoresRight j - mRight) * valuesRight j)) /
      (Real.exp (mLeft - mMerged) *
          (Finset.univ.sum fun j : ι => Real.exp (scoresLeft j - mLeft)) +
        Real.exp (mRight - mMerged) *
          (Finset.univ.sum fun j : κ => Real.exp (scoresRight j - mRight))) =
      ((Finset.univ.sum fun j : ι => Real.exp (scoresLeft j - mMerged) * valuesLeft j) +
        (Finset.univ.sum fun j : κ => Real.exp (scoresRight j - mMerged) * valuesRight j)) /
      ((Finset.univ.sum fun j : ι => Real.exp (scoresLeft j - mMerged)) +
        (Finset.univ.sum fun j : κ => Real.exp (scoresRight j - mMerged))) := by
  rw [fa2_two_fragment_numerator_merge_eq_flat scoresLeft valuesLeft scoresRight valuesRight
      mLeft mRight mMerged,
    fa2_two_fragment_denominator_merge_eq_flat scoresLeft scoresRight mLeft mRight mMerged]

/-- Fully masked blocks contribute zero to the softmax denominator. -/
theorem fa2_masked_sum_eq_zero_of_all_invisible {ι : Type} [Fintype ι]
    (visible : ι → Bool) (scores : ι → ℝ) (m : ℝ)
    (hInvisible : ∀ j : ι, visible j = Bool.false) :
    (Finset.univ.sum fun j : ι =>
      if visible j then Real.exp (scores j - m) else 0) = 0 := by
  simp [hInvisible]

/-- Fully masked blocks contribute zero to the softmax numerator. -/
theorem fa2_masked_weighted_sum_eq_zero_of_all_invisible {ι : Type} [Fintype ι]
    (visible : ι → Bool) (scores values : ι → ℝ) (m : ℝ)
    (hInvisible : ∀ j : ι, visible j = Bool.false) :
    (Finset.univ.sum fun j : ι =>
      if visible j then Real.exp (scores j - m) * values j else 0) = 0 := by
  simp [hInvisible]

/-! ## FA-2 two-fragment forward spec

This is the first partitioned-forward spec surface: split the KV domain into
two `Bk`-sized fragments, compute per-fragment denominator/numerator pairs
under local maxima, then merge them with FA-2's delayed rescale.  The theorem
below states that this two-fragment FA-2 output is exactly the flat
`attentionReal` output over the same two-block KV domain.
-/

private theorem two_block0_le : 0 + 1 ≤ 2 := by omega
private theorem two_block1_le : 1 + 1 ≤ 2 := by omega

/-- Left-fragment denominator contribution. -/
noncomputable def fa2TwoBlockDenomLeft {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * 2, D] → ℝ)
    (scale : ℝ) (mBlock : ℝ) (i : Fin M) : ℝ :=
  Finset.univ.sum fun j : Fin Bk =>
    Real.exp
      (FA1Math.scaledScore Q K scale i
        (FA1Math.blockIndex Bk 2 0 two_block0_le j) - mBlock)

/-- Right-fragment denominator contribution. -/
noncomputable def fa2TwoBlockDenomRight {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * 2, D] → ℝ)
    (scale : ℝ) (mBlock : ℝ) (i : Fin M) : ℝ :=
  Finset.univ.sum fun j : Fin Bk =>
    Real.exp
      (FA1Math.scaledScore Q K scale i
        (FA1Math.blockIndex Bk 2 1 two_block1_le j) - mBlock)

/-- Left-fragment numerator contribution. -/
noncomputable def fa2TwoBlockNumerLeft {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * 2, D] → ℝ)
    (scale : ℝ) (mBlock : ℝ) (idx : TileIndex [M, D]) : ℝ :=
  Finset.univ.sum fun j : Fin Bk =>
    let key := FA1Math.blockIndex Bk 2 0 two_block0_le j
    Real.exp (FA1Math.scaledScore Q K scale idx.1 key - mBlock) *
      V (key, idx.2.1, PUnit.unit)

/-- Right-fragment numerator contribution. -/
noncomputable def fa2TwoBlockNumerRight {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * 2, D] → ℝ)
    (scale : ℝ) (mBlock : ℝ) (idx : TileIndex [M, D]) : ℝ :=
  Finset.univ.sum fun j : Fin Bk =>
    let key := FA1Math.blockIndex Bk 2 1 two_block1_le j
    Real.exp (FA1Math.scaledScore Q K scale idx.1 key - mBlock) *
      V (key, idx.2.1, PUnit.unit)

/-- Two-fragment FA-2 forward output for one `(query, head-dim)` coordinate. -/
noncomputable def fa2TwoBlockForwardSpec {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * 2, D] → ℝ)
    (scale : ℝ) (mLeft mRight mMerged : ℝ) (idx : TileIndex [M, D]) : ℝ :=
  (Real.exp (mLeft - mMerged) *
        fa2TwoBlockNumerLeft Q K V scale mLeft idx +
      Real.exp (mRight - mMerged) *
        fa2TwoBlockNumerRight Q K V scale mRight idx) /
    (Real.exp (mLeft - mMerged) *
        fa2TwoBlockDenomLeft Q K scale mLeft idx.1 +
      Real.exp (mRight - mMerged) *
        fa2TwoBlockDenomRight Q K scale mRight idx.1)

private theorem fa2_two_block_denominator_common_shift {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * 2, D] → ℝ)
    (scale : ℝ) (m : ℝ) (i : Fin M) :
    fa2TwoBlockDenomLeft Q K scale m i +
      fa2TwoBlockDenomRight Q K scale m i =
      Real.exp (0 - m) * FA1Math.lFree Q K scale 2 (le_refl 2) i := by
  rw [FA1Math.lFree_succ Q K scale 1 two_block1_le i,
    FA1Math.lFree_succ Q K scale 0 two_block0_le i,
    FA1Math.lFree_zero]
  simp only [zero_add, fa2TwoBlockDenomLeft, fa2TwoBlockDenomRight]
  rw [mul_add]
  rw [Finset.mul_sum, Finset.mul_sum]
  apply congrArg₂ HAdd.hAdd
  · refine Finset.sum_congr rfl ?_
    intro j _
    rw [← Real.exp_add]
    congr 1
    ring_nf
  · refine Finset.sum_congr rfl ?_
    intro j _
    rw [← Real.exp_add]
    congr 1
    ring_nf

private theorem fa2_two_block_numerator_common_shift {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * 2, D] → ℝ)
    (scale : ℝ) (m : ℝ) (idx : TileIndex [M, D]) :
    fa2TwoBlockNumerLeft Q K V scale m idx +
      fa2TwoBlockNumerRight Q K V scale m idx =
      Real.exp (0 - m) * FA1Math.oFree Q K V scale 2 (le_refl 2) idx := by
  rw [FA1Math.oFree_succ Q K V scale 1 two_block1_le idx,
    FA1Math.oFree_succ Q K V scale 0 two_block0_le idx,
    FA1Math.oFree_zero]
  simp only [zero_add, fa2TwoBlockNumerLeft, fa2TwoBlockNumerRight]
  rw [mul_add]
  rw [Finset.mul_sum, Finset.mul_sum]
  apply congrArg₂ HAdd.hAdd
  · refine Finset.sum_congr rfl ?_
    intro j _
    rw [← mul_assoc, ← Real.exp_add]
    congr 1
    ring_nf
  · refine Finset.sum_congr rfl ?_
    intro j _
    rw [← mul_assoc, ← Real.exp_add]
    congr 1
    ring_nf

/-- Two-fragment FA-2 forward equals the flat FA-1 attention spec over the
same two-block KV domain.  This is the partitioned-forward math bridge that
the executable FA-2 kernel will target. -/
theorem fa2_two_block_forward_eq_attentionReal {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * 2, D] → ℝ)
    (scale : ℝ) (mLeft mRight mMerged : ℝ) (idx : TileIndex [M, D])
    (hlFree : FA1Math.lFree Q K scale 2 (le_refl 2) idx.1 ≠ 0) :
    fa2TwoBlockForwardSpec Q K V scale mLeft mRight mMerged idx =
      attentionReal Q K V scale idx := by
  unfold fa2TwoBlockForwardSpec fa2TwoBlockDenomLeft fa2TwoBlockDenomRight
    fa2TwoBlockNumerLeft fa2TwoBlockNumerRight
  rw [fa2_two_fragment_attention_ratio_eq_flat
    (scoresLeft := fun j : Fin Bk =>
      FA1Math.scaledScore Q K scale idx.1
        (FA1Math.blockIndex Bk 2 0 two_block0_le j))
    (valuesLeft := fun j : Fin Bk =>
      V (FA1Math.blockIndex Bk 2 0 two_block0_le j, idx.2.1, PUnit.unit))
    (scoresRight := fun j : Fin Bk =>
      FA1Math.scaledScore Q K scale idx.1
        (FA1Math.blockIndex Bk 2 1 two_block1_le j))
    (valuesRight := fun j : Fin Bk =>
      V (FA1Math.blockIndex Bk 2 1 two_block1_le j, idx.2.1, PUnit.unit))
    (mLeft := mLeft) (mRight := mRight) (mMerged := mMerged)]
  change
    (fa2TwoBlockNumerLeft Q K V scale mMerged idx +
        fa2TwoBlockNumerRight Q K V scale mMerged idx) /
      (fa2TwoBlockDenomLeft Q K scale mMerged idx.1 +
        fa2TwoBlockDenomRight Q K scale mMerged idx.1) =
      attentionReal Q K V scale idx
  rw [fa2_two_block_numerator_common_shift Q K V scale mMerged idx,
    fa2_two_block_denominator_common_shift Q K scale mMerged idx.1,
    mul_div_mul_left _ _ (Real.exp_ne_zero (0 - mMerged))]
  exact FA1Math.oFree_div_lFree_eq_attentionReal Q K V scale idx hlFree

/-- 4D slice-facing version of the two-fragment FA-2 forward bridge.  This is
the theorem shape that later FA-2 kernels can use at a fixed
`(batch, head, query, d)` coordinate. -/
theorem fa2_two_block_forward_eq_attentionReal4D
    {B H S_q D Bk : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, Bk * 2, D] → ℝ)
    (scale : ℝ) (b : Fin B) (h : Fin H) (i : Fin S_q) (d : Fin D)
    (mLeft mRight mMerged : ℝ)
    (hlFree :
      FA1Math.lFree (sliceBH Q b h) (sliceBH K b h) scale 2 (le_refl 2) i ≠ 0) :
    fa2TwoBlockForwardSpec (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
        scale mLeft mRight mMerged (i, d, PUnit.unit) =
      attentionReal4D Q K V scale (b, h, i, d, PUnit.unit) := by
  rw [attentionReal4D_slice]
  exact fa2_two_block_forward_eq_attentionReal
    (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
    scale mLeft mRight mMerged (i, d, PUnit.unit) hlFree

/-- Headline spec-level FA-1/FA-2 equality for the two-fragment forward
surface: the flat FA-1 4D attention spec equals the delayed-rescale FA-2
two-block partitioned spec at the same coordinate. -/
theorem fa1_eq_fa2_two_block_forward4D
    {B H S_q D Bk : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, Bk * 2, D] → ℝ)
    (scale : ℝ) (b : Fin B) (h : Fin H) (i : Fin S_q) (d : Fin D)
    (mLeft mRight mMerged : ℝ)
    (hlFree :
      FA1Math.lFree (sliceBH Q b h) (sliceBH K b h) scale 2 (le_refl 2) i ≠ 0) :
    attentionReal4D Q K V scale (b, h, i, d, PUnit.unit) =
      fa2TwoBlockForwardSpec (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
        scale mLeft mRight mMerged (i, d, PUnit.unit) := by
  exact (fa2_two_block_forward_eq_attentionReal4D
    Q K V scale b h i d mLeft mRight mMerged hlFree).symm

/-! ## FA-2 scalar score-row max producer

This producer computes the row max consumed by the scalar fragment summary and
fused scalar forward stages.
-/

def fa2ScalarScoreMaxKernel (scoreReg mReg : RegionName) (Bk : Nat) :
    ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(Bk) + tl.arange(0, $(Bk))
  scores := tl.load($(scoreReg) + offs)
  m      := tl.max(scores, axis = 0)
  tl.store($(mReg) + pid, m)
}

/-- Correctness of the executable scalar score-row max producer. -/
theorem fa2ScalarScoreMaxKernel_correct_view
    (scoreReg mReg : RegionName) (Bk : Nat) (hBk : 0 < Bk)
    (s : BlockState) (scores : Fin Bk → ℝ)
    (hScores : InputLoadedAt s scoreReg Bk scores) :
    ComputeCorrect.Realizes
      (kernel := fa2ScalarScoreMaxKernel scoreReg mReg Bk)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar mReg s.pid)
      (expected := fun _ : PUnit => tileMax hBk scores) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hBk.ne'
  simp [exec, fa2ScalarScoreMaxKernel, stepStmts, stepStmt, evalOp,
        Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim,
        TileShape.eraseAxis, TileShape.insertAxisIndex, tileMax,
        BlockState.writeMemTyped_real] at hExec ⊢
  subst s'
  intro _
  simp [ComputeCorrect.WriteMap.scalar]
  unfold InputLoadedAt at hScores
  simpa [NumericDType.add, NumericDType.mul, TiledLogSumExp.blockMax, BlockState.pid] using
    congrArg
      (fun f : Fin (n + 1) → ℝ =>
        Finset.univ.sup'
          (⟨⟨0, hBk⟩, Finset.mem_univ _⟩ :
            (Finset.univ : Finset (Fin (n + 1))).Nonempty) f)
      (funext hScores)

/-- State-parametric handoff for a score-row max producer.  If a later consumer
state has the same scalar pid and agrees with the max producer final state on
`mReg`, then the consumer can read the produced row max. -/
theorem fa2ScalarScoreMaxKernel_loaded_of_agrees
    (scoreReg mReg : RegionName) (Bk : Nat) (hBk : 0 < Bk)
    (s sMax consumer : BlockState) (scores : Fin Bk → ℝ)
    (hExec : exec (fa2ScalarScoreMaxKernel scoreReg mReg Bk).toAlgKernel s = some sMax)
    (hPid : consumer.pid = s.pid)
    (hMem : ∀ offset, consumer.readMem mReg offset = sMax.readMem mReg offset)
    (hScores : InputLoadedAt s scoreReg Bk scores) :
    consumer.readMem mReg consumer.pid = tileMax hBk scores := by
  have hview := fa2ScalarScoreMaxKernel_correct_view
    scoreReg mReg Bk hBk s scores hScores
  unfold ComputeCorrect.Realizes ComputeKernel.ExecCorrect
    ComputeKernel.ComputeCorrect ComputeKernel.ProjectedCorrect
    ComputeKernel.AlgorithmCorrect Kernel.Correct at hview
  rcases hview with ⟨_, hview⟩
  simp at hview
  have hOut := hview s sMax hExec rfl PUnit.unit
  calc
    consumer.readMem mReg consumer.pid = sMax.readMem mReg consumer.pid := hMem _
    _ = sMax.readMem mReg s.pid := by rw [hPid]
    _ = tileMax hBk scores := by
      simpa [ComputeCorrect.WriteMap.scalar] using hOut

/-! ## FA-2 scalar value-fragment staging producer

This producer stages one `Bk`-lane value fragment for a fixed output coordinate
dimension `d`.  It is the value-buffer counterpart to the score-fragment
handoff used by the fused scalar consumer.
-/

def fa2ScalarValueFragmentKernel
    (vReg valueReg : RegionName) (D Bk keyBlock d : Nat) :
    ComputeKernel := triton {
  pid      := tl.program_id(0)
  offs     := tl.arange(0, $(Bk))
  src      := (($(keyBlock) * $(Bk) + offs) * $(D)) + $(d)
  values   := tl.load($(vReg) + src)
  dst      := pid * $(Bk) + offs
  tl.store($(valueReg) + dst, values)
}

/-- Correctness of the executable scalar value-fragment staging producer. -/
theorem fa2ScalarValueFragmentKernel_correct_view
    {D Bk N : Nat}
    (vReg valueReg : RegionName)
    (keyBlock : Nat) (hKeyBlock : keyBlock < N) (d : Fin D)
    (s : BlockState) (V : TileIndex [Bk * N, D] → ℝ)
    (hV : ∀ idx : TileIndex [Bk, D],
      s.readMem vReg ((keyBlock * Bk + idx.1.val) * D + idx.2.1.val) =
        V (FA1Math.blockIndex Bk N keyBlock
            (Nat.succ_le_iff.mpr hKeyBlock) idx.1,
          idx.2.1, PUnit.unit)) :
    ComputeCorrect.Realizes
      (kernel := fa2ScalarValueFragmentKernel vReg valueReg D Bk keyBlock d.val)
      (initialState := s)
      (write := fun j : Fin Bk =>
        some (valueReg, s.pid * Bk + j.val))
      (expected := fun j : Fin Bk =>
        V (FA1Math.blockIndex Bk N keyBlock
            (Nat.succ_le_iff.mpr hKeyBlock) j,
          d, PUnit.unit)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  simp [exec, fa2ScalarValueFragmentKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, NumericDType.add, NumericDType.mul,
        BlockState.writeMemTyped_real] at hExec ⊢
  subst s'
  intro j
  have hInj :
      Function.Injective
        (fun idx : TileIndex [Bk] => s.pid * Bk + idx.1.val) :=
    injective_offset_singleton _
  rw [BlockState.scatter_readback_nd _ _ _ hInj (j, PUnit.unit)]
  have hRead := hV (j, d, PUnit.unit)
  simpa [FA1Math.blockIndex, Nat.mul_assoc, Nat.add_assoc] using hRead

/-- State-parametric handoff for a staged value fragment.  If a later consumer
state has the same scalar pid and agrees with the value producer final state on
`valueReg`, the staged values are available as `InputLoadedAt`. -/
theorem fa2ScalarValueFragmentKernel_loaded_of_agrees
    {D Bk N : Nat}
    (vReg valueReg : RegionName)
    (keyBlock : Nat) (hKeyBlock : keyBlock < N) (d : Fin D)
    (s sValue consumer : BlockState) (V : TileIndex [Bk * N, D] → ℝ)
    (hExec :
      exec (fa2ScalarValueFragmentKernel vReg valueReg D Bk keyBlock d.val).toAlgKernel s =
        some sValue)
    (hPid : consumer.pid = s.pid)
    (hMem : ∀ offset, consumer.readMem valueReg offset = sValue.readMem valueReg offset)
    (hV : ∀ idx : TileIndex [Bk, D],
      s.readMem vReg ((keyBlock * Bk + idx.1.val) * D + idx.2.1.val) =
        V (FA1Math.blockIndex Bk N keyBlock
            (Nat.succ_le_iff.mpr hKeyBlock) idx.1,
          idx.2.1, PUnit.unit)) :
    InputLoadedAt consumer valueReg Bk
      (fun j : Fin Bk =>
        V (FA1Math.blockIndex Bk N keyBlock
            (Nat.succ_le_iff.mpr hKeyBlock) j,
          d, PUnit.unit)) := by
  have hview := fa2ScalarValueFragmentKernel_correct_view
    vReg valueReg keyBlock hKeyBlock d s V hV
  unfold ComputeCorrect.Realizes ComputeKernel.ExecCorrect
    ComputeKernel.ComputeCorrect ComputeKernel.ProjectedCorrect
    ComputeKernel.AlgorithmCorrect Kernel.Correct at hview
  rcases hview with ⟨_, hview⟩
  simp at hview
  intro j
  have hOut := hview s sValue hExec rfl j
  calc
    consumer.readMem valueReg (consumer.pid * Bk + j.val) =
        sValue.readMem valueReg (consumer.pid * Bk + j.val) := hMem _
    _ = sValue.readMem valueReg (s.pid * Bk + j.val) := by rw [hPid]
    _ = V (FA1Math.blockIndex Bk N keyBlock
            (Nat.succ_le_iff.mpr hKeyBlock) j,
          d, PUnit.unit) := by
      simpa using hOut

/-- Two-block value-fragment handoff for the scalar fused-forward consumer. -/
theorem fa2ScalarValueFragmentKernel_twoBlock_loaded_of_agrees
    {D Bk : Nat}
    (vReg valueLeftReg valueRightReg : RegionName) (d : Fin D)
    (s sLeft sRight consumer : BlockState)
    (V : TileIndex [Bk * 2, D] → ℝ)
    (hExecLeft :
      exec (fa2ScalarValueFragmentKernel vReg valueLeftReg D Bk 0 d.val).toAlgKernel s =
        some sLeft)
    (hExecRight :
      exec (fa2ScalarValueFragmentKernel vReg valueRightReg D Bk 1 d.val).toAlgKernel s =
        some sRight)
    (hPid : consumer.pid = s.pid)
    (hValueLeftMem :
      ∀ offset, consumer.readMem valueLeftReg offset = sLeft.readMem valueLeftReg offset)
    (hValueRightMem :
      ∀ offset, consumer.readMem valueRightReg offset = sRight.readMem valueRightReg offset)
    (hV : ∀ idx : TileIndex [Bk * 2, D],
      s.readMem vReg (idx.1.val * D + idx.2.1.val) = V idx) :
    InputLoadedAt consumer valueLeftReg Bk
      (fun j : Fin Bk =>
        V (FA1Math.blockIndex Bk 2 0 two_block0_le j, d, PUnit.unit)) ∧
    InputLoadedAt consumer valueRightReg Bk
      (fun j : Fin Bk =>
        V (FA1Math.blockIndex Bk 2 1 two_block1_le j, d, PUnit.unit)) := by
  have hVLeft : ∀ idx : TileIndex [Bk, D],
      s.readMem vReg ((0 * Bk + idx.1.val) * D + idx.2.1.val) =
        V (FA1Math.blockIndex Bk 2 0 two_block0_le idx.1,
          idx.2.1, PUnit.unit) := by
    intro idx
    simpa [FA1Math.blockIndex] using
      hV (FA1Math.blockIndex Bk 2 0 two_block0_le idx.1,
        idx.2.1, PUnit.unit)
  have hVRight : ∀ idx : TileIndex [Bk, D],
      s.readMem vReg ((1 * Bk + idx.1.val) * D + idx.2.1.val) =
        V (FA1Math.blockIndex Bk 2 1 two_block1_le idx.1,
          idx.2.1, PUnit.unit) := by
    intro idx
    simpa [FA1Math.blockIndex] using
      hV (FA1Math.blockIndex Bk 2 1 two_block1_le idx.1,
        idx.2.1, PUnit.unit)
  constructor
  · exact fa2ScalarValueFragmentKernel_loaded_of_agrees
      vReg valueLeftReg 0 (by omega) d s sLeft consumer V
      hExecLeft hPid hValueLeftMem hVLeft
  · exact fa2ScalarValueFragmentKernel_loaded_of_agrees
      vReg valueRightReg 1 (by omega) d s sRight consumer V
      hExecRight hPid hValueRightMem hVRight

/-! ## FA-2 fragment-summary kernel surface

This scalar helper is the executable producer for one fragment summary:
given a `Bk`-lane score tile, a matching value tile for one output coordinate,
and the fragment max `m`, it writes the denominator `l` and numerator `o`
that the merge-stage kernel consumes.
-/

def fa2ScalarFragmentSummaryKernel
    (scoreReg valueReg mReg lReg oReg : RegionName) (Bk : Nat) :
    ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(Bk) + tl.arange(0, $(Bk))
  scores := tl.load($(scoreReg) + offs)
  values := tl.load($(valueReg) + offs)
  m      := tl.load($(mReg) + pid)
  w      := tl.exp(scores - m)
  l      := tl.sum(w, axis = 0)
  o      := tl.sum(w * values, axis = 0)
  tl.store($(lReg) + pid, l)
  tl.store($(oReg) + pid, o)
}

noncomputable def fa2ScalarFragmentDenom {Bk : Nat}
    (scores : Fin Bk → ℝ) (m : ℝ) : ℝ :=
  Finset.univ.sum fun j : Fin Bk => Real.exp (scores j - m)

noncomputable def fa2ScalarFragmentNumer {Bk : Nat}
    (scores values : Fin Bk → ℝ) (m : ℝ) : ℝ :=
  Finset.univ.sum fun j : Fin Bk => Real.exp (scores j - m) * values j

/-- The scalar fragment-summary kernel writes the fragment denominator and
numerator consumed by the FA-2 merge stage. -/
theorem fa2ScalarFragmentSummaryKernel_correct_view
    (scoreReg valueReg mReg lReg oReg : RegionName) (Bk : Nat)
    (s : BlockState) (scores values : Fin Bk → ℝ) (m : ℝ)
    (hScores : InputLoadedAt s scoreReg Bk scores)
    (hValues : InputLoadedAt s valueReg Bk values)
    (hm : s.readMem mReg s.pid = m)
    (hOutNe : lReg ≠ oReg) :
    ComputeCorrect.Realizes
      (kernel := fa2ScalarFragmentSummaryKernel scoreReg valueReg mReg lReg oReg Bk)
      (initialState := s)
      (write := fun ch : Fin 2 =>
        if ch.val = 0 then some (lReg, s.pid) else some (oReg, s.pid))
      (expected := fun ch : Fin 2 =>
        if ch.val = 0 then fa2ScalarFragmentDenom scores m
        else fa2ScalarFragmentNumer scores values m) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro ch
  fin_cases ch
  · simp [exec, fa2ScalarFragmentSummaryKernel, stepStmts, stepStmt, evalOp,
      hm, Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
      NumericDType.sub, WithBot.realExp, fa2ScalarFragmentDenom,
      BlockState.writeMemTyped_real] at hExec ⊢
    subst s'
    simp [hOutNe, TileShape.insertAxisIndex]
    refine Finset.sum_congr rfl ?_
    intro j _
    simp [hScores j]
  · simp [exec, fa2ScalarFragmentSummaryKernel, stepStmts, stepStmt, evalOp,
      hm, Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
      NumericDType.sub, WithBot.realExp, fa2ScalarFragmentNumer,
      BlockState.writeMemTyped_real] at hExec ⊢
    subst s'
    simp [TileShape.insertAxisIndex]
    refine Finset.sum_congr rfl ?_
    intro j _
    simp [hScores j, hValues j]

/-! ## FA-2 QK score-fragment tile bridge

This helper isolates the pure tile algebra behind the future executable score
producer: `Q @ Kᵀ * scale` computes the local `[M, Bk]` score fragment.
-/

noncomputable def fa2ScoreFragmentSpec {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk, D] → ℝ)
    (scale : ℝ) : TileIndex [M, Bk] → ℝ :=
  fun idx =>
    (Finset.univ.sum fun d : Fin D =>
      Q (idx.1, d, PUnit.unit) * K (idx.2.1, d, PUnit.unit)) * scale

/-- Tile-level QK score-fragment bridge for FA-2 forward. -/
theorem fa2_score_fragment_tile_eq {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk, D] → ℝ)
    (scale : ℝ) :
    Tile.bop NumericDType.real.mul Broadcast.scalarR
        (Tile.dot [] (Tile.ofReal Q) (Tile.transpose [] (Tile.ofReal K)))
        (Tile.scalar ((scale : ℝ) : WithBot ℝ)) =
      Tile.ofReal (fa2ScoreFragmentSpec Q K scale) := by
  ext idx
  rcases idx with ⟨i, j, u⟩
  cases u
  simp [fa2ScoreFragmentSpec, Tile.bop, Tile.dot, Tile.transpose, Tile.ofReal,
    NumericDType.mul]

/-- Local score-fragment spec agrees with the global `scaledScore` view used by
the FA-1/FA-2 attention specs when the local key fragment is a block slice of
the global K tensor. -/
theorem fa2ScoreFragmentSpec_eq_scaledScore {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (keyBlock : Nat) (hKeyBlock : keyBlock < N)
    (idx : TileIndex [M, Bk]) :
    fa2ScoreFragmentSpec
        Q
        (fun localIdx : TileIndex [Bk, D] =>
          K (FA1Math.blockIndex Bk N keyBlock
              (Nat.succ_le_iff.mpr hKeyBlock) localIdx.1,
            localIdx.2.1, PUnit.unit))
        scale idx =
      FA1Math.scaledScore Q K scale idx.1
        (FA1Math.blockIndex Bk N keyBlock
          (Nat.succ_le_iff.mpr hKeyBlock) idx.2.1) := by
  unfold fa2ScoreFragmentSpec FA1Math.scaledScore
  ring_nf

/-- Executable FA-2 QK score-fragment producer.

One program computes one `[M, Bk]` score tile:
`scores = Q_block @ K_fragmentᵀ * scale`, storing it contiguously at
`scoreReg + pid * (M * Bk)`.  The tile algebra is already isolated in
`fa2_score_fragment_tile_eq`; the full executable correctness proof also needs
a prefix-state/register fact and a row-major 2D store-readback helper. -/
def fa2ScoreFragmentKernel
    (qReg kReg scoreReg : RegionName)
    (M D Bk keyBlock : Nat) (scale : ℝ) : ComputeKernel := triton {
  pid         := tl.program_id(0)
  offs_m      := tl.arange(0, $(M))
  offs_n_loc  := tl.arange(0, $(Bk))
  offs_n      := $(keyBlock) * $(Bk) + offs_n_loc
  offs_d      := tl.arange(0, $(D))
  q_ptrs      := (pid * $(M) + offs_m)[:, None] * $(D) + offs_d[None, :]
  k_ptrs      := offs_n[:, None] * $(D) + offs_d[None, :]
  score_ptrs  := pid * $(M * Bk) + offs_m[:, None] * $(Bk) + offs_n_loc[None, :]
  q           := tl.load($(qReg) + q_ptrs)
  k           := tl.load($(kReg) + k_ptrs)
  scores      := tl.dot(q, tl.trans(k)) * $(scale)
  tl.store($(scoreReg) + score_ptrs, scores)
}

/-- Correctness of the executable FA-2 score-fragment producer for one program
instance. -/
theorem fa2ScoreFragmentKernel_correct_view
    (qReg kReg scoreReg : RegionName)
    (M D Bk keyBlock : Nat) (scale : ℝ) (s : BlockState)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk, D] → ℝ)
    (hQ : ∀ idx : TileIndex [M, D],
      s.readMem qReg
          ((s.pid * M + idx.1.val) * D + idx.2.1.val) =
        Q idx)
    (hK : ∀ idx : TileIndex [Bk, D],
      s.readMem kReg
          ((keyBlock * Bk + idx.1.val) * D + idx.2.1.val) =
        K idx) :
    ComputeCorrect.Realizes
      (kernel := fa2ScoreFragmentKernel qReg kReg scoreReg M D Bk keyBlock scale)
      (initialState := s)
      (write := fun idx : TileIndex [M, Bk] =>
        some (scoreReg,
          s.pid * (M * Bk) + idx.1.val * Bk + idx.2.1.val))
      (expected := fa2ScoreFragmentSpec Q K scale) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  simp [exec, fa2ScoreFragmentKernel, stepStmts, stepStmt, evalOp, Option.bind,
    Tile.bop, Tile.dot, Tile.transpose, NumericDType.add, NumericDType.mul,
    BlockState.writeMemTyped_real, fa2ScoreFragmentSpec] at hExec ⊢
  subst s'
  intro i j
  simp only [TileShape.dropInsertedIndex] at *
  have h_inj : Function.Injective
      (fun idx : TileIndex [M, Bk] =>
        s.pids 0 * (M * Bk) + idx.1.val * Bk + idx.2.1.val) := by
    intro a b h
    have hrow : Offset.rowMajor2D (rows := M) (cols := Bk)
        (s.pids 0 * (M * Bk)) Bk a =
        Offset.rowMajor2D (rows := M) (cols := Bk)
          (s.pids 0 * (M * Bk)) Bk b := by
      simpa [Offset.rowMajor2D, Offset.strided] using h
    exact Offset.rowMajor2D_inj
      (base := s.pids 0 * (M * Bk)) (rowStride := Bk) (le_refl Bk) hrow
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, j, PUnit.unit)]
  congr 1
  refine Finset.sum_congr rfl ?_
  intro d _
  simp [hQ (i, d, PUnit.unit), hK (j, d, PUnit.unit)]

/-- Global-score view of `fa2ScoreFragmentKernel_correct_view`: when the local
K fragment is the `keyBlock` slice of a global K tensor, the executable score
producer writes the global `FA1Math.scaledScore` values expected by the later
attention pipeline. -/
theorem fa2ScoreFragmentKernel_scaledScore_correct_view
    {M D Bk N : Nat}
    (qReg kReg scoreReg : RegionName)
    (keyBlock : Nat) (hKeyBlock : keyBlock < N)
    (scale : ℝ) (s : BlockState)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (hQ : ∀ idx : TileIndex [M, D],
      s.readMem qReg
          ((s.pid * M + idx.1.val) * D + idx.2.1.val) =
        Q idx)
    (hK : ∀ idx : TileIndex [Bk, D],
      s.readMem kReg
          ((keyBlock * Bk + idx.1.val) * D + idx.2.1.val) =
        K (FA1Math.blockIndex Bk N keyBlock
            (Nat.succ_le_iff.mpr hKeyBlock) idx.1,
          idx.2.1, PUnit.unit)) :
    ComputeCorrect.Realizes
      (kernel := fa2ScoreFragmentKernel qReg kReg scoreReg M D Bk keyBlock scale)
      (initialState := s)
      (write := fun idx : TileIndex [M, Bk] =>
        some (scoreReg,
          s.pid * (M * Bk) + idx.1.val * Bk + idx.2.1.val))
      (expected := fun idx : TileIndex [M, Bk] =>
        FA1Math.scaledScore Q K scale idx.1
          (FA1Math.blockIndex Bk N keyBlock
            (Nat.succ_le_iff.mpr hKeyBlock) idx.2.1)) := by
  have hview := fa2ScoreFragmentKernel_correct_view
    qReg kReg scoreReg M D Bk keyBlock scale s Q
    (fun idx : TileIndex [Bk, D] =>
      K (FA1Math.blockIndex Bk N keyBlock
          (Nat.succ_le_iff.mpr hKeyBlock) idx.1,
        idx.2.1, PUnit.unit))
    hQ hK
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  unfold ComputeCorrect.Realizes ComputeKernel.ExecCorrect
    ComputeKernel.ComputeCorrect ComputeKernel.ProjectedCorrect
    ComputeKernel.AlgorithmCorrect Kernel.Correct at hview
  rcases hview with ⟨_, hview⟩
  simp at hview
  intro idx
  have hOut := hview s s' hExec rfl idx.1 idx.2.1 idx.2.2
  have hSpec := fa2ScoreFragmentSpec_eq_scaledScore
    Q K scale keyBlock hKeyBlock idx
  simpa [hSpec] using hOut

/-- Row handoff from the score-fragment producer to the scalar fused forward
kernel: after producing the `[M, Bk]` score tile for query block `s.pid`, row
`row` can be consumed as a standard `InputLoadedAt` tile by a scalar kernel
whose `pid` is `s.pid * M + row`. -/
theorem fa2ScoreFragmentKernel_scaledScore_row_loaded
    {M D Bk N : Nat}
    (qReg kReg scoreReg : RegionName)
    (keyBlock : Nat) (hKeyBlock : keyBlock < N)
    (scale : ℝ) (s sScore : BlockState)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (row : Fin M)
    (hExec :
      exec (fa2ScoreFragmentKernel qReg kReg scoreReg M D Bk keyBlock scale).toAlgKernel s =
        some sScore)
    (hQ : ∀ idx : TileIndex [M, D],
      s.readMem qReg
          ((s.pid * M + idx.1.val) * D + idx.2.1.val) =
        Q idx)
    (hK : ∀ idx : TileIndex [Bk, D],
      s.readMem kReg
          ((keyBlock * Bk + idx.1.val) * D + idx.2.1.val) =
        K (FA1Math.blockIndex Bk N keyBlock
            (Nat.succ_le_iff.mpr hKeyBlock) idx.1,
          idx.2.1, PUnit.unit)) :
    InputLoadedAt
      (sScore.withPids (fun ax => if ax = 0 then s.pid * M + row.val else sScore.pids ax))
      scoreReg Bk
      (fun j : Fin Bk =>
        FA1Math.scaledScore Q K scale row
          (FA1Math.blockIndex Bk N keyBlock
            (Nat.succ_le_iff.mpr hKeyBlock) j)) := by
  have hview := fa2ScoreFragmentKernel_scaledScore_correct_view
    qReg kReg scoreReg keyBlock hKeyBlock scale s Q K hQ hK
  unfold ComputeCorrect.Realizes ComputeKernel.ExecCorrect
    ComputeKernel.ComputeCorrect ComputeKernel.ProjectedCorrect
    ComputeKernel.AlgorithmCorrect Kernel.Correct at hview
  rcases hview with ⟨_, hview⟩
  simp at hview
  intro j
  have hOut := hview s sScore hExec rfl row j
  have hAddr :
      (s.pid * M + row.val) * Bk + j.val =
        s.pid * (M * Bk) + row.val * Bk + j.val := by
    ring
  simpa [InputLoadedAt, BlockState.withPids, hAddr] using hOut

/-- State-parametric row handoff from the score-fragment producer to a later
consumer state.  If `consumer` has the scalar fused-forward program id
`queryBlock * M + row` and agrees with the score producer final state on
`scoreReg`, the produced row is available as the consumer's `InputLoadedAt`
score tile. -/
theorem fa2ScoreFragmentKernel_scaledScore_row_loaded_of_agrees
    {M D Bk N : Nat}
    (qReg kReg scoreReg : RegionName)
    (keyBlock : Nat) (hKeyBlock : keyBlock < N)
    (scale : ℝ) (s sScore consumer : BlockState)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (row : Fin M)
    (hExec :
      exec (fa2ScoreFragmentKernel qReg kReg scoreReg M D Bk keyBlock scale).toAlgKernel s =
        some sScore)
    (hPid : consumer.pid = s.pid * M + row.val)
    (hScoreMem : ∀ offset, consumer.readMem scoreReg offset = sScore.readMem scoreReg offset)
    (hQ : ∀ idx : TileIndex [M, D],
      s.readMem qReg
          ((s.pid * M + idx.1.val) * D + idx.2.1.val) =
        Q idx)
    (hK : ∀ idx : TileIndex [Bk, D],
      s.readMem kReg
          ((keyBlock * Bk + idx.1.val) * D + idx.2.1.val) =
        K (FA1Math.blockIndex Bk N keyBlock
            (Nat.succ_le_iff.mpr hKeyBlock) idx.1,
          idx.2.1, PUnit.unit)) :
    InputLoadedAt consumer scoreReg Bk
      (fun j : Fin Bk =>
        FA1Math.scaledScore Q K scale row
          (FA1Math.blockIndex Bk N keyBlock
            (Nat.succ_le_iff.mpr hKeyBlock) j)) := by
  have hLoaded := fa2ScoreFragmentKernel_scaledScore_row_loaded
    qReg kReg scoreReg keyBlock hKeyBlock scale s sScore Q K row hExec hQ hK
  refine InputLoadedAt.transfer ?_ ?_ hLoaded
  · simp [hPid]
  · intro offset
    simpa [BlockState.withPids] using hScoreMem offset

/-- Two-block score-producer handoff for the scalar fused-forward consumer.
This packages the left/right score-row transfer obligations into exactly the
two `InputLoadedAt` hypotheses consumed by
`fa2ScalarTwoBlockForwardKernel_attentionReal_view`. -/
theorem fa2ScoreFragmentKernel_twoBlock_rows_loaded_of_agrees
    {M D Bk : Nat}
    (qReg kReg scoreLeftReg scoreRightReg : RegionName)
    (scale : ℝ) (s sLeft sRight consumer : BlockState)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * 2, D] → ℝ)
    (row : Fin M)
    (hExecLeft :
      exec (fa2ScoreFragmentKernel qReg kReg scoreLeftReg M D Bk 0 scale).toAlgKernel s =
        some sLeft)
    (hExecRight :
      exec (fa2ScoreFragmentKernel qReg kReg scoreRightReg M D Bk 1 scale).toAlgKernel s =
        some sRight)
    (hPid : consumer.pid = s.pid * M + row.val)
    (hScoreLeftMem :
      ∀ offset, consumer.readMem scoreLeftReg offset = sLeft.readMem scoreLeftReg offset)
    (hScoreRightMem :
      ∀ offset, consumer.readMem scoreRightReg offset = sRight.readMem scoreRightReg offset)
    (hQ : ∀ idx : TileIndex [M, D],
      s.readMem qReg
          ((s.pid * M + idx.1.val) * D + idx.2.1.val) =
        Q idx)
    (hK : ∀ idx : TileIndex [Bk * 2, D],
      s.readMem kReg (idx.1.val * D + idx.2.1.val) = K idx) :
    InputLoadedAt consumer scoreLeftReg Bk
      (fun j : Fin Bk =>
        FA1Math.scaledScore Q K scale row
          (FA1Math.blockIndex Bk 2 0 two_block0_le j)) ∧
    InputLoadedAt consumer scoreRightReg Bk
      (fun j : Fin Bk =>
        FA1Math.scaledScore Q K scale row
          (FA1Math.blockIndex Bk 2 1 two_block1_le j)) := by
  have hKLeft : ∀ idx : TileIndex [Bk, D],
      s.readMem kReg ((0 * Bk + idx.1.val) * D + idx.2.1.val) =
        K (FA1Math.blockIndex Bk 2 0 two_block0_le idx.1,
          idx.2.1, PUnit.unit) := by
    intro idx
    simpa [FA1Math.blockIndex] using
      hK (FA1Math.blockIndex Bk 2 0 two_block0_le idx.1,
        idx.2.1, PUnit.unit)
  have hKRight : ∀ idx : TileIndex [Bk, D],
      s.readMem kReg ((1 * Bk + idx.1.val) * D + idx.2.1.val) =
        K (FA1Math.blockIndex Bk 2 1 two_block1_le idx.1,
          idx.2.1, PUnit.unit) := by
    intro idx
    simpa [FA1Math.blockIndex] using
      hK (FA1Math.blockIndex Bk 2 1 two_block1_le idx.1,
        idx.2.1, PUnit.unit)
  constructor
  · exact fa2ScoreFragmentKernel_scaledScore_row_loaded_of_agrees
      qReg kReg scoreLeftReg 0 (by omega) scale s sLeft consumer Q K row
      hExecLeft hPid hScoreLeftMem hQ hKLeft
  · exact fa2ScoreFragmentKernel_scaledScore_row_loaded_of_agrees
      qReg kReg scoreRightReg 1 (by omega) scale s sRight consumer Q K row
      hExecRight hPid hScoreRightMem hQ hKRight

/-! ## FA-2 fused two-block scalar forward kernel surface

This fuses the two fragment-summary computations and the delayed-rescale merge
for one output coordinate.  It is still scalar (one output coordinate per
program), but it is the first executable FA-2 forward slice that contains both
fragment production and merge in one DSL kernel.
-/

def fa2ScalarTwoBlockForwardKernel
    (scoreLeftReg valueLeftReg scoreRightReg valueRightReg
      mLeftReg mRightReg mMergedReg outReg : RegionName)
    (Bk : Nat) : ComputeKernel := triton {
  pid      := tl.program_id(0)
  offs     := pid * $(Bk) + tl.arange(0, $(Bk))
  s_left   := tl.load($(scoreLeftReg) + offs)
  v_left   := tl.load($(valueLeftReg) + offs)
  s_right  := tl.load($(scoreRightReg) + offs)
  v_right  := tl.load($(valueRightReg) + offs)
  m_left   := tl.load($(mLeftReg) + pid)
  m_right  := tl.load($(mRightReg) + pid)
  m        := tl.load($(mMergedReg) + pid)
  w_left   := tl.exp(s_left - m_left)
  w_right  := tl.exp(s_right - m_right)
  l_left   := tl.sum(w_left, axis = 0)
  l_right  := tl.sum(w_right, axis = 0)
  o_left   := tl.sum(w_left * v_left, axis = 0)
  o_right  := tl.sum(w_right * v_right, axis = 0)
  a_left   := tl.exp(m_left - m)
  a_right  := tl.exp(m_right - m)
  l        := a_left * l_left + a_right * l_right
  o        := (a_left * o_left + a_right * o_right) / l
  tl.store($(outReg) + pid, o)
}

noncomputable def fa2ScalarTwoBlockForwardTileSpec {Bk : Nat}
    (scoresLeft valuesLeft scoresRight valuesRight : Fin Bk → ℝ)
    (mLeft mRight mMerged : ℝ) : ℝ :=
  let lLeft := fa2ScalarFragmentDenom scoresLeft mLeft
  let lRight := fa2ScalarFragmentDenom scoresRight mRight
  let oLeft := fa2ScalarFragmentNumer scoresLeft valuesLeft mLeft
  let oRight := fa2ScalarFragmentNumer scoresRight valuesRight mRight
  (Real.exp (mLeft - mMerged) * oLeft +
      Real.exp (mRight - mMerged) * oRight) /
    (Real.exp (mLeft - mMerged) * lLeft +
      Real.exp (mRight - mMerged) * lRight)

set_option maxHeartbeats 4000000

/-- Correctness of the fused scalar two-block FA-2 forward slice against its
tile-level score/value spec. -/
theorem fa2ScalarTwoBlockForwardKernel_correct_view
    (scoreLeftReg valueLeftReg scoreRightReg valueRightReg
      mLeftReg mRightReg mMergedReg outReg : RegionName)
    (Bk : Nat) (s : BlockState)
    (scoresLeft valuesLeft scoresRight valuesRight : Fin Bk → ℝ)
    (mLeft mRight mMerged : ℝ)
    (hScoresLeft : InputLoadedAt s scoreLeftReg Bk scoresLeft)
    (hValuesLeft : InputLoadedAt s valueLeftReg Bk valuesLeft)
    (hScoresRight : InputLoadedAt s scoreRightReg Bk scoresRight)
    (hValuesRight : InputLoadedAt s valueRightReg Bk valuesRight)
    (hmLeft : s.readMem mLeftReg s.pid = mLeft)
    (hmRight : s.readMem mRightReg s.pid = mRight)
    (hmMerged : s.readMem mMergedReg s.pid = mMerged) :
    ComputeCorrect.Realizes
      (kernel := fa2ScalarTwoBlockForwardKernel
        scoreLeftReg valueLeftReg scoreRightReg valueRightReg
        mLeftReg mRightReg mMergedReg outReg Bk)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar outReg s.pid)
      (expected := fun _ : PUnit =>
        fa2ScalarTwoBlockForwardTileSpec
          scoresLeft valuesLeft scoresRight valuesRight mLeft mRight mMerged) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  have hRead :
      exec (fa2ScalarTwoBlockForwardKernel
        scoreLeftReg valueLeftReg scoreRightReg valueRightReg
        mLeftReg mRightReg mMergedReg outReg Bk) s = some s' := hExec
  simp [exec, fa2ScalarTwoBlockForwardKernel, stepStmts, stepStmt, evalOp,
    hmLeft, hmRight, hmMerged, Tile.bop, Tile.uop, Tile.reduceSum,
    Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    WithBot.realExp, WithBot.realDiv, fa2ScalarFragmentDenom,
    fa2ScalarFragmentNumer, fa2ScalarTwoBlockForwardTileSpec,
    BlockState.writeMemTyped_real] at hRead ⊢
  subst s'
  intro _
  simp [ComputeCorrect.WriteMap.scalar, TileShape.insertAxisIndex]
  have hOL :
      (∑ x : Fin Bk,
        Real.exp (s.readMem scoreLeftReg (s.pids 0 * Bk + x.val) - mLeft) *
          s.readMem valueLeftReg (s.pids 0 * Bk + x.val)) =
        ∑ j : Fin Bk, Real.exp (scoresLeft j - mLeft) * valuesLeft j := by
    refine Finset.sum_congr rfl ?_
    intro j _
    simp [hScoresLeft j, hValuesLeft j]
  have hOR :
      (∑ x : Fin Bk,
        Real.exp (s.readMem scoreRightReg (s.pids 0 * Bk + x.val) - mRight) *
          s.readMem valueRightReg (s.pids 0 * Bk + x.val)) =
        ∑ j : Fin Bk, Real.exp (scoresRight j - mRight) * valuesRight j := by
    refine Finset.sum_congr rfl ?_
    intro j _
    simp [hScoresRight j, hValuesRight j]
  have hLL :
      (∑ x : Fin Bk,
        Real.exp (s.readMem scoreLeftReg (s.pids 0 * Bk + x.val) - mLeft)) =
        ∑ j : Fin Bk, Real.exp (scoresLeft j - mLeft) := by
    refine Finset.sum_congr rfl ?_
    intro j _
    simp [hScoresLeft j]
  have hLR :
      (∑ x : Fin Bk,
        Real.exp (s.readMem scoreRightReg (s.pids 0 * Bk + x.val) - mRight)) =
        ∑ j : Fin Bk, Real.exp (scoresRight j - mRight) := by
    refine Finset.sum_congr rfl ?_
    intro j _
    simp [hScoresRight j]
  have hNum :
      Real.exp (mLeft - mMerged) *
          (∑ x : Fin Bk,
            Real.exp (s.readMem scoreLeftReg (s.pids 0 * Bk + x.val) - mLeft) *
              s.readMem valueLeftReg (s.pids 0 * Bk + x.val)) +
        Real.exp (mRight - mMerged) *
          (∑ x : Fin Bk,
            Real.exp (s.readMem scoreRightReg (s.pids 0 * Bk + x.val) - mRight) *
              s.readMem valueRightReg (s.pids 0 * Bk + x.val)) =
        Real.exp (mLeft - mMerged) *
          (∑ j : Fin Bk, Real.exp (scoresLeft j - mLeft) * valuesLeft j) +
        Real.exp (mRight - mMerged) *
          (∑ j : Fin Bk, Real.exp (scoresRight j - mRight) * valuesRight j) := by
    rw [hOL, hOR]
  have hDen :
      Real.exp (mLeft - mMerged) *
          (∑ x : Fin Bk,
            Real.exp (s.readMem scoreLeftReg (s.pids 0 * Bk + x.val) - mLeft)) +
        Real.exp (mRight - mMerged) *
          (∑ x : Fin Bk,
            Real.exp (s.readMem scoreRightReg (s.pids 0 * Bk + x.val) - mRight)) =
        Real.exp (mLeft - mMerged) *
          (∑ j : Fin Bk, Real.exp (scoresLeft j - mLeft)) +
        Real.exp (mRight - mMerged) *
          (∑ j : Fin Bk, Real.exp (scoresRight j - mRight)) := by
    rw [hLL, hLR]
  exact congrArg₂ (fun (num den : ℝ) => num / den) hNum hDen

/-- End-to-end scalar theorem for the fused two-block FA-2 forward slice: if
the score/value buffers contain the two Q/K/V fragments for one output
coordinate, the executable fused scalar kernel writes the flat `attentionReal`
result. -/
theorem fa2ScalarTwoBlockForwardKernel_attentionReal_view
    {M D Bk : Nat}
    (scoreLeftReg valueLeftReg scoreRightReg valueRightReg
      mLeftReg mRightReg mMergedReg outReg : RegionName)
    (s : BlockState)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * 2, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D])
    (mLeft mRight mMerged : ℝ)
    (hScoresLeft : InputLoadedAt s scoreLeftReg Bk
      (fun j : Fin Bk =>
        FA1Math.scaledScore Q K scale idx.1
          (FA1Math.blockIndex Bk 2 0 two_block0_le j)))
    (hValuesLeft : InputLoadedAt s valueLeftReg Bk
      (fun j : Fin Bk =>
        V (FA1Math.blockIndex Bk 2 0 two_block0_le j, idx.2.1, PUnit.unit)))
    (hScoresRight : InputLoadedAt s scoreRightReg Bk
      (fun j : Fin Bk =>
        FA1Math.scaledScore Q K scale idx.1
          (FA1Math.blockIndex Bk 2 1 two_block1_le j)))
    (hValuesRight : InputLoadedAt s valueRightReg Bk
      (fun j : Fin Bk =>
        V (FA1Math.blockIndex Bk 2 1 two_block1_le j, idx.2.1, PUnit.unit)))
    (hmLeft : s.readMem mLeftReg s.pid = mLeft)
    (hmRight : s.readMem mRightReg s.pid = mRight)
    (hmMerged : s.readMem mMergedReg s.pid = mMerged)
    (hlFree : FA1Math.lFree Q K scale 2 (le_refl 2) idx.1 ≠ 0) :
    ComputeCorrect.Realizes
      (kernel := fa2ScalarTwoBlockForwardKernel
        scoreLeftReg valueLeftReg scoreRightReg valueRightReg
        mLeftReg mRightReg mMergedReg outReg Bk)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar outReg s.pid)
      (expected := fun _ : PUnit => attentionReal Q K V scale idx) := by
  have hcorrect := fa2ScalarTwoBlockForwardKernel_correct_view
    scoreLeftReg valueLeftReg scoreRightReg valueRightReg
    mLeftReg mRightReg mMergedReg outReg Bk s
    (fun j : Fin Bk =>
      FA1Math.scaledScore Q K scale idx.1
        (FA1Math.blockIndex Bk 2 0 two_block0_le j))
    (fun j : Fin Bk =>
      V (FA1Math.blockIndex Bk 2 0 two_block0_le j, idx.2.1, PUnit.unit))
    (fun j : Fin Bk =>
      FA1Math.scaledScore Q K scale idx.1
        (FA1Math.blockIndex Bk 2 1 two_block1_le j))
    (fun j : Fin Bk =>
      V (FA1Math.blockIndex Bk 2 1 two_block1_le j, idx.2.1, PUnit.unit))
    mLeft mRight mMerged
    hScoresLeft hValuesLeft hScoresRight hValuesRight hmLeft hmRight hmMerged
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  unfold ComputeCorrect.Realizes ComputeKernel.ExecCorrect
    ComputeKernel.ComputeCorrect ComputeKernel.ProjectedCorrect
    ComputeKernel.AlgorithmCorrect Kernel.Correct at hcorrect
  rcases hcorrect with ⟨_, hcorrect⟩
  simp at hcorrect
  have hOut := hcorrect s s' hExec rfl PUnit.unit
  intro _
  have hSpec :
      fa2ScalarTwoBlockForwardTileSpec
        (fun j : Fin Bk =>
          FA1Math.scaledScore Q K scale idx.1
            (FA1Math.blockIndex Bk 2 0 two_block0_le j))
        (fun j : Fin Bk =>
          V (FA1Math.blockIndex Bk 2 0 two_block0_le j, idx.2.1, PUnit.unit))
        (fun j : Fin Bk =>
          FA1Math.scaledScore Q K scale idx.1
            (FA1Math.blockIndex Bk 2 1 two_block1_le j))
        (fun j : Fin Bk =>
          V (FA1Math.blockIndex Bk 2 1 two_block1_le j, idx.2.1, PUnit.unit))
        mLeft mRight mMerged =
        attentionReal Q K V scale idx := by
    unfold fa2ScalarTwoBlockForwardTileSpec fa2ScalarFragmentDenom
      fa2ScalarFragmentNumer
    change fa2TwoBlockForwardSpec Q K V scale mLeft mRight mMerged idx =
      attentionReal Q K V scale idx
    exact fa2_two_block_forward_eq_attentionReal
      Q K V scale mLeft mRight mMerged idx hlFree
  simpa [ComputeCorrect.WriteMap.scalar, hSpec] using hOut

/-- Producer-consumer wrapper for the fused scalar two-block FA-2 forward slice.
The left/right score inputs are discharged from two executable
`fa2ScoreFragmentKernel` producer runs plus score-buffer agreement with the
consumer state; value buffers and max registers remain explicit consumer-side
inputs. -/
theorem fa2ScalarTwoBlockForwardKernel_attentionReal_of_score_producers_view
    {M D Bk : Nat}
    (qReg kReg scoreLeftReg valueLeftReg scoreRightReg valueRightReg
      mLeftReg mRightReg mMergedReg outReg : RegionName)
    (sProducer sLeft sRight consumer : BlockState)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * 2, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D])
    (mLeft mRight mMerged : ℝ)
    (hExecLeft :
      exec (fa2ScoreFragmentKernel qReg kReg scoreLeftReg M D Bk 0 scale).toAlgKernel
          sProducer =
        some sLeft)
    (hExecRight :
      exec (fa2ScoreFragmentKernel qReg kReg scoreRightReg M D Bk 1 scale).toAlgKernel
          sProducer =
        some sRight)
    (hPid : consumer.pid = sProducer.pid * M + idx.1.val)
    (hScoreLeftMem :
      ∀ offset, consumer.readMem scoreLeftReg offset = sLeft.readMem scoreLeftReg offset)
    (hScoreRightMem :
      ∀ offset, consumer.readMem scoreRightReg offset = sRight.readMem scoreRightReg offset)
    (hQ : ∀ qIdx : TileIndex [M, D],
      sProducer.readMem qReg
          ((sProducer.pid * M + qIdx.1.val) * D + qIdx.2.1.val) =
        Q qIdx)
    (hK : ∀ kIdx : TileIndex [Bk * 2, D],
      sProducer.readMem kReg (kIdx.1.val * D + kIdx.2.1.val) = K kIdx)
    (hValuesLeft : InputLoadedAt consumer valueLeftReg Bk
      (fun j : Fin Bk =>
        V (FA1Math.blockIndex Bk 2 0 two_block0_le j, idx.2.1, PUnit.unit)))
    (hValuesRight : InputLoadedAt consumer valueRightReg Bk
      (fun j : Fin Bk =>
        V (FA1Math.blockIndex Bk 2 1 two_block1_le j, idx.2.1, PUnit.unit)))
    (hmLeft : consumer.readMem mLeftReg consumer.pid = mLeft)
    (hmRight : consumer.readMem mRightReg consumer.pid = mRight)
    (hmMerged : consumer.readMem mMergedReg consumer.pid = mMerged)
    (hlFree : FA1Math.lFree Q K scale 2 (le_refl 2) idx.1 ≠ 0) :
    ComputeCorrect.Realizes
      (kernel := fa2ScalarTwoBlockForwardKernel
        scoreLeftReg valueLeftReg scoreRightReg valueRightReg
        mLeftReg mRightReg mMergedReg outReg Bk)
      (initialState := consumer)
      (write := ComputeCorrect.WriteMap.scalar outReg consumer.pid)
      (expected := fun _ : PUnit => attentionReal Q K V scale idx) := by
  have hScores := fa2ScoreFragmentKernel_twoBlock_rows_loaded_of_agrees
    qReg kReg scoreLeftReg scoreRightReg scale
    sProducer sLeft sRight consumer Q K idx.1
    hExecLeft hExecRight hPid hScoreLeftMem hScoreRightMem hQ hK
  exact fa2ScalarTwoBlockForwardKernel_attentionReal_view
    scoreLeftReg valueLeftReg scoreRightReg valueRightReg
    mLeftReg mRightReg mMergedReg outReg consumer Q K V scale idx
    mLeft mRight mMerged
    hScores.1 hValuesLeft hScores.2 hValuesRight hmLeft hmRight hmMerged hlFree

/-- Producer-consumer wrapper that discharges both score-buffer and value-buffer
inputs for the fused scalar two-block FA-2 forward slice from executable
producer kernels.  Max registers remain explicit consumer-side assumptions. -/
theorem fa2ScalarTwoBlockForwardKernel_attentionReal_of_score_value_producers_view
    {M D Bk : Nat}
    (qReg kReg vReg scoreLeftReg valueLeftReg scoreRightReg valueRightReg
      mLeftReg mRightReg mMergedReg outReg : RegionName)
    (sScoreProducer sScoreLeft sScoreRight
      sValueProducer sValueLeft sValueRight consumer : BlockState)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * 2, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D])
    (mLeft mRight mMerged : ℝ)
    (hScoreExecLeft :
      exec (fa2ScoreFragmentKernel qReg kReg scoreLeftReg M D Bk 0 scale).toAlgKernel
          sScoreProducer =
        some sScoreLeft)
    (hScoreExecRight :
      exec (fa2ScoreFragmentKernel qReg kReg scoreRightReg M D Bk 1 scale).toAlgKernel
          sScoreProducer =
        some sScoreRight)
    (hValueExecLeft :
      exec (fa2ScalarValueFragmentKernel vReg valueLeftReg D Bk 0 idx.2.1.val).toAlgKernel
          sValueProducer =
        some sValueLeft)
    (hValueExecRight :
      exec (fa2ScalarValueFragmentKernel vReg valueRightReg D Bk 1 idx.2.1.val).toAlgKernel
          sValueProducer =
        some sValueRight)
    (hScorePid : consumer.pid = sScoreProducer.pid * M + idx.1.val)
    (hValuePid : consumer.pid = sValueProducer.pid)
    (hScoreLeftMem :
      ∀ offset, consumer.readMem scoreLeftReg offset = sScoreLeft.readMem scoreLeftReg offset)
    (hScoreRightMem :
      ∀ offset, consumer.readMem scoreRightReg offset = sScoreRight.readMem scoreRightReg offset)
    (hValueLeftMem :
      ∀ offset, consumer.readMem valueLeftReg offset = sValueLeft.readMem valueLeftReg offset)
    (hValueRightMem :
      ∀ offset, consumer.readMem valueRightReg offset = sValueRight.readMem valueRightReg offset)
    (hQ : ∀ qIdx : TileIndex [M, D],
      sScoreProducer.readMem qReg
          ((sScoreProducer.pid * M + qIdx.1.val) * D + qIdx.2.1.val) =
        Q qIdx)
    (hK : ∀ kIdx : TileIndex [Bk * 2, D],
      sScoreProducer.readMem kReg (kIdx.1.val * D + kIdx.2.1.val) = K kIdx)
    (hV : ∀ vIdx : TileIndex [Bk * 2, D],
      sValueProducer.readMem vReg (vIdx.1.val * D + vIdx.2.1.val) = V vIdx)
    (hmLeft : consumer.readMem mLeftReg consumer.pid = mLeft)
    (hmRight : consumer.readMem mRightReg consumer.pid = mRight)
    (hmMerged : consumer.readMem mMergedReg consumer.pid = mMerged)
    (hlFree : FA1Math.lFree Q K scale 2 (le_refl 2) idx.1 ≠ 0) :
    ComputeCorrect.Realizes
      (kernel := fa2ScalarTwoBlockForwardKernel
        scoreLeftReg valueLeftReg scoreRightReg valueRightReg
        mLeftReg mRightReg mMergedReg outReg Bk)
      (initialState := consumer)
      (write := ComputeCorrect.WriteMap.scalar outReg consumer.pid)
      (expected := fun _ : PUnit => attentionReal Q K V scale idx) := by
  have hScores := fa2ScoreFragmentKernel_twoBlock_rows_loaded_of_agrees
    qReg kReg scoreLeftReg scoreRightReg scale
    sScoreProducer sScoreLeft sScoreRight consumer Q K idx.1
    hScoreExecLeft hScoreExecRight hScorePid hScoreLeftMem hScoreRightMem hQ hK
  have hValues := fa2ScalarValueFragmentKernel_twoBlock_loaded_of_agrees
    vReg valueLeftReg valueRightReg idx.2.1
    sValueProducer sValueLeft sValueRight consumer V
    hValueExecLeft hValueExecRight hValuePid hValueLeftMem hValueRightMem hV
  exact fa2ScalarTwoBlockForwardKernel_attentionReal_view
    scoreLeftReg valueLeftReg scoreRightReg valueRightReg
    mLeftReg mRightReg mMergedReg outReg consumer Q K V scale idx
    mLeft mRight mMerged
    hScores.1 hValues.1 hScores.2 hValues.2 hmLeft hmRight hmMerged hlFree

/-- 4D-facing wrapper for the fused scalar two-block FA-2 forward slice.  The
kernel still writes one scalar output coordinate, but the theorem is stated in
the same `(batch, head, query, d)` language as the forward user surface. -/
theorem fa2ScalarTwoBlockForwardKernel_attentionReal4D_view
    {B H S_q D Bk : Nat}
    (scoreLeftReg valueLeftReg scoreRightReg valueRightReg
      mLeftReg mRightReg mMergedReg outReg : RegionName)
    (s : BlockState)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, Bk * 2, D] → ℝ)
    (scale : ℝ) (b : Fin B) (h : Fin H) (i : Fin S_q) (d : Fin D)
    (mLeft mRight mMerged : ℝ)
    (hScoresLeft : InputLoadedAt s scoreLeftReg Bk
      (fun j : Fin Bk =>
        FA1Math.scaledScore (sliceBH Q b h) (sliceBH K b h) scale i
          (FA1Math.blockIndex Bk 2 0 two_block0_le j)))
    (hValuesLeft : InputLoadedAt s valueLeftReg Bk
      (fun j : Fin Bk =>
        (sliceBH V b h) (FA1Math.blockIndex Bk 2 0 two_block0_le j,
          d, PUnit.unit)))
    (hScoresRight : InputLoadedAt s scoreRightReg Bk
      (fun j : Fin Bk =>
        FA1Math.scaledScore (sliceBH Q b h) (sliceBH K b h) scale i
          (FA1Math.blockIndex Bk 2 1 two_block1_le j)))
    (hValuesRight : InputLoadedAt s valueRightReg Bk
      (fun j : Fin Bk =>
        (sliceBH V b h) (FA1Math.blockIndex Bk 2 1 two_block1_le j,
          d, PUnit.unit)))
    (hmLeft : s.readMem mLeftReg s.pid = mLeft)
    (hmRight : s.readMem mRightReg s.pid = mRight)
    (hmMerged : s.readMem mMergedReg s.pid = mMerged)
    (hlFree :
      FA1Math.lFree (sliceBH Q b h) (sliceBH K b h) scale 2 (le_refl 2) i ≠ 0) :
    ComputeCorrect.Realizes
      (kernel := fa2ScalarTwoBlockForwardKernel
        scoreLeftReg valueLeftReg scoreRightReg valueRightReg
        mLeftReg mRightReg mMergedReg outReg Bk)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar outReg s.pid)
      (expected := fun _ : PUnit =>
        attentionReal4D Q K V scale (b, h, i, d, PUnit.unit)) := by
  have hview := fa2ScalarTwoBlockForwardKernel_attentionReal_view
    scoreLeftReg valueLeftReg scoreRightReg valueRightReg
    mLeftReg mRightReg mMergedReg outReg s
    (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
    scale (i, d, PUnit.unit) mLeft mRight mMerged
    hScoresLeft hValuesLeft hScoresRight hValuesRight
    hmLeft hmRight hmMerged hlFree
  simpa [attentionReal4D_slice] using hview

/-- Grid-facing wrapper for the fused scalar two-block FA-2 forward slice.
Every program instance may correspond to a different 4D output coordinate; the
kernel writes that coordinate's `attentionReal4D` result at its scalar output
slot. -/
theorem fa2ScalarTwoBlockForwardKernel_forAll_attentionReal4D_view
    {B H S_q D Bk : Nat} {g : Grid}
    (scoreLeftReg valueLeftReg scoreRightReg valueRightReg
      mLeftReg mRightReg mMergedReg outReg : RegionName)
    (s : BlockState)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, Bk * 2, D] → ℝ)
    (scale : ℝ) (coord : GridIndex g → TileIndex [B, H, S_q, D])
    (mLeft mRight mMerged : GridIndex g → ℝ)
    (hScoresLeft : ∀ gridIdx : GridIndex g,
      InputLoadedAt (s.withGridIndex gridIdx) scoreLeftReg Bk
        (fun j : Fin Bk =>
          FA1Math.scaledScore
            (sliceBH Q (coord gridIdx).1 (coord gridIdx).2.1)
            (sliceBH K (coord gridIdx).1 (coord gridIdx).2.1) scale
            (coord gridIdx).2.2.1
            (FA1Math.blockIndex Bk 2 0 two_block0_le j)))
    (hValuesLeft : ∀ gridIdx : GridIndex g,
      InputLoadedAt (s.withGridIndex gridIdx) valueLeftReg Bk
        (fun j : Fin Bk =>
          (sliceBH V (coord gridIdx).1 (coord gridIdx).2.1)
            (FA1Math.blockIndex Bk 2 0 two_block0_le j,
              (coord gridIdx).2.2.2.1, PUnit.unit)))
    (hScoresRight : ∀ gridIdx : GridIndex g,
      InputLoadedAt (s.withGridIndex gridIdx) scoreRightReg Bk
        (fun j : Fin Bk =>
          FA1Math.scaledScore
            (sliceBH Q (coord gridIdx).1 (coord gridIdx).2.1)
            (sliceBH K (coord gridIdx).1 (coord gridIdx).2.1) scale
            (coord gridIdx).2.2.1
            (FA1Math.blockIndex Bk 2 1 two_block1_le j)))
    (hValuesRight : ∀ gridIdx : GridIndex g,
      InputLoadedAt (s.withGridIndex gridIdx) valueRightReg Bk
        (fun j : Fin Bk =>
          (sliceBH V (coord gridIdx).1 (coord gridIdx).2.1)
            (FA1Math.blockIndex Bk 2 1 two_block1_le j,
              (coord gridIdx).2.2.2.1, PUnit.unit)))
    (hmLeft : ∀ gridIdx : GridIndex g,
      (s.withGridIndex gridIdx).readMem mLeftReg (s.withGridIndex gridIdx).pid =
        mLeft gridIdx)
    (hmRight : ∀ gridIdx : GridIndex g,
      (s.withGridIndex gridIdx).readMem mRightReg (s.withGridIndex gridIdx).pid =
        mRight gridIdx)
    (hmMerged : ∀ gridIdx : GridIndex g,
      (s.withGridIndex gridIdx).readMem mMergedReg (s.withGridIndex gridIdx).pid =
        mMerged gridIdx)
    (hlFree : ∀ gridIdx : GridIndex g,
      FA1Math.lFree
          (sliceBH Q (coord gridIdx).1 (coord gridIdx).2.1)
          (sliceBH K (coord gridIdx).1 (coord gridIdx).2.1)
          scale 2 (le_refl 2) (coord gridIdx).2.2.1 ≠ 0) :
    Kernel.ForAllProgramsSome
      (fa2ScalarTwoBlockForwardKernel
        scoreLeftReg valueLeftReg scoreRightReg valueRightReg
        mLeftReg mRightReg mMergedReg outReg Bk).toAlgKernel
      g s
      (fun gridIdx s' =>
        s'.readMem outReg (s.withGridIndex gridIdx).pid =
          attentionReal4D Q K V scale (coord gridIdx)) := by
  intro gridIdx
  let sIdx := s.withGridIndex gridIdx
  have hview := fa2ScalarTwoBlockForwardKernel_attentionReal4D_view
    scoreLeftReg valueLeftReg scoreRightReg valueRightReg
    mLeftReg mRightReg mMergedReg outReg sIdx Q K V scale
    (coord gridIdx).1 (coord gridIdx).2.1 (coord gridIdx).2.2.1
    (coord gridIdx).2.2.2.1
    (mLeft gridIdx) (mRight gridIdx) (mMerged gridIdx)
    (hScoresLeft gridIdx) (hValuesLeft gridIdx)
    (hScoresRight gridIdx) (hValuesRight gridIdx)
    (hmLeft gridIdx) (hmRight gridIdx) (hmMerged gridIdx)
    (hlFree gridIdx)
  obtain ⟨s', hExec⟩ :
      ∃ s',
        exec
            (fa2ScalarTwoBlockForwardKernel
              scoreLeftReg valueLeftReg scoreRightReg valueRightReg
              mLeftReg mRightReg mMergedReg outReg Bk).toAlgKernel
            sIdx = some s' := by
    simp [exec, fa2ScalarTwoBlockForwardKernel, stepStmts, stepStmt, evalOp,
      Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
      NumericDType.sub, NumericDType.div, WithBot.realExp, WithBot.realDiv,
      BlockState.writeMemTyped_real]
  refine ⟨s', hExec, ?_⟩
  unfold ComputeCorrect.Realizes ComputeKernel.ExecCorrect
    ComputeKernel.ComputeCorrect ComputeKernel.ProjectedCorrect
    ComputeKernel.AlgorithmCorrect Kernel.Correct at hview
  rcases hview with ⟨_, hview⟩
  simp at hview
  have hOut := hview sIdx s' hExec rfl PUnit.unit
  simpa [sIdx, ComputeCorrect.WriteMap.scalar] using hOut

/-! ## FA-2 merge-stage kernel surface

This small kernel is the executable core of the FA-2 fragment merge.  Each
program reads two scalar fragments `(m, l, o)` and writes the delayed-rescale
merge result for one output coordinate.
-/

def fa2ScalarTwoFragmentMergeKernel
    (mLeftReg lLeftReg oLeftReg mRightReg lRightReg oRightReg mMergedReg outReg : RegionName) :
    ComputeKernel := triton {
  pid     := tl.program_id(0)
  m_left  := tl.load($(mLeftReg) + pid)
  l_left  := tl.load($(lLeftReg) + pid)
  o_left  := tl.load($(oLeftReg) + pid)
  m_right := tl.load($(mRightReg) + pid)
  l_right := tl.load($(lRightReg) + pid)
  o_right := tl.load($(oRightReg) + pid)
  m       := tl.load($(mMergedReg) + pid)
  a_left  := tl.exp(m_left - m)
  a_right := tl.exp(m_right - m)
  l       := a_left * l_left + a_right * l_right
  o       := (a_left * o_left + a_right * o_right) / l
  tl.store($(outReg) + pid, o)
}

noncomputable def fa2ScalarTwoFragmentMergeSpec
    (mLeft lLeft oLeft mRight lRight oRight mMerged : ℝ) : ℝ :=
  let m := mMerged
  let aLeft := Real.exp (mLeft - m)
  let aRight := Real.exp (mRight - m)
  (aLeft * oLeft + aRight * oRight) / (aLeft * lLeft + aRight * lRight)

theorem fa2ScalarTwoFragmentMergeKernel_correct
    (mLeftReg lLeftReg oLeftReg mRightReg lRightReg oRightReg mMergedReg outReg : RegionName)
    (s : BlockState)
    (mLeft lLeft oLeft mRight lRight oRight mMerged : ℝ)
    (hmLeft : s.readMem mLeftReg s.pid = mLeft)
    (hlLeft : s.readMem lLeftReg s.pid = lLeft)
    (hoLeft : s.readMem oLeftReg s.pid = oLeft)
    (hmRight : s.readMem mRightReg s.pid = mRight)
    (hlRight : s.readMem lRightReg s.pid = lRight)
    (hoRight : s.readMem oRightReg s.pid = oRight)
    (hmMerged : s.readMem mMergedReg s.pid = mMerged) :
    observeRowAt
        (exec (fa2ScalarTwoFragmentMergeKernel
          mLeftReg lLeftReg oLeftReg mRightReg lRightReg oRightReg mMergedReg outReg) s)
        outReg s.pid =
      some (fa2ScalarTwoFragmentMergeSpec mLeft lLeft oLeft mRight lRight oRight mMerged) := by
  simp [observeRowAt, exec, fa2ScalarTwoFragmentMergeKernel,
    fa2ScalarTwoFragmentMergeSpec, stepStmts, stepStmt, evalOp, hmLeft, hlLeft,
    hoLeft, hmRight, hlRight, hoRight, hmMerged, Tile.bop, Tile.uop, NumericDType.add,
    NumericDType.sub, NumericDType.mul, NumericDType.div, WithBot.realExp,
    WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv,
    BlockState.writeMemTyped_real]

/-- Compute-facing surface for the scalar FA-2 merge-stage kernel. -/
theorem fa2ScalarTwoFragmentMergeKernel_correct_view
    (mLeftReg lLeftReg oLeftReg mRightReg lRightReg oRightReg mMergedReg outReg : RegionName)
    (s : BlockState)
    (mLeft lLeft oLeft mRight lRight oRight mMerged : ℝ)
    (hmLeft : s.readMem mLeftReg s.pid = mLeft)
    (hlLeft : s.readMem lLeftReg s.pid = lLeft)
    (hoLeft : s.readMem oLeftReg s.pid = oLeft)
    (hmRight : s.readMem mRightReg s.pid = mRight)
    (hlRight : s.readMem lRightReg s.pid = lRight)
    (hoRight : s.readMem oRightReg s.pid = oRight)
    (hmMerged : s.readMem mMergedReg s.pid = mMerged) :
    ComputeCorrect.Realizes
      (kernel := fa2ScalarTwoFragmentMergeKernel
        mLeftReg lLeftReg oLeftReg mRightReg lRightReg oRightReg mMergedReg outReg)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar outReg s.pid)
      (expected := fun _ : PUnit =>
        fa2ScalarTwoFragmentMergeSpec mLeft lLeft oLeft mRight lRight oRight mMerged) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  have hview := fa2ScalarTwoFragmentMergeKernel_correct
    mLeftReg lLeftReg oLeftReg mRightReg lRightReg oRightReg mMergedReg outReg
    s mLeft lLeft oLeft mRight lRight oRight mMerged
    hmLeft hlLeft hoLeft hmRight hlRight hoRight hmMerged
  rw [hExec] at hview
  intro _
  simpa [observeRowAt, ComputeCorrect.WriteMap.scalar] using hview

/-- End-to-end scalar merge-stage theorem for a two-block FA-2 forward slice:
if the fragment buffers contain the left/right denominator and numerator
contributions for a Q/K/V two-block domain, the executable merge-stage kernel
writes the flat `attentionReal` result. -/
theorem fa2ScalarTwoFragmentMergeKernel_attentionReal_view
    {M D Bk : Nat}
    (mLeftReg lLeftReg oLeftReg mRightReg lRightReg oRightReg mMergedReg outReg : RegionName)
    (s : BlockState)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * 2, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D])
    (mLeft mRight mMerged : ℝ)
    (lLeft oLeft lRight oRight : ℝ)
    (hmLeft : s.readMem mLeftReg s.pid = mLeft)
    (hlLeftMem : s.readMem lLeftReg s.pid = lLeft)
    (hoLeftMem : s.readMem oLeftReg s.pid = oLeft)
    (hmRight : s.readMem mRightReg s.pid = mRight)
    (hlRightMem : s.readMem lRightReg s.pid = lRight)
    (hoRightMem : s.readMem oRightReg s.pid = oRight)
    (hmMerged : s.readMem mMergedReg s.pid = mMerged)
    (hlLeft :
      lLeft = fa2TwoBlockDenomLeft Q K scale mLeft idx.1)
    (hoLeft :
      oLeft = fa2TwoBlockNumerLeft Q K V scale mLeft idx)
    (hlRight :
      lRight = fa2TwoBlockDenomRight Q K scale mRight idx.1)
    (hoRight :
      oRight = fa2TwoBlockNumerRight Q K V scale mRight idx)
    (hlFree : FA1Math.lFree Q K scale 2 (le_refl 2) idx.1 ≠ 0) :
    ComputeCorrect.Realizes
      (kernel := fa2ScalarTwoFragmentMergeKernel
        mLeftReg lLeftReg oLeftReg mRightReg lRightReg oRightReg mMergedReg outReg)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar outReg s.pid)
      (expected := fun _ : PUnit => attentionReal Q K V scale idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  have hmerge := fa2ScalarTwoFragmentMergeKernel_correct
    mLeftReg lLeftReg oLeftReg mRightReg lRightReg oRightReg mMergedReg outReg
    s mLeft lLeft oLeft mRight lRight oRight mMerged
    hmLeft hlLeftMem hoLeftMem hmRight hlRightMem hoRightMem hmMerged
  rw [hExec] at hmerge
  intro _
  have hSpec :
      fa2ScalarTwoFragmentMergeSpec mLeft lLeft oLeft mRight lRight oRight mMerged =
        attentionReal Q K V scale idx := by
    rw [hlLeft, hoLeft, hlRight, hoRight]
    exact fa2_two_block_forward_eq_attentionReal
      Q K V scale mLeft mRight mMerged idx hlFree
  simpa [observeRowAt, ComputeCorrect.WriteMap.scalar, hSpec] using hmerge

/-- FA-2 forward baseline kernel.

At this stage the FA-2 module exposes the same online-softmax block recurrence
as the verified FA-1 4D forward kernel.  This gives Tier 3-B an independent
theorem surface and a stable spec target before the FA-2-specific partitioning
and block-skip rewrites are introduced. -/
def fa2ForwardKernelStrided
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks : Nat)
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    (stride_ob stride_oh stride_om stride_od : Nat)
    (scale : ℝ) : ComputeKernel :=
  fa1ForwardKernelStrided qReg kReg vReg outReg M D Bk numKVBlocks
    stride_qb stride_qh stride_qs stride_qd
    stride_kb stride_kh stride_kn stride_kd
    stride_vb stride_vh stride_vn stride_vd
    stride_ob stride_oh stride_om stride_od scale

/-- FA-2 forward baseline correctness over bundled 4D tensor views. -/
theorem fa2_forward_correct_4D_views
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (hSk : Bk * numKVBlocks = S_k)
    (views : FA1Views4D B H S_q S_k D)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQBnd : s.pids 0 * M + M ≤ S_q)
    (hQ4D : TensorView.loaded s views.qView Q4D)
    (hK4D : TensorView.loaded s views.kView K4D)
    (hV4D : TensorView.loaded s views.vView V4D) :
    ComputeCorrect.Realizes
      (kernel :=
        fa2ForwardKernelStrided views.qReg views.kReg views.vReg views.outReg
          M D Bk numKVBlocks
          views.layout.qB views.layout.qH views.layout.qS views.layout.qD
          views.layout.kB views.layout.kH views.layout.kS views.layout.kD
          views.layout.vB views.layout.vH views.layout.vS views.layout.vD
          views.layout.oB views.layout.oH views.layout.oS views.layout.oD
          scale)
      (initialState := s)
      (write := fun idx : TileIndex [M, D] =>
        some (views.outReg, (views.outBlockOffset s M) idx))
      (expected := fun idx : TileIndex [M, D] =>
        attentionReal4D Q4D K4D V4D scale
          (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
           ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
           idx.2.1, PUnit.unit)) := by
  simpa [fa2ForwardKernelStrided, FA1Views4D.kernel]
    using fa1_forward_correct_4D_views hBk hNumKVBlocks hSk
      views Q4D K4D V4D scale s hPidB hPidH hQBnd hQ4D hK4D hV4D

end VeriTile.Examples
