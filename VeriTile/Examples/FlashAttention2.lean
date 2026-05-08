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

namespace VeriTile.Examples

open VeriTile.Triton
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
