/-
VeriTile.Examples.FlashAttention1

FlashAttention-1 forward kernel — DSL definition + math model + full
correctness theorem.

v0 scope:
  * One `program_id` ↔ one Q-row block of `[BLOCK_M, D]`.
  * K, V are full `[S, D]` matrices in memory (single-block-output).
  * Inner KV-block loop with the canonical online-softmax update
    (running `m_i`, `l_i`, `O`).
  * Arbitrary user-supplied `scale : ℝ` (the standard `1/√D` Triton
    specialization is the caller's responsibility).
  * Non-causal and causal strided kernels are fully proved against
    their 4D reference specs.
  * Full-tile assumptions remain explicit:
    `Bk * numKVBlocks = S_k` and `q_block * M + M <= S_q`.

`fa1_forward_correct_4D` and `fa1_forward_correct_4D_causal` are fully
proven end-to-end via the four-stage decomposition (math identity +
pre-loop init + loop step via `forLoop_inv` + post-loop readout).

v1 / boundary-mask scope:
  * Boundary-masked non-causal and causal kernels are defined and proved
    through 4D/layout/view theorem surfaces.
  * D-tail kernels split logical head dimension `D` from block width `Bd`;
    their DSL surface and padded-D math bridges are in place.
-/

import VeriTile.Examples.FlashAttention1.Common.Spec
import VeriTile.Examples.FlashAttention1.Common.CausalBoundary

namespace VeriTile.Examples

open VeriTile.Triton

/-! ## Causal streaming math model

The causal loop is the same online-softmax recurrence, but each block
score is first filtered by the global predicate
`key_index ≤ qStart + query_lane`. Filtered-out scores are represented
as `⊥`, so `WithBot.realExp` contributes exactly zero softmax mass,
matching the kernel's `tl.where(..., -inf)` followed by `tl.exp`.
-/

namespace FA1MathCausal

/-- Causal scaled score. Returns `⊥` for future keys, otherwise the
ordinary scaled dot-product score. -/
noncomputable def maskedScore {M S D : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ) (scale : ℝ)
    (i : Fin M) (j : Fin S) : WithBot ℝ :=
  if j.val ≤ qStart + i.val then
    (StreamingAccumulator.scaledScore Q K scale i j : ℝ)
  else
    ⊥

@[simp] theorem maskedScore_of_le {M S D : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ) (scale : ℝ)
    (i : Fin M) (j : Fin S) (h : j.val ≤ qStart + i.val) :
    maskedScore qStart Q K scale i j =
      ((StreamingAccumulator.scaledScore Q K scale i j : ℝ) : WithBot ℝ) := by
  simp [maskedScore, h]

@[simp] theorem maskedScore_of_not_le {M S D : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ) (scale : ℝ)
    (i : Fin M) (j : Fin S) (h : ¬ j.val ≤ qStart + i.val) :
    maskedScore qStart Q K scale i j = (⊥ : WithBot ℝ) := by
  simp [maskedScore, h]

/-- Exponentiating a masked score contributes the ordinary exponent on
causally visible keys and zero on future keys. This is the algebraic
shape produced by `tl.where(..., -inf)` followed by `tl.exp`. -/
theorem realExp_maskedScore_sub {M S D : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ) (scale : ℝ)
    (i : Fin M) (j : Fin S) (m : WithBot ℝ) :
    WithBot.realExp
        (Option.map₂ (fun x y : ℝ => x - y)
          (maskedScore qStart Q K scale i j) m)
      =
      if j.val ≤ qStart + i.val then
        WithBot.realExp
          (Option.map (fun y : ℝ => StreamingAccumulator.scaledScore Q K scale i j - y) m)
      else
        some 0 := by
  by_cases h : j.val ≤ qStart + i.val
  · simp [h, maskedScore]
    cases m <;> rfl
  · simp [h, maskedScore, WithBot.realExp]
    cases m <;> rfl

/-- Running per-row max of causal masked scores over the first `k` KV
blocks. Future keys enter as `⊥`, exactly like `tl.where(..., -inf)`. -/
noncomputable def mPartial {M D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) :
    Nat → Fin M → WithBot ℝ
  | 0, _ => ⊥
  | k + 1, i =>
      if h : k + 1 ≤ numKVBlocks then
        max (mPartial Bk qStart Q numKVBlocks K scale k i)
          ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
            maskedScore qStart Q K scale i
              (StreamingAccumulator.blockIndex Bk numKVBlocks k h jLocal))
      else
        mPartial Bk qStart Q numKVBlocks K scale k i

/-- Running causal softmax normalizer, shifted by the running max. -/
noncomputable def lPartial {M D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) :
    Nat → Fin M → ℝ
  | 0, _ => 0
  | k + 1, i =>
      if h : k + 1 ≤ numKVBlocks then
        let alpha :=
          (WithBot.realExp
            (Option.map₂ (fun x y : ℝ => x - y)
              (mPartial Bk qStart Q numKVBlocks K scale k i)
              (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i))).unbotD 0
        alpha * lPartial Bk qStart Q numKVBlocks K scale k i +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            (WithBot.realExp
              (Option.map₂ (fun x y : ℝ => x - y)
                (maskedScore qStart Q K scale i
                  (StreamingAccumulator.blockIndex Bk numKVBlocks k h jLocal))
                (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i))).unbotD 0)
      else
        lPartial Bk qStart Q numKVBlocks K scale k i

/-- Running causal unnormalized output accumulator. -/
noncomputable def oPartial {M D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (numKVBlocks : Nat)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) :
    Nat → TileIndex [M, D] → ℝ
  | 0, _ => 0
  | k + 1, idx =>
      if h : k + 1 ≤ numKVBlocks then
        let alpha :=
          (WithBot.realExp
            (Option.map₂ (fun x y : ℝ => x - y)
              (mPartial Bk qStart Q numKVBlocks K scale k idx.1)
              (mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1))).unbotD 0
        alpha * oPartial Bk qStart Q numKVBlocks K V scale k idx +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            let j := StreamingAccumulator.blockIndex Bk numKVBlocks k h jLocal
            (WithBot.realExp
              (Option.map₂ (fun x y : ℝ => x - y)
                (maskedScore qStart Q K scale idx.1 j)
                (mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1))).unbotD 0 *
              V (j, idx.2.1, PUnit.unit))
      else
        oPartial Bk qStart Q numKVBlocks K V scale k idx

/-- Recurrence unfold for `mPartial` at iteration `k+1`, when `k < N`. -/
theorem mPartial_succ_of_lt {M D Bk : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    mPartial Bk qStart Q numKVBlocks K scale (k + 1) i =
      max (mPartial Bk qStart Q numKVBlocks K scale k i)
        ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
          maskedScore qStart Q K scale i
            (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal)) := by
  conv_lhs => rw [mPartial]
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

/-- Recurrence unfold for `lPartial` at iteration `k+1`, when `k < N`. -/
theorem lPartial_succ_of_lt {M D Bk : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    lPartial Bk qStart Q numKVBlocks K scale (k + 1) i =
      let alpha :=
        (WithBot.realExp
          (Option.map₂ (fun x y : ℝ => x - y)
            (mPartial Bk qStart Q numKVBlocks K scale k i)
            (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i))).unbotD 0
      alpha * lPartial Bk qStart Q numKVBlocks K scale k i +
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          (WithBot.realExp
            (Option.map₂ (fun x y : ℝ => x - y)
              (maskedScore qStart Q K scale i
                (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal))
              (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i))).unbotD 0) := by
  conv_lhs => rw [lPartial]
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

/-- Recurrence unfold for `oPartial` at iteration `k+1`, when `k < N`. -/
theorem oPartial_succ_of_lt {M D Bk : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (idx : TileIndex [M, D]) :
    oPartial Bk qStart Q numKVBlocks K V scale (k + 1) idx =
      let alpha :=
        (WithBot.realExp
          (Option.map₂ (fun x y : ℝ => x - y)
            (mPartial Bk qStart Q numKVBlocks K scale k idx.1)
            (mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1))).unbotD 0
      alpha * oPartial Bk qStart Q numKVBlocks K V scale k idx +
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          let j := StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal
          (WithBot.realExp
            (Option.map₂ (fun x y : ℝ => x - y)
              (maskedScore qStart Q K scale idx.1 j)
              (mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1))).unbotD 0 *
            V (j, idx.2.1, PUnit.unit)) := by
  conv_lhs => rw [oPartial]
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

/-! ### Causal m-free reference sums

These are the causal counterparts of `StreamingAccumulator.lFree` / `StreamingAccumulator.oFree`.
They keep the causal mask in the unshifted reference form, so the final
streaming ratio can be compared directly with `attentionRealCausalBlock`.
-/

/-- Causal m-free normalizer over the first `k` KV blocks. Future keys
contribute zero mass. -/
noncomputable def lFree {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k ≤ N) (i : Fin M) : ℝ :=
  Finset.univ.sum (fun n : Fin k => Finset.univ.sum (fun jL : Fin Bk =>
    let j := StreamingAccumulator.blockIndex Bk N n.val (Nat.lt_of_lt_of_le n.isLt hk) jL
    if j.val ≤ qStart + i.val then
      Real.exp (StreamingAccumulator.scaledScore Q K scale i j)
    else
      0))

/-- Causal m-free unnormalized output over the first `k` KV blocks. -/
noncomputable def oFree {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k ≤ N)
    (idx : TileIndex [M, D]) : ℝ :=
  Finset.univ.sum (fun n : Fin k => Finset.univ.sum (fun jL : Fin Bk =>
    let j := StreamingAccumulator.blockIndex Bk N n.val (Nat.lt_of_lt_of_le n.isLt hk) jL
    (if j.val ≤ qStart + idx.1.val then
      Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1 j)
    else
      0) * V (j, idx.2.1, PUnit.unit)))

@[simp] theorem lFree_zero {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (i : Fin M) :
    lFree qStart Q K scale 0 (Nat.zero_le _) i = 0 := by
  unfold lFree
  simp

@[simp] theorem oFree_zero {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oFree qStart Q K V scale 0 (Nat.zero_le _) idx = 0 := by
  unfold oFree
  simp

/-- Recurrence for causal `lFree`. -/
theorem lFree_succ {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k + 1 ≤ N) (i : Fin M) :
    lFree qStart Q K scale (k + 1) hk i =
      lFree qStart Q K scale k (Nat.le_of_succ_le hk) i +
      Finset.univ.sum (fun jL : Fin Bk =>
        let j := StreamingAccumulator.blockIndex Bk N k hk jL
        if j.val ≤ qStart + i.val then
          Real.exp (StreamingAccumulator.scaledScore Q K scale i j)
        else
          0) := by
  unfold lFree
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

/-- Recurrence for causal `oFree`. -/
theorem oFree_succ {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k + 1 ≤ N)
    (idx : TileIndex [M, D]) :
    oFree qStart Q K V scale (k + 1) hk idx =
      oFree qStart Q K V scale k (Nat.le_of_succ_le hk) idx +
      Finset.univ.sum (fun jL : Fin Bk =>
        let j := StreamingAccumulator.blockIndex Bk N k hk jL
        (if j.val ≤ qStart + idx.1.val then
          Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1 j)
        else
          0) * V (j, idx.2.1, PUnit.unit)) := by
  unfold oFree
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

/-- Flat form of the final causal normalizer. -/
theorem lFree_eq_flat {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (i : Fin M) :
    lFree qStart Q K scale N (le_refl N) i =
      Finset.univ.sum (fun j : Fin (Bk * N) =>
        if j.val ≤ qStart + i.val then
          Real.exp (StreamingAccumulator.scaledScore Q K scale i j)
        else
          0) := by
  unfold lFree
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (StreamingAccumulator.blockIndexEquiv Bk N) ?_ ?_).symm
  · intro _; simp
  · intro j _
    rw [StreamingAccumulator.blockIndex_blockIndexEquiv]

/-- Flat form of the final causal output accumulator. -/
theorem oFree_eq_flat {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oFree qStart Q K V scale N (le_refl N) idx =
      Finset.univ.sum (fun j : Fin (Bk * N) =>
        (if j.val ≤ qStart + idx.1.val then
          Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1 j)
        else
          0) * V (j, idx.2.1, PUnit.unit)) := by
  unfold oFree
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (StreamingAccumulator.blockIndexEquiv Bk N) ?_ ?_).symm
  · intro _; simp
  · intro j _
    rw [StreamingAccumulator.blockIndex_blockIndexEquiv]

/-- For causal streaming, `mPartial (k+1) i` is non-`⊥` whenever there
is at least one block (`hBk : 0 < Bk`) and at least one iteration
(`hk : k+1 ≤ numKVBlocks`). The first block's first lane has global
index `0`, which is always causally visible (`0 ≤ qStart + i.val`),
giving a `some _` contribution that propagates via running max. -/
theorem mPartial_succ_ne_bot {M D Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (i : Fin M) :
    mPartial Bk qStart Q numKVBlocks K scale (k + 1) i ≠ ⊥ := by
  induction k with
  | zero =>
      rw [mPartial_succ_of_lt qStart Q numKVBlocks K scale 0
            (Nat.lt_of_succ_le hk) i]
      have hVis : (StreamingAccumulator.blockIndex Bk numKVBlocks 0
              (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk)) (⟨0, hBk⟩ : Fin Bk)).val
              ≤ qStart + i.val := by
        simp [StreamingAccumulator.blockIndex]
      have h0 : maskedScore qStart Q K scale i
              (StreamingAccumulator.blockIndex Bk numKVBlocks 0
                (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk))
                (⟨0, hBk⟩ : Fin Bk)) ≠ ⊥ := by
        rw [maskedScore_of_le qStart Q K scale i _ hVis]
        exact WithBot.coe_ne_bot
      show mPartial Bk qStart Q numKVBlocks K scale 0 i ⊔ _ ≠ ⊥
      change ⊥ ⊔ _ ≠ ⊥
      rw [bot_sup_eq]
      simp [Finset.sup_eq_bot_iff]
      exact ⟨⟨0, hBk⟩, h0⟩
  | succ k' ih =>
      have hk' : k' + 1 ≤ numKVBlocks := by omega
      rw [mPartial_succ_of_lt qStart Q numKVBlocks K scale (k' + 1)
            (Nat.lt_of_succ_le hk) i]
      intro hcontra
      have h_left : mPartial Bk qStart Q numKVBlocks K scale (k' + 1) i ≤ ⊥ := by
        rw [← hcontra]; exact le_max_left _ _
      exact ih hk' (le_bot_iff.mp h_left)

/-- Causal streaming normalizer equals the causal m-free normalizer times
the final exponential shift. -/
theorem lPartial_eq_mShifted {M D Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (i : Fin M) :
    lPartial Bk qStart Q numKVBlocks K scale k i =
      Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale k i).unbotD 0) *
        lFree qStart Q K scale k hk i := by
  induction k with
  | zero =>
      show (0 : ℝ) =
        Real.exp (-((⊥ : WithBot ℝ).unbotD 0)) *
          lFree qStart Q K scale 0 hk i
      rw [lFree_zero]
      ring
  | succ k ih =>
      have hk' : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      rw [lPartial_succ_of_lt qStart Q numKVBlocks K scale k
        (Nat.lt_of_succ_le hk) i]
      rw [ih hk']
      rw [lFree_succ qStart Q K scale k hk i, mul_add]
      have hExpSub : ∀ s : ℝ,
          Real.exp (s - (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i).unbotD 0) =
            Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale (k + 1) i).unbotD 0) *
              Real.exp s := by
        intro s
        rw [show s - (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i).unbotD 0
              = -(mPartial Bk qStart Q numKVBlocks K scale (k + 1) i).unbotD 0 + s by ring,
            Real.exp_add]
      have hSumB :
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            (WithBot.realExp
              (Option.map₂ (fun x y : ℝ => x - y)
                (maskedScore qStart Q K scale i
                  (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk)) jLocal))
                (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i))).unbotD 0)
          =
          Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale (k + 1) i).unbotD 0) *
            (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
              let j := StreamingAccumulator.blockIndex Bk numKVBlocks k hk jLocal
              if j.val ≤ qStart + i.val then
                Real.exp (StreamingAccumulator.scaledScore Q K scale i j)
              else
                0) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro jLocal _
        let j := StreamingAccumulator.blockIndex Bk numKVBlocks k hk jLocal
        have hjEq :
            StreamingAccumulator.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk)) jLocal = j := by
          rfl
        rw [hjEq]
        by_cases hvis : j.val ≤ qStart + i.val
        · rw [maskedScore_of_le qStart Q K scale i j hvis]
          obtain ⟨m, hm⟩ := WithBot.ne_bot_iff_exists.mp
            (mPartial_succ_ne_bot hBk qStart Q numKVBlocks K scale k hk i)
          rw [← hm]
          simp [hvis]
          rw [show StreamingAccumulator.scaledScore Q K scale i j - m = -m + StreamingAccumulator.scaledScore Q K scale i j by ring,
              Real.exp_add]
        · rw [maskedScore_of_not_le qStart Q K scale i j hvis]
          simp [hvis]
      have hSumA :
          (WithBot.realExp
              (Option.map₂ (fun x y : ℝ => x - y)
                (mPartial Bk qStart Q numKVBlocks K scale k i)
                (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i))).unbotD 0 *
            (Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale k i).unbotD 0) *
              lFree qStart Q K scale k hk' i)
          =
          Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale (k + 1) i).unbotD 0) *
            lFree qStart Q K scale k hk' i := by
        rcases Nat.eq_zero_or_pos k with hkz | hkpos
        · subst hkz
          rw [lFree_zero]
          ring
        · obtain ⟨k', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.pos_iff_ne_zero.mp hkpos)
          have hk_succ : k' + 1 ≤ numKVBlocks := Nat.le_of_succ_le hk
          have hmk_ne := mPartial_succ_ne_bot hBk qStart Q numKVBlocks K scale k' hk_succ i
          have hmk1_ne := mPartial_succ_ne_bot hBk qStart Q numKVBlocks K scale (k' + 1) hk i
          obtain ⟨rk, hrk⟩ := WithBot.ne_bot_iff_exists.mp hmk_ne
          obtain ⟨rk1, hrk1⟩ := WithBot.ne_bot_iff_exists.mp hmk1_ne
          rw [← hrk, ← hrk1]
          simp [WithBot.realExp]
          rw [show Real.exp (rk - rk1) = Real.exp (-rk1) * Real.exp rk by
            rw [show rk - rk1 = -rk1 + rk by ring, Real.exp_add]]
          rw [show Real.exp (-rk1) * Real.exp rk *
                    (Real.exp (-rk) * lFree qStart Q K scale (k' + 1) hk' i)
                = Real.exp (-rk1) *
                    (Real.exp rk * Real.exp (-rk) *
                      lFree qStart Q K scale (k' + 1) hk' i) by ring]
          rw [show Real.exp rk * Real.exp (-rk) = 1 by
            rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]]
          ring
      linarith [hSumA, hSumB]

/-- Causal streaming output accumulator equals the causal m-free output
accumulator times the final exponential shift. -/
theorem oPartial_eq_mShifted {M D Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (idx : TileIndex [M, D]) :
    oPartial Bk qStart Q numKVBlocks K V scale k idx =
      Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale k idx.1).unbotD 0) *
        oFree qStart Q K V scale k hk idx := by
  induction k with
  | zero =>
      show (0 : ℝ) =
        Real.exp (-((⊥ : WithBot ℝ).unbotD 0)) *
          oFree qStart Q K V scale 0 hk idx
      rw [oFree_zero]
      ring
  | succ k ih =>
      have hk' : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      rw [oPartial_succ_of_lt qStart Q numKVBlocks K V scale k
        (Nat.lt_of_succ_le hk) idx]
      rw [ih hk']
      rw [oFree_succ qStart Q K V scale k hk idx, mul_add]
      have hExpSub : ∀ s : ℝ,
          Real.exp (s - (mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) =
            Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
              Real.exp s := by
        intro s
        rw [show s - (mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1).unbotD 0
              = -(mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1).unbotD 0 + s by ring,
            Real.exp_add]
      have hSumB :
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            let j := StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk)) jLocal
            (WithBot.realExp
              (Option.map₂ (fun x y : ℝ => x - y)
                (maskedScore qStart Q K scale idx.1 j)
                (mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1))).unbotD 0 *
              V (j, idx.2.1, PUnit.unit))
          =
          Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
            (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
              let j := StreamingAccumulator.blockIndex Bk numKVBlocks k hk jLocal
              (if j.val ≤ qStart + idx.1.val then
                Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1 j)
              else
                0) * V (j, idx.2.1, PUnit.unit)) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro jLocal _
        let j := StreamingAccumulator.blockIndex Bk numKVBlocks k hk jLocal
        have hjEq :
            StreamingAccumulator.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk)) jLocal = j := by
          rfl
        rw [hjEq]
        dsimp only
        by_cases hvis : j.val ≤ qStart + idx.1.val
        · rw [maskedScore_of_le qStart Q K scale idx.1 j hvis]
          obtain ⟨m, hm⟩ := WithBot.ne_bot_iff_exists.mp
            (mPartial_succ_ne_bot hBk qStart Q numKVBlocks K scale k hk idx.1)
          rw [← hm]
          simp [hvis]
          rw [show StreamingAccumulator.scaledScore Q K scale idx.1 j - m =
                -m + StreamingAccumulator.scaledScore Q K scale idx.1 j by ring,
              Real.exp_add]
          ring
        · rw [maskedScore_of_not_le qStart Q K scale idx.1 j hvis]
          simp [hvis]
      have hSumA :
          (WithBot.realExp
              (Option.map₂ (fun x y : ℝ => x - y)
                (mPartial Bk qStart Q numKVBlocks K scale k idx.1)
                (mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1))).unbotD 0 *
            (Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale k idx.1).unbotD 0) *
              oFree qStart Q K V scale k hk' idx)
          =
          Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
            oFree qStart Q K V scale k hk' idx := by
        rcases Nat.eq_zero_or_pos k with hkz | hkpos
        · subst hkz
          rw [oFree_zero]
          ring
        · obtain ⟨k', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.pos_iff_ne_zero.mp hkpos)
          have hk_succ : k' + 1 ≤ numKVBlocks := Nat.le_of_succ_le hk
          have hmk_ne := mPartial_succ_ne_bot hBk qStart Q numKVBlocks K scale k' hk_succ idx.1
          have hmk1_ne := mPartial_succ_ne_bot hBk qStart Q numKVBlocks K scale (k' + 1) hk idx.1
          obtain ⟨rk, hrk⟩ := WithBot.ne_bot_iff_exists.mp hmk_ne
          obtain ⟨rk1, hrk1⟩ := WithBot.ne_bot_iff_exists.mp hmk1_ne
          rw [← hrk, ← hrk1]
          simp [WithBot.realExp]
          rw [show Real.exp (rk - rk1) = Real.exp (-rk1) * Real.exp rk by
            rw [show rk - rk1 = -rk1 + rk by ring, Real.exp_add]]
          rw [show Real.exp (-rk1) * Real.exp rk *
                    (Real.exp (-rk) * oFree qStart Q K V scale (k' + 1) hk' idx)
                = Real.exp (-rk1) *
                    (Real.exp rk * Real.exp (-rk) *
                      oFree qStart Q K V scale (k' + 1) hk' idx) by ring]
          rw [show Real.exp rk * Real.exp (-rk) = 1 by
            rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]]
          ring
      linarith [hSumA, hSumB]

/-- Causal m-free ratio is exactly the local-block causal attention spec. -/
theorem oFree_div_lFree_eq_attentionRealCausalBlock {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D])
    (_hlFree : lFree qStart Q K scale N (le_refl N) idx.1 ≠ 0) :
    oFree qStart Q K V scale N (le_refl N) idx /
        lFree qStart Q K scale N (le_refl N) idx.1
      = attentionRealCausalBlock qStart Q K V scale idx := by
  rw [oFree_eq_flat, lFree_eq_flat]
  unfold attentionRealCausalBlock
  rfl

/-- The final causal m-free normalizer is positive: key `0` is always
visible to every query row. -/
theorem lFree_final_pos {M D Bk N : Nat} (hBk : 0 < Bk) (hN : 0 < N)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (i : Fin M) :
    0 < lFree qStart Q K scale N (le_refl N) i := by
  rw [lFree_eq_flat]
  apply Finset.sum_pos'
  · intro j _
    by_cases h : j.val ≤ qStart + i.val
    · simp [h, le_of_lt (Real.exp_pos _)]
    · simp [h]
  · refine ⟨⟨0, Nat.mul_pos hBk hN⟩, Finset.mem_univ _, ?_⟩
    simp [Real.exp_pos]

/-- The final causal streaming normalizer is nonzero under non-empty KV
scope. -/
theorem lPartial_final_ne_zero {M D Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (hN : 0 < numKVBlocks)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (i : Fin M) :
    lPartial Bk qStart Q numKVBlocks K scale numKVBlocks i ≠ 0 := by
  rw [lPartial_eq_mShifted hBk qStart Q numKVBlocks K scale numKVBlocks
      (le_refl _) i]
  exact mul_ne_zero (Real.exp_ne_zero _)
    (ne_of_gt (lFree_final_pos hBk hN qStart Q K scale i))

/-- Final causal streaming ratio equals the user-facing local-block
causal attention spec. -/
theorem streaming_eq_attentionRealCausalBlock {M D Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (numKVBlocks : Nat) (hN : 0 < numKVBlocks)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    oPartial Bk qStart Q numKVBlocks K V scale numKVBlocks idx /
        lPartial Bk qStart Q numKVBlocks K scale numKVBlocks idx.1
      = attentionRealCausalBlock qStart Q K V scale idx := by
  have hl : lPartial Bk qStart Q numKVBlocks K scale numKVBlocks idx.1 ≠ 0 :=
    lPartial_final_ne_zero hBk qStart Q numKVBlocks hN K scale idx.1
  rw [oPartial_eq_mShifted hBk qStart Q numKVBlocks K V scale numKVBlocks
        (le_refl _) idx,
      lPartial_eq_mShifted hBk qStart Q numKVBlocks K scale numKVBlocks
        (le_refl _) idx.1]
  rw [mul_div_mul_left _ _ (Real.exp_ne_zero _)]
  have hlFree : lFree qStart Q K scale numKVBlocks (le_refl _) idx.1 ≠ 0 := by
    intro h
    apply hl
    rw [lPartial_eq_mShifted hBk qStart Q numKVBlocks K scale numKVBlocks
        (le_refl _) idx.1, h, mul_zero]
  exact oFree_div_lFree_eq_attentionRealCausalBlock qStart Q K V scale idx hlFree

/-! ### Tile-level block helpers — causal kernel body

Each helper packages one statement of the causal loop body's Tile-level
algebra into a single equation, used as a rewrite in
`fa1_step_strided_causal`.

The chain mirrors the kernel:

```text
scores_raw :=  q @ k.T * scale          -- block_scoresRaw_tile_eq
causal     :=  offs_m[:, None] >= offs_n[None, :]   -- block_causal_mask_tile_eq
scores     :=  tl.where(causal, scores_raw, -inf)   -- block_scores_tile_eq
m_block    :=  tl.max(scores, axis=1)               -- block_mBlock_tile_eq
m_new      :=  tl.max(m_i, m_block)                 -- block_mNew_tile_eq
```

These are the parts where the causal kernel diverges from the non-causal
strided kernel; the `m_new`/`alpha`/`l_new`/`o_acc` lemmas (combining
the mask-aware streaming math with the running accumulators) follow. -/

/-- The kernel's `scores_raw = q @ k.T * scale` Tile expression at this
program-instance evaluates to `Tile.ofReal` of the per-element
`scaledScore`. Same structure as the non-causal kernel's `scores`
register; reused here as the un-masked input to the causal mask. -/
theorem block_scoresRaw_tile_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (n : Nat) (hn : n < N) :
    Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.dot [] (Tile.ofReal Q)
        (Tile.transpose [] (Tile.ofReal
          (fun idx : TileIndex [Bk, D] =>
            K (StreamingAccumulator.blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.1,
              idx.2.1, PUnit.unit)))))
      (Tile.scalar ((scale : ℝ) : WithBot ℝ))
    = Tile.ofReal (fun idx : TileIndex [M, Bk] =>
        StreamingAccumulator.scaledScore Q K scale idx.1
          (StreamingAccumulator.blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.2.1)) := by
  ext idx
  obtain ⟨i, j, _⟩ := idx
  simp only [Tile.bop, Tile.scalar, Tile.ofReal_data,
    Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul]
  exact StreamingAccumulator.block_scaled_data_eq Q K scale n hn i j

/-- The causal mask tile `offs_m[:, None] >= offs_n[None, :]` evaluates
to the `decide` predicate `(n * Bk + j) ≤ (qb * M + i)` at position
`(i, j)`. -/
theorem block_causal_mask_tile_eq {M Bk : Nat} (qb n : Nat) :
    (Tile.cop (ComparableDType.nat.ge)
      (Broadcast.consR (Broadcast.consL Broadcast.nil))
      (Tile.expandDim ⟨1, by simp⟩
        (Tile.vec (fun i : Fin M => qb * M + i.val)))
      (Tile.expandDim ⟨0, by simp⟩
        (Tile.vec (fun j : Fin Bk => n * Bk + j.val))))
      = ⟨fun idx : TileIndex [M, Bk] =>
          decide ((n * Bk + idx.2.1.val) ≤ (qb * M + idx.1.val))⟩ := by
  ext idx
  obtain ⟨i, j, _⟩ := idx
  simp [Tile.cop, ComparableDType.ge, Tile.expandDim, Tile.vec,
    Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex]

/-- Combining the causal mask with the raw scores via `tl.where(causal,
scores_raw, -inf)` yields the `Tile` whose data is exactly
`maskedScore`. Visible lanes carry the real `scaledScore`; masked lanes
carry `⊥`. -/
theorem block_scores_tile_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (n : Nat) (hn : n < N) (qb : Nat) :
    Tile.select
      (⟨fun idx : TileIndex [M, Bk] =>
        decide ((n * Bk + idx.2.1.val) ≤ (qb * M + idx.1.val))⟩ : Tile .bool [M, Bk])
      (Tile.ofReal (fun idx : TileIndex [M, Bk] =>
        StreamingAccumulator.scaledScore Q K scale idx.1
          (StreamingAccumulator.blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.2.1)))
      (⟨fun _ : TileIndex [M, Bk] => (none : WithBot ℝ)⟩ : Tile .real [M, Bk])
    = ⟨fun idx : TileIndex [M, Bk] =>
        maskedScore (qb * M) Q K scale idx.1
          (StreamingAccumulator.blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.2.1)⟩ := by
  ext idx
  obtain ⟨i, j, u⟩ := idx
  simp only [Tile.select_data, Tile.ofReal_data]
  have hIdx : (StreamingAccumulator.blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j).val
            = n * Bk + j.val := by simp [StreamingAccumulator.blockIndex]
  by_cases h_mask : n * Bk + j.val ≤ qb * M + i.val
  · rw [decide_eq_true h_mask]
    simp only [if_true]
    rw [maskedScore_of_le (qb * M) Q K scale i _ (by rw [hIdx]; exact h_mask)]
    rfl
  · rw [decide_eq_false h_mask]
    show (none : WithBot ℝ) = maskedScore (qb * M) Q K scale i
        (StreamingAccumulator.blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j)
    rw [maskedScore_of_not_le (qb * M) Q K scale i _ (by rw [hIdx]; exact h_mask)]
    rfl

/-- Row-max of the causal scores tile: `tl.max(scores, axis=1)` produces
`Finset.univ.sup` of the per-row `maskedScore`. The reduce-max here uses
`WithBot.sup'` internally; we collapse it to `Finset.sup` since the
`WithBot` type already carries `⊥` as its bottom element. -/
theorem block_mBlock_tile_eq {M D Bk N : Nat} (hBk : 0 < Bk)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (n : Nat) (hn : n < N) (qb : Nat) :
    Tile.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false
      ⟨fun idx : TileIndex [M, Bk] =>
        maskedScore (qb * M) Q K scale idx.1
          (StreamingAccumulator.blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.2.1)⟩
    = some ⟨fun idx : TileIndex [M] =>
        (Finset.univ : Finset (Fin Bk)).sup
          (fun jLocal => maskedScore (qb * M) Q K scale idx.1
            (StreamingAccumulator.blockIndex Bk N n (Nat.succ_le_iff.mpr hn) jLocal))⟩ := by
  unfold Tile.reduceMax
  simp [Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, hBk]
  funext idx
  rw [Finset.sup'_eq_sup]
  rfl

/-- Combining the running max `m_i` with the per-block max `m_block`
yields the next-iteration `mPartial`. This is `mPartial_succ_of_lt` at
the `Tile` level. -/
theorem block_mNew_tile_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (n : Nat) (hn : n < N) (qb : Nat) :
    Tile.bop max (Broadcast.consSame Broadcast.nil)
      (⟨fun idx : TileIndex [M] =>
        mPartial Bk (qb * M) Q N K scale n idx.1⟩ : Tile .real [M])
      (⟨fun idx : TileIndex [M] =>
        (Finset.univ : Finset (Fin Bk)).sup
          (fun jLocal => maskedScore (qb * M) Q K scale idx.1
            (StreamingAccumulator.blockIndex Bk N n (Nat.succ_le_iff.mpr hn) jLocal))⟩ : Tile .real [M])
    = ⟨fun idx : TileIndex [M] =>
        mPartial Bk (qb * M) Q N K scale (n + 1) idx.1⟩ := by
  ext idx
  obtain ⟨i, _⟩ := idx
  simp [Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex]
  rw [mPartial_succ_of_lt (qb * M) Q N K scale n hn i]

/-- Causal `α` multiplier — the unbotted real payload of
`exp(mPartial(k) - mPartial(k+1))`. Same role as
`StreamingAccumulator.alphaPartial`, but for the causal streaming. -/
noncomputable def alphaCausal {M D Bk : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) : ℝ :=
  (WithBot.realExp
      (Option.map₂ (fun x y : ℝ => x - y)
        (mPartial Bk qStart Q numKVBlocks K scale k i)
        (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i))).unbotD 0

/-- The causal `l_new = α * l_i + tl.sum(p, axis=1)` update at the
simplified-input form realizes the causal streaming `lPartial(k+1)`.
This is `lPartial_succ_of_lt` lifted to `Tile` level. -/
theorem block_lNew_tile_eq {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < N) :
    Tile.bop WithBot.realAdd (Broadcast.consSame Broadcast.nil)
      (Tile.bop WithBot.realMul (Broadcast.consSame Broadcast.nil)
        (Tile.ofReal fun idx : TileIndex [M] =>
          alphaCausal qStart Q N K scale k idx.1)
        (Tile.ofReal fun idx : TileIndex [M] =>
          lPartial Bk qStart Q N K scale k idx.1))
      (Tile.ofReal fun idx : TileIndex [M] =>
        Finset.univ.sum (fun jLocal : Fin Bk =>
          (WithBot.realExp
            (Option.map₂ (fun x y : ℝ => x - y)
              (maskedScore qStart Q K scale idx.1
                (StreamingAccumulator.blockIndex Bk N k (Nat.succ_le_iff.mpr hk) jLocal))
              (mPartial Bk qStart Q N K scale (k + 1) idx.1))).unbotD 0))
      =
      Tile.ofReal (fun idx : TileIndex [M] =>
        lPartial Bk qStart Q N K scale (k + 1) idx.1) := by
  ext idx
  simp [Tile.bop, Tile.ofReal]
  rw [lPartial_succ_of_lt qStart Q N K scale k hk idx.1]
  rfl

/-- Bridge `if (decide P : Bool) then a else b ↔ if P then a else b`.
The kernel's `Tile.select` unfolds via `Tile.select_data` to a Bool-`ite`
(its `c.data idx : Bool`), while `maskedScore` and similar Prop-`if`
definitions elaborate to a Prop-`ite`. Both are propositionally equal
but not definitionally; this is the canonical simp bridge between them,
used in `fa1_step_strided_causal`'s operational simp. -/
@[simp] theorem ite_decide_bool {P : Prop} [Decidable P] {α : Sort _}
    (a b : α) :
    (if (decide P : Bool) then a else b) = if P then a else b := by
  by_cases h : P
  · simp [h]
  · simp [h]

/-- The causal `o_acc = α * o_acc + p @ V` update at the
simplified-input form realizes the causal streaming `oPartial(k+1)`.
This is `oPartial_succ_of_lt` lifted to `Tile` level. -/
theorem block_oAcc_tile_eq {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < N) :
    Tile.bop WithBot.realAdd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.bop WithBot.realMul (Broadcast.consSame (Broadcast.consL Broadcast.nil))
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.ofReal fun idx : TileIndex [M] =>
            alphaCausal qStart Q N K scale k idx.1))
        (Tile.ofReal fun idx : TileIndex [M, D] =>
          oPartial Bk qStart Q N K V scale k idx))
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        Finset.univ.sum (fun jLocal : Fin Bk =>
          let j := StreamingAccumulator.blockIndex Bk N k (Nat.succ_le_iff.mpr hk) jLocal
          (WithBot.realExp
            (Option.map₂ (fun x y : ℝ => x - y)
              (maskedScore qStart Q K scale idx.1 j)
              (mPartial Bk qStart Q N K scale (k + 1) idx.1))).unbotD 0 *
            V (j, idx.2.1, PUnit.unit)))
      =
      Tile.ofReal (fun idx : TileIndex [M, D] =>
        oPartial Bk qStart Q N K V scale (k + 1) idx) := by
  ext idx
  rcases idx with ⟨i, d, u⟩
  cases u
  simp [Tile.bop, Tile.expandDim, Tile.ofReal, TileShape.dropInsertedIndex]
  rw [oPartial_succ_of_lt qStart Q N K V scale k hk (i, d, PUnit.unit)]
  rfl

/-- Per-row bridge: the `Finset.sup` of the kernel-side
`if-then-some-else-none` lambda equals the `Finset.sup` of `maskedScore`.
The mask predicate matches because `(blockIndex k j).val = k * Bk + j.val`,
and the `some` payload matches because `scaledScore = scale * Σ Q*K`
(differing from kernel's `(Σ Q*K) * scale` only by `mul_comm`). -/
theorem kernelIfSup_eq_maskedScoreSup {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (qb : Nat) (k : Nat) (hk : k < N) (i : Fin M) :
    ((Finset.univ : Finset (Fin Bk)).sup
      (fun x : Fin Bk =>
        (if k * Bk + ↑x ≤ qb * M + ↑i then
          some
            ((∑ x_1 : Fin D,
              Q (i, x_1, PUnit.unit) *
                K (StreamingAccumulator.blockIndex Bk N k (Nat.succ_le_iff.mpr hk) x, x_1, PUnit.unit)) * scale)
        else none : WithBot ℝ))) =
    ((Finset.univ : Finset (Fin Bk)).sup
      (fun x : Fin Bk =>
        maskedScore (qb * M) Q K scale i
          (StreamingAccumulator.blockIndex Bk N k (Nat.succ_le_iff.mpr hk) x))) := by
  apply Finset.sup_congr rfl
  intro x _
  by_cases h_mask : k * Bk + x.val ≤ qb * M + i.val
  · rw [maskedScore_of_le (qb * M) Q K scale i _
        (by simp [StreamingAccumulator.blockIndex]; exact h_mask)]
    rw [if_pos h_mask]
    unfold StreamingAccumulator.scaledScore
    ring_nf
    rfl
  · rw [maskedScore_of_not_le (qb * M) Q K scale i _
        (by simp [StreamingAccumulator.blockIndex]; omega)]
    simp [h_mask]
    rfl

/-- Sup-form `mPartial(k+1)` recurrence over the kernel-side
`if-then-some-else-none` lambda. The kernel's `Tile.reduceMaxDrop`
produces `Finset.sup'`; after normalizing to `Finset.sup` via
`Finset.sup'_eq_sup`, this lemma directly closes the m_new conjunct. -/
theorem mPartial_succ_kernelForm {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (qb : Nat) (k : Nat) (hk : k < N) (i : Fin M) :
    max (mPartial Bk (qb * M) Q N K scale k i)
      ((Finset.univ : Finset (Fin Bk)).sup
        (fun x : Fin Bk =>
          (if k * Bk + ↑x ≤ qb * M + ↑i then
            some
              ((∑ x_1 : Fin D,
                Q (i, x_1, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk N k (Nat.succ_le_iff.mpr hk) x, x_1, PUnit.unit)) * scale)
          else none : WithBot ℝ))) =
    mPartial Bk (qb * M) Q N K scale (k + 1) i := by
  rw [kernelIfSup_eq_maskedScoreSup Q K scale qb k hk i,
      ← mPartial_succ_of_lt (qb * M) Q N K scale k hk i]

/-- `WithBot.realExp` is never bottom, so it is equal to its `unbotD`
payload rewrapped as `some`. This bridges the kernel's optional value
with the ℝ-clean streaming definitions. -/
theorem realExp_eq_some_unbotD (x : WithBot ℝ) :
    WithBot.realExp x = some ((WithBot.realExp x).unbotD 0) := by
  cases x <;> rfl

end FA1MathCausal
def P_fa1
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState) : Prop :=
  -- `pid` register and value preserved.
  s.regs .nat [] "pid" = some (Tile.scalar origPid) ∧
  s.pid = origPid ∧
  -- Q-row-block index vector (set pre-loop, reused post-loop for the
  -- final `tl.store` ptrs).
  s.regs .nat [M] "offs_m" = some
      (Tile.vec (fun i : Fin M => origPid * M + i.val)) ∧
  -- D-axis index vector.
  s.regs .nat [D] "offs_d" = some
      (Tile.vec (fun d : Fin D => d.val)) ∧
  -- Loaded Q-row-block.
  s.regs .real [M, D] "q" = some (Tile.ofReal Q) ∧
  -- Streaming accumulators at iteration `k`.
  s.regs .real [M] "m_i" = some
      ⟨fun idx : TileIndex [M] => StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k idx.1⟩ ∧
  s.regs .real [M] "l_i" = some
      (Tile.ofReal fun idx : TileIndex [M] =>
        StreamingAccumulator.lPartial Q numKVBlocks K scale k idx.1) ∧
  s.regs .real [M, D] "o_acc" = some
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        StreamingAccumulator.oPartial Q numKVBlocks K V scale k idx) ∧
  -- Input regions still loaded as the user's hypothesis says.
  InputAt s qReg
      (Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D) Q ∧
  InputAt s kReg
      (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K ∧
  InputAt s vReg
      (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V

/-- Loop-carried predicate for the strided FA-1 forward kernel
(`fa1ForwardKernelStrided`). Same streaming math content as `P_fa1` —
running `mPartial`/`lPartial`/`oPartial` in their register slots, plus
the initial Q-block load and book-keeping — but the InputAt premises
use the strided / 4D-aware addressing computed by the kernel from its
16 stride parameters and three `program_id` axes.

The static `(qb, headIdx, batch)` triple is the program_id values at
proof entry; the invariant asserts they are preserved across loop
iterations and exposes them as `Tile.scalar` registers.

Tile-local offset perspective: Q is observed at addresses
`qBase + i * sQS + d * sQD` where
`qBase = batch * sQB + headIdx * sQH + qb * M * sQS`. Similarly for
K/V/O. The 4D-wrapper corollary (issue #39 step (iv)) supplies these
offsets via `Offset.strided` over `[B, H, S_q, D]` plus
`Offset.strided_inj` for the readback injectivity. -/
def P_fa1_strided
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    -- Static (qb, headIdx, batch) program-id context.
    (qb headIdx batch : Nat)
    -- 16 stride parameters (matching `fa1ForwardKernelStrided`):
    (sQB sQH sQS sQD : Nat)
    (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat)
    (sOB sOH _sOM _sOD : Nat)
    -- Math inputs:
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState) : Prop :=
  -- Program-id values preserved.
  s.pids 0 = qb ∧ s.pids 1 = headIdx ∧ s.pids 2 = batch ∧
  -- pid_* registers (set by preLoop's three program_id reads).
  s.regs .nat [] "pid_qb" = some (Tile.scalar qb) ∧
  s.regs .nat [] "pid_h"  = some (Tile.scalar headIdx) ∧
  s.regs .nat [] "pid_b"  = some (Tile.scalar batch) ∧
  -- *_base_off registers (set by preLoop's four base-offset assigns).
  s.regs .nat [] "q_base_off" = some (Tile.scalar (batch * sQB + headIdx * sQH)) ∧
  s.regs .nat [] "k_base_off" = some (Tile.scalar (batch * sKB + headIdx * sKH)) ∧
  s.regs .nat [] "v_base_off" = some (Tile.scalar (batch * sVB + headIdx * sVH)) ∧
  s.regs .nat [] "o_base_off" = some (Tile.scalar (batch * sOB + headIdx * sOH)) ∧
  -- Q-block row index vector and D-axis index vector.
  s.regs .nat [M] "offs_m" = some
      (Tile.vec (fun i : Fin M => qb * M + i.val)) ∧
  s.regs .nat [D] "offs_d" = some
      (Tile.vec (fun d : Fin D => d.val)) ∧
  -- Loaded Q-row-block.
  s.regs .real [M, D] "q" = some (Tile.ofReal Q) ∧
  -- Streaming accumulators at iter k.
  s.regs .real [M] "m_i" = some
      ⟨fun idx : TileIndex [M] => StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k idx.1⟩ ∧
  s.regs .real [M] "l_i" = some
      (Tile.ofReal fun idx : TileIndex [M] =>
        StreamingAccumulator.lPartial Q numKVBlocks K scale k idx.1) ∧
  s.regs .real [M, D] "o_acc" = some
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        StreamingAccumulator.oPartial Q numKVBlocks K V scale k idx) ∧
  -- Input regions still loaded under the strided addressing.
  InputAt s qReg
      (fun idx : TileIndex [M, D] =>
        batch * sQB + headIdx * sQH + qb * M * sQS
          + idx.1.val * sQS + idx.2.1.val * sQD) Q ∧
  InputAt s kReg
      (fun idx : TileIndex [Bk * numKVBlocks, D] =>
        batch * sKB + headIdx * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD) K ∧
  InputAt s vReg
      (fun idx : TileIndex [Bk * numKVBlocks, D] =>
        batch * sVB + headIdx * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD) V

/-- Loop-carried predicate for the causal strided FA-1 kernel. This is
the causal analogue of `P_fa1_strided`: the address bookkeeping and
loaded Q/K/V premises are unchanged, but the accumulator registers are
interpreted with the causal streaming recurrence parameterized by
`qStart = qb * M`. -/
def P_fa1_strided_causal
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat)
    (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat)
    (sOB sOH _sOM _sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState) : Prop :=
  s.pids 0 = qb ∧ s.pids 1 = headIdx ∧ s.pids 2 = batch ∧
  s.regs .nat [] "pid_qb" = some (Tile.scalar qb) ∧
  s.regs .nat [] "pid_h"  = some (Tile.scalar headIdx) ∧
  s.regs .nat [] "pid_b"  = some (Tile.scalar batch) ∧
  s.regs .nat [] "q_base_off" = some (Tile.scalar (batch * sQB + headIdx * sQH)) ∧
  s.regs .nat [] "k_base_off" = some (Tile.scalar (batch * sKB + headIdx * sKH)) ∧
  s.regs .nat [] "v_base_off" = some (Tile.scalar (batch * sVB + headIdx * sVH)) ∧
  s.regs .nat [] "o_base_off" = some (Tile.scalar (batch * sOB + headIdx * sOH)) ∧
  s.regs .nat [M] "offs_m" = some
      (Tile.vec (fun i : Fin M => qb * M + i.val)) ∧
  s.regs .nat [D] "offs_d" = some
      (Tile.vec (fun d : Fin D => d.val)) ∧
  s.regs .real [M, D] "q" = some (Tile.ofReal Q) ∧
  s.regs .real [M] "m_i" = some
      ⟨fun idx : TileIndex [M] =>
        FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale k idx.1⟩ ∧
  s.regs .real [M] "l_i" = some
      (Tile.ofReal fun idx : TileIndex [M] =>
        FA1MathCausal.lPartial Bk (qb * M) Q numKVBlocks K scale k idx.1) ∧
  s.regs .real [M, D] "o_acc" = some
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        FA1MathCausal.oPartial Bk (qb * M) Q numKVBlocks K V scale k idx) ∧
  InputAt s qReg
      (fun idx : TileIndex [M, D] =>
        batch * sQB + headIdx * sQH + qb * M * sQS
          + idx.1.val * sQS + idx.2.1.val * sQD) Q ∧
  InputAt s kReg
      (fun idx : TileIndex [Bk * numKVBlocks, D] =>
        batch * sKB + headIdx * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD) K ∧
  InputAt s vReg
      (fun idx : TileIndex [Bk * numKVBlocks, D] =>
        batch * sVB + headIdx * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD) V

/-- Loop-carried predicate for FA-1 v1 / boundary-masked strided kernels.

Compared with `P_fa1_strided`, K/V are logical `[S_k, D]` tensors rather
than padded `[Bk * numKVBlocks, D]` tensors, and the running accumulators use
`FA1MathBoundary`. The kernel still iterates over `numKVBlocks` padded
blocks; out-of-range lanes are masked into `⊥` / zero by the recurrence. -/
def P_fa1_strided_boundary
    {M D Bk numKVBlocks S_k : Nat}
    (_qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH _sQS _sQD : Nat)
    (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat)
    (sOB sOH _sOM _sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState) : Prop :=
  s.pids 0 = qb ∧ s.pids 1 = headIdx ∧ s.pids 2 = batch ∧
  s.regs .nat [] "pid_qb" = some (Tile.scalar qb) ∧
  s.regs .nat [] "pid_h"  = some (Tile.scalar headIdx) ∧
  s.regs .nat [] "pid_b"  = some (Tile.scalar batch) ∧
  s.regs .nat [] "q_base_off" = some (Tile.scalar (batch * sQB + headIdx * sQH)) ∧
  s.regs .nat [] "k_base_off" = some (Tile.scalar (batch * sKB + headIdx * sKH)) ∧
  s.regs .nat [] "v_base_off" = some (Tile.scalar (batch * sVB + headIdx * sVH)) ∧
  s.regs .nat [] "o_base_off" = some (Tile.scalar (batch * sOB + headIdx * sOH)) ∧
  s.regs .nat [M] "offs_m" = some
      (Tile.vec (fun i : Fin M => qb * M + i.val)) ∧
  s.regs .nat [D] "offs_d" = some
      (Tile.vec (fun d : Fin D => d.val)) ∧
  s.regs .real [M, D] "q" = some (Tile.ofReal Q) ∧
  s.regs .real [M] "m_i" = some
      ⟨fun idx : TileIndex [M] =>
        FA1MathBoundary.mPartial Bk Q numKVBlocks K scale k idx.1⟩ ∧
  s.regs .real [M] "l_i" = some
      (Tile.ofReal fun idx : TileIndex [M] =>
        FA1MathBoundary.lPartial Bk Q numKVBlocks K scale k idx.1) ∧
  s.regs .real [M, D] "o_acc" = some
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        FA1MathBoundary.oPartial Bk Q numKVBlocks K V scale k idx) ∧
  InputAt s kReg
      (fun idx : TileIndex [S_k, D] =>
        batch * sKB + headIdx * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD) K ∧
  InputAt s vReg
      (fun idx : TileIndex [S_k, D] =>
        batch * sVB + headIdx * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD) V

/-- Loop-carried predicate for causal boundary-masked strided kernels.

This combines `P_fa1_strided_boundary`'s logical `[S_k, D]` K/V memory
contract with the causal streaming recurrence parameterized by
`qStart = qb * M`. -/
def P_fa1_strided_causal_boundary
    {M D Bk numKVBlocks S_k : Nat}
    (_qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH _sQS _sQD : Nat)
    (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat)
    (sOB sOH _sOM _sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState) : Prop :=
  s.pids 0 = qb ∧ s.pids 1 = headIdx ∧ s.pids 2 = batch ∧
  s.regs .nat [] "pid_qb" = some (Tile.scalar qb) ∧
  s.regs .nat [] "pid_h"  = some (Tile.scalar headIdx) ∧
  s.regs .nat [] "pid_b"  = some (Tile.scalar batch) ∧
  s.regs .nat [] "q_base_off" = some (Tile.scalar (batch * sQB + headIdx * sQH)) ∧
  s.regs .nat [] "k_base_off" = some (Tile.scalar (batch * sKB + headIdx * sKH)) ∧
  s.regs .nat [] "v_base_off" = some (Tile.scalar (batch * sVB + headIdx * sVH)) ∧
  s.regs .nat [] "o_base_off" = some (Tile.scalar (batch * sOB + headIdx * sOH)) ∧
  s.regs .nat [M] "offs_m" = some
      (Tile.vec (fun i : Fin M => qb * M + i.val)) ∧
  s.regs .nat [D] "offs_d" = some
      (Tile.vec (fun d : Fin D => d.val)) ∧
  s.regs .real [M, D] "q" = some (Tile.ofReal Q) ∧
  s.regs .real [M] "m_i" = some
      ⟨fun idx : TileIndex [M] =>
        FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale k idx.1⟩ ∧
  s.regs .real [M] "l_i" = some
      (Tile.ofReal fun idx : TileIndex [M] =>
        FA1MathCausalBoundary.lPartial Bk (qb * M) Q numKVBlocks K scale k idx.1) ∧
  s.regs .real [M, D] "o_acc" = some
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        FA1MathCausalBoundary.oPartial Bk (qb * M) Q numKVBlocks K V scale k idx) ∧
  InputAt s kReg
      (fun idx : TileIndex [S_k, D] =>
        batch * sKB + headIdx * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD) K ∧
  InputAt s vReg
      (fun idx : TileIndex [S_k, D] =>
        batch * sVB + headIdx * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD) V

/-- D-tail boundary invariant: run the existing boundary invariant at block
hidden width `Bd`, with Q/K/V register/math state zero-padded outside `D`.
Unlike the plain padded lift, the K/V memory contract remains logical
`[S_k, D]`; out-of-D lanes are supplied by masked loads, not by memory. -/
def P_fa1_strided_boundaryD
    {M D Bd Bk numKVBlocks S_k : Nat}
    (_qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH _sQS _sQD : Nat)
    (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat)
    (sOB sOH _sOM _sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState) : Prop :=
  s.pids 0 = qb ∧ s.pids 1 = headIdx ∧ s.pids 2 = batch ∧
  s.regs .nat [] "pid_qb" = some (Tile.scalar qb) ∧
  s.regs .nat [] "pid_h"  = some (Tile.scalar headIdx) ∧
  s.regs .nat [] "pid_b"  = some (Tile.scalar batch) ∧
  s.regs .nat [] "q_base_off" = some (Tile.scalar (batch * sQB + headIdx * sQH)) ∧
  s.regs .nat [] "k_base_off" = some (Tile.scalar (batch * sKB + headIdx * sKH)) ∧
  s.regs .nat [] "v_base_off" = some (Tile.scalar (batch * sVB + headIdx * sVH)) ∧
  s.regs .nat [] "o_base_off" = some (Tile.scalar (batch * sOB + headIdx * sOH)) ∧
  s.regs .nat [M] "offs_m" = some
      (Tile.vec (fun i : Fin M => qb * M + i.val)) ∧
  s.regs .nat [Bd] "offs_d" = some
      (Tile.vec (fun d : Fin Bd => d.val)) ∧
  s.regs .real [M, Bd] "q" = some (Tile.ofReal (padHeadD (Bd := Bd) Q)) ∧
  s.regs .real [M] "m_i" = some
      ⟨fun idx : TileIndex [M] =>
        FA1MathBoundary.mPartial Bk (padHeadD (Bd := Bd) Q) numKVBlocks
          (padHeadD (Bd := Bd) K) scale k idx.1⟩ ∧
  s.regs .real [M] "l_i" = some
      (Tile.ofReal fun idx : TileIndex [M] =>
        FA1MathBoundary.lPartial Bk (padHeadD (Bd := Bd) Q) numKVBlocks
          (padHeadD (Bd := Bd) K) scale k idx.1) ∧
  s.regs .real [M, Bd] "o_acc" = some
      (Tile.ofReal fun idx : TileIndex [M, Bd] =>
        FA1MathBoundary.oPartial Bk (padHeadD (Bd := Bd) Q) numKVBlocks
          (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V) scale k idx) ∧
  InputAt s kReg
      (fun idx : TileIndex [S_k, D] =>
        batch * sKB + headIdx * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD) K ∧
  InputAt s vReg
      (fun idx : TileIndex [S_k, D] =>
        batch * sVB + headIdx * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD) V

/-- D-tail causal-boundary invariant: same padded hidden-width lift as
`P_fa1_strided_boundaryD`, but with the causal-boundary recurrence. -/
def P_fa1_strided_causal_boundaryD
    {M D Bd Bk numKVBlocks S_k : Nat}
    (_qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH _sQS _sQD : Nat)
    (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat)
    (sOB sOH _sOM _sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState) : Prop :=
  s.pids 0 = qb ∧ s.pids 1 = headIdx ∧ s.pids 2 = batch ∧
  s.regs .nat [] "pid_qb" = some (Tile.scalar qb) ∧
  s.regs .nat [] "pid_h"  = some (Tile.scalar headIdx) ∧
  s.regs .nat [] "pid_b"  = some (Tile.scalar batch) ∧
  s.regs .nat [] "q_base_off" = some (Tile.scalar (batch * sQB + headIdx * sQH)) ∧
  s.regs .nat [] "k_base_off" = some (Tile.scalar (batch * sKB + headIdx * sKH)) ∧
  s.regs .nat [] "v_base_off" = some (Tile.scalar (batch * sVB + headIdx * sVH)) ∧
  s.regs .nat [] "o_base_off" = some (Tile.scalar (batch * sOB + headIdx * sOH)) ∧
  s.regs .nat [M] "offs_m" = some
      (Tile.vec (fun i : Fin M => qb * M + i.val)) ∧
  s.regs .nat [Bd] "offs_d" = some
      (Tile.vec (fun d : Fin Bd => d.val)) ∧
  s.regs .real [M, Bd] "q" = some (Tile.ofReal (padHeadD (Bd := Bd) Q)) ∧
  s.regs .real [M] "m_i" = some
      ⟨fun idx : TileIndex [M] =>
        FA1MathCausalBoundary.mPartial Bk (qb * M) (padHeadD (Bd := Bd) Q)
          numKVBlocks (padHeadD (Bd := Bd) K) scale k idx.1⟩ ∧
  s.regs .real [M] "l_i" = some
      (Tile.ofReal fun idx : TileIndex [M] =>
        FA1MathCausalBoundary.lPartial Bk (qb * M) (padHeadD (Bd := Bd) Q)
          numKVBlocks (padHeadD (Bd := Bd) K) scale k idx.1) ∧
  s.regs .real [M, Bd] "o_acc" = some
      (Tile.ofReal fun idx : TileIndex [M, Bd] =>
        FA1MathCausalBoundary.oPartial Bk (qb * M) (padHeadD (Bd := Bd) Q)
          numKVBlocks (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
          scale k idx) ∧
  InputAt s kReg
      (fun idx : TileIndex [S_k, D] =>
        batch * sKB + headIdx * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD) K ∧
  InputAt s vReg
      (fun idx : TileIndex [S_k, D] =>
        batch * sVB + headIdx * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD) V
/-! ## attentionReal4D bridge

The slice-form result of `fa1_forward_correct_4D_slice` equals the
user-facing `attentionReal4D` spec at the corresponding 4D index.
The bridge factors through `attentionReal_row_eq` (`attentionReal`'s
output at row `i` depends only on `Q`'s row at `i`), reusing the
fact that K and V slices are byte-identical (modulo Fin proof
irrelevance) under `Bk * numKVBlocks = S_k`.

Three private helpers do the row-factoring through the `Tile.dot` /
`softmaxRow` / `attention` layers; `attentionReal_row_eq` ties them
together at the user-facing `Tile.ofReal`-lifted level. -/

/-- `Tile.dot`'s output at row `i` of the first argument depends only
on that row's data. -/
private theorem dot_data_row_eq {M1 M2 K N : Nat}
    (a1 : Tile .real [M1, K]) (a2 : Tile .real [M2, K])
    (b : Tile .real [K, N])
    (i1 : Fin M1) (i2 : Fin M2) (n : Fin N)
    (hRow : ∀ k, a1.data (i1, k, PUnit.unit) = a2.data (i2, k, PUnit.unit)) :
    (Tile.dot [] a1 b).data (i1, n, PUnit.unit) =
    (Tile.dot [] a2 b).data (i2, n, PUnit.unit) := by
  rw [Tile.dot_nil_data, Tile.dot_nil_data]
  apply Finset.sum_congr rfl
  intro k _
  rw [hRow k]

/-- `softmaxRow`'s output at row `i` depends only on that row's data. -/
private theorem softmaxRow_data_row_eq {M1 M2 N : Nat}
    (s1 : Tile .real [M1, N]) (s2 : Tile .real [M2, N])
    (i1 : Fin M1) (i2 : Fin M2) (n : Fin N)
    (hRow : ∀ n', s1.data (i1, n', PUnit.unit) = s2.data (i2, n', PUnit.unit)) :
    (softmaxRow s1).data (i1, n, PUnit.unit) =
    (softmaxRow s2).data (i2, n, PUnit.unit) := by
  unfold softmaxRow
  simp only []
  show some _ = some _
  congr 2
  · rw [hRow n]
  · apply Finset.sum_congr rfl
    intro j _
    rw [hRow j]

/-- `attention`'s output at row `i` of `Q` depends only on that
row's data (and on `K`, `V`, `scale`). -/
private theorem attention_data_row_eq {M1 M2 S D : Nat}
    (Q1 : Tile .real [M1, D]) (Q2 : Tile .real [M2, D])
    (K V : Tile .real [S, D]) (scale : ℝ)
    (i1 : Fin M1) (i2 : Fin M2) (d : Fin D)
    (hRow : ∀ d' : Fin D, Q1.data (i1, d', PUnit.unit) = Q2.data (i2, d', PUnit.unit)) :
    (attention Q1 K V scale).data (i1, d, PUnit.unit) =
    (attention Q2 K V scale).data (i2, d, PUnit.unit) := by
  unfold attention
  simp only []
  apply dot_data_row_eq
  intro k
  apply softmaxRow_data_row_eq
  intro n'
  show Option.map (· * scale) ((Tile.dot [] Q1 (Tile.transpose [] K)).data (i1, n', PUnit.unit)) =
       Option.map (· * scale) ((Tile.dot [] Q2 (Tile.transpose [] K)).data (i2, n', PUnit.unit))
  congr 1
  exact dot_data_row_eq Q1 Q2 (Tile.transpose [] K) i1 i2 n' hRow

/-- Row-invariance of `attentionReal`. Two `attentionReal` calls (with
possibly different M dimensions on `Q`) produce the same value at
indices `(i1, d)` and `(i2, d)` whenever `Q1`'s row at `i1` equals
`Q2`'s row at `i2`. The K, V, scale arguments are the same. -/
theorem attentionReal_row_eq {M1 M2 S D : Nat}
    (Q1 : TileIndex [M1, D] → ℝ) (Q2 : TileIndex [M2, D] → ℝ)
    (K V : TileIndex [S, D] → ℝ) (scale : ℝ)
    (i1 : Fin M1) (i2 : Fin M2) (d : Fin D)
    (hRow : ∀ d' : Fin D, Q1 (i1, d', PUnit.unit) = Q2 (i2, d', PUnit.unit)) :
    attentionReal Q1 K V scale (i1, d, PUnit.unit)
      = attentionReal Q2 K V scale (i2, d, PUnit.unit) := by
  unfold attentionReal VeriTile.Triton.attentionReal VeriTile.Triton.scaledScore
  simp [hRow]

/-- Bridge: the slice-form `attentionReal` produced by FA-1's strided
kernel equals the `attentionReal4D` spec at the corresponding 4D
index. The proof factors through `attentionReal_row_eq`: K, V slices
become byte-identical to `sliceBH` after substituting `hSk`, so only
the Q-row equality remains, and that is `rfl` by `slice4DQRows`'s
definition. -/
theorem attentionReal_slice_eq_attentionReal4D
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hSk : Bk * numKVBlocks = S_k)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ)
    (b : Fin B) (h : Fin H) (qb : Nat)
    (hQBnd : qb * M + M ≤ S_q)
    (idx : TileIndex [M, D]) :
    attentionReal
        (slice4DQRows M Q4D b h qb hQBnd)
        (slice4DFlat Bk numKVBlocks K4D b h hSk)
        (slice4DFlat Bk numKVBlocks V4D b h hSk)
        scale idx
      = attentionReal4D Q4D K4D V4D scale
          (b, h, ⟨qb * M + idx.1.val, by have := idx.1.isLt; omega⟩,
           idx.2.1, PUnit.unit) := by
  obtain ⟨i, d, _⟩ := idx
  rw [attentionReal4D_slice]
  subst hSk
  have hKeq : slice4DFlat Bk numKVBlocks K4D b h rfl = sliceBH K4D b h := by
    funext idx; obtain ⟨j, d', _⟩ := idx; rfl
  have hVeq : slice4DFlat Bk numKVBlocks V4D b h rfl = sliceBH V4D b h := by
    funext idx; obtain ⟨j, d', _⟩ := idx; rfl
  rw [hKeq, hVeq]
  apply attentionReal_row_eq
  intro d'
  rfl

/-- Causal bridge: the block-local causal spec with global query start
`qb * M` is exactly the user-facing 4D causal spec at the corresponding
global row. This is the causal analogue of
`attentionReal_slice_eq_attentionReal4D`, but it does not need the
row-invariance lemmas because `attentionRealCausalBlock` already carries
the global query offset in its mask. -/
theorem attentionRealCausalBlock_slice_eq_attentionReal4DCausal
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hSk : Bk * numKVBlocks = S_k)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ)
    (b : Fin B) (h : Fin H) (qb : Nat)
    (hQBnd : qb * M + M ≤ S_q)
    (idx : TileIndex [M, D]) :
    attentionRealCausalBlock (qb * M)
        (slice4DQRows M Q4D b h qb hQBnd)
        (slice4DFlat Bk numKVBlocks K4D b h hSk)
        (slice4DFlat Bk numKVBlocks V4D b h hSk)
        scale idx
      = attentionReal4DCausal Q4D K4D V4D scale
          (b, h, ⟨qb * M + idx.1.val, by have := idx.1.isLt; omega⟩,
           idx.2.1, PUnit.unit) := by
  obtain ⟨i, d, _⟩ := idx
  rw [attentionReal4DCausal_slice]
  subst hSk
  unfold attentionRealCausalBlock attentionRealCausal
  simp [slice4DQRows, slice4DFlat, sliceBH]

end VeriTile.Examples
