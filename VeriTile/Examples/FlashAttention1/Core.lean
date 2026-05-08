/-
VeriTile.Examples.FlashAttention1.Core

FlashAttention-1 v0/full-tile proof surface.
-/

import VeriTile.Triton.Float
import VeriTile.Examples.FlashAttention1.Core.Forward

namespace VeriTile.Examples

open VeriTile.Triton

/-- 4D-aware corollary of `fa1_forward_correct_strided`. Given inputs
laid out via `Offset.strided` over `[B, H, S, D]` (with valid strides
producing a non-overlapping memory layout) and a Q-side boundary
hypothesis ensuring the M-row block fits within `S_q`, the strided
FA-1 kernel produces `attentionReal` of the per-`(b, h, q_block)`
slice.

Result is in slice form (`attentionReal` of `slice4DQRows` /
`slice4DFlat`) rather than `attentionReal4D`. This theorem is the
intermediate proof that composes directly with the strided inner
theorem; `fa1_forward_correct_4D` below packages it with
`attentionReal_slice_eq_attentionReal4D` to expose the user-facing
`attentionReal4D` statement. -/
theorem fa1_forward_correct_4D_slice
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (hSk : Bk * numKVBlocks = S_k)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQBnd : s.pids 0 * M + M ≤ S_q)
    (hQ4D : InputAt s qReg
        (Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0) Q4D)
    (hK4D : InputAt s kReg
        (Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0) K4D)
    (hV4D : InputAt s vReg
        (Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0) V4D)
    (hOValid : Offset.StridesValid [B, H, S_q, D] [sOB, sOH, sOM, sOD]) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernelStrided qReg kReg vReg outReg M D Bk numKVBlocks
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal
                (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
                  (s.pids 0) hQBnd)
                (slice4DFlat Bk numKVBlocks K4D
                  ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
                (slice4DFlat Bk numKVBlocks V4D
                  ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
                scale idx) := by
  intro idx
  -- Convert hQ4D's 4D Offset.strided premise to inner-theorem tile-local form.
  have hQ_inner : InputAt s qReg
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
          + idx.1.val * sQS + idx.2.1.val * sQD)
      (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
        (s.pids 0) hQBnd) := by
    intro tileIdx
    obtain ⟨i, d, _⟩ := tileIdx
    have h := hQ4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨s.pids 0 * M + i.val, by have := i.isLt; omega⟩,
                    d, PUnit.unit)
    show s.readMem qReg
      (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
        + i.val * sQS + d.val * sQD) = _
    rw [show s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + i.val * sQS + d.val * sQD =
          Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + i.val, by have := i.isLt; omega⟩,
             d, PUnit.unit) by
        simp [Offset.strided, Nat.add_mul]
        ring]
    exact h
  -- Convert hK4D similarly.
  have hK_inner : InputAt s kReg
      (fun idx : TileIndex [Bk * numKVBlocks, D] =>
        s.pids 2 * sKB + s.pids 1 * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD)
      (slice4DFlat Bk numKVBlocks K4D
        ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hK4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨j.val, by have := j.isLt; omega⟩,
                    d, PUnit.unit)
    show s.readMem kReg
      (s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD) = _
    rw [show s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD =
          Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨j.val, by have := j.isLt; omega⟩, d, PUnit.unit) by
        simp [Offset.strided]]
    exact h
  -- Convert hV4D similarly.
  have hV_inner : InputAt s vReg
      (fun idx : TileIndex [Bk * numKVBlocks, D] =>
        s.pids 2 * sVB + s.pids 1 * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD)
      (slice4DFlat Bk numKVBlocks V4D
        ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hV4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨j.val, by have := j.isLt; omega⟩,
                    d, PUnit.unit)
    show s.readMem vReg
      (s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD) = _
    rw [show s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD =
          Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨j.val, by have := j.isLt; omega⟩, d, PUnit.unit) by
        simp [Offset.strided]]
    exact h
  -- Output tile-local injectivity from `Offset.strided_inj`. The 4D
  -- `StridesValid` decomposes into four nested ∧'s; the third clause
  -- gives `(D-1)*sOD < sOM` and the fourth gives `0 < sOD`, which
  -- together form `Offset.StridesValid [M, D] [sOM, sOD]`.
  have hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD) := by
    have hOValidLocal : Offset.StridesValid [M, D] [sOM, sOD] :=
      ⟨hOValid.2.2.1, hOValid.2.2.2.1, trivial⟩
    have hStrInj := Offset.strided_inj
        (s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM)
        hOValidLocal
    intro a b hab
    apply hStrInj
    obtain ⟨a₁, a₂, _⟩ := a
    obtain ⟨b₁, b₂, _⟩ := b
    show Offset.strided [M, D] [sOM, sOD] _ _ =
         Offset.strided [M, D] [sOM, sOD] _ _
    simp only [Offset.strided]
    have := hab
    simp only at this
    omega
  exact fa1_forward_correct_strided hBk hNumKVBlocks
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD
    (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ (s.pids 0) hQBnd)
    (slice4DFlat Bk numKVBlocks K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
    (slice4DFlat Bk numKVBlocks V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
    scale s
    hQ_inner hK_inner hV_inner hInj idx

/-- Boundary 4D-aware corollary of `fa1_forward_correct_strided_boundary`.
Mirrors `fa1_forward_correct_4D_slice` but uses the boundary kernel and the
boundary-masked Q-row slicer. K and V are sliced directly via `sliceBH`
(shape `[S_k, D]`) rather than `slice4DFlat`, since the boundary kernel
already takes K/V on the logical `[S_k, D]` domain.

Differences from the non-boundary version:
* No `hQBnd : s.pids 0 * M + M ≤ S_q`: the boundary kernel's store mask
  handles the partial Q-row tail. Instead the conclusion is per-`idx`,
  guarded by `s.pids 0 * M + idx.1.val < S_q`.
* `hSk : Bk * numKVBlocks = S_k` becomes `hSkLe : S_k ≤ Bk * numKVBlocks`
  (the boundary kernel only needs the cover-all-of-K/V condition).
* `hSk : 0 < S_k` is required by `streaming_eq_attentionReal` to ensure
  the running normalizer is non-zero. -/
theorem fa1_forward_correct_4D
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (hSk : Bk * numKVBlocks = S_k)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQBnd : s.pids 0 * M + M ≤ S_q)
    (hQ4D : InputAt s qReg
        (Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0) Q4D)
    (hK4D : InputAt s kReg
        (Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0) K4D)
    (hV4D : InputAt s vReg
        (Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0) V4D)
    (hOValid : Offset.StridesValid [B, H, S_q, D] [sOB, sOH, sOM, sOD]) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernelStrided qReg kReg vReg outReg M D Bk numKVBlocks
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal4D Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
             idx.2.1, PUnit.unit)) := by
  intro idx
  rw [fa1_forward_correct_4D_slice hBk hNumKVBlocks hSk
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        Q4D K4D V4D scale s hPidB hPidH hQBnd hQ4D hK4D hV4D hOValid idx]
  congr 1
  exact attentionReal_slice_eq_attentionReal4D hSk Q4D K4D V4D scale
    ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ (s.pids 0) hQBnd idx

/-- Boundary FA-1 forward correctness in user-facing `attentionReal4D`
form. Bundle of `fa1_forward_correct_4D_boundary_slice` plus an inline
`attentionReal_row_eq` bridge: the same statement, with the slice-form
result re-expressed at the corresponding 4D index of
`attentionReal4D Q4D K4D V4D scale`.

Unlike the non-boundary `_4D` theorem, the conclusion is guarded by
the per-row bound `s.pids 0 * M + idx.1.val < S_q`. The bridge cannot
reuse `attentionReal_slice_eq_attentionReal4D` directly because the
boundary K/V slice uses `sliceBH` (shape `[S_k, D]`) instead of
`slice4DFlat` (shape `[Bk * numKVBlocks, D]`), so the row-equality is
derived inline via `attentionReal_row_eq`. -/
theorem fa1_forward_correct_4D_causal_raw_of_step
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hSk : Bk * numKVBlocks = S_k)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQBnd : s.pids 0 * M + M ≤ S_q)
    (hQ4D : InputAt s qReg
        (Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0) Q4D)
    (hK4D : InputAt s kReg
        (Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0) K4D)
    (hV4D : InputAt s vReg
        (Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0) V4D)
    (hOValid : Offset.StridesValid [B, H, S_q, D] [sOB, sOH, sOM, sOD])
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_strided_causal qReg kReg vReg
          (s.pids 0) (s.pids 1) (s.pids 2)
          sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
          sOB sOH sOM sOD
          (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
            (s.pids 0) hQBnd)
          (slice4DFlat Bk numKVBlocks K4D
            ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
          (slice4DFlat Bk numKVBlocks V4D
            ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
          scale i st →
        ∃ st',
          stepStmts (fa1LoopBodyStridedCausal kReg vReg M D Bk sKN sKD sVN sVD scale)
            (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
          P_fa1_strided_causal qReg kReg vReg
            (s.pids 0) (s.pids 1) (s.pids 2)
            sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
            sOB sOH sOM sOD
            (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
              (s.pids 0) hQBnd)
            (slice4DFlat Bk numKVBlocks K4D
              ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
            (slice4DFlat Bk numKVBlocks V4D
              ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
            scale (i + 1) st') :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernelStridedCausal qReg kReg vReg outReg M D Bk numKVBlocks
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausal.oPartial Bk (s.pids 0 * M)
                (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
                  (s.pids 0) hQBnd)
                numKVBlocks
                (slice4DFlat Bk numKVBlocks K4D
                  ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
                (slice4DFlat Bk numKVBlocks V4D
                  ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
                scale numKVBlocks idx /
              FA1MathCausal.lPartial Bk (s.pids 0 * M)
                (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
                  (s.pids 0) hQBnd)
                numKVBlocks
                (slice4DFlat Bk numKVBlocks K4D
                  ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
                scale numKVBlocks idx.1) := by
  intro idx
  have hQ_inner : InputAt s qReg
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
          + idx.1.val * sQS + idx.2.1.val * sQD)
      (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
        (s.pids 0) hQBnd) := by
    intro tileIdx
    obtain ⟨i, d, _⟩ := tileIdx
    have h := hQ4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨s.pids 0 * M + i.val, by have := i.isLt; omega⟩,
                    d, PUnit.unit)
    show s.readMem qReg
      (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
        + i.val * sQS + d.val * sQD) = _
    rw [show s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + i.val * sQS + d.val * sQD =
          Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + i.val, by have := i.isLt; omega⟩,
             d, PUnit.unit) by
        simp [Offset.strided, Nat.add_mul]
        ring]
    exact h
  have hK_inner : InputAt s kReg
      (fun idx : TileIndex [Bk * numKVBlocks, D] =>
        s.pids 2 * sKB + s.pids 1 * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD)
      (slice4DFlat Bk numKVBlocks K4D
        ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hK4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨j.val, by have := j.isLt; omega⟩,
                    d, PUnit.unit)
    show s.readMem kReg
      (s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD) = _
    rw [show s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD =
          Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨j.val, by have := j.isLt; omega⟩, d, PUnit.unit) by
        simp [Offset.strided]]
    exact h
  have hV_inner : InputAt s vReg
      (fun idx : TileIndex [Bk * numKVBlocks, D] =>
        s.pids 2 * sVB + s.pids 1 * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD)
      (slice4DFlat Bk numKVBlocks V4D
        ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hV4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨j.val, by have := j.isLt; omega⟩,
                    d, PUnit.unit)
    show s.readMem vReg
      (s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD) = _
    rw [show s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD =
          Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨j.val, by have := j.isLt; omega⟩, d, PUnit.unit) by
        simp [Offset.strided]]
    exact h
  have hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD) := by
    have hOValidLocal : Offset.StridesValid [M, D] [sOM, sOD] :=
      ⟨hOValid.2.2.1, hOValid.2.2.2.1, trivial⟩
    have hStrInj := Offset.strided_inj
        (s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM)
        hOValidLocal
    intro a b hab
    apply hStrInj
    obtain ⟨a₁, a₂, _⟩ := a
    obtain ⟨b₁, b₂, _⟩ := b
    show Offset.strided [M, D] [sOM, sOD] _ _ =
         Offset.strided [M, D] [sOM, sOD] _ _
    simp only [Offset.strided]
    have := hab
    simp only at this
    omega
  exact fa1_forward_correct_strided_causal_raw_of_step
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD
    (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ (s.pids 0) hQBnd)
    (slice4DFlat Bk numKVBlocks K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
    (slice4DFlat Bk numKVBlocks V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
    scale s hQ_inner hK_inner hV_inner hInj hStep idx

/-- 4D causal FA-1 forward correctness in raw streaming form. This
discharges the causal loop-step obligation using
`fa1_step_strided_causal`, so callers no longer need to thread a
manual invariant-preservation proof. -/
theorem fa1_forward_correct_4D_causal_raw
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk)
    (hSk : Bk * numKVBlocks = S_k)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQBnd : s.pids 0 * M + M ≤ S_q)
    (hQ4D : InputAt s qReg
        (Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0) Q4D)
    (hK4D : InputAt s kReg
        (Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0) K4D)
    (hV4D : InputAt s vReg
        (Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0) V4D)
    (hOValid : Offset.StridesValid [B, H, S_q, D] [sOB, sOH, sOM, sOD]) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernelStridedCausal qReg kReg vReg outReg M D Bk numKVBlocks
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausal.oPartial Bk (s.pids 0 * M)
                (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
                  (s.pids 0) hQBnd)
                numKVBlocks
                (slice4DFlat Bk numKVBlocks K4D
                  ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
                (slice4DFlat Bk numKVBlocks V4D
                  ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
                scale numKVBlocks idx /
              FA1MathCausal.lPartial Bk (s.pids 0 * M)
                (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
                  (s.pids 0) hQBnd)
                numKVBlocks
                (slice4DFlat Bk numKVBlocks K4D
                  ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
                scale numKVBlocks idx.1) := by
  exact fa1_forward_correct_4D_causal_raw_of_step hSk
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD Q4D K4D V4D scale s
    hPidB hPidH hQBnd hQ4D hK4D hV4D hOValid
    (fun i st hi hPi =>
      fa1_step_strided_causal hBk qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
          (s.pids 0) hQBnd)
        (slice4DFlat Bk numKVBlocks K4D
          ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
        (slice4DFlat Bk numKVBlocks V4D
          ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
        scale i st hi hPi)

/-- 4D causal FA-1 forward correctness, stated directly against the
user-facing `attentionReal4DCausal` spec. This is the fully cleaned
causal theorem: no external step obligation and no raw accumulator
ratio in the conclusion. -/
theorem fa1_forward_correct_4D_causal
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (hSk : Bk * numKVBlocks = S_k)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQBnd : s.pids 0 * M + M ≤ S_q)
    (hQ4D : InputAt s qReg
        (Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0) Q4D)
    (hK4D : InputAt s kReg
        (Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0) K4D)
    (hV4D : InputAt s vReg
        (Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0) V4D)
    (hOValid : Offset.StridesValid [B, H, S_q, D] [sOB, sOH, sOM, sOD]) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (fa1ForwardKernelStridedCausal qReg kReg vReg outReg M D Bk numKVBlocks
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal4DCausal Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
             idx.2.1, PUnit.unit)) := by
  intro idx
  rw [fa1_forward_correct_4D_causal_raw hBk hSk
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q4D K4D V4D scale s
        hPidB hPidH hQBnd hQ4D hK4D hV4D hOValid idx]
  congr 1
  rw [FA1MathCausal.streaming_eq_attentionRealCausalBlock hBk
        (s.pids 0 * M)
        (slice4DQRows M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
          (s.pids 0) hQBnd)
        numKVBlocks hNumKVBlocks
        (slice4DFlat Bk numKVBlocks K4D
          ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
        (slice4DFlat Bk numKVBlocks V4D
          ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ hSk)
        scale idx]
  exact attentionRealCausalBlock_slice_eq_attentionReal4DCausal hSk Q4D K4D V4D scale
    ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ (s.pids 0) hQBnd idx

/-! ## Layout-level theorem surface

These are the user-facing wrappers over the final 4D theorems above. They
bundle the sixteen Q/K/V/O stride arguments into `FA1Layout4D`, expose
named offset helpers for the `InputAt` premises, and keep the conclusion in
the same `attentionReal4D` / `attentionReal4DCausal` form. The input
premises are stated through `TensorView.loaded`, which is the memory-contract
surface users should normally see. -/

/-- FA-1 forward correctness over a bundled 4D layout. This is the same
statement as `fa1_forward_correct_4D`, but with the stride plumbing hidden
behind `FA1Layout4D`. -/
theorem fa1_forward_correct_4D_layout
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (hSk : Bk * numKVBlocks = S_k)
    (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQBnd : s.pids 0 * M + M ≤ S_q)
    (hQ4D : TensorView.loaded s (layout.qView qReg) Q4D)
    (hK4D : TensorView.loaded s (layout.kView kReg) K4D)
    (hV4D : TensorView.loaded s (layout.vView vReg) V4D) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (layout.kernel qReg kReg vReg outReg M Bk numKVBlocks scale) s)
          outReg (layout.outBlockOffset s M) idx
        = some (attentionReal4D Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
             idx.2.1, PUnit.unit)) := by
  intro idx
  simpa [FA1Layout4D.kernel, FA1Layout4D.qOffset, FA1Layout4D.kOffset,
         FA1Layout4D.vOffset, FA1Layout4D.outBlockOffset,
         FA1Layout4D.qView, FA1Layout4D.kView, FA1Layout4D.vView,
         TensorView.loaded, TensorView.offset,
         FA1Layout4D.qStrides, FA1Layout4D.kStrides,
         FA1Layout4D.vStrides, FA1Layout4D.oStrides]
    using fa1_forward_correct_4D hBk hNumKVBlocks hSk
      qReg kReg vReg outReg
      layout.qB layout.qH layout.qS layout.qD
      layout.kB layout.kH layout.kS layout.kD
      layout.vB layout.vH layout.vS layout.vD
      layout.oB layout.oH layout.oS layout.oD
      Q4D K4D V4D scale s hPidB hPidH hQBnd
      hQ4D hK4D hV4D layout.hOValid idx

/-- Causal FA-1 forward correctness over a bundled 4D layout. This is the
layout-level version of `fa1_forward_correct_4D_causal`. -/
theorem fa1_forward_correct_4D_causal_layout
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (hSk : Bk * numKVBlocks = S_k)
    (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQBnd : s.pids 0 * M + M ≤ S_q)
    (hQ4D : TensorView.loaded s (layout.qView qReg) Q4D)
    (hK4D : TensorView.loaded s (layout.kView kReg) K4D)
    (hV4D : TensorView.loaded s (layout.vView vReg) V4D) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (layout.causalKernel qReg kReg vReg outReg M Bk numKVBlocks scale) s)
          outReg (layout.outBlockOffset s M) idx
        = some (attentionReal4DCausal Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
             idx.2.1, PUnit.unit)) := by
  intro idx
  simpa [FA1Layout4D.causalKernel, FA1Layout4D.qOffset, FA1Layout4D.kOffset,
         FA1Layout4D.vOffset, FA1Layout4D.outBlockOffset,
         FA1Layout4D.qView, FA1Layout4D.kView, FA1Layout4D.vView,
         TensorView.loaded, TensorView.offset,
         FA1Layout4D.qStrides, FA1Layout4D.kStrides,
         FA1Layout4D.vStrides, FA1Layout4D.oStrides]
    using fa1_forward_correct_4D_causal hBk hNumKVBlocks hSk
      qReg kReg vReg outReg
      layout.qB layout.qH layout.qS layout.qD
      layout.kB layout.kH layout.kS layout.kD
      layout.vB layout.vH layout.vS layout.vD
      layout.oB layout.oH layout.oS layout.oD
      Q4D K4D V4D scale s hPidB hPidH hQBnd
      hQ4D hK4D hV4D layout.hOValid idx

theorem fa1_forward_correct_4D_exec_views
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
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (views.kernel M Bk numKVBlocks scale) s)
          views.outReg (views.outBlockOffset s M) idx
        = some (attentionReal4D Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
             idx.2.1, PUnit.unit)) := by
  intro idx
  simpa [FA1Views4D.kernel, FA1Views4D.outBlockOffset,
         FA1Views4D.qView, FA1Views4D.kView, FA1Views4D.vView]
    using fa1_forward_correct_4D_layout hBk hNumKVBlocks hSk views.layout
      views.qReg views.kReg views.vReg views.outReg
      Q4D K4D V4D scale s hPidB hPidH hQBnd
      hQ4D hK4D hV4D idx

theorem fa1_forward_correct_4D_views
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
      (kernel := views.kernel M Bk numKVBlocks scale)
      (initialState := s)
      (write := fun idx : TileIndex [M, D] =>
        some (views.outReg, (views.outBlockOffset s M) idx))
      (expected := fun idx : TileIndex [M, D] =>
        attentionReal4D Q4D K4D V4D scale
          (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
           ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
           idx.2.1, PUnit.unit)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have hview := fa1_forward_correct_4D_exec_views hBk hNumKVBlocks hSk
    views Q4D K4D V4D scale s hPidB hPidH hQBnd hQ4D hK4D hV4D idx
  rw [hExec] at hview
  simpa [observeTileAt] using hview

/-- Causal FA-1 forward correctness over bundled tensor views. -/
theorem fa1_forward_correct_4D_causal_exec_views
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
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec (views.causalKernel M Bk numKVBlocks scale) s)
          views.outReg (views.outBlockOffset s M) idx
        = some (attentionReal4DCausal Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
             idx.2.1, PUnit.unit)) := by
  intro idx
  simpa [FA1Views4D.causalKernel, FA1Views4D.outBlockOffset,
         FA1Views4D.qView, FA1Views4D.kView, FA1Views4D.vView]
    using fa1_forward_correct_4D_causal_layout hBk hNumKVBlocks hSk views.layout
      views.qReg views.kReg views.vReg views.outReg
      Q4D K4D V4D scale s hPidB hPidH hQBnd
      hQ4D hK4D hV4D idx

/-- Compute-facing causal FA-1 forward correctness over bundled tensor views. -/
theorem fa1_forward_correct_4D_causal_views
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
      (kernel := views.causalKernel M Bk numKVBlocks scale)
      (initialState := s)
      (write := fun idx : TileIndex [M, D] =>
        some (views.outReg, (views.outBlockOffset s M) idx))
      (expected := fun idx : TileIndex [M, D] =>
        attentionReal4DCausal Q4D K4D V4D scale
          (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
           ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
           idx.2.1, PUnit.unit)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have hview := fa1_forward_correct_4D_causal_exec_views hBk hNumKVBlocks hSk
    views Q4D K4D V4D scale s hPidB hPidH hQBnd hQ4D hK4D hV4D idx
  rw [hExec] at hview
  simpa [observeTileAt] using hview



end VeriTile.Examples
