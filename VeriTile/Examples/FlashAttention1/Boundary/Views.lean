/-
VeriTile.Examples.FlashAttention1.Boundary.Views

Split-out support for FlashAttention-1 v1 boundary proofs.
-/

import VeriTile.Examples.FlashAttention1.Boundary.Boundary

namespace VeriTile.Examples

open VeriTile.Triton

theorem fa1_forward_correct_4D_boundary_slice
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : InputAt s qReg
        (Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0) Q4D)
    (hK4D : InputAt s kReg
        (Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0) K4D)
    (hV4D : InputAt s vReg
        (Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0) V4D)
    (hOValid : Offset.StridesValid [B, H, S_q, D] [sOB, sOH, sOM, sOD]) :
    ∀ idx : TileIndex [M, D],
      s.pids 0 * M + idx.1.val < S_q →
      observeTileAt
          (exec (fa1ForwardKernelStridedBoundary qReg kReg vReg outReg
              M D Bk numKVBlocks S_q S_k
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal
                (slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
                  (s.pids 0))
                (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
                (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
                scale idx) := by
  intro idx hIdxIn
  -- `hQIn`: in-bounds Q rows read the logical Q tensor.
  have hQIn_inner : ∀ tileIdx : TileIndex [M, D],
      s.pids 0 * M + tileIdx.1.val < S_q →
      s.readMem qReg
        (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
          + tileIdx.1.val * sQS + tileIdx.2.1.val * sQD) =
        slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
          (s.pids 0) tileIdx := by
    intro tileIdx hIn
    obtain ⟨i, d, _⟩ := tileIdx
    have h := hQ4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨s.pids 0 * M + i.val, hIn⟩,
                    d, PUnit.unit)
    show s.readMem qReg
      (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
        + i.val * sQS + d.val * sQD) = _
    rw [show s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + i.val * sQS + d.val * sQD =
          Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + i.val, hIn⟩,
             d, PUnit.unit) by
        simp [Offset.strided, Nat.add_mul]
        ring]
    rw [slice4DQRowsBoundary_of_lt M Q4D _ _ _ i d hIn]
    exact h
  -- `hQOut`: out-of-bounds Q rows are zero by definition.
  have hQOut_inner : ∀ tileIdx : TileIndex [M, D],
      ¬ s.pids 0 * M + tileIdx.1.val < S_q →
      slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
        (s.pids 0) tileIdx = 0 := by
    intro tileIdx hOut
    obtain ⟨i, d, _⟩ := tileIdx
    exact slice4DQRowsBoundary_of_not_lt M Q4D _ _ _ i d hOut
  -- Convert `hK4D` to inner-theorem form: K-side slice is `sliceBH K4D b h`,
  -- which has shape `[S_k, D]` directly (no `slice4DFlat` rewriting needed).
  have hK_inner : InputAt s kReg
      (fun idx : TileIndex [S_k, D] =>
        s.pids 2 * sKB + s.pids 1 * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD)
      (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hK4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit)
    show s.readMem kReg
      (s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD) = _
    rw [show s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD =
          Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit) by
        simp [Offset.strided]]
    exact h
  -- Convert `hV4D` similarly.
  have hV_inner : InputAt s vReg
      (fun idx : TileIndex [S_k, D] =>
        s.pids 2 * sVB + s.pids 1 * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD)
      (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hV4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit)
    show s.readMem vReg
      (s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD) = _
    rw [show s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD =
          Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit) by
        simp [Offset.strided]]
    exact h
  -- Output tile-local injectivity from `Offset.strided_inj`.
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
  exact fa1_forward_correct_strided_boundary hBk hSk hSkLe
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD
    (slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ (s.pids 0))
    (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
    (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
    scale s hQIn_inner hQOut_inner hK_inner hV_inner hInj idx hIdxIn

theorem fa1_forward_correct_4D_boundary
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : InputAt s qReg
        (Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0) Q4D)
    (hK4D : InputAt s kReg
        (Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0) K4D)
    (hV4D : InputAt s vReg
        (Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0) V4D)
    (hOValid : Offset.StridesValid [B, H, S_q, D] [sOB, sOH, sOM, sOD]) :
    ∀ idx : TileIndex [M, D],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      observeTileAt
          (exec (fa1ForwardKernelStridedBoundary qReg kReg vReg outReg
              M D Bk numKVBlocks S_q S_k
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
  intro idx hLt
  rw [fa1_forward_correct_4D_boundary_slice hBk hSk hSkLe
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        Q4D K4D V4D scale s hPidB hPidH hQ4D hK4D hV4D hOValid idx hLt]
  congr 1
  -- Bridge: `attentionReal` over the boundary Q-row slice + `sliceBH` K/V
  -- equals `attentionReal4D` at the global `(b, h, qb*M + i, d)` index.
  -- For in-bounds rows, `slice4DQRowsBoundary` returns exactly the same
  -- value as `Q4D ∘ globalIndex`, so `attentionReal_row_eq` applies.
  obtain ⟨i, d, _⟩ := idx
  rw [attentionReal4D_slice]
  apply attentionReal_row_eq
  intro d'
  -- Goal: `slice4DQRowsBoundary M Q4D b h qb (i, d', ()) = sliceBH Q4D b h (⟨qb*M+i,_⟩, d', ())`
  rw [slice4DQRowsBoundary_of_lt M Q4D _ _ _ i d' hLt]
  rfl

/-- D-tail boundary FA-1 forward correctness in user-facing `attentionReal4D`
form. The theorem observes a block-width `[M, Bd]` output tile but only
asserts logical lanes `d < D`, so ordinary output stride validity over the
logical `[B,H,S_q,D]` tensor suffices for readback. -/
theorem fa1_forward_correct_4D_boundaryD
    {B H S_q S_k D Bd Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : InputAt s qReg
        (Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0) Q4D)
    (hK4D : InputAt s kReg
        (Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0) K4D)
    (hV4D : InputAt s vReg
        (Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0) V4D)
    (hOValid : Offset.StridesValid [B, H, S_q, D] [sOB, sOH, sOM, sOD]) :
    ∀ idx : TileIndex [M, Bd],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (exec (fa1ForwardKernelStridedBoundaryD qReg kReg vReg outReg
              M Bd Bk numKVBlocks S_q S_k D
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal4D Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, hLt⟩,
             ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hLt hDIdx
  have hQIn_inner : ∀ tileIdx : TileIndex [M, D],
      s.pids 0 * M + tileIdx.1.val < S_q →
      s.readMem qReg
        (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
          + tileIdx.1.val * sQS + tileIdx.2.1.val * sQD) =
        slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
          (s.pids 0) tileIdx := by
    intro tileIdx hIn
    obtain ⟨i, d, _⟩ := tileIdx
    have h := hQ4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨s.pids 0 * M + i.val, hIn⟩,
                    d, PUnit.unit)
    show s.readMem qReg
      (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
        + i.val * sQS + d.val * sQD) = _
    rw [show s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + i.val * sQS + d.val * sQD =
          Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + i.val, hIn⟩,
             d, PUnit.unit) by
        simp [Offset.strided, Nat.add_mul]
        ring]
    rw [slice4DQRowsBoundary_of_lt M Q4D _ _ _ i d hIn]
    exact h
  have hQOut_inner : ∀ tileIdx : TileIndex [M, D],
      ¬ s.pids 0 * M + tileIdx.1.val < S_q →
      slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
        (s.pids 0) tileIdx = 0 := by
    intro tileIdx hOut
    obtain ⟨i, d, _⟩ := tileIdx
    exact slice4DQRowsBoundary_of_not_lt M Q4D _ _ _ i d hOut
  have hK_inner : InputAt s kReg
      (fun idx : TileIndex [S_k, D] =>
        s.pids 2 * sKB + s.pids 1 * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD)
      (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hK4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit)
    show s.readMem kReg
      (s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD) = _
    rw [show s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD =
          Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit) by
        simp [Offset.strided]]
    exact h
  have hV_inner : InputAt s vReg
      (fun idx : TileIndex [S_k, D] =>
        s.pids 2 * sVB + s.pids 1 * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD)
      (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hV4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit)
    show s.readMem vReg
      (s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD) = _
    rw [show s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD =
          Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit) by
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
  rw [fa1_forward_correct_strided_boundaryD hBk hSk hSkLe hDLe
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        (slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ (s.pids 0))
        (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
        (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
        scale s hQIn_inner hQOut_inner hK_inner hV_inner hInj idx hLt hDIdx]
  congr 1
  obtain ⟨i, d, u⟩ := idx
  cases u
  rw [attentionReal4D_slice]
  apply attentionReal_row_eq
  intro d'
  rw [slice4DQRowsBoundary_of_lt M Q4D _ _ _ i d' hLt]
  rfl

/-- D-tail causal-boundary FA-1 forward correctness in user-facing
`attentionReal4DCausal` form. The theorem observes a block-width `[M, Bd]`
output tile but only asserts logical lanes `d < D`. -/
theorem fa1_forward_correct_4D_causal_boundaryD
    {B H S_q S_k D Bd Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : InputAt s qReg
        (Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0) Q4D)
    (hK4D : InputAt s kReg
        (Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0) K4D)
    (hV4D : InputAt s vReg
        (Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0) V4D)
    (hOValid : Offset.StridesValid [B, H, S_q, D] [sOB, sOH, sOM, sOD]) :
    ∀ idx : TileIndex [M, Bd],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (exec (fa1ForwardKernelStridedCausalBoundaryD qReg kReg vReg outReg
              M Bd Bk numKVBlocks S_q S_k D
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionReal4DCausal Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, hLt⟩,
             ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hLt hDIdx
  have hQIn_inner : ∀ tileIdx : TileIndex [M, D],
      s.pids 0 * M + tileIdx.1.val < S_q →
      s.readMem qReg
        (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
          + tileIdx.1.val * sQS + tileIdx.2.1.val * sQD) =
        slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
          (s.pids 0) tileIdx := by
    intro tileIdx hIn
    obtain ⟨i, d, _⟩ := tileIdx
    have h := hQ4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨s.pids 0 * M + i.val, hIn⟩,
                    d, PUnit.unit)
    show s.readMem qReg
      (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
        + i.val * sQS + d.val * sQD) = _
    rw [show s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + i.val * sQS + d.val * sQD =
          Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + i.val, hIn⟩,
             d, PUnit.unit) by
        simp [Offset.strided, Nat.add_mul]
        ring]
    rw [slice4DQRowsBoundary_of_lt M Q4D _ _ _ i d hIn]
    exact h
  have hQOut_inner : ∀ tileIdx : TileIndex [M, D],
      ¬ s.pids 0 * M + tileIdx.1.val < S_q →
      slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
        (s.pids 0) tileIdx = 0 := by
    intro tileIdx hOut
    obtain ⟨i, d, _⟩ := tileIdx
    exact slice4DQRowsBoundary_of_not_lt M Q4D _ _ _ i d hOut
  have hK_inner : InputAt s kReg
      (fun idx : TileIndex [S_k, D] =>
        s.pids 2 * sKB + s.pids 1 * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD)
      (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hK4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit)
    show s.readMem kReg
      (s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD) = _
    rw [show s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD =
          Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit) by
        simp [Offset.strided]]
    exact h
  have hV_inner : InputAt s vReg
      (fun idx : TileIndex [S_k, D] =>
        s.pids 2 * sVB + s.pids 1 * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD)
      (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hV4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit)
    show s.readMem vReg
      (s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD) = _
    rw [show s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD =
          Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit) by
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
  rw [fa1_forward_correct_strided_causal_boundaryD hBk hSk hSkLe hDLe
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        (slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ (s.pids 0))
        (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
        (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
        scale s hQIn_inner hQOut_inner hK_inner hV_inner hInj idx hLt hDIdx]
  congr 1
  obtain ⟨i, d, u⟩ := idx
  cases u
  have hLtI : s.pids 0 * M + i.val < S_q := by
    simpa using hLt
  rw [attentionReal4DCausal_slice]
  unfold attentionRealCausalBlock attentionRealCausal
  simp [sliceBH, slice4DQRowsBoundary, hLtI]

/-- Causal-boundary 4D-aware corollary of
`fa1_forward_correct_strided_causal_boundary`. The result is still stated in
slice-local `attentionRealCausalBlock` form; `fa1_forward_correct_4D_causal_boundary`
below rewrites it to the user-facing 4D causal spec. -/
theorem fa1_forward_correct_4D_causal_boundary_slice
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : InputAt s qReg
        (Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0) Q4D)
    (hK4D : InputAt s kReg
        (Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0) K4D)
    (hV4D : InputAt s vReg
        (Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0) V4D)
    (hOValid : Offset.StridesValid [B, H, S_q, D] [sOB, sOH, sOM, sOD]) :
    ∀ idx : TileIndex [M, D],
      s.pids 0 * M + idx.1.val < S_q →
      observeTileAt
          (exec (fa1ForwardKernelStridedCausalBoundary qReg kReg vReg outReg
              M D Bk numKVBlocks S_q S_k
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, D] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some (attentionRealCausalBlock (s.pids 0 * M)
                (slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
                  (s.pids 0))
                (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
                (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
                scale idx) := by
  intro idx hIdxIn
  have hQIn_inner : ∀ tileIdx : TileIndex [M, D],
      s.pids 0 * M + tileIdx.1.val < S_q →
      s.readMem qReg
        (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
          + tileIdx.1.val * sQS + tileIdx.2.1.val * sQD) =
        slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
          (s.pids 0) tileIdx := by
    intro tileIdx hIn
    obtain ⟨i, d, _⟩ := tileIdx
    have h := hQ4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
                    ⟨s.pids 0 * M + i.val, hIn⟩,
                    d, PUnit.unit)
    show s.readMem qReg
      (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
        + i.val * sQS + d.val * sQD) = _
    rw [show s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + i.val * sQS + d.val * sQD =
          Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + i.val, hIn⟩,
             d, PUnit.unit) by
        simp [Offset.strided, Nat.add_mul]
        ring]
    rw [slice4DQRowsBoundary_of_lt M Q4D _ _ _ i d hIn]
    exact h
  have hQOut_inner : ∀ tileIdx : TileIndex [M, D],
      ¬ s.pids 0 * M + tileIdx.1.val < S_q →
      slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩
        (s.pids 0) tileIdx = 0 := by
    intro tileIdx hOut
    obtain ⟨i, d, _⟩ := tileIdx
    exact slice4DQRowsBoundary_of_not_lt M Q4D _ _ _ i d hOut
  have hK_inner : InputAt s kReg
      (fun idx : TileIndex [S_k, D] =>
        s.pids 2 * sKB + s.pids 1 * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD)
      (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hK4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit)
    show s.readMem kReg
      (s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD) = _
    rw [show s.pids 2 * sKB + s.pids 1 * sKH + j.val * sKN + d.val * sKD =
          Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit) by
        simp [Offset.strided]]
    exact h
  have hV_inner : InputAt s vReg
      (fun idx : TileIndex [S_k, D] =>
        s.pids 2 * sVB + s.pids 1 * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD)
      (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩) := by
    intro tileIdx
    obtain ⟨j, d, _⟩ := tileIdx
    have h := hV4D (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit)
    show s.readMem vReg
      (s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD) = _
    rw [show s.pids 2 * sVB + s.pids 1 * sVH + j.val * sVN + d.val * sVD =
          Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩, j, d, PUnit.unit) by
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
  exact fa1_forward_correct_strided_causal_boundary hBk hSk hSkLe
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD
    (slice4DQRowsBoundary M Q4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩ (s.pids 0))
    (sliceBH K4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
    (sliceBH V4D ⟨s.pids 2, hPidB⟩ ⟨s.pids 1, hPidH⟩)
    scale s hQIn_inner hQOut_inner hK_inner hV_inner hInj idx hIdxIn

/-- Causal-boundary FA-1 forward correctness in user-facing
`attentionReal4DCausal` form. -/
theorem fa1_forward_correct_4D_causal_boundary
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : InputAt s qReg
        (Offset.strided [B, H, S_q, D] [sQB, sQH, sQS, sQD] 0) Q4D)
    (hK4D : InputAt s kReg
        (Offset.strided [B, H, S_k, D] [sKB, sKH, sKN, sKD] 0) K4D)
    (hV4D : InputAt s vReg
        (Offset.strided [B, H, S_k, D] [sVB, sVH, sVN, sVD] 0) V4D)
    (hOValid : Offset.StridesValid [B, H, S_q, D] [sOB, sOH, sOM, sOD]) :
    ∀ idx : TileIndex [M, D],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      observeTileAt
          (exec (fa1ForwardKernelStridedCausalBoundary qReg kReg vReg outReg
              M D Bk numKVBlocks S_q S_k
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
  intro idx hLt
  rw [fa1_forward_correct_4D_causal_boundary_slice hBk hSk hSkLe
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD
        Q4D K4D V4D scale s hPidB hPidH hQ4D hK4D hV4D hOValid idx hLt]
  congr 1
  obtain ⟨i, d, _⟩ := idx
  have hLtI : s.pids 0 * M + i.val < S_q := by
    simpa using hLt
  rw [attentionReal4DCausal_slice]
  unfold attentionRealCausalBlock attentionRealCausal
  simp [sliceBH, slice4DQRowsBoundary, hLtI]

theorem fa1_forward_correct_4D_boundary_layout
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : TensorView.loaded s (layout.qView qReg) Q4D)
    (hK4D : TensorView.loaded s (layout.kView kReg) K4D)
    (hV4D : TensorView.loaded s (layout.vView vReg) V4D) :
    ∀ idx : TileIndex [M, D],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      observeTileAt
          (exec (layout.boundaryKernel qReg kReg vReg outReg M Bk numKVBlocks scale) s)
          outReg (layout.outBlockOffset s M) idx
        = some (attentionReal4D Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
             idx.2.1, PUnit.unit)) := by
  intro idx hLt
  simpa [FA1Layout4D.boundaryKernel, FA1Layout4D.qOffset, FA1Layout4D.kOffset,
         FA1Layout4D.vOffset, FA1Layout4D.outBlockOffset,
         FA1Layout4D.qView, FA1Layout4D.kView, FA1Layout4D.vView,
         TensorView.loaded, TensorView.offset,
         FA1Layout4D.qStrides, FA1Layout4D.kStrides,
         FA1Layout4D.vStrides, FA1Layout4D.oStrides]
    using fa1_forward_correct_4D_boundary hBk hSk hSkLe
      qReg kReg vReg outReg
      layout.qB layout.qH layout.qS layout.qD
      layout.kB layout.kH layout.kS layout.kD
      layout.vB layout.vH layout.vS layout.vD
      layout.oB layout.oH layout.oS layout.oD
      Q4D K4D V4D scale s hPidB hPidH
      hQ4D hK4D hV4D layout.hOValid idx hLt

/-- Causal-boundary FA-1 forward correctness over a bundled 4D layout. -/
theorem fa1_forward_correct_4D_causal_boundary_layout
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : TensorView.loaded s (layout.qView qReg) Q4D)
    (hK4D : TensorView.loaded s (layout.kView kReg) K4D)
    (hV4D : TensorView.loaded s (layout.vView vReg) V4D) :
    ∀ idx : TileIndex [M, D],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      observeTileAt
          (exec (layout.causalBoundaryKernel qReg kReg vReg outReg M Bk numKVBlocks scale) s)
          outReg (layout.outBlockOffset s M) idx
        = some (attentionReal4DCausal Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
             idx.2.1, PUnit.unit)) := by
  intro idx hLt
  simpa [FA1Layout4D.causalBoundaryKernel, FA1Layout4D.qOffset, FA1Layout4D.kOffset,
         FA1Layout4D.vOffset, FA1Layout4D.outBlockOffset,
         FA1Layout4D.qView, FA1Layout4D.kView, FA1Layout4D.vView,
         TensorView.loaded, TensorView.offset,
         FA1Layout4D.qStrides, FA1Layout4D.kStrides,
         FA1Layout4D.vStrides, FA1Layout4D.oStrides]
    using fa1_forward_correct_4D_causal_boundary hBk hSk hSkLe
      qReg kReg vReg outReg
      layout.qB layout.qH layout.qS layout.qD
      layout.kB layout.kH layout.kS layout.kD
      layout.vB layout.vH layout.vS layout.vD
      layout.oB layout.oH layout.oS layout.oD
      Q4D K4D V4D scale s hPidB hPidH
      hQ4D hK4D hV4D layout.hOValid idx hLt

/-- D-tail boundary FA-1 forward correctness over a bundled 4D layout.
The output tile has padded width `Bd`, but the conclusion is stated only for
logical lanes `d < D`. -/
theorem fa1_forward_correct_4D_boundaryD_layout
    {B H S_q S_k D Bd Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
    (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : TensorView.loaded s (layout.qView qReg) Q4D)
    (hK4D : TensorView.loaded s (layout.kView kReg) K4D)
    (hV4D : TensorView.loaded s (layout.vView vReg) V4D) :
    ∀ idx : TileIndex [M, Bd],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (exec (layout.boundaryKernelD qReg kReg vReg outReg M Bd Bk numKVBlocks scale) s)
          outReg (layout.outBlockOffsetD s M Bd) idx
        = some (attentionReal4D Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, hLt⟩,
             ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hLt hDIdx
  simpa [FA1Layout4D.boundaryKernelD, FA1Layout4D.qOffset, FA1Layout4D.kOffset,
         FA1Layout4D.vOffset, FA1Layout4D.outBlockOffsetD,
         FA1Layout4D.qView, FA1Layout4D.kView, FA1Layout4D.vView,
         TensorView.loaded, TensorView.offset,
         FA1Layout4D.qStrides, FA1Layout4D.kStrides,
         FA1Layout4D.vStrides, FA1Layout4D.oStrides]
    using fa1_forward_correct_4D_boundaryD hBk hSk hSkLe hDLe
      qReg kReg vReg outReg
      layout.qB layout.qH layout.qS layout.qD
      layout.kB layout.kH layout.kS layout.kD
      layout.vB layout.vH layout.vS layout.vD
      layout.oB layout.oH layout.oS layout.oD
      Q4D K4D V4D scale s hPidB hPidH
      hQ4D hK4D hV4D layout.hOValid idx hLt hDIdx

/-- D-tail causal-boundary FA-1 forward correctness over a bundled 4D
layout. -/
theorem fa1_forward_correct_4D_causal_boundaryD_layout
    {B H S_q S_k D Bd Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
    (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : TensorView.loaded s (layout.qView qReg) Q4D)
    (hK4D : TensorView.loaded s (layout.kView kReg) K4D)
    (hV4D : TensorView.loaded s (layout.vView vReg) V4D) :
    ∀ idx : TileIndex [M, Bd],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (exec (layout.causalBoundaryKernelD qReg kReg vReg outReg M Bd Bk numKVBlocks scale) s)
          outReg (layout.outBlockOffsetD s M Bd) idx
        = some (attentionReal4DCausal Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, hLt⟩,
             ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hLt hDIdx
  simpa [FA1Layout4D.causalBoundaryKernelD, FA1Layout4D.qOffset, FA1Layout4D.kOffset,
         FA1Layout4D.vOffset, FA1Layout4D.outBlockOffsetD,
         FA1Layout4D.qView, FA1Layout4D.kView, FA1Layout4D.vView,
         TensorView.loaded, TensorView.offset,
         FA1Layout4D.qStrides, FA1Layout4D.kStrides,
         FA1Layout4D.vStrides, FA1Layout4D.oStrides]
    using fa1_forward_correct_4D_causal_boundaryD hBk hSk hSkLe hDLe
      qReg kReg vReg outReg
      layout.qB layout.qH layout.qS layout.qD
      layout.kB layout.kH layout.kS layout.kD
      layout.vB layout.vH layout.vS layout.vD
      layout.oB layout.oH layout.oS layout.oD
      Q4D K4D V4D scale s hPidB hPidH
      hQ4D hK4D hV4D layout.hOValid idx hLt hDIdx

theorem fa1_forward_correct_4D_boundary_views
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (views : FA1Views4D B H S_q S_k D)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : TensorView.loaded s views.qView Q4D)
    (hK4D : TensorView.loaded s views.kView K4D)
    (hV4D : TensorView.loaded s views.vView V4D) :
    ∀ idx : TileIndex [M, D],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      observeTileAt
          (exec (views.boundaryKernel M Bk numKVBlocks scale) s)
          views.outReg (views.outBlockOffset s M) idx
        = some (attentionReal4D Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
             idx.2.1, PUnit.unit)) := by
  intro idx hLt
  simpa [FA1Views4D.boundaryKernel, FA1Views4D.outBlockOffset,
         FA1Views4D.qView, FA1Views4D.kView, FA1Views4D.vView]
    using fa1_forward_correct_4D_boundary_layout hBk hSk hSkLe views.layout
      views.qReg views.kReg views.vReg views.outReg
      Q4D K4D V4D scale s hPidB hPidH
      hQ4D hK4D hV4D idx hLt

/-- Causal-boundary FA-1 forward correctness over bundled tensor views. -/
theorem fa1_forward_correct_4D_causal_boundary_views
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (views : FA1Views4D B H S_q S_k D)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : TensorView.loaded s views.qView Q4D)
    (hK4D : TensorView.loaded s views.kView K4D)
    (hV4D : TensorView.loaded s views.vView V4D) :
    ∀ idx : TileIndex [M, D],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      observeTileAt
          (exec (views.causalBoundaryKernel M Bk numKVBlocks scale) s)
          views.outReg (views.outBlockOffset s M) idx
        = some (attentionReal4DCausal Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, by have := idx.1.isLt; omega⟩,
             idx.2.1, PUnit.unit)) := by
  intro idx hLt
  simpa [FA1Views4D.causalBoundaryKernel, FA1Views4D.outBlockOffset,
         FA1Views4D.qView, FA1Views4D.kView, FA1Views4D.vView]
    using fa1_forward_correct_4D_causal_boundary_layout hBk hSk hSkLe views.layout
      views.qReg views.kReg views.vReg views.outReg
      Q4D K4D V4D scale s hPidB hPidH
      hQ4D hK4D hV4D idx hLt

/-- D-tail boundary FA-1 forward correctness over bundled tensor views. -/
theorem fa1_forward_correct_4D_boundaryD_views
    {B H S_q S_k D Bd Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
    (views : FA1Views4D B H S_q S_k D)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : TensorView.loaded s views.qView Q4D)
    (hK4D : TensorView.loaded s views.kView K4D)
    (hV4D : TensorView.loaded s views.vView V4D) :
    ∀ idx : TileIndex [M, Bd],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (exec (views.boundaryKernelD M Bd Bk numKVBlocks scale) s)
          views.outReg (views.outBlockOffsetD s M Bd) idx
        = some (attentionReal4D Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, hLt⟩,
             ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hLt hDIdx
  simpa [FA1Views4D.boundaryKernelD, FA1Views4D.outBlockOffsetD,
         FA1Views4D.qView, FA1Views4D.kView, FA1Views4D.vView]
    using fa1_forward_correct_4D_boundaryD_layout hBk hSk hSkLe hDLe views.layout
      views.qReg views.kReg views.vReg views.outReg
      Q4D K4D V4D scale s hPidB hPidH
      hQ4D hK4D hV4D idx hLt hDIdx

/-- D-tail causal-boundary FA-1 forward correctness over bundled tensor views. -/
theorem fa1_forward_correct_4D_causal_boundaryD_views
    {B H S_q S_k D Bd Bk numKVBlocks M : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
    (views : FA1Views4D B H S_q S_k D)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (hPidB : s.pids 2 < B) (hPidH : s.pids 1 < H)
    (hQ4D : TensorView.loaded s views.qView Q4D)
    (hK4D : TensorView.loaded s views.kView K4D)
    (hV4D : TensorView.loaded s views.vView V4D) :
    ∀ idx : TileIndex [M, Bd],
      ∀ hLt : s.pids 0 * M + idx.1.val < S_q,
      ∀ hDIdx : idx.2.1.val < D,
      observeTileAt
          (exec (views.causalBoundaryKernelD M Bd Bk numKVBlocks scale) s)
          views.outReg (views.outBlockOffsetD s M Bd) idx
        = some (attentionReal4DCausal Q4D K4D V4D scale
            (⟨s.pids 2, hPidB⟩, ⟨s.pids 1, hPidH⟩,
             ⟨s.pids 0 * M + idx.1.val, hLt⟩,
             ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  intro idx hLt hDIdx
  simpa [FA1Views4D.causalBoundaryKernelD, FA1Views4D.outBlockOffsetD,
         FA1Views4D.qView, FA1Views4D.kView, FA1Views4D.vView]
    using fa1_forward_correct_4D_causal_boundaryD_layout hBk hSk hSkLe hDLe views.layout
      views.qReg views.kReg views.vReg views.outReg
      Q4D K4D V4D scale s hPidB hPidH
      hQ4D hK4D hV4D idx hLt hDIdx



end VeriTile.Examples
