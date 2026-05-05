/-
VeriTile.Examples.FlashAttention1.V1Boundary.Helpers

Split-out support for FlashAttention-1 v1 boundary proofs.
-/

import VeriTile.Examples.FlashAttention1.V1Boundary.Bodies

namespace VeriTile.Examples

open VeriTile.Triton

/-- Reading the `n`-th KV block through the row-major address expression
used by the loop gives the corresponding `blockIndex` cell of a full
`[Bk * numKVBlocks, D]` input. -/
theorem fa1_block_read
    {D Bk numKVBlocks : Nat}
    (region : RegionName) (s : BlockState)
    (X : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (hX : InputAt s region
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) X)
    (n : Nat) (hn : n < numKVBlocks)
    (j : Fin Bk) (d : Fin D) :
    s.readMem region ((n * Bk + j.val) * D + d.val) =
      X (FA1Math.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) j,
        d, PUnit.unit) := by
  rw [BlockState.readMem]
  have haddr :
      (n * Bk + j.val) * D + d.val =
        Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D
          (FA1Math.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) j,
            d, PUnit.unit) := by
    simp [Offset.rowMajor2D, Offset.strided, FA1Math.blockIndex]
  rw [haddr]
  exact hX _

/-- Tile-level version of `fa1_block_read`: the loop-local block load is the
`Tile.ofReal` view of the corresponding full K/V matrix block. -/
theorem fa1_block_load_tile_eq
    {D Bk numKVBlocks : Nat}
    (region : RegionName) (s : BlockState)
    (X : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (hX : InputAt s region
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) X)
    (n : Nat) (hn : n < numKVBlocks) :
    (⟨fun idx : TileIndex [Bk, D] =>
        some (s.readMem region ((n * Bk + idx.1.val) * D + idx.2.1.val))⟩
      : Tile .real [Bk, D])
      =
      Tile.ofReal (fun idx : TileIndex [Bk, D] =>
        X (FA1Math.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) idx.1,
          idx.2.1, PUnit.unit)) := by
  apply load_tile_eq_of_InputAt_map
    (s := s) (region := region)
    (addr := fun idx : TileIndex [Bk, D] =>
      (n * Bk + idx.1.val) * D + idx.2.1.val)
    (offsetFn := Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D)
    (embed := fun idx : TileIndex [Bk, D] =>
      (FA1Math.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) idx.1,
        idx.2.1, PUnit.unit))
    (xs := X)
  · intro idx
    simp [Offset.rowMajor2D, Offset.strided, FA1Math.blockIndex]
  · exact hX

/-- Strided variant of `fa1_block_read`: reading the `n`-th KV block at
the strided address `base + (n*Bk + j) * sN + d * sD` recovers the
`blockIndex` cell of the full input. The strided InputAt premise
matches the K / V branches of `P_fa1_strided` (with
`base = batch * sKB + headIdx * sKH`, `sN = sKN`, `sD = sKD`, etc.). -/
theorem fa1_block_read_strided
    {D Bk numKVBlocks : Nat}
    (region : RegionName) (s : BlockState)
    (base sN sD : Nat)
    (X : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (hX : InputAt s region
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          base + idx.1.val * sN + idx.2.1.val * sD) X)
    (n : Nat) (hn : n < numKVBlocks)
    (j : Fin Bk) (d : Fin D) :
    s.readMem region (base + (n * Bk + j.val) * sN + d.val * sD) =
      X (FA1Math.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) j,
        d, PUnit.unit) := by
  rw [BlockState.readMem]
  have haddr :
      base + (n * Bk + j.val) * sN + d.val * sD =
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
            base + idx.1.val * sN + idx.2.1.val * sD)
          (FA1Math.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) j,
            d, PUnit.unit) := by
    simp [FA1Math.blockIndex]
  rw [haddr]
  exact hX _

/-- Strided tile-level version of `fa1_block_load_tile_eq`. -/
theorem fa1_block_load_tile_eq_strided
    {D Bk numKVBlocks : Nat}
    (region : RegionName) (s : BlockState)
    (base sN sD : Nat)
    (X : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (hX : InputAt s region
        (fun idx : TileIndex [Bk * numKVBlocks, D] =>
          base + idx.1.val * sN + idx.2.1.val * sD) X)
    (n : Nat) (hn : n < numKVBlocks) :
    (⟨fun idx : TileIndex [Bk, D] =>
        some (s.readMem region
          (base + (n * Bk + idx.1.val) * sN + idx.2.1.val * sD))⟩
      : Tile .real [Bk, D])
      =
      Tile.ofReal (fun idx : TileIndex [Bk, D] =>
        X (FA1Math.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) idx.1,
          idx.2.1, PUnit.unit)) := by
  apply load_tile_eq_of_InputAt_map
    (s := s) (region := region)
    (addr := fun idx : TileIndex [Bk, D] =>
      base + (n * Bk + idx.1.val) * sN + idx.2.1.val * sD)
    (offsetFn := fun idx : TileIndex [Bk * numKVBlocks, D] =>
      base + idx.1.val * sN + idx.2.1.val * sD)
    (embed := fun idx : TileIndex [Bk, D] =>
      (FA1Math.blockIndex Bk numKVBlocks n (Nat.succ_le_iff.mpr hn) idx.1,
        idx.2.1, PUnit.unit))
    (xs := X)
  · intro idx
    simp [FA1Math.blockIndex]
  · exact hX

/-- Boundary-masked strided KV block load. Valid local lanes read the logical
`[S_k, D]` tensor through `blockIndex?`; invalid padded lanes are exactly the
explicit `other = 0` value from the Triton load. -/
theorem fa1_block_load_tile_eq_strided_boundary
    {D Bk S_k : Nat}
    (region : RegionName) (s : BlockState)
    (base sN sD : Nat)
    (X : TileIndex [S_k, D] → ℝ)
    (hX : InputAt s region
        (fun idx : TileIndex [S_k, D] =>
          base + idx.1.val * sN + idx.2.1.val * sD) X)
    (n : Nat) :
    (⟨fun idx : TileIndex [Bk, D] =>
        if _h : n * Bk + idx.1.val < S_k then
          some (s.readMem region
            (base + (n * Bk + idx.1.val) * sN + idx.2.1.val * sD))
        else
          some 0⟩
      : Tile .real [Bk, D])
      =
      Tile.ofReal (fun idx : TileIndex [Bk, D] =>
        match FA1MathBoundary.blockIndex? S_k Bk n idx.1 with
        | some j => X (j, idx.2.1, PUnit.unit)
        | none => 0) := by
  ext idx
  rw [Tile.ofReal_data]
  by_cases h : n * Bk + idx.1.val < S_k
  · simp [h, FA1MathBoundary.blockIndex?_of_lt]
    rw [BlockState.readMem]
    have haddr :
        base + (n * Bk + idx.1.val) * sN + idx.2.1.val * sD =
          (fun idx : TileIndex [S_k, D] =>
              base + idx.1.val * sN + idx.2.1.val * sD)
            (⟨n * Bk + idx.1.val, h⟩, idx.2.1, PUnit.unit) := by
      rfl
    rw [haddr]
    exact congrArg some (hX _)
  · simp [h, FA1MathBoundary.blockIndex?_of_not_lt]

/-- D-tail boundary-masked strided KV block load. A lane reads memory only
when both the sequence lane is valid and the hidden lane is logical
(`d < D`); otherwise it is the explicit `other = 0`. The result is the
boundary block view of `padHeadD X`. -/
theorem fa1_block_load_tile_eq_strided_boundaryD
    {D Bd Bk S_k : Nat}
    (region : RegionName) (s : BlockState)
    (base sN sD : Nat)
    (X : TileIndex [S_k, D] → ℝ)
    (hX : InputAt s region
        (fun idx : TileIndex [S_k, D] =>
          base + idx.1.val * sN + idx.2.1.val * sD) X)
    (n : Nat) :
    (⟨fun idx : TileIndex [Bk, Bd] =>
        if _h : (n * Bk + idx.1.val < S_k) ∧ (idx.2.1.val < D) then
          some (s.readMem region
            (base + (n * Bk + idx.1.val) * sN + idx.2.1.val * sD))
        else
          some 0⟩
      : Tile .real [Bk, Bd])
      =
      Tile.ofReal (fun idx : TileIndex [Bk, Bd] =>
        match FA1MathBoundary.blockIndex? S_k Bk n idx.1 with
        | some j => padHeadD (Bd := Bd) X (j, idx.2.1, PUnit.unit)
        | none => 0) := by
  ext idx
  rw [Tile.ofReal_data]
  by_cases hD : idx.2.1.val < D
  · by_cases hRow : n * Bk + idx.1.val < S_k
    · have hBoth : (n * Bk + idx.1.val < S_k) ∧ (idx.2.1.val < D) :=
        ⟨hRow, hD⟩
      simp [hBoth, FA1MathBoundary.blockIndex?_of_lt, padHeadD]
      rw [BlockState.readMem]
      have haddr :
          base + (n * Bk + idx.1.val) * sN + idx.2.1.val * sD =
            (fun idx : TileIndex [S_k, D] =>
                base + idx.1.val * sN + idx.2.1.val * sD)
              (⟨n * Bk + idx.1.val, hRow⟩,
                ⟨idx.2.1.val, hD⟩, PUnit.unit) := by
        rfl
      rw [haddr]
      exact congrArg some (hX _)
    · have hNotBoth : ¬ ((n * Bk + idx.1.val < S_k) ∧ (idx.2.1.val < D)) := by
        intro h; exact hRow h.1
      simp [hRow, FA1MathBoundary.blockIndex?_of_not_lt]
  · have hNotBoth : ¬ ((n * Bk + idx.1.val < S_k) ∧ (idx.2.1.val < D)) := by
      intro h; exact hD h.2
    by_cases hRow : n * Bk + idx.1.val < S_k
    · simp [hRow, FA1MathBoundary.blockIndex?_of_lt, padHeadD, hD]
    · simp [hRow, FA1MathBoundary.blockIndex?_of_not_lt]

/-- Score lane helper for boundary/D-tail loop steps. Once the K tile has
been identified as the boundary block view of `K`, the kernel-side
`scores_raw` followed by the sequence mask is exactly
`FA1MathBoundary.maskedScore`. -/
theorem fa1_boundary_score_lane_eq
    {M D Bk S_k : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat)
    (kLoaded : Tile .real [Bk, D])
    (hK_loaded_eq : kLoaded =
      Tile.ofReal (fun idx : TileIndex [Bk, D] =>
        match FA1MathBoundary.blockIndex? S_k Bk k idx.1 with
        | some j => K (j, idx.2.1, PUnit.unit)
        | none => 0))
    (i : Fin M) (j : Fin Bk) :
    (if k * Bk + j.val < S_k then
      Option.map (fun a : ℝ => a * scale)
        (@Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
          (fun d : Fin D => Option.map (fun b => Q (i, d, PUnit.unit) * b)
            (kLoaded.data (j, d, PUnit.unit))))
    else
      (none : WithBot ℝ))
      =
    FA1MathBoundary.maskedScore Bk k Q K scale i j := by
  by_cases h : k * Bk + j.val < S_k
  · rw [if_pos h]
    have hkLoaded : ∀ d : Fin D,
        kLoaded.data (j, d, PUnit.unit)
          = some (K (⟨k * Bk + j.val, h⟩, d, PUnit.unit)) := by
      intro d
      have := congrArg (fun t : Tile .real [Bk, D] => t.data (j, d, PUnit.unit))
        hK_loaded_eq
      simp [Tile.ofReal, FA1MathBoundary.blockIndex?_of_lt _ _ _ _ h] at this
      exact this
    rw [FA1MathBoundary.maskedScore_of_lt Bk k Q K scale i j h]
    have h_sum :
        (∑ x : Fin D, Option.map (fun b : ℝ => Q (i, x, PUnit.unit) * b)
          (kLoaded.data (j, x, PUnit.unit)) : WithBot ℝ)
        = some (∑ x : Fin D,
            Q (i, x, PUnit.unit) * K (⟨k * Bk + j.val, h⟩, x, PUnit.unit)) := by
      rw [← WithBot.sum_someTerm_eq_some]
      apply Finset.sum_congr rfl
      intro x _
      rw [hkLoaded x]
      rfl
    rw [h_sum]
    show (some _ : WithBot ℝ) = some _
    unfold FA1Math.scaledScore
    congr 1
    ring
  · rw [if_neg h]
    rw [FA1MathBoundary.maskedScore_of_not_lt Bk k Q K scale i j h]
    rfl

/-- Causal score lane helper for boundary/D-tail loop steps. It is the
causal analogue of `fa1_boundary_score_lane_eq`: score computation,
causal masking, then sequence masking equals
`FA1MathCausalBoundary.maskedScore`. -/
theorem fa1_causal_boundary_score_lane_eq
    {M D Bk S_k : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat)
    (kLoaded : Tile .real [Bk, D])
    (hK_loaded_eq : kLoaded =
      Tile.ofReal (fun idx : TileIndex [Bk, D] =>
        match FA1MathBoundary.blockIndex? S_k Bk k idx.1 with
        | some j => K (j, idx.2.1, PUnit.unit)
        | none => 0))
    (i : Fin M) (j : Fin Bk) :
    (if k * Bk + j.val < S_k then
      if k * Bk + j.val ≤ qStart + i.val then
        Option.map (fun a : ℝ => a * scale)
          (@Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
            (fun d : Fin D => Option.map (fun b => Q (i, d, PUnit.unit) * b)
              (kLoaded.data (j, d, PUnit.unit))))
      else
        (none : WithBot ℝ)
    else
      (none : WithBot ℝ))
      =
    FA1MathCausalBoundary.maskedScore Bk k qStart Q K scale i j := by
  by_cases hLt : k * Bk + j.val < S_k
  · rw [if_pos hLt]
    by_cases hLe : k * Bk + j.val ≤ qStart + i.val
    · rw [if_pos hLe]
      have hkLoaded : ∀ d : Fin D,
          kLoaded.data (j, d, PUnit.unit)
            = some (K (⟨k * Bk + j.val, hLt⟩, d, PUnit.unit)) := by
        intro d
        have := congrArg (fun t : Tile .real [Bk, D] => t.data (j, d, PUnit.unit))
          hK_loaded_eq
        simp [Tile.ofReal, FA1MathBoundary.blockIndex?_of_lt _ _ _ _ hLt] at this
        exact this
      rw [FA1MathCausalBoundary.maskedScore_of_lt_of_le Bk k qStart Q K scale i j hLt hLe]
      have h_sum :
          (∑ x : Fin D, Option.map (fun b : ℝ => Q (i, x, PUnit.unit) * b)
            (kLoaded.data (j, x, PUnit.unit)) : WithBot ℝ)
          = some (∑ x : Fin D,
              Q (i, x, PUnit.unit) * K (⟨k * Bk + j.val, hLt⟩, x, PUnit.unit)) := by
        rw [← WithBot.sum_someTerm_eq_some]
        apply Finset.sum_congr rfl
        intro x _
        rw [hkLoaded x]
        rfl
      rw [h_sum]
      show (some _ : WithBot ℝ) = some _
      unfold FA1Math.scaledScore
      congr 1
      ring
    · rw [if_neg hLe]
      rw [FA1MathCausalBoundary.maskedScore_of_lt_of_not_le Bk k qStart Q K scale i j hLt hLe]
      rfl
  · rw [if_neg hLt]
    rw [FA1MathCausalBoundary.maskedScore_of_not_lt Bk k qStart Q K scale i j hLt]
    rfl


end VeriTile.Examples
