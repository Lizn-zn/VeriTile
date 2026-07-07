/-
VeriTile.Examples.FlashAttention1.Common.Boundary

FA-1 boundary-masked non-causal streaming Real recurrence.
-/

import VeriTile.Semantics.StreamingAccumulator

namespace VeriTile.Examples

open VeriTile
/-! ## Boundary-masked streaming math model (FA-1 v1 scaffold)

The full-tile `FA1Math` recurrence indexes K/V by
`Fin (Bk * numKVBlocks)`. Boundary-masked kernels instead iterate over a
padded block domain while the mathematical input has logical length `S_k`.
Invalid local lanes (`k * Bk + jLocal >= S_k`) enter as `⊥`, exactly like
the kernel's score-side `tl.where(score_mask, scores_raw, -inf)`.
-/

namespace FA1MathBoundary

/-- `WithBot.realExp` is never bottom, so it is equal to its `unbotD`
payload rewrapped as `some`. -/
theorem realExp_eq_some_unbotD (x : WithBot ℝ) :
    WithBot.realExp x = some ((WithBot.realExp x).unbotD 0) := by
  cases x <;> rfl

/-- Logical KV index for a padded loop lane, if it is in bounds. -/
def blockIndex? (S_k Bk k : Nat) (jLocal : Fin Bk) : Option (Fin S_k) :=
  if h : k * Bk + jLocal.val < S_k then
    some ⟨k * Bk + jLocal.val, h⟩
  else
    none

@[simp] theorem blockIndex?_of_lt
    (S_k Bk k : Nat) (jLocal : Fin Bk)
    (h : k * Bk + jLocal.val < S_k) :
    blockIndex? S_k Bk k jLocal = some ⟨k * Bk + jLocal.val, h⟩ := by
  simp [blockIndex?, h]

@[simp] theorem blockIndex?_of_not_lt
    (S_k Bk k : Nat) (jLocal : Fin Bk)
    (h : ¬ k * Bk + jLocal.val < S_k) :
    blockIndex? S_k Bk k jLocal = none := by
  simp [blockIndex?, h]

/-- Boundary-masked score for a padded loop lane. Out-of-range KV lanes are
`⊥`, so exponentiating them contributes zero mass. -/
noncomputable def maskedScore {M S_k D : Nat}
    (Bk k : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (i : Fin M) (jLocal : Fin Bk) : WithBot ℝ :=
  match blockIndex? S_k Bk k jLocal with
  | some j => (StreamingAccumulator.scaledScore Q K scale i j : ℝ)
  | none   => ⊥

@[simp] theorem maskedScore_of_lt {M S_k D : Nat}
    (Bk k : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (i : Fin M) (jLocal : Fin Bk)
    (h : k * Bk + jLocal.val < S_k) :
    maskedScore Bk k Q K scale i jLocal =
      ((StreamingAccumulator.scaledScore Q K scale i
        ⟨k * Bk + jLocal.val, h⟩ : ℝ) : WithBot ℝ) := by
  simp [maskedScore, h]

@[simp] theorem maskedScore_of_not_lt {M S_k D : Nat}
    (Bk k : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (i : Fin M) (jLocal : Fin Bk)
    (h : ¬ k * Bk + jLocal.val < S_k) :
    maskedScore Bk k Q K scale i jLocal = (⊥ : WithBot ℝ) := by
  simp [maskedScore, h]

/-- Running per-row max over the first `k` padded KV blocks, ignoring
out-of-range lanes by treating them as `⊥`. -/
noncomputable def mPartial {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ) :
    Nat → Fin M → WithBot ℝ
  | 0, _ => ⊥
  | k + 1, i =>
      if _h : k + 1 ≤ numKVBlocks then
        max (mPartial Bk Q numKVBlocks K scale k i)
          ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
            maskedScore Bk k Q K scale i jLocal)
      else
        mPartial Bk Q numKVBlocks K scale k i

/-- Boundary-aware α multiplier `exp(m_k - m_{k+1})`. -/
noncomputable def alphaPartial {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) : ℝ :=
  (WithBot.realExp
    (WithBot.realSub
      (mPartial Bk Q numKVBlocks K scale k i)
      (mPartial Bk Q numKVBlocks K scale (k + 1) i))).unbotD 0

/-- Boundary-aware running normalizer. -/
noncomputable def lPartial {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ) :
    Nat → Fin M → ℝ
  | 0, _ => 0
  | k + 1, i =>
      if _h : k + 1 ≤ numKVBlocks then
        let mNew := mPartial Bk Q numKVBlocks K scale (k + 1) i
        alphaPartial Bk Q numKVBlocks K scale k i *
          lPartial Bk Q numKVBlocks K scale k i +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            (WithBot.realExp
              (WithBot.realSub
                (maskedScore Bk k Q K scale i jLocal) mNew)).unbotD 0)
      else
        lPartial Bk Q numKVBlocks K scale k i

/-- Boundary-aware running unnormalized output accumulator. -/
noncomputable def oPartial {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ) :
    Nat → TileIndex [M, D] → ℝ
  | 0, _ => 0
  | k + 1, idx =>
      if _h : k + 1 ≤ numKVBlocks then
        let i := idx.1
        let d := idx.2.1
        let mNew := mPartial Bk Q numKVBlocks K scale (k + 1) i
        alphaPartial Bk Q numKVBlocks K scale k i *
          oPartial Bk Q numKVBlocks K V scale k idx +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            match blockIndex? S_k Bk k jLocal with
            | some j =>
                (WithBot.realExp
                  (WithBot.realSub
                    (maskedScore Bk k Q K scale i jLocal) mNew)).unbotD 0 *
                  V (j, d, PUnit.unit)
            | none => 0)
      else
        oPartial Bk Q numKVBlocks K V scale k idx

theorem mPartial_succ_of_lt {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    mPartial Bk Q numKVBlocks K scale (k + 1) i =
      max (mPartial Bk Q numKVBlocks K scale k i)
        ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
          maskedScore Bk k Q K scale i jLocal) := by
  change (if h : k + 1 ≤ numKVBlocks then
      max (mPartial Bk Q numKVBlocks K scale k i)
        ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
          maskedScore Bk k Q K scale i jLocal)
    else mPartial Bk Q numKVBlocks K scale k i) = _
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

theorem lPartial_succ_of_lt {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    lPartial Bk Q numKVBlocks K scale (k + 1) i =
      let mNew := mPartial Bk Q numKVBlocks K scale (k + 1) i
      alphaPartial Bk Q numKVBlocks K scale k i *
        lPartial Bk Q numKVBlocks K scale k i +
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          (WithBot.realExp
            (WithBot.realSub
              (maskedScore Bk k Q K scale i jLocal) mNew)).unbotD 0) := by
  change (if h : k + 1 ≤ numKVBlocks then
      (let mNew := mPartial Bk Q numKVBlocks K scale (k + 1) i
       alphaPartial Bk Q numKVBlocks K scale k i *
          lPartial Bk Q numKVBlocks K scale k i +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            (WithBot.realExp
              (WithBot.realSub
                (maskedScore Bk k Q K scale i jLocal) mNew)).unbotD 0))
    else lPartial Bk Q numKVBlocks K scale k i) = _
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

theorem oPartial_succ_of_lt {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (idx : TileIndex [M, D]) :
    oPartial Bk Q numKVBlocks K V scale (k + 1) idx =
      let i := idx.1
      let d := idx.2.1
      let mNew := mPartial Bk Q numKVBlocks K scale (k + 1) i
      alphaPartial Bk Q numKVBlocks K scale k i *
        oPartial Bk Q numKVBlocks K V scale k idx +
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          match blockIndex? S_k Bk k jLocal with
          | some j =>
              (WithBot.realExp
                (WithBot.realSub
                  (maskedScore Bk k Q K scale i jLocal) mNew)).unbotD 0 *
                V (j, d, PUnit.unit)
          | none => 0) := by
  change (if h : k + 1 ≤ numKVBlocks then
      (let i := idx.1
       let d := idx.2.1
       let mNew := mPartial Bk Q numKVBlocks K scale (k + 1) i
       alphaPartial Bk Q numKVBlocks K scale k i *
          oPartial Bk Q numKVBlocks K V scale k idx +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            match blockIndex? S_k Bk k jLocal with
            | some j =>
                (WithBot.realExp
                  (WithBot.realSub
                    (maskedScore Bk k Q K scale i jLocal) mNew)).unbotD 0 *
                  V (j, d, PUnit.unit)
            | none => 0))
    else oPartial Bk Q numKVBlocks K V scale k idx) = _
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

/-- Boundary loop-local `m_new = max(m_i, m_block)` realizes the masked
streaming `mPartial` recurrence. -/
theorem block_mNew_tile_eq {M S_k D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < N) :
    Tile.bop max (Broadcast.consSame Broadcast.nil)
      (⟨fun idx : TileIndex [M] => mPartial Bk Q N K scale k idx.1⟩ : Tile .real [M])
      (⟨fun idx : TileIndex [M] =>
        (Finset.univ : Finset (Fin Bk)).sup
          (fun j => maskedScore Bk k Q K scale idx.1 j)⟩ : Tile .real [M])
      =
      ⟨fun idx : TileIndex [M] => mPartial Bk Q N K scale (k + 1) idx.1⟩ := by
  ext idx
  simp [Tile.bop]
  rw [mPartial_succ_of_lt Bk Q N K scale k hk idx.1]

/-- Boundary loop-local `alpha = exp(m_i - m_new)` after folding
`m_new = mPartial(k+1)`. -/
theorem block_alpha_tile_eq {M S_k D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < N) :
    Tile.uop WithBot.realExp
      (Tile.bop WithBot.realSub (Broadcast.consSame Broadcast.nil)
        (⟨fun idx : TileIndex [M] =>
          mPartial Bk Q N K scale k idx.1⟩ : Tile .real [M])
        (Tile.bop max (Broadcast.consSame Broadcast.nil)
          (⟨fun idx : TileIndex [M] =>
            mPartial Bk Q N K scale k idx.1⟩ : Tile .real [M])
          (⟨fun idx : TileIndex [M] =>
            (Finset.univ : Finset (Fin Bk)).sup
              (fun j => maskedScore Bk k Q K scale idx.1 j)⟩ : Tile .real [M])))
      =
      Tile.ofReal (fun idx : TileIndex [M] =>
        alphaPartial Bk Q N K scale k idx.1) := by
  ext idx
  have hmTile :
      Tile.bop max (Broadcast.consSame Broadcast.nil)
        (⟨fun idx : TileIndex [M] =>
          mPartial Bk Q N K scale k idx.1⟩ : Tile .real [M])
        (⟨fun idx : TileIndex [M] =>
          (Finset.univ : Finset (Fin Bk)).sup
            (fun j => maskedScore Bk k Q K scale idx.1 j)⟩ : Tile .real [M])
      =
      ⟨fun idx : TileIndex [M] => mPartial Bk Q N K scale (k + 1) idx.1⟩ :=
    block_mNew_tile_eq (Bk := Bk) Q K scale k hk
  have hm :=
    congrArg (fun t : Tile .real [M] => t.data idx)
      hmTile
  simp [Tile.bop] at hm
  simp [Tile.uop, Tile.bop, Tile.ofReal, alphaPartial]
  rw [hm]
  exact realExp_eq_some_unbotD _

/-- Boundary loop-local probability tile `p = exp(scores - m_new[:,None])`
after folding `m_new = mPartial(k+1)`. -/
theorem block_p_tile_eq {M S_k D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < N) :
    Tile.uop WithBot.realExp
      (Tile.bop WithBot.realSub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (⟨fun idx : TileIndex [M, Bk] =>
          maskedScore Bk k Q K scale idx.1 idx.2.1⟩ : Tile .real [M, Bk])
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.bop max (Broadcast.consSame Broadcast.nil)
            (⟨fun idx : TileIndex [M] =>
              mPartial Bk Q N K scale k idx.1⟩ : Tile .real [M])
            (⟨fun idx : TileIndex [M] =>
              (Finset.univ : Finset (Fin Bk)).sup
                (fun j => maskedScore Bk k Q K scale idx.1 j)⟩ : Tile .real [M]))))
      =
      Tile.ofReal (fun idx : TileIndex [M, Bk] =>
        (WithBot.realExp
          (WithBot.realSub
            (maskedScore Bk k Q K scale idx.1 idx.2.1)
            (mPartial Bk Q N K scale (k + 1) idx.1))).unbotD 0) := by
  ext idx
  have hmTile :
      Tile.bop max (Broadcast.consSame Broadcast.nil)
        (⟨fun idx : TileIndex [M] =>
          mPartial Bk Q N K scale k idx.1⟩ : Tile .real [M])
        (⟨fun idx : TileIndex [M] =>
          (Finset.univ : Finset (Fin Bk)).sup
            (fun j => maskedScore Bk k Q K scale idx.1 j)⟩ : Tile .real [M])
      =
      ⟨fun idx : TileIndex [M] => mPartial Bk Q N K scale (k + 1) idx.1⟩ :=
    block_mNew_tile_eq (Bk := Bk) Q K scale k hk
  have hm :=
    congrArg (fun t : Tile .real [M] => t.data (idx.1, PUnit.unit))
      hmTile
  simp [Tile.bop] at hm
  simp [Tile.uop, Tile.bop, Tile.expandDim, Tile.ofReal, TileShape.dropInsertedIndex]
  rw [hm]
  exact realExp_eq_some_unbotD _

/-- Boundary score mask `offs_n < S_k` lifted to the `[M, Bk]` score tile. -/
theorem block_score_mask_tile_eq {M Bk S_k : Nat} (k : Nat) :
    (Tile.cop (ComparableDType.nat.lt)
      Broadcast.scalarR
      (Tile.bop NumericDType.nat.add (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Tile.bop NumericDType.nat.mul Broadcast.scalarR
          (Tile.expandDim ⟨1, by simp⟩
            (Tile.vec (fun i : Fin M => i.val)))
          (Tile.scalar 0))
        (Tile.expandDim ⟨0, by simp⟩
          (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
      (Tile.scalar S_k))
      = ⟨fun idx : TileIndex [M, Bk] =>
          decide (k * Bk + idx.2.1.val < S_k)⟩ := by
  ext idx
  obtain ⟨i, j, _⟩ := idx
  simp only [Tile.cop, Tile.bop, ComparableDType.lt, NumericDType.add,
    NumericDType.mul, Tile.expandDim_data, Tile.vec_data, Tile.scalar_data_index,
    Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex]
  simp

/-- Applying the boundary score mask to the raw score tile produces exactly
`maskedScore`: valid KV lanes carry the real scaled score; padded lanes carry
`⊥` through the explicit `-inf` branch. -/
theorem block_scores_tile_eq {M S_k D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) :
    Tile.select
      (⟨fun idx : TileIndex [M, Bk] =>
        decide (k * Bk + idx.2.1.val < S_k)⟩ : Tile .bool [M, Bk])
      (Tile.ofReal (fun idx : TileIndex [M, Bk] =>
        match blockIndex? S_k Bk k idx.2.1 with
        | some j => StreamingAccumulator.scaledScore Q K scale idx.1 j
        | none => 0))
      (⟨fun _ : TileIndex [M, Bk] => (none : WithBot ℝ)⟩ : Tile .real [M, Bk])
    = ⟨fun idx : TileIndex [M, Bk] =>
        maskedScore Bk k Q K scale idx.1 idx.2.1⟩ := by
  ext idx
  obtain ⟨i, j, u⟩ := idx
  cases u
  simp only [Tile.select_data, Tile.ofReal_data]
  by_cases h : k * Bk + j.val < S_k
  · rw [decide_eq_true h]
    simp [h, maskedScore_of_lt, blockIndex?_of_lt]
    rfl
  · rw [decide_eq_false h]
    simp [h, maskedScore_of_not_lt]
    rfl

/-- Row-max of the boundary-masked scores tile. -/
theorem block_mBlock_tile_eq {M S_k D Bk : Nat} (hBk : 0 < Bk)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) :
    Tile.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false
      ⟨fun idx : TileIndex [M, Bk] =>
        maskedScore Bk k Q K scale idx.1 idx.2.1⟩
    = some ⟨fun idx : TileIndex [M] =>
        (Finset.univ : Finset (Fin Bk)).sup
          (fun jLocal => maskedScore Bk k Q K scale idx.1 jLocal)⟩ := by
  unfold Tile.reduceMax
  simp [Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, hBk]
  funext idx
  rw [Finset.sup'_eq_sup]
  rfl

/-- Boundary loop-local `l_new` expression realizes the masked streaming
`lPartial` recurrence. -/
theorem block_lNew_tile_eq {M S_k D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < N) :
    Tile.bop WithBot.realAdd (Broadcast.consSame Broadcast.nil)
      (Tile.bop WithBot.realMul (Broadcast.consSame Broadcast.nil)
        (Tile.ofReal fun idx : TileIndex [M] =>
          alphaPartial Bk Q N K scale k idx.1)
        (Tile.ofReal fun idx : TileIndex [M] =>
          lPartial Bk Q N K scale k idx.1))
      (Tile.ofReal fun idx : TileIndex [M] =>
        Finset.univ.sum (fun j : Fin Bk =>
          (WithBot.realExp
            (WithBot.realSub
              (maskedScore Bk k Q K scale idx.1 j)
              (mPartial Bk Q N K scale (k + 1) idx.1))).unbotD 0))
      =
      Tile.ofReal (fun idx : TileIndex [M] =>
        lPartial Bk Q N K scale (k + 1) idx.1) := by
  ext idx
  simp [Tile.bop, Tile.ofReal]
  rw [lPartial_succ_of_lt Bk Q N K scale k hk idx.1]
  simp [WithBot.realSub]

/-- Boundary loop-local `o_acc` update realizes the masked streaming
`oPartial` recurrence. -/
theorem block_oAcc_tile_eq {M S_k D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < N) :
    Tile.bop WithBot.realAdd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.bop WithBot.realMul (Broadcast.consSame (Broadcast.consL Broadcast.nil))
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.ofReal fun idx : TileIndex [M] =>
            alphaPartial Bk Q N K scale k idx.1))
        (Tile.ofReal fun idx : TileIndex [M, D] =>
          oPartial Bk Q N K V scale k idx))
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        Finset.univ.sum (fun j : Fin Bk =>
          match blockIndex? S_k Bk k j with
          | some jGlobal =>
              (WithBot.realExp
                (WithBot.realSub
                  (maskedScore Bk k Q K scale idx.1 j)
                  (mPartial Bk Q N K scale (k + 1) idx.1))).unbotD 0 *
                V (jGlobal, idx.2.1, PUnit.unit)
          | none => 0))
      =
      Tile.ofReal (fun idx : TileIndex [M, D] =>
        oPartial Bk Q N K V scale (k + 1) idx) := by
  ext idx
  rcases idx with ⟨i, d, u⟩
  cases u
  simp [Tile.bop, Tile.expandDim, Tile.ofReal, TileShape.dropInsertedIndex]
  rw [oPartial_succ_of_lt Bk Q N K V scale k hk (i, d, PUnit.unit)]
  simp [WithBot.realSub]

/-! ### Boundary m-free reference sums

Mirrors `StreamingAccumulator.lFree` / `StreamingAccumulator.oFree` but indexes over the logical
`[S_k, D]` domain. Out-of-range loop lanes (`k * Bk + jLocal ≥ S_k`)
contribute zero, so summing all `numKVBlocks * Bk` slots yields the same
total as summing over `Fin S_k` whenever `S_k ≤ Bk * numKVBlocks`. -/

/-- Σ over the first `k` padded KV blocks of `exp(scaledScore)` for
in-range lanes only, no `m`-shift. Out-of-range lanes contribute 0. -/
noncomputable def lFreeBoundary {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) : ℝ :=
  Finset.univ.sum (fun n : Fin k => Finset.univ.sum (fun jL : Fin Bk =>
    match blockIndex? S_k Bk n.val jL with
    | some j => Real.exp (StreamingAccumulator.scaledScore Q K scale i j)
    | none => 0))

/-- Σ over the first `k` padded KV blocks of `exp(scaledScore) · V` for
in-range lanes only, no `m`-shift. Out-of-range lanes contribute 0. -/
noncomputable def oFreeBoundary {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (idx : TileIndex [M, D]) : ℝ :=
  Finset.univ.sum (fun n : Fin k => Finset.univ.sum (fun jL : Fin Bk =>
    match blockIndex? S_k Bk n.val jL with
    | some j =>
        Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1 j) *
          V (j, idx.2.1, PUnit.unit)
    | none => 0))

@[simp] theorem lFreeBoundary_zero {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (i : Fin M) :
    lFreeBoundary Bk Q K scale 0 i = 0 := by
  unfold lFreeBoundary
  simp

@[simp] theorem oFreeBoundary_zero {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    oFreeBoundary Bk Q K V scale 0 idx = 0 := by
  unfold oFreeBoundary
  simp

/-- Recurrence: `lFreeBoundary (k+1)` adds the next block's masked sum
on top of `lFreeBoundary k`. -/
theorem lFreeBoundary_succ {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) :
    lFreeBoundary Bk Q K scale (k + 1) i =
      lFreeBoundary Bk Q K scale k i +
      Finset.univ.sum (fun jL : Fin Bk =>
        match blockIndex? S_k Bk k jL with
        | some j => Real.exp (StreamingAccumulator.scaledScore Q K scale i j)
        | none => 0) := by
  unfold lFreeBoundary
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

/-- Recurrence: `oFreeBoundary (k+1)` adds the next block's masked
`exp · V` sum on top of `oFreeBoundary k`. -/
theorem oFreeBoundary_succ {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (idx : TileIndex [M, D]) :
    oFreeBoundary Bk Q K V scale (k + 1) idx =
      oFreeBoundary Bk Q K V scale k idx +
      Finset.univ.sum (fun jL : Fin Bk =>
        match blockIndex? S_k Bk k jL with
        | some j =>
            Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1 j) *
              V (j, idx.2.1, PUnit.unit)
        | none => 0) := by
  unfold oFreeBoundary
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

/-- Whenever the loop has reached at least one iteration AND the logical
KV length `S_k > 0` (so the very first lane `0 * Bk + 0 = 0` is in-range),
the boundary `mPartial (k+1)` is non-`⊥`. This propagates through the
running max from the first in-range contribution. -/
theorem mPartial_succ_ne_bot {M S_k D : Nat} {Bk : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (i : Fin M) :
    mPartial Bk Q numKVBlocks K scale (k + 1) i ≠ ⊥ := by
  induction k with
  | zero =>
      rw [mPartial_succ_of_lt Bk Q numKVBlocks K scale 0
            (Nat.lt_of_succ_le hk) i]
      have hVis : 0 * Bk + (⟨0, hBk⟩ : Fin Bk).val < S_k := by
        simp; exact hSk
      have h0 : maskedScore Bk 0 Q K scale i (⟨0, hBk⟩ : Fin Bk) ≠ ⊥ := by
        rw [maskedScore_of_lt Bk 0 Q K scale i _ hVis]
        exact WithBot.coe_ne_bot
      show mPartial Bk Q numKVBlocks K scale 0 i ⊔ _ ≠ ⊥
      change ⊥ ⊔ _ ≠ ⊥
      rw [bot_sup_eq]
      simp [Finset.sup_eq_bot_iff]
      exact ⟨⟨0, hBk⟩, h0⟩
  | succ k' ih =>
      have hk' : k' + 1 ≤ numKVBlocks := by omega
      rw [mPartial_succ_of_lt Bk Q numKVBlocks K scale (k' + 1)
            (Nat.lt_of_succ_le hk) i]
      intro hcontra
      have h_left : mPartial Bk Q numKVBlocks K scale (k' + 1) i ≤ ⊥ := by
        rw [← hcontra]; exact le_max_left _ _
      exact ih hk' (le_bot_iff.mp h_left)

/-- The boundary streaming `lPartial k i` equals `exp(-m_k) · lFreeBoundary k i`,
where `m_k = (mPartial k i).unbotD 0`. By induction on `k`, parallel to
`StreamingAccumulator.lPartial_eq_mShifted` and `FA1MathCausal.lPartial_eq_mShifted`. -/
theorem lPartial_eq_mShifted {M S_k D : Nat} {Bk : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (i : Fin M) :
    lPartial Bk Q numKVBlocks K scale k i =
      Real.exp (-(mPartial Bk Q numKVBlocks K scale k i).unbotD 0) *
        lFreeBoundary Bk Q K scale k i := by
  induction k with
  | zero =>
      show (0 : ℝ) =
        Real.exp (-((⊥ : WithBot ℝ).unbotD 0)) *
          lFreeBoundary Bk Q K scale 0 i
      rw [lFreeBoundary_zero]
      ring
  | succ k ih =>
      have hk' : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      rw [lPartial_succ_of_lt Bk Q numKVBlocks K scale k
        (Nat.lt_of_succ_le hk) i]
      rw [ih hk']
      rw [lFreeBoundary_succ Bk Q K scale k i, mul_add]
      -- Goal:
      --   αₖ * (exp(-mₖ) * lFreeBoundary k) +
      --   Σ exp(maskedScore - m_{k+1}).unbotD 0
      --   = exp(-m_{k+1}) * lFreeBoundary k
      --     + exp(-m_{k+1}) * Σ_{in-range} exp(scaledScore)
      have hSumB :
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            (WithBot.realExp
              (WithBot.realSub
                (maskedScore Bk k Q K scale i jLocal)
                (mPartial Bk Q numKVBlocks K scale (k + 1) i))).unbotD 0)
          =
          Real.exp (-(mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0) *
            (Finset.univ : Finset (Fin Bk)).sum (fun jL : Fin Bk =>
              match blockIndex? S_k Bk k jL with
              | some j => Real.exp (StreamingAccumulator.scaledScore Q K scale i j)
              | none => 0) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro jLocal _
        by_cases hvis : k * Bk + jLocal.val < S_k
        · rw [maskedScore_of_lt Bk k Q K scale i jLocal hvis,
              blockIndex?_of_lt S_k Bk k jLocal hvis]
          obtain ⟨m, hm⟩ := WithBot.ne_bot_iff_exists.mp
            (mPartial_succ_ne_bot hBk hSk Q numKVBlocks K scale k hk i)
          rw [← hm]
          simp
          rw [show StreamingAccumulator.scaledScore Q K scale i ⟨k * Bk + jLocal.val, hvis⟩ - m =
                -m + StreamingAccumulator.scaledScore Q K scale i ⟨k * Bk + jLocal.val, hvis⟩
                by ring,
              Real.exp_add]
        · rw [maskedScore_of_not_lt Bk k Q K scale i jLocal hvis,
              blockIndex?_of_not_lt S_k Bk k jLocal hvis]
          simp
          cases mPartial Bk Q numKVBlocks K scale (k + 1) i <;>
            (left; unfold WithBot.realExp; rfl)
      have hSumA :
          alphaPartial Bk Q numKVBlocks K scale k i *
            (Real.exp (-(mPartial Bk Q numKVBlocks K scale k i).unbotD 0) *
              lFreeBoundary Bk Q K scale k i)
          =
          Real.exp (-(mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0) *
            lFreeBoundary Bk Q K scale k i := by
        rcases Nat.eq_zero_or_pos k with hkz | hkpos
        · subst hkz
          rw [lFreeBoundary_zero]
          ring
        · obtain ⟨k', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.pos_iff_ne_zero.mp hkpos)
          have hk_succ : k' + 1 ≤ numKVBlocks := Nat.le_of_succ_le hk
          have hmk_ne :=
            mPartial_succ_ne_bot hBk hSk Q numKVBlocks K scale k' hk_succ i
          have hmk1_ne :=
            mPartial_succ_ne_bot hBk hSk Q numKVBlocks K scale (k' + 1) hk i
          obtain ⟨rk, hrk⟩ := WithBot.ne_bot_iff_exists.mp hmk_ne
          obtain ⟨rk1, hrk1⟩ := WithBot.ne_bot_iff_exists.mp hmk1_ne
          have hAlpha :
              alphaPartial Bk Q numKVBlocks K scale (k' + 1) i =
                Real.exp (rk - rk1) := by
            unfold alphaPartial
            rw [← hrk, ← hrk1]
            simp [WithBot.realSub]
          rw [hAlpha, ← hrk, ← hrk1]
          simp only [WithBot.unbotD_coe]
          rw [show Real.exp (rk - rk1) = Real.exp (-rk1) * Real.exp rk by
                rw [show rk - rk1 = -rk1 + rk by ring, Real.exp_add]]
          rw [show Real.exp (-rk1) * Real.exp rk *
                    (Real.exp (-rk) * lFreeBoundary Bk Q K scale (k' + 1) i)
                = Real.exp (-rk1) *
                    (Real.exp rk * Real.exp (-rk) *
                      lFreeBoundary Bk Q K scale (k' + 1) i) by ring]
          rw [show Real.exp rk * Real.exp (-rk) = 1 by
                rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]]
          ring
      linarith [hSumA, hSumB]

/-- Companion identity for `oPartial`: `exp(-m_k) · oFreeBoundary k idx`.
Same shape as `lPartial_eq_mShifted` plus `· V[j, d]` on each summand. -/
theorem oPartial_eq_mShifted {M S_k D : Nat} {Bk : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (idx : TileIndex [M, D]) :
    oPartial Bk Q numKVBlocks K V scale k idx =
      Real.exp (-(mPartial Bk Q numKVBlocks K scale k idx.1).unbotD 0) *
        oFreeBoundary Bk Q K V scale k idx := by
  induction k with
  | zero =>
      show (0 : ℝ) =
        Real.exp (-((⊥ : WithBot ℝ).unbotD 0)) *
          oFreeBoundary Bk Q K V scale 0 idx
      rw [oFreeBoundary_zero]
      ring
  | succ k ih =>
      have hk' : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      rw [oPartial_succ_of_lt Bk Q numKVBlocks K V scale k
        (Nat.lt_of_succ_le hk) idx]
      rw [ih hk']
      rw [oFreeBoundary_succ Bk Q K V scale k idx, mul_add]
      have hSumB :
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            match blockIndex? S_k Bk k jLocal with
            | some j =>
                (WithBot.realExp
                  (WithBot.realSub
                    (maskedScore Bk k Q K scale idx.1 jLocal)
                    (mPartial Bk Q numKVBlocks K scale (k + 1) idx.1))).unbotD 0 *
                  V (j, idx.2.1, PUnit.unit)
            | none => 0)
          =
          Real.exp (-(mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
            (Finset.univ : Finset (Fin Bk)).sum (fun jL : Fin Bk =>
              match blockIndex? S_k Bk k jL with
              | some j =>
                  Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1 j) *
                    V (j, idx.2.1, PUnit.unit)
              | none => 0) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro jLocal _
        by_cases hvis : k * Bk + jLocal.val < S_k
        · rw [maskedScore_of_lt Bk k Q K scale idx.1 jLocal hvis,
              blockIndex?_of_lt S_k Bk k jLocal hvis]
          obtain ⟨m, hm⟩ := WithBot.ne_bot_iff_exists.mp
            (mPartial_succ_ne_bot hBk hSk Q numKVBlocks K scale k hk idx.1)
          rw [← hm]
          simp
          rw [show StreamingAccumulator.scaledScore Q K scale idx.1
                  ⟨k * Bk + jLocal.val, hvis⟩ - m =
                -m + StreamingAccumulator.scaledScore Q K scale idx.1
                  ⟨k * Bk + jLocal.val, hvis⟩ by ring,
              Real.exp_add]
          ring
        · rw [blockIndex?_of_not_lt S_k Bk k jLocal hvis]
          simp
      have hSumA :
          alphaPartial Bk Q numKVBlocks K scale k idx.1 *
            (Real.exp (-(mPartial Bk Q numKVBlocks K scale k idx.1).unbotD 0) *
              oFreeBoundary Bk Q K V scale k idx)
          =
          Real.exp (-(mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
            oFreeBoundary Bk Q K V scale k idx := by
        rcases Nat.eq_zero_or_pos k with hkz | hkpos
        · subst hkz
          rw [oFreeBoundary_zero]
          ring
        · obtain ⟨k', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.pos_iff_ne_zero.mp hkpos)
          have hk_succ : k' + 1 ≤ numKVBlocks := Nat.le_of_succ_le hk
          have hmk_ne :=
            mPartial_succ_ne_bot hBk hSk Q numKVBlocks K scale k' hk_succ idx.1
          have hmk1_ne :=
            mPartial_succ_ne_bot hBk hSk Q numKVBlocks K scale (k' + 1) hk idx.1
          obtain ⟨rk, hrk⟩ := WithBot.ne_bot_iff_exists.mp hmk_ne
          obtain ⟨rk1, hrk1⟩ := WithBot.ne_bot_iff_exists.mp hmk1_ne
          have hAlpha :
              alphaPartial Bk Q numKVBlocks K scale (k' + 1) idx.1 =
                Real.exp (rk - rk1) := by
            unfold alphaPartial
            rw [← hrk, ← hrk1]
            simp [WithBot.realSub]
          rw [hAlpha, ← hrk, ← hrk1]
          simp only [WithBot.unbotD_coe]
          rw [show Real.exp (rk - rk1) = Real.exp (-rk1) * Real.exp rk by
                rw [show rk - rk1 = -rk1 + rk by ring, Real.exp_add]]
          rw [show Real.exp (-rk1) * Real.exp rk *
                    (Real.exp (-rk) * oFreeBoundary Bk Q K V scale (k' + 1) idx)
                = Real.exp (-rk1) *
                    (Real.exp rk * Real.exp (-rk) *
                      oFreeBoundary Bk Q K V scale (k' + 1) idx) by ring]
          rw [show Real.exp rk * Real.exp (-rk) = 1 by
                rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]]
          ring
      linarith [hSumA, hSumB]

/-- Flat-index form of `lFreeBoundary numKVBlocks i`: collapse the
double sum over `Fin numKVBlocks × Fin Bk` to a single sum over
`Fin (Bk * numKVBlocks)` using the same bijection as `StreamingAccumulator.lFree_eq_flat`.
The masked scores (out-of-range lanes give 0) are preserved exactly
under the bijection, since `(blockIndexEquiv j).1.val * Bk + ... = j.val`. -/
theorem lFreeBoundary_eq_flat {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (numKVBlocks : Nat) (i : Fin M) :
    lFreeBoundary Bk Q K scale numKVBlocks i =
      Finset.univ.sum (fun j' : Fin (Bk * numKVBlocks) =>
        if h : j'.val < S_k then
          Real.exp (StreamingAccumulator.scaledScore Q K scale i ⟨j'.val, h⟩)
        else
          0) := by
  unfold lFreeBoundary
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (StreamingAccumulator.blockIndexEquiv Bk numKVBlocks) ?_ ?_).symm
  · intro _; simp
  · intro j' _
    -- The bijection sends `j'` to `(n, jL)` with `n.val * Bk + jL.val = j'.val`.
    set p := StreamingAccumulator.blockIndexEquiv Bk numKVBlocks j' with hp
    show (if h : j'.val < S_k then
            Real.exp (StreamingAccumulator.scaledScore Q K scale i ⟨j'.val, h⟩)
          else
            0)
        = (match blockIndex? S_k Bk p.1.val p.2 with
          | some j => Real.exp (StreamingAccumulator.scaledScore Q K scale i j)
          | none => 0)
    have hValEq : p.1.val * Bk + p.2.val = j'.val := by
      have hBI := StreamingAccumulator.blockIndex_blockIndexEquiv (Bk := Bk) (N := numKVBlocks) j'
      have h1 := congrArg Fin.val hBI
      change p.1.val * Bk + p.2.val = j'.val at h1
      exact h1
    by_cases hLt : p.1.val * Bk + p.2.val < S_k
    · rw [blockIndex?_of_lt S_k Bk p.1.val p.2 hLt]
      have hLt' : j'.val < S_k := hValEq ▸ hLt
      rw [dif_pos hLt']
      congr 2
      apply Fin.ext
      exact hValEq.symm
    · rw [blockIndex?_of_not_lt S_k Bk p.1.val p.2 hLt]
      have hLt' : ¬ j'.val < S_k := hValEq ▸ hLt
      rw [dif_neg hLt']

/-- Flat-index form of `oFreeBoundary numKVBlocks idx`. Companion to
`lFreeBoundary_eq_flat`. -/
theorem oFreeBoundary_eq_flat {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (numKVBlocks : Nat) (idx : TileIndex [M, D]) :
    oFreeBoundary Bk Q K V scale numKVBlocks idx =
      Finset.univ.sum (fun j' : Fin (Bk * numKVBlocks) =>
        if h : j'.val < S_k then
          Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1 ⟨j'.val, h⟩) *
            V (⟨j'.val, h⟩, idx.2.1, PUnit.unit)
        else
          0) := by
  unfold oFreeBoundary
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (StreamingAccumulator.blockIndexEquiv Bk numKVBlocks) ?_ ?_).symm
  · intro _; simp
  · intro j' _
    set p := StreamingAccumulator.blockIndexEquiv Bk numKVBlocks j' with hp
    show (if h : j'.val < S_k then
            Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1 ⟨j'.val, h⟩) *
              V (⟨j'.val, h⟩, idx.2.1, PUnit.unit)
          else
            0)
        = (match blockIndex? S_k Bk p.1.val p.2 with
          | some j =>
              Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1 j) *
                V (j, idx.2.1, PUnit.unit)
          | none => 0)
    have hValEq : p.1.val * Bk + p.2.val = j'.val := by
      have hBI := StreamingAccumulator.blockIndex_blockIndexEquiv (Bk := Bk) (N := numKVBlocks) j'
      have h1 := congrArg Fin.val hBI
      change p.1.val * Bk + p.2.val = j'.val at h1
      exact h1
    by_cases hLt : p.1.val * Bk + p.2.val < S_k
    · rw [blockIndex?_of_lt S_k Bk p.1.val p.2 hLt]
      have hLt' : j'.val < S_k := hValEq ▸ hLt
      rw [dif_pos hLt']
      have hFinEq : (⟨p.1.val * Bk + p.2.val, hLt⟩ : Fin S_k) =
          ⟨j'.val, hLt'⟩ := Fin.ext hValEq
      rw [hFinEq]
    · rw [blockIndex?_of_not_lt S_k Bk p.1.val p.2 hLt]
      have hLt' : ¬ j'.val < S_k := hValEq ▸ hLt
      rw [dif_neg hLt']

/-- Reindex the flat-form `lFreeBoundary` from `Fin (Bk * numKVBlocks)` to
`Fin S_k`: padded indices `j ≥ S_k` contribute 0, so the embedding
`Fin S_k ↪ Fin (Bk * numKVBlocks)` (valid because `S_k ≤ Bk * numKVBlocks`)
exhausts the support. -/
theorem lFreeBoundary_final_eq_finSk_sum {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (numKVBlocks : Nat) (hSkLe : S_k ≤ Bk * numKVBlocks) (i : Fin M) :
    lFreeBoundary Bk Q K scale numKVBlocks i =
      Finset.univ.sum (fun j : Fin S_k =>
        Real.exp (StreamingAccumulator.scaledScore Q K scale i j)) := by
  rw [lFreeBoundary_eq_flat]
  -- Use `Fintype.sum_subset` strategy: only `j' : j'.val < S_k` contribute,
  -- and these biject with `Fin S_k` via `Fin.castLE hSkLe`.
  -- Apply `Finset.sum_bij` from `Fin S_k` to `Fin (Bk * numKVBlocks)`-with-filter.
  rw [show (Finset.univ.sum (fun j' : Fin (Bk * numKVBlocks) =>
        if h : j'.val < S_k then
          Real.exp (StreamingAccumulator.scaledScore Q K scale i ⟨j'.val, h⟩)
        else
          0)) =
        ((Finset.univ : Finset (Fin (Bk * numKVBlocks))).filter
          (fun j' => j'.val < S_k)).sum
          (fun j' : Fin (Bk * numKVBlocks) =>
            if h : j'.val < S_k then
              Real.exp (StreamingAccumulator.scaledScore Q K scale i ⟨j'.val, h⟩)
            else
              0)
        from ?_]
  · -- Now the sum is over `j' < S_k`. Reindex via `Fin.castLE hSkLe`.
    refine (Finset.sum_bij (fun (j : Fin S_k) (_ : j ∈ Finset.univ) =>
      Fin.castLE hSkLe j) ?_ ?_ ?_ ?_).symm
    · -- (mem) `Fin.castLE hSkLe j ∈ filter`
      intro j _
      simp [Fin.val_castLE, j.isLt]
    · -- (inj) `Fin.castLE hSkLe` is injective on its domain
      intro j₁ _ j₂ _ heq
      apply Fin.ext
      have := congrArg Fin.val heq
      simpa [Fin.val_castLE] using this
    · -- (surj) every `j' < S_k` is `Fin.castLE hSkLe j` for some `j`
      intro j' hj'
      simp at hj'
      refine ⟨⟨j'.val, hj'⟩, Finset.mem_univ _, ?_⟩
      apply Fin.ext
      simp
    · -- (val) value match
      intro j _
      have hLt : (Fin.castLE hSkLe j).val < S_k := by
        rw [Fin.val_castLE]; exact j.isLt
      rw [dif_pos hLt]
      congr 1
  · -- Auxiliary: sum-over-univ with 0-on-failed-dite equals filtered sum.
    refine (Finset.sum_filter_of_ne ?_).symm
    intro j' _ hNe
    by_contra hLt
    apply hNe
    rw [dif_neg hLt]

/-- Companion of `lFreeBoundary_final_eq_finSk_sum` for `oFreeBoundary`. -/
theorem oFreeBoundary_final_eq_finSk_sum {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (numKVBlocks : Nat) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (idx : TileIndex [M, D]) :
    oFreeBoundary Bk Q K V scale numKVBlocks idx =
      Finset.univ.sum (fun j : Fin S_k =>
        Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1 j) *
          V (j, idx.2.1, PUnit.unit)) := by
  rw [oFreeBoundary_eq_flat]
  rw [show (Finset.univ.sum (fun j' : Fin (Bk * numKVBlocks) =>
        if h : j'.val < S_k then
          Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1 ⟨j'.val, h⟩) *
            V (⟨j'.val, h⟩, idx.2.1, PUnit.unit)
        else
          0)) =
        ((Finset.univ : Finset (Fin (Bk * numKVBlocks))).filter
          (fun j' => j'.val < S_k)).sum
          (fun j' : Fin (Bk * numKVBlocks) =>
            if h : j'.val < S_k then
              Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1 ⟨j'.val, h⟩) *
                V (⟨j'.val, h⟩, idx.2.1, PUnit.unit)
            else
              0)
        from ?_]
  · refine (Finset.sum_bij (fun (j : Fin S_k) (_ : j ∈ Finset.univ) =>
      Fin.castLE hSkLe j) ?_ ?_ ?_ ?_).symm
    · intro j _
      simp [Fin.val_castLE, j.isLt]
    · intro j₁ _ j₂ _ heq
      apply Fin.ext
      have := congrArg Fin.val heq
      simpa [Fin.val_castLE] using this
    · intro j' hj'
      simp at hj'
      refine ⟨⟨j'.val, hj'⟩, Finset.mem_univ _, ?_⟩
      apply Fin.ext
      simp
    · intro j _
      have hLt : (Fin.castLE hSkLe j).val < S_k := by
        rw [Fin.val_castLE]; exact j.isLt
      rw [dif_pos hLt]
      congr 1
  · refine (Finset.sum_filter_of_ne ?_).symm
    intro j' _ hNe
    by_contra hLt
    apply hNe
    rw [dif_neg hLt]

/-- Boundary spec-side identity: the m-free reference ratio (over `Fin S_k`)
equals `attentionReal` directly. Mirrors `StreamingAccumulator.oFree_div_lFree_eq_attentionReal`,
adapted for `K V : [S_k, D] → ℝ`. -/
theorem oFreeBoundary_div_lFreeBoundary_eq_attentionReal
    {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (numKVBlocks : Nat) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (idx : TileIndex [M, D])
    (_hlFree : lFreeBoundary Bk Q K scale numKVBlocks idx.1 ≠ 0) :
    oFreeBoundary Bk Q K V scale numKVBlocks idx /
        lFreeBoundary Bk Q K scale numKVBlocks idx.1
      = attentionReal Q K V scale idx := by
  rcases idx with ⟨i, d, u⟩
  cases u
  rw [oFreeBoundary_final_eq_finSk_sum Bk Q K V scale numKVBlocks hSkLe
    (i, d, PUnit.unit)]
  rw [lFreeBoundary_final_eq_finSk_sum Bk Q K scale numKVBlocks hSkLe i]
  unfold attentionReal
  rfl

/-- The boundary `lFreeBoundary numKVBlocks` is strictly positive when
the logical KV scope is non-empty (`0 < S_k`): the j=0 lane (in-range
when `0 < S_k`) contributes `Real.exp _ > 0`. -/
theorem lFreeBoundary_final_pos {M S_k D : Nat} (Bk : Nat)
    (hSk : 0 < S_k)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (numKVBlocks : Nat) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (i : Fin M) :
    0 < lFreeBoundary Bk Q K scale numKVBlocks i := by
  rw [lFreeBoundary_final_eq_finSk_sum Bk Q K scale numKVBlocks hSkLe i]
  apply Finset.sum_pos'
  · intro j _
    exact le_of_lt (Real.exp_pos _)
  · refine ⟨⟨0, hSk⟩, Finset.mem_univ _, ?_⟩
    exact Real.exp_pos _

/-- The final boundary streaming normalizer is non-zero whenever the
KV scope is non-empty. Boundary analog of `StreamingAccumulator.lPartial_final_ne_zero`. -/
theorem lPartial_final_ne_zero {M S_k D : Nat} {Bk : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (hSkLe : S_k ≤ Bk * numKVBlocks)
    (i : Fin M) :
    lPartial Bk Q numKVBlocks K scale numKVBlocks i ≠ 0 := by
  rw [lPartial_eq_mShifted hBk hSk Q numKVBlocks K scale numKVBlocks
      (le_refl _) i]
  exact mul_ne_zero (Real.exp_ne_zero _)
    (ne_of_gt (lFreeBoundary_final_pos Bk hSk Q K scale numKVBlocks hSkLe i))

/-- When the logical KV scope is empty (`S_k = 0`), every loop lane is
out-of-range, so the running normalizer stays at zero. This is the
contradiction that excludes the degenerate case from
`streaming_eq_attentionReal`. -/
theorem lPartial_eq_zero_of_S_k_zero {M D : Nat} {Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [0, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) :
    lPartial Bk Q numKVBlocks K scale k i = 0 := by
  induction k with
  | zero => rfl
  | succ k ih =>
      by_cases hk : k + 1 ≤ numKVBlocks
      · rw [lPartial_succ_of_lt Bk Q numKVBlocks K scale k
            (Nat.lt_of_succ_le hk) i]
        simp only
        rw [ih]
        ring_nf
        apply Finset.sum_eq_zero
        intro jLocal _
        have hOOR : ¬ k * Bk + jLocal.val < 0 := by omega
        rw [maskedScore_of_not_lt Bk k Q K scale i jLocal hOOR]
        cases mPartial Bk Q numKVBlocks K scale (1 + k) i with
        | bot => rfl
        | coe _ => rfl
      · -- Out-of-loop branch: lPartial (k+1) = lPartial k = 0.
        change (if h : k + 1 ≤ numKVBlocks then _ else
            lPartial Bk Q numKVBlocks K scale k i) = 0
        rw [dif_neg hk]
        exact ih

/-- **Math identity (boundary).** After all `numKVBlocks` iterations,
the boundary streaming `(oPartial, lPartial)` ratio computes the same
value as `attentionReal` on the logical `[S_k, D]` KV domain. The proof
factors `exp(-m_N)` out of both numerator and denominator and matches
the residual against `attentionReal`. -/
theorem streaming_eq_attentionReal {M D Bk : Nat} (hBk : 0 < Bk)
    (Q : TileIndex [M, D] → ℝ) {S_k numKVBlocks : Nat}
    (hSkLe : S_k ≤ Bk * numKVBlocks)
    (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D])
    (hL : lPartial Bk Q numKVBlocks K scale numKVBlocks idx.1 ≠ 0) :
    oPartial Bk Q numKVBlocks K V scale numKVBlocks idx /
        lPartial Bk Q numKVBlocks K scale numKVBlocks idx.1
      = attentionReal Q K V scale idx := by
  -- We need `0 < S_k` to invoke the m-shifted bridge. Derive it from
  -- `hL`: if `S_k = 0`, then `lPartial` is identically zero
  -- (no in-range lane ever contributes), contradicting `hL`.
  rcases Nat.eq_zero_or_pos S_k with hSkz | hSk
  · subst hSkz
    exact (hL (lPartial_eq_zero_of_S_k_zero Q numKVBlocks K scale numKVBlocks
      idx.1)).elim
  · -- 0 < S_k: standard cancellation via lPartial/oPartial_eq_mShifted.
    rw [oPartial_eq_mShifted hBk hSk Q numKVBlocks K V scale numKVBlocks
          (le_refl _) idx,
        lPartial_eq_mShifted hBk hSk Q numKVBlocks K scale numKVBlocks
          (le_refl _) idx.1]
    rw [mul_div_mul_left _ _ (Real.exp_ne_zero _)]
    have hlFree : lFreeBoundary Bk Q K scale numKVBlocks idx.1 ≠ 0 := by
      intro h
      apply hL
      rw [lPartial_eq_mShifted hBk hSk Q numKVBlocks K scale numKVBlocks
          (le_refl _) idx.1, h, mul_zero]
    exact oFreeBoundary_div_lFreeBoundary_eq_attentionReal Bk Q K V scale
      numKVBlocks hSkLe idx hlFree

end FA1MathBoundary

end VeriTile.Examples
