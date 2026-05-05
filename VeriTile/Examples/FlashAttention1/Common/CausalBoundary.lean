/-
VeriTile.Examples.FlashAttention1.Common.CausalBoundary

FA-1 boundary-masked causal streaming Real recurrence.
-/

import VeriTile.Examples.FlashAttention1.Common.Boundary

namespace VeriTile.Examples

open VeriTile.Triton
/-! ## Causal boundary streaming math model

Combines the KV boundary mask with the causal `j <= qStart + i` mask.
Out-of-range or future lanes are represented as `⊥`, matching the
causal-boundary kernel's two `tl.where(..., -inf)` stages. -/

namespace FA1MathCausalBoundary

/-- `WithBot.realExp` is never bottom, so it is equal to its `unbotD`
payload rewrapped as `some`. -/
theorem realExp_eq_some_unbotD (x : WithBot ℝ) :
    WithBot.realExp x = some ((WithBot.realExp x).unbotD 0) := by
  cases x <;> rfl

@[simp] theorem realExp_realSub_bot_unbotD (m : WithBot ℝ) :
    (WithBot.realExp (WithBot.realSub (⊥ : WithBot ℝ) m)).unbotD 0 = 0 := by
  cases m <;> rfl

@[simp] theorem realExp_optionMap_sub_bot_unbotD (m : WithBot ℝ) :
    (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y) (⊥ : WithBot ℝ) m)).unbotD 0 = 0 := by
  cases m <;> rfl

/-- Causal + boundary masked score for a padded loop lane. -/
noncomputable def maskedScore {M S_k D : Nat}
    (Bk k qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (i : Fin M) (jLocal : Fin Bk) : WithBot ℝ :=
  match FA1MathBoundary.blockIndex? S_k Bk k jLocal with
  | some j =>
      if j.val ≤ qStart + i.val then
        (FA1Math.scaledScore Q K scale i j : ℝ)
      else
        ⊥
  | none => ⊥

@[simp] theorem maskedScore_of_lt_of_le {M S_k D : Nat}
    (Bk k qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (i : Fin M) (jLocal : Fin Bk)
    (hLt : k * Bk + jLocal.val < S_k)
    (hLe : k * Bk + jLocal.val ≤ qStart + i.val) :
    maskedScore Bk k qStart Q K scale i jLocal =
      ((FA1Math.scaledScore Q K scale i
        ⟨k * Bk + jLocal.val, hLt⟩ : ℝ) : WithBot ℝ) := by
  simp [maskedScore, FA1MathBoundary.blockIndex?_of_lt, hLt, hLe]

@[simp] theorem maskedScore_of_lt_of_not_le {M S_k D : Nat}
    (Bk k qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (i : Fin M) (jLocal : Fin Bk)
    (hLt : k * Bk + jLocal.val < S_k)
    (hLe : ¬ k * Bk + jLocal.val ≤ qStart + i.val) :
    maskedScore Bk k qStart Q K scale i jLocal = (⊥ : WithBot ℝ) := by
  simp [maskedScore, FA1MathBoundary.blockIndex?_of_lt, hLt, hLe]

@[simp] theorem maskedScore_of_not_lt {M S_k D : Nat}
    (Bk k qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (i : Fin M) (jLocal : Fin Bk)
    (hLt : ¬ k * Bk + jLocal.val < S_k) :
    maskedScore Bk k qStart Q K scale i jLocal = (⊥ : WithBot ℝ) := by
  simp [maskedScore, FA1MathBoundary.blockIndex?_of_not_lt, hLt]

/-- Running per-row max over the first `k` padded KV blocks. -/
noncomputable def mPartial {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ) :
    Nat → Fin M → WithBot ℝ
  | 0, _ => ⊥
  | k + 1, i =>
      if _h : k + 1 ≤ numKVBlocks then
        max (mPartial Bk qStart Q numKVBlocks K scale k i)
          ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
            maskedScore Bk k qStart Q K scale i jLocal)
      else
        mPartial Bk qStart Q numKVBlocks K scale k i

/-- Boundary-causal α multiplier `exp(m_k - m_{k+1})`. -/
noncomputable def alphaPartial {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) : ℝ :=
  (WithBot.realExp
    (WithBot.realSub
      (mPartial Bk qStart Q numKVBlocks K scale k i)
      (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i))).unbotD 0

/-- Boundary-causal running normalizer. -/
noncomputable def lPartial {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ) :
    Nat → Fin M → ℝ
  | 0, _ => 0
  | k + 1, i =>
      if _h : k + 1 ≤ numKVBlocks then
        let mNew := mPartial Bk qStart Q numKVBlocks K scale (k + 1) i
        alphaPartial Bk qStart Q numKVBlocks K scale k i *
          lPartial Bk qStart Q numKVBlocks K scale k i +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            (WithBot.realExp
              (WithBot.realSub
                (maskedScore Bk k qStart Q K scale i jLocal) mNew)).unbotD 0)
      else
        lPartial Bk qStart Q numKVBlocks K scale k i

/-- Boundary-causal running unnormalized output accumulator. -/
noncomputable def oPartial {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ) :
    Nat → TileIndex [M, D] → ℝ
  | 0, _ => 0
  | k + 1, idx =>
      if _h : k + 1 ≤ numKVBlocks then
        let i := idx.1
        let d := idx.2.1
        let mNew := mPartial Bk qStart Q numKVBlocks K scale (k + 1) i
        alphaPartial Bk qStart Q numKVBlocks K scale k i *
          oPartial Bk qStart Q numKVBlocks K V scale k idx +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            match FA1MathBoundary.blockIndex? S_k Bk k jLocal with
            | some j =>
                if j.val ≤ qStart + i.val then
                  (WithBot.realExp
                    (WithBot.realSub
                      (maskedScore Bk k qStart Q K scale i jLocal) mNew)).unbotD 0 *
                    V (j, d, PUnit.unit)
                else
                  0
            | none => 0)
      else
        oPartial Bk qStart Q numKVBlocks K V scale k idx

theorem mPartial_succ_of_lt {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    mPartial Bk qStart Q numKVBlocks K scale (k + 1) i =
      max (mPartial Bk qStart Q numKVBlocks K scale k i)
        ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
          maskedScore Bk k qStart Q K scale i jLocal) := by
  change (if h : k + 1 ≤ numKVBlocks then
      max (mPartial Bk qStart Q numKVBlocks K scale k i)
        ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
          maskedScore Bk k qStart Q K scale i jLocal)
    else mPartial Bk qStart Q numKVBlocks K scale k i) = _
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

theorem lPartial_succ_of_lt {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    lPartial Bk qStart Q numKVBlocks K scale (k + 1) i =
      let mNew := mPartial Bk qStart Q numKVBlocks K scale (k + 1) i
      alphaPartial Bk qStart Q numKVBlocks K scale k i *
        lPartial Bk qStart Q numKVBlocks K scale k i +
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          (WithBot.realExp
            (WithBot.realSub
              (maskedScore Bk k qStart Q K scale i jLocal) mNew)).unbotD 0) := by
  change (if h : k + 1 ≤ numKVBlocks then
      (let mNew := mPartial Bk qStart Q numKVBlocks K scale (k + 1) i
       alphaPartial Bk qStart Q numKVBlocks K scale k i *
          lPartial Bk qStart Q numKVBlocks K scale k i +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            (WithBot.realExp
              (WithBot.realSub
                (maskedScore Bk k qStart Q K scale i jLocal) mNew)).unbotD 0))
    else lPartial Bk qStart Q numKVBlocks K scale k i) = _
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

theorem oPartial_succ_of_lt {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (idx : TileIndex [M, D]) :
    oPartial Bk qStart Q numKVBlocks K V scale (k + 1) idx =
      let i := idx.1
      let d := idx.2.1
      let mNew := mPartial Bk qStart Q numKVBlocks K scale (k + 1) i
      alphaPartial Bk qStart Q numKVBlocks K scale k i *
        oPartial Bk qStart Q numKVBlocks K V scale k idx +
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          match FA1MathBoundary.blockIndex? S_k Bk k jLocal with
          | some j =>
              if j.val ≤ qStart + i.val then
                (WithBot.realExp
                  (WithBot.realSub
                    (maskedScore Bk k qStart Q K scale i jLocal) mNew)).unbotD 0 *
                  V (j, d, PUnit.unit)
              else
                0
          | none => 0) := by
  change (if h : k + 1 ≤ numKVBlocks then
      (let i := idx.1
       let d := idx.2.1
       let mNew := mPartial Bk qStart Q numKVBlocks K scale (k + 1) i
       alphaPartial Bk qStart Q numKVBlocks K scale k i *
          oPartial Bk qStart Q numKVBlocks K V scale k idx +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            match FA1MathBoundary.blockIndex? S_k Bk k jLocal with
            | some j =>
                if j.val ≤ qStart + i.val then
                  (WithBot.realExp
                    (WithBot.realSub
                      (maskedScore Bk k qStart Q K scale i jLocal) mNew)).unbotD 0 *
                    V (j, d, PUnit.unit)
                else
                  0
            | none => 0))
    else oPartial Bk qStart Q numKVBlocks K V scale k idx) = _
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

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

/-- Causal mask `offs_m[:, None] >= offs_n[None, :]`, stated against the
same `qStart` used by the recurrence. -/
theorem block_causal_mask_tile_eq {M Bk : Nat} (qStart k : Nat) :
    (Tile.cop (ComparableDType.nat.ge)
      (Broadcast.consR (Broadcast.consL Broadcast.nil))
      (Tile.expandDim ⟨1, by simp⟩
        (Tile.vec (fun i : Fin M => qStart + i.val)))
      (Tile.expandDim ⟨0, by simp⟩
        (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
      = ⟨fun idx : TileIndex [M, Bk] =>
          decide ((k * Bk + idx.2.1.val) ≤ (qStart + idx.1.val))⟩ := by
  ext idx
  obtain ⟨i, j, _⟩ := idx
  simp [Tile.cop, ComparableDType.ge, Tile.expandDim, Tile.vec,
    Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex]

/-- Applying the causal mask and then the boundary score mask to the raw
score tile produces exactly `maskedScore`. -/
theorem block_scores_tile_eq {M S_k D Bk : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) :
    Tile.select
      (⟨fun idx : TileIndex [M, Bk] =>
        decide (k * Bk + idx.2.1.val < S_k)⟩ : Tile .bool [M, Bk])
      (Tile.select
        (⟨fun idx : TileIndex [M, Bk] =>
          decide ((k * Bk + idx.2.1.val) ≤ (qStart + idx.1.val))⟩ :
            Tile .bool [M, Bk])
        (Tile.ofReal (fun idx : TileIndex [M, Bk] =>
          match FA1MathBoundary.blockIndex? S_k Bk k idx.2.1 with
          | some j => FA1Math.scaledScore Q K scale idx.1 j
          | none => 0))
        (⟨fun _ : TileIndex [M, Bk] => (none : WithBot ℝ)⟩ : Tile .real [M, Bk]))
      (⟨fun _ : TileIndex [M, Bk] => (none : WithBot ℝ)⟩ : Tile .real [M, Bk])
    = ⟨fun idx : TileIndex [M, Bk] =>
        maskedScore Bk k qStart Q K scale idx.1 idx.2.1⟩ := by
  ext idx
  obtain ⟨i, j, u⟩ := idx
  cases u
  simp only [Tile.select_data, Tile.ofReal_data]
  by_cases hLt : k * Bk + j.val < S_k
  · rw [decide_eq_true hLt]
    simp only [if_true]
    by_cases hLe : k * Bk + j.val ≤ qStart + i.val
    · rw [decide_eq_true hLe]
      simp [FA1MathBoundary.blockIndex?_of_lt, maskedScore_of_lt_of_le,
        hLt, hLe]
      rfl
    · rw [decide_eq_false hLe]
      simp [maskedScore_of_lt_of_not_le, hLt, hLe]
      rfl
  · rw [decide_eq_false hLt]
    simp [maskedScore_of_not_lt, hLt]
    rfl

/-- Row-max of the causal-boundary masked scores tile. -/
theorem block_mBlock_tile_eq {M S_k D Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) :
    Tile.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false
      ⟨fun idx : TileIndex [M, Bk] =>
        maskedScore Bk k qStart Q K scale idx.1 idx.2.1⟩
    = some ⟨fun idx : TileIndex [M] =>
        (Finset.univ : Finset (Fin Bk)).sup
          (fun jLocal => maskedScore Bk k qStart Q K scale idx.1 jLocal)⟩ := by
  unfold Tile.reduceMax
  simp [Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, hBk]
  funext idx
  rw [Finset.sup'_eq_sup]
  rfl

/-- Boundary-causal loop-local `m_new = max(m_i, m_block)` realizes the
streaming `mPartial` recurrence. -/
theorem block_mNew_tile_eq {M S_k D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < N) :
    Tile.bop max (Broadcast.consSame Broadcast.nil)
      (⟨fun idx : TileIndex [M] => mPartial Bk qStart Q N K scale k idx.1⟩ : Tile .real [M])
      (⟨fun idx : TileIndex [M] =>
        (Finset.univ : Finset (Fin Bk)).sup
          (fun j => maskedScore Bk k qStart Q K scale idx.1 j)⟩ : Tile .real [M])
      =
      ⟨fun idx : TileIndex [M] => mPartial Bk qStart Q N K scale (k + 1) idx.1⟩ := by
  ext idx
  simp [Tile.bop]
  rw [mPartial_succ_of_lt Bk qStart Q N K scale k hk idx.1]

/-- Boundary-causal loop-local `l_new` expression realizes the streaming
`lPartial` recurrence. -/
theorem block_lNew_tile_eq {M S_k D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < N) :
    Tile.bop WithBot.realAdd (Broadcast.consSame Broadcast.nil)
      (Tile.bop WithBot.realMul (Broadcast.consSame Broadcast.nil)
        (Tile.ofReal fun idx : TileIndex [M] =>
          alphaPartial Bk qStart Q N K scale k idx.1)
        (Tile.ofReal fun idx : TileIndex [M] =>
          lPartial Bk qStart Q N K scale k idx.1))
      (Tile.ofReal fun idx : TileIndex [M] =>
        Finset.univ.sum (fun j : Fin Bk =>
          (WithBot.realExp
            (WithBot.realSub
              (maskedScore Bk k qStart Q K scale idx.1 j)
              (mPartial Bk qStart Q N K scale (k + 1) idx.1))).unbotD 0))
      =
      Tile.ofReal (fun idx : TileIndex [M] =>
        lPartial Bk qStart Q N K scale (k + 1) idx.1) := by
  ext idx
  simp [Tile.bop, Tile.ofReal]
  rw [lPartial_succ_of_lt Bk qStart Q N K scale k hk idx.1]
  simp [WithBot.realSub]

/-- Boundary-causal loop-local `o_acc` update realizes the streaming
`oPartial` recurrence. -/
theorem block_oAcc_tile_eq {M S_k D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < N) :
    Tile.bop WithBot.realAdd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.bop WithBot.realMul (Broadcast.consSame (Broadcast.consL Broadcast.nil))
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.ofReal fun idx : TileIndex [M] =>
            alphaPartial Bk qStart Q N K scale k idx.1))
        (Tile.ofReal fun idx : TileIndex [M, D] =>
          oPartial Bk qStart Q N K V scale k idx))
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        Finset.univ.sum (fun j : Fin Bk =>
          match FA1MathBoundary.blockIndex? S_k Bk k j with
          | some jGlobal =>
              if jGlobal.val ≤ qStart + idx.1.val then
                (WithBot.realExp
                  (WithBot.realSub
                    (maskedScore Bk k qStart Q K scale idx.1 j)
                    (mPartial Bk qStart Q N K scale (k + 1) idx.1))).unbotD 0 *
                  V (jGlobal, idx.2.1, PUnit.unit)
              else
                0
          | none => 0))
      =
      Tile.ofReal (fun idx : TileIndex [M, D] =>
        oPartial Bk qStart Q N K V scale (k + 1) idx) := by
  ext idx
  rcases idx with ⟨i, d, u⟩
  cases u
  simp [Tile.bop, Tile.expandDim, Tile.ofReal, TileShape.dropInsertedIndex]
  rw [oPartial_succ_of_lt Bk qStart Q N K V scale k hk (i, d, PUnit.unit)]
  simp [WithBot.realSub]

/-! ### Causal boundary m-free reference sums -/

/-- Causal-boundary m-free normalizer over the first `k` padded KV blocks.
Out-of-range and future lanes contribute zero mass. -/
noncomputable def lFreeBoundary {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) : ℝ :=
  Finset.univ.sum (fun n : Fin k => Finset.univ.sum (fun jL : Fin Bk =>
    match FA1MathBoundary.blockIndex? S_k Bk n.val jL with
    | some j =>
        if j.val ≤ qStart + i.val then
          Real.exp (FA1Math.scaledScore Q K scale i j)
        else
          0
    | none => 0))

/-- Causal-boundary m-free unnormalized output over the first `k` padded
KV blocks. -/
noncomputable def oFreeBoundary {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (idx : TileIndex [M, D]) : ℝ :=
  Finset.univ.sum (fun n : Fin k => Finset.univ.sum (fun jL : Fin Bk =>
    match FA1MathBoundary.blockIndex? S_k Bk n.val jL with
    | some j =>
        (if j.val ≤ qStart + idx.1.val then
          Real.exp (FA1Math.scaledScore Q K scale idx.1 j)
        else
          0) * V (j, idx.2.1, PUnit.unit)
    | none => 0))

@[simp] theorem lFreeBoundary_zero {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (i : Fin M) :
    lFreeBoundary Bk qStart Q K scale 0 i = 0 := by
  unfold lFreeBoundary
  simp

@[simp] theorem oFreeBoundary_zero {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    oFreeBoundary Bk qStart Q K V scale 0 idx = 0 := by
  unfold oFreeBoundary
  simp

/-- Recurrence: `lFreeBoundary (k+1)` adds the next masked causal block. -/
theorem lFreeBoundary_succ {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) :
    lFreeBoundary Bk qStart Q K scale (k + 1) i =
      lFreeBoundary Bk qStart Q K scale k i +
      Finset.univ.sum (fun jL : Fin Bk =>
        match FA1MathBoundary.blockIndex? S_k Bk k jL with
        | some j =>
            if j.val ≤ qStart + i.val then
              Real.exp (FA1Math.scaledScore Q K scale i j)
            else
              0
        | none => 0) := by
  unfold lFreeBoundary
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

/-- Recurrence: `oFreeBoundary (k+1)` adds the next masked causal block. -/
theorem oFreeBoundary_succ {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (idx : TileIndex [M, D]) :
    oFreeBoundary Bk qStart Q K V scale (k + 1) idx =
      oFreeBoundary Bk qStart Q K V scale k idx +
      Finset.univ.sum (fun jL : Fin Bk =>
        match FA1MathBoundary.blockIndex? S_k Bk k jL with
        | some j =>
            (if j.val ≤ qStart + idx.1.val then
              Real.exp (FA1Math.scaledScore Q K scale idx.1 j)
            else
              0) * V (j, idx.2.1, PUnit.unit)
        | none => 0) := by
  unfold oFreeBoundary
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

/-- Flat-index form of the final causal-boundary normalizer. -/
theorem lFreeBoundary_eq_flat {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (numKVBlocks : Nat) (i : Fin M) :
    lFreeBoundary Bk qStart Q K scale numKVBlocks i =
      Finset.univ.sum (fun j' : Fin (Bk * numKVBlocks) =>
        if h : j'.val < S_k then
          if j'.val ≤ qStart + i.val then
            Real.exp (FA1Math.scaledScore Q K scale i ⟨j'.val, h⟩)
          else
            0
        else
          0) := by
  unfold lFreeBoundary
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (FA1Math.blockIndexEquiv Bk numKVBlocks) ?_ ?_).symm
  · intro _; simp
  · intro j' _
    set p := FA1Math.blockIndexEquiv Bk numKVBlocks j' with hp
    show (if h : j'.val < S_k then
            if j'.val ≤ qStart + i.val then
              Real.exp (FA1Math.scaledScore Q K scale i ⟨j'.val, h⟩)
            else
              0
          else
            0)
        = (match FA1MathBoundary.blockIndex? S_k Bk p.1.val p.2 with
          | some j =>
              if j.val ≤ qStart + i.val then
                Real.exp (FA1Math.scaledScore Q K scale i j)
              else
                0
          | none => 0)
    have hValEq : p.1.val * Bk + p.2.val = j'.val := by
      have hBI := FA1Math.blockIndex_blockIndexEquiv (Bk := Bk) (N := numKVBlocks) j'
      have h1 := congrArg Fin.val hBI
      change p.1.val * Bk + p.2.val = j'.val at h1
      exact h1
    by_cases hLt : p.1.val * Bk + p.2.val < S_k
    · rw [FA1MathBoundary.blockIndex?_of_lt S_k Bk p.1.val p.2 hLt]
      have hLt' : j'.val < S_k := hValEq ▸ hLt
      rw [dif_pos hLt']
      have hFinEq : (⟨p.1.val * Bk + p.2.val, hLt⟩ : Fin S_k) =
          ⟨j'.val, hLt'⟩ := Fin.ext hValEq
      rw [hFinEq]
    · rw [FA1MathBoundary.blockIndex?_of_not_lt S_k Bk p.1.val p.2 hLt]
      have hLt' : ¬ j'.val < S_k := hValEq ▸ hLt
      rw [dif_neg hLt']

/-- Flat-index form of the final causal-boundary output accumulator. -/
theorem oFreeBoundary_eq_flat {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (numKVBlocks : Nat) (idx : TileIndex [M, D]) :
    oFreeBoundary Bk qStart Q K V scale numKVBlocks idx =
      Finset.univ.sum (fun j' : Fin (Bk * numKVBlocks) =>
        if h : j'.val < S_k then
          (if j'.val ≤ qStart + idx.1.val then
            Real.exp (FA1Math.scaledScore Q K scale idx.1 ⟨j'.val, h⟩)
          else
            0) * V (⟨j'.val, h⟩, idx.2.1, PUnit.unit)
        else
          0) := by
  unfold oFreeBoundary
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (FA1Math.blockIndexEquiv Bk numKVBlocks) ?_ ?_).symm
  · intro _; simp
  · intro j' _
    set p := FA1Math.blockIndexEquiv Bk numKVBlocks j' with hp
    show (if h : j'.val < S_k then
            (if j'.val ≤ qStart + idx.1.val then
              Real.exp (FA1Math.scaledScore Q K scale idx.1 ⟨j'.val, h⟩)
            else
              0) * V (⟨j'.val, h⟩, idx.2.1, PUnit.unit)
          else
            0)
        = (match FA1MathBoundary.blockIndex? S_k Bk p.1.val p.2 with
          | some j =>
              (if j.val ≤ qStart + idx.1.val then
                Real.exp (FA1Math.scaledScore Q K scale idx.1 j)
              else
                0) * V (j, idx.2.1, PUnit.unit)
          | none => 0)
    have hValEq : p.1.val * Bk + p.2.val = j'.val := by
      have hBI := FA1Math.blockIndex_blockIndexEquiv (Bk := Bk) (N := numKVBlocks) j'
      have h1 := congrArg Fin.val hBI
      change p.1.val * Bk + p.2.val = j'.val at h1
      exact h1
    by_cases hLt : p.1.val * Bk + p.2.val < S_k
    · rw [FA1MathBoundary.blockIndex?_of_lt S_k Bk p.1.val p.2 hLt]
      have hLt' : j'.val < S_k := hValEq ▸ hLt
      rw [dif_pos hLt']
      have hFinEq : (⟨p.1.val * Bk + p.2.val, hLt⟩ : Fin S_k) =
          ⟨j'.val, hLt'⟩ := Fin.ext hValEq
      rw [hFinEq]
    · rw [FA1MathBoundary.blockIndex?_of_not_lt S_k Bk p.1.val p.2 hLt]
      have hLt' : ¬ j'.val < S_k := hValEq ▸ hLt
      rw [dif_neg hLt']

/-- Reindex the final causal-boundary normalizer from padded loop lanes to
the logical KV domain. Out-of-range padded lanes contribute zero. -/
theorem lFreeBoundary_final_eq_finSk_sum {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (numKVBlocks : Nat) (hSkLe : S_k ≤ Bk * numKVBlocks) (i : Fin M) :
    lFreeBoundary Bk qStart Q K scale numKVBlocks i =
      Finset.univ.sum (fun j : Fin S_k =>
        if j.val ≤ qStart + i.val then
          Real.exp (FA1Math.scaledScore Q K scale i j)
        else
          0) := by
  rw [lFreeBoundary_eq_flat]
  rw [show (Finset.univ.sum (fun j' : Fin (Bk * numKVBlocks) =>
        if h : j'.val < S_k then
          if j'.val ≤ qStart + i.val then
            Real.exp (FA1Math.scaledScore Q K scale i ⟨j'.val, h⟩)
          else
            0
        else
          0)) =
        ((Finset.univ : Finset (Fin (Bk * numKVBlocks))).filter
          (fun j' => j'.val < S_k)).sum
          (fun j' : Fin (Bk * numKVBlocks) =>
            if h : j'.val < S_k then
              if j'.val ≤ qStart + i.val then
                Real.exp (FA1Math.scaledScore Q K scale i ⟨j'.val, h⟩)
              else
                0
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
      simp [Fin.val_castLE]
  · refine (Finset.sum_filter_of_ne ?_).symm
    intro j' _ hNe
    by_contra hLt
    apply hNe
    rw [dif_neg hLt]

/-- Reindex the final causal-boundary output accumulator from padded loop
lanes to the logical KV domain. -/
theorem oFreeBoundary_final_eq_finSk_sum {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (numKVBlocks : Nat) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (idx : TileIndex [M, D]) :
    oFreeBoundary Bk qStart Q K V scale numKVBlocks idx =
      Finset.univ.sum (fun j : Fin S_k =>
        (if j.val ≤ qStart + idx.1.val then
          Real.exp (FA1Math.scaledScore Q K scale idx.1 j)
        else
          0) * V (j, idx.2.1, PUnit.unit)) := by
  rw [oFreeBoundary_eq_flat]
  rw [show (Finset.univ.sum (fun j' : Fin (Bk * numKVBlocks) =>
        if h : j'.val < S_k then
          (if j'.val ≤ qStart + idx.1.val then
            Real.exp (FA1Math.scaledScore Q K scale idx.1 ⟨j'.val, h⟩)
          else
            0) * V (⟨j'.val, h⟩, idx.2.1, PUnit.unit)
        else
          0)) =
        ((Finset.univ : Finset (Fin (Bk * numKVBlocks))).filter
          (fun j' => j'.val < S_k)).sum
          (fun j' : Fin (Bk * numKVBlocks) =>
            if h : j'.val < S_k then
              (if j'.val ≤ qStart + idx.1.val then
                Real.exp (FA1Math.scaledScore Q K scale idx.1 ⟨j'.val, h⟩)
              else
                0) * V (⟨j'.val, h⟩, idx.2.1, PUnit.unit)
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
      simp [Fin.val_castLE]
  · refine (Finset.sum_filter_of_ne ?_).symm
    intro j' _ hNe
    by_contra hLt
    apply hNe
    rw [dif_neg hLt]

/-- Causal-boundary m-free ratio is exactly the local-block causal attention
spec over the logical KV domain. -/
theorem oFreeBoundary_div_lFreeBoundary_eq_attentionRealCausalBlock
    {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (numKVBlocks : Nat) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (idx : TileIndex [M, D])
    (_hlFree : lFreeBoundary Bk qStart Q K scale numKVBlocks idx.1 ≠ 0) :
    oFreeBoundary Bk qStart Q K V scale numKVBlocks idx /
        lFreeBoundary Bk qStart Q K scale numKVBlocks idx.1
      = attentionRealCausalBlock qStart Q K V scale idx := by
  rw [oFreeBoundary_final_eq_finSk_sum Bk qStart Q K V scale numKVBlocks hSkLe idx]
  rw [lFreeBoundary_final_eq_finSk_sum Bk qStart Q K scale numKVBlocks hSkLe idx.1]
  unfold attentionRealCausalBlock
  rfl

/-- The final causal-boundary m-free normalizer is positive when the logical
KV domain is non-empty: key 0 is in range and causally visible. -/
theorem lFreeBoundary_final_pos {M S_k D : Nat} (Bk : Nat)
    (hSk : 0 < S_k)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (numKVBlocks : Nat) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (i : Fin M) :
    0 < lFreeBoundary Bk qStart Q K scale numKVBlocks i := by
  rw [lFreeBoundary_final_eq_finSk_sum Bk qStart Q K scale numKVBlocks hSkLe i]
  apply Finset.sum_pos'
  · intro j _
    by_cases h : j.val ≤ qStart + i.val
    · simp [h, le_of_lt (Real.exp_pos _)]
    · simp [h]
  · refine ⟨⟨0, hSk⟩, Finset.mem_univ _, ?_⟩
    simp [Real.exp_pos]

/-- Once the loop has processed at least one block and `S_k > 0`,
causal-boundary `mPartial` is non-bottom. The first logical key is both
in range and causally visible. -/
theorem mPartial_succ_ne_bot {M S_k D : Nat} {Bk : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (i : Fin M) :
    mPartial Bk qStart Q numKVBlocks K scale (k + 1) i ≠ ⊥ := by
  induction k with
  | zero =>
      rw [mPartial_succ_of_lt Bk qStart Q numKVBlocks K scale 0
            (Nat.lt_of_succ_le hk) i]
      have hLt : 0 * Bk + (⟨0, hBk⟩ : Fin Bk).val < S_k := by
        simp; exact hSk
      have hLe : 0 * Bk + (⟨0, hBk⟩ : Fin Bk).val ≤ qStart + i.val := by
        simp
      have h0 : maskedScore Bk 0 qStart Q K scale i (⟨0, hBk⟩ : Fin Bk) ≠ ⊥ := by
        rw [maskedScore_of_lt_of_le Bk 0 qStart Q K scale i _ hLt hLe]
        exact WithBot.coe_ne_bot
      show mPartial Bk qStart Q numKVBlocks K scale 0 i ⊔ _ ≠ ⊥
      change ⊥ ⊔ _ ≠ ⊥
      rw [bot_sup_eq]
      simp [Finset.sup_eq_bot_iff]
      exact ⟨⟨0, hBk⟩, h0⟩
  | succ k' ih =>
      have hk' : k' + 1 ≤ numKVBlocks := by omega
      rw [mPartial_succ_of_lt Bk qStart Q numKVBlocks K scale (k' + 1)
            (Nat.lt_of_succ_le hk) i]
      intro hcontra
      have h_left : mPartial Bk qStart Q numKVBlocks K scale (k' + 1) i ≤ ⊥ := by
        rw [← hcontra]; exact le_max_left _ _
      exact ih hk' (le_bot_iff.mp h_left)

/-- Causal-boundary streaming normalizer equals the m-free normalizer times
the final exponential shift. -/
theorem lPartial_eq_mShifted {M S_k D : Nat} {Bk : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (i : Fin M) :
    lPartial Bk qStart Q numKVBlocks K scale k i =
      Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale k i).unbotD 0) *
        lFreeBoundary Bk qStart Q K scale k i := by
  induction k with
  | zero =>
      show (0 : ℝ) =
        Real.exp (-((⊥ : WithBot ℝ).unbotD 0)) *
          lFreeBoundary Bk qStart Q K scale 0 i
      rw [lFreeBoundary_zero]
      ring
  | succ k ih =>
      have hk' : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      rw [lPartial_succ_of_lt Bk qStart Q numKVBlocks K scale k
        (Nat.lt_of_succ_le hk) i]
      rw [ih hk']
      rw [lFreeBoundary_succ Bk qStart Q K scale k i, mul_add]
      have hSumB :
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            (WithBot.realExp
              (WithBot.realSub
                (maskedScore Bk k qStart Q K scale i jLocal)
                (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i))).unbotD 0)
          =
          Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale (k + 1) i).unbotD 0) *
            (Finset.univ : Finset (Fin Bk)).sum (fun jL : Fin Bk =>
              match FA1MathBoundary.blockIndex? S_k Bk k jL with
              | some j =>
                  if j.val ≤ qStart + i.val then
                    Real.exp (FA1Math.scaledScore Q K scale i j)
                  else
                    0
              | none => 0) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro jLocal _
        by_cases hLt : k * Bk + jLocal.val < S_k
        · rw [FA1MathBoundary.blockIndex?_of_lt S_k Bk k jLocal hLt]
          by_cases hLe : k * Bk + jLocal.val ≤ qStart + i.val
          · rw [maskedScore_of_lt_of_le Bk k qStart Q K scale i jLocal hLt hLe]
            obtain ⟨m, hm⟩ := WithBot.ne_bot_iff_exists.mp
              (mPartial_succ_ne_bot hBk hSk qStart Q numKVBlocks K scale k hk i)
            rw [← hm]
            simp [hLe, WithBot.realSub]
            rw [show FA1Math.scaledScore Q K scale i ⟨k * Bk + jLocal.val, hLt⟩ - m =
                  -m + FA1Math.scaledScore Q K scale i ⟨k * Bk + jLocal.val, hLt⟩
                  by ring,
                Real.exp_add]
          · rw [maskedScore_of_lt_of_not_le Bk k qStart Q K scale i jLocal hLt hLe]
            simp [hLe]
        · rw [maskedScore_of_not_lt Bk k qStart Q K scale i jLocal hLt,
            FA1MathBoundary.blockIndex?_of_not_lt S_k Bk k jLocal hLt]
          simp
      have hSumA :
          alphaPartial Bk qStart Q numKVBlocks K scale k i *
            (Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale k i).unbotD 0) *
              lFreeBoundary Bk qStart Q K scale k i)
          =
          Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale (k + 1) i).unbotD 0) *
            lFreeBoundary Bk qStart Q K scale k i := by
        rcases Nat.eq_zero_or_pos k with hkz | hkpos
        · subst hkz
          rw [lFreeBoundary_zero]
          ring
        · obtain ⟨k', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.pos_iff_ne_zero.mp hkpos)
          have hk_succ : k' + 1 ≤ numKVBlocks := Nat.le_of_succ_le hk
          have hmk_ne :=
            mPartial_succ_ne_bot hBk hSk qStart Q numKVBlocks K scale k' hk_succ i
          have hmk1_ne :=
            mPartial_succ_ne_bot hBk hSk qStart Q numKVBlocks K scale (k' + 1) hk i
          obtain ⟨rk, hrk⟩ := WithBot.ne_bot_iff_exists.mp hmk_ne
          obtain ⟨rk1, hrk1⟩ := WithBot.ne_bot_iff_exists.mp hmk1_ne
          have hAlpha :
              alphaPartial Bk qStart Q numKVBlocks K scale (k' + 1) i =
                Real.exp (rk - rk1) := by
            unfold alphaPartial
            rw [← hrk, ← hrk1]
            simp [WithBot.realSub]
          rw [hAlpha, ← hrk, ← hrk1]
          simp only [WithBot.unbotD_coe]
          rw [show Real.exp (rk - rk1) = Real.exp (-rk1) * Real.exp rk by
                rw [show rk - rk1 = -rk1 + rk by ring, Real.exp_add]]
          rw [show Real.exp (-rk1) * Real.exp rk *
                    (Real.exp (-rk) * lFreeBoundary Bk qStart Q K scale (k' + 1) i)
                = Real.exp (-rk1) *
                    (Real.exp rk * Real.exp (-rk) *
                      lFreeBoundary Bk qStart Q K scale (k' + 1) i) by ring]
          rw [show Real.exp rk * Real.exp (-rk) = 1 by
                rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]]
          ring
      linarith [hSumA, hSumB]

/-- Causal-boundary streaming output accumulator equals the m-free output
accumulator times the final exponential shift. -/
theorem oPartial_eq_mShifted {M S_k D : Nat} {Bk : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (idx : TileIndex [M, D]) :
    oPartial Bk qStart Q numKVBlocks K V scale k idx =
      Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale k idx.1).unbotD 0) *
        oFreeBoundary Bk qStart Q K V scale k idx := by
  induction k with
  | zero =>
      show (0 : ℝ) =
        Real.exp (-((⊥ : WithBot ℝ).unbotD 0)) *
          oFreeBoundary Bk qStart Q K V scale 0 idx
      rw [oFreeBoundary_zero]
      ring
  | succ k ih =>
      have hk' : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      rw [oPartial_succ_of_lt Bk qStart Q numKVBlocks K V scale k
        (Nat.lt_of_succ_le hk) idx]
      rw [ih hk']
      rw [oFreeBoundary_succ Bk qStart Q K V scale k idx, mul_add]
      have hSumB :
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            match FA1MathBoundary.blockIndex? S_k Bk k jLocal with
            | some j =>
                if j.val ≤ qStart + idx.1.val then
                  (WithBot.realExp
                    (WithBot.realSub
                      (maskedScore Bk k qStart Q K scale idx.1 jLocal)
                      (mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1))).unbotD 0 *
                    V (j, idx.2.1, PUnit.unit)
                else
                  0
            | none => 0)
          =
          Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
            (Finset.univ : Finset (Fin Bk)).sum (fun jL : Fin Bk =>
              match FA1MathBoundary.blockIndex? S_k Bk k jL with
              | some j =>
                  (if j.val ≤ qStart + idx.1.val then
                    Real.exp (FA1Math.scaledScore Q K scale idx.1 j)
                  else
                    0) * V (j, idx.2.1, PUnit.unit)
              | none => 0) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro jLocal _
        by_cases hLt : k * Bk + jLocal.val < S_k
        · rw [FA1MathBoundary.blockIndex?_of_lt S_k Bk k jLocal hLt]
          by_cases hLe : k * Bk + jLocal.val ≤ qStart + idx.1.val
          · rw [maskedScore_of_lt_of_le Bk k qStart Q K scale idx.1 jLocal hLt hLe]
            obtain ⟨m, hm⟩ := WithBot.ne_bot_iff_exists.mp
              (mPartial_succ_ne_bot hBk hSk qStart Q numKVBlocks K scale k hk idx.1)
            rw [← hm]
            simp [hLe, WithBot.realSub]
            rw [show FA1Math.scaledScore Q K scale idx.1 ⟨k * Bk + jLocal.val, hLt⟩ - m =
                  -m + FA1Math.scaledScore Q K scale idx.1 ⟨k * Bk + jLocal.val, hLt⟩
                  by ring,
                Real.exp_add]
            ring
          · rw [maskedScore_of_lt_of_not_le Bk k qStart Q K scale idx.1 jLocal hLt hLe]
            simp [hLe]
        · rw [FA1MathBoundary.blockIndex?_of_not_lt S_k Bk k jLocal hLt]
          simp
      have hSumA :
          alphaPartial Bk qStart Q numKVBlocks K scale k idx.1 *
            (Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale k idx.1).unbotD 0) *
              oFreeBoundary Bk qStart Q K V scale k idx)
          =
          Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
            oFreeBoundary Bk qStart Q K V scale k idx := by
        rcases Nat.eq_zero_or_pos k with hkz | hkpos
        · subst hkz
          rw [oFreeBoundary_zero]
          ring
        · obtain ⟨k', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.pos_iff_ne_zero.mp hkpos)
          have hk_succ : k' + 1 ≤ numKVBlocks := Nat.le_of_succ_le hk
          have hmk_ne :=
            mPartial_succ_ne_bot hBk hSk qStart Q numKVBlocks K scale k' hk_succ idx.1
          have hmk1_ne :=
            mPartial_succ_ne_bot hBk hSk qStart Q numKVBlocks K scale (k' + 1) hk idx.1
          obtain ⟨rk, hrk⟩ := WithBot.ne_bot_iff_exists.mp hmk_ne
          obtain ⟨rk1, hrk1⟩ := WithBot.ne_bot_iff_exists.mp hmk1_ne
          have hAlpha :
              alphaPartial Bk qStart Q numKVBlocks K scale (k' + 1) idx.1 =
                Real.exp (rk - rk1) := by
            unfold alphaPartial
            rw [← hrk, ← hrk1]
            simp [WithBot.realSub]
          rw [hAlpha, ← hrk, ← hrk1]
          simp only [WithBot.unbotD_coe]
          rw [show Real.exp (rk - rk1) = Real.exp (-rk1) * Real.exp rk by
                rw [show rk - rk1 = -rk1 + rk by ring, Real.exp_add]]
          rw [show Real.exp (-rk1) * Real.exp rk *
                    (Real.exp (-rk) * oFreeBoundary Bk qStart Q K V scale (k' + 1) idx)
                = Real.exp (-rk1) *
                    (Real.exp rk * Real.exp (-rk) *
                      oFreeBoundary Bk qStart Q K V scale (k' + 1) idx) by ring]
          rw [show Real.exp rk * Real.exp (-rk) = 1 by
                rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]]
          ring
      linarith [hSumA, hSumB]

/-- The final causal-boundary streaming normalizer is nonzero when the
logical KV domain is non-empty. -/
theorem lPartial_final_ne_zero {M S_k D : Nat} {Bk : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (hSkLe : S_k ≤ Bk * numKVBlocks)
    (i : Fin M) :
    lPartial Bk qStart Q numKVBlocks K scale numKVBlocks i ≠ 0 := by
  rw [lPartial_eq_mShifted hBk hSk qStart Q numKVBlocks K scale numKVBlocks
      (le_refl _) i]
  exact mul_ne_zero (Real.exp_ne_zero _)
    (ne_of_gt (lFreeBoundary_final_pos Bk hSk qStart Q K scale numKVBlocks hSkLe i))

/-- Final causal-boundary streaming ratio equals the local-block causal
attention spec over the logical `[S_k, D]` KV domain. -/
theorem streaming_eq_attentionRealCausalBlock {M D Bk : Nat} (hBk : 0 < Bk)
    {S_k numKVBlocks : Nat} (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    oPartial Bk qStart Q numKVBlocks K V scale numKVBlocks idx /
        lPartial Bk qStart Q numKVBlocks K scale numKVBlocks idx.1
      = attentionRealCausalBlock qStart Q K V scale idx := by
  have hl : lPartial Bk qStart Q numKVBlocks K scale numKVBlocks idx.1 ≠ 0 :=
    lPartial_final_ne_zero hBk hSk qStart Q numKVBlocks K scale hSkLe idx.1
  rw [oPartial_eq_mShifted hBk hSk qStart Q numKVBlocks K V scale numKVBlocks
        (le_refl _) idx,
      lPartial_eq_mShifted hBk hSk qStart Q numKVBlocks K scale numKVBlocks
        (le_refl _) idx.1]
  rw [mul_div_mul_left _ _ (Real.exp_ne_zero _)]
  have hlFree : lFreeBoundary Bk qStart Q K scale numKVBlocks idx.1 ≠ 0 := by
    intro h
    apply hl
    rw [lPartial_eq_mShifted hBk hSk qStart Q numKVBlocks K scale numKVBlocks
        (le_refl _) idx.1, h, mul_zero]
  exact oFreeBoundary_div_lFreeBoundary_eq_attentionRealCausalBlock Bk qStart Q K V scale
    numKVBlocks hSkLe idx hlFree

end FA1MathCausalBoundary

end VeriTile.Examples
