/-
VeriTile.Examples.FlashAttention1.ScoreVariants.Block

Block-local score views, block-partial recurrences, and tile bridges.
-/

import VeriTile.Examples.FlashAttention1.ScoreVariants.Math

namespace VeriTile.Examples

open VeriTile.Triton

namespace FA1Score
/-! ## Block-local score views

The executable loop consumes one `Bk`-wide KV block at a time. These helpers
repackage the generic key-indexed score recurrence into the block-local tile
shape that the loop body computes.
-/

def scoreBlockIndex (Bk numKVBlocks k : Nat) (h : k + 1 ≤ numKVBlocks)
    (jLocal : Fin Bk) : Fin (Bk * numKVBlocks) :=
  StreamingAccumulator.blockIndex Bk numKVBlocks k h jLocal

noncomputable def scoreBlockLane {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (h : k + 1 ≤ numKVBlocks) :
    Tile .real [M, Bk] :=
  ⟨fun idx : TileIndex [M, Bk] =>
    scoreLane visible score idx.1 (scoreBlockIndex Bk numKVBlocks k h idx.2.1)⟩

def visibleBlock {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (k : Nat) (h : k + 1 ≤ numKVBlocks) :
    Tile .bool [M, Bk] :=
  ⟨fun idx : TileIndex [M, Bk] =>
    visible idx.1 (scoreBlockIndex Bk numKVBlocks k h idx.2.1)⟩

def valueBlock {D Bk numKVBlocks : Nat}
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (h : k + 1 ≤ numKVBlocks) :
    Tile .real [Bk, D] :=
  Tile.ofReal (fun idx : TileIndex [Bk, D] =>
    V (scoreBlockIndex Bk numKVBlocks k h idx.1, idx.2.1, PUnit.unit))

noncomputable def lScoreBlockFree {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (i : Fin M) : ℝ :=
  Finset.univ.sum (fun n : Fin k =>
    Finset.univ.sum (fun jLocal : Fin Bk =>
      let j := scoreBlockIndex Bk numKVBlocks n.val
        (Nat.lt_of_lt_of_le n.isLt hk) jLocal
      if visible i j then Real.exp (score i j) else 0))

noncomputable def oScoreBlockFree {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (idx : TileIndex [M, D]) : ℝ :=
  Finset.univ.sum (fun n : Fin k =>
    Finset.univ.sum (fun jLocal : Fin Bk =>
      let j := scoreBlockIndex Bk numKVBlocks n.val
        (Nat.lt_of_lt_of_le n.isLt hk) jLocal
      (if visible idx.1 j then Real.exp (score idx.1 j) else 0) *
        V (j, idx.2.1, PUnit.unit)))

@[simp] theorem lScoreBlockFree_zero {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ) (i : Fin M) :
    lScoreBlockFree visible score 0 (Nat.zero_le _) i = 0 := by
  simp [lScoreBlockFree]

@[simp] theorem oScoreBlockFree_zero {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (idx : TileIndex [M, D]) :
    oScoreBlockFree visible score V 0 (Nat.zero_le _) idx = 0 := by
  simp [oScoreBlockFree]

theorem lScoreBlockFree_succ {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (i : Fin M) :
    lScoreBlockFree visible score (k + 1) hk i =
      lScoreBlockFree visible score k (Nat.le_of_succ_le hk) i +
      Finset.univ.sum (fun jLocal : Fin Bk =>
        let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
        if visible i j then Real.exp (score i j) else 0) := by
  unfold lScoreBlockFree
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

theorem oScoreBlockFree_succ {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (idx : TileIndex [M, D]) :
    oScoreBlockFree visible score V (k + 1) hk idx =
      oScoreBlockFree visible score V k (Nat.le_of_succ_le hk) idx +
      Finset.univ.sum (fun jLocal : Fin Bk =>
        let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
        (if visible idx.1 j then Real.exp (score idx.1 j) else 0) *
          V (j, idx.2.1, PUnit.unit)) := by
  unfold oScoreBlockFree
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

theorem lScoreBlockFree_final_eq_lFreeScore {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ) (i : Fin M) :
    lScoreBlockFree visible score numKVBlocks (le_refl _) i =
      lFreeScore visible score i := by
  unfold lScoreBlockFree lFreeScore
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (StreamingAccumulator.blockIndexEquiv Bk numKVBlocks) ?_ ?_).symm
  · intro _; simp
  · intro j _
    simp [scoreBlockIndex, StreamingAccumulator.blockIndex_blockIndexEquiv]

theorem oScoreBlockFree_final_eq_oFreeScore {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (idx : TileIndex [M, D]) :
    oScoreBlockFree visible score V numKVBlocks (le_refl _) idx =
      oFreeScore visible score V idx := by
  unfold oScoreBlockFree oFreeScore
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (StreamingAccumulator.blockIndexEquiv Bk numKVBlocks) ?_ ?_).symm
  · intro _; simp
  · intro j _
    simp [scoreBlockIndex, StreamingAccumulator.blockIndex_blockIndexEquiv]

theorem scoreBlock_exp_shift_sum_eq {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (i : Fin M) (m : ℝ) :
    (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
      (WithBot.realExp
        (WithBot.realSub
          (scoreLane visible score i
            (scoreBlockIndex Bk numKVBlocks k hk jLocal))
          ((m : ℝ) : WithBot ℝ))).unbotD 0) =
      Real.exp (-m) *
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
          if visible i j then Real.exp (score i j) else 0) := by
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro jLocal _
  have hShift :=
    scoreLane_exp_shift_eq visible score i
      (scoreBlockIndex Bk numKVBlocks k hk jLocal) m
  simpa [WithBot.realSub] using hShift

theorem scoreBlock_exp_shift_v_sum_eq {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (idx : TileIndex [M, D]) (m : ℝ) :
    (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
      let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
      (WithBot.realExp
        (WithBot.realSub
          (scoreLane visible score idx.1 j)
          ((m : ℝ) : WithBot ℝ))).unbotD 0 *
        V (j, idx.2.1, PUnit.unit)) =
      Real.exp (-m) *
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
          (if visible idx.1 j then Real.exp (score idx.1 j) else 0) *
            V (j, idx.2.1, PUnit.unit)) := by
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro jLocal _
  let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
  have hShift :=
    scoreLane_exp_shift_eq visible score idx.1 j m
  have hShift' :
      (WithBot.realExp
        (WithBot.realSub
          (scoreLane visible score idx.1 j)
          ((m : ℝ) : WithBot ℝ))).unbotD 0 =
        Real.exp (-m) *
          if visible idx.1 j then
            Real.exp (score idx.1 j)
          else 0 := by
    simpa [WithBot.realSub] using hShift
  change
    (WithBot.realExp
      (WithBot.realSub (scoreLane visible score idx.1 j)
        ((m : ℝ) : WithBot ℝ))).unbotD 0 *
        V (j, idx.2.1, PUnit.unit) =
      Real.exp (-m) *
        ((if visible idx.1 j then Real.exp (score idx.1 j) else 0) *
          V (j, idx.2.1, PUnit.unit))
  rw [hShift']
  by_cases hVisible : visible idx.1 j = Bool.true
  · simp [hVisible]
    ring
  · have hFalse : visible idx.1 j = Bool.false := by
      cases hv : visible idx.1 j
      · rfl
      · exact (hVisible hv).elim
    simp [hFalse]

theorem scoreBlock_visible_false_of_sup_eq_bot {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (i : Fin M)
    (hSup :
      ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
        scoreLane visible score i
          (scoreBlockIndex Bk numKVBlocks k hk jLocal)) = (⊥ : WithBot ℝ))
    (jLocal : Fin Bk) :
    visible i (scoreBlockIndex Bk numKVBlocks k hk jLocal) = Bool.false := by
  have hAll :
      ∀ j ∈ (Finset.univ : Finset (Fin Bk)),
        scoreLane visible score i
          (scoreBlockIndex Bk numKVBlocks k hk j) = (⊥ : WithBot ℝ) := by
    simpa [Finset.sup_eq_bot_iff] using hSup
  have hLane := hAll jLocal (Finset.mem_univ _)
  cases hv : visible i (scoreBlockIndex Bk numKVBlocks k hk jLocal)
  · rfl
  · have hLane' :
        scoreLane visible score i
            (scoreBlockIndex Bk numKVBlocks k hk jLocal) =
          ((score i (scoreBlockIndex Bk numKVBlocks k hk jLocal) : ℝ) :
            WithBot ℝ) := by
      simp [scoreLane, hv]
    rw [hLane'] at hLane
    exact (WithBot.coe_ne_bot hLane).elim

@[simp] theorem scoreBlockLane_data {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (h : k + 1 ≤ numKVBlocks) (i : Fin M) (j : Fin Bk) :
    (scoreBlockLane visible score k h).data (i, j, PUnit.unit) =
      scoreLane visible score i (scoreBlockIndex Bk numKVBlocks k h j) := rfl

@[simp] theorem visibleBlock_data {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (k : Nat) (h : k + 1 ≤ numKVBlocks) (i : Fin M) (j : Fin Bk) :
    (visibleBlock visible k h).data (i, j, PUnit.unit) =
      visible i (scoreBlockIndex Bk numKVBlocks k h j) := rfl

@[simp] theorem valueBlock_data {D Bk numKVBlocks : Nat}
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (h : k + 1 ≤ numKVBlocks) (j : Fin Bk) (d : Fin D) :
    (valueBlock V k h).data (j, d, PUnit.unit) =
      ((V (scoreBlockIndex Bk numKVBlocks k h j, d, PUnit.unit) : ℝ) :
        WithBot ℝ) := rfl

theorem fa1_score_block_read
    {D Bk numKVBlocks : Nat}
    (region : RegionName) (s : BlockState)
    (X : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (hX : InputAt s region
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) X)
    (k : Nat) (hk : k < numKVBlocks)
    (j : Fin Bk) (d : Fin D) :
    s.readMem region ((k * Bk + j.val) * D + d.val) =
      X (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j,
        d, PUnit.unit) := by
  rw [BlockState.readMem]
  have haddr :
      (k * Bk + j.val) * D + d.val =
        Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D
          (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j,
            d, PUnit.unit) := by
    simp [Offset.rowMajor2D, Offset.strided, scoreBlockIndex, StreamingAccumulator.blockIndex]
  rw [haddr]
  exact hX _

theorem fa1_score_block_load_tile_eq
    {D Bk numKVBlocks : Nat}
    (region : RegionName) (s : BlockState)
    (X : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (hX : InputAt s region
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) X)
    (k : Nat) (hk : k < numKVBlocks) :
    (⟨fun idx : TileIndex [Bk, D] =>
        some (s.readMem region ((k * Bk + idx.1.val) * D + idx.2.1.val))⟩
      : Tile .real [Bk, D])
      =
      valueBlock X k (Nat.succ_le_iff.mpr hk) := by
  ext idx
  rw [valueBlock_data]
  exact congrArg some (fa1_score_block_read region s X hX k hk idx.1 idx.2.1)

theorem fa1_score_block_mem_load_tile_eq
    {D Bk numKVBlocks : Nat}
    (region : RegionName) (s : BlockState)
    (X : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (hX : InputAt s region
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) X)
    (k : Nat) (hk : k < numKVBlocks) :
    (⟨fun idx : TileIndex [Bk, D] =>
        some (s.readMem region ((k * Bk + idx.1.val) * D + idx.2.1.val))⟩
      : Tile .real [Bk, D])
      =
      valueBlock X k (Nat.succ_le_iff.mpr hk) := by
  ext idx
  rw [valueBlock_data]
  exact congrArg some (by
    simpa [BlockState.readMem] using
      fa1_score_block_read region s X hX k hk idx.1 idx.2.1)

noncomputable def rawScoreBlockTile {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (h : k + 1 ≤ numKVBlocks) :
    Tile .real [M, Bk] :=
  Tile.bop NumericDType.real.mul Broadcast.scalarR
    (Tile.dot [] (Tile.ofReal Q)
      (Tile.transpose [] (Tile.ofReal
        (fun idx : TileIndex [Bk, D] =>
          K (scoreBlockIndex Bk numKVBlocks k h idx.1, idx.2.1, PUnit.unit)))))
    (Tile.scalar ((scale : ℝ) : WithBot ℝ))

def distanceBlockTile {M Bk : Nat} (qStart : Nat) (k : Nat) :
    Tile .real [M, Bk] :=
  Tile.bop max (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    (Tile.bop WithBot.realSub (Broadcast.consL (Broadcast.consR Broadcast.nil))
      (Tile.natToReal
        (Tile.expandDim ⟨0, by simp⟩
          (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
      (Tile.natToReal
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.vec (fun i : Fin M => qStart + i.val)))))
    (Tile.bop WithBot.realSub Broadcast.scalarL
      (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
      (Tile.bop WithBot.realSub (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Tile.natToReal
          (Tile.expandDim ⟨0, by simp⟩
            (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
        (Tile.natToReal
          (Tile.expandDim ⟨1, by simp⟩
          (Tile.vec (fun i : Fin M => qStart + i.val))))))

def distanceBlockTileNumeric {M Bk : Nat} (qStart : Nat) (k : Nat) :
    Tile .real [M, Bk] :=
  Tile.bop max (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    (Tile.bop NumericDType.real.sub (Broadcast.consL (Broadcast.consR Broadcast.nil))
      (Tile.natToReal
        (Tile.expandDim ⟨0, by simp⟩
          (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
      (Tile.natToReal
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.vec (fun i : Fin M => qStart + i.val)))))
    (Tile.bop NumericDType.real.sub Broadcast.scalarL
      (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
      (Tile.bop NumericDType.real.sub (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Tile.natToReal
          (Tile.expandDim ⟨0, by simp⟩
            (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
        (Tile.natToReal
          (Tile.expandDim ⟨1, by simp⟩
            (Tile.vec (fun i : Fin M => qStart + i.val))))))

theorem distanceBlockTileNumeric_eq {M Bk : Nat} (qStart : Nat) (k : Nat) :
    distanceBlockTileNumeric (M := M) (Bk := Bk) qStart k =
      distanceBlockTile (M := M) (Bk := Bk) qStart k := by
  rfl

noncomputable def slidingVisibleBlockTile {M Bk : Nat}
    (qStart window : Nat) (k : Nat) : Tile .bool [M, Bk] :=
  Tile.cop ComparableDType.real.lt Broadcast.scalarR
    (distanceBlockTile (M := M) (Bk := Bk) qStart k)
    (Tile.scalar (((window : ℝ) : WithBot ℝ)))

noncomputable def alibiScoreBlockTile {M D Bk numKVBlocks : Nat}
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (h : k + 1 ≤ numKVBlocks) :
    Tile .real [M, Bk] :=
  Tile.bop WithBot.realAdd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    (rawScoreBlockTile Q K scale k h)
    (Tile.bop WithBot.realSub Broadcast.scalarL
      (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
      (Tile.bop WithBot.realMul Broadcast.scalarL
        (Tile.scalar ((slope : ℝ) : WithBot ℝ))
        (distanceBlockTile (M := M) (Bk := Bk) qStart k)))

noncomputable def softcapScoreTile {M Bk : Nat}
    (softcap : ℝ) (scores : Tile .real [M, Bk]) : Tile .real [M, Bk] :=
  Tile.bop WithBot.realMul Broadcast.scalarL
    (Tile.scalar ((softcap : ℝ) : WithBot ℝ))
    (Tile.uop WithBot.realTanh
      (Tile.bop WithBot.realDiv Broadcast.scalarR
        scores
        (Tile.scalar ((softcap : ℝ) : WithBot ℝ))))

theorem softcapScoreTileNumeric_eq {M Bk : Nat}
    (softcap : ℝ) (scores : Tile .real [M, Bk]) :
    Tile.bop NumericDType.real.mul Broadcast.scalarL
      (Tile.scalar ((softcap : ℝ) : WithBot ℝ))
      (Tile.uop WithBot.realTanh
        (Tile.bop NumericDType.real.div Broadcast.scalarR
          scores
          (Tile.scalar ((softcap : ℝ) : WithBot ℝ)))) =
      softcapScoreTile softcap scores := by
  rfl

def maskedScoreTile {M Bk : Nat}
    (visible : Tile .bool [M, Bk]) (scores : Tile .real [M, Bk]) :
    Tile .real [M, Bk] :=
  Tile.select visible scores
    (⟨fun _ : TileIndex [M, Bk] => (⊥ : WithBot ℝ)⟩ : Tile .real [M, Bk])

/-! ## Block-partial score recurrences

These mirror the executable FA-1 loop's block granularity while keeping the
score and visibility predicates generic. They are the math bridge needed by
the loop-step proof: one loop iteration consumes exactly one `Bk`-wide score
block.
-/

noncomputable def mScoreBlockPartial {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ) :
    Nat → Fin M → WithBot ℝ
  | 0, _ => ⊥
  | k + 1, i =>
      if h : k + 1 ≤ numKVBlocks then
        max (mScoreBlockPartial visible score k i)
          ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
            scoreLane visible score i (scoreBlockIndex Bk numKVBlocks k h jLocal))
      else
        mScoreBlockPartial visible score k i

noncomputable def alphaScoreBlockPartial {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (i : Fin M) : ℝ :=
  (WithBot.realExp
    (WithBot.realSub
      (mScoreBlockPartial visible score k i)
      (mScoreBlockPartial visible score (k + 1) i))).unbotD 0

noncomputable def lScoreBlockPartial {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ) :
    Nat → Fin M → ℝ
  | 0, _ => 0
  | k + 1, i =>
      if h : k + 1 ≤ numKVBlocks then
        alphaScoreBlockPartial visible score k i *
          lScoreBlockPartial visible score k i +
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          (WithBot.realExp
            (WithBot.realSub
              (scoreLane visible score i (scoreBlockIndex Bk numKVBlocks k h jLocal))
              (mScoreBlockPartial visible score (k + 1) i))).unbotD 0)
      else
        lScoreBlockPartial visible score k i

noncomputable def oScoreBlockPartial {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ) :
    Nat → TileIndex [M, D] → ℝ
  | 0, _ => 0
  | k + 1, idx =>
      if h : k + 1 ≤ numKVBlocks then
        alphaScoreBlockPartial visible score k idx.1 *
          oScoreBlockPartial visible score V k idx +
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          let j := scoreBlockIndex Bk numKVBlocks k h jLocal
          (WithBot.realExp
            (WithBot.realSub
              (scoreLane visible score idx.1 j)
              (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0 *
            V (j, idx.2.1, PUnit.unit))
      else
        oScoreBlockPartial visible score V k idx

theorem mScoreBlockPartial_succ_of_lt {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    mScoreBlockPartial visible score (k + 1) i =
      max (mScoreBlockPartial visible score k i)
        ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
          scoreLane visible score i
            (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal)) := by
  conv_lhs => rw [mScoreBlockPartial]
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

theorem mScoreBlockPartial_succ {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (i : Fin M) :
    mScoreBlockPartial visible score (k + 1) i =
      max (mScoreBlockPartial visible score k i)
        ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
          scoreLane visible score i
            (scoreBlockIndex Bk numKVBlocks k hk jLocal)) := by
  conv_lhs => rw [mScoreBlockPartial]
  rw [dif_pos hk]

theorem lScoreBlockPartial_succ_of_lt {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    lScoreBlockPartial visible score (k + 1) i =
      alphaScoreBlockPartial visible score k i *
        lScoreBlockPartial visible score k i +
      (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
        (WithBot.realExp
          (WithBot.realSub
            (scoreLane visible score i
              (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal))
            (mScoreBlockPartial visible score (k + 1) i))).unbotD 0) := by
  conv_lhs => rw [lScoreBlockPartial]
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

theorem lScoreBlockPartial_succ {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (i : Fin M) :
    lScoreBlockPartial visible score (k + 1) i =
      alphaScoreBlockPartial visible score k i *
        lScoreBlockPartial visible score k i +
      (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
        (WithBot.realExp
          (WithBot.realSub
            (scoreLane visible score i
              (scoreBlockIndex Bk numKVBlocks k hk jLocal))
            (mScoreBlockPartial visible score (k + 1) i))).unbotD 0) := by
  conv_lhs => rw [lScoreBlockPartial]
  rw [dif_pos hk]

theorem oScoreBlockPartial_succ_of_lt {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (hk : k < numKVBlocks) (idx : TileIndex [M, D]) :
    oScoreBlockPartial visible score V (k + 1) idx =
      alphaScoreBlockPartial visible score k idx.1 *
        oScoreBlockPartial visible score V k idx +
      (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
        let j := scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal
        (WithBot.realExp
          (WithBot.realSub
            (scoreLane visible score idx.1 j)
            (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0 *
          V (j, idx.2.1, PUnit.unit)) := by
  conv_lhs => rw [oScoreBlockPartial]
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

theorem oScoreBlockPartial_succ {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (idx : TileIndex [M, D]) :
    oScoreBlockPartial visible score V (k + 1) idx =
      alphaScoreBlockPartial visible score k idx.1 *
        oScoreBlockPartial visible score V k idx +
      (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
        let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
        (WithBot.realExp
          (WithBot.realSub
            (scoreLane visible score idx.1 j)
            (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0 *
          V (j, idx.2.1, PUnit.unit)) := by
  conv_lhs => rw [oScoreBlockPartial]
  rw [dif_pos hk]

theorem lScoreBlockFree_eq_zero_of_mScoreBlockPartial_eq_bot {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (i : Fin M)
    (hm : mScoreBlockPartial visible score k i = (⊥ : WithBot ℝ)) :
    lScoreBlockFree visible score k hk i = 0 := by
  induction k with
  | zero =>
      simp
  | succ k ih =>
      have hkPrev : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      have hkLt : k < numKVBlocks := Nat.lt_of_succ_le hk
      have hmStep := hm
      rw [mScoreBlockPartial_succ visible score k hk i] at hmStep
      cases hOld : mScoreBlockPartial visible score k i <;>
        cases hSup :
          ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
            scoreLane visible score i
              (scoreBlockIndex Bk numKVBlocks k hk jLocal)) <;>
        simp [hOld, hSup] at hmStep
      rw [lScoreBlockFree_succ visible score k hk i]
      have hFreeOld :
          lScoreBlockFree visible score k hkPrev i = 0 := ih hkPrev hOld
      have hBlock :
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
            if visible i j then Real.exp (score i j) else 0) = 0 := by
        apply Finset.sum_eq_zero
        intro jLocal _
        have hFalse :=
          scoreBlock_visible_false_of_sup_eq_bot visible score k hk i hSup jLocal
        simp [hFalse]
      simp [hFreeOld, hBlock]

theorem oScoreBlockFree_eq_zero_of_mScoreBlockPartial_eq_bot {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (idx : TileIndex [M, D])
    (hm : mScoreBlockPartial visible score k idx.1 = (⊥ : WithBot ℝ)) :
    oScoreBlockFree visible score V k hk idx = 0 := by
  induction k with
  | zero =>
      simp
  | succ k ih =>
      have hkPrev : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      have hkLt : k < numKVBlocks := Nat.lt_of_succ_le hk
      have hmStep := hm
      rw [mScoreBlockPartial_succ visible score k hk idx.1] at hmStep
      cases hOld : mScoreBlockPartial visible score k idx.1 <;>
        cases hSup :
          ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
            scoreLane visible score idx.1
              (scoreBlockIndex Bk numKVBlocks k hk jLocal)) <;>
        simp [hOld, hSup] at hmStep
      rw [oScoreBlockFree_succ visible score V k hk idx]
      have hFreeOld :
          oScoreBlockFree visible score V k hkPrev idx = 0 := ih hkPrev hOld
      have hBlock :
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
            if visible idx.1 j then
              Real.exp (score idx.1 j) * V (j, idx.2.1, PUnit.unit)
            else
              0) = 0 := by
        apply Finset.sum_eq_zero
        intro jLocal _
        have hFalse :=
          scoreBlock_visible_false_of_sup_eq_bot visible score k hk idx.1 hSup jLocal
        simp [hFalse]
      simp [hFreeOld, hBlock]

theorem lScoreBlockPartial_eq_mShifted {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (i : Fin M) :
    lScoreBlockPartial visible score k i =
      Real.exp (-(mScoreBlockPartial visible score k i).unbotD 0) *
        lScoreBlockFree visible score k hk i := by
  induction k with
  | zero =>
      rw [lScoreBlockFree_zero]
      simp [lScoreBlockPartial]
  | succ k ih =>
      have hkPrev : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      have hkLt : k < numKVBlocks := Nat.lt_of_succ_le hk
      rw [lScoreBlockPartial_succ visible score k hk i]
      rw [lScoreBlockFree_succ visible score k hk i]
      rw [ih hkPrev]
      cases hmNew : mScoreBlockPartial visible score (k + 1) i with
      | bot =>
          have hmStep := hmNew
          rw [mScoreBlockPartial_succ visible score k hk i] at hmStep
          cases hmOld : mScoreBlockPartial visible score k i <;>
            cases hSup :
              ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
                scoreLane visible score i
                  (scoreBlockIndex Bk numKVBlocks k hk jLocal)) <;>
            simp [hmOld, hSup] at hmStep
          have hFreeOld :
              lScoreBlockFree visible score k hkPrev i = 0 :=
            lScoreBlockFree_eq_zero_of_mScoreBlockPartial_eq_bot
              visible score k hkPrev i hmOld
          have hBlockFree :
              (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
                let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
                if visible i j then Real.exp (score i j) else 0) = 0 := by
            apply Finset.sum_eq_zero
            intro jLocal _
            have hFalse :=
              scoreBlock_visible_false_of_sup_eq_bot visible score k hk i hSup jLocal
            simp [hFalse]
          have hBlockPartial :
              (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
                (WithBot.realExp
                  (WithBot.realSub
                    (scoreLane visible score i
                      (scoreBlockIndex Bk numKVBlocks k hk jLocal))
                    (mScoreBlockPartial visible score (k + 1) i))).unbotD 0) = 0 := by
            apply Finset.sum_eq_zero
            intro jLocal _
            have hFalse :=
              scoreBlock_visible_false_of_sup_eq_bot visible score k hk i hSup jLocal
            simp [hmNew, hFalse, scoreLane_of_false]
          simp [alphaScoreBlockPartial, hmOld, hmNew, hFreeOld, hBlockFree]
          simpa [WithBot.realSub, hmNew] using hBlockPartial
      | coe mNew =>
          cases hmOld : mScoreBlockPartial visible score k i with
          | bot =>
              have hFreeOld :
                  lScoreBlockFree visible score k hkPrev i = 0 :=
                lScoreBlockFree_eq_zero_of_mScoreBlockPartial_eq_bot
                  visible score k hkPrev i hmOld
              simp [alphaScoreBlockPartial, hmOld, hmNew, hFreeOld]
              simpa [WithBot.realSub] using
                scoreBlock_exp_shift_sum_eq visible score k hk i mNew
          | coe mOld =>
              simp [alphaScoreBlockPartial, hmOld, hmNew]
              rw [show Real.exp (mOld - mNew) *
                      (Real.exp (-mOld) *
                        lScoreBlockFree visible score k hkPrev i) =
                    Real.exp (-mNew) *
                        lScoreBlockFree visible score k hkPrev i
                  from alphaScore_shift_cancel mOld mNew
                    (lScoreBlockFree visible score k hkPrev i)]
              rw [show
                  (Finset.univ : Finset (Fin Bk)).sum (fun x =>
                    (WithBot.realExp
                      (Option.map₂ (fun x1 x2 => x1 - x2)
                        (scoreLane visible score i
                          (scoreBlockIndex Bk numKVBlocks k hk x))
                        ((mNew : ℝ) : WithBot ℝ))).unbotD 0) =
                    Real.exp (-mNew) *
                      (Finset.univ : Finset (Fin Bk)).sum (fun x =>
                        let j := scoreBlockIndex Bk numKVBlocks k hk x
                        if visible i j then Real.exp (score i j) else 0) by
                simpa [WithBot.realSub] using
                  scoreBlock_exp_shift_sum_eq visible score k hk i mNew]
              ring

theorem oScoreBlockPartial_eq_mShifted {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (idx : TileIndex [M, D]) :
    oScoreBlockPartial visible score V k idx =
      Real.exp (-(mScoreBlockPartial visible score k idx.1).unbotD 0) *
        oScoreBlockFree visible score V k hk idx := by
  induction k with
  | zero =>
      rw [oScoreBlockFree_zero]
      simp [oScoreBlockPartial]
  | succ k ih =>
      have hkPrev : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      rw [oScoreBlockPartial_succ visible score V k hk idx]
      rw [oScoreBlockFree_succ visible score V k hk idx]
      rw [ih hkPrev]
      cases hmNew : mScoreBlockPartial visible score (k + 1) idx.1 with
      | bot =>
          have hmStep := hmNew
          rw [mScoreBlockPartial_succ visible score k hk idx.1] at hmStep
          cases hmOld : mScoreBlockPartial visible score k idx.1 <;>
            cases hSup :
              ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
                scoreLane visible score idx.1
                  (scoreBlockIndex Bk numKVBlocks k hk jLocal)) <;>
            simp [hmOld, hSup] at hmStep
          have hFreeOld :
              oScoreBlockFree visible score V k hkPrev idx = 0 :=
            oScoreBlockFree_eq_zero_of_mScoreBlockPartial_eq_bot
              visible score V k hkPrev idx hmOld
          have hBlockFree :
              (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
                let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
                if visible idx.1 j then
                  Real.exp (score idx.1 j) * V (j, idx.2.1, PUnit.unit)
                else
                  0) = 0 := by
            apply Finset.sum_eq_zero
            intro jLocal _
            have hFalse :=
              scoreBlock_visible_false_of_sup_eq_bot visible score k hk idx.1 hSup jLocal
            simp [hFalse]
          have hBlockPartial :
              (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
                let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
                (WithBot.realExp
                  (WithBot.realSub
                    (scoreLane visible score idx.1 j)
                    (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0 *
                  V (j, idx.2.1, PUnit.unit)) = 0 := by
            apply Finset.sum_eq_zero
            intro jLocal _
            have hFalse :=
              scoreBlock_visible_false_of_sup_eq_bot visible score k hk idx.1 hSup jLocal
            simp [hmNew, hFalse, scoreLane_of_false]
          simp [alphaScoreBlockPartial, hmOld, hmNew, hFreeOld, hBlockFree]
          simpa [WithBot.realSub, hmNew] using hBlockPartial
      | coe mNew =>
          cases hmOld : mScoreBlockPartial visible score k idx.1 with
          | bot =>
              have hFreeOld :
                  oScoreBlockFree visible score V k hkPrev idx = 0 :=
                oScoreBlockFree_eq_zero_of_mScoreBlockPartial_eq_bot
                  visible score V k hkPrev idx hmOld
              simp [alphaScoreBlockPartial, hmOld, hmNew, hFreeOld]
              simpa [WithBot.realSub] using
                scoreBlock_exp_shift_v_sum_eq visible score V k hk idx mNew
          | coe mOld =>
              simp [alphaScoreBlockPartial, hmOld, hmNew]
              rw [show Real.exp (mOld - mNew) *
                      (Real.exp (-mOld) *
                        oScoreBlockFree visible score V k hkPrev idx) =
                    Real.exp (-mNew) *
                        oScoreBlockFree visible score V k hkPrev idx
                  from alphaScore_shift_cancel mOld mNew
                    (oScoreBlockFree visible score V k hkPrev idx)]
              rw [show
                  (Finset.univ : Finset (Fin Bk)).sum (fun x =>
                    let j := scoreBlockIndex Bk numKVBlocks k hk x
                    (WithBot.realExp
                      (Option.map₂ (fun x1 x2 => x1 - x2)
                        (scoreLane visible score idx.1 j)
                        ((mNew : ℝ) : WithBot ℝ))).unbotD 0 *
                      V (j, idx.2.1, PUnit.unit)) =
                    Real.exp (-mNew) *
                      (Finset.univ : Finset (Fin Bk)).sum (fun x =>
                        let j := scoreBlockIndex Bk numKVBlocks k hk x
                        (if visible idx.1 j then Real.exp (score idx.1 j) else 0) *
                          V (j, idx.2.1, PUnit.unit)) by
                simpa [WithBot.realSub] using
                  scoreBlock_exp_shift_v_sum_eq visible score V k hk idx mNew]
              have hIfMul :
                  (Finset.univ : Finset (Fin Bk)).sum (fun x =>
                    (if visible idx.1 (scoreBlockIndex Bk numKVBlocks k hk x) then
                        Real.exp (score idx.1 (scoreBlockIndex Bk numKVBlocks k hk x))
                      else 0) *
                      V (scoreBlockIndex Bk numKVBlocks k hk x, idx.2.1, PUnit.unit)) =
                  (Finset.univ : Finset (Fin Bk)).sum (fun x =>
                    if visible idx.1 (scoreBlockIndex Bk numKVBlocks k hk x) then
                      Real.exp (score idx.1 (scoreBlockIndex Bk numKVBlocks k hk x)) *
                        V (scoreBlockIndex Bk numKVBlocks k hk x, idx.2.1, PUnit.unit)
                    else 0) := by
                apply Finset.sum_congr rfl
                intro x _
                by_cases hVisible :
                    visible idx.1 (scoreBlockIndex Bk numKVBlocks k hk x) = Bool.true
                · simp [hVisible]
                · have hFalse :
                      visible idx.1 (scoreBlockIndex Bk numKVBlocks k hk x) = Bool.false := by
                    cases hv : visible idx.1 (scoreBlockIndex Bk numKVBlocks k hk x)
                    · rfl
                    · exact (hVisible hv).elim
                  simp [hFalse]
              rw [hIfMul]
              ring

theorem oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealMaskedScore
    {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (idx : TileIndex [M, D]) :
    oScoreBlockPartial visible score V numKVBlocks idx /
        lScoreBlockPartial visible score numKVBlocks idx.1 =
      attentionRealMaskedScore visible score V idx := by
  rw [oScoreBlockPartial_eq_mShifted visible score V numKVBlocks (le_refl _) idx]
  rw [lScoreBlockPartial_eq_mShifted visible score numKVBlocks (le_refl _) idx.1]
  rw [mul_div_mul_left _ _ (Real.exp_ne_zero _)]
  rw [oScoreBlockFree_final_eq_oFreeScore, lScoreBlockFree_final_eq_lFreeScore]
  rw [oFreeScore_div_lFreeScore_eq_attentionRealMaskedScore]

theorem oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealAlibi
    {M D Bk numKVBlocks : Nat}
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oScoreBlockPartial allVisible (alibiScore qStart slope Q K scale) V
        numKVBlocks idx /
      lScoreBlockPartial allVisible (alibiScore qStart slope Q K scale)
        numKVBlocks idx.1 =
      attentionRealAlibi qStart slope Q K V scale idx := by
  rw [oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealMaskedScore]
  rfl

theorem oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealSlidingWindow
    {M D Bk numKVBlocks : Nat}
    (qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oScoreBlockPartial (slidingVisible window qStart) (dotScore Q K scale) V
        numKVBlocks idx /
      lScoreBlockPartial (slidingVisible window qStart) (dotScore Q K scale)
        numKVBlocks idx.1 =
      attentionRealSlidingWindow qStart window Q K V scale idx := by
  rw [oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealMaskedScore]
  rfl

theorem oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealSoftcap
    {M D Bk numKVBlocks : Nat}
    (softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oScoreBlockPartial allVisible (softcapDotScore softcap Q K scale) V
        numKVBlocks idx /
      lScoreBlockPartial allVisible (softcapDotScore softcap Q K scale)
        numKVBlocks idx.1 =
      attentionRealSoftcap softcap Q K V scale idx := by
  rw [oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealMaskedScore]
  rfl

theorem oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealAlibiSlidingSoftcap
    {M D Bk numKVBlocks : Nat}
    (qStart window : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oScoreBlockPartial (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        V numKVBlocks idx /
      lScoreBlockPartial (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        numKVBlocks idx.1 =
      attentionRealAlibiSlidingSoftcap qStart window slope softcap Q K V scale idx := by
  rw [oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealMaskedScore]
  rfl

theorem score_block_mNew_tile_eq {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) :
    Tile.bop max (Broadcast.consSame Broadcast.nil)
      (⟨fun idx : TileIndex [M] =>
        mScoreBlockPartial visible score k idx.1⟩ : Tile .real [M])
      (⟨fun idx : TileIndex [M] =>
        (Finset.univ : Finset (Fin Bk)).sup
          (fun j => scoreLane visible score idx.1
            (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j))⟩ :
        Tile .real [M])
      =
      ⟨fun idx : TileIndex [M] =>
        mScoreBlockPartial visible score (k + 1) idx.1⟩ := by
  ext idx
  simp [Tile.bop]
  rw [mScoreBlockPartial_succ_of_lt visible score k hk idx.1]

theorem score_block_alpha_tile_eq {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) :
    Tile.uop WithBot.realExp
      (Tile.bop WithBot.realSub (Broadcast.consSame Broadcast.nil)
        (⟨fun idx : TileIndex [M] =>
          mScoreBlockPartial visible score k idx.1⟩ : Tile .real [M])
        (Tile.bop max (Broadcast.consSame Broadcast.nil)
          (⟨fun idx : TileIndex [M] =>
            mScoreBlockPartial visible score k idx.1⟩ : Tile .real [M])
          (⟨fun idx : TileIndex [M] =>
            (Finset.univ : Finset (Fin Bk)).sup
              (fun j => scoreLane visible score idx.1
                (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j))⟩ :
            Tile .real [M])))
      =
      Tile.ofReal (fun idx : TileIndex [M] =>
        alphaScoreBlockPartial visible score k idx.1) := by
  ext idx
  have hmTile := score_block_mNew_tile_eq visible score k hk
  have hm := congrArg (fun t : Tile .real [M] => t.data idx) hmTile
  simp [Tile.bop] at hm
  simp [Tile.uop, Tile.bop, Tile.ofReal, alphaScoreBlockPartial]
  rw [hm]
  exact FA1MathBoundary.realExp_eq_some_unbotD _

theorem score_block_p_tile_eq {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) :
    Tile.uop WithBot.realExp
      (Tile.bop WithBot.realSub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (scoreBlockLane visible score k (Nat.succ_le_iff.mpr hk))
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.bop max (Broadcast.consSame Broadcast.nil)
            (⟨fun idx : TileIndex [M] =>
              mScoreBlockPartial visible score k idx.1⟩ : Tile .real [M])
            (⟨fun idx : TileIndex [M] =>
              (Finset.univ : Finset (Fin Bk)).sup
                (fun j => scoreLane visible score idx.1
                  (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j))⟩ :
              Tile .real [M]))))
      =
      Tile.ofReal (fun idx : TileIndex [M, Bk] =>
        (WithBot.realExp
          (WithBot.realSub
            (scoreLane visible score idx.1
              (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1))
            (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0) := by
  ext idx
  have hmTile := score_block_mNew_tile_eq visible score k hk
  have hm := congrArg (fun t : Tile .real [M] => t.data (idx.1, PUnit.unit)) hmTile
  simp [Tile.bop] at hm
  simp [Tile.uop, Tile.bop, Tile.expandDim, Tile.ofReal,
    TileShape.dropInsertedIndex]
  rw [hm]
  exact FA1MathBoundary.realExp_eq_some_unbotD _

theorem score_block_mBlock_reduceMax_eq {M Bk numKVBlocks : Nat}
    (hBk : 0 < Bk)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) :
    Tile.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false
      (scoreBlockLane visible score k (Nat.succ_le_iff.mpr hk))
      =
      some
        (⟨fun idx : TileIndex [M] =>
          (Finset.univ : Finset (Fin Bk)).sup
            (fun j => scoreLane visible score idx.1
              (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j))⟩ :
          Tile .real [M]) := by
  unfold Tile.reduceMax
  simp [Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, hBk, scoreBlockLane]
  funext idx
  rw [Finset.sup'_eq_sup]
  apply Finset.sup_congr rfl
  intro j _
  simp [scoreBlockIndex]

theorem score_block_mBlock_sup'_tile_eq {M Bk numKVBlocks : Nat}
    (hU : (Finset.univ : Finset (Fin Bk)).Nonempty)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) :
    (⟨fun outIdx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup' hU
        (fun j =>
          (scoreBlockLane visible score k
            (Nat.succ_le_iff.mpr hk)).data (outIdx.1, j, outIdx.2))⟩ :
      Tile .real [M])
      =
      ⟨fun idx : TileIndex [M] =>
        (Finset.univ : Finset (Fin Bk)).sup
          (fun j => scoreLane visible score idx.1
            (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j))⟩ := by
  ext idx
  change
    (Finset.univ : Finset (Fin Bk)).sup' hU
      (fun j =>
        (scoreBlockLane visible score k
          (Nat.succ_le_iff.mpr hk)).data (idx.1, j, idx.2))
      =
      (Finset.univ : Finset (Fin Bk)).sup
        (fun j => scoreLane visible score idx.1
          (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j))
  rw [Finset.sup'_eq_sup]
  apply Finset.sup_congr rfl
  intro j _
  simp [scoreBlockLane, scoreBlockIndex]

theorem score_block_p_rowSum_eq {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) :
    Tile.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false
      (Tile.ofReal fun idx : TileIndex [M, Bk] =>
        (WithBot.realExp
          (WithBot.realSub
            (scoreLane visible score idx.1
              (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1))
            (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0)
      =
      Tile.ofReal (fun idx : TileIndex [M] =>
        Finset.univ.sum (fun j : Fin Bk =>
          (WithBot.realExp
            (WithBot.realSub
              (scoreLane visible score idx.1
                (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j))
              (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0)) := by
  ext idx
  simp [Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
    TileShape.eraseAxis, TileShape.insertAxisIndex, Tile.ofReal]
  rfl

theorem score_block_pv_dot_eq {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (hk : k < numKVBlocks) :
    Tile.dot [] (M := M) (K := Bk) (N := D)
      (Tile.ofReal fun idx : TileIndex [M, Bk] =>
        (WithBot.realExp
          (WithBot.realSub
            (scoreLane visible score idx.1
              (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1))
            (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0)
      (valueBlock V k (Nat.succ_le_iff.mpr hk))
      =
      Tile.ofReal (fun idx : TileIndex [M, D] =>
        Finset.univ.sum (fun jLocal : Fin Bk =>
          let j := scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal
          (WithBot.realExp
            (WithBot.realSub
              (scoreLane visible score idx.1 j)
              (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0 *
            V (j, idx.2.1, PUnit.unit))) := by
  ext idx
  rcases idx with ⟨i, d, u⟩
  cases u
  simp [Tile.dot, Tile.ofReal, valueBlock]

theorem rawScoreBlockTile_eq {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk) =
      scoreBlockLane allVisible (dotScore Q K scale) k
        (Nat.succ_le_iff.mpr hk) := by
  ext idx
  obtain ⟨i, j, u⟩ := idx
  cases u
  simp only [rawScoreBlockTile, Tile.bop, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.mul]
  unfold scoreBlockIndex
  rw [StreamingAccumulator.block_scaled_data_eq Q K scale k hk i j]
  simp [scoreBlockLane, scoreLane, dotScore, allVisible, scoreBlockIndex]

theorem score_block_lNew_tile_eq {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) :
    Tile.bop WithBot.realAdd (Broadcast.consSame Broadcast.nil)
      (Tile.bop WithBot.realMul (Broadcast.consSame Broadcast.nil)
        (Tile.ofReal fun idx : TileIndex [M] =>
          alphaScoreBlockPartial visible score k idx.1)
        (Tile.ofReal fun idx : TileIndex [M] =>
          lScoreBlockPartial visible score k idx.1))
      (Tile.ofReal fun idx : TileIndex [M] =>
        Finset.univ.sum (fun j : Fin Bk =>
          (WithBot.realExp
            (WithBot.realSub
              (scoreLane visible score idx.1
                (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j))
              (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0))
      =
      Tile.ofReal (fun idx : TileIndex [M] =>
        lScoreBlockPartial visible score (k + 1) idx.1) := by
  ext idx
  simp [Tile.bop, Tile.ofReal]
  rw [lScoreBlockPartial_succ_of_lt visible score k hk idx.1]
  simp [WithBot.realSub]

theorem score_block_oAcc_tile_eq {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (hk : k < numKVBlocks) :
    Tile.bop WithBot.realAdd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.bop WithBot.realMul (Broadcast.consSame (Broadcast.consL Broadcast.nil))
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.ofReal fun idx : TileIndex [M] =>
            alphaScoreBlockPartial visible score k idx.1))
        (Tile.ofReal fun idx : TileIndex [M, D] =>
          oScoreBlockPartial visible score V k idx))
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        Finset.univ.sum (fun jLocal : Fin Bk =>
          let j := scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal
          (WithBot.realExp
            (WithBot.realSub
              (scoreLane visible score idx.1 j)
              (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0 *
            V (j, idx.2.1, PUnit.unit)))
      =
      Tile.ofReal (fun idx : TileIndex [M, D] =>
        oScoreBlockPartial visible score V (k + 1) idx) := by
  ext idx
  rcases idx with ⟨i, d, u⟩
  cases u
  simp [Tile.bop, Tile.expandDim, Tile.ofReal, TileShape.dropInsertedIndex]
  rw [oScoreBlockPartial_succ_of_lt visible score V k hk (i, d, PUnit.unit)]
  simp [WithBot.realSub]

theorem softcapScoreBlock_tile_eq {M D Bk numKVBlocks : Nat}
    (softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    Tile.bop WithBot.realMul Broadcast.scalarL
      (Tile.scalar ((softcap : ℝ) : WithBot ℝ))
      (Tile.uop WithBot.realTanh
        (Tile.bop WithBot.realDiv Broadcast.scalarR
          (Tile.bop NumericDType.real.mul Broadcast.scalarR
            (Tile.dot [] (Tile.ofReal Q)
              (Tile.transpose [] (Tile.ofReal
                (fun idx : TileIndex [Bk, D] =>
                  K (scoreBlockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) idx.1, idx.2.1, PUnit.unit)))))
            (Tile.scalar ((scale : ℝ) : WithBot ℝ)))
          (Tile.scalar ((softcap : ℝ) : WithBot ℝ))))
      =
      scoreBlockLane allVisible (softcapDotScore softcap Q K scale) k
        (Nat.succ_le_iff.mpr hk) := by
  ext idx
  obtain ⟨i, j, u⟩ := idx
  cases u
  simp only [Tile.bop, Tile.scalar, Tile.uop, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.mul]
  unfold scoreBlockIndex
  rw [StreamingAccumulator.block_scaled_data_eq Q K scale k hk i j]
  exact softcapScore_lane_eq softcap Q K scale i
    (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j)

theorem softcapScoreBlock_numeric_tile_eq {M D Bk numKVBlocks : Nat}
    (softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    Tile.bop NumericDType.real.mul Broadcast.scalarL
      (Tile.scalar ((softcap : ℝ) : WithBot ℝ))
      (Tile.uop WithBot.realTanh
        (Tile.bop NumericDType.real.div Broadcast.scalarR
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
          (Tile.scalar ((softcap : ℝ) : WithBot ℝ))))
      =
      scoreBlockLane allVisible (softcapDotScore softcap Q K scale) k
        (Nat.succ_le_iff.mpr hk) := by
  unfold rawScoreBlockTile
  exact softcapScoreBlock_tile_eq softcap Q K scale k hk

theorem distanceBlock_tile_eq {M Bk numKVBlocks : Nat}
    (qStart : Nat) (k : Nat) (hk : k < numKVBlocks) :
    Tile.bop max (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.bop WithBot.realSub (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Tile.natToReal
          (Tile.expandDim ⟨0, by simp⟩
            (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
        (Tile.natToReal
          (Tile.expandDim ⟨1, by simp⟩
            (Tile.vec (fun i : Fin M => qStart + i.val)))))
      (Tile.bop WithBot.realSub Broadcast.scalarL
        (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
        (Tile.bop WithBot.realSub (Broadcast.consL (Broadcast.consR Broadcast.nil))
          (Tile.natToReal
            (Tile.expandDim ⟨0, by simp⟩
              (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
          (Tile.natToReal
            (Tile.expandDim ⟨1, by simp⟩
              (Tile.vec (fun i : Fin M => qStart + i.val))))))
      =
      Tile.ofReal (fun idx : TileIndex [M, Bk] =>
        (natAbsDiff (qStart + idx.1.val)
          (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1).val :
            Nat)) := by
  ext idx
  obtain ⟨i, j, u⟩ := idx
  cases u
  simp only [Tile.bop, Tile.scalar, Tile.natToReal, Tile.expandDim_data,
    Tile.vec_data, Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex, Tile.ofReal_data]
  unfold scoreBlockIndex
  change (((max (((k * Bk + j.val : Nat) : ℝ) - ((qStart + i.val : Nat) : ℝ))
        (0 - (((k * Bk + j.val : Nat) : ℝ) - ((qStart + i.val : Nat) : ℝ))) : ℝ) :
        WithBot ℝ) =
      (((natAbsDiff (qStart + i.val) (k * Bk + j.val) : Nat) : ℝ) : WithBot ℝ))
  rw [real_max_delta_eq_natAbsDiff]

theorem slidingVisibleBlock_tile_eq {M Bk numKVBlocks : Nat}
    (qStart window : Nat) (k : Nat) (hk : k < numKVBlocks) :
    Tile.cop ComparableDType.real.lt Broadcast.scalarR
      (Tile.bop max (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Tile.bop WithBot.realSub (Broadcast.consL (Broadcast.consR Broadcast.nil))
          (Tile.natToReal
            (Tile.expandDim ⟨0, by simp⟩
              (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
          (Tile.natToReal
            (Tile.expandDim ⟨1, by simp⟩
              (Tile.vec (fun i : Fin M => qStart + i.val)))))
        (Tile.bop WithBot.realSub Broadcast.scalarL
          (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
          (Tile.bop WithBot.realSub (Broadcast.consL (Broadcast.consR Broadcast.nil))
            (Tile.natToReal
              (Tile.expandDim ⟨0, by simp⟩
                (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
            (Tile.natToReal
              (Tile.expandDim ⟨1, by simp⟩
                (Tile.vec (fun i : Fin M => qStart + i.val)))))))
      (Tile.scalar (((window : ℝ) : WithBot ℝ)))
      =
      visibleBlock (slidingVisible window qStart) k (Nat.succ_le_iff.mpr hk) := by
  rw [distanceBlock_tile_eq qStart k hk]
  ext idx
  obtain ⟨i, j, u⟩ := idx
  cases u
  simp only [Tile.cop, Tile.scalar, Tile.ofReal_data, Broadcast.leftIndex,
    Broadcast.rightIndex, ComparableDType.lt]
  unfold visibleBlock slidingVisible scoreBlockIndex
  simp [StreamingAccumulator.blockIndex]
  constructor
  · intro hlt
    change ((((natAbsDiff (qStart + i.val) (k * Bk + j.val) : Nat) : ℝ) :
      WithBot ℝ) < (((window : Nat) : ℝ) : WithBot ℝ)) at hlt
    rw [WithBot.coe_lt_coe] at hlt
    exact_mod_cast hlt
  · intro hlt
    change ((((natAbsDiff (qStart + i.val) (k * Bk + j.val) : Nat) : ℝ) :
      WithBot ℝ) < (((window : Nat) : ℝ) : WithBot ℝ))
    rw [WithBot.coe_lt_coe]
    exact_mod_cast hlt

theorem alibiScoreBlock_tile_eq {M D Bk numKVBlocks : Nat}
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    Tile.bop WithBot.realAdd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.bop NumericDType.real.mul Broadcast.scalarR
        (Tile.dot [] (Tile.ofReal Q)
          (Tile.transpose [] (Tile.ofReal
            (fun idx : TileIndex [Bk, D] =>
              K (scoreBlockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) idx.1, idx.2.1, PUnit.unit)))))
        (Tile.scalar ((scale : ℝ) : WithBot ℝ)))
      (Tile.bop WithBot.realSub Broadcast.scalarL
        (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
        (Tile.bop WithBot.realMul Broadcast.scalarL
          (Tile.scalar ((slope : ℝ) : WithBot ℝ))
          (Tile.bop max (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Tile.bop WithBot.realSub (Broadcast.consL (Broadcast.consR Broadcast.nil))
              (Tile.natToReal
                (Tile.expandDim ⟨0, by simp⟩
                  (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
              (Tile.natToReal
                (Tile.expandDim ⟨1, by simp⟩
                  (Tile.vec (fun i : Fin M => qStart + i.val)))))
            (Tile.bop WithBot.realSub Broadcast.scalarL
              (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
              (Tile.bop WithBot.realSub (Broadcast.consL (Broadcast.consR Broadcast.nil))
                (Tile.natToReal
                  (Tile.expandDim ⟨0, by simp⟩
                    (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
                (Tile.natToReal
                  (Tile.expandDim ⟨1, by simp⟩
                    (Tile.vec (fun i : Fin M => qStart + i.val)))))))))
      =
      scoreBlockLane allVisible (alibiScore qStart slope Q K scale) k
        (Nat.succ_le_iff.mpr hk) := by
  ext idx
  obtain ⟨i, j, u⟩ := idx
  cases u
  simp only [Tile.bop, Tile.scalar, Tile.natToReal, Tile.expandDim_data,
    Tile.vec_data, Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex, NumericDType.mul]
  unfold scoreBlockIndex
  rw [StreamingAccumulator.block_scaled_data_eq Q K scale k hk i j]
  have hdist :
      max (WithBot.realSub (some (((k * Bk + j.val : Nat) : ℝ)))
            (some (((qStart + i.val : Nat) : ℝ))))
          (WithBot.realSub (((0 : ℝ) : WithBot ℝ))
            (WithBot.realSub (some (((k * Bk + j.val : Nat) : ℝ)))
              (some (((qStart + i.val : Nat) : ℝ))))) =
        (((natAbsDiff (qStart + i.val) (k * Bk + j.val) : Nat) : ℝ) :
          WithBot ℝ) := by
    change (((max (((k * Bk + j.val : Nat) : ℝ) - ((qStart + i.val : Nat) : ℝ))
        (0 - (((k * Bk + j.val : Nat) : ℝ) - ((qStart + i.val : Nat) : ℝ))) : ℝ) :
        WithBot ℝ) =
      (((natAbsDiff (qStart + i.val) (k * Bk + j.val) : Nat) : ℝ) :
        WithBot ℝ))
    rw [real_max_delta_eq_natAbsDiff]
  rw [hdist]
  rw [alibiBiasSub_toWithBot slope (qStart + i.val) (k * Bk + j.val)]
  simpa [scoreBlockLane, scoreBlockIndex, StreamingAccumulator.blockIndex] using
    alibiScore_lane_eq qStart slope Q K scale i
      (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j)

theorem slidingScoreBlock_tile_eq {M D Bk numKVBlocks : Nat}
    (qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    Tile.select
      (Tile.cop ComparableDType.real.lt Broadcast.scalarR
        (Tile.bop max (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Tile.bop WithBot.realSub (Broadcast.consL (Broadcast.consR Broadcast.nil))
            (Tile.natToReal
              (Tile.expandDim ⟨0, by simp⟩
                (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
            (Tile.natToReal
              (Tile.expandDim ⟨1, by simp⟩
                (Tile.vec (fun i : Fin M => qStart + i.val)))))
          (Tile.bop WithBot.realSub Broadcast.scalarL
            (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
            (Tile.bop WithBot.realSub (Broadcast.consL (Broadcast.consR Broadcast.nil))
              (Tile.natToReal
                (Tile.expandDim ⟨0, by simp⟩
                  (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
              (Tile.natToReal
                (Tile.expandDim ⟨1, by simp⟩
                  (Tile.vec (fun i : Fin M => qStart + i.val)))))))
        (Tile.scalar (((window : ℝ) : WithBot ℝ))))
      (Tile.bop NumericDType.real.mul Broadcast.scalarR
        (Tile.dot [] (Tile.ofReal Q)
          (Tile.transpose [] (Tile.ofReal
            (fun idx : TileIndex [Bk, D] =>
              K (scoreBlockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) idx.1, idx.2.1, PUnit.unit)))))
        (Tile.scalar ((scale : ℝ) : WithBot ℝ)))
      (⟨fun _ : TileIndex [M, Bk] => (⊥ : WithBot ℝ)⟩ : Tile .real [M, Bk])
      =
      scoreBlockLane (slidingVisible window qStart) (dotScore Q K scale) k
        (Nat.succ_le_iff.mpr hk) := by
  rw [slidingVisibleBlock_tile_eq qStart window k hk]
  ext idx
  obtain ⟨i, j, u⟩ := idx
  cases u
  simp only [Tile.select_data, Tile.bop, Tile.scalar,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul]
  unfold scoreBlockIndex
  let jg := StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j
  by_cases hvis : slidingVisible window qStart i jg = Bool.true
  · simp [visibleBlock, scoreBlockIndex, jg, hvis]
    have hraw := StreamingAccumulator.block_scaled_data_eq Q K scale k hk i j
    simp only [WithBot.realMul] at hraw
    exact hraw.trans (by
      simp [scoreBlockLane, scoreBlockIndex, dotScore, hvis, jg] at ⊢
    )
  · simp [visibleBlock, scoreBlockIndex, jg, Bool.eq_false_iff.mpr hvis]
    simp [scoreBlockLane, scoreBlockIndex, Bool.eq_false_iff.mpr hvis, jg]

theorem slidingScoreBlock_numeric_tile_eq {M D Bk numKVBlocks : Nat}
    (qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    Tile.select
      (slidingVisibleBlockTile (M := M) (Bk := Bk) qStart window k)
      (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
      (⟨fun _ : TileIndex [M, Bk] => (⊥ : WithBot ℝ)⟩ : Tile .real [M, Bk])
      =
      scoreBlockLane (slidingVisible window qStart) (dotScore Q K scale) k
        (Nat.succ_le_iff.mpr hk) := by
  unfold slidingVisibleBlockTile rawScoreBlockTile distanceBlockTile
  exact slidingScoreBlock_tile_eq qStart window Q K scale k hk

theorem alibiScoreBlockTile_eq {M D Bk numKVBlocks : Nat}
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    alibiScoreBlockTile qStart slope Q K scale k (Nat.succ_le_iff.mpr hk) =
      scoreBlockLane allVisible (alibiScore qStart slope Q K scale) k
        (Nat.succ_le_iff.mpr hk) := by
  unfold alibiScoreBlockTile rawScoreBlockTile distanceBlockTile
  exact alibiScoreBlock_tile_eq qStart slope Q K scale k hk

theorem alibiScoreBlock_numeric_tile_eq {M D Bk numKVBlocks : Nat}
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    Tile.bop NumericDType.real.add
      (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
      (Tile.bop NumericDType.real.sub Broadcast.scalarL
        (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
        (Tile.bop NumericDType.real.mul Broadcast.scalarL
          (Tile.scalar ((slope : ℝ) : WithBot ℝ))
          (distanceBlockTile (M := M) (Bk := Bk) qStart k)))
      =
      scoreBlockLane allVisible (alibiScore qStart slope Q K scale) k
        (Nat.succ_le_iff.mpr hk) := by
  unfold rawScoreBlockTile distanceBlockTile
  exact alibiScoreBlock_tile_eq qStart slope Q K scale k hk

theorem alibiScoreBlock_numeric_eq_tile {M D Bk numKVBlocks : Nat}
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    Tile.bop NumericDType.real.add
      (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
      (Tile.bop NumericDType.real.sub Broadcast.scalarL
        (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
        (Tile.bop NumericDType.real.mul Broadcast.scalarL
          (Tile.scalar ((slope : ℝ) : WithBot ℝ))
          (distanceBlockTile (M := M) (Bk := Bk) qStart k)))
      =
      alibiScoreBlockTile qStart slope Q K scale k (Nat.succ_le_iff.mpr hk) := by
  rfl

theorem slidingVisibleBlockTile_eq {M Bk numKVBlocks : Nat}
    (qStart window : Nat) (k : Nat) (hk : k < numKVBlocks) :
    slidingVisibleBlockTile (M := M) (Bk := Bk) qStart window k =
      visibleBlock (numKVBlocks := numKVBlocks) (slidingVisible window qStart)
        k (Nat.succ_le_iff.mpr hk) := by
  unfold slidingVisibleBlockTile distanceBlockTile
  exact slidingVisibleBlock_tile_eq qStart window k hk

theorem softcapAlibiScoreBlockTile_eq {M D Bk numKVBlocks : Nat}
    (qStart : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    softcapScoreTile softcap
        (alibiScoreBlockTile qStart slope Q K scale k (Nat.succ_le_iff.mpr hk)) =
      scoreBlockLane allVisible
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        k (Nat.succ_le_iff.mpr hk) := by
  rw [alibiScoreBlockTile_eq qStart slope Q K scale k hk]
  ext idx
  obtain ⟨i, j, u⟩ := idx
  cases u
  simp only [softcapScoreTile, Tile.bop, Tile.scalar, Tile.uop,
    Broadcast.leftIndex, Broadcast.rightIndex, WithBot.realMul_coe_coe,
    WithBot.realDiv_coe_coe, WithBot.realTanh_coe, scoreBlockLane, scoreLane,
    allVisible, softcapScore, if_true]

theorem alibiSlidingSoftcapScoreBlockTile_eq {M D Bk numKVBlocks : Nat}
    (qStart window : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    maskedScoreTile
        (slidingVisibleBlockTile (M := M) (Bk := Bk) qStart window k)
        (softcapScoreTile softcap
          (alibiScoreBlockTile qStart slope Q K scale k (Nat.succ_le_iff.mpr hk))) =
      scoreBlockLane (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        k (Nat.succ_le_iff.mpr hk) := by
  rw [slidingVisibleBlockTile_eq (M := M) (Bk := Bk) (numKVBlocks := numKVBlocks)
      qStart window k hk,
    softcapAlibiScoreBlockTile_eq qStart slope softcap Q K scale k hk]
  ext idx
  obtain ⟨i, j, u⟩ := idx
  cases u
  let jg := scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j
  by_cases hvis : slidingVisible window qStart i jg = Bool.true
  · simp [maskedScoreTile, visibleBlock, scoreBlockLane, scoreLane, allVisible,
      jg, hvis]
  · simp [maskedScoreTile, visibleBlock, scoreBlockLane, scoreLane, allVisible, jg,
      Bool.eq_false_iff.mpr hvis]

theorem alibiSlidingSoftcapScoreBlock_numeric_tile_eq {M D Bk numKVBlocks : Nat}
    (qStart window : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    Tile.select
      (slidingVisibleBlockTile (M := M) (Bk := Bk) qStart window k)
      (Tile.bop NumericDType.real.mul Broadcast.scalarL
        (Tile.scalar ((softcap : ℝ) : WithBot ℝ))
        (Tile.uop WithBot.realTanh
          (Tile.bop NumericDType.real.div Broadcast.scalarR
            (Tile.bop NumericDType.real.add
              (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
              (Tile.bop NumericDType.real.sub Broadcast.scalarL
                (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
                (Tile.bop NumericDType.real.mul Broadcast.scalarL
                  (Tile.scalar ((slope : ℝ) : WithBot ℝ))
                  (distanceBlockTile (M := M) (Bk := Bk) qStart k))))
            (Tile.scalar ((softcap : ℝ) : WithBot ℝ)))))
      (⟨fun _ : TileIndex [M, Bk] => (⊥ : WithBot ℝ)⟩ : Tile .real [M, Bk])
      =
      scoreBlockLane (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        k (Nat.succ_le_iff.mpr hk) := by
  change maskedScoreTile
      (slidingVisibleBlockTile (M := M) (Bk := Bk) qStart window k)
      (softcapScoreTile softcap
        (alibiScoreBlockTile qStart slope Q K scale k (Nat.succ_le_iff.mpr hk))) =
      scoreBlockLane (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        k (Nat.succ_le_iff.mpr hk)
  exact alibiSlidingSoftcapScoreBlockTile_eq qStart window slope softcap Q K scale k hk

end FA1Score

end VeriTile.Examples
