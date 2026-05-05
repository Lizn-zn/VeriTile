/-
VeriTile.Examples.FlashAttention1.V0.Steps

Split-out support for FlashAttention-1 v0/full-tile proofs.
-/

import VeriTile.Examples.FlashAttention1.V0.PreLoop

namespace VeriTile.Examples

open VeriTile.Triton

set_option maxHeartbeats 800000 in
/-- One FA-1 KV-block iteration preserves the loop invariant. -/
theorem fa1_step
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (origPid k : Nat) (s : BlockState)
    (hk : k < numKVBlocks)
    (hP : P_fa1 qReg kReg vReg origPid Q K V scale k s) :
    ∃ s',
      stepStmts (fa1LoopBody kReg vReg M D Bk scale)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1 qReg kReg vReg origPid Q K V scale (k + 1) s' := by
  rcases hP with
    ⟨hpidReg, hpid, hoffs_m, hoffs_d, hq, hm, hl, ho, hQ, hK, hV⟩
  let offsN : Tile .nat [Bk] :=
    Tile.vec fun j : Fin Bk => k * Bk + j.val
  let ptrs : Tile .nat [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] => (k * Bk + idx.1.val) * D + idx.2.1.val⟩
  let kTile : Tile .real [Bk, D] :=
    Tile.ofReal fun idx : TileIndex [Bk, D] =>
      K (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.1,
        idx.2.1, PUnit.unit)
  let vTile : Tile .real [Bk, D] :=
    Tile.ofReal fun idx : TileIndex [Bk, D] =>
      V (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.1,
        idx.2.1, PUnit.unit)
  let scores : Tile .real [M, Bk] :=
    Tile.ofReal fun idx : TileIndex [M, Bk] =>
      FA1Math.scaledScore Q K scale idx.1
        (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1)
  let mBlock : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup
        (fun j => ((FA1Math.scaledScore Q K scale idx.1
          (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j)
          : ℝ) : WithBot ℝ))⟩
  let mNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1⟩
  let alpha : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      FA1Math.alphaPartial Q numKVBlocks K scale k idx.1
  let p : Tile .real [M, Bk] :=
    Tile.ofReal fun idx : TileIndex [M, Bk] =>
      Real.exp (FA1Math.scaledScore Q K scale idx.1
        (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0)
  let lNew : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      FA1Math.lPartial Q numKVBlocks K scale (k + 1) idx.1
  let oNew : Tile .real [M, D] :=
    Tile.ofReal fun idx : TileIndex [M, D] =>
      FA1Math.oPartial Q numKVBlocks K V scale (k + 1) idx
  let s0 := s.setReg "n" .nat [] (Tile.scalar k)
  let s1 := s0.setReg "offs_n" .nat [Bk] offsN
  let s2 := s1.setReg "k_ptrs" .nat [Bk, D] ptrs
  let s3 := s2.setReg "v_ptrs" .nat [Bk, D] ptrs
  let s4 := s3.setReg "k" .real [Bk, D] kTile
  let s5 := s4.setReg "v" .real [Bk, D] vTile
  let s6 := s5.setReg "scores" .real [M, Bk] scores
  let s7 := s6.setReg "m_block" .real [M] mBlock
  let s8 := s7.setReg "m_new" .real [M] mNew
  let s9 := s8.setReg "alpha" .real [M] alpha
  let s10 := s9.setReg "p" .real [M, Bk] p
  let s11 := s10.setReg "l_new" .real [M] lNew
  let s12 := s11.setReg "o_acc" .real [M, D] oNew
  let s13 := s12.setReg "m_i" .real [M] mNew
  let s' := s13.setReg "l_i" .real [M] lNew
  apply Exists.intro
  constructor
  · simp [fa1LoopBody, stepStmts, stepStmt, evalOp, Tile.bop, Tile.expandDim,
      Tile.transpose, Tile.dot, Tile.reduceMax, Tile.reduceMaxDrop,
      Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
      TileShape.eraseAxis, TileShape.insertAxisIndex, TileShape.dropInsertedIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, Option.bind, hBk, hoffs_d, hq, hm, hl, ho]
    have hKmem : ∀ (j : Fin Bk) (d : Fin D),
        s.readMem kReg ((k * Bk + j.val) * D + d.val) =
          K (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit) :=
      fa1_block_read kReg s K hK k hk
    have hVmem : ∀ (j : Fin Bk) (d : Fin D),
        s.readMem vReg ((k * Bk + j.val) * D + d.val) =
          V (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit) :=
      fa1_block_read vReg s V hV k hk
    simp_rw [hKmem, hVmem]
    rfl
  · have hmNewData : ∀ i : Fin M,
        max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
          (some ((Finset.univ : Finset (Fin Bk)).sup'
            (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
          FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i
      rw [FA1Math.mPartial_succ_of_lt Q numKVBlocks K scale k hk i]
      congr 1
      change ((((Finset.univ : Finset (Fin Bk)).sup'
          (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
          (fun j => (Finset.univ.sum (fun d : Fin D =>
            Q (i, d, PUnit.unit) *
              K (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) : ℝ) : WithBot ℝ) =
        (Finset.univ : Finset (Fin Bk)).sup (fun j =>
          ((FA1Math.scaledScore Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) : ℝ) : WithBot ℝ))
      rw [← WithBot.sup'_coe]
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      simp [FA1Math.scaledScore, mul_comm]
    have hmNewDataOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        (FA1Math.mPartial Bk Q numKVBlocks K scale k i) ⊔
          (some ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
          FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i H
      rw [FA1Math.mPartial_succ_of_lt Q numKVBlocks K scale k hk i]
      congr 1
      change ((((Finset.univ : Finset (Fin Bk)).sup' H
          (fun j => (Finset.univ.sum (fun d : Fin D =>
            Q (i, d, PUnit.unit) *
              K (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) : ℝ) : WithBot ℝ) =
        (Finset.univ : Finset (Fin Bk)).sup (fun j =>
          ((FA1Math.scaledScore Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) : ℝ) : WithBot ℝ))
      rw [← WithBot.sup'_coe]
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      simp [FA1Math.scaledScore, mul_comm]
    have hmNewDataComm : ∀ i : Fin M,
        max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
          (some ((Finset.univ : Finset (Fin Bk)).sup'
            (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit)))))) =
          FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i
      rw [← hmNewData i]
      congr 2
      apply Finset.sup'_congr (H := by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩) rfl
      intro j _
      rw [mul_comm]
    have hAlphaData : ∀ i : Fin M,
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
            (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i
      simpa [WithBot.realSub] using
        FA1Math.alphaPartial_toWithBot Q numKVBlocks K scale k i
    have hPData : ∀ (i : Fin M) (j : Fin Bk),
        WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i))
          =
          some (Real.exp (FA1Math.scaledScore Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) -
            (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j
      have hm : FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i ≠ ⊥ :=
        FA1Math.mPartial_succ_ne_bot hBk Q numKVBlocks K scale k
          (Nat.succ_le_iff.mpr hk) i
      obtain ⟨m, hm_eq⟩ := WithBot.ne_bot_iff_exists.mp hm
      rw [← hm_eq]
      simp [FA1Math.scaledScore, mul_comm]
    have hAlphaDataOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      change WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i)
      have hM :
          @max (Option ℝ) Option.instMax
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
            (some ((Finset.univ : Finset (Fin Bk)).sup' H
              (fun j =>
                (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
            FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
        have hOptionMax :
            @max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))
            =
            max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) := by
          cases FA1Math.mPartial Bk Q numKVBlocks K scale k i <;> rfl
        rw [hOptionMax]
        exact hmNewDataOf i H
      exact (congrArg
        (fun z => WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i) z)) hM).trans
        (hAlphaData i)
    have hPDataOf : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (@max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      change WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (Real.exp (FA1Math.scaledScore Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) -
            (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0))
      have hM :
          @max (Option ℝ) Option.instMax
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
            (some ((Finset.univ : Finset (Fin Bk)).sup' H
              (fun j =>
                (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
            FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
        have hOptionMax :
            @max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))
            =
            max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) := by
          cases FA1Math.mPartial Bk Q numKVBlocks K scale k i <;> rfl
        rw [hOptionMax]
        exact hmNewDataOf i H
      exact (congrArg
        (fun z => WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b) z)) hM).trans
          (hPData i j)
    have hAlphaDataComm : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      have hSup :
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))) =
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) := by
        apply Finset.sup'_congr (H := H) rfl
        intro j _
        rw [mul_comm]
      rw [hSup]
      exact hAlphaDataOf i H
    have hPDataComm : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            scale * (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (@max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      have hSup :
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))) =
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) := by
        apply Finset.sup'_congr (H := H) rfl
        intro j _
        rw [mul_comm]
      rw [hSup]
      simpa [mul_comm] using hPDataOf i j H
    have hAlphaDataMaxComm : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataComm i H
    have hPDataMaxComm : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            scale * (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (@max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataComm i j H
    have hAlphaDataMaxComm' : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataComm i H
    have hPDataMaxComm' : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            scale * (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataComm i j H
    have hAlphaDataMaxOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataOf i H
    have hPDataMaxOf : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataOf i j H
    have max_eq_sup : ∀ (a b : WithBot ℝ), max a b = a ⊔ b := by
      intro a b
      rfl
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [hpidReg]
    · simp [hpid]
    · simp [hoffs_m]
    · simp [hoffs_d]
    · simp [hq]
    · simp
      ext idx
      exact hmNewData idx.1
    · rw [← FA1Math.block_lNew_tile_eq Q K scale k hk]
      simp
      ext idx
      simp only [Tile.bop, Tile.ofReal, Tile.uop,
        Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
      congr 1
      · -- LHS: Option.map (· * lPartial k) (WithBot.realExp (Option.map₂ - mPartial(k) (max ...)))
        -- RHS: Option.map₂ * (some alphaPartial) (some lPartial k)
        rw [show (Option.map₂ (· * ·) (some (FA1Math.alphaPartial Q numKVBlocks K scale k idx.1))
              (some (FA1Math.lPartial Q numKVBlocks K scale k idx.1)) :
              WithBot ℝ) =
            Option.map (· * FA1Math.lPartial Q numKVBlocks K scale k idx.1)
              (some (FA1Math.alphaPartial Q numKVBlocks K scale k idx.1)) from rfl]
        congr 1
        -- Goal: WithBot.realExp (Option.map₂ - mPartial(k) (max mPartial(k) (some sup'))) =
        --       some alphaPartial.
        -- hAlphaData idx.1 says it's WithBot.realExp (Option.map₂ - mPartial(k) mPartial(k+1)).
        -- We bridge via congrArg.
        refine Eq.trans ?_ (hAlphaData idx.1)
        congr 2
        exact hmNewData idx.1
      · -- LHS: ∑ x, WithBot.realExp (Option.map (... - max ...) ...)
        -- RHS: some (∑ j, Real.exp (...))
        -- Step 1: replace each summand by `some (Real.exp ...)` via the same
        -- congr-2 / hmNewData / hPData pattern used in the alpha branch.
        -- Step 2: collapse `∑ some _ = some (∑ _)` via `WithBot.sum_someTerm_eq_some`.
        refine Eq.trans
          (@Finset.sum_congr (Fin Bk) (WithBot ℝ) _ _ _ _ _ rfl
            (fun j _ => ?_))
          (WithBot.sum_someTerm_eq_some _ _)
        refine Eq.trans ?_ (hPData idx.1 j)
        congr 2
        exact hmNewData idx.1
    · rw [← FA1Math.block_oAcc_tile_eq Q K V scale k hk]
      simp
      ext idx
      simp only [Tile.bop, Tile.ofReal, Tile.uop, Tile.expandDim,
        Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
        Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
        TileShape.dropInsertedIndex]
      congr 1
      · -- LHS: Option.map (· * oPartial k (idx)) (WithBot.realExp (Option.map₂ - mPartial(k) (max ...)))
        -- RHS: Option.map₂ * (some alphaPartial) (some (oPartial k idx))
        rw [show (Option.map₂ (· * ·)
              (some (FA1Math.alphaPartial Q numKVBlocks K scale k idx.1))
              (some (FA1Math.oPartial Q numKVBlocks K V scale k
                (idx.1, idx.2.1, PUnit.unit))) :
              WithBot ℝ) =
            Option.map (· * FA1Math.oPartial Q numKVBlocks K V scale k
                (idx.1, idx.2.1, PUnit.unit))
              (some (FA1Math.alphaPartial Q numKVBlocks K scale k idx.1)) from rfl]
        congr 1
        refine Eq.trans ?_ (hAlphaData idx.1)
        congr 2
        exact hmNewData idx.1
      · -- LHS: ∑ x, Option.map (· * V (...)) (WithBot.realExp (Option.map (... - max ...)))
        -- RHS: some (∑ j, Real.exp (...) * V (...))
        refine Eq.trans
          (@Finset.sum_congr (Fin Bk) (WithBot ℝ) _ _ _ _ _ rfl
            (fun j _ => ?_))
          (WithBot.sum_someTerm_eq_some _ _)
        -- Goal at j: Option.map (· * V (...)) (WithBot.realExp (Option.map (...) (max ...)))
        --           = some (Real.exp (...) * V (...))
        -- Strategy: descend through Option.map (·*V) and WithBot.realExp via `congr 1`s
        -- (the `· * V` factor on the outside aligns with the `· * V` factor in the RHS),
        -- then use the same congr-2 + hmNewData + hPData pattern as the alpha branch.
        -- We can't use `congr 1` directly because LHS has `Option.map` head, RHS has
        -- `some` head. So we first reshape RHS to match: `some y = Option.map (·*V) (some r)`
        -- where `r * V = y`, holds by `rfl`.
        change Option.map _ _ =
            Option.map (fun a : ℝ => a *
              V (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, idx.2.1, PUnit.unit))
              (some (Real.exp (FA1Math.scaledScore Q K scale idx.1
                (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j) -
                  WithBot.unbotD 0
                    (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1))))
        congr 1
        refine Eq.trans ?_ (hPData idx.1 j)
        congr 2
        exact hmNewData idx.1
    · intro idx
      simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hQ idx
    · intro idx
      simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hK idx
    · intro idx
      simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hV idx

set_option maxHeartbeats 800000 in
/-- Strided variant of `fa1_step`. One iteration of the strided KV-block
loop preserves `P_fa1_strided`. Same streaming-math content as
`fa1_step`; the only operational differences are the `k_ptrs` / `v_ptrs`
trees (assembled from `*_base_off` plus `offs_n[:, None] * stride_*N +
offs_d[None, :] * stride_*D`) and the `*_base_off` / `pid_*` /
`s.pids 0/1/2` registers being threaded through unchanged. -/
theorem fa1_step_strided
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState)
    (hk : k < numKVBlocks)
    (hP : P_fa1_strided qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale k s) :
    ∃ s',
      stepStmts (fa1LoopBodyStrided kReg vReg M D Bk sKN sKD sVN sVD scale)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1_strided qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale (k + 1) s' := by
  rcases hP with
    ⟨hpids0, hpids1, hpids2,
     hpid_qb, hpid_h, hpid_b,
     hq_base, hk_base, hv_base, ho_base,
     hoffs_m, hoffs_d, hq, hm, hl, ho, hQ, hK, hV⟩
  let kBase : Nat := batch * sKB + headIdx * sKH
  let vBase : Nat := batch * sVB + headIdx * sVH
  let offsN : Tile .nat [Bk] :=
    Tile.vec fun j : Fin Bk => k * Bk + j.val
  let kPtrs : Tile .nat [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] =>
      kBase + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD⟩
  let vPtrs : Tile .nat [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] =>
      vBase + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD⟩
  let kTile : Tile .real [Bk, D] :=
    Tile.ofReal fun idx : TileIndex [Bk, D] =>
      K (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.1,
        idx.2.1, PUnit.unit)
  let vTile : Tile .real [Bk, D] :=
    Tile.ofReal fun idx : TileIndex [Bk, D] =>
      V (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.1,
        idx.2.1, PUnit.unit)
  let scores : Tile .real [M, Bk] :=
    Tile.ofReal fun idx : TileIndex [M, Bk] =>
      FA1Math.scaledScore Q K scale idx.1
        (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1)
  let mBlock : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup
        (fun j => ((FA1Math.scaledScore Q K scale idx.1
          (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j)
          : ℝ) : WithBot ℝ))⟩
  let mNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1⟩
  let alpha : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      FA1Math.alphaPartial Q numKVBlocks K scale k idx.1
  let p : Tile .real [M, Bk] :=
    Tile.ofReal fun idx : TileIndex [M, Bk] =>
      Real.exp (FA1Math.scaledScore Q K scale idx.1
        (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0)
  let lNew : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      FA1Math.lPartial Q numKVBlocks K scale (k + 1) idx.1
  let oNew : Tile .real [M, D] :=
    Tile.ofReal fun idx : TileIndex [M, D] =>
      FA1Math.oPartial Q numKVBlocks K V scale (k + 1) idx
  let s0 := s.setReg "n" .nat [] (Tile.scalar k)
  let s1 := s0.setReg "offs_n" .nat [Bk] offsN
  let s2 := s1.setReg "k_ptrs" .nat [Bk, D] kPtrs
  let s3 := s2.setReg "v_ptrs" .nat [Bk, D] vPtrs
  let s4 := s3.setReg "k" .real [Bk, D] kTile
  let s5 := s4.setReg "v" .real [Bk, D] vTile
  let s6 := s5.setReg "scores" .real [M, Bk] scores
  let s7 := s6.setReg "m_block" .real [M] mBlock
  let s8 := s7.setReg "m_new" .real [M] mNew
  let s9 := s8.setReg "alpha" .real [M] alpha
  let s10 := s9.setReg "p" .real [M, Bk] p
  let s11 := s10.setReg "l_new" .real [M] lNew
  let s12 := s11.setReg "o_acc" .real [M, D] oNew
  let s13 := s12.setReg "m_i" .real [M] mNew
  let s' := s13.setReg "l_i" .real [M] lNew
  apply Exists.intro
  constructor
  · simp [fa1LoopBodyStrided, stepStmts, stepStmt, evalOp, Tile.bop, Tile.expandDim,
      Tile.transpose, Tile.dot, Tile.reduceMax, Tile.reduceMaxDrop,
      Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
      TileShape.eraseAxis, TileShape.insertAxisIndex, TileShape.dropInsertedIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, Option.bind, hBk, hoffs_d, hq, hm, hl, ho,
      hk_base, hv_base]
    have hKmem : ∀ (j : Fin Bk) (d : Fin D),
        s.readMem kReg (batch * sKB + headIdx * sKH
            + (k * Bk + j.val) * sKN + d.val * sKD) =
          K (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit) :=
      fa1_block_read_strided kReg s
        (batch * sKB + headIdx * sKH) sKN sKD K hK k hk
    have hVmem : ∀ (j : Fin Bk) (d : Fin D),
        s.readMem vReg (batch * sVB + headIdx * sVH
            + (k * Bk + j.val) * sVN + d.val * sVD) =
          V (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit) :=
      fa1_block_read_strided vReg s
        (batch * sVB + headIdx * sVH) sVN sVD V hV k hk
    simp_rw [hKmem, hVmem]
    rfl
  · have hmNewData : ∀ i : Fin M,
        max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
          (some ((Finset.univ : Finset (Fin Bk)).sup'
            (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
          FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i
      rw [FA1Math.mPartial_succ_of_lt Q numKVBlocks K scale k hk i]
      congr 1
      change ((((Finset.univ : Finset (Fin Bk)).sup'
          (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
          (fun j => (Finset.univ.sum (fun d : Fin D =>
            Q (i, d, PUnit.unit) *
              K (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) : ℝ) : WithBot ℝ) =
        (Finset.univ : Finset (Fin Bk)).sup (fun j =>
          ((FA1Math.scaledScore Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) : ℝ) : WithBot ℝ))
      rw [← WithBot.sup'_coe]
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      simp [FA1Math.scaledScore, mul_comm]
    have hmNewDataOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        (FA1Math.mPartial Bk Q numKVBlocks K scale k i) ⊔
          (some ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
          FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i H
      rw [FA1Math.mPartial_succ_of_lt Q numKVBlocks K scale k hk i]
      congr 1
      change ((((Finset.univ : Finset (Fin Bk)).sup' H
          (fun j => (Finset.univ.sum (fun d : Fin D =>
            Q (i, d, PUnit.unit) *
              K (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) : ℝ) : WithBot ℝ) =
        (Finset.univ : Finset (Fin Bk)).sup (fun j =>
          ((FA1Math.scaledScore Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) : ℝ) : WithBot ℝ))
      rw [← WithBot.sup'_coe]
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      simp [FA1Math.scaledScore, mul_comm]
    have hmNewDataComm : ∀ i : Fin M,
        max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
          (some ((Finset.univ : Finset (Fin Bk)).sup'
            (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit)))))) =
          FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i
      rw [← hmNewData i]
      congr 2
      apply Finset.sup'_congr (H := by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩) rfl
      intro j _
      rw [mul_comm]
    have hAlphaData : ∀ i : Fin M,
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
            (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i
      simpa [WithBot.realSub] using
        FA1Math.alphaPartial_toWithBot Q numKVBlocks K scale k i
    have hPData : ∀ (i : Fin M) (j : Fin Bk),
        WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i))
          =
          some (Real.exp (FA1Math.scaledScore Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) -
            (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j
      have hm : FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i ≠ ⊥ :=
        FA1Math.mPartial_succ_ne_bot hBk Q numKVBlocks K scale k
          (Nat.succ_le_iff.mpr hk) i
      obtain ⟨m, hm_eq⟩ := WithBot.ne_bot_iff_exists.mp hm
      rw [← hm_eq]
      simp [FA1Math.scaledScore, mul_comm]
    have hAlphaDataOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      change WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i)
      have hM :
          @max (Option ℝ) Option.instMax
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
            (some ((Finset.univ : Finset (Fin Bk)).sup' H
              (fun j =>
                (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
            FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
        have hOptionMax :
            @max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))
            =
            max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) := by
          cases FA1Math.mPartial Bk Q numKVBlocks K scale k i <;> rfl
        rw [hOptionMax]
        exact hmNewDataOf i H
      exact (congrArg
        (fun z => WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i) z)) hM).trans
        (hAlphaData i)
    have hPDataOf : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (@max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      change WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (Real.exp (FA1Math.scaledScore Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) -
            (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0))
      have hM :
          @max (Option ℝ) Option.instMax
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
            (some ((Finset.univ : Finset (Fin Bk)).sup' H
              (fun j =>
                (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
            FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
        have hOptionMax :
            @max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))
            =
            max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) := by
          cases FA1Math.mPartial Bk Q numKVBlocks K scale k i <;> rfl
        rw [hOptionMax]
        exact hmNewDataOf i H
      exact (congrArg
        (fun z => WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b) z)) hM).trans
          (hPData i j)
    have hAlphaDataComm : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      have hSup :
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))) =
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) := by
        apply Finset.sup'_congr (H := H) rfl
        intro j _
        rw [mul_comm]
      rw [hSup]
      exact hAlphaDataOf i H
    have hPDataComm : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            scale * (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (@max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      have hSup :
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))) =
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) := by
        apply Finset.sup'_congr (H := H) rfl
        intro j _
        rw [mul_comm]
      rw [hSup]
      simpa [mul_comm] using hPDataOf i j H
    have hAlphaDataMaxComm : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataComm i H
    have hPDataMaxComm : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            scale * (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (@max (Option ℝ) Option.instMax
              (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataComm i j H
    have hAlphaDataMaxComm' : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataComm i H
    have hPDataMaxComm' : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            scale * (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataComm i j H
    have hAlphaDataMaxOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (FA1Math.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataOf i H
    have hPDataMaxOf : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (max (FA1Math.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
        =
        some (Real.exp (FA1Math.scaledScore Q K scale i
          (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataOf i j H
    have max_eq_sup : ∀ (a b : WithBot ℝ), max a b = a ⊔ b := by
      intro a b
      rfl
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [hpids0]
    · simp [hpids1]
    · simp [hpids2]
    · simp [hpid_qb]
    · simp [hpid_h]
    · simp [hpid_b]
    · simp [hq_base]
    · simp [hk_base]
    · simp [hv_base]
    · simp [ho_base]
    · simp [hoffs_m]
    · simp [hoffs_d]
    · simp [hq]
    · simp
      ext idx
      exact hmNewData idx.1
    · rw [← FA1Math.block_lNew_tile_eq Q K scale k hk]
      simp
      ext idx
      simp only [Tile.bop, Tile.ofReal, Tile.uop,
        Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
      congr 1
      · rw [show (Option.map₂ (· * ·) (some (FA1Math.alphaPartial Q numKVBlocks K scale k idx.1))
              (some (FA1Math.lPartial Q numKVBlocks K scale k idx.1)) :
              WithBot ℝ) =
            Option.map (· * FA1Math.lPartial Q numKVBlocks K scale k idx.1)
              (some (FA1Math.alphaPartial Q numKVBlocks K scale k idx.1)) from rfl]
        congr 1
        refine Eq.trans ?_ (hAlphaData idx.1)
        congr 2
        exact hmNewData idx.1
      · refine Eq.trans
          (@Finset.sum_congr (Fin Bk) (WithBot ℝ) _ _ _ _ _ rfl
            (fun j _ => ?_))
          (WithBot.sum_someTerm_eq_some _ _)
        refine Eq.trans ?_ (hPData idx.1 j)
        congr 2
        exact hmNewData idx.1
    · rw [← FA1Math.block_oAcc_tile_eq Q K V scale k hk]
      simp
      ext idx
      simp only [Tile.bop, Tile.ofReal, Tile.uop, Tile.expandDim,
        Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
        Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
        TileShape.dropInsertedIndex]
      congr 1
      · rw [show (Option.map₂ (· * ·)
              (some (FA1Math.alphaPartial Q numKVBlocks K scale k idx.1))
              (some (FA1Math.oPartial Q numKVBlocks K V scale k
                (idx.1, idx.2.1, PUnit.unit))) :
              WithBot ℝ) =
            Option.map (· * FA1Math.oPartial Q numKVBlocks K V scale k
                (idx.1, idx.2.1, PUnit.unit))
              (some (FA1Math.alphaPartial Q numKVBlocks K scale k idx.1)) from rfl]
        congr 1
        refine Eq.trans ?_ (hAlphaData idx.1)
        congr 2
        exact hmNewData idx.1
      · refine Eq.trans
          (@Finset.sum_congr (Fin Bk) (WithBot ℝ) _ _ _ _ _ rfl
            (fun j _ => ?_))
          (WithBot.sum_someTerm_eq_some _ _)
        change Option.map _ _ =
            Option.map (fun a : ℝ => a *
              V (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, idx.2.1, PUnit.unit))
              (some (Real.exp (FA1Math.scaledScore Q K scale idx.1
                (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j) -
                  WithBot.unbotD 0
                    (FA1Math.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1))))
        congr 1
        refine Eq.trans ?_ (hPData idx.1 j)
        congr 2
        exact hmNewData idx.1
    · intro idx
      simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hQ idx
    · intro idx
      simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hK idx
    · intro idx
      simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hV idx

set_option maxHeartbeats 800000 in
/-- Causal strided loop step. This is the causal analogue of
`fa1_step_strided`: the operational body additionally constructs the
causal mask and `tl.where`-masked scores before the same online-softmax
update. -/
theorem fa1_step_strided_causal
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState)
    (hk : k < numKVBlocks)
    (hP : P_fa1_strided_causal qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale k s) :
    ∃ s',
      stepStmts (fa1LoopBodyStridedCausal kReg vReg M D Bk sKN sKD sVN sVD scale)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1_strided_causal qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale (k + 1) s' := by
  rcases hP with
    ⟨hpids0, hpids1, hpids2,
     hpid_qb, hpid_h, hpid_b,
     hq_base, hk_base, hv_base, ho_base,
     hoffs_m, hoffs_d, hq, hm, hl, ho, hQ, hK, hV⟩
  let kBase : Nat := batch * sKB + headIdx * sKH
  let vBase : Nat := batch * sVB + headIdx * sVH
  let offsN : Tile .nat [Bk] :=
    Tile.vec fun j : Fin Bk => k * Bk + j.val
  let kPtrs : Tile .nat [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] =>
      kBase + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD⟩
  let vPtrs : Tile .nat [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] =>
      vBase + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD⟩
  let kTile : Tile .real [Bk, D] :=
    Tile.ofReal fun idx : TileIndex [Bk, D] =>
      K (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.1,
        idx.2.1, PUnit.unit)
  let vTile : Tile .real [Bk, D] :=
    Tile.ofReal fun idx : TileIndex [Bk, D] =>
      V (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.1,
        idx.2.1, PUnit.unit)
  let scoresRaw : Tile .real [M, Bk] :=
    Tile.ofReal fun idx : TileIndex [M, Bk] =>
      FA1Math.scaledScore Q K scale idx.1
        (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1)
  let causal : Tile .bool [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      decide (k * Bk + idx.2.1.val ≤ qb * M + idx.1.val)⟩
  let scores : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
        (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1)⟩
  let mBlock : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup
        (fun jLocal =>
          FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
            (FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) jLocal))⟩
  let mNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) idx.1⟩
  let alpha : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      FA1MathCausal.alphaCausal (qb * M) Q numKVBlocks K scale k idx.1
  let p : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      WithBot.realExp
        (Option.map₂ (fun x y : ℝ => x - y)
          (FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
            (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1))
          (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) idx.1))⟩
  let lNew : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      FA1MathCausal.lPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) idx.1
  let oNew : Tile .real [M, D] :=
    Tile.ofReal fun idx : TileIndex [M, D] =>
      FA1MathCausal.oPartial Bk (qb * M) Q numKVBlocks K V scale (k + 1) idx
  let s0 := s.setReg "n" .nat [] (Tile.scalar k)
  let s1 := s0.setReg "offs_n" .nat [Bk] offsN
  let s2 := s1.setReg "k_ptrs" .nat [Bk, D] kPtrs
  let s3 := s2.setReg "v_ptrs" .nat [Bk, D] vPtrs
  let s4 := s3.setReg "k" .real [Bk, D] kTile
  let s5 := s4.setReg "v" .real [Bk, D] vTile
  let s6 := s5.setReg "scores_raw" .real [M, Bk] scoresRaw
  let s7 := s6.setReg "causal" .bool [M, Bk] causal
  let s8 := s7.setReg "scores" .real [M, Bk] scores
  let s9 := s8.setReg "m_block" .real [M] mBlock
  let s10 := s9.setReg "m_new" .real [M] mNew
  let s11 := s10.setReg "alpha" .real [M] alpha
  let s12 := s11.setReg "p" .real [M, Bk] p
  let s13 := s12.setReg "l_new" .real [M] lNew
  let s14 := s13.setReg "o_acc" .real [M, D] oNew
  let s15 := s14.setReg "m_i" .real [M] mNew
  let s' := s15.setReg "l_i" .real [M] lNew
  apply Exists.intro
  constructor
  · simp [fa1LoopBodyStridedCausal, stepStmts, stepStmt, evalOp,
      Tile.bop, Tile.cop, Tile.select, Tile.expandDim, Tile.transpose, Tile.dot,
      Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
      NumericDType.sub, ComparableDType.ge, Option.bind,
      hBk, hoffs_m, hoffs_d, hq, hm, hl, ho, hk_base, hv_base]
    have hKmem : ∀ (j : Fin Bk) (d : Fin D),
        s.readMem kReg (batch * sKB + headIdx * sKH
            + (k * Bk + j.val) * sKN + d.val * sKD) =
          K (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit) :=
      fa1_block_read_strided kReg s
        (batch * sKB + headIdx * sKH) sKN sKD K hK k hk
    have hVmem : ∀ (j : Fin Bk) (d : Fin D),
        s.readMem vReg (batch * sVB + headIdx * sVH
            + (k * Bk + j.val) * sVN + d.val * sVD) =
          V (FA1Math.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit) :=
      fa1_block_read_strided vReg s
        (batch * sVB + headIdx * sVH) sVN sVD V hV k hk
    simp_rw [hKmem, hVmem]
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [hpids0]
    · simp [hpids1]
    · simp [hpids2]
    · simp [hpid_qb]
    · simp [hpid_h]
    · simp [hpid_b]
    · simp [hq_base]
    · simp [hk_base]
    · simp [hv_base]
    · simp [ho_base]
    · simp [hoffs_m]
    · simp [hoffs_d]
    · simp [hq]
    · simp
      funext idx
      simp_rw [Finset.sup'_eq_sup]
      exact FA1MathCausal.mPartial_succ_kernelForm Q K scale qb k hk idx.1
    · rw [← FA1MathCausal.block_lNew_tile_eq (qb * M) Q K scale k hk]
      simp [Tile.bop, Tile.uop, Tile.ofReal,
        Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
      funext idx
      rw [show some
            (FA1MathCausal.alphaCausal (qb * M) Q numKVBlocks K scale k idx.1 *
                FA1MathCausal.lPartial Bk (qb * M) Q numKVBlocks K scale k idx.1 +
              ∑ x : Fin Bk,
                (WithBot.realExp
                  (Option.map₂ (fun x y : ℝ => x - y)
                    (FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
                      (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) x))
                    (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale
                      (k + 1) idx.1))).unbotD 0)
            =
            Option.map₂ (fun x y : ℝ => x + y)
              (some
                (FA1MathCausal.alphaCausal (qb * M) Q numKVBlocks K scale k idx.1 *
                  FA1MathCausal.lPartial Bk (qb * M) Q numKVBlocks K scale k idx.1))
              (some
                (∑ x : Fin Bk,
                  (WithBot.realExp
                    (Option.map₂ (fun x y : ℝ => x - y)
                      (FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
                        (FA1Math.blockIndex Bk numKVBlocks k
                          (Nat.succ_le_iff.mpr hk) x))
                      (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale
                        (k + 1) idx.1))).unbotD 0)) from rfl]
      congr 1
      · change Option.map _ _ =
            Option.map
              (fun a : ℝ =>
                a * FA1MathCausal.lPartial Bk (qb * M) Q numKVBlocks K scale k idx.1)
              (some (FA1MathCausal.alphaCausal (qb * M) Q numKVBlocks K scale k idx.1))
        congr 1
        unfold FA1MathCausal.alphaCausal
        simp_rw [Finset.sup'_eq_sup]
        rw [← FA1MathCausal.realExp_eq_some_unbotD
          (Option.map₂ (fun x y : ℝ => x - y)
            (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale k idx.1)
            (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale
              (k + 1) idx.1))]
        apply congrArg WithBot.realExp
        congr 2
        exact FA1MathCausal.mPartial_succ_kernelForm Q K scale qb k hk idx.1
      · refine Eq.trans
          (@Finset.sum_congr (Fin Bk) (WithBot ℝ) _ _ _ _ _ rfl
            (fun j _ => ?_))
          (WithBot.sum_someTerm_eq_some _ _)
        simp_rw [Finset.sup'_eq_sup]
        have hscore :
            (if k * Bk + ↑j ≤ qb * M + ↑idx.1 then
              some
                ((∑ x_1 : Fin D,
                  Q (idx.1, x_1, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, x_1, PUnit.unit)) * scale)
            else none : WithBot ℝ) =
              FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
                (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j) := by
          by_cases h_mask : k * Bk + ↑j ≤ qb * M + ↑idx.1
          · rw [if_pos h_mask]
            rw [FA1MathCausal.maskedScore_of_le (qb * M) Q K scale idx.1 _
              (by simp [FA1Math.blockIndex]; exact h_mask)]
            unfold FA1Math.scaledScore
            ring_nf
            rfl
          · rw [if_neg h_mask]
            rw [FA1MathCausal.maskedScore_of_not_le (qb * M) Q K scale idx.1 _
              (by simp [FA1Math.blockIndex]; omega)]
            rfl
        rw [← FA1MathCausal.realExp_eq_some_unbotD
          (Option.map₂ (fun x y : ℝ => x - y)
            (FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
            (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j))
            (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale
              (k + 1) idx.1))]
        apply congrArg WithBot.realExp
        congr 2
        · unfold FA1Math.scaledScore
          ring_nf
          rfl
        · exact FA1MathCausal.mPartial_succ_kernelForm Q K scale qb k hk idx.1
    · rw [← FA1MathCausal.block_oAcc_tile_eq (qb * M) Q K V scale k hk]
      simp [Tile.bop, Tile.uop, Tile.ofReal, Tile.expandDim,
        Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
        Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
        TileShape.dropInsertedIndex]
      funext idx
      rw [show some
            (FA1MathCausal.alphaCausal (qb * M) Q numKVBlocks K scale k idx.1 *
                FA1MathCausal.oPartial Bk (qb * M) Q numKVBlocks K V scale k
                  (idx.1, idx.2.1, PUnit.unit) +
              ∑ x : Fin Bk,
                (WithBot.realExp
                  (Option.map₂ (fun x y : ℝ => x - y)
                    (FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
                      (FA1Math.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) x))
                    (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale
                      (k + 1) idx.1))).unbotD 0 *
                  V (FA1Math.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) x, idx.2.1, PUnit.unit))
            =
            Option.map₂ (fun x y : ℝ => x + y)
              (some
                (FA1MathCausal.alphaCausal (qb * M) Q numKVBlocks K scale k idx.1 *
                  FA1MathCausal.oPartial Bk (qb * M) Q numKVBlocks K V scale k
                    (idx.1, idx.2.1, PUnit.unit)))
              (some
                (∑ x : Fin Bk,
                  (WithBot.realExp
                    (Option.map₂ (fun x y : ℝ => x - y)
                      (FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
                        (FA1Math.blockIndex Bk numKVBlocks k
                          (Nat.succ_le_iff.mpr hk) x))
                      (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale
                        (k + 1) idx.1))).unbotD 0 *
                    V (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) x, idx.2.1, PUnit.unit))) from rfl]
      congr 1
      · change Option.map _ _ =
            Option.map
              (fun a : ℝ =>
                a * FA1MathCausal.oPartial Bk (qb * M) Q numKVBlocks K V scale k
                  (idx.1, idx.2.1, PUnit.unit))
              (some (FA1MathCausal.alphaCausal (qb * M) Q numKVBlocks K scale k idx.1))
        congr 1
        unfold FA1MathCausal.alphaCausal
        simp_rw [Finset.sup'_eq_sup]
        rw [← FA1MathCausal.realExp_eq_some_unbotD
          (Option.map₂ (fun x y : ℝ => x - y)
            (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale k idx.1)
            (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale
              (k + 1) idx.1))]
        apply congrArg WithBot.realExp
        congr 2
        exact FA1MathCausal.mPartial_succ_kernelForm Q K scale qb k hk idx.1
      · refine Eq.trans
          (@Finset.sum_congr (Fin Bk) (WithBot ℝ) _ _ _ _ _ rfl
            (fun j _ => ?_))
          (WithBot.sum_someTerm_eq_some _ _)
        change Option.map _ _ =
            Option.map (fun a : ℝ => a *
              V (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, idx.2.1, PUnit.unit))
              (some ((WithBot.realExp
                (Option.map₂ (fun x y : ℝ => x - y)
                  (FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
                    (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j))
                  (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale
                    (k + 1) idx.1))).unbotD 0))
        congr 1
        simp_rw [Finset.sup'_eq_sup]
        have hscore :
            (if k * Bk + ↑j ≤ qb * M + ↑idx.1 then
              some
                ((∑ x_1 : Fin D,
                  Q (idx.1, x_1, PUnit.unit) *
                    K (FA1Math.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, x_1, PUnit.unit)) * scale)
            else none : WithBot ℝ) =
              FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
                (FA1Math.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j) := by
          by_cases h_mask : k * Bk + ↑j ≤ qb * M + ↑idx.1
          · rw [if_pos h_mask]
            rw [FA1MathCausal.maskedScore_of_le (qb * M) Q K scale idx.1 _
              (by simp [FA1Math.blockIndex]; exact h_mask)]
            unfold FA1Math.scaledScore
            ring_nf
            rfl
          · rw [if_neg h_mask]
            rw [FA1MathCausal.maskedScore_of_not_le (qb * M) Q K scale idx.1 _
              (by simp [FA1Math.blockIndex]; omega)]
            rfl
        rw [← FA1MathCausal.realExp_eq_some_unbotD
          (Option.map₂ (fun x y : ℝ => x - y)
            (FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
            (FA1Math.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j))
            (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale
              (k + 1) idx.1))]
        apply congrArg WithBot.realExp
        congr 2
        · unfold FA1Math.scaledScore
          ring_nf
          rfl
        · exact FA1MathCausal.mPartial_succ_kernelForm Q K scale qb k hk idx.1
    · intro idx
      simpa [s', s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hQ idx
    · intro idx
      simpa [s', s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hK idx
    · intro idx
      simpa [s', s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hV idx


end VeriTile.Examples
