/-
VeriTile.Examples.FlashAttention1.V1Boundary.BoundaryD

Split-out support for FlashAttention-1 v1 boundary proofs.
-/

import VeriTile.Examples.FlashAttention1.V1Boundary.Core

namespace VeriTile.Examples

open VeriTile.Triton

set_option maxHeartbeats 5000000 in
/-- D-tail boundary strided loop step. This is the D-tail analogue of
`fa1_step_strided_boundary`: K/V loads are guarded by both sequence and
hidden-dimension masks, producing the padded K/V block used by the
boundary recurrence over block width `Bd`. -/
theorem fa1_step_strided_boundaryD
    {M D Bd Bk numKVBlocks S_k : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState)
    (hk : k < numKVBlocks)
    (hP : P_fa1_strided_boundaryD (Bd := Bd) (Bk := Bk)
        (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale k s) :
    ∃ s',
      stepStmts (fa1LoopBodyStridedBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1_strided_boundaryD (Bd := Bd) (Bk := Bk)
        (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale (k + 1) s' := by
  rcases hP with
    ⟨hpids0, hpids1, hpids2,
     hpid_qb, hpid_h, hpid_b,
     hq_base, hk_base, hv_base, ho_base,
     hoffs_m, hoffs_d, hq, hm, hl, ho, hK, hV⟩
  let Qp : TileIndex [M, Bd] → ℝ := padHeadD (Bd := Bd) Q
  let Kp : TileIndex [S_k, Bd] → ℝ := padHeadD (Bd := Bd) K
  let Vp : TileIndex [S_k, Bd] → ℝ := padHeadD (Bd := Bd) V
  let kBase : Nat := batch * sKB + headIdx * sKH
  let vBase : Nat := batch * sVB + headIdx * sVH
  let offsN : Tile .nat [Bk] := Tile.vec fun j : Fin Bk => k * Bk + j.val
  let kPtrs : Tile .nat [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] =>
      kBase + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD⟩
  let vPtrs : Tile .nat [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] =>
      vBase + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD⟩
  let kvSeqMask : Tile .bool [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] => decide (k * Bk + idx.1.val < S_k)⟩
  let kvDMask : Tile .bool [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] => decide (idx.2.1.val < D)⟩
  let kvMask : Tile .bool [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] =>
      decide (k * Bk + idx.1.val < S_k) && decide (idx.2.1.val < D)⟩
  let kLoaded : Tile .real [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] =>
      if h : (k * Bk + idx.1.val < S_k) ∧ (idx.2.1.val < D) then
        some (s.readMem kReg
          (kBase + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD))
      else
        some 0⟩
  let vLoaded : Tile .real [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] =>
      if h : (k * Bk + idx.1.val < S_k) ∧ (idx.2.1.val < D) then
        some (s.readMem vReg
          (vBase + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD))
      else
        some 0⟩
  have hK_loaded_eq : kLoaded =
      Tile.ofReal (fun idx : TileIndex [Bk, Bd] =>
        match FA1MathBoundary.blockIndex? S_k Bk k idx.1 with
        | some j => Kp (j, idx.2.1, PUnit.unit)
        | none => 0) := by
    simpa [Kp] using
      fa1_block_load_tile_eq_strided_boundaryD kReg s kBase sKN sKD K hK k
  have hV_loaded_eq : vLoaded =
      Tile.ofReal (fun idx : TileIndex [Bk, Bd] =>
        match FA1MathBoundary.blockIndex? S_k Bk k idx.1 with
        | some j => Vp (j, idx.2.1, PUnit.unit)
        | none => 0) := by
    simpa [Vp] using
      fa1_block_load_tile_eq_strided_boundaryD vReg s vBase sVN sVD V hV k
  let scoresRaw : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      Option.map (fun a => a * scale)
        (@Finset.sum (Fin Bd) (WithBot ℝ) _ Finset.univ
          (fun d : Fin Bd => Option.map (fun b => Qp (idx.1, d, PUnit.unit) * b)
            (kLoaded.data (idx.2.1, d, PUnit.unit))))⟩
  let scoreMask : Tile .bool [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] => decide (k * Bk + idx.2.1.val < S_k)⟩
  let scores : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      if k * Bk + idx.2.1.val < S_k then scoresRaw.data idx else (none : WithBot ℝ)⟩
  let mBlock : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup'
        (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
        (fun j : Fin Bk => scores.data (idx.1, j, PUnit.unit))⟩
  let mNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      max (FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale k idx.1)
        (mBlock.data idx)⟩
  let alpha : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      WithBot.realExp
        (Option.map₂ (· - ·)
          (FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale k idx.1)
          (mNew.data idx))⟩
  let p : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      WithBot.realExp
        (Option.map₂ (· - ·) (scores.data idx) (mNew.data (idx.1, PUnit.unit)))⟩
  let lNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      Option.map₂ (· + ·)
        (Option.map (· * FA1MathBoundary.lPartial Bk Qp numKVBlocks Kp scale k idx.1)
          (alpha.data idx))
        (@Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ
          (fun j : Fin Bk => p.data (idx.1, j, PUnit.unit)))⟩
  let oNew : Tile .real [M, Bd] :=
    ⟨fun idx : TileIndex [M, Bd] =>
      Option.map₂ (· + ·)
        (Option.map
          (· * FA1MathBoundary.oPartial Bk Qp numKVBlocks Kp Vp scale k
            (idx.1, idx.2.1, PUnit.unit))
          (alpha.data (idx.1, PUnit.unit)))
        (@Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ
          (fun j : Fin Bk =>
            Option.map₂ (· * ·)
              (p.data (idx.1, j, PUnit.unit))
              (vLoaded.data (j, idx.2.1, PUnit.unit))))⟩
  let s0 := s.setReg "n" .nat [] (Tile.scalar k)
  let s1 := s0.setReg "offs_n" .nat [Bk] offsN
  let s2 := s1.setReg "k_ptrs" .nat [Bk, Bd] kPtrs
  let s3 := s2.setReg "v_ptrs" .nat [Bk, Bd] vPtrs
  let s4 := s3.setReg "kv_seq_mask" .bool [Bk, Bd] kvSeqMask
  let s5 := s4.setReg "kv_d_mask" .bool [Bk, Bd] kvDMask
  let s6 := s5.setReg "kv_mask" .bool [Bk, Bd] kvMask
  let s7 := s6.setReg "k" .real [Bk, Bd] kLoaded
  let s8 := s7.setReg "v" .real [Bk, Bd] vLoaded
  let s9 := s8.setReg "scores_raw" .real [M, Bk] scoresRaw
  let s10 := s9.setReg "score_mask" .bool [M, Bk] scoreMask
  let s11 := s10.setReg "scores" .real [M, Bk] scores
  let s12 := s11.setReg "m_block" .real [M] mBlock
  let s13 := s12.setReg "m_new" .real [M] mNew
  let s14 := s13.setReg "alpha" .real [M] alpha
  let s15 := s14.setReg "p" .real [M, Bk] p
  let s16 := s15.setReg "l_new" .real [M] lNew
  let s17 := s16.setReg "o_acc" .real [M, Bd] oNew
  let s18 := s17.setReg "m_i" .real [M] mNew
  let s' := s18.setReg "l_i" .real [M] lNew
  have h_score_per_j : ∀ (i : Fin M) (j : Fin Bk),
      scores.data (i, j, PUnit.unit)
        = FA1MathBoundary.maskedScore Bk k Qp Kp scale i j := by
    intro i j
    exact fa1_boundary_score_lane_eq Qp Kp scale k kLoaded hK_loaded_eq i j
  refine ⟨s', ?_, ?_⟩
  · simp [fa1LoopBodyStridedBoundaryD, stepStmts, stepStmt, evalOp,
      Tile.bop, Tile.cop, Tile.select, Tile.expandDim,
      Tile.transpose, Tile.dot, Tile.reduceMax, Tile.reduceMaxDrop,
      Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
      TileShape.eraseAxis, TileShape.insertAxisIndex, TileShape.dropInsertedIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      ComparableDType.lt, Option.bind,
      hBk, hoffs_m, hoffs_d, hq, hm, hl, ho, hk_base, hv_base]
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · exact hpids0
    · exact hpids1
    · exact hpids2
    · exact hpid_qb
    · exact hpid_h
    · exact hpid_b
    · exact hq_base
    · exact hk_base
    · exact hv_base
    · exact ho_base
    · simp [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hoffs_m]
    · simp [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hoffs_d]
    · simp [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hq]
    · have h_regs_m_i : s'.regs .real [M] "m_i" = some mNew := by
        simp [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_m_i]
      congr 1
      ext idx
      show max (FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale k idx.1) (mBlock.data idx)
        = FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale (k + 1) idx.1
      rw [FA1MathBoundary.mPartial_succ_of_lt Bk Qp numKVBlocks Kp scale k hk idx.1]
      congr 1
      show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
          (fun j => scores.data (idx.1, j, PUnit.unit))
          = Finset.univ.sup
              (fun j => FA1MathBoundary.maskedScore Bk k Qp Kp scale idx.1 j)
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      exact h_score_per_j idx.1 j
    · have h_regs_l_i : s'.regs .real [M] "l_i" = some lNew := by
        simp [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_l_i]
      apply congrArg some
      apply Tile.ext
      intro idx
      have h_mNew : mNew.data (idx.1, PUnit.unit)
          = FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale (k + 1) idx.1 := by
        show max (FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale k idx.1)
            (mBlock.data (idx.1, PUnit.unit)) = _
        rw [FA1MathBoundary.mPartial_succ_of_lt Bk Qp numKVBlocks Kp scale k hk idx.1]
        congr 1
        show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
            (fun j => scores.data (idx.1, j, PUnit.unit))
            = Finset.univ.sup
                (fun j => FA1MathBoundary.maskedScore Bk k Qp Kp scale idx.1 j)
        rw [Finset.sup'_eq_sup]
        apply Finset.sup_congr rfl
        intro j _
        exact h_score_per_j idx.1 j
      have h_alpha : alpha.data idx
          = some (FA1MathBoundary.alphaPartial Bk Qp numKVBlocks Kp scale k idx.1) := by
        show WithBot.realExp _ = _
        rw [h_mNew]
        unfold FA1MathBoundary.alphaPartial
        exact FA1MathBoundary.realExp_eq_some_unbotD _
      have h_p_sum : (∑ j : Fin Bk, p.data (idx.1, j, PUnit.unit) : WithBot ℝ)
          = some (∑ j : Fin Bk,
              (WithBot.realExp
                (WithBot.realSub (FA1MathBoundary.maskedScore Bk k Qp Kp scale idx.1 j)
                  (FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale (k + 1) idx.1))
              ).unbotD 0) := by
        rw [← WithBot.sum_someTerm_eq_some]
        apply Finset.sum_congr rfl
        intro j _
        show WithBot.realExp _ = _
        rw [h_score_per_j idx.1 j, h_mNew]
        exact FA1MathBoundary.realExp_eq_some_unbotD _
      show Option.map₂ (fun x1 x2 : ℝ => x1 + x2)
          (Option.map (fun x : ℝ =>
            x * FA1MathBoundary.lPartial Bk Qp numKVBlocks Kp scale k idx.1)
            (alpha.data idx))
          (∑ j : Fin Bk, p.data (idx.1, j, PUnit.unit) : WithBot ℝ)
          = some (FA1MathBoundary.lPartial Bk Qp numKVBlocks Kp scale (k + 1) idx.1)
      rw [h_alpha, h_p_sum,
        FA1MathBoundary.lPartial_succ_of_lt Bk Qp numKVBlocks Kp scale k hk idx.1]
      rfl
    · have h_regs_o_acc : s'.regs .real [M, Bd] "o_acc" = some oNew := by
        simp [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_o_acc]
      apply congrArg some
      apply Tile.ext
      intro idx
      have h_mNew : mNew.data (idx.1, PUnit.unit)
          = FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale (k + 1) idx.1 := by
        show max (FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale k idx.1)
            (mBlock.data (idx.1, PUnit.unit)) = _
        rw [FA1MathBoundary.mPartial_succ_of_lt Bk Qp numKVBlocks Kp scale k hk idx.1]
        congr 1
        show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
            (fun j => scores.data (idx.1, j, PUnit.unit))
            = Finset.univ.sup
                (fun j => FA1MathBoundary.maskedScore Bk k Qp Kp scale idx.1 j)
        rw [Finset.sup'_eq_sup]
        apply Finset.sup_congr rfl
        intro j _
        exact h_score_per_j idx.1 j
      have h_alpha : alpha.data (idx.1, PUnit.unit)
          = some (FA1MathBoundary.alphaPartial Bk Qp numKVBlocks Kp scale k idx.1) := by
        show WithBot.realExp _ = _
        rw [h_mNew]
        unfold FA1MathBoundary.alphaPartial
        exact FA1MathBoundary.realExp_eq_some_unbotD _
      have h_vLoaded : ∀ j : Fin Bk,
          vLoaded.data (j, idx.2.1, PUnit.unit)
            = some (match FA1MathBoundary.blockIndex? S_k Bk k j with
                    | some jGlobal => Vp (jGlobal, idx.2.1, PUnit.unit)
                    | none => 0) := by
        intro j
        have := congrArg (fun t : Tile .real [Bk, Bd] => t.data (j, idx.2.1, PUnit.unit))
          hV_loaded_eq
        simp [Tile.ofReal] at this
        exact this
      have h_p_per_j : ∀ j : Fin Bk,
          p.data (idx.1, j, PUnit.unit)
            = some ((WithBot.realExp
                (WithBot.realSub (FA1MathBoundary.maskedScore Bk k Qp Kp scale idx.1 j)
                  (FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale (k + 1) idx.1))
              ).unbotD 0) := by
        intro j
        show WithBot.realExp _ = _
        rw [h_score_per_j idx.1 j, h_mNew]
        exact FA1MathBoundary.realExp_eq_some_unbotD _
      have h_pv_sum :
          (∑ j : Fin Bk, Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
              (p.data (idx.1, j, PUnit.unit))
              (vLoaded.data (j, idx.2.1, PUnit.unit)) : WithBot ℝ)
            = some (∑ j : Fin Bk,
                match FA1MathBoundary.blockIndex? S_k Bk k j with
                | some jGlobal =>
                    (WithBot.realExp
                      (WithBot.realSub
                        (FA1MathBoundary.maskedScore Bk k Qp Kp scale idx.1 j)
                        (FA1MathBoundary.mPartial Bk Qp numKVBlocks Kp scale (k + 1) idx.1))
                    ).unbotD 0 * Vp (jGlobal, idx.2.1, PUnit.unit)
                | none => 0) := by
        rw [← WithBot.sum_someTerm_eq_some]
        apply Finset.sum_congr rfl
        intro j _
        rw [h_p_per_j j, h_vLoaded j]
        by_cases h : k * Bk + j.val < S_k
        · simp [FA1MathBoundary.blockIndex?_of_lt _ _ _ _ h]
        · simp [FA1MathBoundary.blockIndex?_of_not_lt _ _ _ _ h]
      show Option.map₂ (fun x1 x2 : ℝ => x1 + x2)
          (Option.map (fun x : ℝ =>
            x * FA1MathBoundary.oPartial Bk Qp numKVBlocks Kp Vp scale k
              (idx.1, idx.2.1, PUnit.unit))
            (alpha.data (idx.1, PUnit.unit)))
          (∑ j : Fin Bk, Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (p.data (idx.1, j, PUnit.unit))
            (vLoaded.data (j, idx.2.1, PUnit.unit)) : WithBot ℝ)
          = some (FA1MathBoundary.oPartial Bk Qp numKVBlocks Kp Vp scale (k + 1) idx)
      rw [h_alpha, h_pv_sum,
        FA1MathBoundary.oPartial_succ_of_lt Bk Qp numKVBlocks Kp Vp scale k hk idx]
      rfl
    · intro idx
      simpa [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hK idx
    · intro idx
      simpa [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hV idx

/-- D-tail boundary strided FA-1 forward correctness, raw form. Bundles
`fa1_forward_correct_strided_boundaryD_raw_of_step` with the proven
`fa1_step_strided_boundaryD`. -/
theorem fa1_forward_correct_strided_boundaryD_raw
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, Bd],
      s.pids 0 * M + idx.1.val < S_q →
      idx.2.1.val < D →
      observeTileAt
          (exec (fa1ForwardKernelStridedBoundaryD qReg kReg vReg outReg
              M Bd Bk numKVBlocks S_q S_k D
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathBoundary.oPartial Bk
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
                scale numKVBlocks idx /
              FA1MathBoundary.lPartial Bk
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) scale numKVBlocks idx.1) := by
  exact fa1_forward_correct_strided_boundaryD_raw_of_step
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV hInj
    (fun i st hi hPi =>
      fa1_step_strided_boundaryD hBk qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale i st hi hPi)

/-- D-tail boundary strided FA-1 forward correctness in canonical spec form. -/
theorem fa1_forward_correct_strided_boundaryD
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, Bd],
      s.pids 0 * M + idx.1.val < S_q →
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
        = some (attentionReal Q K V scale
            (idx.1, ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  exact fa1_forward_correct_strided_boundaryD_of_step hBk hSk hSkLe hDLe
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV hInj
    (fun i st hi hPi =>
      fa1_step_strided_boundaryD hBk qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale i st hi hPi)

set_option maxHeartbeats 5000000 in
/-- D-tail causal-boundary strided loop step. This is the causal analogue of
`fa1_step_strided_boundaryD`: K/V loads are D-tail masked, scores are causal
masked and then sequence masked, and the recurrence is
`FA1MathCausalBoundary` over padded hidden width `Bd`. -/
theorem fa1_step_strided_causal_boundaryD
    {M D Bd Bk numKVBlocks S_k : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState)
    (hk : k < numKVBlocks)
    (hP : P_fa1_strided_causal_boundaryD (Bd := Bd) (Bk := Bk)
        (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale k s) :
    ∃ s',
      stepStmts (fa1LoopBodyStridedCausalBoundaryD kReg vReg M Bd Bk S_k D sKN sKD sVN sVD scale)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1_strided_causal_boundaryD (Bd := Bd) (Bk := Bk)
        (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale (k + 1) s' := by
  rcases hP with
    ⟨hpids0, hpids1, hpids2,
     hpid_qb, hpid_h, hpid_b,
     hq_base, hk_base, hv_base, ho_base,
     hoffs_m, hoffs_d, hq, hm, hl, ho, hK, hV⟩
  let Qp : TileIndex [M, Bd] → ℝ := padHeadD (Bd := Bd) Q
  let Kp : TileIndex [S_k, Bd] → ℝ := padHeadD (Bd := Bd) K
  let Vp : TileIndex [S_k, Bd] → ℝ := padHeadD (Bd := Bd) V
  let qStart : Nat := qb * M
  let kBase : Nat := batch * sKB + headIdx * sKH
  let vBase : Nat := batch * sVB + headIdx * sVH
  let offsN : Tile .nat [Bk] := Tile.vec fun j : Fin Bk => k * Bk + j.val
  let kPtrs : Tile .nat [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] =>
      kBase + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD⟩
  let vPtrs : Tile .nat [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] =>
      vBase + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD⟩
  let kvSeqMask : Tile .bool [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] => decide (k * Bk + idx.1.val < S_k)⟩
  let kvDMask : Tile .bool [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] => decide (idx.2.1.val < D)⟩
  let kvMask : Tile .bool [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] =>
      decide (k * Bk + idx.1.val < S_k) && decide (idx.2.1.val < D)⟩
  let kLoaded : Tile .real [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] =>
      if h : (k * Bk + idx.1.val < S_k) ∧ (idx.2.1.val < D) then
        some (s.readMem kReg
          (kBase + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD))
      else
        some 0⟩
  let vLoaded : Tile .real [Bk, Bd] :=
    ⟨fun idx : TileIndex [Bk, Bd] =>
      if h : (k * Bk + idx.1.val < S_k) ∧ (idx.2.1.val < D) then
        some (s.readMem vReg
          (vBase + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD))
      else
        some 0⟩
  have hK_loaded_eq : kLoaded =
      Tile.ofReal (fun idx : TileIndex [Bk, Bd] =>
        match FA1MathBoundary.blockIndex? S_k Bk k idx.1 with
        | some j => Kp (j, idx.2.1, PUnit.unit)
        | none => 0) := by
    simpa [Kp] using
      fa1_block_load_tile_eq_strided_boundaryD kReg s kBase sKN sKD K hK k
  have hV_loaded_eq : vLoaded =
      Tile.ofReal (fun idx : TileIndex [Bk, Bd] =>
        match FA1MathBoundary.blockIndex? S_k Bk k idx.1 with
        | some j => Vp (j, idx.2.1, PUnit.unit)
        | none => 0) := by
    simpa [Vp] using
      fa1_block_load_tile_eq_strided_boundaryD vReg s vBase sVN sVD V hV k
  let scoresRaw : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      Option.map (fun a => a * scale)
        (@Finset.sum (Fin Bd) (WithBot ℝ) _ Finset.univ
          (fun d : Fin Bd => Option.map (fun b => Qp (idx.1, d, PUnit.unit) * b)
            (kLoaded.data (idx.2.1, d, PUnit.unit))))⟩
  let causal : Tile .bool [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      decide (k * Bk + idx.2.1.val ≤ qStart + idx.1.val)⟩
  let causalScores : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      if k * Bk + idx.2.1.val ≤ qStart + idx.1.val then scoresRaw.data idx
      else (none : WithBot ℝ)⟩
  let scoreMask : Tile .bool [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] => decide (k * Bk + idx.2.1.val < S_k)⟩
  let scores : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      if k * Bk + idx.2.1.val < S_k then causalScores.data idx else (none : WithBot ℝ)⟩
  let mBlock : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup'
        (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
        (fun j : Fin Bk => scores.data (idx.1, j, PUnit.unit))⟩
  let mNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      max (FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale k idx.1)
        (mBlock.data idx)⟩
  let alpha : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      WithBot.realExp
        (WithBot.realSub
          (FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale k idx.1)
          (mNew.data idx))⟩
  let p : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      WithBot.realExp
        (WithBot.realSub (scores.data idx) (mNew.data (idx.1, PUnit.unit)))⟩
  let lNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      Option.map₂ (· + ·)
        (Option.map (· * FA1MathCausalBoundary.lPartial Bk qStart Qp numKVBlocks Kp scale k idx.1)
          (alpha.data idx))
        (@Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ
          (fun j : Fin Bk => p.data (idx.1, j, PUnit.unit)))⟩
  let oNew : Tile .real [M, Bd] :=
    ⟨fun idx : TileIndex [M, Bd] =>
      Option.map₂ (· + ·)
        (Option.map
          (· * FA1MathCausalBoundary.oPartial Bk qStart Qp numKVBlocks Kp Vp scale k
            (idx.1, idx.2.1, PUnit.unit))
          (alpha.data (idx.1, PUnit.unit)))
        (@Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ
          (fun j : Fin Bk =>
            Option.map₂ (· * ·)
              (p.data (idx.1, j, PUnit.unit))
              (vLoaded.data (j, idx.2.1, PUnit.unit))))⟩
  let s0 := s.setReg "n" .nat [] (Tile.scalar k)
  let s1 := s0.setReg "offs_n" .nat [Bk] offsN
  let s2 := s1.setReg "k_ptrs" .nat [Bk, Bd] kPtrs
  let s3 := s2.setReg "v_ptrs" .nat [Bk, Bd] vPtrs
  let s4 := s3.setReg "kv_seq_mask" .bool [Bk, Bd] kvSeqMask
  let s5 := s4.setReg "kv_d_mask" .bool [Bk, Bd] kvDMask
  let s6 := s5.setReg "kv_mask" .bool [Bk, Bd] kvMask
  let s7 := s6.setReg "k" .real [Bk, Bd] kLoaded
  let s8 := s7.setReg "v" .real [Bk, Bd] vLoaded
  let s9 := s8.setReg "scores_raw" .real [M, Bk] scoresRaw
  let s10 := s9.setReg "causal" .bool [M, Bk] causal
  let s11 := s10.setReg "causal_scores" .real [M, Bk] causalScores
  let s12 := s11.setReg "score_mask" .bool [M, Bk] scoreMask
  let s13 := s12.setReg "scores" .real [M, Bk] scores
  let s14 := s13.setReg "m_block" .real [M] mBlock
  let s15 := s14.setReg "m_new" .real [M] mNew
  let s16 := s15.setReg "alpha" .real [M] alpha
  let s17 := s16.setReg "p" .real [M, Bk] p
  let s18 := s17.setReg "l_new" .real [M] lNew
  let s19 := s18.setReg "o_acc" .real [M, Bd] oNew
  let s20 := s19.setReg "m_i" .real [M] mNew
  let s' := s20.setReg "l_i" .real [M] lNew
  have h_score_per_j : ∀ (i : Fin M) (j : Fin Bk),
      scores.data (i, j, PUnit.unit)
        = FA1MathCausalBoundary.maskedScore Bk k qStart Qp Kp scale i j := by
    intro i j
    exact fa1_causal_boundary_score_lane_eq qStart Qp Kp scale k kLoaded hK_loaded_eq i j
  refine ⟨s', ?_, ?_⟩
  · simp [fa1LoopBodyStridedCausalBoundaryD, stepStmts, stepStmt, evalOp,
      Tile.bop, Tile.cop, Tile.select, Tile.expandDim,
      Tile.transpose, Tile.dot, Tile.reduceMax, Tile.reduceMaxDrop,
      Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
      TileShape.eraseAxis, TileShape.insertAxisIndex, TileShape.dropInsertedIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      ComparableDType.lt, ComparableDType.ge, Option.bind,
      hBk, hoffs_m, hoffs_d, hq, hm, hl, ho, hk_base, hv_base]
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · exact hpids0
    · exact hpids1
    · exact hpids2
    · exact hpid_qb
    · exact hpid_h
    · exact hpid_b
    · exact hq_base
    · exact hk_base
    · exact hv_base
    · exact ho_base
    · simp [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hoffs_m]
    · simp [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hoffs_d]
    · simp [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hq]
    · have h_regs_m_i : s'.regs .real [M] "m_i" = some mNew := by
        simp [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_m_i]
      congr 1
      ext idx
      show max (FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale k idx.1) (mBlock.data idx)
        = FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale (k + 1) idx.1
      rw [FA1MathCausalBoundary.mPartial_succ_of_lt Bk qStart Qp numKVBlocks Kp scale k hk idx.1]
      congr 1
      show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
          (fun j => scores.data (idx.1, j, PUnit.unit))
          = Finset.univ.sup
              (fun j => FA1MathCausalBoundary.maskedScore Bk k qStart Qp Kp scale idx.1 j)
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      exact h_score_per_j idx.1 j
    · have h_regs_l_i : s'.regs .real [M] "l_i" = some lNew := by
        simp [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_l_i]
      apply congrArg some
      apply Tile.ext
      intro idx
      have h_mNew : mNew.data (idx.1, PUnit.unit)
          = FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale (k + 1) idx.1 := by
        show max (FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale k idx.1)
            (mBlock.data (idx.1, PUnit.unit)) = _
        rw [FA1MathCausalBoundary.mPartial_succ_of_lt Bk qStart Qp numKVBlocks Kp scale k hk idx.1]
        congr 1
        show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
            (fun j => scores.data (idx.1, j, PUnit.unit))
            = Finset.univ.sup
                (fun j => FA1MathCausalBoundary.maskedScore Bk k qStart Qp Kp scale idx.1 j)
        rw [Finset.sup'_eq_sup]
        apply Finset.sup_congr rfl
        intro j _
        exact h_score_per_j idx.1 j
      have h_alpha : alpha.data idx
          = some (FA1MathCausalBoundary.alphaPartial Bk qStart Qp numKVBlocks Kp scale k idx.1) := by
        show WithBot.realExp _ = _
        rw [h_mNew]
        unfold FA1MathCausalBoundary.alphaPartial
        exact FA1MathCausalBoundary.realExp_eq_some_unbotD _
      have h_p_sum : (∑ j : Fin Bk, p.data (idx.1, j, PUnit.unit) : WithBot ℝ)
          = some (∑ j : Fin Bk,
              (WithBot.realExp
                (WithBot.realSub
                  (FA1MathCausalBoundary.maskedScore Bk k qStart Qp Kp scale idx.1 j)
                  (FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale (k + 1) idx.1))
              ).unbotD 0) := by
        rw [← WithBot.sum_someTerm_eq_some]
        apply Finset.sum_congr rfl
        intro j _
        show WithBot.realExp _ = _
        rw [h_score_per_j idx.1 j, h_mNew]
        exact FA1MathCausalBoundary.realExp_eq_some_unbotD _
      show Option.map₂ (fun x1 x2 : ℝ => x1 + x2)
          (Option.map (fun x : ℝ =>
            x * FA1MathCausalBoundary.lPartial Bk qStart Qp numKVBlocks Kp scale k idx.1)
            (alpha.data idx))
          (∑ j : Fin Bk, p.data (idx.1, j, PUnit.unit) : WithBot ℝ)
          = some (FA1MathCausalBoundary.lPartial Bk qStart Qp numKVBlocks Kp scale (k + 1) idx.1)
      rw [h_alpha, h_p_sum,
        FA1MathCausalBoundary.lPartial_succ_of_lt Bk qStart Qp numKVBlocks Kp scale k hk idx.1]
      rfl
    · have h_regs_o_acc : s'.regs .real [M, Bd] "o_acc" = some oNew := by
        simp [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_o_acc]
      apply congrArg some
      apply Tile.ext
      intro idx
      have h_mNew : mNew.data (idx.1, PUnit.unit)
          = FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale (k + 1) idx.1 := by
        show max (FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale k idx.1)
            (mBlock.data (idx.1, PUnit.unit)) = _
        rw [FA1MathCausalBoundary.mPartial_succ_of_lt Bk qStart Qp numKVBlocks Kp scale k hk idx.1]
        congr 1
        show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
            (fun j => scores.data (idx.1, j, PUnit.unit))
            = Finset.univ.sup
                (fun j => FA1MathCausalBoundary.maskedScore Bk k qStart Qp Kp scale idx.1 j)
        rw [Finset.sup'_eq_sup]
        apply Finset.sup_congr rfl
        intro j _
        exact h_score_per_j idx.1 j
      have h_alpha : alpha.data (idx.1, PUnit.unit)
          = some (FA1MathCausalBoundary.alphaPartial Bk qStart Qp numKVBlocks Kp scale k idx.1) := by
        show WithBot.realExp _ = _
        rw [h_mNew]
        unfold FA1MathCausalBoundary.alphaPartial
        exact FA1MathCausalBoundary.realExp_eq_some_unbotD _
      have h_vLoaded : ∀ j : Fin Bk,
          vLoaded.data (j, idx.2.1, PUnit.unit)
            = some (match FA1MathBoundary.blockIndex? S_k Bk k j with
                    | some jGlobal => Vp (jGlobal, idx.2.1, PUnit.unit)
                    | none => 0) := by
        intro j
        have := congrArg (fun t : Tile .real [Bk, Bd] => t.data (j, idx.2.1, PUnit.unit))
          hV_loaded_eq
        simp [Tile.ofReal] at this
        exact this
      have h_p_per_j : ∀ j : Fin Bk,
          p.data (idx.1, j, PUnit.unit)
            = some ((WithBot.realExp
                (WithBot.realSub
                  (FA1MathCausalBoundary.maskedScore Bk k qStart Qp Kp scale idx.1 j)
                  (FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale (k + 1) idx.1))
              ).unbotD 0) := by
        intro j
        show WithBot.realExp _ = _
        rw [h_score_per_j idx.1 j, h_mNew]
        exact FA1MathCausalBoundary.realExp_eq_some_unbotD _
      have h_pv_sum :
          (∑ j : Fin Bk, Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
              (p.data (idx.1, j, PUnit.unit))
              (vLoaded.data (j, idx.2.1, PUnit.unit)) : WithBot ℝ)
            = some (∑ j : Fin Bk,
                match FA1MathBoundary.blockIndex? S_k Bk k j with
                | some jGlobal =>
                    if jGlobal.val ≤ qStart + idx.1.val then
                      (WithBot.realExp
                        (WithBot.realSub
                          (FA1MathCausalBoundary.maskedScore Bk k qStart Qp Kp scale idx.1 j)
                          (FA1MathCausalBoundary.mPartial Bk qStart Qp numKVBlocks Kp scale (k + 1) idx.1))
                      ).unbotD 0 * Vp (jGlobal, idx.2.1, PUnit.unit)
                    else
                      0
                | none => 0) := by
        rw [← WithBot.sum_someTerm_eq_some]
        apply Finset.sum_congr rfl
        intro j _
        rw [h_p_per_j j, h_vLoaded j]
        by_cases hLt : k * Bk + j.val < S_k
        · rw [FA1MathBoundary.blockIndex?_of_lt S_k Bk k j hLt]
          by_cases hLe : k * Bk + j.val ≤ qStart + idx.1.val
          · simp [hLe]
          · rw [FA1MathCausalBoundary.maskedScore_of_lt_of_not_le Bk k qStart Qp Kp scale idx.1 j hLt hLe]
            simp [hLe]
        · simp [FA1MathBoundary.blockIndex?_of_not_lt _ _ _ _ hLt]
      show Option.map₂ (fun x1 x2 : ℝ => x1 + x2)
          (Option.map (fun x : ℝ =>
            x * FA1MathCausalBoundary.oPartial Bk qStart Qp numKVBlocks Kp Vp scale k
              (idx.1, idx.2.1, PUnit.unit))
            (alpha.data (idx.1, PUnit.unit)))
          (∑ j : Fin Bk, Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (p.data (idx.1, j, PUnit.unit))
            (vLoaded.data (j, idx.2.1, PUnit.unit)) : WithBot ℝ)
          = some (FA1MathCausalBoundary.oPartial Bk qStart Qp numKVBlocks Kp Vp scale (k + 1) idx)
      rw [h_alpha, h_pv_sum,
        FA1MathCausalBoundary.oPartial_succ_of_lt Bk qStart Qp numKVBlocks Kp Vp scale k hk idx]
      rfl
    · intro idx
      simpa [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hK idx
    · intro idx
      simpa [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hV idx

/-- D-tail causal-boundary strided FA-1 forward correctness, raw form. -/
theorem fa1_forward_correct_strided_causal_boundaryD_raw
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, Bd],
      s.pids 0 * M + idx.1.val < S_q →
      idx.2.1.val < D →
      observeTileAt
          (exec (fa1ForwardKernelStridedCausalBoundaryD qReg kReg vReg outReg
              M Bd Bk numKVBlocks S_q S_k D
              sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
              sOB sOH sOM sOD scale) s)
          outReg
          (fun idx : TileIndex [M, Bd] =>
            s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
              + idx.1.val * sOM + idx.2.1.val * sOD) idx
        = some
            (FA1MathCausalBoundary.oPartial Bk (s.pids 0 * M)
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
                scale numKVBlocks idx /
              FA1MathCausalBoundary.lPartial Bk (s.pids 0 * M)
                (padHeadD (Bd := Bd) Q) numKVBlocks
                (padHeadD (Bd := Bd) K) scale numKVBlocks idx.1) := by
  exact fa1_forward_correct_strided_causal_boundaryD_raw_of_step
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV hInj
    (fun i st hi hPi =>
      fa1_step_strided_causal_boundaryD hBk qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale i st hi hPi)

/-- D-tail causal-boundary strided FA-1 forward correctness in canonical
causal-block spec form. -/
theorem fa1_forward_correct_strided_causal_boundaryD
    {M D Bd Bk numKVBlocks S_q S_k : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (hDLe : D ≤ Bd)
    (qReg kReg vReg outReg : RegionName)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQIn : ∀ idx : TileIndex [M, D],
        s.pids 0 * M + idx.1.val < S_q →
        s.readMem qReg
          (s.pids 2 * sQB + s.pids 1 * sQH + s.pids 0 * M * sQS
            + idx.1.val * sQS + idx.2.1.val * sQD) = Q idx)
    (hQOut : ∀ idx : TileIndex [M, D],
        ¬ s.pids 0 * M + idx.1.val < S_q → Q idx = 0)
    (hK : InputAt s kReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sKB + s.pids 1 * sKH
            + idx.1.val * sKN + idx.2.1.val * sKD) K)
    (hV : InputAt s vReg
        (fun idx : TileIndex [S_k, D] =>
          s.pids 2 * sVB + s.pids 1 * sVH
            + idx.1.val * sVN + idx.2.1.val * sVD) V)
    (hInj : Function.Injective
      (fun idx : TileIndex [M, D] =>
        s.pids 2 * sOB + s.pids 1 * sOH + s.pids 0 * M * sOM
          + idx.1.val * sOM + idx.2.1.val * sOD)) :
    ∀ idx : TileIndex [M, Bd],
      s.pids 0 * M + idx.1.val < S_q →
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
        = some (attentionRealCausalBlock (s.pids 0 * M) Q K V scale
            (idx.1, ⟨idx.2.1.val, hDIdx⟩, PUnit.unit)) := by
  exact fa1_forward_correct_strided_causal_boundaryD_of_step hBk hSk hSkLe hDLe
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV hInj
    (fun i st hi hPi =>
      fa1_step_strided_causal_boundaryD hBk qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale i st hi hPi)


end VeriTile.Examples
