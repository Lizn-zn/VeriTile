/-
VeriTile.Examples.FlashAttention1.Boundary.BoundaryD

Split-out support for FlashAttention-1 v1 boundary proofs.
-/

import VeriTile.Examples.FlashAttention1.Boundary.Core

namespace VeriTile.Examples

open VeriTile

private theorem fa1_boundaryD_eval_k_ptrs
    (s : BlockState) (Bk Bd k batch headIdx sKB sKH sKN sKD : Nat)
    (hk_base : s.regs .nat [] "k_base_off" =
      some (Tile.scalar (batch * sKB + headIdx * sKH)))
    (hoffs_d : s.regs .nat [Bd] "offs_d" =
      some (Tile.vec fun j : Fin Bd => j.val)) :
    evalOp
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.add .nat Broadcast.scalarL
          (Op.ref .nat [] "k_base_off")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.constNat sKN)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d"))
          (Op.constNat sKD)))
      ((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
        ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])) =
    some (⟨fun idx : TileIndex [Bk, Bd] =>
      batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
        + idx.2.1.val * sKD⟩ : Tile .nat [Bk, Bd]) := by
  have hn :
      evalOp (Op.expandDim 1 (Op.ref .nat [Bk] "offs_n"))
          ((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
            ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])) =
        some (Tile.expandDim 1
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])) := by
    simp
  have hd :
      evalOp (Op.expandDim 0 (Op.ref .nat [Bd] "offs_d"))
          ((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
            ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])) =
        some (Tile.expandDim 0 (Tile.vec fun j : Fin Bd => j.val)) := by
    simp [hoffs_d]
  simp only [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat]
  erw [hn, hd]
  simp [Option.bind, option_match_bind, Tile.bop, Tile.expandDim,
    NumericDType.add, NumericDType.mul,
    hk_base, hoffs_d, hn, hd]
  funext idx
  cases idx with
  | mk i rest =>
    cases rest with
    | mk j rest2 =>
      cases rest2
      rfl

private theorem fa1_boundaryD_eval_v_ptrs
    (s : BlockState) (Bk Bd k batch headIdx sVB sVH sVN sVD : Nat)
    (kPtrs : Tile .nat [Bk, Bd])
    (hv_base : s.regs .nat [] "v_base_off" =
      some (Tile.scalar (batch * sVB + headIdx * sVH)))
    (hoffs_d : s.regs .nat [Bd] "offs_d" =
      some (Tile.vec fun j : Fin Bd => j.val)) :
    evalOp
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.add .nat Broadcast.scalarL
          (Op.ref .nat [] "v_base_off")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.constNat sVN)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d"))
          (Op.constNat sVD)))
      (((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
        ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
          "k_ptrs" .nat [Bk, Bd] kPtrs) =
    some (⟨fun idx : TileIndex [Bk, Bd] =>
      batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
        + idx.2.1.val * sVD⟩ : Tile .nat [Bk, Bd]) := by
  have hn :
      evalOp (Op.expandDim 1 (Op.ref .nat [Bk] "offs_n"))
          (((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
            ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
              "k_ptrs" .nat [Bk, Bd] kPtrs) =
        some (Tile.expandDim 1
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])) := by
    simp
  have hd :
      evalOp (Op.expandDim 0 (Op.ref .nat [Bd] "offs_d"))
          (((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
            ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
              "k_ptrs" .nat [Bk, Bd] kPtrs) =
        some (Tile.expandDim 0 (Tile.vec fun j : Fin Bd => j.val)) := by
    simp [hoffs_d]
  simp only [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat]
  erw [hn, hd]
  simp [Option.bind, option_match_bind, Tile.bop, Tile.expandDim,
    NumericDType.add, NumericDType.mul,
    hv_base, hoffs_d, hn, hd]
  funext idx
  cases idx with
  | mk i rest =>
    cases rest with
    | mk j rest2 =>
      cases rest2
      rfl

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
  have h_kPtrs_eval :=
    fa1_boundaryD_eval_k_ptrs s Bk Bd k batch headIdx sKB sKH sKN sKD hk_base hoffs_d
  have h_vPtrs_eval :=
    fa1_boundaryD_eval_v_ptrs s Bk Bd k batch headIdx sVB sVH sVN sVD kPtrs hv_base hoffs_d
  have h_offsN_s3 :
      evalOp (Op.expandDim 1 (Op.ref .nat [Bk] "offs_n")) s3 =
        some (Tile.expandDim 1 offsN) := by
    simp [s3, s2, s1, s0, offsN]
  have h_offsD_s3 :
      evalOp (Op.expandDim 0 (Op.ref .nat [Bd] "offs_d")) s3 =
        some (Tile.expandDim 0 (Tile.vec fun d : Fin Bd => d.val)) := by
    simp [s3, s2, s1, s0, offsN, hoffs_d]
  have h_offsN_s3_exp :
      evalOp (Op.expandDim 1 (Op.ref .nat [Bk] "offs_n"))
        ((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, Bd])) =
        some (Tile.expandDim 1
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])) := by
    simpa [s3, s2, s1, s0, offsN, kPtrs, vPtrs, kBase, vBase] using h_offsN_s3
  have h_offsD_s3_exp :
      evalOp (Op.expandDim 0 (Op.ref .nat [Bd] "offs_d"))
        ((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, Bd])) =
        some (Tile.expandDim 0 (Tile.vec fun d : Fin Bd => d.val)) := by
    simpa [s3, s2, s1, s0, offsN, kPtrs, vPtrs, kBase, vBase] using h_offsD_s3
  have h_offsN_s4 :
      evalOp (Op.expandDim 1 (Op.ref .nat [Bk] "offs_n")) s4 =
        some (Tile.expandDim 1 offsN) := by
    simp [s4, s3, s2, s1, s0, offsN]
  have h_offsD_s4 :
      evalOp (Op.expandDim 0 (Op.ref .nat [Bd] "offs_d")) s4 =
        some (Tile.expandDim 0 (Tile.vec fun d : Fin Bd => d.val)) := by
    simp [s4, s3, s2, s1, s0, offsN, hoffs_d]
  have h_offsN_s4_exp :
      evalOp (Op.expandDim 1 (Op.ref .nat [Bk] "offs_n"))
        (((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, Bd])).setReg
            "kv_seq_mask" .bool [Bk, Bd]
            ({ data := fun i : TileIndex [Bk, Bd] =>
              decide (k * Bk + (TileShape.dropInsertedIndex [Bk] 1 1 (i.1, 0, PUnit.unit)).1.val < S_k) } :
              Tile .bool [Bk, Bd])) =
        some (Tile.expandDim 1
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])) := by
    simpa [s4, s3, s2, s1, s0, offsN, kPtrs, vPtrs, kvSeqMask, kBase, vBase,
      Tile.expandDim, Tile.bop, TileShape.dropInsertedIndex, NumericDType.add,
      NumericDType.mul] using h_offsN_s4
  have h_offsD_s4_exp :
      evalOp (Op.expandDim 0 (Op.ref .nat [Bd] "offs_d"))
        (((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, Bd])).setReg
            "kv_seq_mask" .bool [Bk, Bd]
            ({ data := fun i : TileIndex [Bk, Bd] =>
              decide (k * Bk + (TileShape.dropInsertedIndex [Bk] 1 1 (i.1, 0, PUnit.unit)).1.val < S_k) } :
              Tile .bool [Bk, Bd])) =
        some (Tile.expandDim 0 (Tile.vec fun d : Fin Bd => d.val)) := by
    simpa [s4, s3, s2, s1, s0, offsN, kPtrs, vPtrs, kvSeqMask, kBase, vBase,
      Tile.expandDim, Tile.bop, TileShape.dropInsertedIndex, NumericDType.add,
      NumericDType.mul] using h_offsD_s4
  have h_kvSeqMask_raw :
      ({ data := fun i : TileIndex [Bk, Bd] =>
        decide (k * Bk + (TileShape.dropInsertedIndex [Bk] 1 1 (i.1, 0, PUnit.unit)).1.val < S_k) } :
        Tile .bool [Bk, Bd]) = kvSeqMask := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_kvDMask_raw :
      ({ data := fun i : TileIndex [Bk, Bd] =>
        decide ((TileShape.dropInsertedIndex [Bd] 0 1 (0, i.2.1, PUnit.unit)).1.val < D) } :
        Tile .bool [Bk, Bd]) = kvDMask := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_kvSeqMask_lane (i : TileIndex [Bk, Bd]) :
      decide (k * Bk + (TileShape.dropInsertedIndex [Bk] 1 1 (i.1, 0, PUnit.unit)).1.val < S_k) =
        kvSeqMask.data (i.1, i.2.1, PUnit.unit) := by
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_kvDMask_lane (i : TileIndex [Bk, Bd]) :
      decide ((TileShape.dropInsertedIndex [Bd] 0 1 (0, i.2.1, PUnit.unit)).1.val < D) =
        kvDMask.data (i.1, i.2.1, PUnit.unit) := by
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_kvMask_raw :
      ({ data := fun i : TileIndex [Bk, Bd] =>
        decide (k * Bk + (TileShape.dropInsertedIndex [Bk] 1 1 (i.1, 0, PUnit.unit)).1.val < S_k) &&
          decide ((TileShape.dropInsertedIndex [Bd] 0 1 (0, i.2.1, PUnit.unit)).1.val < D) } :
        Tile .bool [Bk, Bd]) = kvMask := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_kLoaded_raw :
      ({ data := fun i : TileIndex [Bk, Bd] =>
        if k * Bk + (TileShape.dropInsertedIndex [Bk] 1 1 (i.1, 0, PUnit.unit)).1.val < S_k ∧
            (TileShape.dropInsertedIndex [Bd] 0 1 (0, i.2.1, PUnit.unit)).1.val < D then
          some (s.readMem kReg (batch * sKB + headIdx * sKH + (k * Bk + i.1.val) * sKN + i.2.1.val * sKD))
        else some 0 } : Tile .real [Bk, Bd]) = kLoaded := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_vLoaded_raw :
      ({ data := fun i : TileIndex [Bk, Bd] =>
        if k * Bk + (TileShape.dropInsertedIndex [Bk] 1 1 (i.1, 0, PUnit.unit)).1.val < S_k ∧
            (TileShape.dropInsertedIndex [Bd] 0 1 (0, i.2.1, PUnit.unit)).1.val < D then
          some (s.readMem vReg (batch * sVB + headIdx * sVH + (k * Bk + i.1.val) * sVN + i.2.1.val * sVD))
        else some 0 } : Tile .real [Bk, Bd]) = vLoaded := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_kvMask_from_parts :
      ({ data := fun i : TileIndex [Bk, Bd] =>
        kvSeqMask.data (i.1, i.2.1, PUnit.unit) && kvDMask.data (i.1, i.2.1, PUnit.unit) } :
        Tile .bool [Bk, Bd]) = kvMask := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    simp [kvSeqMask, kvDMask, kvMask]
  have h_kLoaded_from_parts :
      ({ data := fun i : TileIndex [Bk, Bd] =>
        if kvSeqMask.data (i.1, i.2.1, PUnit.unit) = Bool.true ∧
            kvDMask.data (i.1, i.2.1, PUnit.unit) = Bool.true then
          some (s.readMem kReg (batch * sKB + headIdx * sKH + (k * Bk + i.1.val) * sKN + i.2.1.val * sKD))
        else some 0 } : Tile .real [Bk, Bd]) = kLoaded := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    simp [kvSeqMask, kvDMask, kLoaded, kBase]
  have h_vLoaded_from_parts :
      ({ data := fun i : TileIndex [Bk, Bd] =>
        if kvSeqMask.data (i.1, i.2.1, PUnit.unit) = Bool.true ∧
            kvDMask.data (i.1, i.2.1, PUnit.unit) = Bool.true then
          some (s.readMem vReg (batch * sVB + headIdx * sVH + (k * Bk + i.1.val) * sVN + i.2.1.val * sVD))
        else some 0 } : Tile .real [Bk, Bd]) = vLoaded := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    simp [kvSeqMask, kvDMask, vLoaded, vBase]
  have h_kvDMask_eta :
      ({ data := fun i : TileIndex [Bk, Bd] => kvDMask.data (i.1, i.2.1, PUnit.unit) } :
        Tile .bool [Bk, Bd]) = kvDMask := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_kLoaded_from_mask :
      ({ data := fun i : TileIndex [Bk, Bd] =>
        if kvMask.data i = Bool.true then
          some (s.readMem kReg (batch * sKB + headIdx * sKH + (k * Bk + i.1.val) * sKN + i.2.1.val * sKD))
        else some 0 } : Tile .real [Bk, Bd]) = kLoaded := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    simp [kvMask, kvSeqMask, kvDMask, kLoaded, kBase]
  have h_vLoaded_from_mask :
      ({ data := fun i : TileIndex [Bk, Bd] =>
        if kvMask.data i = Bool.true then
          some (s.readMem vReg (batch * sVB + headIdx * sVH + (k * Bk + i.1.val) * sVN + i.2.1.val * sVD))
        else some 0 } : Tile .real [Bk, Bd]) = vLoaded := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    simp [kvMask, kvSeqMask, kvDMask, vLoaded, vBase]
  have hq_s8 :
      evalOp (Op.ref .real [M, Bd] "q") s8 = some (Tile.ofReal Qp) := by
    simpa [s8, s7, s6, s5, s4, s3, s2, s1, s0, evalOp_ref, Qp] using hq
  have hk_s8 :
      evalOp (Op.ref .real [Bk, Bd] "k") s8 = some kLoaded := by
    simp [s8, s7]
  have hkt_s8 :
      evalOp ((Op.ref .real [Bk, Bd] "k").transpose) s8 =
        some (Tile.transpose [] kLoaded) := by
    erw [evalOp_transpose]
    simp [hk_s8]
  have h_scoresRaw_eval :
      evalOp ((Op.ref .real [M, Bd] "q").dot (Op.ref .real [Bk, Bd] "k").transpose) s8 =
        some (Tile.dot [] (Tile.ofReal Qp) (Tile.transpose [] kLoaded)) := by
    erw [evalOp_dot]
    erw [hq_s8, hkt_s8]
    rfl
  have h_scoresRaw_eval_exp :
      evalOp ((Op.ref .real [M, Bd] "q").dot (Op.ref .real [Bk, Bd] "k").transpose)
        (((((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD } :
              Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD } :
              Tile .nat [Bk, Bd])).setReg
            "kv_seq_mask" .bool [Bk, Bd] kvSeqMask).setReg
            "kv_d_mask" .bool [Bk, Bd] kvDMask).setReg
            "kv_mask" .bool [Bk, Bd] kvMask).setReg
            "k" .real [Bk, Bd] kLoaded).setReg
            "v" .real [Bk, Bd] vLoaded) =
        some (Tile.dot [] (Tile.ofReal Qp) (Tile.transpose [] kLoaded)) := by
    simpa [s8, s7, s6, s5, s4, s3, s2, s1, s0, offsN, kPtrs, vPtrs, kBase, vBase]
      using h_scoresRaw_eval
  have h_scoresRawScaled_eval_exp :
      evalOp
        (Op.mul .real Broadcast.scalarR
          (Op.dot (batch := []) (M := M) (K := Bd) (N := Bk)
            (Op.ref .real [M, Bd] "q") (Op.ref .real [Bk, Bd] "k").transpose)
          (Op.const scale))
        (((((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD } :
              Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD } :
              Tile .nat [Bk, Bd])).setReg
            "kv_seq_mask" .bool [Bk, Bd] kvSeqMask).setReg
            "kv_d_mask" .bool [Bk, Bd] kvDMask).setReg
            "kv_mask" .bool [Bk, Bd] kvMask).setReg
            "k" .real [Bk, Bd] kLoaded).setReg
            "v" .real [Bk, Bd] vLoaded) =
        some scoresRaw := by
    rw [evalOp_mul]
    erw [h_scoresRaw_eval_exp]
    simp [evalOp, scoresRaw, Tile.bop, Tile.dot]
    ext idx
    rcases idx with ⟨i, j, ⟨⟩⟩
    have hsum :
        (@Finset.sum (Fin Bd) (WithBot ℝ) _ Finset.univ
          (fun x => Option.map (fun b => Qp (i, x, PUnit.unit) * b)
            ((Tile.transpose [] kLoaded).data (x, j, PUnit.unit)))) =
        (@Finset.sum (Fin Bd) (WithBot ℝ) _ Finset.univ
          (fun x => Option.map (fun b => Qp (i, x, PUnit.unit) * b)
            (kLoaded.data (j, x, PUnit.unit)))) := by
      apply Finset.sum_congr rfl
      intro x hx
      rw [Tile.transpose_nil_data]
    rw [hsum]
    simp [NumericDType.mul, WithBot.realMul]
  have h_scoresRawScaled_eval_exp :
      evalOp
        (Op.mul .real Broadcast.scalarR
          (Op.dot (batch := []) (M := M) (K := Bd) (N := Bk)
            (Op.ref .real [M, Bd] "q") (Op.ref .real [Bk, Bd] "k").transpose)
          (Op.const scale))
        (((((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD } :
              Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD } :
              Tile .nat [Bk, Bd])).setReg
            "kv_seq_mask" .bool [Bk, Bd] kvSeqMask).setReg
            "kv_d_mask" .bool [Bk, Bd] kvDMask).setReg
            "kv_mask" .bool [Bk, Bd] kvMask).setReg
            "k" .real [Bk, Bd] kLoaded).setReg
            "v" .real [Bk, Bd] vLoaded) =
        some scoresRaw := by
    rw [evalOp_mul]
    erw [h_scoresRaw_eval_exp]
    simp [evalOp, scoresRaw, Tile.bop, Tile.dot]
    ext idx
    rcases idx with ⟨i, j, ⟨⟩⟩
    have hsum :
        (@Finset.sum (Fin Bd) (WithBot ℝ) _ Finset.univ
          (fun x => Option.map (fun b => Qp (i, x, PUnit.unit) * b)
            ((Tile.transpose [] kLoaded).data (x, j, PUnit.unit)))) =
        (@Finset.sum (Fin Bd) (WithBot ℝ) _ Finset.univ
          (fun x => Option.map (fun b => Qp (i, x, PUnit.unit) * b)
            (kLoaded.data (j, x, PUnit.unit)))) := by
      apply Finset.sum_congr rfl
      intro x hx
      rw [Tile.transpose_nil_data]
    rw [hsum]
    simp [NumericDType.mul, WithBot.realMul]
  have h_offsM_s9 :
      evalOp (Op.expandDim 1 (Op.ref .nat [M] "offs_m")) s9 =
        some (Tile.expandDim 1 (Tile.vec fun i : Fin M => qb * M + i.val)) := by
    simp [s9, s8, s7, s6, s5, s4, s3, s2, s1, s0, evalOp, hoffs_m]
  have h_offsN_s9 :
      evalOp (Op.expandDim 0 (Op.ref .nat [Bk] "offs_n")) s9 =
        some (Tile.expandDim 0 offsN) := by
    simp [s9, s8, s7, s6, s5, s4, s3, s2, s1, s0, evalOp, offsN]
  have h_scoreMask_raw :
      ({ data := fun i : TileIndex [M, Bk] =>
        decide (k * Bk + (TileShape.dropInsertedIndex [Bk] 0 1
          (0, i.2.1, PUnit.unit)).1.val < S_k) } : Tile .bool [M, Bk]) = scoreMask := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_scoreMask_eval :
      evalOp
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
              (Op.constNat 0))
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
          (Op.constNat S_k)) s9 =
        some scoreMask := by
    simp only [evalOp_lt, evalOp_add, evalOp_mul, evalOp_constNat,
      Option.bind_eq_bind, Option.bind_some]
    erw [h_offsM_s9, h_offsN_s9]
    simpa [Option.bind, Tile.bop, Tile.cop, Tile.expandDim,
      TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
      ComparableDType.lt] using congrArg some h_scoreMask_raw
  have h_scoreMask_eval_exp :
      evalOp
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
              (Op.constNat 0))
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
          (Op.constNat S_k))
        ((((((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, Bd])).setReg
            "kv_seq_mask" .bool [Bk, Bd] kvSeqMask).setReg
            "kv_d_mask" .bool [Bk, Bd] kvDMask).setReg
            "kv_mask" .bool [Bk, Bd] kvMask).setReg
            "k" .real [Bk, Bd] kLoaded).setReg
            "v" .real [Bk, Bd] vLoaded).setReg
            "scores_raw" .real [M, Bk] scoresRaw) =
        some scoreMask := by
    simpa [s9, s8, s7, s6, s5, s4, s3, s2, s1, s0, offsN, kPtrs, vPtrs,
      kvSeqMask, kvDMask, kvMask, kLoaded, vLoaded, scoresRaw, kBase, vBase]
      using h_scoreMask_eval
  have h_scores_raw :
      ({ data := fun idx : TileIndex [M, Bk] =>
        if k * Bk + (TileShape.dropInsertedIndex [Bk] 0 1
            (0, idx.2.1, PUnit.unit)).1.val < S_k then
          Option.map (fun a => a * scale)
            (@Finset.sum (Fin Bd) (WithBot ℝ) _ Finset.univ
              (fun x : Fin Bd => Option.map (fun b => Qp (idx.1, x, PUnit.unit) * b)
                (if k * Bk + idx.2.1.val < S_k ∧ x.val < D then
                  some (s.readMem kReg (kBase + (k * Bk + idx.2.1.val) * sKN + x.val * sKD))
                else some 0 : WithBot ℝ)))
        else (none : WithBot ℝ) } : Tile .real [M, Bk]) = scores := by
    ext idx
    rcases idx with ⟨i, j, ⟨⟩⟩
    have hdrop :
        (TileShape.dropInsertedIndex [Bk] 0 1 (0, j, PUnit.unit)).1.val = j.val := rfl
    by_cases hj : k * Bk + j.val < S_k
    · simp [scores, scoresRaw, kLoaded, kBase, hdrop, hj]
    · simp [scores, scoresRaw, kLoaded, kBase, hdrop, hj]
  have h_scores_from_mask :
      ({ data := fun idx : TileIndex [M, Bk] =>
        if scoreMask.data idx = Bool.true then scoresRaw.data idx else none } :
        Tile .real [M, Bk]) = scores := by
    ext idx
    rcases idx with ⟨i, j, ⟨⟩⟩
    by_cases hj : k * Bk + j.val < S_k
    · simp [scoreMask, scores, hj]
    · simp [scoreMask, scores, hj]
  have h_scores_eval_exp :
      evalOp
        (Op.where
          (Op.ref .bool [M, Bk] "score_mask")
          (Op.ref .real [M, Bk] "scores_raw")
          (Op.broadcast Op.negInf [M, Bk]))
        (((((((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, Bd])).setReg
            "kv_seq_mask" .bool [Bk, Bd] kvSeqMask).setReg
            "kv_d_mask" .bool [Bk, Bd] kvDMask).setReg
            "kv_mask" .bool [Bk, Bd] kvMask).setReg
            "k" .real [Bk, Bd] kLoaded).setReg
            "v" .real [Bk, Bd] vLoaded).setReg
            "scores_raw" .real [M, Bk] scoresRaw).setReg
            "score_mask" .bool [M, Bk] scoreMask) =
        some scores := by
    simp [evalOp, Tile.select, h_scores_from_mask]
  have h_mBlock_eval :
      evalOp (Op.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩
        (keepDims := Bool.false) (Op.ref .real [M, Bk] "scores")) s11 =
        some mBlock := by
    erw [evalOp_reduceMax]
    simp [s11, s10]
    simp [Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
      TileShape.insertAxisIndex, TileShape.dropInsertedIndex, hBk, mBlock]
    funext idx
    rfl
  have h_kvSeqMask_eval :
      evalOp
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d"))
              (Op.constNat 0)))
          (Op.constNat S_k)) s3 =
        some kvSeqMask := by
    simp only [evalOp_lt, evalOp_add, evalOp_mul, evalOp_constNat,
      Option.bind_eq_bind, Option.bind_some]
    erw [h_offsN_s3, h_offsD_s3]
    simpa [Option.bind, offsN, Tile.bop, Tile.cop, Tile.expandDim,
      TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
      ComparableDType.lt] using congrArg some h_kvSeqMask_raw
  have h_kvDMask_eval :
      evalOp
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
              (Op.constNat 0))
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d")))
          (Op.constNat D)) s4 =
        some kvDMask := by
    simp only [evalOp_lt, evalOp_add, evalOp_mul, evalOp_constNat,
      Option.bind_eq_bind, Option.bind_some]
    erw [h_offsN_s4, h_offsD_s4]
    simpa [Option.bind, offsN, Tile.bop, Tile.cop, Tile.expandDim,
      TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
      ComparableDType.lt] using congrArg some h_kvDMask_raw
  have h_kvMask_eval :
      evalOp
        (Op.boolAnd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .bool [Bk, Bd] "kv_seq_mask")
          (Op.ref .bool [Bk, Bd] "kv_d_mask")) s5 =
        some kvMask := by
    simpa [evalOp, s5, s4, s3, s2, s1, s0, kvMask, kvSeqMask, kvDMask,
      offsN, kPtrs, vPtrs, kBase, vBase, Tile.bop]
      using congrArg some h_kvMask_raw
  have h_kvSeqMask_eval_exp :
      evalOp
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d"))
              (Op.constNat 0)))
          (Op.constNat S_k))
        ((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, Bd])) =
        some kvSeqMask := by
    simpa [s3, s2, s1, s0, offsN, kPtrs, vPtrs, kBase, vBase] using h_kvSeqMask_eval
  have h_kvDMask_eval_exp :
      evalOp
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
              (Op.constNat 0))
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d")))
          (Op.constNat D))
        (((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, Bd])).setReg
            "kv_seq_mask" .bool [Bk, Bd] kvSeqMask) =
        some kvDMask := by
    simpa [s4, s3, s2, s1, s0, offsN, kPtrs, vPtrs, kvSeqMask, kBase, vBase]
      using h_kvDMask_eval
  have h_kvMask_eval_exp :
      evalOp
        (Op.boolAnd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .bool [Bk, Bd] "kv_seq_mask")
          (Op.ref .bool [Bk, Bd] "kv_d_mask"))
        ((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, Bd])).setReg
            "kv_seq_mask" .bool [Bk, Bd] kvSeqMask).setReg
            "kv_d_mask" .bool [Bk, Bd] kvDMask) =
        some kvMask := by
    simpa [s5, s4, s3, s2, s1, s0, offsN, kPtrs, vPtrs, kvSeqMask, kvDMask,
      kBase, vBase] using h_kvMask_eval
  have h_kLoaded_eval :
      evalOp
        (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, Bd] "k_ptrs"))
          (MaskOpt.maskOther (Op.ref .bool [Bk, Bd] "kv_mask")
            (Op.broadcast (Op.const 0) [Bk, Bd]))) s6 =
        some kLoaded := by
    simp [evalOp, s6, s5, s4, s3, s2, s1, s0, kPtrs, kvMask, kLoaded,
      kBase, Region.cast, Tile.bop, Tile.expandDim, h_kLoaded_from_mask]
  have h_vLoaded_eval :
      evalOp
        (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, Bd] "v_ptrs"))
          (MaskOpt.maskOther (Op.ref .bool [Bk, Bd] "kv_mask")
            (Op.broadcast (Op.const 0) [Bk, Bd]))) s7 =
        some vLoaded := by
    simp [evalOp, s7, s6, s5, s4, s3, s2, s1, s0, vPtrs, kvMask, vLoaded,
      vBase, Region.cast, Tile.bop, Tile.expandDim, h_vLoaded_from_mask]
  have h_offsN_step :
      stepStmt
        (Stmt.assign .nat [Bk] "offs_n"
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil
              (Op.ref .nat [] "n")
              (Op.constNat Bk))
            (Op.arange Bk))) s0 = some s1 := by
    simp [stepStmt, s1, s0, offsN, Tile.bop, NumericDType.add, NumericDType.mul]
    rfl
  have h_offsN_eval :
      evalOp
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil
            (Op.ref .nat [] "n")
            (Op.constNat Bk))
          (Op.arange Bk))
        (s.setReg "n" .nat [] (Tile.scalar k)) =
      some ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk]) := by
    simp [evalOp, Tile.bop, NumericDType.add, NumericDType.mul]
  have h_mNew_eval :
      evalOp
        (Op.max2 (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "m_i")
          (Op.ref .real [M] "m_block")) s12 = some mNew := by
    simp [evalOp, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      mNew, Qp, Kp, Tile.bop, hm]
  have h_alpha_eval :
      evalOp
        (Op.exp
          (Op.sub .real (Broadcast.consSame Broadcast.nil)
            (Op.ref .real [M] "m_i")
            (Op.ref .real [M] "m_new"))) s13 = some alpha := by
    simp [evalOp_exp, evalOp_sub, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      alpha, mNew, Qp, Kp, Tile.bop, Tile.uop, NumericDType.sub, hm]
  have h_p_eval :
      evalOp
        (Op.exp
          (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (Op.ref .real [M, Bk] "scores")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "m_new")))) s14 =
        some p := by
    simp only [evalOp_exp, evalOp_sub, Option.bind_eq_bind, Option.bind_some]
    unfold evalOp
    simp [evalOp_expandDim, evalOp_expandDim_ref, evalOp_ref,
      s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      p, mNew, Tile.bop, Tile.uop, Tile.expandDim, NumericDType.sub]
    rfl
  have h_lNew_eval :
      evalOp
        (Op.add .real (Broadcast.consSame Broadcast.nil)
          (Op.mul .real (Broadcast.consSame Broadcast.nil)
            (Op.ref .real [M] "alpha")
            (Op.ref .real [M] "l_i"))
          (Op.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
            (Op.ref .real [M, Bk] "p"))) s15 = some lNew := by
    simp [evalOp_add, evalOp_mul, evalOp_reduceSum,
      s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      lNew, alpha, p, Qp, Kp, Tile.bop, Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      hBk, NumericDType.add, NumericDType.mul, hl]
    unfold evalOp
    simp [s15, lNew, alpha, p, Qp, Kp, Tile.bop, Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      hBk, NumericDType.add, NumericDType.mul, WithBot.realAdd, WithBot.realMul]
    funext idx
    rfl
  have h_oNew_eval :
      evalOp
        (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "alpha"))
            (Op.ref .real [M, Bd] "o_acc"))
          (Op.dot (batch := []) (M := M) (K := Bk) (N := Bd)
            (Op.ref .real [M, Bk] "p")
            (Op.ref .real [Bk, Bd] "v"))) s16 = some oNew := by
    simp [evalOp_add, evalOp_mul, evalOp_dot,
      s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      oNew, alpha, p, Qp, Kp, Vp, Tile.bop, Tile.dot, Tile.expandDim,
      TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, ho]
    unfold evalOp
    simp [s16, oNew, alpha, p, Qp, Kp, Vp, Tile.bop, Tile.dot, Tile.expandDim,
      TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul]
    rfl
  refine ⟨s', ?_, ?_⟩
  · simp only [fa1LoopBodyStridedBoundaryD, stepStmts, stepStmt]
    rw [h_offsN_eval]
    dsimp
    erw [h_kPtrs_eval]
    simp only [Option.bind_some]
    erw [h_vPtrs_eval]
    simp only [Option.bind_some]
    erw [h_kvSeqMask_eval_exp]
    simp only [Option.bind_some]
    erw [h_kvDMask_eval_exp]
    simp only [Option.bind_some]
    erw [h_kvMask_eval_exp]
    simp only [Option.bind_some]
    erw [h_kLoaded_eval]
    simp only [Option.bind_some]
    erw [h_vLoaded_eval]
    simp only [Option.bind_some]
    erw [h_scoresRawScaled_eval_exp]
    simp only [Option.bind_some]
    erw [h_scoreMask_eval_exp]
    simp only [Option.bind_some]
    erw [h_scores_eval_exp]
    simp only [Option.bind_some]
    erw [h_mBlock_eval]
    simp only [Option.bind_some]
    erw [h_mNew_eval]
    simp only [Option.bind_some]
    erw [h_alpha_eval]
    simp only [Option.bind_some]
    erw [h_p_eval]
    simp only [Option.bind_some]
    erw [h_lNew_eval]
    simp only [Option.bind_some]
    erw [h_oNew_eval]
    simp only [Option.bind_some]
    simp [evalOp, s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      offsN, kPtrs, vPtrs, kBase, vBase, Tile.vec]
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simpa [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hpids0
    · simpa [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hpids1
    · simpa [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hpids2
    · simpa [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hpid_qb
    · simpa [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hpid_h
    · simpa [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hpid_b
    · simpa [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hq_base
    · simpa [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hk_base
    · simpa [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hv_base
    · simpa [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using ho_base
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
  have h_kPtrs_eval :=
    fa1_boundaryD_eval_k_ptrs s Bk Bd k batch headIdx sKB sKH sKN sKD hk_base hoffs_d
  have h_vPtrs_eval :=
    fa1_boundaryD_eval_v_ptrs s Bk Bd k batch headIdx sVB sVH sVN sVD kPtrs hv_base hoffs_d
  have h_offsN_s3 :
      evalOp (Op.expandDim 1 (Op.ref .nat [Bk] "offs_n")) s3 =
        some (Tile.expandDim 1 offsN) := by
    simp [s3, s2, s1, s0, offsN]
  have h_offsD_s3 :
      evalOp (Op.expandDim 0 (Op.ref .nat [Bd] "offs_d")) s3 =
        some (Tile.expandDim 0 (Tile.vec fun d : Fin Bd => d.val)) := by
    simp [s3, s2, s1, s0, offsN, hoffs_d]
  have h_offsN_s3_exp :
      evalOp (Op.expandDim 1 (Op.ref .nat [Bk] "offs_n"))
        ((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, Bd])) =
        some (Tile.expandDim 1
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])) := by
    simpa [s3, s2, s1, s0, offsN, kPtrs, vPtrs, kBase, vBase] using h_offsN_s3
  have h_offsD_s3_exp :
      evalOp (Op.expandDim 0 (Op.ref .nat [Bd] "offs_d"))
        ((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, Bd])) =
        some (Tile.expandDim 0 (Tile.vec fun d : Fin Bd => d.val)) := by
    simpa [s3, s2, s1, s0, offsN, kPtrs, vPtrs, kBase, vBase] using h_offsD_s3
  have h_offsN_s4 :
      evalOp (Op.expandDim 1 (Op.ref .nat [Bk] "offs_n")) s4 =
        some (Tile.expandDim 1 offsN) := by
    simp [s4, s3, s2, s1, s0, offsN]
  have h_offsD_s4 :
      evalOp (Op.expandDim 0 (Op.ref .nat [Bd] "offs_d")) s4 =
        some (Tile.expandDim 0 (Tile.vec fun d : Fin Bd => d.val)) := by
    simp [s4, s3, s2, s1, s0, offsN, hoffs_d]
  have h_offsN_s4_exp :
      evalOp (Op.expandDim 1 (Op.ref .nat [Bk] "offs_n"))
        (((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, Bd])).setReg
            "kv_seq_mask" .bool [Bk, Bd]
            ({ data := fun i : TileIndex [Bk, Bd] =>
              decide (k * Bk + (TileShape.dropInsertedIndex [Bk] 1 1 (i.1, 0, PUnit.unit)).1.val < S_k) } :
              Tile .bool [Bk, Bd])) =
        some (Tile.expandDim 1
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])) := by
    simpa [s4, s3, s2, s1, s0, offsN, kPtrs, vPtrs, kvSeqMask, kBase, vBase,
      Tile.expandDim, Tile.bop, TileShape.dropInsertedIndex, NumericDType.add,
      NumericDType.mul] using h_offsN_s4
  have h_offsD_s4_exp :
      evalOp (Op.expandDim 0 (Op.ref .nat [Bd] "offs_d"))
        (((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, Bd])).setReg
            "kv_seq_mask" .bool [Bk, Bd]
            ({ data := fun i : TileIndex [Bk, Bd] =>
              decide (k * Bk + (TileShape.dropInsertedIndex [Bk] 1 1 (i.1, 0, PUnit.unit)).1.val < S_k) } :
              Tile .bool [Bk, Bd])) =
        some (Tile.expandDim 0 (Tile.vec fun d : Fin Bd => d.val)) := by
    simpa [s4, s3, s2, s1, s0, offsN, kPtrs, vPtrs, kvSeqMask, kBase, vBase,
      Tile.expandDim, Tile.bop, TileShape.dropInsertedIndex, NumericDType.add,
      NumericDType.mul] using h_offsD_s4
  have h_kvSeqMask_raw :
      ({ data := fun i : TileIndex [Bk, Bd] =>
        decide (k * Bk + (TileShape.dropInsertedIndex [Bk] 1 1 (i.1, 0, PUnit.unit)).1.val < S_k) } :
        Tile .bool [Bk, Bd]) = kvSeqMask := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_kvDMask_raw :
      ({ data := fun i : TileIndex [Bk, Bd] =>
        decide ((TileShape.dropInsertedIndex [Bd] 0 1 (0, i.2.1, PUnit.unit)).1.val < D) } :
        Tile .bool [Bk, Bd]) = kvDMask := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_kvSeqMask_lane (i : TileIndex [Bk, Bd]) :
      decide (k * Bk + (TileShape.dropInsertedIndex [Bk] 1 1 (i.1, 0, PUnit.unit)).1.val < S_k) =
        kvSeqMask.data (i.1, i.2.1, PUnit.unit) := by
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_kvDMask_lane (i : TileIndex [Bk, Bd]) :
      decide ((TileShape.dropInsertedIndex [Bd] 0 1 (0, i.2.1, PUnit.unit)).1.val < D) =
        kvDMask.data (i.1, i.2.1, PUnit.unit) := by
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_kvMask_raw :
      ({ data := fun i : TileIndex [Bk, Bd] =>
        decide (k * Bk + (TileShape.dropInsertedIndex [Bk] 1 1 (i.1, 0, PUnit.unit)).1.val < S_k) &&
          decide ((TileShape.dropInsertedIndex [Bd] 0 1 (0, i.2.1, PUnit.unit)).1.val < D) } :
        Tile .bool [Bk, Bd]) = kvMask := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_kLoaded_raw :
      ({ data := fun i : TileIndex [Bk, Bd] =>
        if k * Bk + (TileShape.dropInsertedIndex [Bk] 1 1 (i.1, 0, PUnit.unit)).1.val < S_k ∧
            (TileShape.dropInsertedIndex [Bd] 0 1 (0, i.2.1, PUnit.unit)).1.val < D then
          some (s.readMem kReg (batch * sKB + headIdx * sKH + (k * Bk + i.1.val) * sKN + i.2.1.val * sKD))
        else some 0 } : Tile .real [Bk, Bd]) = kLoaded := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_vLoaded_raw :
      ({ data := fun i : TileIndex [Bk, Bd] =>
        if k * Bk + (TileShape.dropInsertedIndex [Bk] 1 1 (i.1, 0, PUnit.unit)).1.val < S_k ∧
            (TileShape.dropInsertedIndex [Bd] 0 1 (0, i.2.1, PUnit.unit)).1.val < D then
          some (s.readMem vReg (batch * sVB + headIdx * sVH + (k * Bk + i.1.val) * sVN + i.2.1.val * sVD))
        else some 0 } : Tile .real [Bk, Bd]) = vLoaded := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_kvMask_from_parts :
      ({ data := fun i : TileIndex [Bk, Bd] =>
        kvSeqMask.data (i.1, i.2.1, PUnit.unit) && kvDMask.data (i.1, i.2.1, PUnit.unit) } :
        Tile .bool [Bk, Bd]) = kvMask := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    simp [kvSeqMask, kvDMask, kvMask]
  have h_kLoaded_from_parts :
      ({ data := fun i : TileIndex [Bk, Bd] =>
        if kvSeqMask.data (i.1, i.2.1, PUnit.unit) = Bool.true ∧
            kvDMask.data (i.1, i.2.1, PUnit.unit) = Bool.true then
          some (s.readMem kReg (batch * sKB + headIdx * sKH + (k * Bk + i.1.val) * sKN + i.2.1.val * sKD))
        else some 0 } : Tile .real [Bk, Bd]) = kLoaded := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    simp [kvSeqMask, kvDMask, kLoaded, kBase]
  have h_vLoaded_from_parts :
      ({ data := fun i : TileIndex [Bk, Bd] =>
        if kvSeqMask.data (i.1, i.2.1, PUnit.unit) = Bool.true ∧
            kvDMask.data (i.1, i.2.1, PUnit.unit) = Bool.true then
          some (s.readMem vReg (batch * sVB + headIdx * sVH + (k * Bk + i.1.val) * sVN + i.2.1.val * sVD))
        else some 0 } : Tile .real [Bk, Bd]) = vLoaded := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    simp [kvSeqMask, kvDMask, vLoaded, vBase]
  have h_kvDMask_eta :
      ({ data := fun i : TileIndex [Bk, Bd] => kvDMask.data (i.1, i.2.1, PUnit.unit) } :
        Tile .bool [Bk, Bd]) = kvDMask := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_kLoaded_from_mask :
      ({ data := fun i : TileIndex [Bk, Bd] =>
        if kvMask.data i = Bool.true then
          some (s.readMem kReg (batch * sKB + headIdx * sKH + (k * Bk + i.1.val) * sKN + i.2.1.val * sKD))
        else some 0 } : Tile .real [Bk, Bd]) = kLoaded := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    simp [kvMask, kvSeqMask, kvDMask, kLoaded, kBase]
  have h_vLoaded_from_mask :
      ({ data := fun i : TileIndex [Bk, Bd] =>
        if kvMask.data i = Bool.true then
          some (s.readMem vReg (batch * sVB + headIdx * sVH + (k * Bk + i.1.val) * sVN + i.2.1.val * sVD))
        else some 0 } : Tile .real [Bk, Bd]) = vLoaded := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    simp [kvMask, kvSeqMask, kvDMask, vLoaded, vBase]
  have hq_s8 :
      evalOp (Op.ref .real [M, Bd] "q") s8 = some (Tile.ofReal Qp) := by
    simpa [s8, s7, s6, s5, s4, s3, s2, s1, s0, evalOp_ref, Qp] using hq
  have hk_s8 :
      evalOp (Op.ref .real [Bk, Bd] "k") s8 = some kLoaded := by
    simp [s8, s7]
  have hkt_s8 :
      evalOp ((Op.ref .real [Bk, Bd] "k").transpose) s8 =
        some (Tile.transpose [] kLoaded) := by
    erw [evalOp_transpose]
    simp [hk_s8]
  have h_scoresRaw_eval :
      evalOp ((Op.ref .real [M, Bd] "q").dot (Op.ref .real [Bk, Bd] "k").transpose) s8 =
        some (Tile.dot [] (Tile.ofReal Qp) (Tile.transpose [] kLoaded)) := by
    erw [evalOp_dot]
    erw [hq_s8, hkt_s8]
    rfl
  have h_scoresRaw_eval_exp :
      evalOp ((Op.ref .real [M, Bd] "q").dot (Op.ref .real [Bk, Bd] "k").transpose)
        (((((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD } :
              Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD } :
              Tile .nat [Bk, Bd])).setReg
            "kv_seq_mask" .bool [Bk, Bd] kvSeqMask).setReg
            "kv_d_mask" .bool [Bk, Bd] kvDMask).setReg
            "kv_mask" .bool [Bk, Bd] kvMask).setReg
            "k" .real [Bk, Bd] kLoaded).setReg
            "v" .real [Bk, Bd] vLoaded) =
        some (Tile.dot [] (Tile.ofReal Qp) (Tile.transpose [] kLoaded)) := by
    simpa [s8, s7, s6, s5, s4, s3, s2, s1, s0, offsN, kPtrs, vPtrs, kBase, vBase]
      using h_scoresRaw_eval
  have h_scoresRawScaled_eval_exp :
      evalOp
        (Op.mul .real Broadcast.scalarR
          (Op.dot (batch := []) (M := M) (K := Bd) (N := Bk)
            (Op.ref .real [M, Bd] "q") (Op.ref .real [Bk, Bd] "k").transpose)
          (Op.const scale))
        (((((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD } :
              Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD } :
              Tile .nat [Bk, Bd])).setReg
            "kv_seq_mask" .bool [Bk, Bd] kvSeqMask).setReg
            "kv_d_mask" .bool [Bk, Bd] kvDMask).setReg
            "kv_mask" .bool [Bk, Bd] kvMask).setReg
            "k" .real [Bk, Bd] kLoaded).setReg
            "v" .real [Bk, Bd] vLoaded) =
        some scoresRaw := by
    rw [evalOp_mul]
    erw [h_scoresRaw_eval_exp]
    simp [evalOp, scoresRaw, Tile.bop, Tile.dot]
    ext idx
    rcases idx with ⟨i, j, ⟨⟩⟩
    have hsum :
        (@Finset.sum (Fin Bd) (WithBot ℝ) _ Finset.univ
          (fun x => Option.map (fun b => Qp (i, x, PUnit.unit) * b)
            ((Tile.transpose [] kLoaded).data (x, j, PUnit.unit)))) =
        (@Finset.sum (Fin Bd) (WithBot ℝ) _ Finset.univ
          (fun x => Option.map (fun b => Qp (i, x, PUnit.unit) * b)
            (kLoaded.data (j, x, PUnit.unit)))) := by
      apply Finset.sum_congr rfl
      intro x hx
      rw [Tile.transpose_nil_data]
    rw [hsum]
    simp [NumericDType.mul, WithBot.realMul]
  have h_offsM_s9 :
      evalOp (Op.expandDim 1 (Op.ref .nat [M] "offs_m")) s9 =
        some (Tile.expandDim 1 (Tile.vec fun i : Fin M => qb * M + i.val)) := by
    simp [s9, s8, s7, s6, s5, s4, s3, s2, s1, s0, evalOp, hoffs_m]
  have h_offsN_s9 :
      evalOp (Op.expandDim 0 (Op.ref .nat [Bk] "offs_n")) s9 =
        some (Tile.expandDim 0 offsN) := by
    simp [s9, s8, s7, s6, s5, s4, s3, s2, s1, s0, evalOp, offsN]
  have h_scoreMask_raw :
      ({ data := fun i : TileIndex [M, Bk] =>
        decide (k * Bk + (TileShape.dropInsertedIndex [Bk] 0 1
          (0, i.2.1, PUnit.unit)).1.val < S_k) } : Tile .bool [M, Bk]) = scoreMask := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_causal_raw :
      ({ data := fun i : TileIndex [M, Bk] =>
        decide (k * Bk + (TileShape.dropInsertedIndex [Bk] 0 1
          (0, i.2.1, PUnit.unit)).1.val ≤
            qb * M + (TileShape.dropInsertedIndex [M] 1 1 (i.1, 0, PUnit.unit)).1.val) } :
        Tile .bool [M, Bk]) = causal := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_causalScores_raw :
      ({ data := fun idx : TileIndex [M, Bk] =>
        if k * Bk + idx.2.1.val ≤ qStart + idx.1.val then
          Option.map (fun a => a * scale)
            (@Finset.sum (Fin Bd) (WithBot ℝ) _ Finset.univ
              (fun x : Fin Bd => Option.map (fun b => Qp (idx.1, x, PUnit.unit) * b)
                (if k * Bk + idx.2.1.val < S_k ∧ x.val < D then
                  some (s.readMem kReg (kBase + (k * Bk + idx.2.1.val) * sKN + x.val * sKD))
                else some 0 : WithBot ℝ)))
        else (none : WithBot ℝ) } : Tile .real [M, Bk]) = causalScores := by
    ext idx
    rcases idx with ⟨i, j, ⟨⟩⟩
    have hdropN :
        (TileShape.dropInsertedIndex [Bk] 0 1 (0, j, PUnit.unit)).1.val = j.val := rfl
    have hdropM :
        (TileShape.dropInsertedIndex [M] 1 1 (i, 0, PUnit.unit)).1.val = i.val := rfl
    by_cases hj : k * Bk + j.val ≤ qStart + i.val
    · simp [causalScores, scoresRaw, kLoaded, kBase, hdropN, hdropM, hj]
    · simp [causalScores, scoresRaw, kLoaded, kBase, hdropN, hdropM, hj]
  have h_causalScores_from_mask :
      ({ data := fun idx : TileIndex [M, Bk] =>
        if causal.data idx = Bool.true then scoresRaw.data idx else none } :
        Tile .real [M, Bk]) = causalScores := by
    ext idx
    rcases idx with ⟨i, j, ⟨⟩⟩
    by_cases hj : k * Bk + j.val ≤ qStart + i.val
    · simp [causal, causalScores, hj]
    · simp [causal, causalScores, hj]
  have h_scores_raw :
      ({ data := fun idx : TileIndex [M, Bk] =>
        if k * Bk + (TileShape.dropInsertedIndex [Bk] 0 1
            (0, idx.2.1, PUnit.unit)).1.val < S_k then
          if k * Bk + idx.2.1.val ≤ qStart + idx.1.val then
            Option.map (fun a => a * scale)
              (@Finset.sum (Fin Bd) (WithBot ℝ) _ Finset.univ
                (fun x : Fin Bd => Option.map (fun b => Qp (idx.1, x, PUnit.unit) * b)
                  (if k * Bk + idx.2.1.val < S_k ∧ x.val < D then
                    some (s.readMem kReg (kBase + (k * Bk + idx.2.1.val) * sKN + x.val * sKD))
                  else some 0 : WithBot ℝ)))
          else (none : WithBot ℝ)
        else (none : WithBot ℝ) } : Tile .real [M, Bk]) = scores := by
    ext idx
    rcases idx with ⟨i, j, ⟨⟩⟩
    have hdrop :
        (TileShape.dropInsertedIndex [Bk] 0 1 (0, j, PUnit.unit)).1.val = j.val := rfl
    by_cases hSeq : k * Bk + j.val < S_k
    · by_cases hCausal : k * Bk + j.val ≤ qStart + i.val
      · simp [scores, causalScores, scoresRaw, kLoaded, kBase,
          hdrop, hSeq, hCausal]
      · simp [scores, causalScores, scoresRaw, kLoaded, kBase,
          hdrop, hSeq, hCausal]
    · simp [scores, causalScores, scoresRaw, kLoaded, kBase,
        hdrop, hSeq]
  have h_scores_from_mask :
      ({ data := fun idx : TileIndex [M, Bk] =>
        if scoreMask.data idx = Bool.true then causalScores.data idx else none } :
        Tile .real [M, Bk]) = scores := by
    ext idx
    rcases idx with ⟨i, j, ⟨⟩⟩
    by_cases hj : k * Bk + j.val < S_k
    · simp [scoreMask, scores, hj]
    · simp [scoreMask, scores, hj]
  have h_kvSeqMask_eval :
      evalOp
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d"))
              (Op.constNat 0)))
          (Op.constNat S_k)) s3 =
        some kvSeqMask := by
    simp only [evalOp_lt, evalOp_add, evalOp_mul, evalOp_constNat,
      Option.bind_eq_bind, Option.bind_some]
    erw [h_offsN_s3, h_offsD_s3]
    simpa [Option.bind, offsN, Tile.bop, Tile.cop, Tile.expandDim,
      TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
      ComparableDType.lt] using congrArg some h_kvSeqMask_raw
  have h_kvDMask_eval :
      evalOp
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
              (Op.constNat 0))
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d")))
          (Op.constNat D)) s4 =
        some kvDMask := by
    simp only [evalOp_lt, evalOp_add, evalOp_mul, evalOp_constNat,
      Option.bind_eq_bind, Option.bind_some]
    erw [h_offsN_s4, h_offsD_s4]
    simpa [Option.bind, offsN, Tile.bop, Tile.cop, Tile.expandDim,
      TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
      ComparableDType.lt] using congrArg some h_kvDMask_raw
  have h_kvMask_eval :
      evalOp
        (Op.boolAnd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .bool [Bk, Bd] "kv_seq_mask")
          (Op.ref .bool [Bk, Bd] "kv_d_mask")) s5 =
        some kvMask := by
    simpa [evalOp, s5, s4, s3, s2, s1, s0, kvMask, kvSeqMask, kvDMask,
      offsN, kPtrs, vPtrs, kBase, vBase, Tile.bop]
      using congrArg some h_kvMask_raw
  have h_kvSeqMask_eval_exp :
      evalOp
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d"))
              (Op.constNat 0)))
          (Op.constNat S_k))
        ((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, Bd])) =
        some kvSeqMask := by
    simpa [s3, s2, s1, s0, offsN, kPtrs, vPtrs, kBase, vBase] using h_kvSeqMask_eval
  have h_kvDMask_eval_exp :
      evalOp
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
              (Op.constNat 0))
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bd] "offs_d")))
          (Op.constNat D))
        (((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, Bd])).setReg
            "kv_seq_mask" .bool [Bk, Bd] kvSeqMask) =
        some kvDMask := by
    simpa [s4, s3, s2, s1, s0, offsN, kPtrs, vPtrs, kvSeqMask, kBase, vBase]
      using h_kvDMask_eval
  have h_kvMask_eval_exp :
      evalOp
        (Op.boolAnd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .bool [Bk, Bd] "kv_seq_mask")
          (Op.ref .bool [Bk, Bd] "kv_d_mask"))
        ((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, Bd])).setReg
            "kv_seq_mask" .bool [Bk, Bd] kvSeqMask).setReg
            "kv_d_mask" .bool [Bk, Bd] kvDMask) =
        some kvMask := by
    simpa [s5, s4, s3, s2, s1, s0, offsN, kPtrs, vPtrs, kvSeqMask, kvDMask,
      kBase, vBase] using h_kvMask_eval
  have h_kLoaded_eval :
      evalOp
        (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, Bd] "k_ptrs"))
          (MaskOpt.maskOther (Op.ref .bool [Bk, Bd] "kv_mask")
            (Op.broadcast (Op.const 0) [Bk, Bd]))) s6 =
        some kLoaded := by
    simp [evalOp, s6, s5, s4, s3, s2, s1, s0, kPtrs, kvMask, kLoaded,
      kBase, Region.cast, Tile.bop, Tile.expandDim, h_kLoaded_from_mask]
  have h_vLoaded_eval :
      evalOp
        (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, Bd] "v_ptrs"))
          (MaskOpt.maskOther (Op.ref .bool [Bk, Bd] "kv_mask")
            (Op.broadcast (Op.const 0) [Bk, Bd]))) s7 =
        some vLoaded := by
    simp [evalOp, s7, s6, s5, s4, s3, s2, s1, s0, vPtrs, kvMask, vLoaded,
      vBase, Region.cast, Tile.bop, Tile.expandDim, h_vLoaded_from_mask]
  have h_causal_eval :
      evalOp
        (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n"))) s9 =
        some causal := by
    simp only [evalOp, Option.bind_eq_bind, Option.bind_some]
    erw [h_offsM_s9, h_offsN_s9]
    simp [Option.bind, qStart, offsN, causal, Tile.cop, Tile.expandDim,
      TileShape.dropInsertedIndex, ComparableDType.ge]
    ext idx
    rcases idx with ⟨i, j, ⟨⟩⟩
    rfl
  have h_causal_eval_exp :
      evalOp
        (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
        ((((((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, Bd])).setReg
            "kv_seq_mask" .bool [Bk, Bd] kvSeqMask).setReg
            "kv_d_mask" .bool [Bk, Bd] kvDMask).setReg
            "kv_mask" .bool [Bk, Bd] kvMask).setReg
            "k" .real [Bk, Bd] kLoaded).setReg
            "v" .real [Bk, Bd] vLoaded).setReg
            "scores_raw" .real [M, Bk] scoresRaw) =
        some causal := by
    simpa [s9, s8, s7, s6, s5, s4, s3, s2, s1, s0, offsN, kPtrs, vPtrs,
      kvSeqMask, kvDMask, kvMask, kLoaded, vLoaded, scoresRaw, kBase, vBase]
      using h_causal_eval
  have h_causalScores_eval_exp :
      evalOp
        (Op.where
          (Op.ref .bool [M, Bk] "causal")
          (Op.ref .real [M, Bk] "scores_raw")
          (Op.broadcast Op.negInf [M, Bk]))
        (((((((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, Bd])).setReg
            "kv_seq_mask" .bool [Bk, Bd] kvSeqMask).setReg
            "kv_d_mask" .bool [Bk, Bd] kvDMask).setReg
            "kv_mask" .bool [Bk, Bd] kvMask).setReg
            "k" .real [Bk, Bd] kLoaded).setReg
            "v" .real [Bk, Bd] vLoaded).setReg
            "scores_raw" .real [M, Bk] scoresRaw).setReg
            "causal" .bool [M, Bk] causal) =
        some causalScores := by
    simp [evalOp, Tile.select, h_causalScores_from_mask]
  have h_offsM_s11 :
      evalOp (Op.expandDim 1 (Op.ref .nat [M] "offs_m")) s11 =
        some (Tile.expandDim 1 (Tile.vec fun i : Fin M => qb * M + i.val)) := by
    simpa [s11, s10] using h_offsM_s9
  have h_offsN_s11 :
      evalOp (Op.expandDim 0 (Op.ref .nat [Bk] "offs_n")) s11 =
        some (Tile.expandDim 0 offsN) := by
    simpa [s11, s10] using h_offsN_s9
  have h_scoreMask_eval :
      evalOp
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
              (Op.constNat 0))
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
          (Op.constNat S_k)) s11 =
        some scoreMask := by
    simp only [evalOp_lt, evalOp_add, evalOp_mul, evalOp_constNat,
      Option.bind_eq_bind, Option.bind_some]
    erw [h_offsM_s11, h_offsN_s11]
    simpa [Option.bind, Tile.bop, Tile.cop, Tile.expandDim,
      TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
      ComparableDType.lt] using congrArg some h_scoreMask_raw
  have h_scoreMask_eval_exp :
      evalOp
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
              (Op.constNat 0))
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
          (Op.constNat S_k))
        ((((((((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, Bd])).setReg
            "kv_seq_mask" .bool [Bk, Bd] kvSeqMask).setReg
            "kv_d_mask" .bool [Bk, Bd] kvDMask).setReg
            "kv_mask" .bool [Bk, Bd] kvMask).setReg
            "k" .real [Bk, Bd] kLoaded).setReg
            "v" .real [Bk, Bd] vLoaded).setReg
            "scores_raw" .real [M, Bk] scoresRaw).setReg
            "causal" .bool [M, Bk] causal).setReg
            "causal_scores" .real [M, Bk] causalScores) =
        some scoreMask := by
    simpa [s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0, offsN, kPtrs, vPtrs,
      kvSeqMask, kvDMask, kvMask, kLoaded, vLoaded, scoresRaw, causal, causalScores,
      kBase, vBase] using h_scoreMask_eval
  have h_scores_eval_exp :
      evalOp
        (Op.where
          (Op.ref .bool [M, Bk] "score_mask")
          (Op.ref .real [M, Bk] "causal_scores")
          (Op.broadcast Op.negInf [M, Bk]))
        (((((((((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, Bd])).setReg
            "v_ptrs" .nat [Bk, Bd]
            ({ data := fun idx : TileIndex [Bk, Bd] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, Bd])).setReg
            "kv_seq_mask" .bool [Bk, Bd] kvSeqMask).setReg
            "kv_d_mask" .bool [Bk, Bd] kvDMask).setReg
            "kv_mask" .bool [Bk, Bd] kvMask).setReg
            "k" .real [Bk, Bd] kLoaded).setReg
            "v" .real [Bk, Bd] vLoaded).setReg
            "scores_raw" .real [M, Bk] scoresRaw).setReg
            "causal" .bool [M, Bk] causal).setReg
            "causal_scores" .real [M, Bk] causalScores).setReg
            "score_mask" .bool [M, Bk] scoreMask) =
        some scores := by
    simp [evalOp, Tile.select, h_scores_from_mask]
  have h_mBlock_eval :
      evalOp (Op.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩
        (keepDims := Bool.false) (Op.ref .real [M, Bk] "scores")) s13 =
        some mBlock := by
    erw [evalOp_reduceMax]
    simp [s13, s12]
    simp [Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
      TileShape.insertAxisIndex, TileShape.dropInsertedIndex, hBk, mBlock]
    funext idx
    rfl
  have h_offsN_step :
      stepStmt
        (Stmt.assign .nat [Bk] "offs_n"
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil
              (Op.ref .nat [] "n")
              (Op.constNat Bk))
            (Op.arange Bk))) s0 = some s1 := by
    simp [stepStmt, s1, s0, offsN, Tile.bop, NumericDType.add, NumericDType.mul]
    rfl
  have h_offsN_eval :
      evalOp
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil
            (Op.ref .nat [] "n")
            (Op.constNat Bk))
          (Op.arange Bk))
        (s.setReg "n" .nat [] (Tile.scalar k)) =
      some ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk]) := by
    simp [evalOp, Tile.bop, NumericDType.add, NumericDType.mul]
  have h_mNew_eval :
      evalOp
        (Op.max2 (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "m_i")
          (Op.ref .real [M] "m_block")) s14 = some mNew := by
    simp [evalOp, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      mNew, Qp, Kp, qStart, Tile.bop, hm]
  have h_alpha_eval :
      evalOp
        (Op.exp
          (Op.sub .real (Broadcast.consSame Broadcast.nil)
            (Op.ref .real [M] "m_i")
            (Op.ref .real [M] "m_new"))) s15 = some alpha := by
    simp [evalOp_exp, evalOp_sub, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      alpha, mNew, Qp, Kp, qStart, Tile.bop, Tile.uop, NumericDType.sub, WithBot.realSub, hm]
  have h_p_eval :
      evalOp
        (Op.exp
          (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (Op.ref .real [M, Bk] "scores")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "m_new")))) s16 =
        some p := by
    simp only [evalOp_exp, evalOp_sub, Option.bind_eq_bind, Option.bind_some]
    unfold evalOp
    simp [evalOp_expandDim, evalOp_expandDim_ref, evalOp_ref,
      s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      p, mNew, Qp, Kp, qStart, Tile.bop, Tile.uop, Tile.expandDim, NumericDType.sub,
      WithBot.realSub]
    rfl
  have h_lNew_eval :
      evalOp
        (Op.add .real (Broadcast.consSame Broadcast.nil)
          (Op.mul .real (Broadcast.consSame Broadcast.nil)
            (Op.ref .real [M] "alpha")
            (Op.ref .real [M] "l_i"))
          (Op.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
            (Op.ref .real [M, Bk] "p"))) s17 = some lNew := by
    simp [evalOp_add, evalOp_mul, evalOp_reduceSum,
      s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      lNew, alpha, p, Qp, Kp, qStart, Tile.bop, Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      hBk, NumericDType.add, NumericDType.mul, hl]
    unfold evalOp
    simp [s17, lNew, alpha, p, Qp, Kp, qStart, Tile.bop, Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      hBk, NumericDType.add, NumericDType.mul, WithBot.realAdd, WithBot.realMul]
    funext idx
    rfl
  have h_oNew_eval :
      evalOp
        (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "alpha"))
            (Op.ref .real [M, Bd] "o_acc"))
          (Op.dot (batch := []) (M := M) (K := Bk) (N := Bd)
            (Op.ref .real [M, Bk] "p")
            (Op.ref .real [Bk, Bd] "v"))) s18 = some oNew := by
    simp [evalOp_add, evalOp_mul, evalOp_dot,
      s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      oNew, alpha, p, Qp, Kp, Vp, qStart, Tile.bop, Tile.dot, Tile.expandDim,
      TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, ho]
    unfold evalOp
    simp [s18, oNew, alpha, p, Qp, Kp, Vp, qStart, Tile.bop, Tile.dot, Tile.expandDim,
      TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul]
    rfl
  refine ⟨s', ?_, ?_⟩
  · simp only [fa1LoopBodyStridedCausalBoundaryD, stepStmts, stepStmt]
    rw [h_offsN_eval]
    dsimp
    erw [h_kPtrs_eval]
    simp only [Option.bind_some]
    erw [h_vPtrs_eval]
    simp only [Option.bind_some]
    erw [h_kvSeqMask_eval_exp]
    simp only [Option.bind_some]
    erw [h_kvDMask_eval_exp]
    simp only [Option.bind_some]
    erw [h_kvMask_eval_exp]
    simp only [Option.bind_some]
    erw [h_kLoaded_eval]
    simp only [Option.bind_some]
    erw [h_vLoaded_eval]
    simp only [Option.bind_some]
    erw [h_scoresRawScaled_eval_exp]
    simp only [Option.bind_some]
    erw [h_causal_eval_exp]
    simp only [Option.bind_some]
    erw [h_causalScores_eval_exp]
    simp only [Option.bind_some]
    erw [h_scoreMask_eval_exp]
    simp only [Option.bind_some]
    erw [h_scores_eval_exp]
    simp only [Option.bind_some]
    erw [h_mBlock_eval]
    simp only [Option.bind_some]
    erw [h_mNew_eval]
    simp only [Option.bind_some]
    erw [h_alpha_eval]
    simp only [Option.bind_some]
    erw [h_p_eval]
    simp only [Option.bind_some]
    erw [h_lNew_eval]
    simp only [Option.bind_some]
    erw [h_oNew_eval]
    simp only [Option.bind_some]
    simp [evalOp, s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      offsN, kPtrs, vPtrs, kBase, vBase, Tile.vec]
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simpa [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hpids0
    · simpa [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hpids1
    · simpa [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hpids2
    · simpa [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hpid_qb
    · simpa [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hpid_h
    · simpa [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hpid_b
    · simpa [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hq_base
    · simpa [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hk_base
    · simpa [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hv_base
    · simpa [s', s20, s19, s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using ho_base
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
