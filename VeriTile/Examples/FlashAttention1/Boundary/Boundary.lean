/-
VeriTile.Examples.FlashAttention1.Boundary.Boundary

Split-out support for FlashAttention-1 v1 boundary proofs.
-/

import VeriTile.Examples.FlashAttention1.Boundary.BoundaryD

set_option maxHeartbeats 800000

namespace VeriTile.Examples

open VeriTile

private theorem fa1_boundary_eval_k_ptrs
    (s : BlockState) (Bk D k batch headIdx sKB sKH sKN sKD : Nat)
    (hk_base : s.regs .nat [] "k_base_off" =
      some (Tile.scalar (batch * sKB + headIdx * sKH)))
    (hoffs_d : s.regs .nat [D] "offs_d" =
      some (Tile.vec fun j : Fin D => j.val)) :
    evalOp
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.add .nat Broadcast.scalarL
          (Op.ref .nat [] "k_base_off")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.constNat sKN)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat sKD)))
      ((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
        ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])) =
    some (⟨fun idx : TileIndex [Bk, D] =>
      batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
        + idx.2.1.val * sKD⟩ : Tile .nat [Bk, D]) := by
  have hn :
      evalOp (Op.expandDim 1 (Op.ref .nat [Bk] "offs_n"))
          ((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
            ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])) =
        some (Tile.expandDim 1
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])) := by
    simp
  have hd :
      evalOp (Op.expandDim 0 (Op.ref .nat [D] "offs_d"))
          ((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
            ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])) =
        some (Tile.expandDim 0 (Tile.vec fun j : Fin D => j.val)) := by
    simp [hoffs_d]
  simp only [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat]
  erw [hn, hd]
  simp [Option.bind, option_match_bind, Tile.bop, Tile.expandDim,
    NumericDType.add, NumericDType.mul, hk_base, hoffs_d, hn, hd]
  funext idx
  cases idx with
  | mk i rest =>
    cases rest with
    | mk j rest2 =>
      cases rest2
      rfl

private theorem fa1_boundary_eval_v_ptrs
    (s : BlockState) (Bk D k batch headIdx sVB sVH sVN sVD : Nat)
    (kPtrs : Tile .nat [Bk, D])
    (hv_base : s.regs .nat [] "v_base_off" =
      some (Tile.scalar (batch * sVB + headIdx * sVH)))
    (hoffs_d : s.regs .nat [D] "offs_d" =
      some (Tile.vec fun j : Fin D => j.val)) :
    evalOp
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.add .nat Broadcast.scalarL
          (Op.ref .nat [] "v_base_off")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.constNat sVN)))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
          (Op.constNat sVD)))
      (((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
        ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
          "k_ptrs" .nat [Bk, D] kPtrs) =
    some (⟨fun idx : TileIndex [Bk, D] =>
      batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
        + idx.2.1.val * sVD⟩ : Tile .nat [Bk, D]) := by
  have hn :
      evalOp (Op.expandDim 1 (Op.ref .nat [Bk] "offs_n"))
          (((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
            ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
              "k_ptrs" .nat [Bk, D] kPtrs) =
        some (Tile.expandDim 1
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])) := by
    simp
  have hd :
      evalOp (Op.expandDim 0 (Op.ref .nat [D] "offs_d"))
          (((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
            ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
              "k_ptrs" .nat [Bk, D] kPtrs) =
        some (Tile.expandDim 0 (Tile.vec fun j : Fin D => j.val)) := by
    simp [hoffs_d]
  simp only [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat]
  erw [hn, hd]
  simp [Option.bind, option_match_bind, Tile.bop, Tile.expandDim,
    NumericDType.add, NumericDType.mul, hv_base, hoffs_d, hn, hd]
  funext idx
  cases idx with
  | mk i rest =>
    cases rest with
    | mk j rest2 =>
      cases rest2
      rfl

/-- Causal boundary strided loop step. One iteration preserves
`P_fa1_strided_causal_boundary`: K/V loads use the logical boundary mask,
scores are first causal-masked and then boundary-masked, and the
online-softmax update follows `FA1MathCausalBoundary`. -/
theorem fa1_step_strided_causal_boundary
    {M D Bk numKVBlocks S_k : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState)
    (hk : k < numKVBlocks)
    (hP : P_fa1_strided_causal_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale k s) :
    ∃ s',
      stepStmts (fa1LoopBodyStridedCausalBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1_strided_causal_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale (k + 1) s' := by
  rcases hP with
    ⟨hpids0, hpids1, hpids2,
     hpid_qb, hpid_h, hpid_b,
     hq_base, hk_base, hv_base, ho_base,
     hoffs_m, hoffs_d, hq, hm, hl, ho, hK, hV⟩
  let kBase : Nat := batch * sKB + headIdx * sKH
  let vBase : Nat := batch * sVB + headIdx * sVH
  let offsN : Tile .nat [Bk] := Tile.vec fun j : Fin Bk => k * Bk + j.val
  let kPtrs : Tile .nat [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] =>
      kBase + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD⟩
  let vPtrs : Tile .nat [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] =>
      vBase + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD⟩
  let kvMask : Tile .bool [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] => decide (k * Bk + idx.1.val < S_k)⟩
  let kLoaded : Tile .real [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] =>
      if h : k * Bk + idx.1.val < S_k then
        some (s.readMem kReg
          (kBase + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD))
      else
        some 0⟩
  let vLoaded : Tile .real [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] =>
      if h : k * Bk + idx.1.val < S_k then
        some (s.readMem vReg
          (vBase + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD))
      else
        some 0⟩
  have hK_loaded_eq : kLoaded =
      Tile.ofReal (fun idx : TileIndex [Bk, D] =>
        match FA1MathBoundary.blockIndex? S_k Bk k idx.1 with
        | some j => K (j, idx.2.1, PUnit.unit)
        | none => 0) :=
    fa1_block_load_tile_eq_strided_boundary kReg s kBase sKN sKD K hK k
  have hV_loaded_eq : vLoaded =
      Tile.ofReal (fun idx : TileIndex [Bk, D] =>
        match FA1MathBoundary.blockIndex? S_k Bk k idx.1 with
        | some j => V (j, idx.2.1, PUnit.unit)
        | none => 0) :=
    fa1_block_load_tile_eq_strided_boundary vReg s vBase sVN sVD V hV k
  let scoresRaw : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      Option.map (fun a => a * scale)
        (@Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
          (fun d : Fin D => Option.map (fun b => Q (idx.1, d, PUnit.unit) * b)
            (kLoaded.data (idx.2.1, d, PUnit.unit))))⟩
  let causal : Tile .bool [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      decide (k * Bk + idx.2.1.val ≤ qb * M + idx.1.val)⟩
  let causalScores : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      if k * Bk + idx.2.1.val ≤ qb * M + idx.1.val then scoresRaw.data idx
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
      max (FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale k idx.1)
        (mBlock.data idx)⟩
  let alpha : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      WithBot.realExp
        (WithBot.realSub
          (FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale k idx.1)
          (mNew.data idx))⟩
  let p : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      WithBot.realExp
        (WithBot.realSub (scores.data idx) (mNew.data (idx.1, PUnit.unit)))⟩
  let lNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      Option.map₂ (· + ·)
        (Option.map (· * FA1MathCausalBoundary.lPartial Bk (qb * M) Q numKVBlocks K scale k idx.1)
          (alpha.data idx))
        (@Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ
          (fun j : Fin Bk => p.data (idx.1, j, PUnit.unit)))⟩
  let oNew : Tile .real [M, D] :=
    ⟨fun idx : TileIndex [M, D] =>
      Option.map₂ (· + ·)
        (Option.map
          (· * FA1MathCausalBoundary.oPartial Bk (qb * M) Q numKVBlocks K V scale k
            (idx.1, idx.2.1, PUnit.unit))
          (alpha.data (idx.1, PUnit.unit)))
        (@Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ
          (fun j : Fin Bk =>
            Option.map₂ (· * ·)
              (p.data (idx.1, j, PUnit.unit))
              (vLoaded.data (j, idx.2.1, PUnit.unit))))⟩
  let s0 := s.setReg "n" .nat [] (Tile.scalar k)
  let s1 := s0.setReg "offs_n" .nat [Bk] offsN
  let s2 := s1.setReg "k_ptrs" .nat [Bk, D] kPtrs
  let s3 := s2.setReg "v_ptrs" .nat [Bk, D] vPtrs
  let s4 := s3.setReg "kv_mask" .bool [Bk, D] kvMask
  let s5 := s4.setReg "k" .real [Bk, D] kLoaded
  let s6 := s5.setReg "v" .real [Bk, D] vLoaded
  let s7 := s6.setReg "scores_raw" .real [M, Bk] scoresRaw
  let s8 := s7.setReg "causal" .bool [M, Bk] causal
  let s9 := s8.setReg "causal_scores" .real [M, Bk] causalScores
  let s10 := s9.setReg "score_mask" .bool [M, Bk] scoreMask
  let s11 := s10.setReg "scores" .real [M, Bk] scores
  let s12 := s11.setReg "m_block" .real [M] mBlock
  let s13 := s12.setReg "m_new" .real [M] mNew
  let s14 := s13.setReg "alpha" .real [M] alpha
  let s15 := s14.setReg "p" .real [M, Bk] p
  let s16 := s15.setReg "l_new" .real [M] lNew
  let s17 := s16.setReg "o_acc" .real [M, D] oNew
  let s18 := s17.setReg "m_i" .real [M] mNew
  let s' := s18.setReg "l_i" .real [M] lNew
  have h_score_per_j : ∀ (i : Fin M) (j : Fin Bk),
        scores.data (i, j, PUnit.unit)
          = FA1MathCausalBoundary.maskedScore Bk k (qb * M) Q K scale i j := by
    intro i j
    by_cases hLt : k * Bk + j.val < S_k
    · show (if k * Bk + ↑j < S_k then causalScores.data (i, j, PUnit.unit)
          else (none : WithBot ℝ)) = _
      rw [if_pos hLt]
      by_cases hLe : k * Bk + j.val ≤ qb * M + i.val
      · show (if k * Bk + ↑j ≤ qb * M + ↑i then scoresRaw.data (i, j, PUnit.unit)
            else (none : WithBot ℝ)) = _
        rw [if_pos hLe]
        have hkLoaded : ∀ d : Fin D,
            kLoaded.data (j, d, PUnit.unit)
              = some (K (⟨k * Bk + j.val, hLt⟩, d, PUnit.unit)) := by
          intro d
          have := congrArg (fun t : Tile .real [Bk, D] => t.data (j, d, PUnit.unit))
            hK_loaded_eq
          simp [Tile.ofReal, FA1MathBoundary.blockIndex?_of_lt _ _ _ _ hLt] at this
          exact this
        rw [FA1MathCausalBoundary.maskedScore_of_lt_of_le Bk k (qb * M) Q K scale i j hLt hLe]
        show Option.map (· * scale) _ = _
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
        unfold StreamingAccumulator.scaledScore
        congr 1
        ring
      · show (if k * Bk + ↑j ≤ qb * M + ↑i then scoresRaw.data (i, j, PUnit.unit)
            else (none : WithBot ℝ)) = _
        rw [if_neg hLe]
        rw [FA1MathCausalBoundary.maskedScore_of_lt_of_not_le Bk k (qb * M) Q K scale i j hLt hLe]
        rfl
    · show (if k * Bk + ↑j < S_k then causalScores.data (i, j, PUnit.unit)
        else (none : WithBot ℝ)) = _
      rw [if_neg hLt]
      rw [FA1MathCausalBoundary.maskedScore_of_not_lt Bk k (qb * M) Q K scale i j hLt]
      rfl
  have h_kPtrs_eval :=
    fa1_boundary_eval_k_ptrs s Bk D k batch headIdx sKB sKH sKN sKD hk_base hoffs_d
  have h_vPtrs_eval :=
    fa1_boundary_eval_v_ptrs s Bk D k batch headIdx sVB sVH sVN sVD kPtrs hv_base hoffs_d
  have h_offsN_s3 :
      evalOp (Op.expandDim 1 (Op.ref .nat [Bk] "offs_n")) s3 =
        some (Tile.expandDim 1 offsN) := by
    simp [s3, s2, s1, s0, offsN]
  have h_offsD_s3 :
      evalOp (Op.expandDim 0 (Op.ref .nat [D] "offs_d")) s3 =
        some (Tile.expandDim 0 (Tile.vec fun d : Fin D => d.val)) := by
    simp [s3, s2, s1, s0, offsN, hoffs_d]
  have h_kvMask_raw :
      ({ data := fun i : TileIndex [Bk, D] =>
        decide (k * Bk + (TileShape.dropInsertedIndex [Bk] 1 1 (i.1, 0, PUnit.unit)).1.val < S_k) } :
        Tile .bool [Bk, D]) = kvMask := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_kLoaded_from_mask :
      ({ data := fun i : TileIndex [Bk, D] =>
        if kvMask.data i = Bool.true then
          some (s.readMem kReg (batch * sKB + headIdx * sKH + (k * Bk + i.1.val) * sKN + i.2.1.val * sKD))
        else some 0 } : Tile .real [Bk, D]) = kLoaded := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    simp [kvMask, kLoaded, kBase]
  have h_vLoaded_from_mask :
      ({ data := fun i : TileIndex [Bk, D] =>
        if kvMask.data i = Bool.true then
          some (s.readMem vReg (batch * sVB + headIdx * sVH + (k * Bk + i.1.val) * sVN + i.2.1.val * sVD))
        else some 0 } : Tile .real [Bk, D]) = vLoaded := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    simp [kvMask, vLoaded, vBase]
  have h_kvMask_eval :
      evalOp
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
              (Op.constNat 0)))
          (Op.constNat S_k)) s3 =
        some kvMask := by
    simp only [evalOp_lt, evalOp_add, evalOp_mul, evalOp_constNat,
      Option.bind_eq_bind, Option.bind_some]
    erw [h_offsN_s3, h_offsD_s3]
    simpa [Option.bind, offsN, Tile.bop, Tile.cop, Tile.expandDim,
      TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
      ComparableDType.lt] using congrArg some h_kvMask_raw
  have h_kvMask_eval_exp :
      evalOp
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
              (Op.constNat 0)))
          (Op.constNat S_k))
        ((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, D])).setReg
            "v_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, D])) =
        some kvMask := by
    simpa [s3, s2, s1, s0, offsN, kPtrs, vPtrs, kBase, vBase] using h_kvMask_eval
  have h_kLoaded_eval :
      evalOp
        (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, D] "k_ptrs"))
          (MaskOpt.maskOther (Op.ref .bool [Bk, D] "kv_mask")
            (Op.broadcast (Op.const 0) [Bk, D]))) s4 =
        some kLoaded := by
    simp [evalOp, s4, s3, s2, s1, s0, kPtrs, kvMask, kLoaded,
      kBase, Region.cast, Tile.bop, Tile.expandDim, h_kLoaded_from_mask]
  have h_vLoaded_eval :
      evalOp
        (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, D] "v_ptrs"))
          (MaskOpt.maskOther (Op.ref .bool [Bk, D] "kv_mask")
            (Op.broadcast (Op.const 0) [Bk, D]))) s5 =
        some vLoaded := by
    simp [evalOp, s5, s4, s3, s2, s1, s0, vPtrs, kvMask, vLoaded,
      vBase, Region.cast, Tile.bop, Tile.expandDim, h_vLoaded_from_mask]
  have hq_s6 :
      evalOp (Op.ref .real [M, D] "q") s6 = some (Tile.ofReal Q) := by
    simpa [s6, s5, s4, s3, s2, s1, s0, evalOp_ref] using hq
  have hk_s6 :
      evalOp (Op.ref .real [Bk, D] "k") s6 = some kLoaded := by
    simp [s6, s5]
  have hkt_s6 :
      evalOp (Op.transpose (batch := []) (M := Bk) (N := D)
        (Op.ref .real [Bk, D] "k")) s6 =
        some (Tile.transpose [] kLoaded) := by
    erw [evalOp_transpose]
    simp [hk_s6]
  have h_scoresRaw_eval :
      evalOp (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
        (Op.ref .real [M, D] "q")
        (Op.transpose (batch := []) (M := Bk) (N := D)
          (Op.ref .real [Bk, D] "k"))) s6 =
        some (Tile.dot [] (Tile.ofReal Q) (Tile.transpose [] kLoaded)) := by
    erw [evalOp_dot]
    erw [hq_s6, hkt_s6]
    rfl
  have h_scoresRaw_eval_exp :
      evalOp (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
        (Op.ref .real [M, D] "q")
        (Op.transpose (batch := []) (M := Bk) (N := D)
          (Op.ref .real [Bk, D] "k")))
        (((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD } :
              Tile .nat [Bk, D])).setReg
            "v_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD } :
              Tile .nat [Bk, D])).setReg
            "kv_mask" .bool [Bk, D] kvMask).setReg
            "k" .real [Bk, D] kLoaded).setReg
            "v" .real [Bk, D] vLoaded) =
        some (Tile.dot [] (Tile.ofReal Q) (Tile.transpose [] kLoaded)) := by
    simpa [s6, s5, s4, s3, s2, s1, s0, offsN, kPtrs, vPtrs, kBase, vBase]
      using h_scoresRaw_eval
  have h_scoresRawScaled_eval_exp :
      evalOp
        (Op.mul .real Broadcast.scalarR
          (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
            (Op.ref .real [M, D] "q")
            (Op.transpose (batch := []) (M := Bk) (N := D)
              (Op.ref .real [Bk, D] "k")))
          (Op.const scale))
        (((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD } :
              Tile .nat [Bk, D])).setReg
            "v_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD } :
              Tile .nat [Bk, D])).setReg
            "kv_mask" .bool [Bk, D] kvMask).setReg
            "k" .real [Bk, D] kLoaded).setReg
            "v" .real [Bk, D] vLoaded) =
        some scoresRaw := by
    rw [evalOp_mul]
    erw [h_scoresRaw_eval_exp]
    simp [evalOp, scoresRaw, Tile.bop, Tile.dot]
    ext idx
    rcases idx with ⟨i, j, ⟨⟩⟩
    have hsum :
        (@Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
          (fun x => Option.map (fun b => Q (i, x, PUnit.unit) * b)
            ((Tile.transpose [] kLoaded).data (x, j, PUnit.unit)))) =
        (@Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
          (fun x => Option.map (fun b => Q (i, x, PUnit.unit) * b)
            (kLoaded.data (j, x, PUnit.unit)))) := by
      apply Finset.sum_congr rfl
      intro x hx
      rw [Tile.transpose_nil_data]
    rw [hsum]
    simp [NumericDType.mul, WithBot.realMul]
  have h_offsM_s7 :
      evalOp (Op.expandDim 1 (Op.ref .nat [M] "offs_m")) s7 =
        some (Tile.expandDim 1 (Tile.vec fun i : Fin M => qb * M + i.val)) := by
    simp [s7, s6, s5, s4, s3, s2, s1, s0, evalOp, hoffs_m]
  have h_offsN_s7 :
      evalOp (Op.expandDim 0 (Op.ref .nat [Bk] "offs_n")) s7 =
        some (Tile.expandDim 0 offsN) := by
    simp [s7, s6, s5, s4, s3, s2, s1, s0, evalOp, offsN]
  have h_causal_raw :
      ({ data := fun i : TileIndex [M, Bk] =>
        decide (k * Bk + (TileShape.dropInsertedIndex [Bk] 0 1
          (0, i.2.1, PUnit.unit)).1.val ≤
            qb * M + (TileShape.dropInsertedIndex [M] 1 1 (i.1, 0, PUnit.unit)).1.val) } :
        Tile .bool [M, Bk]) = causal := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_causalScores_from_mask :
      ({ data := fun idx : TileIndex [M, Bk] =>
        if causal.data idx = Bool.true then scoresRaw.data idx else none } :
        Tile .real [M, Bk]) = causalScores := by
    ext idx
    rcases idx with ⟨i, j, ⟨⟩⟩
    by_cases hj : k * Bk + j.val ≤ qb * M + i.val
    · simp [causal, causalScores, hj]
    · simp [causal, causalScores, hj]
  have h_causal_eval :
      evalOp
        (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n"))) s7 =
        some causal := by
    simp only [evalOp, Option.bind_eq_bind, Option.bind_some]
    erw [h_offsM_s7, h_offsN_s7]
    simpa [Option.bind, offsN, causal, Tile.cop, Tile.expandDim,
      TileShape.dropInsertedIndex, ComparableDType.ge] using congrArg some h_causal_raw
  have h_causal_eval_exp :
      evalOp
        (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
        ((((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, D])).setReg
            "v_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, D])).setReg
            "kv_mask" .bool [Bk, D] kvMask).setReg
            "k" .real [Bk, D] kLoaded).setReg
            "v" .real [Bk, D] vLoaded).setReg
            "scores_raw" .real [M, Bk] scoresRaw) =
        some causal := by
    simpa [s7, s6, s5, s4, s3, s2, s1, s0, offsN, kPtrs, vPtrs,
      kvMask, kLoaded, vLoaded, scoresRaw, kBase, vBase] using h_causal_eval
  have h_causalScores_eval_exp :
      evalOp
        (Op.where
          (Op.ref .bool [M, Bk] "causal")
          (Op.ref .real [M, Bk] "scores_raw")
          (Op.broadcast Op.negInf [M, Bk]))
        (((((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, D])).setReg
            "v_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, D])).setReg
            "kv_mask" .bool [Bk, D] kvMask).setReg
            "k" .real [Bk, D] kLoaded).setReg
            "v" .real [Bk, D] vLoaded).setReg
            "scores_raw" .real [M, Bk] scoresRaw).setReg
            "causal" .bool [M, Bk] causal) =
        some causalScores := by
    simp [evalOp, Tile.select, h_causalScores_from_mask]
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
            "k_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, D])).setReg
            "v_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, D])).setReg
            "kv_mask" .bool [Bk, D] kvMask).setReg
            "k" .real [Bk, D] kLoaded).setReg
            "v" .real [Bk, D] vLoaded).setReg
            "scores_raw" .real [M, Bk] scoresRaw).setReg
            "causal" .bool [M, Bk] causal).setReg
            "causal_scores" .real [M, Bk] causalScores) =
        some scoreMask := by
    simpa [s9, s8, s7, s6, s5, s4, s3, s2, s1, s0, offsN, kPtrs, vPtrs,
      kvMask, kLoaded, vLoaded, scoresRaw, causal, causalScores, kBase, vBase]
      using h_scoreMask_eval
  have h_scores_from_mask :
      ({ data := fun idx : TileIndex [M, Bk] =>
        if scoreMask.data idx = Bool.true then causalScores.data idx else none } :
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
          (Op.ref .real [M, Bk] "causal_scores")
          (Op.broadcast Op.negInf [M, Bk]))
        (((((((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, D])).setReg
            "v_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, D])).setReg
            "kv_mask" .bool [Bk, D] kvMask).setReg
            "k" .real [Bk, D] kLoaded).setReg
            "v" .real [Bk, D] vLoaded).setReg
            "scores_raw" .real [M, Bk] scoresRaw).setReg
            "causal" .bool [M, Bk] causal).setReg
            "causal_scores" .real [M, Bk] causalScores).setReg
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
      mNew, Tile.bop, hm]
  have h_alpha_eval :
      evalOp
        (Op.exp
          (Op.sub .real (Broadcast.consSame Broadcast.nil)
            (Op.ref .real [M] "m_i")
            (Op.ref .real [M] "m_new"))) s13 = some alpha := by
    simp [evalOp_exp, evalOp_sub, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      alpha, mNew, Tile.bop, Tile.uop, NumericDType.sub, WithBot.realSub, hm]
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
      p, mNew, Tile.bop, Tile.uop, Tile.expandDim, NumericDType.sub,
      WithBot.realSub]
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
      lNew, alpha, p, Tile.bop, Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      hBk, NumericDType.add, NumericDType.mul, hl]
    unfold evalOp
    simp [s15, lNew, alpha, p, Tile.bop, Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      hBk, NumericDType.add, NumericDType.mul, WithBot.realAdd, WithBot.realMul]
    funext idx
    rfl
  have h_oNew_eval :
      evalOp
        (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "alpha"))
            (Op.ref .real [M, D] "o_acc"))
          (Op.dot (batch := []) (M := M) (K := Bk) (N := D)
            (Op.ref .real [M, Bk] "p")
            (Op.ref .real [Bk, D] "v"))) s16 = some oNew := by
    simp [evalOp_add, evalOp_mul, evalOp_dot,
      s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      oNew, alpha, p, Tile.bop, Tile.dot, Tile.expandDim,
      TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, ho]
    unfold evalOp
    simp [s16, oNew, alpha, p, Tile.bop, Tile.dot, Tile.expandDim,
      TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul]
    rfl
  refine ⟨s', ?_, ?_⟩
  · simp only [fa1LoopBodyStridedCausalBoundary, stepStmts, stepStmt]
    rw [h_offsN_eval]
    dsimp
    erw [h_kPtrs_eval]
    simp only [Option.bind_some]
    erw [h_vPtrs_eval]
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
    simp [evalOp, s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      offsN, kPtrs, vPtrs, kBase, vBase, Tile.vec]
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
      show max (FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale k idx.1) (mBlock.data idx)
        = FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) idx.1
      rw [FA1MathCausalBoundary.mPartial_succ_of_lt Bk (qb * M) Q numKVBlocks K scale k hk idx.1]
      congr 1
      show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
          (fun j => scores.data (idx.1, j, PUnit.unit))
          = Finset.univ.sup
              (fun j => FA1MathCausalBoundary.maskedScore Bk k (qb * M) Q K scale idx.1 j)
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
          = FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) idx.1 := by
        show max (FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale k idx.1)
            (mBlock.data (idx.1, PUnit.unit)) = _
        rw [FA1MathCausalBoundary.mPartial_succ_of_lt Bk (qb * M) Q numKVBlocks K scale k hk idx.1]
        congr 1
        show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
            (fun j => scores.data (idx.1, j, PUnit.unit))
            = Finset.univ.sup
                (fun j => FA1MathCausalBoundary.maskedScore Bk k (qb * M) Q K scale idx.1 j)
        rw [Finset.sup'_eq_sup]
        apply Finset.sup_congr rfl
        intro j _
        exact h_score_per_j idx.1 j
      have h_alpha : alpha.data idx
          = some (FA1MathCausalBoundary.alphaPartial Bk (qb * M) Q numKVBlocks K scale k idx.1) := by
        show WithBot.realExp _ = _
        rw [h_mNew]
        unfold FA1MathCausalBoundary.alphaPartial
        exact FA1MathCausalBoundary.realExp_eq_some_unbotD _
      have h_p_sum : (∑ j : Fin Bk, p.data (idx.1, j, PUnit.unit) : WithBot ℝ)
          = some (∑ j : Fin Bk,
              (WithBot.realExp
                (WithBot.realSub
                  (FA1MathCausalBoundary.maskedScore Bk k (qb * M) Q K scale idx.1 j)
                  (FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) idx.1))
              ).unbotD 0) := by
        rw [← WithBot.sum_someTerm_eq_some]
        apply Finset.sum_congr rfl
        intro j _
        show WithBot.realExp _ = _
        rw [h_score_per_j idx.1 j, h_mNew]
        exact FA1MathCausalBoundary.realExp_eq_some_unbotD _
      show Option.map₂ (fun x1 x2 : ℝ => x1 + x2)
          (Option.map (fun x : ℝ =>
            x * FA1MathCausalBoundary.lPartial Bk (qb * M) Q numKVBlocks K scale k idx.1)
            (alpha.data idx))
          (∑ j : Fin Bk, p.data (idx.1, j, PUnit.unit) : WithBot ℝ)
          = some (FA1MathCausalBoundary.lPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) idx.1)
      rw [h_alpha, h_p_sum,
        FA1MathCausalBoundary.lPartial_succ_of_lt Bk (qb * M) Q numKVBlocks K scale k hk idx.1]
      rfl
    · have h_regs_o_acc : s'.regs .real [M, D] "o_acc" = some oNew := by
        simp [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_o_acc]
      apply congrArg some
      apply Tile.ext
      intro idx
      have h_mNew : mNew.data (idx.1, PUnit.unit)
          = FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) idx.1 := by
        show max (FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale k idx.1)
            (mBlock.data (idx.1, PUnit.unit)) = _
        rw [FA1MathCausalBoundary.mPartial_succ_of_lt Bk (qb * M) Q numKVBlocks K scale k hk idx.1]
        congr 1
        show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
            (fun j => scores.data (idx.1, j, PUnit.unit))
            = Finset.univ.sup
                (fun j => FA1MathCausalBoundary.maskedScore Bk k (qb * M) Q K scale idx.1 j)
        rw [Finset.sup'_eq_sup]
        apply Finset.sup_congr rfl
        intro j _
        exact h_score_per_j idx.1 j
      have h_alpha : alpha.data (idx.1, PUnit.unit)
          = some (FA1MathCausalBoundary.alphaPartial Bk (qb * M) Q numKVBlocks K scale k idx.1) := by
        show WithBot.realExp _ = _
        rw [h_mNew]
        unfold FA1MathCausalBoundary.alphaPartial
        exact FA1MathCausalBoundary.realExp_eq_some_unbotD _
      have h_vLoaded : ∀ j : Fin Bk,
          vLoaded.data (j, idx.2.1, PUnit.unit)
            = some (match FA1MathBoundary.blockIndex? S_k Bk k j with
                    | some jGlobal => V (jGlobal, idx.2.1, PUnit.unit)
                    | none => 0) := by
        intro j
        have := congrArg (fun t : Tile .real [Bk, D] => t.data (j, idx.2.1, PUnit.unit))
          hV_loaded_eq
        simp [Tile.ofReal] at this
        exact this
      have h_p_per_j : ∀ j : Fin Bk,
          p.data (idx.1, j, PUnit.unit)
            = some ((WithBot.realExp
                (WithBot.realSub
                  (FA1MathCausalBoundary.maskedScore Bk k (qb * M) Q K scale idx.1 j)
                  (FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) idx.1))
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
                    if jGlobal.val ≤ qb * M + idx.1.val then
                      (WithBot.realExp
                        (WithBot.realSub
                          (FA1MathCausalBoundary.maskedScore Bk k (qb * M) Q K scale idx.1 j)
                          (FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) idx.1))
                      ).unbotD 0 * V (jGlobal, idx.2.1, PUnit.unit)
                    else
                      0
                | none => 0) := by
        rw [← WithBot.sum_someTerm_eq_some]
        apply Finset.sum_congr rfl
        intro j _
        rw [h_p_per_j j, h_vLoaded j]
        by_cases hLt : k * Bk + j.val < S_k
        · rw [FA1MathBoundary.blockIndex?_of_lt S_k Bk k j hLt]
          by_cases hLe : k * Bk + j.val ≤ qb * M + idx.1.val
          · simp [hLe]
          · rw [FA1MathCausalBoundary.maskedScore_of_lt_of_not_le Bk k (qb * M) Q K scale idx.1 j hLt hLe]
            simp [hLe]
        · simp [FA1MathBoundary.blockIndex?_of_not_lt _ _ _ _ hLt]
      show Option.map₂ (fun x1 x2 : ℝ => x1 + x2)
          (Option.map (fun x : ℝ =>
            x * FA1MathCausalBoundary.oPartial Bk (qb * M) Q numKVBlocks K V scale k
              (idx.1, idx.2.1, PUnit.unit))
            (alpha.data (idx.1, PUnit.unit)))
          (∑ j : Fin Bk, Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (p.data (idx.1, j, PUnit.unit))
            (vLoaded.data (j, idx.2.1, PUnit.unit)) : WithBot ℝ)
          = some (FA1MathCausalBoundary.oPartial Bk (qb * M) Q numKVBlocks K V scale (k + 1) idx)
      rw [h_alpha, h_pv_sum,
        FA1MathCausalBoundary.oPartial_succ_of_lt Bk (qb * M) Q numKVBlocks K V scale k hk idx]
      rfl
    · intro idx
      simpa [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hK idx
    · intro idx
      simpa [s', s18, s17, s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hV idx

/-- Causal boundary strided FA-1 forward correctness, raw form. Bundles
`fa1_forward_correct_strided_causal_boundary_raw_of_step` with the proven
causal-boundary loop step. -/
theorem fa1_forward_correct_strided_causal_boundary_raw
    {M D Bk numKVBlocks S_q S_k : Nat}
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
        = some
            (FA1MathCausalBoundary.oPartial Bk (s.pids 0 * M) Q numKVBlocks K V scale
                numKVBlocks idx /
              FA1MathCausalBoundary.lPartial Bk (s.pids 0 * M) Q numKVBlocks K scale
                numKVBlocks idx.1) := by
  exact fa1_forward_correct_strided_causal_boundary_raw_of_step
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV hInj
    (fun i st hi hPi =>
      fa1_step_strided_causal_boundary hBk qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale i st hi hPi)

/-- Causal-boundary strided FA-1 forward correctness in canonical spec form. -/
theorem fa1_forward_correct_strided_causal_boundary
    {M D Bk numKVBlocks S_q S_k : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
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
        = some (attentionRealCausalBlock (s.pids 0 * M) Q K V scale idx) := by
  intro idx hIdx
  rw [fa1_forward_correct_strided_causal_boundary_raw hBk
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV hInj idx hIdx]
  congr 1
  exact FA1MathCausalBoundary.streaming_eq_attentionRealCausalBlock hBk hSk hSkLe
    (s.pids 0 * M) Q K V scale idx

set_option maxHeartbeats 4000000 in
/-- Boundary strided loop step. One iteration of the v1 KV loop preserves
`P_fa1_strided_boundary`: masked K/V loads read logical K/V cells for
in-range lanes and zero for padded lanes; the score mask turns padded score
lanes into `-inf`; the online-softmax update is discharged by the boundary
block lemmas in `FA1MathBoundary`. -/
theorem fa1_step_strided_boundary
    {M D Bk numKVBlocks S_k : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat) (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat) (sOB sOH sOM sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState)
    (hk : k < numKVBlocks)
    (hP : P_fa1_strided_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale k s) :
    ∃ s',
      stepStmts (fa1LoopBodyStridedBoundary kReg vReg M D Bk S_k sKN sKD sVN sVD scale)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1_strided_boundary (Bk := Bk) (numKVBlocks := numKVBlocks)
        qReg kReg vReg qb headIdx batch
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale (k + 1) s' := by
  rcases hP with
    ⟨hpids0, hpids1, hpids2,
     hpid_qb, hpid_h, hpid_b,
     hq_base, hk_base, hv_base, ho_base,
     hoffs_m, hoffs_d, hq, hm, hl, ho, hK, hV⟩
  -- Kernel-form witnesses for K/V (matching `tl.load(..., mask, other=0)` exactly).
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
  let kvMask : Tile .bool [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] => decide (k * Bk + idx.1.val < S_k)⟩
  -- Kernel-form K/V tiles: `if h: in-bounds then some(readMem) else some 0`.
  let kLoaded : Tile .real [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] =>
      if h : k * Bk + idx.1.val < S_k then
        some (s.readMem kReg
          (kBase + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD))
      else
        some 0⟩
  let vLoaded : Tile .real [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] =>
      if h : k * Bk + idx.1.val < S_k then
        some (s.readMem vReg
          (vBase + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD))
      else
        some 0⟩
  -- Bridges: kLoaded / vLoaded equal the canonical match-on-blockIndex? form
  -- used by `block_scores_tile_eq` etc.
  have hK_loaded_eq : kLoaded =
      Tile.ofReal (fun idx : TileIndex [Bk, D] =>
        match FA1MathBoundary.blockIndex? S_k Bk k idx.1 with
        | some j => K (j, idx.2.1, PUnit.unit)
        | none => 0) :=
    fa1_block_load_tile_eq_strided_boundary kReg s kBase sKN sKD K hK k
  have hV_loaded_eq : vLoaded =
      Tile.ofReal (fun idx : TileIndex [Bk, D] =>
        match FA1MathBoundary.blockIndex? S_k Bk k idx.1 with
        | some j => V (j, idx.2.1, PUnit.unit)
        | none => 0) :=
    fa1_block_load_tile_eq_strided_boundary vReg s vBase sVN sVD V hV k
  -- Kernel-form downstream tiles. Each one is exactly what `evalOp` produces
  -- on the masked K/V load, so the operational first branch closes by `rfl`.
  -- The bridge to canonical (`maskedScore` / `mPartial` / `lPartial` / `oPartial`)
  -- happens in the invariant branch via the `FA1MathBoundary.block_*_tile_eq`
  -- lemmas applied per P-clause.
  let scoresRaw : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      Option.map (fun a => a * scale)
        (@Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
          (fun d : Fin D => Option.map (fun b => Q (idx.1, d, PUnit.unit) * b)
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
      max (FA1MathBoundary.mPartial Bk Q numKVBlocks K scale k idx.1)
        (mBlock.data idx)⟩
  let alpha : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      WithBot.realExp
        (Option.map₂ (· - ·)
          (FA1MathBoundary.mPartial Bk Q numKVBlocks K scale k idx.1)
          (mNew.data idx))⟩
  let p : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      WithBot.realExp
        (Option.map₂ (· - ·) (scores.data idx) (mNew.data (idx.1, PUnit.unit)))⟩
  let lNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      Option.map₂ (· + ·)
        (Option.map (· * FA1MathBoundary.lPartial Bk Q numKVBlocks K scale k idx.1)
          (alpha.data idx))
        (@Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ
          (fun j : Fin Bk => p.data (idx.1, j, PUnit.unit)))⟩
  let oNew : Tile .real [M, D] :=
    ⟨fun idx : TileIndex [M, D] =>
      Option.map₂ (· + ·)
        (Option.map
          (· * FA1MathBoundary.oPartial Bk Q numKVBlocks K V scale k
            (idx.1, idx.2.1, PUnit.unit))
          (alpha.data (idx.1, PUnit.unit)))
        (@Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ
          (fun j : Fin Bk =>
            Option.map₂ (· * ·)
              (p.data (idx.1, j, PUnit.unit))
              (vLoaded.data (j, idx.2.1, PUnit.unit))))⟩
  let s0 := s.setReg "n" .nat [] (Tile.scalar k)
  let s1 := s0.setReg "offs_n" .nat [Bk] offsN
  let s2 := s1.setReg "k_ptrs" .nat [Bk, D] kPtrs
  let s3 := s2.setReg "v_ptrs" .nat [Bk, D] vPtrs
  let s4 := s3.setReg "kv_mask" .bool [Bk, D] kvMask
  let s5 := s4.setReg "k" .real [Bk, D] kLoaded
  let s6 := s5.setReg "v" .real [Bk, D] vLoaded
  let s7 := s6.setReg "scores_raw" .real [M, Bk] scoresRaw
  let s8 := s7.setReg "score_mask" .bool [M, Bk] scoreMask
  let s9 := s8.setReg "scores" .real [M, Bk] scores
  let s10 := s9.setReg "m_block" .real [M] mBlock
  let s11 := s10.setReg "m_new" .real [M] mNew
  let s12 := s11.setReg "alpha" .real [M] alpha
  let s13 := s12.setReg "p" .real [M, Bk] p
  let s14 := s13.setReg "l_new" .real [M] lNew
  let s15 := s14.setReg "o_acc" .real [M, D] oNew
  let s16 := s15.setReg "m_i" .real [M] mNew
  let s' := s16.setReg "l_i" .real [M] lNew
  have h_kPtrs_eval :=
    fa1_boundary_eval_k_ptrs s Bk D k batch headIdx sKB sKH sKN sKD hk_base hoffs_d
  have h_vPtrs_eval :=
    fa1_boundary_eval_v_ptrs s Bk D k batch headIdx sVB sVH sVN sVD kPtrs hv_base hoffs_d
  have h_offsN_s3 :
      evalOp (Op.expandDim 1 (Op.ref .nat [Bk] "offs_n")) s3 =
        some (Tile.expandDim 1 offsN) := by
    simp [s3, s2, s1, s0, offsN]
  have h_offsD_s3 :
      evalOp (Op.expandDim 0 (Op.ref .nat [D] "offs_d")) s3 =
        some (Tile.expandDim 0 (Tile.vec fun d : Fin D => d.val)) := by
    simp [s3, s2, s1, s0, offsN, hoffs_d]
  have h_kvMask_raw :
      ({ data := fun i : TileIndex [Bk, D] =>
        decide (k * Bk + (TileShape.dropInsertedIndex [Bk] 1 1 (i.1, 0, PUnit.unit)).1.val < S_k) } :
        Tile .bool [Bk, D]) = kvMask := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    rfl
  have h_kLoaded_from_mask :
      ({ data := fun i : TileIndex [Bk, D] =>
        if kvMask.data i = Bool.true then
          some (s.readMem kReg (batch * sKB + headIdx * sKH + (k * Bk + i.1.val) * sKN + i.2.1.val * sKD))
        else some 0 } : Tile .real [Bk, D]) = kLoaded := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    simp [kvMask, kLoaded, kBase]
  have h_vLoaded_from_mask :
      ({ data := fun i : TileIndex [Bk, D] =>
        if kvMask.data i = Bool.true then
          some (s.readMem vReg (batch * sVB + headIdx * sVH + (k * Bk + i.1.val) * sVN + i.2.1.val * sVD))
        else some 0 } : Tile .real [Bk, D]) = vLoaded := by
    ext i
    rcases i with ⟨i, j, ⟨⟩⟩
    simp [kvMask, vLoaded, vBase]
  have h_kvMask_eval :
      evalOp
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
              (Op.constNat 0)))
          (Op.constNat S_k)) s3 =
        some kvMask := by
    simp only [evalOp_lt, evalOp_add, evalOp_mul, evalOp_constNat,
      Option.bind_eq_bind, Option.bind_some]
    erw [h_offsN_s3, h_offsD_s3]
    simpa [Option.bind, offsN, Tile.bop, Tile.cop, Tile.expandDim,
      TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
      ComparableDType.lt] using congrArg some h_kvMask_raw
  have h_kvMask_eval_exp :
      evalOp
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
              (Op.constNat 0)))
          (Op.constNat S_k))
        ((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, D])).setReg
            "v_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, D])) =
        some kvMask := by
    simpa [s3, s2, s1, s0, offsN, kPtrs, vPtrs, kBase, vBase] using h_kvMask_eval
  have h_kLoaded_eval :
      evalOp
        (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, D] "k_ptrs"))
          (MaskOpt.maskOther (Op.ref .bool [Bk, D] "kv_mask")
            (Op.broadcast (Op.const 0) [Bk, D]))) s4 =
        some kLoaded := by
    simp [evalOp, s4, s3, s2, s1, s0, kPtrs, kvMask, kLoaded,
      kBase, Region.cast, Tile.bop, Tile.expandDim, h_kLoaded_from_mask]
  have h_vLoaded_eval :
      evalOp
        (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, D] "v_ptrs"))
          (MaskOpt.maskOther (Op.ref .bool [Bk, D] "kv_mask")
            (Op.broadcast (Op.const 0) [Bk, D]))) s5 =
        some vLoaded := by
    simp [evalOp, s5, s4, s3, s2, s1, s0, vPtrs, kvMask, vLoaded,
      vBase, Region.cast, Tile.bop, Tile.expandDim, h_vLoaded_from_mask]
  have hq_s6 :
      evalOp (Op.ref .real [M, D] "q") s6 = some (Tile.ofReal Q) := by
    simpa [s6, s5, s4, s3, s2, s1, s0, evalOp_ref] using hq
  have hk_s6 :
      evalOp (Op.ref .real [Bk, D] "k") s6 = some kLoaded := by
    simp [s6, s5]
  have hkt_s6 :
      evalOp (Op.transpose (batch := []) (M := Bk) (N := D)
        (Op.ref .real [Bk, D] "k")) s6 =
        some (Tile.transpose [] kLoaded) := by
    erw [evalOp_transpose]
    simp [hk_s6]
  have h_scoresRaw_eval :
      evalOp (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
        (Op.ref .real [M, D] "q")
        (Op.transpose (batch := []) (M := Bk) (N := D)
          (Op.ref .real [Bk, D] "k"))) s6 =
        some (Tile.dot [] (Tile.ofReal Q) (Tile.transpose [] kLoaded)) := by
    erw [evalOp_dot]
    erw [hq_s6, hkt_s6]
    rfl
  have h_scoresRaw_eval_exp :
      evalOp (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
        (Op.ref .real [M, D] "q")
        (Op.transpose (batch := []) (M := Bk) (N := D)
          (Op.ref .real [Bk, D] "k")))
        (((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD } :
              Tile .nat [Bk, D])).setReg
            "v_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD } :
              Tile .nat [Bk, D])).setReg
            "kv_mask" .bool [Bk, D] kvMask).setReg
            "k" .real [Bk, D] kLoaded).setReg
            "v" .real [Bk, D] vLoaded) =
        some (Tile.dot [] (Tile.ofReal Q) (Tile.transpose [] kLoaded)) := by
    simpa [s6, s5, s4, s3, s2, s1, s0, offsN, kPtrs, vPtrs, kBase, vBase]
      using h_scoresRaw_eval
  have h_scoresRawScaled_eval_exp :
      evalOp
        (Op.mul .real Broadcast.scalarR
          (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
            (Op.ref .real [M, D] "q")
            (Op.transpose (batch := []) (M := Bk) (N := D)
              (Op.ref .real [Bk, D] "k")))
          (Op.const scale))
        (((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN + idx.2.1.val * sKD } :
              Tile .nat [Bk, D])).setReg
            "v_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN + idx.2.1.val * sVD } :
              Tile .nat [Bk, D])).setReg
            "kv_mask" .bool [Bk, D] kvMask).setReg
            "k" .real [Bk, D] kLoaded).setReg
            "v" .real [Bk, D] vLoaded) =
        some scoresRaw := by
    rw [evalOp_mul]
    erw [h_scoresRaw_eval_exp]
    simp [evalOp, scoresRaw, Tile.bop, Tile.dot]
    ext idx
    rcases idx with ⟨i, j, ⟨⟩⟩
    have hsum :
        (@Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
          (fun x => Option.map (fun b => Q (i, x, PUnit.unit) * b)
            ((Tile.transpose [] kLoaded).data (x, j, PUnit.unit)))) =
        (@Finset.sum (Fin D) (WithBot ℝ) _ Finset.univ
          (fun x => Option.map (fun b => Q (i, x, PUnit.unit) * b)
            (kLoaded.data (j, x, PUnit.unit)))) := by
      apply Finset.sum_congr rfl
      intro x hx
      rw [Tile.transpose_nil_data]
    rw [hsum]
    simp [NumericDType.mul, WithBot.realMul]
  have h_offsM_s7 :
      evalOp (Op.expandDim 1 (Op.ref .nat [M] "offs_m")) s7 =
        some (Tile.expandDim 1 (Tile.vec fun i : Fin M => qb * M + i.val)) := by
    simp [s7, s6, s5, s4, s3, s2, s1, s0, evalOp, hoffs_m]
  have h_offsN_s7 :
      evalOp (Op.expandDim 0 (Op.ref .nat [Bk] "offs_n")) s7 =
        some (Tile.expandDim 0 offsN) := by
    simp [s7, s6, s5, s4, s3, s2, s1, s0, evalOp, offsN]
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
          (Op.constNat S_k)) s7 =
        some scoreMask := by
    simp only [evalOp_lt, evalOp_add, evalOp_mul, evalOp_constNat,
      Option.bind_eq_bind, Option.bind_some]
    erw [h_offsM_s7, h_offsN_s7]
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
        ((((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, D])).setReg
            "v_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, D])).setReg
            "kv_mask" .bool [Bk, D] kvMask).setReg
            "k" .real [Bk, D] kLoaded).setReg
            "v" .real [Bk, D] vLoaded).setReg
            "scores_raw" .real [M, Bk] scoresRaw) =
        some scoreMask := by
    simpa [s7, s6, s5, s4, s3, s2, s1, s0, offsN, kPtrs, vPtrs,
      kvMask, kLoaded, vLoaded, scoresRaw, kBase, vBase] using h_scoreMask_eval
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
        (((((((((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
          ({ data := fun i : TileIndex [Bk] => k * Bk + i.1.val } : Tile .nat [Bk])).setReg
            "k_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sKB + headIdx * sKH + (k * Bk + idx.1.val) * sKN
                + idx.2.1.val * sKD } : Tile .nat [Bk, D])).setReg
            "v_ptrs" .nat [Bk, D]
            ({ data := fun idx : TileIndex [Bk, D] =>
              batch * sVB + headIdx * sVH + (k * Bk + idx.1.val) * sVN
                + idx.2.1.val * sVD } : Tile .nat [Bk, D])).setReg
            "kv_mask" .bool [Bk, D] kvMask).setReg
            "k" .real [Bk, D] kLoaded).setReg
            "v" .real [Bk, D] vLoaded).setReg
            "scores_raw" .real [M, Bk] scoresRaw).setReg
            "score_mask" .bool [M, Bk] scoreMask) =
        some scores := by
    simp [evalOp, Tile.select, h_scores_from_mask]
  have h_mBlock_eval :
      evalOp (Op.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩
        (keepDims := Bool.false) (Op.ref .real [M, Bk] "scores")) s9 =
        some mBlock := by
    erw [evalOp_reduceMax]
    simp [s9, s8]
    simp [Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
      TileShape.insertAxisIndex, TileShape.dropInsertedIndex, hBk, mBlock]
    funext idx
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
          (Op.ref .real [M] "m_block")) s10 = some mNew := by
    simp [evalOp, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      mNew, Tile.bop, hm]
  have h_alpha_eval :
      evalOp
        (Op.exp
          (Op.sub .real (Broadcast.consSame Broadcast.nil)
            (Op.ref .real [M] "m_i")
            (Op.ref .real [M] "m_new"))) s11 = some alpha := by
    simp [evalOp_exp, evalOp_sub, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      alpha, mNew, Tile.bop, Tile.uop, NumericDType.sub, hm]
  have h_p_eval :
      evalOp
        (Op.exp
          (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (Op.ref .real [M, Bk] "scores")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "m_new")))) s12 =
        some p := by
    simp only [evalOp_exp, evalOp_sub, Option.bind_eq_bind, Option.bind_some]
    unfold evalOp
    simp [evalOp_expandDim, evalOp_expandDim_ref, evalOp_ref,
      s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      p, mNew, Tile.bop, Tile.uop, Tile.expandDim, NumericDType.sub]
    rfl
  have h_lNew_eval :
      evalOp
        (Op.add .real (Broadcast.consSame Broadcast.nil)
          (Op.mul .real (Broadcast.consSame Broadcast.nil)
            (Op.ref .real [M] "alpha")
            (Op.ref .real [M] "l_i"))
          (Op.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
            (Op.ref .real [M, Bk] "p"))) s13 = some lNew := by
    simp [evalOp_add, evalOp_mul, evalOp_reduceSum,
      s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      lNew, alpha, p, Tile.bop, Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      hBk, NumericDType.add, NumericDType.mul, hl]
    unfold evalOp
    simp [s13, lNew, alpha, p, Tile.bop, Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      hBk, NumericDType.add, NumericDType.mul, WithBot.realAdd, WithBot.realMul]
    funext idx
    rfl
  have h_oNew_eval :
      evalOp
        (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "alpha"))
            (Op.ref .real [M, D] "o_acc"))
          (Op.dot (batch := []) (M := M) (K := Bk) (N := D)
            (Op.ref .real [M, Bk] "p")
            (Op.ref .real [Bk, D] "v"))) s14 = some oNew := by
    simp [evalOp_add, evalOp_mul, evalOp_dot,
      s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      oNew, alpha, p, Tile.bop, Tile.dot, Tile.expandDim,
      TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, ho]
    unfold evalOp
    simp [s14, oNew, alpha, p, Tile.bop, Tile.dot, Tile.expandDim,
      TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul]
    rfl
  refine ⟨s', ?_, ?_⟩
  · simp only [fa1LoopBodyStridedBoundary, stepStmts, stepStmt]
    rw [h_offsN_eval]
    dsimp
    erw [h_kPtrs_eval]
    simp only [Option.bind_some]
    erw [h_vPtrs_eval]
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
    simp [evalOp, s', s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
      offsN, kPtrs, vPtrs, kBase, vBase, Tile.vec]
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · show s.pids 0 = qb; exact hpids0
    · show s.pids 1 = headIdx; exact hpids1
    · show s.pids 2 = batch; exact hpids2
    · show s.regs .nat [] "pid_qb" = some (Tile.scalar qb); exact hpid_qb
    · show s.regs .nat [] "pid_h" = some (Tile.scalar headIdx); exact hpid_h
    · show s.regs .nat [] "pid_b" = some (Tile.scalar batch); exact hpid_b
    · show s.regs .nat [] "q_base_off" = some (Tile.scalar (batch * sQB + headIdx * sQH))
      exact hq_base
    · show s.regs .nat [] "k_base_off" = some (Tile.scalar (batch * sKB + headIdx * sKH))
      exact hk_base
    · show s.regs .nat [] "v_base_off" = some (Tile.scalar (batch * sVB + headIdx * sVH))
      exact hv_base
    · show s.regs .nat [] "o_base_off" = some (Tile.scalar (batch * sOB + headIdx * sOH))
      exact ho_base
    · -- offs_m
      simp [s', s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hoffs_m]
    · -- offs_d
      simp [s', s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hoffs_d]
    · -- q
      simp [s', s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        hq]
    · -- m_i
      have h_regs_m_i : s'.regs .real [M] "m_i" = some mNew := by
        simp [s', s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_m_i]
      congr 1
      ext idx
      have h_score_per_j : ∀ j : Fin Bk,
          scores.data (idx.1, j, PUnit.unit)
            = FA1MathBoundary.maskedScore Bk k Q K scale idx.1 j := by
        intro j
        by_cases h : k * Bk + j.val < S_k
        · show (if k * Bk + ↑j < S_k then scoresRaw.data (idx.1, j, PUnit.unit)
              else (none : WithBot ℝ)) = _
          rw [if_pos h]
          have hkLoaded : ∀ d : Fin D,
              kLoaded.data (j, d, PUnit.unit)
                = some (K (⟨k * Bk + j.val, h⟩, d, PUnit.unit)) := by
            intro d
            have := congrArg (fun t : Tile .real [Bk, D] => t.data (j, d, PUnit.unit))
              hK_loaded_eq
            simp [Tile.ofReal, FA1MathBoundary.blockIndex?_of_lt _ _ _ _ h] at this
            exact this
          rw [FA1MathBoundary.maskedScore_of_lt Bk k Q K scale idx.1 j h]
          show Option.map (· * scale) _ = _
          have h_sum :
              (∑ x : Fin D, Option.map (fun b : ℝ => Q (idx.1, x, PUnit.unit) * b)
                (kLoaded.data (j, x, PUnit.unit)) : WithBot ℝ)
              = some (∑ x : Fin D,
                  Q (idx.1, x, PUnit.unit) * K (⟨k * Bk + j.val, h⟩, x, PUnit.unit)) := by
            rw [← WithBot.sum_someTerm_eq_some]
            apply Finset.sum_congr rfl
            intro x _
            rw [hkLoaded x]
            rfl
          rw [h_sum]
          show (some _ : WithBot ℝ) = some _
          unfold StreamingAccumulator.scaledScore
          congr 1
          ring
        · show (if k * Bk + ↑j < S_k then scoresRaw.data (idx.1, j, PUnit.unit)
              else (none : WithBot ℝ)) = _
          rw [if_neg h]
          rw [FA1MathBoundary.maskedScore_of_not_lt Bk k Q K scale idx.1 j h]
          rfl
      show max (FA1MathBoundary.mPartial Bk Q numKVBlocks K scale k idx.1) (mBlock.data idx)
        = FA1MathBoundary.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1
      rw [FA1MathBoundary.mPartial_succ_of_lt Bk Q numKVBlocks K scale k hk idx.1]
      congr 1
      show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
          (fun j => scores.data (idx.1, j, PUnit.unit))
          = Finset.univ.sup
              (fun j => FA1MathBoundary.maskedScore Bk k Q K scale idx.1 j)
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      exact h_score_per_j j
    · -- l_i
      have h_regs_l_i : s'.regs .real [M] "l_i" = some lNew := by
        simp [s', s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_l_i]
      apply congrArg some
      show lNew = Tile.ofReal (fun idx : TileIndex [M] =>
          FA1MathBoundary.lPartial Bk Q numKVBlocks K scale (k + 1) idx.1)
      apply Tile.ext
      intro idx
      have h_score_per_j : ∀ j : Fin Bk,
          scores.data (idx.1, j, PUnit.unit)
            = FA1MathBoundary.maskedScore Bk k Q K scale idx.1 j := by
        intro j
        by_cases h : k * Bk + j.val < S_k
        · show (if k * Bk + ↑j < S_k then scoresRaw.data (idx.1, j, PUnit.unit)
              else (none : WithBot ℝ)) = _
          rw [if_pos h]
          have hkLoaded : ∀ d : Fin D,
              kLoaded.data (j, d, PUnit.unit)
                = some (K (⟨k * Bk + j.val, h⟩, d, PUnit.unit)) := by
            intro d
            have := congrArg (fun t : Tile .real [Bk, D] => t.data (j, d, PUnit.unit))
              hK_loaded_eq
            simp [Tile.ofReal, FA1MathBoundary.blockIndex?_of_lt _ _ _ _ h] at this
            exact this
          rw [FA1MathBoundary.maskedScore_of_lt Bk k Q K scale idx.1 j h]
          show Option.map (· * scale) _ = _
          have h_sum :
              (∑ x : Fin D, Option.map (fun b : ℝ => Q (idx.1, x, PUnit.unit) * b)
                (kLoaded.data (j, x, PUnit.unit)) : WithBot ℝ)
              = some (∑ x : Fin D,
                  Q (idx.1, x, PUnit.unit) * K (⟨k * Bk + j.val, h⟩, x, PUnit.unit)) := by
            rw [← WithBot.sum_someTerm_eq_some]
            apply Finset.sum_congr rfl
            intro x _
            rw [hkLoaded x]
            rfl
          rw [h_sum]
          show (some _ : WithBot ℝ) = some _
          unfold StreamingAccumulator.scaledScore
          congr 1
          ring
        · show (if k * Bk + ↑j < S_k then scoresRaw.data (idx.1, j, PUnit.unit)
              else (none : WithBot ℝ)) = _
          rw [if_neg h, FA1MathBoundary.maskedScore_of_not_lt Bk k Q K scale idx.1 j h]
          rfl
      have h_mNew : mNew.data (idx.1, PUnit.unit)
          = FA1MathBoundary.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1 := by
        show max (FA1MathBoundary.mPartial Bk Q numKVBlocks K scale k idx.1)
            (mBlock.data (idx.1, PUnit.unit)) = _
        rw [FA1MathBoundary.mPartial_succ_of_lt Bk Q numKVBlocks K scale k hk idx.1]
        congr 1
        show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
            (fun j => scores.data (idx.1, j, PUnit.unit))
            = Finset.univ.sup
                (fun j => FA1MathBoundary.maskedScore Bk k Q K scale idx.1 j)
        rw [Finset.sup'_eq_sup]
        apply Finset.sup_congr rfl
        intro j _
        exact h_score_per_j j
      have h_alpha : alpha.data idx
          = some (FA1MathBoundary.alphaPartial Bk Q numKVBlocks K scale k idx.1) := by
        show WithBot.realExp _ = _
        rw [h_mNew]
        unfold FA1MathBoundary.alphaPartial
        exact FA1MathBoundary.realExp_eq_some_unbotD _
      have h_p_sum : (∑ j : Fin Bk, p.data (idx.1, j, PUnit.unit) : WithBot ℝ)
          = some (∑ j : Fin Bk,
              (WithBot.realExp
                (WithBot.realSub (FA1MathBoundary.maskedScore Bk k Q K scale idx.1 j)
                  (FA1MathBoundary.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1))
              ).unbotD 0) := by
        rw [← WithBot.sum_someTerm_eq_some]
        apply Finset.sum_congr rfl
        intro j _
        show WithBot.realExp _ = _
        rw [h_score_per_j j, h_mNew]
        show WithBot.realExp _ = _
        exact FA1MathBoundary.realExp_eq_some_unbotD _
      show Option.map₂ (fun x1 x2 : ℝ => x1 + x2)
          (Option.map (fun x : ℝ =>
            x * FA1MathBoundary.lPartial Bk Q numKVBlocks K scale k idx.1)
            (alpha.data idx))
          (∑ j : Fin Bk, p.data (idx.1, j, PUnit.unit) : WithBot ℝ)
          = some (FA1MathBoundary.lPartial Bk Q numKVBlocks K scale (k + 1) idx.1)
      rw [h_alpha, h_p_sum,
        FA1MathBoundary.lPartial_succ_of_lt Bk Q numKVBlocks K scale k hk idx.1]
      rfl
    · -- o_acc
      have h_regs_o_acc : s'.regs .real [M, D] "o_acc" = some oNew := by
        simp [s', s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
      rw [h_regs_o_acc]
      apply congrArg some
      show oNew = Tile.ofReal (fun idx : TileIndex [M, D] =>
          FA1MathBoundary.oPartial Bk Q numKVBlocks K V scale (k + 1) idx)
      apply Tile.ext
      intro idx
      have h_score_per_j : ∀ j : Fin Bk,
          scores.data (idx.1, j, PUnit.unit)
            = FA1MathBoundary.maskedScore Bk k Q K scale idx.1 j := by
        intro j
        by_cases h : k * Bk + j.val < S_k
        · show (if k * Bk + ↑j < S_k then scoresRaw.data (idx.1, j, PUnit.unit)
              else (none : WithBot ℝ)) = _
          rw [if_pos h]
          have hkLoaded : ∀ d : Fin D,
              kLoaded.data (j, d, PUnit.unit)
                = some (K (⟨k * Bk + j.val, h⟩, d, PUnit.unit)) := by
            intro d
            have := congrArg (fun t : Tile .real [Bk, D] => t.data (j, d, PUnit.unit))
              hK_loaded_eq
            simp [Tile.ofReal, FA1MathBoundary.blockIndex?_of_lt _ _ _ _ h] at this
            exact this
          rw [FA1MathBoundary.maskedScore_of_lt Bk k Q K scale idx.1 j h]
          show Option.map (· * scale) _ = _
          have h_sum :
              (∑ x : Fin D, Option.map (fun b : ℝ => Q (idx.1, x, PUnit.unit) * b)
                (kLoaded.data (j, x, PUnit.unit)) : WithBot ℝ)
              = some (∑ x : Fin D,
                  Q (idx.1, x, PUnit.unit) * K (⟨k * Bk + j.val, h⟩, x, PUnit.unit)) := by
            rw [← WithBot.sum_someTerm_eq_some]
            apply Finset.sum_congr rfl
            intro x _
            rw [hkLoaded x]
            rfl
          rw [h_sum]
          show (some _ : WithBot ℝ) = some _
          unfold StreamingAccumulator.scaledScore
          congr 1
          ring
        · show (if k * Bk + ↑j < S_k then scoresRaw.data (idx.1, j, PUnit.unit)
              else (none : WithBot ℝ)) = _
          rw [if_neg h, FA1MathBoundary.maskedScore_of_not_lt Bk k Q K scale idx.1 j h]
          rfl
      have h_mNew : mNew.data (idx.1, PUnit.unit)
          = FA1MathBoundary.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1 := by
        show max (FA1MathBoundary.mPartial Bk Q numKVBlocks K scale k idx.1)
            (mBlock.data (idx.1, PUnit.unit)) = _
        rw [FA1MathBoundary.mPartial_succ_of_lt Bk Q numKVBlocks K scale k hk idx.1]
        congr 1
        show Finset.univ.sup' ⟨⟨0, hBk⟩, Finset.mem_univ _⟩
            (fun j => scores.data (idx.1, j, PUnit.unit))
            = Finset.univ.sup
                (fun j => FA1MathBoundary.maskedScore Bk k Q K scale idx.1 j)
        rw [Finset.sup'_eq_sup]
        apply Finset.sup_congr rfl
        intro j _
        exact h_score_per_j j
      have h_alpha : alpha.data (idx.1, PUnit.unit)
          = some (FA1MathBoundary.alphaPartial Bk Q numKVBlocks K scale k idx.1) := by
        show WithBot.realExp _ = _
        rw [h_mNew]
        unfold FA1MathBoundary.alphaPartial
        exact FA1MathBoundary.realExp_eq_some_unbotD _
      have h_vLoaded : ∀ j : Fin Bk,
          vLoaded.data (j, idx.2.1, PUnit.unit)
            = some (match FA1MathBoundary.blockIndex? S_k Bk k j with
                    | some jGlobal => V (jGlobal, idx.2.1, PUnit.unit)
                    | none => 0) := by
        intro j
        have := congrArg (fun t : Tile .real [Bk, D] => t.data (j, idx.2.1, PUnit.unit))
          hV_loaded_eq
        simp [Tile.ofReal] at this
        exact this
      have h_p_per_j : ∀ j : Fin Bk,
          p.data (idx.1, j, PUnit.unit)
            = some ((WithBot.realExp
                (WithBot.realSub (FA1MathBoundary.maskedScore Bk k Q K scale idx.1 j)
                  (FA1MathBoundary.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1))
              ).unbotD 0) := by
        intro j
        show WithBot.realExp _ = _
        rw [h_score_per_j j, h_mNew]
        show WithBot.realExp _ = _
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
                        (FA1MathBoundary.maskedScore Bk k Q K scale idx.1 j)
                        (FA1MathBoundary.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1))
                    ).unbotD 0 * V (jGlobal, idx.2.1, PUnit.unit)
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
            x * FA1MathBoundary.oPartial Bk Q numKVBlocks K V scale k
              (idx.1, idx.2.1, PUnit.unit))
            (alpha.data (idx.1, PUnit.unit)))
          (∑ j : Fin Bk, Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
            (p.data (idx.1, j, PUnit.unit))
            (vLoaded.data (j, idx.2.1, PUnit.unit)) : WithBot ℝ)
          = some (FA1MathBoundary.oPartial Bk Q numKVBlocks K V scale (k + 1) idx)
      rw [h_alpha, h_pv_sum,
        FA1MathBoundary.oPartial_succ_of_lt Bk Q numKVBlocks K V scale k hk idx]
      rfl
    · -- hK : InputAt s' kReg ...
      intro idx
      simpa [s', s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hK idx
    · -- hV : InputAt s' vReg ...
      intro idx
      simpa [s', s16, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hV idx

/-- Boundary strided FA-1 forward correctness, raw form. Bundles
`fa1_forward_correct_strided_boundary_raw_of_step` with the proven
`fa1_step_strided_boundary` so callers no longer need to supply the
loop-step lemma explicitly. The output is observed only on in-range
Q rows (`s.pids 0 * M + idx.1.val < S_q`); out-of-range rows are
masked off by the kernel's store mask and outside this theorem's
guarantee. The conclusion is the raw streaming-accumulator ratio
`oPartial / lPartial` at `numKVBlocks`; the canonical-form bridge
to `attentionReal` over the logical `[S_k, D]` domain is left to
follow-up. -/
theorem fa1_forward_correct_strided_boundary_raw
    {M D Bk numKVBlocks S_q S_k : Nat}
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
        = some
            (FA1MathBoundary.oPartial Bk Q numKVBlocks K V scale
                numKVBlocks idx /
              FA1MathBoundary.lPartial Bk Q numKVBlocks K scale
                numKVBlocks idx.1) := by
  exact fa1_forward_correct_strided_boundary_raw_of_step
    qReg kReg vReg outReg
    sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
    sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV hInj
    (fun i st hi hPi =>
      fa1_step_strided_boundary hBk qReg kReg vReg
        (s.pids 0) (s.pids 1) (s.pids 2)
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale i st hi hPi)

/-- Boundary strided forward correctness, stated against the canonical
`attentionReal` over the logical `[S_k, D]` KV domain (rather than the
raw streaming-accumulator ratio). Bridges `_raw` through
`FA1MathBoundary.streaming_eq_attentionReal`. The `0 < S_k` hypothesis
ensures the running normalizer is non-zero, allowing the m-shifted
algebra to cancel cleanly. -/
theorem fa1_forward_correct_strided_boundary
    {M D Bk numKVBlocks S_q S_k : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
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
        = some (attentionReal Q K V scale idx) := by
  intro idx hIdx
  rw [fa1_forward_correct_strided_boundary_raw hBk
        qReg kReg vReg outReg
        sQB sQH sQS sQD sKB sKH sKN sKD sVB sVH sVN sVD
        sOB sOH sOM sOD Q K V scale s hQIn hQOut hK hV hInj idx hIdx]
  congr 1
  have hL := FA1MathBoundary.lPartial_final_ne_zero hBk hSk Q numKVBlocks K
    scale hSkLe idx.1
  exact FA1MathBoundary.streaming_eq_attentionReal hBk Q hSkLe K V scale idx hL

end VeriTile.Examples
