/-
VeriTile.Examples.FlashAttention1.Core.Steps

Split-out support for FlashAttention-1 v0/full-tile proofs.
-/

import VeriTile.Examples.FlashAttention1.Core.PreLoop

namespace VeriTile.Examples

open VeriTile

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
      K (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.1,
        idx.2.1, PUnit.unit)
  let vTile : Tile .real [Bk, D] :=
    Tile.ofReal fun idx : TileIndex [Bk, D] =>
      V (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.1,
        idx.2.1, PUnit.unit)
  let scores : Tile .real [M, Bk] :=
    Tile.ofReal fun idx : TileIndex [M, Bk] =>
      StreamingAccumulator.scaledScore Q K scale idx.1
        (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1)
  let mBlock : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup
        (fun j => ((StreamingAccumulator.scaledScore Q K scale idx.1
          (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j)
          : ℝ) : WithBot ℝ))⟩
  let mNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1⟩
  let alpha : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      StreamingAccumulator.alphaPartial Q numKVBlocks K scale k idx.1
  let p : Tile .real [M, Bk] :=
    Tile.ofReal fun idx : TileIndex [M, Bk] =>
      Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1
        (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1) -
          (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0)
  let lNew : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      StreamingAccumulator.lPartial Q numKVBlocks K scale (k + 1) idx.1
  let oNew : Tile .real [M, D] :=
    Tile.ofReal fun idx : TileIndex [M, D] =>
      StreamingAccumulator.oPartial Q numKVBlocks K V scale (k + 1) idx
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
  refine ⟨s', ?_, ?_⟩
  · have hKmem : ∀ (j : Fin Bk) (d : Fin D),
        s.readMem kReg ((k * Bk + j.val) * D + d.val) =
          K (StreamingAccumulator.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit) :=
      fa1_block_read kReg s K hK k hk
    have hVmem : ∀ (j : Fin Bk) (d : Fin D),
        s.readMem vReg ((k * Bk + j.val) * D + d.val) =
          V (StreamingAccumulator.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit) :=
      fa1_block_read vReg s V hV k hk
    have hOffsNReg : s1.regs .nat [Bk] "offs_n" = some offsN := by
      simp [s1, s0, offsN, BlockState.setReg]
    have hOffsDReg : s1.regs .nat [D] "offs_d" = some (Tile.vec fun d : Fin D => d.val) := by
      simp [s1, s0, BlockState.setReg, hoffs_d]
    have hKPtrsEval :
        evalOp
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
              (Op.constNat D))
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))) s1 =
          some ptrs := by
      simp only [evalOp_add, evalOp_mul, evalOp_constNat, Option.bind_eq_bind,
        Option.bind_some]
      rw [evalOp_expandDim_one_nat, hOffsNReg]
      simp only [Option.bind_some]
      rw [evalOp_expandDim_zero_nat, hOffsDReg]
      simp [ptrs, offsN, Tile.bop, NumericDType.add, NumericDType.mul]
    have hKPtrsEvalState :
        evalOp
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
              (Op.constNat D))
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
          ((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
            (Tile.vec fun j : Fin Bk => k * Bk + j.val)) =
          some ptrs := by
      simpa [s1, s0, offsN] using hKPtrsEval
    have hOffsNReg2 : s2.regs .nat [Bk] "offs_n" = some offsN := by
      simp [s2, s1, s0, offsN, BlockState.setReg]
    have hOffsDReg2 : s2.regs .nat [D] "offs_d" = some (Tile.vec fun d : Fin D => d.val) := by
      simp [s2, s1, s0, BlockState.setReg, hoffs_d]
    have hVPtrsEval :
        evalOp
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
              (Op.constNat D))
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))) s2 =
          some ptrs := by
      simp only [evalOp_add, evalOp_mul, evalOp_constNat, Option.bind_eq_bind,
        Option.bind_some]
      rw [evalOp_expandDim_one_nat, hOffsNReg2]
      simp only [Option.bind_some]
      rw [evalOp_expandDim_zero_nat, hOffsDReg2]
      simp [ptrs, offsN, Tile.bop, NumericDType.add, NumericDType.mul]
    have hVPtrsEvalState :
        evalOp
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
              (Op.constNat D))
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
          (((s.setReg "n" .nat [] (Tile.scalar k)).setReg "offs_n" .nat [Bk]
              (Tile.vec fun j : Fin Bk => k * Bk + j.val)).setReg
            "k_ptrs" .nat [Bk, D] ptrs) =
          some ptrs := by
      simpa [s2, s1, s0, offsN] using hVPtrsEval
    have h1 :
        stepStmt
          (Stmt.assign .nat [Bk] "offs_n"
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil
                (Op.ref .nat [] "n")
                (Op.constNat Bk))
              (Op.arange Bk))) s0 = some s1 := by
      simp [stepStmt, s1, s0, offsN, Tile.bop, NumericDType.add, NumericDType.mul]
      rfl
    have h2 :
        stepStmt
          (Stmt.assign .nat [Bk, D] "k_ptrs"
            (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
                (Op.constNat D))
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))) s1 =
          some s2 := by
      simp only [stepStmt, hKPtrsEval, Option.bind_some, s2]
      rfl
    have h3 :
        stepStmt
          (Stmt.assign .nat [Bk, D] "v_ptrs"
            (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
                (Op.constNat D))
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))) s2 =
          some s3 := by
      simp only [stepStmt, hVPtrsEval, Option.bind_some, s3]
      rfl
    have h4 :
        stepStmt
          (Stmt.assign .real [Bk, D] "k"
            (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, D] "k_ptrs"))
              MaskOpt.none)) s3 = some s4 := by
      simp [stepStmt, s4, s3, s2, s1, s0, kTile, ptrs, evalOp_load_region_none,
        Region.cast, hKmem]
      rfl
    have h5 :
        stepStmt
          (Stmt.assign .real [Bk, D] "v"
            (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, D] "v_ptrs"))
              MaskOpt.none)) s4 = some s5 := by
      simp [stepStmt, s5, s4, s3, s2, s1, s0, vTile, ptrs, evalOp_load_region_none,
        Region.cast, hVmem]
      rfl
    have h6 :
        stepStmt
          (Stmt.assign .real [M, Bk] "scores"
            (Op.mul .real Broadcast.scalarR
              (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
                (Op.ref .real [M, D] "q")
                (Op.transpose (batch := []) (M := Bk) (N := D)
                  (Op.ref .real [Bk, D] "k")))
              (Op.const scale))) s5 = some s6 := by
      simp only [stepStmt, evalOp_mul, Option.bind_eq_bind, Option.bind_some]
      change (((evalOp
            (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
              (Op.ref .real [M, D] "q")
              (Op.transpose (batch := []) (M := Bk) (N := D)
                (Op.ref .real [Bk, D] "k"))) s5).bind fun vx =>
          (evalOp (Op.const scale) s5).bind fun vy =>
            some (Tile.bop NumericDType.real.mul Broadcast.scalarR vx vy)).bind
        fun v => some (s5.setReg "scores" .real [M, Bk] v)) = some s6
      rw [evalOp_dot]
      rw [evalOp_transpose]
      simp [s6, s5, s4, s3, s2, s1, s0, scores, kTile,
        evalOp_transpose, Tile.bop, Tile.transpose, Tile.dot, NumericDType.mul,
        StreamingAccumulator.scaledScore, hq]
      ext idx <;> simp [Tile.ofReal, mul_comm]
    have h7 :
        stepStmt
          (Stmt.assign .real [M] "m_block"
            (Op.reduceMax ⟨1, by simp⟩ «false»
              (Op.ref .real [M, Bk] "scores"))) s6 = some s7 := by
      simp only [stepStmt]
      change ((evalOp
        (Op.reduceMax ⟨1, by simp⟩ «false»
          (Op.ref .real [M, Bk] "scores")) s6).bind
        fun v => some (s6.setReg "m_block" .real [M] v)) = some s7
      rw [VeriTile.evalOp_reduceMax]
      simp [s7, s6, mBlock, scores, Tile.reduceMax, Tile.reduceMaxDrop, Tile.ofReal,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex, hBk]
      apply congrArg (fun t =>
        (s5.setReg "scores" .real [M, Bk] scores).setReg "m_block" .real [M] t)
      ext idx
      change ((((Finset.univ : Finset (Fin Bk)).sup'
          (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
          (fun j => StreamingAccumulator.scaledScore Q K scale idx.1
            (StreamingAccumulator.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j))) : ℝ) : WithBot ℝ) =
        (Finset.univ : Finset (Fin Bk)).sup (fun j =>
          ((StreamingAccumulator.scaledScore Q K scale idx.1
            (StreamingAccumulator.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) : ℝ) : WithBot ℝ))
      rw [← WithBot.sup'_coe]
      rw [Finset.sup'_eq_sup]
    have hmNewDataStep : ∀ i : Fin M,
        max (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i) (mBlock.data (i, PUnit.unit)) =
          StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i
      simp [mBlock]
      rw [StreamingAccumulator.mPartial_succ_of_lt Q numKVBlocks K scale k hk i]
    have hAlphaDataStep : ∀ i : Fin M,
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i))
          =
          some (StreamingAccumulator.alphaPartial Q numKVBlocks K scale k i) := by
      intro i
      simpa [WithBot.realSub] using
        StreamingAccumulator.alphaPartial_toWithBot Q numKVBlocks K scale k i
    have hPDataStep : ∀ (i : Fin M) (j : Fin Bk),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (some (StreamingAccumulator.scaledScore Q K scale i
              (StreamingAccumulator.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j)))
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i))
          =
          some (Real.exp
            (StreamingAccumulator.scaledScore Q K scale i
              (StreamingAccumulator.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j) -
            WithBot.unbotD 0
              (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i))) := by
      intro i j
      cases h : StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i with
      | bot =>
          exact (StreamingAccumulator.mPartial_succ_ne_bot hBk Q numKVBlocks K scale k
            (Nat.succ_le_iff.mpr hk) i h).elim
      | coe m =>
          simp [h, WithBot.realSub]
    have h8 :
        stepStmt
          (Stmt.assign .real [M] "m_new"
            (Op.max2 (Broadcast.consSame Broadcast.nil)
              (Op.ref .real [M] "m_i")
              (Op.ref .real [M] "m_block"))) s7 = some s8 := by
      simp [stepStmt, evalOp_max2, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        mNew, mBlock, Tile.bop, hm, hmNewDataStep]
    have h9 :
        stepStmt
          (Stmt.assign .real [M] "alpha"
            (Op.exp
              (Op.sub .real (Broadcast.consSame Broadcast.nil)
                (Op.ref .real [M] "m_i")
                (Op.ref .real [M] "m_new")))) s8 = some s9 := by
      simp [stepStmt, evalOp_exp, evalOp_sub, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        alpha, mNew, Tile.bop, Tile.uop, Tile.ofReal, NumericDType.sub, hm, hmNewDataStep,
        hAlphaDataStep]
    have h10 :
        stepStmt
          (Stmt.assign .real [M, Bk] "p"
            (Op.exp
              (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
                (Op.ref .real [M, Bk] "scores")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "m_new"))))) s9 =
          some s10 := by
      simp only [stepStmt, evalOp_exp, evalOp_sub, Option.bind_eq_bind, Option.bind_some]
      unfold evalOp
      simp [VeriTile.evalOp_expandDim, VeriTile.evalOp_expandDim_ref,
        VeriTile.evalOp_ref,
        evalOp_expandDim_one_real,
        s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        p, scores, mNew, Tile.bop, Tile.uop, Tile.ofReal, Tile.expandDim, NumericDType.sub,
        hmNewDataStep, hPDataStep]
      change some (s9.setReg "p" .real [M, Bk]
          ({ data := fun idx : TileIndex [M, Bk] =>
            WithBot.realExp
              (Option.map
                (fun b => StreamingAccumulator.scaledScore Q K scale idx.1
                  (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) idx.2.1) - b)
                (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1)) } :
            Tile .real [M, Bk])) =
        some (s9.setReg "p" .real [M, Bk] p)
      apply congrArg (fun t => some (s9.setReg "p" .real [M, Bk] t))
      ext idx
      simpa [p, Tile.ofReal] using hPDataStep idx.1 idx.2.1
    have h11 :
        stepStmt
          (Stmt.assign .real [M] "l_new"
            (Op.add .real (Broadcast.consSame Broadcast.nil)
              (Op.mul .real (Broadcast.consSame Broadcast.nil)
                (Op.ref .real [M] "alpha")
                (Op.ref .real [M] "l_i"))
                (Op.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
                (Op.ref .real [M, Bk] "p")))) s10 = some s11 := by
      simp [stepStmt, evalOp_add, evalOp_mul, evalOp_reduceSum,
        s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        lNew, alpha, p, Tile.bop, Tile.ofReal, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        hBk, NumericDType.add, NumericDType.mul, hl]
      unfold evalOp
      simp [s11, lNew, alpha, p, Tile.bop, Tile.ofReal, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        hBk, NumericDType.add, NumericDType.mul, WithBot.realAdd, WithBot.realMul,
        StreamingAccumulator.lPartial_succ_of_lt Q numKVBlocks K scale k hk]
      rfl
    have h12 :
        stepStmt
          (Stmt.assign .real [M, D] "o_acc"
            (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "alpha"))
                (Op.ref .real [M, D] "o_acc"))
              (Op.dot (batch := []) (M := M) (K := Bk) (N := D)
                  (Op.ref .real [M, Bk] "p")
                  (Op.ref .real [Bk, D] "v")))) s11 = some s12 := by
        simp [stepStmt, evalOp_add, evalOp_mul, evalOp_dot,
          s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
          oNew, alpha, p, vTile, Tile.bop, Tile.ofReal, Tile.dot,
          Tile.expandDim, TileShape.dropInsertedIndex, NumericDType.add,
          NumericDType.mul, ho]
        unfold evalOp
        simp [Tile.dot, Tile.expandDim, TileShape.dropInsertedIndex,
          StreamingAccumulator.oPartial_succ_of_lt Q numKVBlocks K V scale k hk]
        simp only [Option.bind, Option.map]
    have h13 :
        stepStmt
          (Stmt.assign .real [M] "m_i" (Op.ref .real [M] "m_new")) s12 =
          some s13 := by
      simp [stepStmt, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
    have h14 :
        stepStmt
          (Stmt.assign .real [M] "l_i" (Op.ref .real [M] "l_new")) s13 =
          some s' := by
      simp [stepStmt, s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
    change stepStmts (fa1LoopBody kReg vReg M D Bk scale) s0 = some s'
    unfold fa1LoopBody
    rw [stepStmts.cons_some h1]
    rw [stepStmts.cons_some h2]
    rw [stepStmts.cons_some h3]
    rw [stepStmts.cons_some h4]
    rw [stepStmts.cons_some h5]
    rw [stepStmts.cons_some h6]
    rw [stepStmts.cons_some h7]
    rw [stepStmts.cons_some h8]
    rw [stepStmts.cons_some h9]
    rw [stepStmts.cons_some h10]
    rw [stepStmts.cons_some h11]
    rw [stepStmts.cons_some h12]
    rw [stepStmts.cons_some h13]
    rw [stepStmts.cons_some h14]
    simp [stepStmts]
  · have hmNewData : ∀ i : Fin M,
        max (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
          (some ((Finset.univ : Finset (Fin Bk)).sup'
            (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
          StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i
      rw [StreamingAccumulator.mPartial_succ_of_lt Q numKVBlocks K scale k hk i]
      congr 1
      change ((((Finset.univ : Finset (Fin Bk)).sup'
          (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
          (fun j => (Finset.univ.sum (fun d : Fin D =>
            Q (i, d, PUnit.unit) *
              K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) : ℝ) : WithBot ℝ) =
        (Finset.univ : Finset (Fin Bk)).sup (fun j =>
          ((StreamingAccumulator.scaledScore Q K scale i
            (StreamingAccumulator.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) : ℝ) : WithBot ℝ))
      rw [← WithBot.sup'_coe]
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      simp [StreamingAccumulator.scaledScore, mul_comm]
    have hmNewDataOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i) ⊔
          (some ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
          StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i H
      rw [StreamingAccumulator.mPartial_succ_of_lt Q numKVBlocks K scale k hk i]
      congr 1
      change ((((Finset.univ : Finset (Fin Bk)).sup' H
          (fun j => (Finset.univ.sum (fun d : Fin D =>
            Q (i, d, PUnit.unit) *
              K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) : ℝ) : WithBot ℝ) =
        (Finset.univ : Finset (Fin Bk)).sup (fun j =>
          ((StreamingAccumulator.scaledScore Q K scale i
            (StreamingAccumulator.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) : ℝ) : WithBot ℝ))
      rw [← WithBot.sup'_coe]
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      simp [StreamingAccumulator.scaledScore, mul_comm]
    have hmNewDataComm : ∀ i : Fin M,
        max (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
          (some ((Finset.univ : Finset (Fin Bk)).sup'
            (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit)))))) =
          StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i
      rw [← hmNewData i]
      congr 2
      apply Finset.sup'_congr (H := by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩) rfl
      intro j _
      rw [mul_comm]
    have hAlphaData : ∀ i : Fin M,
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i))
          =
          some (StreamingAccumulator.alphaPartial Q numKVBlocks K scale k i) := by
      intro i
      simpa [WithBot.realSub] using
        StreamingAccumulator.alphaPartial_toWithBot Q numKVBlocks K scale k i
    have hPData : ∀ (i : Fin M) (j : Fin Bk),
        WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i))
          =
          some (Real.exp (StreamingAccumulator.scaledScore Q K scale i
            (StreamingAccumulator.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) -
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j
      have hm : StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i ≠ ⊥ :=
        StreamingAccumulator.mPartial_succ_ne_bot hBk Q numKVBlocks K scale k
          (Nat.succ_le_iff.mpr hk) i
      obtain ⟨m, hm_eq⟩ := WithBot.ne_bot_iff_exists.mp hm
      rw [← hm_eq]
      simp [StreamingAccumulator.scaledScore, mul_comm]
    have hAlphaDataOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (StreamingAccumulator.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      change WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (StreamingAccumulator.alphaPartial Q numKVBlocks K scale k i)
      have hM :
          @max (Option ℝ) Option.instMax
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
            (some ((Finset.univ : Finset (Fin Bk)).sup' H
              (fun j =>
                (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
            StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
        have hOptionMax :
            @max (Option ℝ) Option.instMax
              (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))
            =
            max (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) := by
          cases StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i <;> rfl
        rw [hOptionMax]
        exact hmNewDataOf i H
      exact (congrArg
        (fun z => WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i) z)) hM).trans
        (hAlphaData i)
    have hPDataOf : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (@max (Option ℝ) Option.instMax
              (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
        =
        some (Real.exp (StreamingAccumulator.scaledScore Q K scale i
          (StreamingAccumulator.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      change WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
              (@max (Option ℝ) Option.instMax
                (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (Real.exp (StreamingAccumulator.scaledScore Q K scale i
            (StreamingAccumulator.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) -
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0))
      have hM :
          @max (Option ℝ) Option.instMax
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
            (some ((Finset.univ : Finset (Fin Bk)).sup' H
              (fun j =>
                (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
            StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
        have hOptionMax :
            @max (Option ℝ) Option.instMax
              (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))
            =
            max (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) := by
          cases StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i <;> rfl
        rw [hOptionMax]
        exact hmNewDataOf i H
      exact (congrArg
        (fun z => WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b) z)) hM).trans
          (hPData i j)
    have hAlphaDataComm : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (StreamingAccumulator.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      have hSup :
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))) =
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks k
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
                K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (@max (Option ℝ) Option.instMax
              (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (StreamingAccumulator.scaledScore Q K scale i
          (StreamingAccumulator.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      have hSup :
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))) =
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks k
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
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (StreamingAccumulator.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataComm i H
    have hPDataMaxComm : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            scale * (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (@max (Option ℝ) Option.instMax
              (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (StreamingAccumulator.scaledScore Q K scale i
          (StreamingAccumulator.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataComm i j H
    have hAlphaDataMaxComm' : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (max (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (StreamingAccumulator.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataComm i H
    have hPDataMaxComm' : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            scale * (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (max (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (StreamingAccumulator.scaledScore Q K scale i
          (StreamingAccumulator.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataComm i j H
    have hAlphaDataMaxOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (max (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (StreamingAccumulator.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataOf i H
    have hPDataMaxOf : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (max (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
        =
        some (Real.exp (StreamingAccumulator.scaledScore Q K scale i
          (StreamingAccumulator.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataOf i j H
    have max_eq_sup : ∀ (a b : WithBot ℝ), max a b = a ⊔ b := by
      intro a b
      rfl
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hpidReg
    · simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hpid
    · simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hoffs_m
    · simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hoffs_d
    · simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
        using hq
    · simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        mNew, BlockState.setReg]
    · simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        lNew, BlockState.setReg]
    · simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        oNew, BlockState.setReg]
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
      K (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.1,
        idx.2.1, PUnit.unit)
  let vTile : Tile .real [Bk, D] :=
    Tile.ofReal fun idx : TileIndex [Bk, D] =>
      V (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.1,
        idx.2.1, PUnit.unit)
  let scores : Tile .real [M, Bk] :=
    Tile.ofReal fun idx : TileIndex [M, Bk] =>
      StreamingAccumulator.scaledScore Q K scale idx.1
        (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1)
  let mBlock : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup
        (fun j => ((StreamingAccumulator.scaledScore Q K scale idx.1
          (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j)
          : ℝ) : WithBot ℝ))⟩
  let mNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1⟩
  let alpha : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      StreamingAccumulator.alphaPartial Q numKVBlocks K scale k idx.1
  let p : Tile .real [M, Bk] :=
    Tile.ofReal fun idx : TileIndex [M, Bk] =>
      Real.exp (StreamingAccumulator.scaledScore Q K scale idx.1
        (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1) -
          (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0)
  let lNew : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      StreamingAccumulator.lPartial Q numKVBlocks K scale (k + 1) idx.1
  let oNew : Tile .real [M, D] :=
    Tile.ofReal fun idx : TileIndex [M, D] =>
      StreamingAccumulator.oPartial Q numKVBlocks K V scale (k + 1) idx
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
  refine ⟨s', ?_, ?_⟩
  · have hKmem : ∀ (j : Fin Bk) (d : Fin D),
        s.readMem kReg (batch * sKB + headIdx * sKH
            + (k * Bk + j.val) * sKN + d.val * sKD) =
          K (StreamingAccumulator.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit) :=
      fa1_block_read_strided kReg s
        (batch * sKB + headIdx * sKH) sKN sKD K hK k hk
    have hVmem : ∀ (j : Fin Bk) (d : Fin D),
        s.readMem vReg (batch * sVB + headIdx * sVH
            + (k * Bk + j.val) * sVN + d.val * sVD) =
          V (StreamingAccumulator.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit) :=
      fa1_block_read_strided vReg s
        (batch * sVB + headIdx * sVH) sVN sVD V hV k hk
    have hOffsNReg : s1.regs .nat [Bk] "offs_n" = some offsN := by
      simp [s1, s0, offsN, BlockState.setReg]
    have hOffsDReg : s1.regs .nat [D] "offs_d" = some (Tile.vec fun d : Fin D => d.val) := by
      simp [s1, s0, BlockState.setReg, hoffs_d]
    have hKBaseReg : s1.regs .nat [] "k_base_off" =
        some (Tile.scalar (batch * sKB + headIdx * sKH)) := by
      simp [s1, s0, BlockState.setReg, hk_base]
    have hKPtrsEval :
        evalOp
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.add .nat Broadcast.scalarL
              (Op.ref .nat [] "k_base_off")
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
                (Op.constNat sKN)))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
              (Op.constNat sKD))) s1 =
          some kPtrs := by
      simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
        Option.bind_eq_bind, Option.bind_some]
      rw [hKBaseReg]
      simp only [Option.bind_some]
      rw [evalOp_expandDim_one_nat, hOffsNReg]
      simp only [Option.bind_some]
      rw [evalOp_expandDim_zero_nat, hOffsDReg]
      simp [kPtrs, kBase, offsN, Tile.bop, NumericDType.add, NumericDType.mul]
    have hOffsNReg2 : s2.regs .nat [Bk] "offs_n" = some offsN := by
      simp [s2, s1, s0, offsN, BlockState.setReg]
    have hOffsDReg2 : s2.regs .nat [D] "offs_d" = some (Tile.vec fun d : Fin D => d.val) := by
      simp [s2, s1, s0, BlockState.setReg, hoffs_d]
    have hVBaseReg2 : s2.regs .nat [] "v_base_off" =
        some (Tile.scalar (batch * sVB + headIdx * sVH)) := by
      simp [s2, s1, s0, BlockState.setReg, hv_base]
    have hVPtrsEval :
        evalOp
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.add .nat Broadcast.scalarL
              (Op.ref .nat [] "v_base_off")
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
                (Op.constNat sVN)))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
              (Op.constNat sVD))) s2 =
          some vPtrs := by
      simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
        Option.bind_eq_bind, Option.bind_some]
      rw [hVBaseReg2]
      simp only [Option.bind_some]
      rw [evalOp_expandDim_one_nat, hOffsNReg2]
      simp only [Option.bind_some]
      rw [evalOp_expandDim_zero_nat, hOffsDReg2]
      simp [vPtrs, vBase, offsN, Tile.bop, NumericDType.add, NumericDType.mul]
    have h1 :
        stepStmt
          (Stmt.assign .nat [Bk] "offs_n"
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil
                (Op.ref .nat [] "n")
                (Op.constNat Bk))
              (Op.arange Bk))) s0 = some s1 := by
      simp [stepStmt, s1, s0, offsN, Tile.bop, NumericDType.add, NumericDType.mul]
      rfl
    have h2 :
        stepStmt
          (Stmt.assign .nat [Bk, D] "k_ptrs"
            (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.add .nat Broadcast.scalarL
                (Op.ref .nat [] "k_base_off")
                (Op.mul .nat Broadcast.scalarR
                  (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
                  (Op.constNat sKN)))
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
                (Op.constNat sKD)))) s1 = some s2 := by
      simp only [stepStmt, hKPtrsEval, Option.bind_some, s2]
      rfl
    have h3 :
        stepStmt
          (Stmt.assign .nat [Bk, D] "v_ptrs"
            (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.add .nat Broadcast.scalarL
                (Op.ref .nat [] "v_base_off")
                (Op.mul .nat Broadcast.scalarR
                  (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
                  (Op.constNat sVN)))
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
                (Op.constNat sVD)))) s2 = some s3 := by
      simp only [stepStmt, hVPtrsEval, Option.bind_some, s3]
      rfl
    have h4 :
        stepStmt
          (Stmt.assign .real [Bk, D] "k"
            (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, D] "k_ptrs"))
              MaskOpt.none)) s3 = some s4 := by
      simp [stepStmt, s4, s3, s2, s1, s0, kTile, kPtrs, kBase, evalOp_load_region_none,
        Region.cast, hKmem]
      rfl
    have h5 :
        stepStmt
          (Stmt.assign .real [Bk, D] "v"
            (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, D] "v_ptrs"))
              MaskOpt.none)) s4 = some s5 := by
      simp [stepStmt, s5, s4, s3, s2, s1, s0, vTile, vPtrs, vBase, evalOp_load_region_none,
        Region.cast, hVmem]
      rfl
    have h6 :
        stepStmt
          (Stmt.assign .real [M, Bk] "scores"
            (Op.mul .real Broadcast.scalarR
              (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
                (Op.ref .real [M, D] "q")
                (Op.transpose (batch := []) (M := Bk) (N := D)
                  (Op.ref .real [Bk, D] "k")))
              (Op.const scale))) s5 = some s6 := by
      simp only [stepStmt, evalOp_mul, Option.bind_eq_bind, Option.bind_some]
      change (((evalOp
            (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
              (Op.ref .real [M, D] "q")
              (Op.transpose (batch := []) (M := Bk) (N := D)
                (Op.ref .real [Bk, D] "k"))) s5).bind fun vx =>
          (evalOp (Op.const scale) s5).bind fun vy =>
            some (Tile.bop NumericDType.real.mul Broadcast.scalarR vx vy)).bind
        fun v => some (s5.setReg "scores" .real [M, Bk] v)) = some s6
      rw [evalOp_dot]
      rw [evalOp_transpose]
      simp [s6, s5, s4, s3, s2, s1, s0, scores, kTile,
        evalOp_transpose, Tile.bop, Tile.transpose, Tile.dot, NumericDType.mul,
        StreamingAccumulator.scaledScore, hq]
      ext idx <;> simp [Tile.ofReal, mul_comm]
    have h7 :
        stepStmt
          (Stmt.assign .real [M] "m_block"
            (Op.reduceMax ⟨1, by simp⟩ «false»
              (Op.ref .real [M, Bk] "scores"))) s6 = some s7 := by
      simp only [stepStmt]
      change ((evalOp
        (Op.reduceMax ⟨1, by simp⟩ «false»
          (Op.ref .real [M, Bk] "scores")) s6).bind
        fun v => some (s6.setReg "m_block" .real [M] v)) = some s7
      rw [VeriTile.evalOp_reduceMax]
      simp [s7, s6, mBlock, scores, Tile.reduceMax, Tile.reduceMaxDrop, Tile.ofReal,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex, hBk]
      apply congrArg (fun t =>
        (s5.setReg "scores" .real [M, Bk] scores).setReg "m_block" .real [M] t)
      ext idx
      change ((((Finset.univ : Finset (Fin Bk)).sup'
          (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
          (fun j => StreamingAccumulator.scaledScore Q K scale idx.1
            (StreamingAccumulator.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j))) : ℝ) : WithBot ℝ) =
        (Finset.univ : Finset (Fin Bk)).sup (fun j =>
          ((StreamingAccumulator.scaledScore Q K scale idx.1
            (StreamingAccumulator.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) : ℝ) : WithBot ℝ))
      rw [← WithBot.sup'_coe]
      rw [Finset.sup'_eq_sup]
    have hmNewDataStep : ∀ i : Fin M,
        max (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i) (mBlock.data (i, PUnit.unit)) =
          StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i
      simp [mBlock]
      rw [StreamingAccumulator.mPartial_succ_of_lt Q numKVBlocks K scale k hk i]
    have hAlphaDataStep : ∀ i : Fin M,
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i))
          =
          some (StreamingAccumulator.alphaPartial Q numKVBlocks K scale k i) := by
      intro i
      simpa [WithBot.realSub] using
        StreamingAccumulator.alphaPartial_toWithBot Q numKVBlocks K scale k i
    have hPDataStep : ∀ (i : Fin M) (j : Fin Bk),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (some (StreamingAccumulator.scaledScore Q K scale i
              (StreamingAccumulator.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j)))
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i))
          =
          some (Real.exp
            (StreamingAccumulator.scaledScore Q K scale i
              (StreamingAccumulator.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j) -
            WithBot.unbotD 0
              (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i))) := by
      intro i j
      cases h : StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i with
      | bot =>
          exact (StreamingAccumulator.mPartial_succ_ne_bot hBk Q numKVBlocks K scale k
            (Nat.succ_le_iff.mpr hk) i h).elim
      | coe m =>
          simp [h, WithBot.realSub]
    have h8 :
        stepStmt
          (Stmt.assign .real [M] "m_new"
            (Op.max2 (Broadcast.consSame Broadcast.nil)
              (Op.ref .real [M] "m_i")
              (Op.ref .real [M] "m_block"))) s7 = some s8 := by
      simp [stepStmt, evalOp_max2, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        mNew, mBlock, Tile.bop, hm, hmNewDataStep]
    have h9 :
        stepStmt
          (Stmt.assign .real [M] "alpha"
            (Op.exp
              (Op.sub .real (Broadcast.consSame Broadcast.nil)
                (Op.ref .real [M] "m_i")
                (Op.ref .real [M] "m_new")))) s8 = some s9 := by
      simp [stepStmt, evalOp_exp, evalOp_sub, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        alpha, mNew, Tile.bop, Tile.uop, Tile.ofReal, NumericDType.sub, hm, hmNewDataStep,
        hAlphaDataStep]
    have h10 :
        stepStmt
          (Stmt.assign .real [M, Bk] "p"
            (Op.exp
              (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
                (Op.ref .real [M, Bk] "scores")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "m_new"))))) s9 =
          some s10 := by
      simp only [stepStmt, evalOp_exp, evalOp_sub, Option.bind_eq_bind, Option.bind_some]
      unfold evalOp
      simp [VeriTile.evalOp_expandDim, VeriTile.evalOp_expandDim_ref,
        VeriTile.evalOp_ref,
        evalOp_expandDim_one_real,
        s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        p, scores, mNew, Tile.bop, Tile.uop, Tile.ofReal, Tile.expandDim, NumericDType.sub,
        hmNewDataStep, hPDataStep]
      change some (s9.setReg "p" .real [M, Bk]
          ({ data := fun idx : TileIndex [M, Bk] =>
            WithBot.realExp
              (Option.map
                (fun b => StreamingAccumulator.scaledScore Q K scale idx.1
                  (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) idx.2.1) - b)
                (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) idx.1)) } :
            Tile .real [M, Bk])) =
        some (s9.setReg "p" .real [M, Bk] p)
      apply congrArg (fun t => some (s9.setReg "p" .real [M, Bk] t))
      ext idx
      simpa [p, Tile.ofReal] using hPDataStep idx.1 idx.2.1
    have h11 :
        stepStmt
          (Stmt.assign .real [M] "l_new"
            (Op.add .real (Broadcast.consSame Broadcast.nil)
              (Op.mul .real (Broadcast.consSame Broadcast.nil)
                (Op.ref .real [M] "alpha")
                (Op.ref .real [M] "l_i"))
                (Op.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
                (Op.ref .real [M, Bk] "p")))) s10 = some s11 := by
      simp [stepStmt, evalOp_add, evalOp_mul, evalOp_reduceSum,
        s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        lNew, alpha, p, Tile.bop, Tile.ofReal, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        hBk, NumericDType.add, NumericDType.mul, hl]
      unfold evalOp
      simp [s11, lNew, alpha, p, Tile.bop, Tile.ofReal, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        hBk, NumericDType.add, NumericDType.mul, WithBot.realAdd, WithBot.realMul,
        StreamingAccumulator.lPartial_succ_of_lt Q numKVBlocks K scale k hk]
      rfl
    have h12 :
        stepStmt
          (Stmt.assign .real [M, D] "o_acc"
            (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "alpha"))
                (Op.ref .real [M, D] "o_acc"))
              (Op.dot (batch := []) (M := M) (K := Bk) (N := D)
                (Op.ref .real [M, Bk] "p")
                (Op.ref .real [Bk, D] "v")))) s11 = some s12 := by
      simp [stepStmt, evalOp_add, evalOp_mul, evalOp_dot,
        s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        oNew, alpha, p, vTile, Tile.bop, Tile.ofReal, Tile.dot,
        Tile.expandDim, TileShape.dropInsertedIndex, NumericDType.add,
        NumericDType.mul, ho]
      unfold evalOp
      simp [Tile.dot, Tile.expandDim, TileShape.dropInsertedIndex,
        StreamingAccumulator.oPartial_succ_of_lt Q numKVBlocks K V scale k hk]
      simp only [Option.bind, Option.map]
    have h13 :
        stepStmt
          (Stmt.assign .real [M] "m_i" (Op.ref .real [M] "m_new")) s12 =
          some s13 := by
      simp [stepStmt, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
    have h14 :
        stepStmt
          (Stmt.assign .real [M] "l_i" (Op.ref .real [M] "l_new")) s13 =
          some s' := by
      simp [stepStmt, s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
    change stepStmts (fa1LoopBodyStrided kReg vReg M D Bk sKN sKD sVN sVD scale) s0 = some s'
    unfold fa1LoopBodyStrided
    rw [stepStmts.cons_some h1]
    rw [stepStmts.cons_some h2]
    rw [stepStmts.cons_some h3]
    rw [stepStmts.cons_some h4]
    rw [stepStmts.cons_some h5]
    rw [stepStmts.cons_some h6]
    rw [stepStmts.cons_some h7]
    rw [stepStmts.cons_some h8]
    rw [stepStmts.cons_some h9]
    rw [stepStmts.cons_some h10]
    rw [stepStmts.cons_some h11]
    rw [stepStmts.cons_some h12]
    rw [stepStmts.cons_some h13]
    rw [stepStmts.cons_some h14]
    simp [stepStmts]
  · have hmNewData : ∀ i : Fin M,
        max (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
          (some ((Finset.univ : Finset (Fin Bk)).sup'
            (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
          StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i
      rw [StreamingAccumulator.mPartial_succ_of_lt Q numKVBlocks K scale k hk i]
      congr 1
      change ((((Finset.univ : Finset (Fin Bk)).sup'
          (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
          (fun j => (Finset.univ.sum (fun d : Fin D =>
            Q (i, d, PUnit.unit) *
              K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) : ℝ) : WithBot ℝ) =
        (Finset.univ : Finset (Fin Bk)).sup (fun j =>
          ((StreamingAccumulator.scaledScore Q K scale i
            (StreamingAccumulator.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) : ℝ) : WithBot ℝ))
      rw [← WithBot.sup'_coe]
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      simp [StreamingAccumulator.scaledScore, mul_comm]
    have hmNewDataOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i) ⊔
          (some ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
          StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i H
      rw [StreamingAccumulator.mPartial_succ_of_lt Q numKVBlocks K scale k hk i]
      congr 1
      change ((((Finset.univ : Finset (Fin Bk)).sup' H
          (fun j => (Finset.univ.sum (fun d : Fin D =>
            Q (i, d, PUnit.unit) *
              K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)) : ℝ) : WithBot ℝ) =
        (Finset.univ : Finset (Fin Bk)).sup (fun j =>
          ((StreamingAccumulator.scaledScore Q K scale i
            (StreamingAccumulator.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) : ℝ) : WithBot ℝ))
      rw [← WithBot.sup'_coe]
      rw [Finset.sup'_eq_sup]
      apply Finset.sup_congr rfl
      intro j _
      simp [StreamingAccumulator.scaledScore, mul_comm]
    have hmNewDataComm : ∀ i : Fin M,
        max (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
          (some ((Finset.univ : Finset (Fin Bk)).sup'
            (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit)))))) =
          StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
      intro i
      rw [← hmNewData i]
      congr 2
      apply Finset.sup'_congr (H := by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩) rfl
      intro j _
      rw [mul_comm]
    have hAlphaData : ∀ i : Fin M,
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i))
          =
          some (StreamingAccumulator.alphaPartial Q numKVBlocks K scale k i) := by
      intro i
      simpa [WithBot.realSub] using
        StreamingAccumulator.alphaPartial_toWithBot Q numKVBlocks K scale k i
    have hPData : ∀ (i : Fin M) (j : Fin Bk),
        WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i))
          =
          some (Real.exp (StreamingAccumulator.scaledScore Q K scale i
            (StreamingAccumulator.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) -
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j
      have hm : StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i ≠ ⊥ :=
        StreamingAccumulator.mPartial_succ_ne_bot hBk Q numKVBlocks K scale k
          (Nat.succ_le_iff.mpr hk) i
      obtain ⟨m, hm_eq⟩ := WithBot.ne_bot_iff_exists.mp hm
      rw [← hm_eq]
      simp [StreamingAccumulator.scaledScore, mul_comm]
    have hAlphaDataOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (StreamingAccumulator.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      change WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (StreamingAccumulator.alphaPartial Q numKVBlocks K scale k i)
      have hM :
          @max (Option ℝ) Option.instMax
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
            (some ((Finset.univ : Finset (Fin Bk)).sup' H
              (fun j =>
                (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
            StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
        have hOptionMax :
            @max (Option ℝ) Option.instMax
              (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))
            =
            max (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) := by
          cases StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i <;> rfl
        rw [hOptionMax]
        exact hmNewDataOf i H
      exact (congrArg
        (fun z => WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i) z)) hM).trans
        (hAlphaData i)
    have hPDataOf : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (@max (Option ℝ) Option.instMax
              (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
        =
        some (Real.exp (StreamingAccumulator.scaledScore Q K scale i
          (StreamingAccumulator.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      change WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
              (@max (Option ℝ) Option.instMax
                (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (Real.exp (StreamingAccumulator.scaledScore Q K scale i
            (StreamingAccumulator.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) j) -
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0))
      have hM :
          @max (Option ℝ) Option.instMax
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
            (some ((Finset.univ : Finset (Fin Bk)).sup' H
              (fun j =>
                (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) =
            StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i := by
        have hOptionMax :
            @max (Option ℝ) Option.instMax
              (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))
            =
            max (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale))) := by
          cases StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i <;> rfl
        rw [hOptionMax]
        exact hmNewDataOf i H
      exact (congrArg
        (fun z => WithBot.realExp
          (Option.map
            (fun b =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b) z)) hM).trans
          (hPData i j)
    have hAlphaDataComm : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (StreamingAccumulator.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      have hSup :
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))) =
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks k
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
                K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (@max (Option ℝ) Option.instMax
              (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (StreamingAccumulator.scaledScore Q K scale i
          (StreamingAccumulator.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      have hSup :
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              scale * (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))) =
          ((Finset.univ : Finset (Fin Bk)).sup' H
            (fun j =>
              (Finset.univ.sum (fun d : Fin D =>
                Q (i, d, PUnit.unit) *
                  K (StreamingAccumulator.blockIndex Bk numKVBlocks k
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
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (@max (Option ℝ) Option.instMax
                (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (StreamingAccumulator.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataComm i H
    have hPDataMaxComm : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            scale * (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (@max (Option ℝ) Option.instMax
              (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (StreamingAccumulator.scaledScore Q K scale i
          (StreamingAccumulator.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataComm i j H
    have hAlphaDataMaxComm' : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (max (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    scale * (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
          =
          some (StreamingAccumulator.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataComm i H
    have hPDataMaxComm' : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            scale * (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) - b)
            (max (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  scale * (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))))))))
        =
        some (Real.exp (StreamingAccumulator.scaledScore Q K scale i
          (StreamingAccumulator.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataComm i j H
    have hAlphaDataMaxOf : ∀ (i : Fin M)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (max (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
                (some ((Finset.univ : Finset (Fin Bk)).sup' H
                  (fun j =>
                    (Finset.univ.sum (fun d : Fin D =>
                    Q (i, d, PUnit.unit) *
                      K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
          =
          some (StreamingAccumulator.alphaPartial Q numKVBlocks K scale k i) := by
      intro i H
      exact hAlphaDataOf i H
    have hPDataMaxOf : ∀ (i : Fin M) (j : Fin Bk)
        (H : (Finset.univ : Finset (Fin Bk)).Nonempty),
      WithBot.realExp
        (Option.map
          (fun b =>
            (Finset.univ.sum (fun d : Fin D =>
              Q (i, d, PUnit.unit) *
                K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale - b)
            (max (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale k i)
              (some ((Finset.univ : Finset (Fin Bk)).sup' H
                (fun j =>
                  (Finset.univ.sum (fun d : Fin D =>
                  Q (i, d, PUnit.unit) *
                    K (StreamingAccumulator.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit))) * scale)))))
        =
        some (Real.exp (StreamingAccumulator.scaledScore Q K scale i
          (StreamingAccumulator.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j) -
          (StreamingAccumulator.mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0)) := by
      intro i j H
      exact hPDataOf i j H
    have max_eq_sup : ∀ (a b : WithBot ℝ), max a b = a ⊔ b := by
      intro a b
      rfl
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simpa using hpids0
    · simpa using hpids1
    · simpa using hpids2
    · simpa using hpid_qb
    · simpa using hpid_h
    · simpa using hpid_b
    · simpa using hq_base
    · simpa using hk_base
    · simpa using hv_base
    · simpa using ho_base
    · simpa using hoffs_m
    · simpa using hoffs_d
    · simpa using hq
    · simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        mNew, BlockState.setReg]
    · simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        lNew, BlockState.setReg]
    · simpa [s', s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        oNew, BlockState.setReg]
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
      K (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.1,
        idx.2.1, PUnit.unit)
  let vTile : Tile .real [Bk, D] :=
    Tile.ofReal fun idx : TileIndex [Bk, D] =>
      V (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.1,
        idx.2.1, PUnit.unit)
  let scoresRaw : Tile .real [M, Bk] :=
    Tile.ofReal fun idx : TileIndex [M, Bk] =>
      StreamingAccumulator.scaledScore Q K scale idx.1
        (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1)
  let causal : Tile .bool [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      decide (k * Bk + idx.2.1.val ≤ qb * M + idx.1.val)⟩
  let scores : Tile .real [M, Bk] :=
    ⟨fun idx : TileIndex [M, Bk] =>
      FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
        (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1)⟩
  let mBlock : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup
        (fun jLocal =>
          FA1MathCausal.maskedScore (qb * M) Q K scale idx.1
            (StreamingAccumulator.blockIndex Bk numKVBlocks k
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
            (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1))
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
  refine ⟨s', ?_, ?_⟩
  · have hKmem : ∀ (j : Fin Bk) (d : Fin D),
        s.readMem kReg (batch * sKB + headIdx * sKH
            + (k * Bk + j.val) * sKN + d.val * sKD) =
          K (StreamingAccumulator.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit) :=
      fa1_block_read_strided kReg s
        (batch * sKB + headIdx * sKH) sKN sKD K hK k hk
    have hVmem : ∀ (j : Fin Bk) (d : Fin D),
        s.readMem vReg (batch * sVB + headIdx * sVH
            + (k * Bk + j.val) * sVN + d.val * sVD) =
          V (StreamingAccumulator.blockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) j, d, PUnit.unit) :=
      fa1_block_read_strided vReg s
        (batch * sVB + headIdx * sVH) sVN sVD V hV k hk
    have hOffsNReg : s1.regs .nat [Bk] "offs_n" = some offsN := by
      simp [s1, s0, offsN, BlockState.setReg]
    have hOffsDReg : s1.regs .nat [D] "offs_d" = some (Tile.vec fun d : Fin D => d.val) := by
      simp [s1, s0, BlockState.setReg, hoffs_d]
    have hKBaseReg : s1.regs .nat [] "k_base_off" =
        some (Tile.scalar (batch * sKB + headIdx * sKH)) := by
      simp [s1, s0, BlockState.setReg, hk_base]
    have hKPtrsEval :
        evalOp
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.add .nat Broadcast.scalarL
              (Op.ref .nat [] "k_base_off")
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
                (Op.constNat sKN)))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
              (Op.constNat sKD))) s1 =
          some kPtrs := by
      simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
        Option.bind_eq_bind, Option.bind_some]
      rw [hKBaseReg]
      simp only [Option.bind_some]
      rw [evalOp_expandDim_one_nat, hOffsNReg]
      simp only [Option.bind_some]
      rw [evalOp_expandDim_zero_nat, hOffsDReg]
      simp [kPtrs, kBase, offsN, Tile.bop, NumericDType.add, NumericDType.mul]
    have hOffsNReg2 : s2.regs .nat [Bk] "offs_n" = some offsN := by
      simp [s2, s1, s0, offsN, BlockState.setReg]
    have hOffsDReg2 : s2.regs .nat [D] "offs_d" = some (Tile.vec fun d : Fin D => d.val) := by
      simp [s2, s1, s0, BlockState.setReg, hoffs_d]
    have hVBaseReg2 : s2.regs .nat [] "v_base_off" =
        some (Tile.scalar (batch * sVB + headIdx * sVH)) := by
      simp [s2, s1, s0, BlockState.setReg, hv_base]
    have hVPtrsEval :
        evalOp
          (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.add .nat Broadcast.scalarL
              (Op.ref .nat [] "v_base_off")
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
                (Op.constNat sVN)))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
              (Op.constNat sVD))) s2 =
          some vPtrs := by
      simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
        Option.bind_eq_bind, Option.bind_some]
      rw [hVBaseReg2]
      simp only [Option.bind_some]
      rw [evalOp_expandDim_one_nat, hOffsNReg2]
      simp only [Option.bind_some]
      rw [evalOp_expandDim_zero_nat, hOffsDReg2]
      simp [vPtrs, vBase, offsN, Tile.bop, NumericDType.add, NumericDType.mul]
    have h1 :
        stepStmt
          (Stmt.assign .nat [Bk] "offs_n"
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil
                (Op.ref .nat [] "n")
                (Op.constNat Bk))
              (Op.arange Bk))) s0 = some s1 := by
      simp [stepStmt, s1, s0, offsN, Tile.bop, NumericDType.add, NumericDType.mul]
      rfl
    have h2 :
        stepStmt
          (Stmt.assign .nat [Bk, D] "k_ptrs"
            (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.add .nat Broadcast.scalarL
                (Op.ref .nat [] "k_base_off")
                (Op.mul .nat Broadcast.scalarR
                  (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
                  (Op.constNat sKN)))
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
                (Op.constNat sKD)))) s1 = some s2 := by
      simp only [stepStmt, hKPtrsEval, Option.bind_some, s2]
      rfl
    have h3 :
        stepStmt
          (Stmt.assign .nat [Bk, D] "v_ptrs"
            (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.add .nat Broadcast.scalarL
                (Op.ref .nat [] "v_base_off")
                (Op.mul .nat Broadcast.scalarR
                  (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
                  (Op.constNat sVN)))
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d"))
                (Op.constNat sVD)))) s2 = some s3 := by
      simp only [stepStmt, hVPtrsEval, Option.bind_some, s3]
      rfl
    have h4 :
        stepStmt
          (Stmt.assign .real [Bk, D] "k"
            (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, D] "k_ptrs"))
              MaskOpt.none)) s3 = some s4 := by
      simp [stepStmt, s4, s3, s2, s1, s0, kTile, kPtrs, kBase, evalOp_load_region_none,
        Region.cast, hKmem]
      rfl
    have h5 :
        stepStmt
          (Stmt.assign .real [Bk, D] "v"
            (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, D] "v_ptrs"))
              MaskOpt.none)) s4 = some s5 := by
      simp [stepStmt, s5, s4, s3, s2, s1, s0, vTile, vPtrs, vBase, evalOp_load_region_none,
        Region.cast, hVmem]
      rfl
    have h6 :
        stepStmt
          (Stmt.assign .real [M, Bk] "scores_raw"
            (Op.mul .real Broadcast.scalarR
              (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
                (Op.ref .real [M, D] "q")
                (Op.transpose (batch := []) (M := Bk) (N := D)
                  (Op.ref .real [Bk, D] "k")))
              (Op.const scale))) s5 = some s6 := by
      simp only [stepStmt, evalOp_mul, Option.bind_eq_bind, Option.bind_some]
      change (((evalOp
            (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
              (Op.ref .real [M, D] "q")
              (Op.transpose (batch := []) (M := Bk) (N := D)
                (Op.ref .real [Bk, D] "k"))) s5).bind fun vx =>
          (evalOp (Op.const scale) s5).bind fun vy =>
            some (Tile.bop NumericDType.real.mul Broadcast.scalarR vx vy)).bind
        fun v => some (s5.setReg "scores_raw" .real [M, Bk] v)) = some s6
      rw [evalOp_dot]
      rw [evalOp_transpose]
      simp [s6, s5, s4, s3, s2, s1, s0, scoresRaw, kTile,
        evalOp_transpose, Tile.bop, Tile.transpose, Tile.dot, NumericDType.mul,
        hq, FA1MathCausal.block_scoresRaw_tile_eq Q K scale k hk]
      ext idx <;> simp [scoresRaw, StreamingAccumulator.scaledScore, Tile.ofReal, mul_comm]
    have h7 :
        stepStmt
          (Stmt.assign .bool [M, Bk] "causal"
            (Op.ge ComparableDType.nat
              (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))) s6 =
          some s7 := by
      simp only [stepStmt, evalOp, Option.bind_eq_bind, Option.bind_some]
      rw [evalOp_expandDim_one_nat]
      rw [evalOp_expandDim_zero_nat]
      simp [s7, s6, s5, s4, s3, s2, s1, s0, causal, Tile.cop,
        ComparableDType.ge, hoffs_m, offsN, Tile.expandDim,
        TileShape.dropInsertedIndex]
    have hScoresSelect :
        Tile.select causal scoresRaw
          (⟨fun _ : TileIndex [M, Bk] => (none : WithBot ℝ)⟩ : Tile .real [M, Bk]) =
        scores := by
      simpa [causal, scoresRaw, scores] using
        FA1MathCausal.block_scores_tile_eq Q K scale k hk qb
    have h8 :
        stepStmt
          (Stmt.assign .real [M, Bk] "scores"
            (Op.where
              (Op.ref .bool [M, Bk] "causal")
              (Op.ref .real [M, Bk] "scores_raw")
              (Op.broadcast Op.negInf [M, Bk]))) s7 = some s8 := by
      simp only [stepStmt, evalOp, evalOp_ref, evalOp_negInf, Option.bind_eq_bind,
        Option.bind_some]
      simp [hScoresSelect,
        s8, s7, s6, s5, s4, s3, s2, s1, s0,
        scores, scoresRaw, causal]
    have hScoresReg : s8.regs .real [M, Bk] "scores" = some scores := by
      simp [s8]
    have hMBlockReduce :
        Tile.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false scores =
        some mBlock := by
      simpa [scores, mBlock] using
        FA1MathCausal.block_mBlock_tile_eq hBk Q K scale k hk qb
    have h9 :
        stepStmt
          (Stmt.assign .real [M] "m_block"
            (Op.reduceMax ⟨1, by simp⟩ «false»
              (Op.ref .real [M, Bk] "scores"))) s8 = some s9 := by
      simp only [stepStmt]
      change ((evalOp (Op.reduceMax ⟨1, by simp⟩ «false»
          (Op.ref .real [M, Bk] "scores")) s8).bind
        fun v => some (s8.setReg "m_block" .real [M] v)) = some s9
      rw [evalOp_reduceMax]
      rw [evalOp_ref]
      rw [hScoresReg]
      change ((Tile.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false scores).bind
        fun v => some (s8.setReg "m_block" .real [M] v)) = some s9
      rw [hMBlockReduce]
      change some (s8.setReg "m_block" .real [M] mBlock) = some s9
      rfl
    have hMBlockReg : s9.regs .real [M] "m_block" = some mBlock := by
      simp [s9]
    have hMNewTile :
        Tile.bop max (Broadcast.consSame Broadcast.nil)
          (⟨fun idx : TileIndex [M] =>
            FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale k idx.1⟩ : Tile .real [M])
          mBlock =
        mNew := by
      simpa [mBlock, mNew] using
        FA1MathCausal.block_mNew_tile_eq Q K scale k hk qb
    have h10 :
        stepStmt
          (Stmt.assign .real [M] "m_new"
            (Op.max2 (Broadcast.consSame Broadcast.nil)
              (Op.ref .real [M] "m_i")
              (Op.ref .real [M] "m_block"))) s9 = some s10 := by
      simp only [stepStmt, evalOp_max2, Option.bind_eq_bind, Option.bind_some]
      rw [evalOp_ref]
      rw [show s9.regs .real [M] "m_i" =
          some (⟨fun idx : TileIndex [M] =>
            FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale k idx.1⟩ : Tile .real [M]) by
        simp [s9, s8, s7, s6, s5, s4, s3, s2, s1, s0, hm]]
      rw [evalOp_ref]
      rw [hMBlockReg]
      simp only [Option.bind_some]
      rw [hMNewTile]
    have hAlphaDataStep : ∀ i : Fin M,
        WithBot.realExp
          (Option.map₂ (fun x1 x2 => x1 - x2)
            (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale k i)
            (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) i))
          =
          some (FA1MathCausal.alphaCausal (qb * M) Q numKVBlocks K scale k i) := by
      intro i
      simpa [FA1MathCausal.alphaCausal] using
        FA1MathCausal.realExp_eq_some_unbotD
          (Option.map₂ (fun x1 x2 : ℝ => x1 - x2)
            (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale k i)
            (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) i))
    have h11 :
        stepStmt
          (Stmt.assign .real [M] "alpha"
            (Op.exp
              (Op.sub .real (Broadcast.consSame Broadcast.nil)
                (Op.ref .real [M] "m_i")
                (Op.ref .real [M] "m_new")))) s10 = some s11 := by
      simp [stepStmt, evalOp_exp, evalOp_sub, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        alpha, mNew, Tile.bop, Tile.uop, Tile.ofReal, NumericDType.sub, hm,
        FA1MathCausal.block_mNew_tile_eq Q K scale k hk qb, hAlphaDataStep]
    have h12 :
        stepStmt
          (Stmt.assign .real [M, Bk] "p"
            (Op.exp
              (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
                (Op.ref .real [M, Bk] "scores")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "m_new"))))) s11 =
          some s12 := by
      simp only [stepStmt, evalOp_exp, evalOp_sub, Option.bind_eq_bind, Option.bind_some]
      unfold evalOp
      simp [VeriTile.evalOp_expandDim, VeriTile.evalOp_expandDim_ref,
        VeriTile.evalOp_ref,
        evalOp_expandDim_one_real,
        s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        p, scores, mNew, Tile.bop, Tile.uop, Tile.expandDim, NumericDType.sub,
        FA1MathCausal.block_mNew_tile_eq Q K scale k hk qb]
      change some (s11.setReg "p" .real [M, Bk] p) = some s12
      rfl
    have h13 :
        stepStmt
          (Stmt.assign .real [M] "l_new"
            (Op.add .real (Broadcast.consSame Broadcast.nil)
              (Op.mul .real (Broadcast.consSame Broadcast.nil)
                (Op.ref .real [M] "alpha")
                (Op.ref .real [M] "l_i"))
                (Op.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
                (Op.ref .real [M, Bk] "p")))) s12 = some s13 := by
      simp [stepStmt, evalOp_add, evalOp_mul, evalOp_reduceSum,
        s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        lNew, alpha, p, Tile.bop, Tile.ofReal, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        hBk, NumericDType.add, NumericDType.mul, hl,
        FA1MathCausal.block_lNew_tile_eq (qb * M) Q K scale k hk]
      unfold evalOp
      simp [s13, lNew, alpha, p, Tile.bop, Tile.ofReal, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        hBk, NumericDType.add, NumericDType.mul, WithBot.realAdd, WithBot.realMul,
        FA1MathCausal.lPartial_succ_of_lt (qb * M) Q numKVBlocks K scale k hk]
      have hLTile :
          ({ data := fun i : TileIndex [M] =>
            Option.map
              (fun b =>
                FA1MathCausal.alphaCausal (qb * M) Q numKVBlocks K scale k i.1 *
                    FA1MathCausal.lPartial Bk (qb * M) Q numKVBlocks K scale k i.1 +
                  b)
              (@Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ (fun x =>
                WithBot.realExp
                  (Option.map₂ (fun x y : ℝ => x - y)
                    (FA1MathCausal.maskedScore (qb * M) Q K scale i.1
                      (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) x))
                    (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) i.1)))) } :
              Tile .real [M]) =
            { data := fun i : TileIndex [M] =>
              some
                (WithBot.unbotD 0
                    (WithBot.realExp
                      (Option.map₂ (fun x y : ℝ => x - y)
                        (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale k i.1)
                        (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) i.1))) *
                  FA1MathCausal.lPartial Bk (qb * M) Q numKVBlocks K scale k i.1 +
                ∑ jLocal : Fin Bk,
                  WithBot.unbotD 0
                    (WithBot.realExp
                      (Option.map₂ (fun x y : ℝ => x - y)
                        (FA1MathCausal.maskedScore (qb * M) Q K scale i.1
                          (StreamingAccumulator.blockIndex Bk numKVBlocks k
                            (Nat.succ_le_iff.mpr hk) jLocal))
                        (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) i.1)))) } := by
        ext i
        have hSum :
            (@Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ (fun x =>
              WithBot.realExp
                (Option.map₂ (fun x y : ℝ => x - y)
                  (FA1MathCausal.maskedScore (qb * M) Q K scale i.1
                    (StreamingAccumulator.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) x))
                  (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) i.1)))) =
              (some
                (∑ x : Fin Bk,
                  WithBot.unbotD 0
                    (WithBot.realExp
                      (Option.map₂ (fun x y : ℝ => x - y)
                        (FA1MathCausal.maskedScore (qb * M) Q K scale i.1
                          (StreamingAccumulator.blockIndex Bk numKVBlocks k
                            (Nat.succ_le_iff.mpr hk) x))
                        (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) i.1)))) :
                  WithBot ℝ) := by
          calc
            (∑ x : Fin Bk,
              WithBot.realExp
                (Option.map₂ (fun x y : ℝ => x - y)
                  (FA1MathCausal.maskedScore (qb * M) Q K scale i.1
                    (StreamingAccumulator.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) x))
                  (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) i.1))) =
                @Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ (fun x =>
                  (some
                    (WithBot.unbotD 0
                      (WithBot.realExp
                        (Option.map₂ (fun x y : ℝ => x - y)
                          (FA1MathCausal.maskedScore (qb * M) Q K scale i.1
                            (StreamingAccumulator.blockIndex Bk numKVBlocks k
                              (Nat.succ_le_iff.mpr hk) x))
                          (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) i.1)))) :
                  WithBot ℝ)) := by
                  apply Finset.sum_congr rfl
                  intro x _
                  exact FA1MathCausal.realExp_eq_some_unbotD _
            _ = (some
                (∑ x : Fin Bk,
                  WithBot.unbotD 0
                    (WithBot.realExp
                      (Option.map₂ (fun x y : ℝ => x - y)
                        (FA1MathCausal.maskedScore (qb * M) Q K scale i.1
                          (StreamingAccumulator.blockIndex Bk numKVBlocks k
                            (Nat.succ_le_iff.mpr hk) x))
                        (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) i.1)))) :
                  WithBot ℝ) := by
                simpa using
                  (WithBot.sum_someTerm_eq_some
                    (Finset.univ : Finset (Fin Bk))
                    (fun x =>
                      WithBot.unbotD 0
                        (WithBot.realExp
                          (Option.map₂ (fun x y : ℝ => x - y)
                            (FA1MathCausal.maskedScore (qb * M) Q K scale i.1
                              (StreamingAccumulator.blockIndex Bk numKVBlocks k
                                (Nat.succ_le_iff.mpr hk) x))
                            (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) i.1)))))
        simp only [hSum, Option.map_some, FA1MathCausal.alphaCausal]
      exact (congrArg (fun t : Tile .real [M] =>
        (s11.setReg "p" .real [M, Bk] p).setReg "l_new" .real [M] t) hLTile)
    have h14 :
        stepStmt
          (Stmt.assign .real [M, D] "o_acc"
            (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "alpha"))
                (Op.ref .real [M, D] "o_acc"))
              (Op.dot (batch := []) (M := M) (K := Bk) (N := D)
                (Op.ref .real [M, Bk] "p")
                (Op.ref .real [Bk, D] "v")))) s13 = some s14 := by
      simp [stepStmt, evalOp_add, evalOp_mul, evalOp_dot,
        s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        oNew, alpha, p, vTile, Tile.bop, Tile.ofReal, Tile.dot,
        Tile.expandDim, TileShape.dropInsertedIndex, NumericDType.add,
        NumericDType.mul, ho,
        FA1MathCausal.block_oAcc_tile_eq (qb * M) Q K V scale k hk]
      unfold evalOp
      simp [Tile.dot, Tile.expandDim, TileShape.dropInsertedIndex,
        FA1MathCausal.oPartial_succ_of_lt (qb * M) Q numKVBlocks K V scale k hk]
      simp only [Option.bind, Option.map]
      congr
      ext idx
      rcases idx with ⟨i, d, u⟩
      cases u
      let dotSum : ℝ :=
        ∑ x : Fin Bk,
          WithBot.unbotD (0 : ℝ)
              (WithBot.realExp
                (Option.map₂ (fun x y : ℝ => x - y)
                  (FA1MathCausal.maskedScore (qb * M) Q K scale i
                    (StreamingAccumulator.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) x))
                    (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) i))) *
              V (StreamingAccumulator.blockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) x, d, PUnit.unit)
      let pVSum : WithBot ℝ :=
        @Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ (fun x =>
            (match
              WithBot.realExp
                (Option.map₂ (fun x y : ℝ => x - y)
                  (FA1MathCausal.maskedScore (qb * M) Q K scale i
                    (StreamingAccumulator.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) x))
                  (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) i)) with
              | some x_1 =>
                  some (x_1 * V (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) x, d, PUnit.unit))
              | none => none : WithBot ℝ))
      have hSum : pVSum = (some dotSum : WithBot ℝ) := by
        dsimp [pVSum]
        calc
          (∑ x : Fin Bk,
            (match
              WithBot.realExp
                (Option.map₂ (fun x y : ℝ => x - y)
                  (FA1MathCausal.maskedScore (qb * M) Q K scale i
                    (StreamingAccumulator.blockIndex Bk numKVBlocks k
                      (Nat.succ_le_iff.mpr hk) x))
                  (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) i)) with
              | some x_1 =>
                  some (x_1 * V (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) x, d, PUnit.unit))
              | none => none : WithBot ℝ)) =
              @Finset.sum (Fin Bk) (WithBot ℝ) _ Finset.univ (fun x =>
                (some
                  (WithBot.unbotD 0
                      (WithBot.realExp
                        (Option.map₂ (fun x y : ℝ => x - y)
                          (FA1MathCausal.maskedScore (qb * M) Q K scale i
                            (StreamingAccumulator.blockIndex Bk numKVBlocks k
                              (Nat.succ_le_iff.mpr hk) x))
                          (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) i))) *
                  V (StreamingAccumulator.blockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) x, d, PUnit.unit)) :
                WithBot ℝ)) := by
                apply Finset.sum_congr rfl
                intro x _
                conv_lhs => rw [FA1MathCausal.realExp_eq_some_unbotD]
          _ = (some dotSum : WithBot ℝ) := by
              dsimp [dotSum]
              simpa using
                (WithBot.sum_someTerm_eq_some
                  (Finset.univ : Finset (Fin Bk))
                  (fun x =>
                    WithBot.unbotD 0
                        (WithBot.realExp
                          (Option.map₂ (fun x y : ℝ => x - y)
                            (FA1MathCausal.maskedScore (qb * M) Q K scale i
                              (StreamingAccumulator.blockIndex Bk numKVBlocks k
                                (Nat.succ_le_iff.mpr hk) x))
                            (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) i))) *
                      V (StreamingAccumulator.blockIndex Bk numKVBlocks k
                        (Nat.succ_le_iff.mpr hk) x, d, PUnit.unit)))
      simp only [FA1MathCausal.alphaCausal]
      rename_i out
      convert (show (Option.map₂ (fun x1 x2 : ℝ => x1 + x2)
          (some
            (WithBot.unbotD 0
                (WithBot.realExp
                  (Option.map₂ (fun x y : ℝ => x - y)
                    (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale k i)
                    (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) i))) *
              FA1MathCausal.oPartial Bk (qb * M) Q numKVBlocks K V scale k
                (i, d, PUnit.unit)) : WithBot ℝ)
          pVSum = some out ↔
          (some
            (WithBot.unbotD 0
                  (WithBot.realExp
                  (Option.map₂ (fun x y : ℝ => x - y)
                    (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale k i)
                    (FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale (k + 1) i))) *
                FA1MathCausal.oPartial Bk (qb * M) Q numKVBlocks K V scale k
                  (i, d, PUnit.unit) + dotSum) : WithBot ℝ) = some out) from by
          rw [hSum]
          simp) <;> (
            dsimp [pVSum]
            first
            | rfl
            | congr
              ext x <;> simp [StreamingAccumulator.blockIndex]
              aesop)
    have h15 :
        stepStmt
          (Stmt.assign .real [M] "m_i" (Op.ref .real [M] "m_new")) s14 =
          some s15 := by
      simp [stepStmt, s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
    have h16 :
        stepStmt
          (Stmt.assign .real [M] "l_i" (Op.ref .real [M] "l_new")) s15 =
          some s' := by
      simp [stepStmt, s', s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0]
    change stepStmts (fa1LoopBodyStridedCausal kReg vReg M D Bk sKN sKD sVN sVD scale) s0 = some s'
    unfold fa1LoopBodyStridedCausal
    rw [stepStmts.cons_some h1]
    rw [stepStmts.cons_some h2]
    rw [stepStmts.cons_some h3]
    rw [stepStmts.cons_some h4]
    rw [stepStmts.cons_some h5]
    rw [stepStmts.cons_some h6]
    rw [stepStmts.cons_some h7]
    rw [stepStmts.cons_some h8]
    rw [stepStmts.cons_some h9]
    rw [stepStmts.cons_some h10]
    rw [stepStmts.cons_some h11]
    rw [stepStmts.cons_some h12]
    rw [stepStmts.cons_some h13]
    rw [stepStmts.cons_some h14]
    rw [stepStmts.cons_some h15]
    rw [stepStmts.cons_some h16]
    simp [stepStmts]
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simpa using hpids0
    · simpa using hpids1
    · simpa using hpids2
    · simpa using hpid_qb
    · simpa using hpid_h
    · simpa using hpid_b
    · simpa using hq_base
    · simpa using hk_base
    · simpa using hv_base
    · simpa using ho_base
    · simpa using hoffs_m
    · simpa using hoffs_d
    · simpa using hq
    · simpa [s', s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        mNew, BlockState.setReg]
    · simpa [s', s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        lNew, BlockState.setReg]
    · simpa [s', s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0,
        oNew, BlockState.setReg]
    · intro idx
      simpa [s', s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hQ idx
    · intro idx
      simpa [s', s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hK idx
    · intro idx
      simpa [s', s15, s14, s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, s0] using hV idx


end VeriTile.Examples
