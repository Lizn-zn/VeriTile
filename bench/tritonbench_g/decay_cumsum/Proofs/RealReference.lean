import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Internal real reference and its existing complete streaming IO proofs.
This is a proof adapter, not the faithful typed source; in particular its
prepare/backward bodies cache arange in an extra offs register. The new source
must be connected by a checked execution bridge before public use. -/

namespace VeriTile.Bench.TritonBenchG.DecayCumsum.RealReference

open VeriTile.Triton

set_option linter.unusedSimpArgs false

section Correct_without_Rounding

def fwd_decay_cumsum_surface
    (G GO : RegionName)
    (s_qk_h _s_qk_t _s_qk_d _B _H _T : Nat) (_scale : ℝ)
    (BT BK DK : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  p_g = G + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) +
    i_k * $(BK) + tl.arange(0, $(BK))
  p_go = GO + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) +
    i_k * $(BK) + tl.arange(0, $(BK))
  cum_decay = tl.zeros([$(BK)], dtype=tl.float32)
  mask = (i_k * $(BK) + tl.arange(0, $(BK))) < $(DK)
  for i in range($(0), $(BT), $(1)) {
    _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
    cum_decay += _g * 1.44269504
    tl.store(p_go, (cum_decay).to(p_go.dtype.element_ty), mask=mask)
    p_g += $(DK)
    p_go += $(DK)
  }
}

def prepare_qg_kg_surface
    (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat)
    (scale : ℝ) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs = tl.arange(0, $(BK))
  p_q = Q + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + offs
  p_g = G + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + offs
  p_k = K + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + offs
  p_qg = QG + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + offs
  p_kg = KG + i_bh * $(s_qk_h) + i_c * $(BT) * $(DK) + i_k * $(BK) + offs
  mask = (i_k * $(BK) + offs) < $(DK)
  last_decay = tl.load(G + i_bh * $(s_qk_h) +
    (i_c * $(BT) + $(BT) - $(1)) * $(DK) + i_k * $(BK) + offs)
  for i in range($(0), $(BT), $(1)) {
    _q = tl.load(p_q, mask=mask, other=0)
    _k = tl.load(p_k, mask=mask, other=0)
    _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
    _q *= tl.math.exp2(_g) * $((scale : ℝ))
    _k *= tl.math.exp2(last_decay - _g)
    tl.store(p_kg, (_k).to(p_kg.dtype.element_ty), mask=mask)
    tl.store(p_qg, (_q).to(p_qg.dtype.element_ty), mask=mask)
    p_q += $(DK)
    p_g += $(DK)
    p_k += $(DK)
    p_kg += $(DK)
    p_qg += $(DK)
  }
}

def bwd_decay_global_cumsum_surface
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(0)
  i_c = tl.program_id(1)
  i_bh = tl.program_id(2)
  offs = tl.arange(0, $(BK))
  p_q = Q + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_k = K + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_g = G + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dg = DG + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dq_inner = DQInner + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dk_inner = DKInner + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dq_inter = DQInter + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  p_dk_inter = DKInter + i_bh * $(s_qk_h) + i_k * $(BK) + offs + (i_c * $(BT) + $(BT) - $(1)) * $(DK)
  cum_grad_dg = tl.zeros([$(BK)], dtype=tl.float32)
  mask = (i_k * $(BK) + offs) < $(DK)
  last_g = tl.zeros([$(BK)], dtype=tl.float32)
  for j in range($(BT) - $(1), -$(1), -$(1)) {
    _g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
    if j == $(BT) - $(1) {
      last_g = _g
    }
    _dq1 = tl.load(p_dq_inner, mask=mask, other=0)
    _dq2 = tl.load(p_dq_inter, mask=mask, other=0)
    _dq2 *= tl.math.exp2(_g)
    _dq = _dq1 + _dq2
    tl.store(p_dq_inter, _dq, mask=mask)
    _dk1 = tl.load(p_dk_inner, mask=mask, other=0)
    _dk2 = tl.load(p_dk_inter, mask=mask, other=0)
    _dk2 *= tl.math.exp2(last_g - _g)
    _dk = _dk1 + _dk2
    tl.store(p_dk_inter, _dk, mask=mask)
    _q = tl.load(p_q, mask=mask, other=0)
    _k = tl.load(p_k, mask=mask, other=0)
    _dg = _dq * _q - _dk * _k
    cum_grad_dg += _dg
    tl.store(p_dg, (cum_grad_dg).to(p_dg.dtype.element_ty), mask=mask)
    p_g -= $(DK)
    p_k -= $(DK)
    p_q -= $(DK)
    p_dq_inner -= $(DK)
    p_dk_inner -= $(DK)
    p_dq_inter -= $(DK)
    p_dk_inter -= $(DK)
    p_dg -= $(DK)
  }
}

def elemIndex (s : BlockState) (BK : Nat) (i : Fin BK) : Nat :=
  s.pids 0 * BK + i.val

def baseOffset (s : BlockState) (s_qk_h DK t_rel BT BK : Nat) : Nat :=
  s.pids 2 * s_qk_h + (s.pids 1 * BT + t_rel) * DK + s.pids 0 * BK

def offset
    (s : BlockState) (s_qk_h DK t_rel BT BK : Nat) (i : Fin BK) : Nat :=
  baseOffset s s_qk_h DK t_rel BT BK + i.val

def active (s : BlockState) (DK BK : Nat) (i : Fin BK) : Prop :=
  elemIndex s BK i < DK

instance activeDecidable (s : BlockState) (DK BK : Nat) (i : Fin BK) :
    Decidable (active s DK BK i) := by
  unfold active
  infer_instance

noncomputable def fwdDecayClosed
    (s : BlockState) (G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) : ℝ :=
  1.44269504 *
    ∑ k : Fin (t_rel.val + 1),
      s.readMem G (offset s s_qk_h DK k.val BT BK i)

noncomputable def fwdPartial
    (s : BlockState) (G : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) (i : Fin BK) : ℝ :=
  1.44269504 * ∑ k : Fin m, s.readMem G (offset s s_qk_h DK k.val BT BK i)

noncomputable def fwdCumTile
    (s : BlockState) (G : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) : Tile .real [BK] :=
  ⟨fun idx => if active s DK BK idx.1 then
      some (fwdPartial s G s_qk_h DK BT BK m idx.1) else some 0⟩

def fwdPgTile
    (s : BlockState) (G : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) : Tile .ptr [BK] :=
  ⟨fun idx => (G, offset s s_qk_h DK m BT BK idx.1)⟩

def fwdPgoTile
    (s : BlockState) (GO : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) : Tile .ptr [BK] :=
  ⟨fun idx => (GO, offset s s_qk_h DK m BT BK idx.1)⟩

def fwdMaskTile (s : BlockState) (DK BK : Nat) : Tile .bool [BK] :=
  ⟨fun idx => decide (active s DK BK idx.1)⟩

noncomputable def fwdInv
    (G GO : RegionName) (s : BlockState) (s_qk_h DK BT BK : Nat) :
    Nat → BlockState → Prop :=
  fun m sc =>
    sc.pids = s.pids ∧
    (∀ a, sc.readMem G a = s.readMem G a) ∧
    sc.regs .real [BK] "cum_decay" = some (fwdCumTile s G s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_g" = some (fwdPgTile s G s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_go" = some (fwdPgoTile s GO s_qk_h DK BT BK m) ∧
    sc.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) ∧
    (∀ j, m ≤ j → ∀ i : Fin BK,
      sc.readMem GO (offset s s_qk_h DK j BT BK i) =
        s.readMem GO (offset s s_qk_h DK j BT BK i)) ∧
    (∀ j, j < m → ∀ i : Fin BK,
      sc.readMem GO (offset s s_qk_h DK j BT BK i) =
        if active s DK BK i then
          fwdPartial s G s_qk_h DK BT BK (j + 1) i
        else s.readMem GO (offset s s_qk_h DK j BT BK i))

theorem fwdPartial_succ
    (s : BlockState) (G : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) (i : Fin BK) :
    fwdPartial s G s_qk_h DK BT BK (m + 1) i =
      fwdPartial s G s_qk_h DK BT BK m i +
        1.44269504 * s.readMem G (offset s s_qk_h DK m BT BK i) := by
  simp only [fwdPartial, Fin.sum_univ_castSucc, Fin.val_last, Fin.val_castSucc]
  ring

noncomputable def fwdGTile
    (s : BlockState) (G : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) : Tile .real [BK] :=
  ⟨fun idx => if active s DK BK idx.1 then
      some (s.readMem G (offset s s_qk_h DK m BT BK idx.1)) else some 0⟩

def fwdBody (G GO : RegionName) (s_qk_h BT BK DK : Nat) : List Stmt :=
  [Stmt.assign TileDType.real [BK] "_g"
      (Op.load ComputeDType.fp32.eraseDType (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_g"))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign TileDType.real [BK] "cum_decay"
      (Op.add NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "cum_decay")
        (Op.mul NumericDType.real Broadcast.scalarR (Op.ref TileDType.real [BK] "_g") (Op.const 1.44269504))),
    Stmt.store TileDType.real [BK] (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_go"))
      (Op.ref TileDType.real [BK] "cum_decay") (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask")),
    Stmt.assign TileDType.ptr [BK] "p_g"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_g") (Op.constNat DK)),
    Stmt.assign TileDType.ptr [BK] "p_go"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_go") (Op.constNat DK))]

set_option maxHeartbeats 4000000 in
                                                  
theorem fwd_decay_cumsum_step
    (G GO : RegionName) (s : BlockState) (s_qk_h BT BK DK : Nat)
    (hne : G ≠ GO) (hBK : BK ≤ DK)
    (m : Nat) (sc : BlockState) (hP : fwdInv G GO s s_qk_h DK BT BK m sc) :
    ∃ s',
      stepStmts (fwdBody G GO s_qk_h BT BK DK)
        (sc.setReg "i" .nat [] (Tile.scalar m)) = some s' ∧
      fwdInv G GO s s_qk_h DK BT BK (m + 1) s' := by
  obtain ⟨hpids, hG, hcum, hpg, hpgo, hmask, hUntouched, hreadGO⟩ := hP
  set sloop := sc.setReg "i" .nat [] (Tile.scalar m) with hsloop
  
  have hcum' : sloop.regs .real [BK] "cum_decay" = some (fwdCumTile s G s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hcum
  have hpg' : sloop.regs .ptr [BK] "p_g" = some (fwdPgTile s G s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpg
  have hpgo' : sloop.regs .ptr [BK] "p_go" = some (fwdPgoTile s GO s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpgo
  have hmask' : sloop.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
    rw [hsloop]; simpa using hmask
  
  have hgeval : evalOp (Op.load TileDType.real (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_g"))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BK] "mask") ((Op.const 0).broadcast [BK]))) sloop
      = some (fwdGTile s G s_qk_h DK BT BK m) := by
    simp only [evalOp, Option.bind, hpg', hmask', evalOp_const]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [fwdGTile, fwdMaskTile, fwdPgTile]
    by_cases ha : active s DK BK i
    · simp only [ha, decide_true, if_true, if_pos]
      rw [hsloop, BlockState.setReg_readMemValue, BlockState.readMemValue_real, hG]
    · simp only [ha, decide_false, Bool.false_eq_true, if_false]
      rfl
  
  set s1 := sloop.setReg "_g" .real [BK] (fwdGTile s G s_qk_h DK BT BK m) with hs1
  have hstep1 : stepStmt (Stmt.assign TileDType.real [BK] "_g"
      (Op.load ComputeDType.fp32.eraseDType (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_g"))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BK] "mask") ((Op.const 0).broadcast [BK])))) sloop
      = some s1 := by
    rw [hs1]; exact stepStmt_assign_eq_some (by simpa [ComputeDType.eraseDType] using hgeval)
  
  have hcum1 : s1.regs .real [BK] "cum_decay" = some (fwdCumTile s G s_qk_h DK BT BK m) := by
    rw [hs1]; simpa using hcum'
  have hg1 : s1.regs .real [BK] "_g" = some (fwdGTile s G s_qk_h DK BT BK m) := by
    rw [hs1]; simp
  have hcumeval : evalOp (Op.add NumericDType.real Broadcast.nil.consSame
      (Op.ref TileDType.real [BK] "cum_decay")
      (Op.mul NumericDType.real Broadcast.scalarR (Op.ref TileDType.real [BK] "_g")
        (Op.const 1.44269504))) s1
      = some (fwdCumTile s G s_qk_h DK BT BK (m + 1)) := by
    simp only [evalOp, Option.bind, hcum1, hg1, evalOp_const]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [fwdCumTile, fwdGTile, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.add, NumericDType.mul]
    by_cases ha : active s DK BK i
    · simp only [ha, if_true, WithBot.realAdd, WithBot.realMul]
      rw [fwdPartial_succ]
      norm_num
      ring
    · simp only [ha, if_false, WithBot.realAdd, WithBot.realMul]
      norm_num
  set s2 := s1.setReg "cum_decay" .real [BK] (fwdCumTile s G s_qk_h DK BT BK (m + 1)) with hs2
  have hstep2 : stepStmt (Stmt.assign TileDType.real [BK] "cum_decay"
      (Op.add NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "cum_decay")
        (Op.mul NumericDType.real Broadcast.scalarR (Op.ref TileDType.real [BK] "_g")
          (Op.const 1.44269504)))) s1 = some s2 := by
    rw [hs2]; exact stepStmt_assign_eq_some hcumeval
  
  have hpgo2 : s2.regs .ptr [BK] "p_go" = some (fwdPgoTile s GO s_qk_h DK BT BK m) := by
    rw [hs2, hs1]; simp [hpgo']
  have hmask2 : s2.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
    rw [hs2, hs1]; simp [hmask']
  have hcum2 : s2.regs .real [BK] "cum_decay" = some (fwdCumTile s G s_qk_h DK BT BK (m + 1)) := by
    rw [hs2]; simp
  
  set vfun : TileIndex [BK] → ℝ := fun k =>
    (fwdCumTile s G s_qk_h DK BT BK (m + 1)).data k |>.unbotD 0 with hvfun
  
  set sst := (TileShape.allIndices [BK]).foldl
    (fun (acc : BlockState) k =>
      if active s DK BK k.1 then
        acc.writeMem GO (offset s s_qk_h DK m BT BK k.1) (vfun k)
      else acc) s2 with hsst
  have hstore : stepStmt (Stmt.store TileDType.real [BK] (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_go"))
      (Op.ref TileDType.real [BK] "cum_decay") (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask"))) s2
      = some sst := by
    simp only [stepStmt, evalOp, Option.bind, Option.map, hcum2, hpgo2, hmask2]
    apply congrArg some
    rw [hsst]
    congr 1
    funext acc k
    simp only [fwdPgoTile, fwdMaskTile]
    by_cases ha : active s DK BK k.1
    · rw [if_pos (by simpa using ha), if_pos ha, BlockState.writeMemTyped_real]
      rfl
    · rw [if_neg (by simpa using ha), if_neg ha]
  
  have hpg2 : sst.regs .ptr [BK] "p_g" = some (fwdPgTile s G s_qk_h DK BT BK m) := by
    rw [hsst, BlockState.foldl_writeMem_prop_masked_regs, hs2, hs1]; simp [hpg']
  have hpgadd : evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_g")
      (Op.constNat DK)) sst = some (fwdPgTile s G s_qk_h DK BT BK (m + 1)) := by
    simp only [evalOp, Option.bind, hpg2, evalOp_constNat]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [Tile.ptrAdd_data, Broadcast.leftIndex, Broadcast.rightIndex, fwdPgTile, Tile.scalar]
    refine Prod.ext rfl ?_
    simp only [offset, baseOffset]
    ring
  set s3 := sst.setReg "p_g" .ptr [BK] (fwdPgTile s G s_qk_h DK BT BK (m + 1)) with hs3
  have hstep4 : stepStmt (Stmt.assign TileDType.ptr [BK] "p_g"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_g") (Op.constNat DK))) sst
      = some s3 := by
    rw [hs3]; exact stepStmt_assign_eq_some hpgadd
  
  have hpgo3 : s3.regs .ptr [BK] "p_go" = some (fwdPgoTile s GO s_qk_h DK BT BK m) := by
    rw [hs3, BlockState.setReg_ne_name (h := by decide)]
    rw [hsst, BlockState.foldl_writeMem_prop_masked_regs, hs2, hs1]; simp [hpgo']
  have hpgoadd : evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_go")
      (Op.constNat DK)) s3 = some (fwdPgoTile s GO s_qk_h DK BT BK (m + 1)) := by
    simp only [evalOp, Option.bind, hpgo3, evalOp_constNat]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [Tile.ptrAdd_data, Broadcast.leftIndex, Broadcast.rightIndex, fwdPgoTile, Tile.scalar]
    refine Prod.ext rfl ?_
    simp only [offset, baseOffset]
    ring
  set s4 := s3.setReg "p_go" .ptr [BK] (fwdPgoTile s GO s_qk_h DK BT BK (m + 1)) with hs4
  have hstep5 : stepStmt (Stmt.assign TileDType.ptr [BK] "p_go"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_go") (Op.constNat DK))) s3
      = some s4 := by
    rw [hs4]; exact stepStmt_assign_eq_some hpgoadd
  refine ⟨s4, ?_, ?_⟩
  · simp only [fwdBody]
    rw [stepStmts.cons_some hstep1, stepStmts.cons_some hstep2,
        stepStmts.cons_some hstore, stepStmts.cons_some hstep4,
        stepStmts.cons_some hstep5, stepStmts.nil]
  · 
    
    have hInj : Function.Injective (fun k : TileIndex [BK] =>
        offset s s_qk_h DK m BT BK k.1) := by
      rintro ⟨a, _⟩ ⟨b, _⟩ hab
      simp only [offset] at hab
      obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
      rfl
    
    have hmemG4 : ∀ a, s4.readMem G a = sst.readMem G a := by
      intro a; rw [hs4, hs3]; simp only [BlockState.setReg_readMem]
    have hmemGO4 : ∀ a, s4.readMem GO a = sst.readMem GO a := by
      intro a; rw [hs4, hs3]; simp only [BlockState.setReg_readMem]
    
    have hsstG : ∀ a, sst.readMem G a = s.readMem G a := by
      intro a
      rw [hsst, BlockState.scatter_prop_masked_preserves_other_region
        (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := hne)]
      rw [hs2, hs1]; simp only [BlockState.setReg_readMem]; exact hG a
    
    have hsep : ∀ (j : Nat), j ≠ m → ∀ (i : Fin BK),
        (∀ k : TileIndex [BK], active s DK BK k.1 →
          (fun k : TileIndex [BK] => offset s s_qk_h DK m BT BK k.1) k
            ≠ offset s s_qk_h DK j BT BK i) := by
      intro j hj i k _
      simp only [offset, baseOffset]
      have hik : (i : ℕ) < BK := i.isLt
      have hiDK : (i : ℕ) < DK := lt_of_lt_of_le hik hBK
      rcases Nat.lt_or_ge j m with hlt | hge'
      · have hge : (s.pids 1 * BT + j) * DK + DK ≤ (s.pids 1 * BT + m) * DK := by
          have : (s.pids 1 * BT + j) + 1 ≤ s.pids 1 * BT + m := by omega
          calc (s.pids 1 * BT + j) * DK + DK
              = ((s.pids 1 * BT + j) + 1) * DK := by ring
            _ ≤ (s.pids 1 * BT + m) * DK := Nat.mul_le_mul_right DK this
        omega
      · have hjm' : m < j := by omega
        have hkBK : (k.1 : ℕ) < BK := k.1.isLt
        have hkDK : (k.1 : ℕ) < DK := lt_of_lt_of_le hkBK hBK
        have hge : (s.pids 1 * BT + m) * DK + DK ≤ (s.pids 1 * BT + j) * DK := by
          have : (s.pids 1 * BT + m) + 1 ≤ s.pids 1 * BT + j := by omega
          calc (s.pids 1 * BT + m) * DK + DK
              = ((s.pids 1 * BT + m) + 1) * DK := by ring
            _ ≤ (s.pids 1 * BT + j) * DK := Nat.mul_le_mul_right DK this
        omega
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · 
      rw [hs4, hs3]; simp only [BlockState.setReg_pids]
      rw [hsst, BlockState.foldl_writeMem_prop_masked_pids]
      rw [hs2, hs1]; simp only [BlockState.setReg_pids]; exact hpids
    · 
      intro a; rw [hmemG4]; exact hsstG a
    · 
      rw [hs4, hs3, BlockState.setReg_ne_name (h := by decide),
        BlockState.setReg_ne_name (h := by decide), hsst,
        BlockState.foldl_writeMem_prop_masked_regs, hs2]; simp
    · 
      rw [hs4, BlockState.setReg_ne_name (h := by decide), hs3]; simp
    · 
      rw [hs4]; simp
    · 
      rw [hs4, hs3, BlockState.setReg_ne_name (h := by decide),
        BlockState.setReg_ne_name (h := by decide), hsst,
        BlockState.foldl_writeMem_prop_masked_regs, hs2, hs1]; simp [hmask']
    · 
      intro j hj i
      rw [hmemGO4 (offset s s_qk_h DK j BT BK i), hsst]
      rw [BlockState.scatter_prop_masked_preserves_other_offset
        (region := GO)
        (offsetFn := fun k : TileIndex [BK] => offset s s_qk_h DK m BT BK k.1)
        (valueFn := vfun)
        (P := fun k : TileIndex [BK] => active s DK BK k.1)
        (off := offset s s_qk_h DK j BT BK i)
        (h_ne := hsep j (by omega) i)]
      rw [hs2, hs1, hsloop]; simp only [BlockState.setReg_readMem]
      exact hUntouched j (by omega) i
    · 
      intro j hj i
      rw [hmemGO4 (offset s s_qk_h DK j BT BK i), hsst]
      by_cases hjm : j = m
      · subst hjm
        rw [BlockState.scatter_readback_prop_masked_nd s2
          (fun k : TileIndex [BK] => offset s s_qk_h DK j BT BK k.1) vfun
          (fun k => active s DK BK k.1) hInj (i, PUnit.unit)]
        by_cases ha : active s DK BK i
        · rw [if_pos ha, if_pos ha]
          simp only [hvfun, fwdCumTile]
          rw [if_pos ha]; rfl
        · rw [if_neg ha, if_neg ha]
          rw [hs2, hs1, hsloop]; simp only [BlockState.setReg_readMem]
          exact hUntouched j (le_refl j) i
      · 
        have hjlt : j < m := by omega
        rw [BlockState.scatter_prop_masked_preserves_other_offset
          (region := GO)
          (offsetFn := fun k : TileIndex [BK] => offset s s_qk_h DK m BT BK k.1)
          (valueFn := vfun)
          (P := fun k : TileIndex [BK] => active s DK BK k.1)
          (off := offset s s_qk_h DK j BT BK i)
          (h_ne := hsep j (by omega) i)]
        rw [hs2, hs1, hsloop]; simp only [BlockState.setReg_readMem]
        rw [hreadGO j hjlt i]

set_option maxHeartbeats 4000000 in

theorem fwd_decay_cumsum_full_surface_closed_general
    (G GO : RegionName)
    (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ)
    (BT BK DK : Nat) (s : BlockState) (t_rel : Fin BT) (i : Fin BK)
    (hne : G ≠ GO) (hBK : BK ≤ DK) :
    (exec (fwd_decay_cumsum_surface G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK) s).map
      (·.readMem GO (offset s s_qk_h DK t_rel.val BT BK i))
      = some (if active s DK BK i then fwdDecayClosed s G s_qk_h DK BT BK t_rel i
              else s.readMem GO (offset s s_qk_h DK t_rel.val BT BK i)) := by
  
  simp only [exec, fwd_decay_cumsum_surface, ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, Except.pure, bind, pure]
  
  set s6 := ((((((s.setReg "i_k" .nat [] (Tile.scalar (s.pids 0))).setReg
      "i_c" .nat [] (Tile.scalar (s.pids 1))).setReg
      "i_bh" .nat [] (Tile.scalar (s.pids 2))).setReg
      "p_g" .ptr [BK] (fwdPgTile s G s_qk_h DK BT BK 0)).setReg
      "p_go" .ptr [BK] (fwdPgoTile s GO s_qk_h DK BT BK 0)).setReg
      "cum_decay" .real [BK] (fwdCumTile s G s_qk_h DK BT BK 0)).setReg
      "mask" .bool [BK] (fwdMaskTile s DK BK) with hs6
  
  set s_ik := s.setReg "i_k" .nat [] (Tile.scalar (s.pids 0)) with hs_ik
  set s_ic := s_ik.setReg "i_c" .nat [] (Tile.scalar (s.pids 1)) with hs_ic
  set s_ibh := s_ic.setReg "i_bh" .nat [] (Tile.scalar (s.pids 2)) with hs_ibh
  have hibh : s_ibh.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by simp [hs_ibh]
  have hic : s_ibh.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by
    rw [hs_ibh, hs_ic]; simp
  have hik : s_ibh.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by
    rw [hs_ibh, hs_ic, hs_ik]; simp
  
  have hpgeval : evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase G)
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.add NumericDType.nat Broadcast.nil
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
            (Op.mul NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
              (Op.constNat DK)))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
        (Op.arange BK))) s_ibh = some (fwdPgTile s G s_qk_h DK BT BK 0) := by
    simp only [evalOp, Option.bind, hibh, hic, hik, evalOp_constNat, evalOp_arange,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
      Tile.bop, Tile.scalar]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [Tile.ptrAdd_data, fwdPgTile, offset, baseOffset,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
    refine Prod.ext rfl ?_
    simp only [Tile.vec]
    ring
  set s_pg := s_ibh.setReg "p_g" .ptr [BK] (fwdPgTile s G s_qk_h DK BT BK 0) with hs_pg
  have hpgostep_ibh : s_pg.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by
    rw [hs_pg]; simp [hibh]
  have hpgostep_ic : s_pg.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by
    rw [hs_pg]; simp [hic]
  have hpgostep_ik : s_pg.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by
    rw [hs_pg]; simp [hik]
  
  have hpgoeval : evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase GO)
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.add NumericDType.nat Broadcast.nil
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
            (Op.mul NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
              (Op.constNat DK)))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
        (Op.arange BK))) s_pg = some (fwdPgoTile s GO s_qk_h DK BT BK 0) := by
    simp only [evalOp, Option.bind, hpgostep_ibh, hpgostep_ic, hpgostep_ik, evalOp_constNat,
      evalOp_arange, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
      Tile.bop, Tile.scalar]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [Tile.ptrAdd_data, fwdPgoTile, offset, baseOffset,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
    refine Prod.ext rfl ?_
    simp only [Tile.vec]
    ring
  set s_pgo := s_pg.setReg "p_go" .ptr [BK] (fwdPgoTile s GO s_qk_h DK BT BK 0) with hs_pgo
  
  have hcumeval : evalOp (Op.full [BK] (Op.const 0)) s_pgo = some (fwdCumTile s G s_qk_h DK BT BK 0) := by
    simp only [evalOp_full, evalOp_const, Option.bind]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [fwdCumTile, Tile.scalar, fwdPartial, Finset.univ_eq_empty, Finset.sum_empty,
      mul_zero]
    by_cases ha : active s DK BK i <;> simp [ha]
  set s_cum := s_pgo.setReg "cum_decay" .real [BK] (fwdCumTile s G s_qk_h DK BT BK 0) with hs_cum
  have hcum_ik : s_cum.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by
    rw [hs_cum, hs_pgo, hs_pg]; simp [hik]
  
  have hmaskeval : evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK))
        (Op.arange BK)) (Op.constNat DK)) s_cum = some (fwdMaskTile s DK BK) := by
    simp only [evalOp, Option.bind, hcum_ik, evalOp_constNat, evalOp_arange,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
      ComparableDType.lt, Tile.bop, Tile.cop, Tile.scalar]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [fwdMaskTile, active, elemIndex, Tile.vec,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
    rfl
  
  have hstepK : stepStmt (Stmt.assign TileDType.nat [] "i_k" (Op.programId 0)) s = some s_ik := by
    rw [hs_ik]; exact stepStmt_assign_eq_some (by simp [evalOp])
  have hstepC : stepStmt (Stmt.assign TileDType.nat [] "i_c" (Op.programId 1)) s_ik = some s_ic := by
    rw [hs_ic]; exact stepStmt_assign_eq_some (show evalOp (Op.programId 1) s_ik = _ from by
      simp [evalOp, hs_ik])
  have hstepB : stepStmt (Stmt.assign TileDType.nat [] "i_bh" (Op.programId 2)) s_ic = some s_ibh := by
    rw [hs_ibh]; exact stepStmt_assign_eq_some (show evalOp (Op.programId 2) s_ic = _ from by
      simp [evalOp, hs_ic, hs_ik])
  have hstepPg : stepStmt (Stmt.assign TileDType.ptr [BK] "p_g"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase G)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.arange BK)))) s_ibh = some s_pg := by
    rw [hs_pg]; exact stepStmt_assign_eq_some hpgeval
  have hstepPgo : stepStmt (Stmt.assign TileDType.ptr [BK] "p_go"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase GO)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.arange BK)))) s_pg = some s_pgo := by
    rw [hs_pgo]; exact stepStmt_assign_eq_some hpgoeval
  have hstepCum : stepStmt (Stmt.assign TileDType.real [BK] "cum_decay" (Op.full [BK] (Op.const 0)))
      s_pgo = some s_cum := by
    rw [hs_cum]; exact stepStmt_assign_eq_some hcumeval
  have hstepMask : stepStmt (Stmt.assign TileDType.bool [BK] "mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK))
          (Op.arange BK)) (Op.constNat DK))) s_cum
      = some (s_cum.setReg "mask" .bool [BK] (fwdMaskTile s DK BK)) :=
    stepStmt_assign_eq_some hmaskeval
  rw [stepStmts.cons_some hstepK]
  rw [stepStmts.cons_some hstepC]
  rw [stepStmts.cons_some hstepB]
  rw [stepStmts.cons_some hstepPg]
  rw [stepStmts.cons_some hstepPgo]
  erw [stepStmts.cons_some hstepCum]
  erw [stepStmts.cons_some hstepMask]
  have hs_cum_to_s6 :
      s_cum.setReg "mask" .bool [BK] (fwdMaskTile s DK BK) = s6 := by
    rw [hs6, hs_cum, hs_pgo, hs_pg, hs_ibh, hs_ic, hs_ik]
  
  rw [show ∀ (X : Stmt), stepStmts [X] (s_cum.setReg "mask" .bool [BK] (fwdMaskTile s DK BK))
        = stepStmt X (s_cum.setReg "mask" .bool [BK] (fwdMaskTile s DK BK)) from
      fun X => by cases h : stepStmt X (s_cum.setReg "mask" .bool [BK] (fwdMaskTile s DK BK)) <;>
        simp [stepStmts, h]]
  rw [hs_cum_to_s6]
  
  have hP0 : fwdInv G GO s s_qk_h DK BT BK 0 s6 := by
    have hsub : s6 = ((((((s.setReg "i_k" .nat [] (Tile.scalar (s.pids 0))).setReg
        "i_c" .nat [] (Tile.scalar (s.pids 1))).setReg
        "i_bh" .nat [] (Tile.scalar (s.pids 2))).setReg
        "p_g" .ptr [BK] (fwdPgTile s G s_qk_h DK BT BK 0)).setReg
        "p_go" .ptr [BK] (fwdPgoTile s GO s_qk_h DK BT BK 0)).setReg
        "cum_decay" .real [BK] (fwdCumTile s G s_qk_h DK BT BK 0)).setReg
        "mask" .bool [BK] (fwdMaskTile s DK BK) := by
      rw [hs6, hs_cum, hs_pgo, hs_pg, hs_ibh, hs_ic, hs_ik]
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hsub]; simp only [BlockState.setReg_pids]
    · intro a; rw [hsub]; simp only [BlockState.setReg_readMem]
    · rw [hsub, BlockState.setReg_ne_name (h := by decide), BlockState.setReg_same]
    · rw [hsub, BlockState.setReg_ne_name (h := by decide),
        BlockState.setReg_ne_name (h := by decide), BlockState.setReg_ne_name (h := by decide),
        BlockState.setReg_same]
    · rw [hsub, BlockState.setReg_ne_name (h := by decide),
        BlockState.setReg_ne_name (h := by decide), BlockState.setReg_same]
    · rw [hsub, BlockState.setReg_same]
    · intro j hj i; rw [hsub]; simp only [BlockState.setReg_readMem]
    · intro j hj i; omega
  
  obtain ⟨final, sfinal, hLoop, hfinal, hPfinal⟩ :=
    forRange_inv (idx := "i") (start := 0) (stop := BT) (step := 1)
      (body := fwdBody G GO s_qk_h BT BK DK)
      (P := fwdInv G GO s s_qk_h DK BT BK) (s_init := s6) (Nat.one_ne_zero) hP0
      (fun j st _ hPj => fwd_decay_cumsum_step G GO s s_qk_h BT BK DK hne hBK j st hPj)
  simp only [fwdBody] at hLoop
  erw [hLoop]
  simp only [Option.map_some]
  
  obtain ⟨_, _, _, _, _, _, _, hreadGO⟩ := hPfinal
  have ht : t_rel.val < final := lt_of_lt_of_le t_rel.isLt hfinal
  rw [hreadGO t_rel.val ht i]
  simp only [fwdDecayClosed, fwdPartial]

theorem fwd_decay_cumsum_surface_closed_compute_correct_general
    (G GO : RegionName)
    (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ)
    (BT BK DK : Nat) (s : BlockState) (t_rel : Fin BT)
    (hne : G ≠ GO) (hBK : BK ≤ DK) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := fwd_decay_cumsum_surface G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s DK BK)
        (fun i => (GO, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        fwdDecayClosed s G s_qk_h DK BT BK t_rel i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fwd_decay_cumsum_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fwd_decay_cumsum_full_surface_closed_general G GO
    s_qk_h s_qk_t s_qk_d B H T scale BT BK DK s t_rel i hne hBK
  rw [hExec] at h
  simp only [Option.map_some] at h
  rw [Option.some.injEq] at h
  show s'.readMem GO (offset s s_qk_h DK t_rel.val BT BK i) = _
  rw [h, if_pos hActive]

noncomputable def prepareQgClosed
    (s : BlockState) (Q G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) (scale : ℝ) : ℝ :=
  s.readMem Q (offset s s_qk_h DK t_rel.val BT BK i) *
    Real.exp (s.readMem G (offset s s_qk_h DK t_rel.val BT BK i) * Real.log 2) *
    scale

noncomputable def prepareKgClosed
    (s : BlockState) (K G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) : ℝ :=
  s.readMem K (offset s s_qk_h DK t_rel.val BT BK i) *
    Real.exp ((s.readMem G (offset s s_qk_h DK (BT - 1) BT BK i) -
      s.readMem G (offset s s_qk_h DK t_rel.val BT BK i)) * Real.log 2)

noncomputable def prepLastDecayTile
    (s : BlockState) (G : RegionName)
    (s_qk_h DK BT BK : Nat) : Tile .real [BK] :=
  ⟨fun idx => some (s.readMem G (offset s s_qk_h DK (BT - 1) BT BK idx.1))⟩

def prepOffsTile (BK : Nat) : Tile .nat [BK] := Tile.vec (fun i : Fin BK => i.val)

def prepPtrTile
    (s : BlockState) (R : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) : Tile .ptr [BK] :=
  ⟨fun idx => (R, offset s s_qk_h DK m BT BK idx.1)⟩

noncomputable def prepLoadTile
    (s : BlockState) (R : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) : Tile .real [BK] :=
  ⟨fun idx => if active s DK BK idx.1 then
      some (s.readMem R (offset s s_qk_h DK m BT BK idx.1)) else some 0⟩

noncomputable def prepQValTile
    (s : BlockState) (Q G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (scale : ℝ) : Tile .real [BK] :=
  ⟨fun idx => if active s DK BK idx.1 then
      some (prepareQgClosed s Q G s_qk_h DK BT BK t_rel idx.1 scale) else some 0⟩

noncomputable def prepKValTile
    (s : BlockState) (K G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) : Tile .real [BK] :=
  ⟨fun idx => if active s DK BK idx.1 then
      some (prepareKgClosed s K G s_qk_h DK BT BK t_rel idx.1) else some 0⟩

noncomputable def prepInv
    (Q K G QG KG : RegionName) (s : BlockState) (s_qk_h DK BT BK : Nat) (scale : ℝ) :
    Nat → BlockState → Prop :=
  fun m sc =>
    sc.pids = s.pids ∧
    (∀ a, sc.readMem Q a = s.readMem Q a) ∧
    (∀ a, sc.readMem K a = s.readMem K a) ∧
    (∀ a, sc.readMem G a = s.readMem G a) ∧
    sc.regs .ptr [BK] "p_q" = some (prepPtrTile s Q s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_g" = some (prepPtrTile s G s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_k" = some (prepPtrTile s K s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_qg" = some (prepPtrTile s QG s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_kg" = some (prepPtrTile s KG s_qk_h DK BT BK m) ∧
    sc.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) ∧
    sc.regs .real [BK] "last_decay" = some (prepLastDecayTile s G s_qk_h DK BT BK) ∧
    (∀ j, m ≤ j → ∀ i : Fin BK,
      sc.readMem QG (offset s s_qk_h DK j BT BK i) =
        s.readMem QG (offset s s_qk_h DK j BT BK i)) ∧
    (∀ j, m ≤ j → ∀ i : Fin BK,
      sc.readMem KG (offset s s_qk_h DK j BT BK i) =
        s.readMem KG (offset s s_qk_h DK j BT BK i)) ∧
    (∀ j, j < m → ∀ i : Fin BK, ∀ (hj : j < BT),
      sc.readMem QG (offset s s_qk_h DK j BT BK i) =
        if active s DK BK i then
          prepareQgClosed s Q G s_qk_h DK BT BK ⟨j, hj⟩ i scale
        else s.readMem QG (offset s s_qk_h DK j BT BK i)) ∧
    (∀ j, j < m → ∀ i : Fin BK, ∀ (hj : j < BT),
      sc.readMem KG (offset s s_qk_h DK j BT BK i) =
        if active s DK BK i then
          prepareKgClosed s K G s_qk_h DK BT BK ⟨j, hj⟩ i
        else s.readMem KG (offset s s_qk_h DK j BT BK i))

def prepBody (Q K G QG KG : RegionName) (s_qk_h DK BT BK : Nat) (scale : ℝ) : List Stmt :=
  [Stmt.assign TileDType.real [BK] "_q"
      (Op.load TileDType.real (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_q"))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign TileDType.real [BK] "_k"
      (Op.load TileDType.real (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_k"))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign TileDType.real [BK] "_g"
      (Op.load ComputeDType.fp32.eraseDType (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_g"))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign TileDType.real [BK] "_q"
      (Op.mul NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "_q")
        (Op.mul NumericDType.real Broadcast.scalarR (Op.ref TileDType.real [BK] "_g").exp2 (Op.const scale))),
    Stmt.assign TileDType.real [BK] "_k"
      (Op.mul NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "_k")
        (Op.sub NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "last_decay")
            (Op.ref TileDType.real [BK] "_g")).exp2),
    Stmt.store TileDType.real [BK] (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_kg"))
      (Op.ref TileDType.real [BK] "_k") (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask")),
    Stmt.store TileDType.real [BK] (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_qg"))
      (Op.ref TileDType.real [BK] "_q") (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask")),
    Stmt.assign TileDType.ptr [BK] "p_q"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_q") (Op.constNat DK)),
    Stmt.assign TileDType.ptr [BK] "p_g"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_g") (Op.constNat DK)),
    Stmt.assign TileDType.ptr [BK] "p_k"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_k") (Op.constNat DK)),
    Stmt.assign TileDType.ptr [BK] "p_kg"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_kg") (Op.constNat DK)),
    Stmt.assign TileDType.ptr [BK] "p_qg"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_qg") (Op.constNat DK))]

set_option maxHeartbeats 4000000 in
                                                          
theorem prepare_qg_kg_step
    (Q K G QG KG : RegionName) (s : BlockState) (s_qk_h DK BT BK : Nat) (scale : ℝ)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hK_QG : K ≠ QG) (hK_KG : K ≠ KG)
    (hG_QG : G ≠ QG) (hG_KG : G ≠ KG) (hQG_KG : QG ≠ KG) (hBK : BK ≤ DK) (hBT : 0 < BT)
    (m : Nat) (hm : m < BT) (sc : BlockState) (hP : prepInv Q K G QG KG s s_qk_h DK BT BK scale m sc) :
    ∃ s',
      stepStmts (prepBody Q K G QG KG s_qk_h DK BT BK scale)
        (sc.setReg "i" .nat [] (Tile.scalar m)) = some s' ∧
      prepInv Q K G QG KG s s_qk_h DK BT BK scale (m + 1) s' := by
  obtain ⟨hpids, hQmem, hKmem, hGmem, hpq, hpg, hpk, hpqg, hpkg, hmask, hld,
    hUQG, hUKG, hreadQG, hreadKG⟩ := hP
  set sloop := sc.setReg "i" .nat [] (Tile.scalar m) with hsloop
  
  have hpq' : sloop.regs .ptr [BK] "p_q" = some (prepPtrTile s Q s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpq
  have hpg' : sloop.regs .ptr [BK] "p_g" = some (prepPtrTile s G s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpg
  have hpk' : sloop.regs .ptr [BK] "p_k" = some (prepPtrTile s K s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpk
  have hpqg' : sloop.regs .ptr [BK] "p_qg" = some (prepPtrTile s QG s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpqg
  have hpkg' : sloop.regs .ptr [BK] "p_kg" = some (prepPtrTile s KG s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpkg
  have hmask' : sloop.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
    rw [hsloop]; simpa using hmask
  have hld' : sloop.regs .real [BK] "last_decay" = some (prepLastDecayTile s G s_qk_h DK BT BK) := by
    rw [hsloop]; simpa using hld
  
  have hloadeval : ∀ (R : RegionName) (name : RegName) (st : BlockState),
      st.regs .ptr [BK] name = some (prepPtrTile s R s_qk_h DK BT BK m) →
      st.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) →
      (∀ a, st.readMem R a = s.readMem R a) →
      evalOp (Op.load TileDType.real (MemAccess.ptr (Op.ref TileDType.ptr [BK] name))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BK] "mask") ((Op.const 0).broadcast [BK]))) st
        = some (prepLoadTile s R s_qk_h DK BT BK m) := by
    intro R name st hp hmk hmem
    simp only [evalOp, Option.bind, hp, hmk, evalOp_const]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [prepLoadTile, fwdMaskTile, prepPtrTile]
    by_cases ha : active s DK BK i
    · simp only [ha, decide_true, if_true, if_pos]
      rw [BlockState.readMemValue_real, hmem]
    · simp only [ha, decide_false, Bool.false_eq_true, if_false]
      rfl
  
  have hqeval := hloadeval Q "p_q" sloop hpq' hmask' (fun a => by rw [hsloop]; simpa using hQmem a)
  set s1 := sloop.setReg "_q" .real [BK] (prepLoadTile s Q s_qk_h DK BT BK m) with hs1
  have hstep1 : stepStmt (Stmt.assign TileDType.real [BK] "_q"
      (Op.load TileDType.real (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_q"))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BK] "mask") ((Op.const 0).broadcast [BK])))) sloop
      = some s1 := by
    rw [hs1]; exact stepStmt_assign_eq_some hqeval
  
  have hs1_pk : s1.regs .ptr [BK] "p_k" = some (prepPtrTile s K s_qk_h DK BT BK m) := by rw [hs1]; simpa using hpk'
  have hs1_mask : s1.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by rw [hs1]; simpa using hmask'
  have hs1_Kmem : ∀ a, s1.readMem K a = s.readMem K a := by intro a; rw [hs1]; simp only [BlockState.setReg_readMem]; rw [hsloop]; simpa using hKmem a
  have hkeval := hloadeval K "p_k" s1 hs1_pk hs1_mask hs1_Kmem
  set s2 := s1.setReg "_k" .real [BK] (prepLoadTile s K s_qk_h DK BT BK m) with hs2
  have hstep2 : stepStmt (Stmt.assign TileDType.real [BK] "_k"
      (Op.load TileDType.real (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_k"))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BK] "mask") ((Op.const 0).broadcast [BK])))) s1
      = some s2 := by
    rw [hs2]; exact stepStmt_assign_eq_some hkeval
  
  have hs2_pg : s2.regs .ptr [BK] "p_g" = some (prepPtrTile s G s_qk_h DK BT BK m) := by
    rw [hs2, hs1]; simpa using hpg'
  have hs2_mask : s2.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by rw [hs2, hs1]; simpa using hmask'
  have hs2_Gmem : ∀ a, s2.readMem G a = s.readMem G a := by intro a; rw [hs2, hs1]; simp only [BlockState.setReg_readMem]; rw [hsloop]; simpa using hGmem a
  have hgeval := hloadeval G "p_g" s2 hs2_pg hs2_mask hs2_Gmem
  set s3 := s2.setReg "_g" .real [BK] (prepLoadTile s G s_qk_h DK BT BK m) with hs3
  have hstep3 : stepStmt (Stmt.assign TileDType.real [BK] "_g"
      (Op.load ComputeDType.fp32.eraseDType (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_g"))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BK] "mask") ((Op.const 0).broadcast [BK])))) s2
      = some s3 := by
    rw [hs3]; exact stepStmt_assign_eq_some (by simpa [ComputeDType.eraseDType] using hgeval)
  
  have hs3_qval : s3.regs .real [BK] "_q" = some (prepLoadTile s Q s_qk_h DK BT BK m) := by
    rw [hs3, hs2]; simp [hs1]
  have hs3_gval : s3.regs .real [BK] "_g" = some (prepLoadTile s G s_qk_h DK BT BK m) := by
    rw [hs3]; simp
  have hqscaleeval : evalOp (Op.mul NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "_q")
      (Op.mul NumericDType.real Broadcast.scalarR (Op.ref TileDType.real [BK] "_g").exp2 (Op.const scale))) s3
      = some (prepQValTile s Q G s_qk_h DK BT BK ⟨m, hm⟩ scale) := by
    simp only [evalOp, Option.bind, hs3_qval, hs3_gval, evalOp_const]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [prepQValTile, prepLoadTile, Tile.uop_data, Tile.bop_data, Broadcast.leftIndex,
      Broadcast.rightIndex, NumericDType.mul]
    by_cases ha : active s DK BK i
    · simp only [ha, if_true, WithBot.realExp2, WithBot.realMul]
      simp only [prepareQgClosed, offset, baseOffset, Tile.scalar]
      simp [Tile.vec, mul_assoc]
    · simp only [ha, if_false, WithBot.realExp2, WithBot.realMul, Tile.scalar]
      simp [Tile.vec]
  set s4 := s3.setReg "_q" .real [BK] (prepQValTile s Q G s_qk_h DK BT BK ⟨m, hm⟩ scale) with hs4
  have hstep4 : stepStmt (Stmt.assign TileDType.real [BK] "_q"
      (Op.mul NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "_q")
        (Op.mul NumericDType.real Broadcast.scalarR (Op.ref TileDType.real [BK] "_g").exp2 (Op.const scale)))) s3
      = some s4 := by
    rw [hs4]; exact stepStmt_assign_eq_some hqscaleeval
  
  have hs4_kval : s4.regs .real [BK] "_k" = some (prepLoadTile s K s_qk_h DK BT BK m) := by
    rw [hs4, hs3, hs2]; simp
  have hs4_ld : s4.regs .real [BK] "last_decay" = some (prepLastDecayTile s G s_qk_h DK BT BK) := by
    rw [hs4, hs3, hs2, hs1]; simpa using hld'
  have hs4_gval : s4.regs .real [BK] "_g" = some (prepLoadTile s G s_qk_h DK BT BK m) := by
    rw [hs4, hs3]; simp
  have hkscaleeval : evalOp (Op.mul NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "_k")
      (Op.sub NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "last_decay")
          (Op.ref TileDType.real [BK] "_g")).exp2) s4
      = some (prepKValTile s K G s_qk_h DK BT BK ⟨m, hm⟩) := by
    simp only [evalOp, Option.bind, hs4_kval, hs4_ld, hs4_gval]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [prepKValTile, prepLoadTile, prepLastDecayTile, Tile.uop_data, Tile.bop_data,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, NumericDType.sub]
    by_cases ha : active s DK BK i
    · simp only [ha, if_true, WithBot.realExp2, WithBot.realMul, WithBot.realSub]
      simp only [prepareKgClosed, offset, baseOffset]
      simp [Tile.vec]
    · simp only [ha, if_false, WithBot.realExp2, WithBot.realMul, WithBot.realSub]
      simp [Tile.vec]
  set s5 := s4.setReg "_k" .real [BK] (prepKValTile s K G s_qk_h DK BT BK ⟨m, hm⟩) with hs5
  have hstep5 : stepStmt (Stmt.assign TileDType.real [BK] "_k"
      (Op.mul NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "_k")
        (Op.sub NumericDType.real Broadcast.nil.consSame (Op.ref TileDType.real [BK] "last_decay")
            (Op.ref TileDType.real [BK] "_g")).exp2)) s4
      = some s5 := by
    rw [hs5]; exact stepStmt_assign_eq_some hkscaleeval
  
  have hs5_pkg : s5.regs .ptr [BK] "p_kg" = some (prepPtrTile s KG s_qk_h DK BT BK m) := by
    rw [hs5, hs4, hs3, hs2, hs1]; simpa using hpkg'
  have hs5_mask : s5.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
    rw [hs5, hs4, hs3, hs2, hs1]; simpa using hmask'
  have hs5_kval : s5.regs .real [BK] "_k" = some (prepKValTile s K G s_qk_h DK BT BK ⟨m, hm⟩) := by
    rw [hs5]; simp
  set kvfun : TileIndex [BK] → ℝ := fun k =>
    (prepKValTile s K G s_qk_h DK BT BK ⟨m, hm⟩).data k |>.unbotD 0 with hkvfun
  set sstK := (TileShape.allIndices [BK]).foldl
    (fun (acc : BlockState) k =>
      if active s DK BK k.1 then
        acc.writeMem KG (offset s s_qk_h DK m BT BK k.1) (kvfun k)
      else acc) s5 with hsstK
  have hstoreK : stepStmt (Stmt.store TileDType.real [BK] (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_kg"))
      (Op.ref TileDType.real [BK] "_k") (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask"))) s5
      = some sstK := by
    simp only [stepStmt, evalOp, Option.bind, Option.map, hs5_kval, hs5_pkg, hs5_mask]
    apply congrArg some
    rw [hsstK]
    congr 1
    funext acc k
    simp only [prepPtrTile, fwdMaskTile]
    by_cases ha : active s DK BK k.1
    · rw [if_pos (by simpa using ha), if_pos ha, BlockState.writeMemTyped_real]; rfl
    · rw [if_neg (by simpa using ha), if_neg ha]
  
  have hsstK_pqg : sstK.regs .ptr [BK] "p_qg" = some (prepPtrTile s QG s_qk_h DK BT BK m) := by
    rw [hsstK, BlockState.foldl_writeMem_prop_masked_regs, hs5, hs4, hs3, hs2, hs1]; simpa using hpqg'
  have hsstK_mask : sstK.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
    rw [hsstK, BlockState.foldl_writeMem_prop_masked_regs, hs5, hs4, hs3, hs2, hs1]; simpa using hmask'
  have hsstK_qval : sstK.regs .real [BK] "_q" = some (prepQValTile s Q G s_qk_h DK BT BK ⟨m, hm⟩ scale) := by
    rw [hsstK, BlockState.foldl_writeMem_prop_masked_regs, hs5, hs4]; simp
  set qvfun : TileIndex [BK] → ℝ := fun k =>
    (prepQValTile s Q G s_qk_h DK BT BK ⟨m, hm⟩ scale).data k |>.unbotD 0 with hqvfun
  set sstQ := (TileShape.allIndices [BK]).foldl
    (fun (acc : BlockState) k =>
      if active s DK BK k.1 then
        acc.writeMem QG (offset s s_qk_h DK m BT BK k.1) (qvfun k)
      else acc) sstK with hsstQ
  have hstoreQ : stepStmt (Stmt.store TileDType.real [BK] (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_qg"))
      (Op.ref TileDType.real [BK] "_q") (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask"))) sstK
      = some sstQ := by
    simp only [stepStmt, evalOp, Option.bind, Option.map, hsstK_qval, hsstK_pqg, hsstK_mask]
    apply congrArg some
    rw [hsstQ]
    congr 1
    funext acc k
    simp only [prepPtrTile, fwdMaskTile]
    by_cases ha : active s DK BK k.1
    · rw [if_pos (by simpa using ha), if_pos ha, BlockState.writeMemTyped_real]; rfl
    · rw [if_neg (by simpa using ha), if_neg ha]
  
  have hincr : ∀ (R : RegionName) (name : RegName) (st : BlockState),
      st.regs .ptr [BK] name = some (prepPtrTile s R s_qk_h DK BT BK m) →
      evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] name) (Op.constNat DK)) st
        = some (prepPtrTile s R s_qk_h DK BT BK (m + 1)) := by
    intro R name st hp
    simp only [evalOp, Option.bind, hp, evalOp_constNat]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [Tile.ptrAdd_data, Broadcast.leftIndex, Broadcast.rightIndex, prepPtrTile, Tile.scalar]
    refine Prod.ext rfl ?_
    simp only [offset, baseOffset]; ring
  
  have hsstQ_pq : sstQ.regs .ptr [BK] "p_q" = some (prepPtrTile s Q s_qk_h DK BT BK m) := by
    rw [hsstQ, BlockState.foldl_writeMem_prop_masked_regs, hsstK,
      BlockState.foldl_writeMem_prop_masked_regs, hs5, hs4, hs3, hs2, hs1]; simpa using hpq'
  set s6 := sstQ.setReg "p_q" .ptr [BK] (prepPtrTile s Q s_qk_h DK BT BK (m + 1)) with hs6
  have hstep6 : stepStmt (Stmt.assign TileDType.ptr [BK] "p_q"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_q") (Op.constNat DK))) sstQ = some s6 := by
    rw [hs6]; exact stepStmt_assign_eq_some (hincr Q "p_q" sstQ hsstQ_pq)
  have hs6_pg : s6.regs .ptr [BK] "p_g" = some (prepPtrTile s G s_qk_h DK BT BK m) := by
    rw [hs6, BlockState.setReg_ne_name (h := by decide), hsstQ,
      BlockState.foldl_writeMem_prop_masked_regs, hsstK,
      BlockState.foldl_writeMem_prop_masked_regs, hs5, hs4, hs3, hs2, hs1]; simpa using hpg'
  set s7 := s6.setReg "p_g" .ptr [BK] (prepPtrTile s G s_qk_h DK BT BK (m + 1)) with hs7
  have hstep7 : stepStmt (Stmt.assign TileDType.ptr [BK] "p_g"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_g") (Op.constNat DK))) s6 = some s7 := by
    rw [hs7]; exact stepStmt_assign_eq_some (hincr G "p_g" s6 hs6_pg)
  have hs7_pk : s7.regs .ptr [BK] "p_k" = some (prepPtrTile s K s_qk_h DK BT BK m) := by
    rw [hs7, BlockState.setReg_ne_name (h := by decide),
      hs6, BlockState.setReg_ne_name (h := by decide), hsstQ,
      BlockState.foldl_writeMem_prop_masked_regs, hsstK,
      BlockState.foldl_writeMem_prop_masked_regs, hs5, hs4, hs3, hs2]; simpa using hpk'
  set s8 := s7.setReg "p_k" .ptr [BK] (prepPtrTile s K s_qk_h DK BT BK (m + 1)) with hs8
  have hstep8 : stepStmt (Stmt.assign TileDType.ptr [BK] "p_k"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_k") (Op.constNat DK))) s7 = some s8 := by
    rw [hs8]; exact stepStmt_assign_eq_some (hincr K "p_k" s7 hs7_pk)
  have hs8_pkg : s8.regs .ptr [BK] "p_kg" = some (prepPtrTile s KG s_qk_h DK BT BK m) := by
    rw [hs8, BlockState.setReg_ne_name (h := by decide),
      hs7, BlockState.setReg_ne_name (h := by decide),
      hs6, BlockState.setReg_ne_name (h := by decide), hsstQ,
      BlockState.foldl_writeMem_prop_masked_regs, hsstK,
      BlockState.foldl_writeMem_prop_masked_regs, hs5, hs4, hs3, hs2, hs1]; simpa using hpkg'
  set s9 := s8.setReg "p_kg" .ptr [BK] (prepPtrTile s KG s_qk_h DK BT BK (m + 1)) with hs9
  have hstep9 : stepStmt (Stmt.assign TileDType.ptr [BK] "p_kg"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_kg") (Op.constNat DK))) s8 = some s9 := by
    rw [hs9]; exact stepStmt_assign_eq_some (hincr KG "p_kg" s8 hs8_pkg)
  have hs9_pqg : s9.regs .ptr [BK] "p_qg" = some (prepPtrTile s QG s_qk_h DK BT BK m) := by
    rw [hs9, BlockState.setReg_ne_name (h := by decide),
      hs8, BlockState.setReg_ne_name (h := by decide),
      hs7, BlockState.setReg_ne_name (h := by decide),
      hs6, BlockState.setReg_ne_name (h := by decide), hsstQ,
      BlockState.foldl_writeMem_prop_masked_regs, hsstK,
      BlockState.foldl_writeMem_prop_masked_regs, hs5, hs4, hs3, hs2, hs1]; simpa using hpqg'
  set s10 := s9.setReg "p_qg" .ptr [BK] (prepPtrTile s QG s_qk_h DK BT BK (m + 1)) with hs10
  have hstep10 : stepStmt (Stmt.assign TileDType.ptr [BK] "p_qg"
      (Op.ptrAdd Broadcast.scalarR (Op.ref TileDType.ptr [BK] "p_qg") (Op.constNat DK))) s9 = some s10 := by
    rw [hs10]; exact stepStmt_assign_eq_some (hincr QG "p_qg" s9 hs9_pqg)
  refine ⟨s10, ?_, ?_⟩
  · simp only [prepBody]
    rw [stepStmts.cons_some hstep1, stepStmts.cons_some hstep2, stepStmts.cons_some hstep3,
        stepStmts.cons_some hstep4, stepStmts.cons_some hstep5, stepStmts.cons_some hstoreK,
        stepStmts.cons_some hstoreQ, stepStmts.cons_some hstep6, stepStmts.cons_some hstep7,
        stepStmts.cons_some hstep8, stepStmts.cons_some hstep9, stepStmts.cons_some hstep10,
        stepStmts.nil]
  · 
    
    have hInj : Function.Injective (fun k : TileIndex [BK] =>
        offset s s_qk_h DK m BT BK k.1) := by
      rintro ⟨a, _⟩ ⟨b, _⟩ hab
      simp only [offset] at hab
      obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
      rfl
    
    have hsep : ∀ (j : Nat), j ≠ m → ∀ (i : Fin BK),
        (∀ k : TileIndex [BK], active s DK BK k.1 →
          (fun k : TileIndex [BK] => offset s s_qk_h DK m BT BK k.1) k
            ≠ offset s s_qk_h DK j BT BK i) := by
      intro j hj i k _
      simp only [offset, baseOffset]
      have hik : (i : ℕ) < BK := i.isLt
      have hiDK : (i : ℕ) < DK := lt_of_lt_of_le hik hBK
      rcases Nat.lt_or_ge j m with hlt | hge'
      · have hge : (s.pids 1 * BT + j) * DK + DK ≤ (s.pids 1 * BT + m) * DK := by
          have : (s.pids 1 * BT + j) + 1 ≤ s.pids 1 * BT + m := by omega
          calc (s.pids 1 * BT + j) * DK + DK
              = ((s.pids 1 * BT + j) + 1) * DK := by ring
            _ ≤ (s.pids 1 * BT + m) * DK := Nat.mul_le_mul_right DK this
        omega
      · have hjm' : m < j := by omega
        have hkBK : (k.1 : ℕ) < BK := k.1.isLt
        have hkDK : (k.1 : ℕ) < DK := lt_of_lt_of_le hkBK hBK
        have hge : (s.pids 1 * BT + m) * DK + DK ≤ (s.pids 1 * BT + j) * DK := by
          have : (s.pids 1 * BT + m) + 1 ≤ s.pids 1 * BT + j := by omega
          calc (s.pids 1 * BT + m) * DK + DK
              = ((s.pids 1 * BT + m) + 1) * DK := by ring
            _ ≤ (s.pids 1 * BT + j) * DK := Nat.mul_le_mul_right DK this
        omega
    
    have hmemQG10 : ∀ a, s10.readMem QG a = sstQ.readMem QG a := by
      intro a; rw [hs10, hs9, hs8, hs7, hs6]; simp only [BlockState.setReg_readMem]
    have hmemKG10 : ∀ a, s10.readMem KG a = sstQ.readMem KG a := by
      intro a; rw [hs10, hs9, hs8, hs7, hs6]; simp only [BlockState.setReg_readMem]
    
    have hs5QG : ∀ a, s5.readMem QG a = sc.readMem QG a := by
      intro a; rw [hs5, hs4, hs3, hs2, hs1]; simp only [BlockState.setReg_readMem]
      rw [hsloop]; simp only [BlockState.setReg_readMem]
    have hs5KG : ∀ a, s5.readMem KG a = sc.readMem KG a := by
      intro a; rw [hs5, hs4, hs3, hs2, hs1]; simp only [BlockState.setReg_readMem]
      rw [hsloop]; simp only [BlockState.setReg_readMem]
    
    have hqvfun_active : ∀ i : Fin BK, active s DK BK i →
        qvfun (i, PUnit.unit) = prepareQgClosed s Q G s_qk_h DK BT BK ⟨m, hm⟩ i scale := by
      intro i ha; simp only [hqvfun, prepQValTile, ha, if_true]; rfl
    have hkvfun_active : ∀ i : Fin BK, active s DK BK i →
        kvfun (i, PUnit.unit) = prepareKgClosed s K G s_qk_h DK BT BK ⟨m, hm⟩ i := by
      intro i ha; simp only [hkvfun, prepKValTile, ha, if_true]; rfl
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · 
      rw [hs10, hs9, hs8, hs7, hs6]; simp only [BlockState.setReg_pids]
      rw [hsstQ, BlockState.foldl_writeMem_prop_masked_pids,
        hsstK, BlockState.foldl_writeMem_prop_masked_pids,
        hs5, hs4, hs3, hs2, hs1]; simp only [BlockState.setReg_pids]
      rw [hsloop]; simp only [BlockState.setReg_pids]; exact hpids
    · 
      intro a; rw [hs10, hs9, hs8, hs7, hs6]; simp only [BlockState.setReg_readMem]
      rw [hsstQ, BlockState.scatter_prop_masked_preserves_other_region
        (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := hQ_QG)]
      rw [hsstK, BlockState.scatter_prop_masked_preserves_other_region
        (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := hQ_KG)]
      rw [hs5, hs4, hs3, hs2, hs1]; simp only [BlockState.setReg_readMem]
      rw [hsloop]; simp only [BlockState.setReg_readMem]; exact hQmem a
    · 
      intro a; rw [hs10, hs9, hs8, hs7, hs6]; simp only [BlockState.setReg_readMem]
      rw [hsstQ, BlockState.scatter_prop_masked_preserves_other_region
        (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := hK_QG)]
      rw [hsstK, BlockState.scatter_prop_masked_preserves_other_region
        (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := hK_KG)]
      rw [hs5, hs4, hs3, hs2, hs1]; simp only [BlockState.setReg_readMem]
      rw [hsloop]; simp only [BlockState.setReg_readMem]; exact hKmem a
    · 
      intro a; rw [hs10, hs9, hs8, hs7, hs6]; simp only [BlockState.setReg_readMem]
      rw [hsstQ, BlockState.scatter_prop_masked_preserves_other_region
        (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := hG_QG)]
      rw [hsstK, BlockState.scatter_prop_masked_preserves_other_region
        (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := hG_KG)]
      rw [hs5, hs4, hs3, hs2, hs1]; simp only [BlockState.setReg_readMem]
      rw [hsloop]; simp only [BlockState.setReg_readMem]; exact hGmem a
    · 
      rw [hs10, BlockState.setReg_ne_name (h := by decide),
        hs9, BlockState.setReg_ne_name (h := by decide),
        hs8, BlockState.setReg_ne_name (h := by decide),
        hs7, BlockState.setReg_ne_name (h := by decide), hs6, BlockState.setReg_same]
    · 
      rw [hs10, BlockState.setReg_ne_name (h := by decide),
        hs9, BlockState.setReg_ne_name (h := by decide),
        hs8, BlockState.setReg_ne_name (h := by decide), hs7, BlockState.setReg_same]
    · 
      rw [hs10, BlockState.setReg_ne_name (h := by decide),
        hs9, BlockState.setReg_ne_name (h := by decide), hs8, BlockState.setReg_same]
    · 
      rw [hs10, BlockState.setReg_same]
    · 
      rw [hs10, BlockState.setReg_ne_name (h := by decide), hs9, BlockState.setReg_same]
    · 
      rw [hs10, BlockState.setReg_ne_name (h := by decide),
        hs9, BlockState.setReg_ne_name (h := by decide),
        hs8, BlockState.setReg_ne_name (h := by decide),
        hs7, BlockState.setReg_ne_name (h := by decide),
        hs6, BlockState.setReg_ne_name (h := by decide)]
      rw [hsstQ, BlockState.foldl_writeMem_prop_masked_regs,
        hsstK, BlockState.foldl_writeMem_prop_masked_regs,
        hs5, hs4, hs3, hs2, hs1]; simpa using hmask'
    · 
      rw [hs10, BlockState.setReg_ne_name (h := by decide),
        hs9, BlockState.setReg_ne_name (h := by decide),
        hs8, BlockState.setReg_ne_name (h := by decide),
        hs7, BlockState.setReg_ne_name (h := by decide),
        hs6, BlockState.setReg_ne_name (h := by decide)]
      rw [hsstQ, BlockState.foldl_writeMem_prop_masked_regs,
        hsstK, BlockState.foldl_writeMem_prop_masked_regs,
        hs5, hs4, hs3, hs2, hs1]; simpa using hld'
    · 
      intro j hj i
      rw [hmemQG10 (offset s s_qk_h DK j BT BK i), hsstQ]
      rw [BlockState.scatter_prop_masked_preserves_other_offset
        (region := QG)
        (offsetFn := fun k : TileIndex [BK] => offset s s_qk_h DK m BT BK k.1)
        (valueFn := qvfun)
        (P := fun k : TileIndex [BK] => active s DK BK k.1)
        (off := offset s s_qk_h DK j BT BK i)
        (h_ne := hsep j (by omega) i)]
      rw [hsstK, BlockState.scatter_prop_masked_preserves_other_region
        (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := hQG_KG)]
      rw [hs5QG]
      exact hUQG j (by omega) i
    · 
      intro j hj i
      rw [hmemKG10 (offset s s_qk_h DK j BT BK i), hsstQ]
      rw [BlockState.scatter_prop_masked_preserves_other_region
        (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := Ne.symm hQG_KG)]
      rw [hsstK, BlockState.scatter_prop_masked_preserves_other_offset
        (region := KG)
        (offsetFn := fun k : TileIndex [BK] => offset s s_qk_h DK m BT BK k.1)
        (valueFn := kvfun)
        (P := fun k : TileIndex [BK] => active s DK BK k.1)
        (off := offset s s_qk_h DK j BT BK i)
        (h_ne := hsep j (by omega) i)]
      rw [hs5KG]
      exact hUKG j (by omega) i
    · 
      intro j hj i hjBT
      rw [hmemQG10 (offset s s_qk_h DK j BT BK i), hsstQ]
      by_cases hjm : j = m
      · subst hjm
        rw [BlockState.scatter_readback_prop_masked_nd sstK
          (fun k : TileIndex [BK] => offset s s_qk_h DK j BT BK k.1) qvfun
          (fun k => active s DK BK k.1) hInj (i, PUnit.unit)]
        by_cases ha : active s DK BK i
        · rw [if_pos ha, if_pos ha, hqvfun_active i ha]
        · rw [if_neg ha, if_neg ha]
          rw [hsstK, BlockState.scatter_prop_masked_preserves_other_region
            (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := hQG_KG)]
          rw [hs5QG]
          exact hUQG j (le_refl j) i
      · have hjlt : j < m := by omega
        rw [BlockState.scatter_prop_masked_preserves_other_offset
          (region := QG)
          (offsetFn := fun k : TileIndex [BK] => offset s s_qk_h DK m BT BK k.1)
          (valueFn := qvfun)
          (P := fun k : TileIndex [BK] => active s DK BK k.1)
          (off := offset s s_qk_h DK j BT BK i)
          (h_ne := hsep j (by omega) i)]
        rw [hsstK, BlockState.scatter_prop_masked_preserves_other_region
          (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := hQG_KG)]
        rw [hs5QG]
        exact hreadQG j hjlt i hjBT
    · 
      intro j hj i hjBT
      rw [hmemKG10 (offset s s_qk_h DK j BT BK i), hsstQ]
      rw [BlockState.scatter_prop_masked_preserves_other_region
        (P := fun k : TileIndex [BK] => active s DK BK k.1) (h_ne := Ne.symm hQG_KG)]
      by_cases hjm : j = m
      · subst hjm
        rw [hsstK, BlockState.scatter_readback_prop_masked_nd s5
          (fun k : TileIndex [BK] => offset s s_qk_h DK j BT BK k.1) kvfun
          (fun k => active s DK BK k.1) hInj (i, PUnit.unit)]
        by_cases ha : active s DK BK i
        · rw [if_pos ha, if_pos ha, hkvfun_active i ha]
        · rw [if_neg ha, if_neg ha]
          rw [hs5KG]
          exact hUKG j (le_refl j) i
      · have hjlt : j < m := by omega
        rw [hsstK, BlockState.scatter_prop_masked_preserves_other_offset
          (region := KG)
          (offsetFn := fun k : TileIndex [BK] => offset s s_qk_h DK m BT BK k.1)
          (valueFn := kvfun)
          (P := fun k : TileIndex [BK] => active s DK BK k.1)
          (off := offset s s_qk_h DK j BT BK i)
          (h_ne := hsep j (by omega) i)]
        rw [hs5KG]
        exact hreadKG j hjlt i hjBT

set_option maxHeartbeats 4000000 in

theorem prepare_qg_kg_loop_drive
    (Q K G QG KG : RegionName) (s : BlockState) (s_qk_h DK BT BK : Nat) (scale : ℝ)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hK_QG : K ≠ QG) (hK_KG : K ≠ KG)
    (hG_QG : G ≠ QG) (hG_KG : G ≠ KG) (hQG_KG : QG ≠ KG) (hBK : BK ≤ DK) (hBT : 0 < BT) :
    ∃ final s_final,
      exec (prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale) s = some s_final ∧
      BT ≤ final ∧ prepInv Q K G QG KG s s_qk_h DK BT BK scale final s_final := by
  simp only [exec, prepare_qg_kg_surface, ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, Except.pure, bind, pure]
  
  set s_ik := s.setReg "i_k" .nat [] (Tile.scalar (s.pids 0)) with hs_ik
  set s_ic := s_ik.setReg "i_c" .nat [] (Tile.scalar (s.pids 1)) with hs_ic
  set s_ibh := s_ic.setReg "i_bh" .nat [] (Tile.scalar (s.pids 2)) with hs_ibh
  set s_offs := s_ibh.setReg "offs" .nat [BK] (prepOffsTile BK) with hs_offs
  set s_pq := s_offs.setReg "p_q" .ptr [BK] (prepPtrTile s Q s_qk_h DK BT BK 0) with hs_pq
  set s_pg := s_pq.setReg "p_g" .ptr [BK] (prepPtrTile s G s_qk_h DK BT BK 0) with hs_pg
  set s_pk := s_pg.setReg "p_k" .ptr [BK] (prepPtrTile s K s_qk_h DK BT BK 0) with hs_pk
  set s_pqg := s_pk.setReg "p_qg" .ptr [BK] (prepPtrTile s QG s_qk_h DK BT BK 0) with hs_pqg
  set s_pkg := s_pqg.setReg "p_kg" .ptr [BK] (prepPtrTile s KG s_qk_h DK BT BK 0) with hs_pkg
  set s_mask := s_pkg.setReg "mask" .bool [BK] (fwdMaskTile s DK BK) with hs_mask
  set s11 := s_mask.setReg "last_decay" .real [BK] (prepLastDecayTile s G s_qk_h DK BT BK) with hs11
  
  have hibh : s_ibh.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by simp [hs_ibh]
  have hic : s_ibh.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by
    rw [hs_ibh, hs_ic]; simp
  have hik : s_ibh.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by
    rw [hs_ibh, hs_ic, hs_ik]; simp
  
  have hptr : ∀ (R : RegionName) (st : BlockState),
      st.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) →
      st.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) →
      st.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) →
      st.regs .nat [BK] "offs" = some (prepOffsTile BK) →
      evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase R)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.ref TileDType.nat [BK] "offs"))) st = some (prepPtrTile s R s_qk_h DK BT BK 0) := by
    intro R st hbh hc hk ho
    simp only [evalOp, Option.bind, hbh, hc, hk, ho, evalOp_constNat,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
      Tile.bop, Tile.scalar]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [Tile.ptrAdd_data, prepPtrTile, offset, baseOffset, prepOffsTile,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil, Tile.vec]
    refine Prod.ext rfl ?_
    ring
  
  have hstepK : stepStmt (Stmt.assign TileDType.nat [] "i_k" (Op.programId 0)) s = some s_ik := by
    rw [hs_ik]; exact stepStmt_assign_eq_some (by simp [evalOp])
  have hstepC : stepStmt (Stmt.assign TileDType.nat [] "i_c" (Op.programId 1)) s_ik = some s_ic := by
    rw [hs_ic]; exact stepStmt_assign_eq_some (show evalOp (Op.programId 1) s_ik = _ from by simp [evalOp, hs_ik])
  have hstepB : stepStmt (Stmt.assign TileDType.nat [] "i_bh" (Op.programId 2)) s_ic = some s_ibh := by
    rw [hs_ibh]; exact stepStmt_assign_eq_some (show evalOp (Op.programId 2) s_ic = _ from by simp [evalOp, hs_ic, hs_ik])
  have hstepOffs : stepStmt (Stmt.assign TileDType.nat [BK] "offs" (Op.arange BK)) s_ibh = some s_offs := by
    rw [hs_offs]; exact stepStmt_assign_eq_some (by simp only [evalOp_arange, prepOffsTile])
  
  have hoffs_at : ∀ (st : BlockState), st.regs .nat [BK] "offs" = some (prepOffsTile BK) →
      True := fun _ _ => trivial
  have hsoffs_bh : s_offs.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by rw [hs_offs]; simp [hibh]
  have hsoffs_c : s_offs.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by rw [hs_offs]; simp [hic]
  have hsoffs_k : s_offs.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by rw [hs_offs]; simp [hik]
  have hsoffs_o : s_offs.regs .nat [BK] "offs" = some (prepOffsTile BK) := by rw [hs_offs]; simp
  have hstepPq : stepStmt (Stmt.assign TileDType.ptr [BK] "p_q"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.ref TileDType.nat [BK] "offs")))) s_offs = some s_pq := by
    rw [hs_pq]; exact stepStmt_assign_eq_some (hptr Q s_offs hsoffs_bh hsoffs_c hsoffs_k hsoffs_o)
  have hspq_bh : s_pq.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by rw [hs_pq]; simp [hsoffs_bh]
  have hspq_c : s_pq.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by rw [hs_pq]; simp [hsoffs_c]
  have hspq_k : s_pq.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by rw [hs_pq]; simp [hsoffs_k]
  have hspq_o : s_pq.regs .nat [BK] "offs" = some (prepOffsTile BK) := by rw [hs_pq]; simp [hsoffs_o]
  have hstepPg : stepStmt (Stmt.assign TileDType.ptr [BK] "p_g"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase G)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.ref TileDType.nat [BK] "offs")))) s_pq = some s_pg := by
    rw [hs_pg]; exact stepStmt_assign_eq_some (hptr G s_pq hspq_bh hspq_c hspq_k hspq_o)
  have hspg_bh : s_pg.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by rw [hs_pg]; simp [hspq_bh]
  have hspg_c : s_pg.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by rw [hs_pg]; simp [hspq_c]
  have hspg_k : s_pg.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by rw [hs_pg]; simp [hspq_k]
  have hspg_o : s_pg.regs .nat [BK] "offs" = some (prepOffsTile BK) := by rw [hs_pg]; simp [hspq_o]
  have hstepPk : stepStmt (Stmt.assign TileDType.ptr [BK] "p_k"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.ref TileDType.nat [BK] "offs")))) s_pg = some s_pk := by
    rw [hs_pk]; exact stepStmt_assign_eq_some (hptr K s_pg hspg_bh hspg_c hspg_k hspg_o)
  have hspk_bh : s_pk.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by rw [hs_pk]; simp [hspg_bh]
  have hspk_c : s_pk.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by rw [hs_pk]; simp [hspg_c]
  have hspk_k : s_pk.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by rw [hs_pk]; simp [hspg_k]
  have hspk_o : s_pk.regs .nat [BK] "offs" = some (prepOffsTile BK) := by rw [hs_pk]; simp [hspg_o]
  have hstepPqg : stepStmt (Stmt.assign TileDType.ptr [BK] "p_qg"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase QG)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.ref TileDType.nat [BK] "offs")))) s_pk = some s_pqg := by
    rw [hs_pqg]; exact stepStmt_assign_eq_some (hptr QG s_pk hspk_bh hspk_c hspk_k hspk_o)
  have hspqg_bh : s_pqg.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by rw [hs_pqg]; simp [hspk_bh]
  have hspqg_c : s_pqg.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by rw [hs_pqg]; simp [hspk_c]
  have hspqg_k : s_pqg.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by rw [hs_pqg]; simp [hspk_k]
  have hspqg_o : s_pqg.regs .nat [BK] "offs" = some (prepOffsTile BK) := by rw [hs_pqg]; simp [hspk_o]
  have hstepPkg : stepStmt (Stmt.assign TileDType.ptr [BK] "p_kg"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase KG)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.ref TileDType.nat [BK] "offs")))) s_pqg = some s_pkg := by
    rw [hs_pkg]; exact stepStmt_assign_eq_some (hptr KG s_pqg hspqg_bh hspqg_c hspqg_k hspqg_o)
  have hspkg_k : s_pkg.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by rw [hs_pkg]; simp [hspqg_k]
  have hspkg_o : s_pkg.regs .nat [BK] "offs" = some (prepOffsTile BK) := by rw [hs_pkg]; simp [hspqg_o]
  
  have hmaskeval : evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK))
        (Op.ref TileDType.nat [BK] "offs")) (Op.constNat DK)) s_pkg = some (fwdMaskTile s DK BK) := by
    simp only [evalOp, Option.bind, hspkg_k, hspkg_o, evalOp_constNat,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
      ComparableDType.lt, Tile.bop, Tile.cop, Tile.scalar]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [fwdMaskTile, active, elemIndex, prepOffsTile, Tile.vec,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
    rfl
  have hstepMask : stepStmt (Stmt.assign TileDType.bool [BK] "mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK))
          (Op.ref TileDType.nat [BK] "offs")) (Op.constNat DK))) s_pkg = some s_mask := by
    rw [hs_mask]; exact stepStmt_assign_eq_some hmaskeval
  have hsmask_bh : s_mask.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by
    rw [hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq]; simp [hsoffs_bh]
  have hsmask_c : s_mask.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by
    rw [hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq]; simp [hsoffs_c]
  have hsmask_k : s_mask.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by
    rw [hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq]; simp [hsoffs_k]
  have hsmask_o : s_mask.regs .nat [BK] "offs" = some (prepOffsTile BK) := by
    rw [hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq]; simp [hsoffs_o]
  
  have hldeval : evalOp (Op.load TileDType.real
      (MemAccess.region G
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.sub NumericDType.nat Broadcast.nil
                  (Op.add NumericDType.nat Broadcast.nil
                    (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                    (Op.constNat BT))
                  (Op.constNat 1))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.ref TileDType.nat [BK] "offs")))
      MaskOpt.none) s_mask = some (prepLastDecayTile s G s_qk_h DK BT BK) := by
    simp only [evalOp, Option.bind, hsmask_bh, hsmask_c, hsmask_k, hsmask_o, evalOp_constNat,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul, NumericDType.sub,
      Tile.bop, Tile.scalar]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [prepLastDecayTile, prepOffsTile, Tile.vec,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
    rw [BlockState.readMemValue_real]
    simp only [offset, baseOffset]
    rw [show s.pids 1 * BT + BT - 1 = s.pids 1 * BT + (BT - 1) by omega]
    simp only [if_true]
    rw [hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
    simp only [BlockState.setReg_readMem]
    rfl
  have hstepLd : stepStmt (Stmt.assign TileDType.real [BK] "last_decay"
      (Op.load TileDType.real
        (MemAccess.region G
          (Op.add NumericDType.nat Broadcast.scalarL
            (Op.add NumericDType.nat Broadcast.nil
              (Op.add NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
                (Op.mul NumericDType.nat Broadcast.nil
                  (Op.sub NumericDType.nat Broadcast.nil
                    (Op.add NumericDType.nat Broadcast.nil
                      (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                      (Op.constNat BT))
                    (Op.constNat 1))
                  (Op.constNat DK)))
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
            (Op.ref TileDType.nat [BK] "offs")))
        MaskOpt.none)) s_mask = some s11 := by
    rw [hs11]; exact stepStmt_assign_eq_some hldeval
  
  rw [stepStmts.cons_some hstepK, stepStmts.cons_some hstepC, stepStmts.cons_some hstepB,
      stepStmts.cons_some hstepOffs, stepStmts.cons_some hstepPq, stepStmts.cons_some hstepPg,
      stepStmts.cons_some hstepPk, stepStmts.cons_some hstepPqg, stepStmts.cons_some hstepPkg]
  erw [stepStmts.cons_some hstepMask]
  erw [stepStmts.cons_some hstepLd]
  rw [show ∀ (X : Stmt), stepStmts [X] s11 = stepStmt X s11 from
      fun X => by cases h : stepStmt X s11 <;> simp [stepStmts, h]]
  
  have hP0 : prepInv Q K G QG KG s s_qk_h DK BT BK scale 0 s11 := by
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
      simp only [BlockState.setReg_pids]
    · intro a; rw [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
      simp only [BlockState.setReg_readMem]
    · intro a; rw [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
      simp only [BlockState.setReg_readMem]
    · intro a; rw [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
      simp only [BlockState.setReg_readMem]
    · 
      simp only [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq,
        BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, reduceCtorEq,
        not_false_eq_true, String.reduceEq]
    · 
      simp only [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg,
        BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, reduceCtorEq,
        not_false_eq_true, String.reduceEq]
    · 
      simp only [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk,
        BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, reduceCtorEq,
        not_false_eq_true, String.reduceEq]
    · 
      simp only [hs11, hs_mask, hs_pkg, hs_pqg,
        BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, reduceCtorEq,
        not_false_eq_true, String.reduceEq]
    · 
      simp only [hs11, hs_mask, hs_pkg,
        BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, reduceCtorEq,
        not_false_eq_true, String.reduceEq]
    · 
      simp only [hs11, hs_mask,
        BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, reduceCtorEq,
        not_false_eq_true, String.reduceEq]
    · 
      rw [hs11, BlockState.setReg_same]
    · intro j hj i; rw [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
      simp only [BlockState.setReg_readMem]
    · intro j hj i; rw [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
      simp only [BlockState.setReg_readMem]
    · intro j hj i hjBT; omega
    · intro j hj i hjBT; omega
  
  obtain ⟨final, sfinal, hLoop, hfinal, hPfinal⟩ :=
    forRange_inv (idx := "i") (start := 0) (stop := BT) (step := 1)
      (body := prepBody Q K G QG KG s_qk_h DK BT BK scale)
      (P := prepInv Q K G QG KG s s_qk_h DK BT BK scale) (s_init := s11) (Nat.one_ne_zero) hP0
      (fun j st hj hPj => prepare_qg_kg_step Q K G QG KG s s_qk_h DK BT BK scale
        hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK hBT j hj st hPj)
  refine ⟨final, sfinal, ?_, hfinal, hPfinal⟩
  simp only [prepBody] at hLoop
  rw [hLoop]

set_option maxHeartbeats 4000000 in

theorem prepare_qg_kg_full_surface_qg_closed_general
    (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) (s : BlockState) (t_rel : Fin BT) (i : Fin BK)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hK_QG : K ≠ QG) (hK_KG : K ≠ KG)
    (hG_QG : G ≠ QG) (hG_KG : G ≠ KG) (hQG_KG : QG ≠ KG) (hBK : BK ≤ DK) :
    (exec (prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale) s).map
      (·.readMem QG (offset s s_qk_h DK t_rel.val BT BK i))
      = some (if active s DK BK i then prepareQgClosed s Q G s_qk_h DK BT BK t_rel i scale
              else s.readMem QG (offset s s_qk_h DK t_rel.val BT BK i)) := by
  have hBT : 0 < BT := Fin.pos t_rel
  obtain ⟨final, sfinal, hExec, hfinal, hPfinal⟩ :=
    prepare_qg_kg_loop_drive Q K G QG KG s s_qk_h DK BT BK scale
      hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK hBT
  rw [hExec]
  simp only [Option.map_some]
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, hreadQG, _⟩ := hPfinal
  have ht : t_rel.val < final := lt_of_lt_of_le t_rel.isLt hfinal
  rw [hreadQG t_rel.val ht i t_rel.isLt]

set_option maxHeartbeats 4000000 in

theorem prepare_qg_kg_full_surface_kg_closed_general
    (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) (s : BlockState) (t_rel : Fin BT) (i : Fin BK)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hK_QG : K ≠ QG) (hK_KG : K ≠ KG)
    (hG_QG : G ≠ QG) (hG_KG : G ≠ KG) (hQG_KG : QG ≠ KG) (hBK : BK ≤ DK) :
    (exec (prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale) s).map
      (·.readMem KG (offset s s_qk_h DK t_rel.val BT BK i))
      = some (if active s DK BK i then prepareKgClosed s K G s_qk_h DK BT BK t_rel i
              else s.readMem KG (offset s s_qk_h DK t_rel.val BT BK i)) := by
  have hBT : 0 < BT := Fin.pos t_rel
  obtain ⟨final, sfinal, hExec, hfinal, hPfinal⟩ :=
    prepare_qg_kg_loop_drive Q K G QG KG s s_qk_h DK BT BK scale
      hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK hBT
  rw [hExec]
  simp only [Option.map_some]
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, hreadKG⟩ := hPfinal
  have ht : t_rel.val < final := lt_of_lt_of_le t_rel.isLt hfinal
  rw [hreadKG t_rel.val ht i t_rel.isLt]

set_option maxHeartbeats 4000000 in

theorem prepare_qg_kg_surface_qg_closed_compute_correct_general
    (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) (s : BlockState) (t_rel : Fin BT)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hK_QG : K ≠ QG) (hK_KG : K ≠ KG)
    (hG_QG : G ≠ QG) (hG_KG : G ≠ KG) (hQG_KG : QG ≠ KG) (hBK : BK ≤ DK) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s DK BK)
        (fun i => (QG, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        prepareQgClosed s Q G s_qk_h DK BT BK t_rel i scale) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [prepare_qg_kg_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := prepare_qg_kg_full_surface_qg_closed_general Q K G QG KG
    s_qk_h DK BT BK scale s t_rel i
    hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK
  rw [hExec] at h
  simp only [Option.map_some] at h
  rw [Option.some.injEq] at h
  show s'.readMem QG (offset s s_qk_h DK t_rel.val BT BK i) = _
  rw [h, if_pos hActive]

set_option maxHeartbeats 4000000 in

theorem prepare_qg_kg_surface_kg_closed_compute_correct_general
    (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) (s : BlockState) (t_rel : Fin BT)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hK_QG : K ≠ QG) (hK_KG : K ≠ KG)
    (hG_QG : G ≠ QG) (hG_KG : G ≠ KG) (hQG_KG : QG ≠ KG) (hBK : BK ≤ DK) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s DK BK)
        (fun i => (KG, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        prepareKgClosed s K G s_qk_h DK BT BK t_rel i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [prepare_qg_kg_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := prepare_qg_kg_full_surface_kg_closed_general Q K G QG KG
    s_qk_h DK BT BK scale s t_rel i
    hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK
  rw [hExec] at h
  simp only [Option.map_some] at h
  rw [Option.some.injEq] at h
  show s'.readMem KG (offset s s_qk_h DK t_rel.val BT BK i) = _
  rw [h, if_pos hActive]

noncomputable def bwdDQInterClosed
    (s : BlockState) (DQInner DQInter G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) : ℝ :=
  s.readMem DQInner (offset s s_qk_h DK t_rel.val BT BK i) +
    s.readMem DQInter (offset s s_qk_h DK t_rel.val BT BK i) *
      Real.exp (s.readMem G (offset s s_qk_h DK t_rel.val BT BK i) * Real.log 2)

noncomputable def bwdDKInterClosed
    (s : BlockState) (DKInner DKInter G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) : ℝ :=
  s.readMem DKInner (offset s s_qk_h DK t_rel.val BT BK i) +
    s.readMem DKInter (offset s s_qk_h DK t_rel.val BT BK i) *
      Real.exp ((s.readMem G (offset s s_qk_h DK (BT - 1) BT BK i) -
        s.readMem G (offset s s_qk_h DK t_rel.val BT BK i)) * Real.log 2)

noncomputable def bwdDGSummand
    (s : BlockState) (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s_qk_h DK BT BK : Nat) (j : Fin BT) (i : Fin BK) : ℝ :=
  bwdDQInterClosed s DQInner DQInter G s_qk_h DK BT BK j i *
      s.readMem Q (offset s s_qk_h DK j.val BT BK i) -
    bwdDKInterClosed s DKInner DKInter G s_qk_h DK BT BK j i *
      s.readMem K (offset s s_qk_h DK j.val BT BK i)

noncomputable def bwdDGClosed
    (s : BlockState) (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) : ℝ :=
  ∑ d : Fin (BT - t_rel.val),
    bwdDGSummand s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK
      ⟨t_rel.val + d.val, by omega⟩ i

section BwdRecipes

variable {BK : Nat} (s : BlockState)

theorem bwdEval_assign_subNat (name : RegName) (a b : Op .nat []) (va vb : Nat)
    (ha : evalOp a s = some (Tile.scalar va))
    (hb : evalOp b s = some (Tile.scalar vb)) :
    stepStmt (.assign .nat [] name (.sub NumericDType.nat Broadcast.nil a b)) s
      = some (s.setReg name .nat [] (Tile.scalar (va - vb))) := by
  simp [stepStmt, evalOp, ha, hb]
  rfl

theorem bwdEval_assign_load_ptr_maskOther (name pname : RegName)
    (mask : Op .bool [BK]) (other : Op .real [BK])
    (ptrT : Tile .ptr [BK]) (maskT : Tile .bool [BK]) (otherT : Tile .real [BK])
    (hptr : s.regs .ptr [BK] pname = some ptrT)
    (hmask : evalOp mask s = some maskT)
    (hother : evalOp other s = some otherT) :
    stepStmt (.assign .real [BK] name
        (.load .real (MemAccess.ptr (Op.ref .ptr [BK] pname))
          (MaskOpt.maskOther mask other))) s
      = some (s.setReg name .real [BK]
          ⟨fun i => if maskT.data i then
                some (s.readMem (ptrT.data i).1 (ptrT.data i).2)
              else otherT.data i⟩) := by
  simp [stepStmt, evalOp, hptr, hmask, hother]

theorem bwdEval_store_ptr_masked (pname : RegName)
    (val : Op .real [BK]) (mask : Op .bool [BK])
    (ptrT : Tile .ptr [BK]) (valT : Tile .real [BK]) (maskT : Tile .bool [BK])
    (hptr : s.regs .ptr [BK] pname = some ptrT)
    (hval : evalOp val s = some valT)
    (hmask : evalOp mask s = some maskT) :
    stepStmt (.store .real [BK] (MemAccess.ptr (Op.ref .ptr [BK] pname))
        val (MaskOpt.mask mask)) s
      = some ((TileShape.allIndices [BK]).foldl
          (fun acc i =>
            if maskT.data i then
              acc.writeMemTyped .real (ptrT.data i).1 (ptrT.data i).2 (valT.data i)
            else acc) s) := by
  simp [stepStmt, evalOp, hptr, hval, hmask]

theorem bwdEval_assign_ptrSub (name pname : RegName) (d : Nat)
    (ptrT : Tile .ptr [BK])
    (hptr : s.regs .ptr [BK] pname = some ptrT) :
    stepStmt (.assign .ptr [BK] name
        (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] pname) (Op.constNat d))) s
      = some (s.setReg name .ptr [BK]
          ⟨fun i => ((ptrT.data i).1, (ptrT.data i).2 - d)⟩) := by
  simp [stepStmt, evalOp, hptr, Tile.ptrSub]

theorem bwdEval_eqNat (a b : Op .nat []) (av bv : Tile .nat [])
    (ha : evalOp a s = some av) (hb : evalOp b s = some bv) :
    evalOp (.eq ComparableDType.nat Broadcast.nil a b) s
      = some (Tile.cop ComparableDType.nat.eq Broadcast.nil av bv) := by
  simp [evalOp, ha, hb]

theorem bwdEval_ifThen_true (cond : Op .bool []) (body : List Stmt)
    (hc : evalOp cond s = some (Tile.scalar Bool.true)) :
    stepStmt (.ifThen cond body) s = stepStmts body s := by
  simp [stepStmt, hc]

theorem bwdEval_ifThen_false (cond : Op .bool []) (body : List Stmt)
    (hc : evalOp cond s = some (Tile.scalar Bool.false)) :
    stepStmt (.ifThen cond body) s = some s := by
  simp [stepStmt, hc]

end BwdRecipes

section BwdComputeRecipes

variable {BK : Nat} (s : BlockState)

theorem bwdEval_mul (a b : Op .real [BK]) (aT bT : Tile .real [BK])
    (ha : evalOp a s = some aT) (hb : evalOp b s = some bT) :
    evalOp (.mul NumericDType.real (Broadcast.consSame Broadcast.nil) a b) s
      = some (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) aT bT) := by
  simp [evalOp, ha, hb]

theorem bwdEval_add (a b : Op .real [BK]) (aT bT : Tile .real [BK])
    (ha : evalOp a s = some aT) (hb : evalOp b s = some bT) :
    evalOp (.add NumericDType.real (Broadcast.consSame Broadcast.nil) a b) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil) aT bT) := by
  simp [evalOp, ha, hb]

theorem bwdEval_sub (a b : Op .real [BK]) (aT bT : Tile .real [BK])
    (ha : evalOp a s = some aT) (hb : evalOp b s = some bT) :
    evalOp (.sub NumericDType.real (Broadcast.consSame Broadcast.nil) a b) s
      = some (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) aT bT) := by
  simp [evalOp, ha, hb]

theorem bwdEval_exp2 (x : Op .real [BK]) (xT : Tile .real [BK])
    (hx : evalOp x s = some xT) :
    evalOp (.exp2 x) s = some (Tile.uop WithBot.realExp2 xT) := by
  simp [evalOp, hx]

end BwdComputeRecipes

section BwdAssembly

private noncomputable def ldVal (s : BlockState) (region : RegionName) (R : Nat)
    (i : TileIndex [4]) : WithBot ℝ :=
  if decide (s.pids 0 * 4 + i.1.val < 8) then
    some (s.readMem region (s.pids 2 * 64 + s.pids 0 * 4 + i.1.val + R))
  else (0 : WithBot ℝ)

private noncomputable def bwdIterHeadState
    (G : RegionName) (sin : BlockState) (rt R : Nat) : BlockState :=
  (sin.setReg "j" .nat [] (Tile.scalar (2 - 1 - rt * 1))).setReg
    "_g" .real [4] ⟨fun i => ldVal sin G R i⟩

def bwdIterBodyG (DK BT BK : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "j"
      (Op.sub .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil (Op.constNat BT) (Op.constNat 1))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "__rev_j") (Op.constNat 1))),
    Stmt.assign .real [BK] "_g"
      (Op.load ComputeDType.fp32.eraseDType (MemAccess.ptr (Op.ref .ptr [BK] "p_g"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.ifThen
      (Op.eq ComparableDType.nat Broadcast.nil (Op.ref .nat [] "j")
        (Op.sub .nat Broadcast.nil (Op.constNat BT) (Op.constNat 1)))
      [Stmt.assign .real [BK] "last_g" (Op.ref .real [BK] "_g")],
    Stmt.assign .real [BK] "_dq1"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_dq_inner"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign .real [BK] "_dq2"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_dq_inter"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign .real [BK] "_dq2"
      (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "_dq2")
        (Op.ref .real [BK] "_g").exp2),
    Stmt.assign .real [BK] "_dq"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [BK] "_dq1")
        (Op.ref .real [BK] "_dq2")),
    Stmt.store .real [BK] (MemAccess.ptr (Op.ref .ptr [BK] "p_dq_inter"))
      (Op.ref .real [BK] "_dq") (MaskOpt.mask (Op.ref .bool [BK] "mask")),
    Stmt.assign .real [BK] "_dk1"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_dk_inner"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign .real [BK] "_dk2"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_dk_inter"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign .real [BK] "_dk2"
      (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "_dk2")
        (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BK] "last_g")
            (Op.ref .real [BK] "_g")).exp2),
    Stmt.assign .real [BK] "_dk"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [BK] "_dk1")
        (Op.ref .real [BK] "_dk2")),
    Stmt.store .real [BK] (MemAccess.ptr (Op.ref .ptr [BK] "p_dk_inter"))
      (Op.ref .real [BK] "_dk") (MaskOpt.mask (Op.ref .bool [BK] "mask")),
    Stmt.assign .real [BK] "_q"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_q"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign .real [BK] "_k"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_k"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign .real [BK] "_dg"
      (Op.sub .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "_dq")
          (Op.ref .real [BK] "_q"))
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "_dk")
          (Op.ref .real [BK] "_k"))),
    Stmt.assign .real [BK] "cum_grad_dg"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [BK] "cum_grad_dg")
        (Op.ref .real [BK] "_dg")),
    Stmt.store .real [BK] (MemAccess.ptr (Op.ref .ptr [BK] "p_dg"))
      (Op.ref .real [BK] "cum_grad_dg") (MaskOpt.mask (Op.ref .bool [BK] "mask")),
    Stmt.assign .ptr [BK] "p_g"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_g") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_k"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_k") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_q"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_q") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_dq_inner"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dq_inner") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_dk_inner"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dk_inner") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_dq_inter"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dq_inter") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_dk_inter"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dk_inter") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_dg"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dg") (Op.constNat DK)) ]

def bwdPtrTileG (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) : Tile .ptr [BK] :=
  ⟨fun idx => (region, offset s s_qk_h DK r BT BK idx.1)⟩

def bwdPtrDecTileG (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) : Tile .ptr [BK] :=
  ⟨fun idx => (region, offset s s_qk_h DK (BT - 1) BT BK idx.1 - m * DK)⟩

theorem bwdPtrDecTileG_zero (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) :
    bwdPtrDecTileG s region s_qk_h DK BT BK 0 = bwdPtrTileG s region s_qk_h DK BT BK (BT - 1) := by
  apply Tile.ext; intro idx
  simp only [bwdPtrDecTileG, bwdPtrTileG, Nat.zero_mul, Nat.sub_zero]

theorem bwdPtrDecTileG_row (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK m : Nat) (hm : m < BT) :
    bwdPtrDecTileG s region s_qk_h DK BT BK m = bwdPtrTileG s region s_qk_h DK BT BK (BT - 1 - m) := by
  apply Tile.ext; intro idx
  simp only [bwdPtrDecTileG, bwdPtrTileG]
  refine Prod.ext rfl ?_
  simp only [offset, baseOffset]
  have key : (s.pids 1 * BT + (BT - 1)) * DK
      = (s.pids 1 * BT + (BT - 1 - m)) * DK + m * DK := by
    have hsplit : s.pids 1 * BT + (BT - 1) = (s.pids 1 * BT + (BT - 1 - m)) + m := by omega
    rw [hsplit, Nat.add_mul]
  rw [key]; omega

theorem bwdPtrTileG_dec_succ (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK m : Nat) (hm : m < BT) :
    (⟨fun idx => (region, offset s s_qk_h DK (BT - 1 - m) BT BK idx.1 - DK)⟩ : Tile .ptr [BK])
      = bwdPtrDecTileG s region s_qk_h DK BT BK (m + 1) := by
  apply Tile.ext; intro idx
  simp only [bwdPtrDecTileG]
  refine Prod.ext rfl ?_
  simp only [offset, baseOffset]
  have key : (s.pids 1 * BT + (BT - 1)) * DK
      = (s.pids 1 * BT + (BT - 1 - m)) * DK + m * DK := by
    have hsplit : s.pids 1 * BT + (BT - 1) = (s.pids 1 * BT + (BT - 1 - m)) + m := by omega
    rw [hsplit, Nat.add_mul]
  rw [key]
  have hkey2 : (m + 1) * DK = m * DK + DK := by ring
  rw [hkey2]; omega

def bwdMaskTileG (s : BlockState) (DK BK : Nat) : Tile .bool [BK] :=
  ⟨fun idx => decide (active s DK BK idx.1)⟩

noncomputable def bwdGatedG (s : BlockState) (DK BK : Nat) (f : Fin BK → ℝ) :
    Tile .real [BK] :=
  ⟨fun idx => if active s DK BK idx.1 then some (f idx.1) else some 0⟩

noncomputable def bwdDqOutG (s : BlockState) (DQInner DQInter G : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (i : Fin BK) : ℝ :=
  s.readMem DQInner (offset s s_qk_h DK r BT BK i) +
    s.readMem DQInter (offset s s_qk_h DK r BT BK i) *
      Real.exp (s.readMem G (offset s s_qk_h DK r BT BK i) * Real.log 2)

noncomputable def bwdDkOutG (s : BlockState) (DKInner DKInter G : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (lg : Fin BK → ℝ) (i : Fin BK) : ℝ :=
  s.readMem DKInner (offset s s_qk_h DK r BT BK i) +
    s.readMem DKInter (offset s s_qk_h DK r BT BK i) *
      Real.exp ((lg i - s.readMem G (offset s s_qk_h DK r BT BK i)) * Real.log 2)

noncomputable def bwdDgSumG (s : BlockState)
    (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (lg : Fin BK → ℝ) (i : Fin BK) : ℝ :=
  bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r i *
      s.readMem Q (offset s s_qk_h DK r BT BK i) -
    bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lg i *
      s.readMem K (offset s s_qk_h DK r BT BK i)

noncomputable def bwdCumPartialG (s : BlockState)
    (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) (i : Fin BK) : ℝ :=
  ∑ d : Fin m,
    bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK
      (BT - 1 - d.val)
      (fun i' => s.readMem G (offset s s_qk_h DK (BT - 1) BT BK i'))
      i

structure BwdPrologueShapeG (s s0 : BlockState)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) : Prop where
  pid : s0.pids = s.pids
  mem : s0.mem = s.mem
  mask : s0.regs .bool [BK] "mask" = some (bwdMaskTileG s DK BK)
  cum : s0.regs .real [BK] "cum_grad_dg" = some (⟨fun _ => some 0⟩ : Tile .real [BK])
  lastg : s0.regs .real [BK] "last_g" = some (⟨fun _ => some 0⟩ : Tile .real [BK])
  pg : s0.regs .ptr [BK] "p_g" = some (bwdPtrTileG s G s_qk_h DK BT BK (BT - 1))
  pk : s0.regs .ptr [BK] "p_k" = some (bwdPtrTileG s K s_qk_h DK BT BK (BT - 1))
  pq : s0.regs .ptr [BK] "p_q" = some (bwdPtrTileG s Q s_qk_h DK BT BK (BT - 1))
  pdqi : s0.regs .ptr [BK] "p_dq_inner" = some (bwdPtrTileG s DQInner s_qk_h DK BT BK (BT - 1))
  pdki : s0.regs .ptr [BK] "p_dk_inner" = some (bwdPtrTileG s DKInner s_qk_h DK BT BK (BT - 1))
  pdqt : s0.regs .ptr [BK] "p_dq_inter" = some (bwdPtrTileG s DQInter s_qk_h DK BT BK (BT - 1))
  pdkt : s0.regs .ptr [BK] "p_dk_inter" = some (bwdPtrTileG s DKInter s_qk_h DK BT BK (BT - 1))
  pdg : s0.regs .ptr [BK] "p_dg" = some (bwdPtrTileG s DG s_qk_h DK BT BK (BT - 1))

noncomputable def bwdLastgG (s : BlockState) (G : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) (i : Fin BK) : ℝ :=
  if 0 < m then s.readMem G (offset s s_qk_h DK (BT - 1) BT BK i) else 0

noncomputable def bwdInvG
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s : BlockState) (s_qk_h DK BT BK : Nat) :
    Nat → BlockState → Prop :=
  fun m sc =>
    sc.pids = s.pids ∧
    (∀ a, sc.readMem G a = s.readMem G a) ∧
    (∀ a, sc.readMem Q a = s.readMem Q a) ∧
    (∀ a, sc.readMem K a = s.readMem K a) ∧
    (∀ a, sc.readMem DQInner a = s.readMem DQInner a) ∧
    (∀ a, sc.readMem DKInner a = s.readMem DKInner a) ∧
    sc.regs .bool [BK] "mask" = some (bwdMaskTileG s DK BK) ∧
    sc.regs .real [BK] "cum_grad_dg" = some
      (bwdGatedG s DK BK (bwdCumPartialG s DQInner DQInter DKInner DKInter Q K G
        s_qk_h DK BT BK m)) ∧
    sc.regs .real [BK] "last_g" = some
      (bwdGatedG s DK BK (bwdLastgG s G s_qk_h DK BT BK m)) ∧
    sc.regs .ptr [BK] "p_g" = some (bwdPtrDecTileG s G s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_k" = some (bwdPtrDecTileG s K s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_q" = some (bwdPtrDecTileG s Q s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_dq_inner" = some (bwdPtrDecTileG s DQInner s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_dk_inner" = some (bwdPtrDecTileG s DKInner s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_dq_inter" = some (bwdPtrDecTileG s DQInter s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_dk_inter" = some (bwdPtrDecTileG s DKInter s_qk_h DK BT BK m) ∧
    sc.regs .ptr [BK] "p_dg" = some (bwdPtrDecTileG s DG s_qk_h DK BT BK m) ∧

    (∀ r, r < BT - m → ∀ i : Fin BK,
      sc.readMem DQInter (offset s s_qk_h DK r BT BK i)
        = s.readMem DQInter (offset s s_qk_h DK r BT BK i)) ∧
    (∀ r, r < BT - m → ∀ i : Fin BK,
      sc.readMem DKInter (offset s s_qk_h DK r BT BK i)
        = s.readMem DKInter (offset s s_qk_h DK r BT BK i)) ∧
    (∀ r, r < BT - m → ∀ i : Fin BK,
      sc.readMem DG (offset s s_qk_h DK r BT BK i)
        = s.readMem DG (offset s s_qk_h DK r BT BK i)) ∧
    
    (∀ r, BT - m ≤ r → r < BT → ∀ i : Fin BK,
      sc.readMem DQInter (offset s s_qk_h DK r BT BK i)
        = if active s DK BK i then
            bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r i
          else s.readMem DQInter (offset s s_qk_h DK r BT BK i)) ∧
    (∀ r, BT - m ≤ r → r < BT → ∀ i : Fin BK,
      sc.readMem DKInter (offset s s_qk_h DK r BT BK i)
        = if active s DK BK i then
            bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r
              (fun i' => s.readMem G (offset s s_qk_h DK (BT - 1) BT BK i')) i
          else s.readMem DKInter (offset s s_qk_h DK r BT BK i)) ∧
    (∀ r, BT - m ≤ r → r < BT → ∀ i : Fin BK,
      sc.readMem DG (offset s s_qk_h DK r BT BK i)
        = if active s DK BK i then
            bwdCumPartialG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK (BT - r) i
          else s.readMem DG (offset s s_qk_h DK r BT BK i))

theorem bwd_body_decomp_general
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) :
    (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
        s_qk_h DK BT BK).toAlgKernel.body
      = (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
          s_qk_h DK BT BK).toAlgKernel.body.take 15
        ++ [Stmt.forRangeDyn "__rev_j" (Op.constNat 0)
              (Op.add .nat Broadcast.nil
                (Op.div .nat Broadcast.nil
                  (Op.sub .nat Broadcast.nil (Op.constNat BT) (Op.constNat 1)) (Op.constNat 1))
                (Op.constNat 1))
              (Op.constNat 1) (bwdIterBodyG DK BT BK)] := by
  rfl

def bwdPrologueG (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) : List Stmt :=
  let pgen : RegName → (RegionName) → Stmt := fun nm reg =>
    Stmt.assign .ptr [BK] nm
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase reg)
        (Op.add .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK)))
            (Op.ref .nat [BK] "offs"))
          (Op.mul .nat Broadcast.nil
            (Op.sub .nat Broadcast.nil
              (Op.add .nat Broadcast.nil
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BT)) (Op.constNat BT))
              (Op.constNat 1))
            (Op.constNat DK))))
  [ Stmt.assign .nat [] "i_k" (Op.programId 0),
    Stmt.assign .nat [] "i_c" (Op.programId 1),
    Stmt.assign .nat [] "i_bh" (Op.programId 2),
    Stmt.assign .nat [BK] "offs" (Op.arange BK),
    pgen "p_q" Q, pgen "p_k" K, pgen "p_g" G, pgen "p_dg" DG,
    pgen "p_dq_inner" DQInner, pgen "p_dk_inner" DKInner,
    pgen "p_dq_inter" DQInter, pgen "p_dk_inter" DKInter,
    Stmt.assign .real [BK] "cum_grad_dg" (Op.full [BK] (Op.const 0)),
    Stmt.assign .bool [BK] "mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK))
          (Op.ref .nat [BK] "offs"))
        (Op.constNat DK)),
    Stmt.assign .real [BK] "last_g" (Op.full [BK] (Op.const 0)) ]

theorem bwd_prologue_take_general
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) :
    ((bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
        s_qk_h DK BT BK).toAlgKernel.body.take 15)
      = bwdPrologueG DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK := by
  rfl

set_option maxHeartbeats 2000000 in

theorem bwd_prologue_eval_general
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) (hBT : 0 < BT) (s : BlockState) :
    ∃ s0,
      stepStmts ((bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK).toAlgKernel.body.take 15) s = some s0
      ∧ BwdPrologueShapeG s s0 DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK := by
  rw [bwd_prologue_take_general]
  
  have hptr : ∀ (reg : RegionName) (c : BlockState),
      c.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) →
      c.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) →
      c.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) →
      c.regs .nat [BK] "offs" = some (Tile.vec (fun i : Fin BK => i.val)) →
      evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase reg)
        (Op.add .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK)))
            (Op.ref .nat [BK] "offs"))
          (Op.mul .nat Broadcast.nil
            (Op.sub .nat Broadcast.nil
              (Op.add .nat Broadcast.nil
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_c") (Op.constNat BT)) (Op.constNat BT))
              (Op.constNat 1))
            (Op.constNat DK)))) c
        = some (bwdPtrTileG s reg s_qk_h DK BT BK (BT - 1)) := by
    intro reg c hik hic hibh hoffs
    simp only [evalOp, hik, hic, hibh, hoffs, Option.bind, evalOp_constNat, evalOp_ref,
      Tile.bop, Tile.ptrAdd, NumericDType.add, NumericDType.mul, NumericDType.sub]
    apply congrArg some; apply Tile.ext; intro idx; obtain ⟨i, u⟩ := idx
    simp only [bwdPtrTileG, offset, baseOffset, Tile.vec, Tile.scalar,
      Broadcast.leftIndex, Broadcast.rightIndex,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR]
    refine Prod.ext rfl ?_
    simp only []
    have hbt1 : s.pids 1 * BT + BT - 1 = s.pids 1 * BT + (BT - 1) := by omega
    rw [hbt1]; ring
  
  have hmaskE : ∀ c : BlockState,
      c.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) →
      c.regs .nat [BK] "offs" = some (Tile.vec (fun i : Fin BK => i.val)) →
      evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK))
          (Op.ref .nat [BK] "offs"))
        (Op.constNat DK)) c
        = some (bwdMaskTileG s DK BK) := by
    intro c hik hoffs
    simp only [evalOp, hik, hoffs, Option.bind, evalOp_constNat, evalOp_ref,
      Tile.bop, NumericDType.mul, ComparableDType.lt]
    apply congrArg some; apply Tile.ext; intro idx; obtain ⟨i, u⟩ := idx
    simp only [bwdMaskTileG, active, elemIndex, Tile.cop_data, Tile.vec, Tile.scalar,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL, ComparableDType.lt]
    rfl
  
  have hfullE : ∀ c : BlockState,
      evalOp (Op.full [BK] (Op.const (0:ℝ))) c
        = some (⟨fun _ => some 0⟩ : Tile .real [BK]) := by
    intro c; simp [evalOp]
  
  set c1 := s.setReg "i_k" .nat [] (Tile.scalar (s.pids 0)) with hc1
  set c2 := c1.setReg "i_c" .nat [] (Tile.scalar (s.pids 1)) with hc2
  set c3 := c2.setReg "i_bh" .nat [] (Tile.scalar (s.pids 2)) with hc3
  set c4 := c3.setReg "offs" .nat [BK] (Tile.vec (fun i : Fin BK => i.val)) with hc4
  
  have hpidc4 : c4.pids = s.pids := by
    rw [hc4, hc3, hc2, hc1]; simp only [BlockState.setReg_pids]

  have hik4 : c4.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by
    rw [hc4, BlockState.setReg_ne_name (h := by decide), hc3,
      BlockState.setReg_ne_name (h := by decide), hc2,
      BlockState.setReg_ne_name (h := by decide), hc1]
    exact BlockState.setReg_same _ _ _ _ _
  have hic4 : c4.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by
    rw [hc4, BlockState.setReg_ne_name (h := by decide), hc3,
      BlockState.setReg_ne_name (h := by decide), hc2]
    exact BlockState.setReg_same _ _ _ _ _
  have hibh4 : c4.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by
    rw [hc4, BlockState.setReg_ne_name (h := by decide), hc3]
    exact BlockState.setReg_same _ _ _ _ _
  have hoffs4 : c4.regs .nat [BK] "offs" = some (Tile.vec (fun i : Fin BK => i.val)) := by
    rw [hc4]; exact BlockState.setReg_same _ _ _ _ _
  
  set c5 := c4.setReg "p_q" .ptr [BK] (bwdPtrTileG s Q s_qk_h DK BT BK (BT - 1)) with hc5
  set c6 := c5.setReg "p_k" .ptr [BK] (bwdPtrTileG s K s_qk_h DK BT BK (BT - 1)) with hc6
  set c7 := c6.setReg "p_g" .ptr [BK] (bwdPtrTileG s G s_qk_h DK BT BK (BT - 1)) with hc7
  set c8 := c7.setReg "p_dg" .ptr [BK] (bwdPtrTileG s DG s_qk_h DK BT BK (BT - 1)) with hc8
  set c9 := c8.setReg "p_dq_inner" .ptr [BK] (bwdPtrTileG s DQInner s_qk_h DK BT BK (BT - 1)) with hc9
  set c10 := c9.setReg "p_dk_inner" .ptr [BK] (bwdPtrTileG s DKInner s_qk_h DK BT BK (BT - 1)) with hc10
  set c11 := c10.setReg "p_dq_inter" .ptr [BK] (bwdPtrTileG s DQInter s_qk_h DK BT BK (BT - 1)) with hc11
  set c12 := c11.setReg "p_dk_inter" .ptr [BK] (bwdPtrTileG s DKInter s_qk_h DK BT BK (BT - 1)) with hc12
  set c13 := c12.setReg "cum_grad_dg" .real [BK] (⟨fun _ => some 0⟩ : Tile .real [BK]) with hc13
  set c14 := c13.setReg "mask" .bool [BK] (bwdMaskTileG s DK BK) with hc14
  set c15 := c14.setReg "last_g" .real [BK] (⟨fun _ => some 0⟩ : Tile .real [BK]) with hc15

  have hbase : ∀ (c : BlockState) (nm : RegName) (t : Tile .ptr [BK]),
      nm ≠ "i_k" → nm ≠ "i_c" → nm ≠ "i_bh" → nm ≠ "offs" →
      c.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) →
      c.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) →
      c.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) →
      c.regs .nat [BK] "offs" = some (Tile.vec (fun i : Fin BK => i.val)) →
      (c.setReg nm .ptr [BK] t).regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) ∧
      (c.setReg nm .ptr [BK] t).regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) ∧
      (c.setReg nm .ptr [BK] t).regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) ∧
      (c.setReg nm .ptr [BK] t).regs .nat [BK] "offs" = some (Tile.vec (fun i : Fin BK => i.val)) := by
    intro c nm t n1 n2 n3 n4 h1 h2 h3 h4
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [BlockState.setReg_ne_name (h := fun h => n1 h.symm)]; exact h1
    · rw [BlockState.setReg_ne_name (h := fun h => n2 h.symm)]; exact h2
    · rw [BlockState.setReg_ne_name (h := fun h => n3 h.symm)]; exact h3
    · rw [BlockState.setReg_ne_name (h := fun h => n4 h.symm)]; exact h4
  obtain ⟨q1, q2, q3, q4⟩ := hbase c4 "p_q" _ (by decide) (by decide) (by decide) (by decide) hik4 hic4 hibh4 hoffs4
  obtain ⟨k1, k2, k3, k4⟩ := hbase c5 "p_k" _ (by decide) (by decide) (by decide) (by decide) q1 q2 q3 q4
  obtain ⟨g1, g2, g3, g4⟩ := hbase c6 "p_g" _ (by decide) (by decide) (by decide) (by decide) k1 k2 k3 k4
  obtain ⟨d1, d2, d3, d4⟩ := hbase c7 "p_dg" _ (by decide) (by decide) (by decide) (by decide) g1 g2 g3 g4
  obtain ⟨a1, a2, a3, a4⟩ := hbase c8 "p_dq_inner" _ (by decide) (by decide) (by decide) (by decide) d1 d2 d3 d4
  obtain ⟨b1, b2, b3, b4⟩ := hbase c9 "p_dk_inner" _ (by decide) (by decide) (by decide) (by decide) a1 a2 a3 a4
  obtain ⟨e1, e2, e3, e4⟩ := hbase c10 "p_dq_inter" _ (by decide) (by decide) (by decide) (by decide) b1 b2 b3 b4
  refine ⟨c15, ?_, ?_⟩
  · 
    simp only [bwdPrologueG]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.programId 1) c1 = some (Tile.scalar (s.pids 1)) by
          rw [evalOp_programId, hc1, BlockState.setReg_pids]))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.programId 2) c2 = some (Tile.scalar (s.pids 2)) by
          rw [evalOp_programId, hc2, hc1]; simp only [BlockState.setReg_pids]))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_arange BK c3))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hptr Q c4 hik4 hic4 hibh4 hoffs4))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hptr K c5 q1 q2 q3 q4))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hptr G c6 k1 k2 k3 k4))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hptr DG c7 g1 g2 g3 g4))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hptr DQInner c8 d1 d2 d3 d4))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hptr DKInner c9 a1 a2 a3 a4))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hptr DQInter c10 b1 b2 b3 b4))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hptr DKInter c11 e1 e2 e3 e4))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hfullE c12))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (hmaskE c13
          (by rw [hc13, BlockState.setReg_ne_name (h := by decide), hc12,
                BlockState.setReg_ne_name (h := by decide), hc11,
                BlockState.setReg_ne_name (h := by decide), hc10,
                BlockState.setReg_ne_name (h := by decide), hc9,
                BlockState.setReg_ne_name (h := by decide), hc8,
                BlockState.setReg_ne_name (h := by decide), hc7,
                BlockState.setReg_ne_name (h := by decide), hc6,
                BlockState.setReg_ne_name (h := by decide), hc5,
                BlockState.setReg_ne_name (h := by decide)]
              exact hik4)
          (by rw [hc13, BlockState.setReg_ne_name (h := by decide), hc12,
                BlockState.setReg_ne_name (h := by decide), hc11,
                BlockState.setReg_ne_name (h := by decide), hc10,
                BlockState.setReg_ne_name (h := by decide), hc9,
                BlockState.setReg_ne_name (h := by decide), hc8,
                BlockState.setReg_ne_name (h := by decide), hc7,
                BlockState.setReg_ne_name (h := by decide), hc6,
                BlockState.setReg_ne_name (h := by decide), hc5,
                BlockState.setReg_ne_name (h := by decide)]
              exact hoffs4)))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (hfullE c14))]
    rw [stepStmts.nil]
  · 
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · 
      rw [hc15, hc14, hc13]; simp only [BlockState.setReg_pids]
      rw [hc12, hc11, hc10, hc9, hc8, hc7, hc6, hc5]; simp only [BlockState.setReg_pids]
      exact hpidc4
    · 
      rw [hc15, hc14, hc13, hc12, hc11, hc10, hc9, hc8, hc7, hc6, hc5, hc4, hc3, hc2, hc1]
      rfl
    · 
      rw [hc15, BlockState.setReg_ne_name (h := by decide), hc14]
      exact BlockState.setReg_same _ _ _ _ _
    · 
      rw [hc15, BlockState.setReg_ne_name (h := by decide), hc14,
        BlockState.setReg_ne_name (h := by decide), hc13]
      exact BlockState.setReg_same _ _ _ _ _
    · 
      rw [hc15]; exact BlockState.setReg_same _ _ _ _ _
    all_goals (
      rw [hc15, BlockState.setReg_ne_name (h := by decide), hc14,
        BlockState.setReg_ne_name (h := by decide), hc13,
        BlockState.setReg_ne_name (h := by decide)])
    · 
      rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10,
        BlockState.setReg_ne_name (h := by decide), hc9,
        BlockState.setReg_ne_name (h := by decide), hc8,
        BlockState.setReg_ne_name (h := by decide), hc7]
      exact BlockState.setReg_same _ _ _ _ _
    · 
      rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10,
        BlockState.setReg_ne_name (h := by decide), hc9,
        BlockState.setReg_ne_name (h := by decide), hc8,
        BlockState.setReg_ne_name (h := by decide), hc7,
        BlockState.setReg_ne_name (h := by decide), hc6]
      exact BlockState.setReg_same _ _ _ _ _
    · 
      rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10,
        BlockState.setReg_ne_name (h := by decide), hc9,
        BlockState.setReg_ne_name (h := by decide), hc8,
        BlockState.setReg_ne_name (h := by decide), hc7,
        BlockState.setReg_ne_name (h := by decide), hc6,
        BlockState.setReg_ne_name (h := by decide), hc5]
      exact BlockState.setReg_same _ _ _ _ _
    · 
      rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10,
        BlockState.setReg_ne_name (h := by decide), hc9]
      exact BlockState.setReg_same _ _ _ _ _
    · 
      rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10]
      exact BlockState.setReg_same _ _ _ _ _
    · 
      rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11]
      exact BlockState.setReg_same _ _ _ _ _
    · 
      rw [hc12]; exact BlockState.setReg_same _ _ _ _ _
    · 
      rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10,
        BlockState.setReg_ne_name (h := by decide), hc9,
        BlockState.setReg_ne_name (h := by decide), hc8]
      exact BlockState.setReg_same _ _ _ _ _

theorem bwdCumPartialG_zero (s : BlockState)
    (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s_qk_h DK BT BK : Nat) (i : Fin BK) :
    bwdCumPartialG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK 0 i = 0 := by
  simp [bwdCumPartialG]

noncomputable def bwdLdRowG (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (i : Fin BK) : ℝ :=
  s.readMem region (offset s s_qk_h DK r BT BK i)

theorem bwdGatedG_add (s : BlockState) (DK BK : Nat) (a b : Fin BK → ℝ) :
    Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
        (bwdGatedG s DK BK a) (bwdGatedG s DK BK b)
      = bwdGatedG s DK BK (fun i => a i + b i) := by
  apply Tile.ext; intro idx
  simp only [bwdGatedG, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, WithBot.realAdd]
  by_cases ha : active s DK BK idx.1
  · simp only [ha, if_true, Option.map₂_some_some]
  · simp only [ha, if_false]
    show Option.map₂ (·+·) (some (0:ℝ)) (some 0) = some 0
    rw [Option.map₂_some_some]; norm_num

theorem bwdGatedG_sub (s : BlockState) (DK BK : Nat) (a b : Fin BK → ℝ) :
    Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil)
        (bwdGatedG s DK BK a) (bwdGatedG s DK BK b)
      = bwdGatedG s DK BK (fun i => a i - b i) := by
  apply Tile.ext; intro idx
  simp only [bwdGatedG, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.sub, WithBot.realSub]
  by_cases ha : active s DK BK idx.1
  · simp only [ha, if_true, Option.map₂_some_some]
  · simp only [ha, if_false]
    show Option.map₂ (·-·) (some (0:ℝ)) (some 0) = some 0
    rw [Option.map₂_some_some]; norm_num

theorem bwdGatedG_mul_exp2 (s : BlockState) (DK BK : Nat) (a b : Fin BK → ℝ) :
    Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
        (bwdGatedG s DK BK a) (Tile.uop WithBot.realExp2 (bwdGatedG s DK BK b))
      = bwdGatedG s DK BK (fun i => a i * Real.exp (b i * Real.log 2)) := by
  apply Tile.ext; intro idx
  simp only [bwdGatedG, Tile.bop, Tile.uop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, WithBot.realMul]
  by_cases ha : active s DK BK idx.1
  · simp only [ha, if_true, WithBot.realExp2_some, Option.map₂_some_some]
  · simp only [ha, if_false]
    show Option.map₂ (·*·) (some (0:ℝ)) (WithBot.realExp2 (some 0)) = some 0
    rw [WithBot.realExp2_some, Option.map₂_some_some]; norm_num

theorem bwdGatedG_mul (s : BlockState) (DK BK : Nat) (a b : Fin BK → ℝ) :
    Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
        (bwdGatedG s DK BK a) (bwdGatedG s DK BK b)
      = bwdGatedG s DK BK (fun i => a i * b i) := by
  apply Tile.ext; intro idx
  simp only [bwdGatedG, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, WithBot.realMul]
  by_cases ha : active s DK BK idx.1
  · simp only [ha, if_true, Option.map₂_some_some]
  · simp only [ha, if_false]
    show Option.map₂ (·*·) (some (0:ℝ)) (some 0) = some 0
    rw [Option.map₂_some_some]; norm_num

theorem bwdLoad_gated (s c : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat)
    (hread : ∀ i : Fin BK,
      c.readMem region (offset s s_qk_h DK r BT BK i)
        = s.readMem region (offset s s_qk_h DK r BT BK i)) :
    (⟨fun idx => if (bwdMaskTileG s DK BK).data idx then
          some (c.readMem ((bwdPtrTileG s region s_qk_h DK BT BK r).data idx).1
            ((bwdPtrTileG s region s_qk_h DK BT BK r).data idx).2)
        else (⟨fun _ => some 0⟩ : Tile .real [BK]).data idx⟩ : Tile .real [BK])
      = bwdGatedG s DK BK (bwdLdRowG s region s_qk_h DK BT BK r) := by
  apply Tile.ext; intro idx
  simp only [bwdMaskTileG, bwdPtrTileG, bwdGatedG, bwdLdRowG]
  by_cases ha : active s DK BK idx.1
  · simp only [ha, decide_true, if_true]; rw [hread idx.1]
  · simp only [ha, decide_false, Bool.false_eq_true, if_false]

theorem bwdStore_scatter (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (f : Fin BK → ℝ) (c : BlockState) :
    ((TileShape.allIndices [BK]).foldl
        (fun acc idx =>
          if (bwdMaskTileG s DK BK).data idx then
            acc.writeMemTyped .real
              ((bwdPtrTileG s region s_qk_h DK BT BK r).data idx).1
              ((bwdPtrTileG s region s_qk_h DK BT BK r).data idx).2
              ((bwdGatedG s DK BK f).data idx)
          else acc) c)
      = (TileShape.allIndices [BK]).foldl
          (fun acc idx =>
            if active s DK BK idx.1 then
              acc.writeMem region (offset s s_qk_h DK r BT BK idx.1) (f idx.1)
            else acc) c := by
  congr 1
  funext acc idx
  simp only [bwdMaskTileG, bwdPtrTileG, bwdGatedG]
  by_cases ha : active s DK BK idx.1
  · simp only [ha, decide_true, if_true, BlockState.writeMemTyped_real,
      FloatDType.real_storeValue, WithBot.unbotD_some]
  · simp only [ha, decide_false, Bool.false_eq_true, if_false]

noncomputable def bwdScatterRow (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (f : Fin BK → ℝ) (c : BlockState) : BlockState :=
  (TileShape.allIndices [BK]).foldl
    (fun acc idx =>
      if active s DK BK idx.1 then
        acc.writeMem region (offset s s_qk_h DK r BT BK idx.1) (f idx.1)
      else acc) c

theorem bwdScatterRow_pids (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (f : Fin BK → ℝ) (c : BlockState) :
    (bwdScatterRow s region s_qk_h DK BT BK r f c).pids = c.pids := by
  unfold bwdScatterRow
  exact BlockState.foldl_writeMem_prop_masked_pids region
    (fun k : TileIndex [BK] => offset s s_qk_h DK r BT BK k.1) (fun k => f k.1)
    (fun k => active s DK BK k.1) _ c

@[simp] theorem bwdScatterRow_regs (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (f : Fin BK → ℝ) (c : BlockState)
    (d : TileDType) (sh : TileShape) (n : RegName) :
    (bwdScatterRow s region s_qk_h DK BT BK r f c).regs d sh n = c.regs d sh n := by
  unfold bwdScatterRow
  exact BlockState.foldl_writeMem_prop_masked_regs region
    (fun k : TileIndex [BK] => offset s s_qk_h DK r BT BK k.1) (fun k => f k.1)
    (fun k => active s DK BK k.1) _ c d sh n

theorem bwdScatterRow_readback (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (f : Fin BK → ℝ) (c : BlockState) (i : Fin BK) :
    (bwdScatterRow s region s_qk_h DK BT BK r f c).readMem region (offset s s_qk_h DK r BT BK i)
      = if active s DK BK i then f i
        else c.readMem region (offset s s_qk_h DK r BT BK i) := by
  have hinj : Function.Injective (fun k : TileIndex [BK] => offset s s_qk_h DK r BT BK k.1) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    simp only [offset] at hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  unfold bwdScatterRow
  have := BlockState.scatter_readback_prop_masked_nd (region := region) c
    (fun k : TileIndex [BK] => offset s s_qk_h DK r BT BK k.1) (fun k => f k.1)
    (fun k => active s DK BK k.1) hinj (i, PUnit.unit)
  simpa using this

theorem bwdScatterRow_other_region (s : BlockState) (region R : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (f : Fin BK → ℝ) (c : BlockState) (off : Nat)
    (h : R ≠ region) :
    (bwdScatterRow s region s_qk_h DK BT BK r f c).readMem R off = c.readMem R off := by
  unfold bwdScatterRow
  exact BlockState.scatter_prop_masked_preserves_other_region region
    (fun k : TileIndex [BK] => offset s s_qk_h DK r BT BK k.1) (fun k => f k.1)
    (fun k => active s DK BK k.1) R h off _ c

theorem bwdScatterRow_other_offset (s : BlockState) (region : RegionName)
    (s_qk_h DK BT BK : Nat) (r : Nat) (f : Fin BK → ℝ) (c : BlockState) (off : Nat)
    (h : ∀ i : Fin BK, offset s s_qk_h DK r BT BK i ≠ off) :
    (bwdScatterRow s region s_qk_h DK BT BK r f c).readMem region off = c.readMem region off := by
  unfold bwdScatterRow
  exact BlockState.scatter_prop_masked_preserves_other_offset region
    (fun k : TileIndex [BK] => offset s s_qk_h DK r BT BK k.1) (fun k => f k.1)
    (fun k => active s DK BK k.1) off (fun k _ => h k.1) _ c

theorem bwd_stopOp_eval (c : BlockState) (BT : Nat) (hBT : 0 < BT) :
    evalOp (Op.add .nat Broadcast.nil
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.constNat BT) (Op.constNat 1)) (Op.constNat 1))
        (Op.constNat 1)) c
      = some (Tile.scalar BT) := by
  simp only [evalOp, evalOp_constNat, Option.bind, Tile.bop, NumericDType.add,
    NumericDType.sub, NumericDType.div]
  apply congrArg some; apply Tile.ext; intro idx
  show (BT - 1) / 1 + 1 = BT
  omega

theorem bwdInvG_entry
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s s0 : BlockState) (s_qk_h DK BT BK : Nat)
    (hsh : BwdPrologueShapeG s s0 DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK) :
    bwdInvG DQInner DQInter DKInner DKInter Q K G DG s s_qk_h DK BT BK 0 s0 := by
  have hmemrd : ∀ (r : RegionName) (a : Nat), s0.readMem r a = s.readMem r a := by
    intro r a; simp only [BlockState.readMem, hsh.mem]
  refine ⟨hsh.pid, (fun a => hmemrd G a), (fun a => hmemrd Q a), (fun a => hmemrd K a),
    (fun a => hmemrd DQInner a), (fun a => hmemrd DKInner a), hsh.mask, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · 
    rw [hsh.cum]; apply congrArg some; apply Tile.ext; intro idx
    simp only [bwdGatedG, bwdCumPartialG_zero]
    by_cases ha : active s DK BK idx.1 <;> simp [ha]
  · 
    rw [hsh.lastg]; apply congrArg some; apply Tile.ext; intro idx
    simp only [bwdGatedG, bwdLastgG, lt_irrefl, if_false]
    by_cases ha : active s DK BK idx.1 <;> simp [ha]
  · rw [bwdPtrDecTileG_zero]; simpa using hsh.pg
  · rw [bwdPtrDecTileG_zero]; simpa using hsh.pk
  · rw [bwdPtrDecTileG_zero]; simpa using hsh.pq
  · rw [bwdPtrDecTileG_zero]; simpa using hsh.pdqi
  · rw [bwdPtrDecTileG_zero]; simpa using hsh.pdki
  · rw [bwdPtrDecTileG_zero]; simpa using hsh.pdqt
  · rw [bwdPtrDecTileG_zero]; simpa using hsh.pdkt
  · rw [bwdPtrDecTileG_zero]; simpa using hsh.pdg
  · intro r _ i; exact hmemrd DQInter _
  · intro r _ i; exact hmemrd DKInter _
  · intro r _ i; exact hmemrd DG _
  · intro r hr hr2 i; omega
  · intro r hr hr2 i; omega
  · intro r hr hr2 i; omega

theorem bwdCumPartialG_succ (s : BlockState)
    (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s_qk_h DK BT BK : Nat) (m : Nat) (hm : m < BT) (i : Fin BK) :
    bwdCumPartialG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK (m + 1) i =
      bwdCumPartialG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK m i +
        bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK (BT - 1 - m)
          (fun i' => s.readMem G (offset s s_qk_h DK (BT - 1) BT BK i')) i := by
  simp only [bwdCumPartialG, Fin.sum_univ_castSucc, Fin.val_last, Fin.val_castSucc]

set_option maxHeartbeats 4000000 in
                                                    
theorem bwd_decay_cumsum_step_general
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s : BlockState) (s_qk_h DK BT BK : Nat) (hBK : BK ≤ DK)
    (hDKInner_DQInter : DKInner ≠ DQInter) (hDKInter_DQInter : DKInter ≠ DQInter)
    (hQ_DQInter : Q ≠ DQInter) (hQ_DKInter : Q ≠ DKInter)
    (hK_DQInter : K ≠ DQInter) (hK_DKInter : K ≠ DKInter)
    (hDQInter_DKInter : DQInter ≠ DKInter)
    (hDQInter_DG : DQInter ≠ DG) (hDKInter_DG : DKInter ≠ DG)
    (hG_DQInter : G ≠ DQInter) (hG_DKInter : G ≠ DKInter) (hG_DG : G ≠ DG)
    (hQ_DG : Q ≠ DG) (hK_DG : K ≠ DG)
    (hDQInner_DQInter : DQInner ≠ DQInter) (hDQInner_DKInter : DQInner ≠ DKInter)
    (hDQInner_DG : DQInner ≠ DG)
    (hDKInner_DKInter : DKInner ≠ DKInter) (hDKInner_DG : DKInner ≠ DG)
    (m : Nat) (hm : m < BT) (sc : BlockState)
    (hP : bwdInvG DQInner DQInter DKInner DKInter Q K G DG s s_qk_h DK BT BK m sc) :
    ∃ s',
      stepStmts (bwdIterBodyG DK BT BK) (sc.setReg "__rev_j" .nat [] (Tile.scalar m)) = some s' ∧
      bwdInvG DQInner DQInter DKInner DKInter Q K G DG s s_qk_h DK BT BK (m + 1) s' := by
  obtain ⟨hpid, hGrd, hQrd, hKrd, hDQird, hDKird, hmaskP, hcumP, hlastgP,
    hpgP, hpkP, hpqP, hpdqiP, hpdkiP, hpdqtP, hpdktP, hpdgP,
    hDQIunt, hDKIunt, hDGunt, hDQIwr, hDKIwr, hDGwr⟩ := hP
  
  set r := BT - 1 - m with hr
  
  rw [bwdPtrDecTileG_row s G s_qk_h DK BT BK m hm] at hpgP
  rw [bwdPtrDecTileG_row s K s_qk_h DK BT BK m hm] at hpkP
  rw [bwdPtrDecTileG_row s Q s_qk_h DK BT BK m hm] at hpqP
  rw [bwdPtrDecTileG_row s DQInner s_qk_h DK BT BK m hm] at hpdqiP
  rw [bwdPtrDecTileG_row s DKInner s_qk_h DK BT BK m hm] at hpdkiP
  rw [bwdPtrDecTileG_row s DQInter s_qk_h DK BT BK m hm] at hpdqtP
  rw [bwdPtrDecTileG_row s DKInter s_qk_h DK BT BK m hm] at hpdktP
  rw [bwdPtrDecTileG_row s DG s_qk_h DK BT BK m hm] at hpdgP
  
  set lgF : Fin BK → ℝ := fun i' => s.readMem G (offset s s_qk_h DK (BT - 1) BT BK i') with hlgF
  
  set cumF : Fin BK → ℝ :=
    bwdCumPartialG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK m with hcumF
  set sin := sc.setReg "__rev_j" .nat [] (Tile.scalar m) with hsin
  have hsinpid : sin.pids = s.pids := by rw [hsin, BlockState.setReg_pids]; exact hpid
  
  have hsinG : ∀ a, sin.readMem G a = s.readMem G a := by
    intro a; rw [hsin, BlockState.setReg_readMem]; exact hGrd a
  have hsinQ : ∀ a, sin.readMem Q a = s.readMem Q a := by
    intro a; rw [hsin, BlockState.setReg_readMem]; exact hQrd a
  have hsinK : ∀ a, sin.readMem K a = s.readMem K a := by
    intro a; rw [hsin, BlockState.setReg_readMem]; exact hKrd a
  have hsinDQi : ∀ a, sin.readMem DQInner a = s.readMem DQInner a := by
    intro a; rw [hsin, BlockState.setReg_readMem]; exact hDQird a
  have hsinDKi : ∀ a, sin.readMem DKInner a = s.readMem DKInner a := by
    intro a; rw [hsin, BlockState.setReg_readMem]; exact hDKird a
  
  set maskT := bwdMaskTileG s DK BK with hmaskT
  
  have hsinrev : sin.regs .nat [] "__rev_j" = some (Tile.scalar m) := by
    rw [hsin]; exact BlockState.setReg_same _ _ _ _ _
  have hsinmask : sin.regs .bool [BK] "mask" = some maskT := by
    rw [hsin, BlockState.setReg_ne_name (h := by decide)]; exact hmaskP
  have hsinpg : sin.regs .ptr [BK] "p_g" = some (bwdPtrTileG s G s_qk_h DK BT BK r) := by
    rw [hsin, BlockState.setReg_ne_name (h := by decide)]; exact hpgP
  
  simp only [bwdIterBodyG]
  
  have e0 : stepStmt (Stmt.assign .nat [] "j"
      (Op.sub .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil (Op.constNat BT) (Op.constNat 1))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "__rev_j") (Op.constNat 1)))) sin
      = some (sin.setReg "j" .nat [] (Tile.scalar (BT - 1 - m * 1))) := by
    rw [bwdEval_assign_subNat sin "j" _ _ (BT - 1) (m * 1)
        (by simp [evalOp, Tile.bop, NumericDType.sub])
        (by simp [evalOp, hsinrev, Tile.bop, NumericDType.mul])]
  rw [stepStmts.cons_some e0]
  set e0s := sin.setReg "j" .nat [] (Tile.scalar (BT - 1 - m * 1)) with he0s
  have e0smask : e0s.regs .bool [BK] "mask" = some maskT := by
    rw [he0s, BlockState.setReg_ne_name (h := by decide)]; exact hsinmask
  have e0spg : e0s.regs .ptr [BK] "p_g" = some (bwdPtrTileG s G s_qk_h DK BT BK r) := by
    rw [he0s, BlockState.setReg_ne_name (h := by decide)]; exact hsinpg
  
  have e1 : stepStmt (Stmt.assign .real [BK] "_g"
      (Op.load ComputeDType.fp32.eraseDType (MemAccess.ptr (Op.ref .ptr [BK] "p_g"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK])))) e0s
      = some (e0s.setReg "_g" .real [BK]
          (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r))) := by
    simp only [ComputeDType.eraseDType]
    rw [bwdEval_assign_load_ptr_maskOther e0s "_g" "p_g" _ _
        (bwdPtrTileG s G s_qk_h DK BT BK r) maskT ⟨fun _ => some 0⟩
        e0spg (by rw [evalOp_ref]; exact e0smask) (by simp [evalOp])]
    rw [bwdLoad_gated s e0s G s_qk_h DK BT BK r
        (fun i => by rw [he0s, BlockState.setReg_readMem, hsinG])]
  rw [stepStmts.cons_some e1]
  set e1s := e0s.setReg "_g" .real [BK]
    (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r)) with he1s

  have e1st : e1s.regs .nat [] "j" = some (Tile.scalar (BT - 1 - m * 1)) := by
    rw [he1s, BlockState.setReg_ne_name (h := by decide), he0s]
    exact BlockState.setReg_same _ _ _ _ _
  have e1sgval : e1s.regs .real [BK] "_g" = some
      (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r)) := by
    rw [he1s]; exact BlockState.setReg_same _ _ _ _ _
  have hcond : evalOp (Op.eq ComparableDType.nat Broadcast.nil (Op.ref .nat [] "j")
      (Op.sub .nat Broadcast.nil (Op.constNat BT) (Op.constNat 1))) e1s
      = some (Tile.scalar (decide (BT - 1 - m * 1 = BT - 1))) := by
    rw [bwdEval_eqNat e1s _ _ (Tile.scalar (BT - 1 - m * 1)) (Tile.scalar (BT - 1))
        (by rw [evalOp_ref]; exact e1st)
        (by simp [evalOp, Tile.bop, NumericDType.sub])]
    rfl

  obtain ⟨s3, hhead, hs3t, hs3gval, hs3mask, hs3lastg, hs3cum,
      hs3pg, hs3pk, hs3pq, hs3pdqi, hs3pdki, hs3pdqt, hs3pdkt, hs3pdg, hs3peelmem⟩ :
      ∃ s3 : BlockState,
        stepStmt (Stmt.ifThen
          (Op.eq ComparableDType.nat Broadcast.nil (Op.ref .nat [] "j")
            (Op.sub .nat Broadcast.nil (Op.constNat BT) (Op.constNat 1)))
          [Stmt.assign .real [BK] "last_g" (Op.ref .real [BK] "_g")]) e1s = some s3 ∧
        s3.regs .nat [] "j" = some (Tile.scalar (BT - 1 - m * 1)) ∧
        s3.regs .real [BK] "_g" = some (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r)) ∧
        s3.regs .bool [BK] "mask" = some maskT ∧
        s3.regs .real [BK] "last_g" = some (bwdGatedG s DK BK lgF) ∧
        s3.regs .real [BK] "cum_grad_dg" = some (bwdGatedG s DK BK cumF) ∧
        s3.regs .ptr [BK] "p_g" = some (bwdPtrTileG s G s_qk_h DK BT BK r) ∧
        s3.regs .ptr [BK] "p_k" = some (bwdPtrTileG s K s_qk_h DK BT BK r) ∧
        s3.regs .ptr [BK] "p_q" = some (bwdPtrTileG s Q s_qk_h DK BT BK r) ∧
        s3.regs .ptr [BK] "p_dq_inner" = some (bwdPtrTileG s DQInner s_qk_h DK BT BK r) ∧
        s3.regs .ptr [BK] "p_dk_inner" = some (bwdPtrTileG s DKInner s_qk_h DK BT BK r) ∧
        s3.regs .ptr [BK] "p_dq_inter" = some (bwdPtrTileG s DQInter s_qk_h DK BT BK r) ∧
        s3.regs .ptr [BK] "p_dk_inter" = some (bwdPtrTileG s DKInter s_qk_h DK BT BK r) ∧
        s3.regs .ptr [BK] "p_dg" = some (bwdPtrTileG s DG s_qk_h DK BT BK r) ∧
        (s3.pids = s.pids ∧ s3.mem = sc.mem) := by
    
    have e1speel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
        n ≠ "_g" → n ≠ "j" → n ≠ "__rev_j" →
        e1s.regs d sh n = sc.regs d sh n := by
      intro d sh n h1 h2 h3
      rw [he1s, BlockState.setReg_ne_name (h := h1), he0s,
          BlockState.setReg_ne_name (h := h2), hsin, BlockState.setReg_ne_name (h := h3)]
    have e1smask : e1s.regs .bool [BK] "mask" = some maskT := by
      rw [e1speel .bool [BK] "mask" (by decide) (by decide) (by decide)]; exact hmaskP
    have e1scum : e1s.regs .real [BK] "cum_grad_dg" = some (bwdGatedG s DK BK cumF) := by
      rw [e1speel .real [BK] "cum_grad_dg" (by decide) (by decide) (by decide)]; exact hcumP
    have e1spg : e1s.regs .ptr [BK] "p_g" = some (bwdPtrTileG s G s_qk_h DK BT BK r) := by
      rw [e1speel .ptr [BK] "p_g" (by decide) (by decide) (by decide)]; exact hpgP
    have e1spk : e1s.regs .ptr [BK] "p_k" = some (bwdPtrTileG s K s_qk_h DK BT BK r) := by
      rw [e1speel .ptr [BK] "p_k" (by decide) (by decide) (by decide)]; exact hpkP
    have e1spq : e1s.regs .ptr [BK] "p_q" = some (bwdPtrTileG s Q s_qk_h DK BT BK r) := by
      rw [e1speel .ptr [BK] "p_q" (by decide) (by decide) (by decide)]; exact hpqP
    have e1spdqi : e1s.regs .ptr [BK] "p_dq_inner" = some (bwdPtrTileG s DQInner s_qk_h DK BT BK r) := by
      rw [e1speel .ptr [BK] "p_dq_inner" (by decide) (by decide) (by decide)]; exact hpdqiP
    have e1spdki : e1s.regs .ptr [BK] "p_dk_inner" = some (bwdPtrTileG s DKInner s_qk_h DK BT BK r) := by
      rw [e1speel .ptr [BK] "p_dk_inner" (by decide) (by decide) (by decide)]; exact hpdkiP
    have e1spdqt : e1s.regs .ptr [BK] "p_dq_inter" = some (bwdPtrTileG s DQInter s_qk_h DK BT BK r) := by
      rw [e1speel .ptr [BK] "p_dq_inter" (by decide) (by decide) (by decide)]; exact hpdqtP
    have e1spdkt : e1s.regs .ptr [BK] "p_dk_inter" = some (bwdPtrTileG s DKInter s_qk_h DK BT BK r) := by
      rw [e1speel .ptr [BK] "p_dk_inter" (by decide) (by decide) (by decide)]; exact hpdktP
    have e1spdg : e1s.regs .ptr [BK] "p_dg" = some (bwdPtrTileG s DG s_qk_h DK BT BK r) := by
      rw [e1speel .ptr [BK] "p_dg" (by decide) (by decide) (by decide)]; exact hpdgP
    have e1spid : e1s.pids = s.pids := by
      rw [he1s, BlockState.setReg_pids, he0s, BlockState.setReg_pids]; exact hsinpid
    have e1smem : e1s.mem = sc.mem := by
      rw [he1s]; simp only [BlockState.setReg]; rw [he0s]; simp only [BlockState.setReg]
      rw [hsin]; simp only [BlockState.setReg]
    by_cases hm0 : m = 0
    · 
      have hrBT : r = BT - 1 := by rw [hr, hm0]; omega
      have htrue : (decide (BT - 1 - m * 1 = BT - 1)) = Bool.true := by
        rw [hm0]; simp
      rw [bwdEval_ifThen_true e1s _ _
        (show evalOp _ e1s = some (Tile.scalar Bool.true) by rw [hcond, htrue])]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
          (show evalOp (Op.ref .real [BK] "_g") e1s
              = some (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r)) by
            rw [evalOp_ref]; exact e1sgval))]
      rw [stepStmts.nil]
      refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1st
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1sgval
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1smask
      · rw [BlockState.setReg_same]
        show some (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r)) = some (bwdGatedG s DK BK lgF)
        apply congrArg; apply congrArg; rw [hlgF]; funext i'
        simp only [bwdLdRowG]; rw [hrBT]
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1scum
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1spg
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1spk
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1spq
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1spdqi
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1spdki
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1spdqt
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1spdkt
      · rw [BlockState.setReg_ne_name (h := by decide)]; exact e1spdg
      · rw [BlockState.setReg_pids]; exact e1spid
      · simp only [BlockState.setReg]; exact e1smem
    · 
      have hfalse : (decide (BT - 1 - m * 1 = BT - 1)) = Bool.false := by
        have hmpos : 0 < m := Nat.pos_of_ne_zero hm0
        simp only [decide_eq_false_iff_not]; omega
      have e1slastg : e1s.regs .real [BK] "last_g" = some (bwdGatedG s DK BK lgF) := by
        rw [e1speel .real [BK] "last_g" (by decide) (by decide) (by decide), hlastgP]
        apply congrArg; apply congrArg; rw [hlgF]; funext i'; simp only [bwdLastgG]
        rw [if_pos (Nat.pos_of_ne_zero hm0)]
      rw [bwdEval_ifThen_false e1s _ _
        (show evalOp _ e1s = some (Tile.scalar Bool.false) by rw [hcond, hfalse])]
      exact ⟨e1s, rfl, e1st, e1sgval, e1smask, e1slastg, e1scum,
        e1spg, e1spk, e1spq, e1spdqi, e1spdki, e1spdqt, e1spdkt, e1spdg, e1spid, e1smem⟩
  rw [stepStmts.cons_some hhead]

  have hrlt : r < BT - m := by rw [hr]; omega

  have hs3mem : ∀ region a, s3.readMem region a = sc.readMem region a := by
    intro region a; simp only [BlockState.readMem, hs3peelmem.2]
  have rdG : ∀ i : Fin BK, s3.readMem G (offset s s_qk_h DK r BT BK i)
      = s.readMem G (offset s s_qk_h DK r BT BK i) := by intro i; rw [hs3mem]; exact hGrd _
  have rdQ : ∀ i : Fin BK, s3.readMem Q (offset s s_qk_h DK r BT BK i)
      = s.readMem Q (offset s s_qk_h DK r BT BK i) := by intro i; rw [hs3mem]; exact hQrd _
  have rdK : ∀ i : Fin BK, s3.readMem K (offset s s_qk_h DK r BT BK i)
      = s.readMem K (offset s s_qk_h DK r BT BK i) := by intro i; rw [hs3mem]; exact hKrd _
  have rdDQi : ∀ i : Fin BK, s3.readMem DQInner (offset s s_qk_h DK r BT BK i)
      = s.readMem DQInner (offset s s_qk_h DK r BT BK i) := by intro i; rw [hs3mem]; exact hDQird _
  have rdDKi : ∀ i : Fin BK, s3.readMem DKInner (offset s s_qk_h DK r BT BK i)
      = s.readMem DKInner (offset s s_qk_h DK r BT BK i) := by intro i; rw [hs3mem]; exact hDKird _
  have rdDQt : ∀ i : Fin BK, s3.readMem DQInter (offset s s_qk_h DK r BT BK i)
      = s.readMem DQInter (offset s s_qk_h DK r BT BK i) := by
    intro i; rw [hs3mem]; exact hDQIunt r hrlt i
  have rdDKt : ∀ i : Fin BK, s3.readMem DKInter (offset s s_qk_h DK r BT BK i)
      = s.readMem DKInter (offset s s_qk_h DK r BT BK i) := by
    intro i; rw [hs3mem]; exact hDKIunt r hrlt i
  
  set otherT : Tile .real [BK] := ⟨fun _ => some 0⟩ with hotherT
  have hother : ∀ c : BlockState, evalOp ((Op.const (0 : ℝ)).broadcast [BK]) c = some otherT := by
    intro c; simp [evalOp, hotherT]
  
  have d3 : stepStmt (Stmt.assign .real [BK] "_dq1"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_dq_inner"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK])))) s3
      = some (s3.setReg "_dq1" .real [BK]
          (bwdGatedG s DK BK (bwdLdRowG s DQInner s_qk_h DK BT BK r))) := by
    rw [bwdEval_assign_load_ptr_maskOther s3 "_dq1" "p_dq_inner" _ _
        (bwdPtrTileG s DQInner s_qk_h DK BT BK r) maskT otherT
        hs3pdqi (by rw [evalOp_ref]; exact hs3mask) (hother s3)]
    rw [bwdLoad_gated s s3 DQInner s_qk_h DK BT BK r rdDQi]
  rw [stepStmts.cons_some d3]
  set c1 := s3.setReg "_dq1" .real [BK]
    (bwdGatedG s DK BK (bwdLdRowG s DQInner s_qk_h DK BT BK r)) with hc1
  have c1mask : c1.regs .bool [BK] "mask" = some maskT := by
    rw [hc1, BlockState.setReg_ne_name (h := by decide)]; exact hs3mask
  have c1pdqt : c1.regs .ptr [BK] "p_dq_inter" = some (bwdPtrTileG s DQInter s_qk_h DK BT BK r) := by
    rw [hc1, BlockState.setReg_ne_name (h := by decide)]; exact hs3pdqt
  have c1rdDQt : ∀ i : Fin BK, c1.readMem DQInter (offset s s_qk_h DK r BT BK i)
      = s.readMem DQInter (offset s s_qk_h DK r BT BK i) := by
    intro i; rw [hc1, BlockState.setReg_readMem]; exact rdDQt i
  
  have d4 : stepStmt (Stmt.assign .real [BK] "_dq2"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_dq_inter"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK])))) c1
      = some (c1.setReg "_dq2" .real [BK]
          (bwdGatedG s DK BK (bwdLdRowG s DQInter s_qk_h DK BT BK r))) := by
    rw [bwdEval_assign_load_ptr_maskOther c1 "_dq2" "p_dq_inter" _ _
        (bwdPtrTileG s DQInter s_qk_h DK BT BK r) maskT otherT
        c1pdqt (by rw [evalOp_ref]; exact c1mask) (hother c1)]
    rw [bwdLoad_gated s c1 DQInter s_qk_h DK BT BK r c1rdDQt]
  rw [stepStmts.cons_some d4]
  set c2 := c1.setReg "_dq2" .real [BK]
    (bwdGatedG s DK BK (bwdLdRowG s DQInter s_qk_h DK BT BK r)) with hc2
  have c2gval : c2.regs .real [BK] "_g" = some
      (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r)) := by
    rw [hc2, BlockState.setReg_ne_name (h := by decide), hc1,
        BlockState.setReg_ne_name (h := by decide)]; exact hs3gval
  
  have d5 : stepStmt (Stmt.assign .real [BK] "_dq2"
      (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "_dq2")
        (Op.ref .real [BK] "_g").exp2)) c2
      = some (c2.setReg "_dq2" .real [BK]
          (bwdGatedG s DK BK (fun i => bwdLdRowG s DQInter s_qk_h DK BT BK r i *
            Real.exp (bwdLdRowG s G s_qk_h DK BT BK r i * Real.log 2)))) := by
    apply stepStmt_assign_eq_some
    rw [bwdEval_mul c2 _ _ (bwdGatedG s DK BK (bwdLdRowG s DQInter s_qk_h DK BT BK r))
        (Tile.uop WithBot.realExp2 (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r)))
        (by rw [evalOp_ref, hc2, BlockState.setReg_same])
        (by rw [bwdEval_exp2 c2 _ (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r))
            (by rw [evalOp_ref]; exact c2gval)])]
    rw [bwdGatedG_mul_exp2]
  rw [stepStmts.cons_some d5]
  set c3 := c2.setReg "_dq2" .real [BK]
    (bwdGatedG s DK BK (fun i => bwdLdRowG s DQInter s_qk_h DK BT BK r i *
      Real.exp (bwdLdRowG s G s_qk_h DK BT BK r i * Real.log 2))) with hc3
  have c3dq1 : c3.regs .real [BK] "_dq1" = some
      (bwdGatedG s DK BK (bwdLdRowG s DQInner s_qk_h DK BT BK r)) := by
    rw [hc3, BlockState.setReg_ne_name (h := by decide), hc2,
        BlockState.setReg_ne_name (h := by decide), hc1, BlockState.setReg_same]
  
  have d6 : stepStmt (Stmt.assign .real [BK] "_dq"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [BK] "_dq1")
        (Op.ref .real [BK] "_dq2"))) c3
      = some (c3.setReg "_dq" .real [BK]
          (bwdGatedG s DK BK (bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r))) := by
    apply stepStmt_assign_eq_some
    rw [bwdEval_add c3 _ _ (bwdGatedG s DK BK (bwdLdRowG s DQInner s_qk_h DK BT BK r))
        (bwdGatedG s DK BK (fun i => bwdLdRowG s DQInter s_qk_h DK BT BK r i *
          Real.exp (bwdLdRowG s G s_qk_h DK BT BK r i * Real.log 2)))
        (by rw [evalOp_ref]; exact c3dq1)
        (by rw [evalOp_ref, hc3, BlockState.setReg_same])]
    rw [bwdGatedG_add]; rfl
  rw [stepStmts.cons_some d6]
  set c4 := c3.setReg "_dq" .real [BK]
    (bwdGatedG s DK BK (bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r)) with hc4
  have c4mask : c4.regs .bool [BK] "mask" = some maskT := by
    rw [hc4, BlockState.setReg_ne_name (h := by decide), hc3,
        BlockState.setReg_ne_name (h := by decide), hc2,
        BlockState.setReg_ne_name (h := by decide), hc1,
        BlockState.setReg_ne_name (h := by decide)]; exact hs3mask
  have c4pdqt : c4.regs .ptr [BK] "p_dq_inter" = some (bwdPtrTileG s DQInter s_qk_h DK BT BK r) := by
    rw [hc4, BlockState.setReg_ne_name (h := by decide), hc3,
        BlockState.setReg_ne_name (h := by decide), hc2,
        BlockState.setReg_ne_name (h := by decide), hc1,
        BlockState.setReg_ne_name (h := by decide)]; exact hs3pdqt
  have c4dq : c4.regs .real [BK] "_dq" = some
      (bwdGatedG s DK BK (bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r)) := by
    rw [hc4]; exact BlockState.setReg_same _ _ _ _ _
  
  have d7 : stepStmt (Stmt.store .real [BK] (MemAccess.ptr (Op.ref .ptr [BK] "p_dq_inter"))
      (Op.ref .real [BK] "_dq") (MaskOpt.mask (Op.ref .bool [BK] "mask"))) c4
      = some (bwdScatterRow s DQInter s_qk_h DK BT BK r
          (bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r) c4) := by
    rw [bwdEval_store_ptr_masked c4 "p_dq_inter" _ _
        (bwdPtrTileG s DQInter s_qk_h DK BT BK r)
        (bwdGatedG s DK BK (bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r)) maskT
        c4pdqt (by rw [evalOp_ref]; exact c4dq) (by rw [evalOp_ref]; exact c4mask)]
    rw [bwdStore_scatter]; rfl
  rw [stepStmts.cons_some d7]
  set c5 := bwdScatterRow s DQInter s_qk_h DK BT BK r
    (bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r) c4 with hc5
  
  have c4peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "_dq" → n ≠ "_dq2" → n ≠ "_dq1" → c4.regs d sh n = s3.regs d sh n := by
    intro d sh n h1 h2 h3
    rw [hc4, BlockState.setReg_ne_name (h := h1), hc3,
        BlockState.setReg_ne_name (h := h2), hc2,
        BlockState.setReg_ne_name (h := h2), hc1,
        BlockState.setReg_ne_name (h := h3)]
  have c4read : ∀ (region : RegionName) (i : Fin BK),
      c4.readMem region (offset s s_qk_h DK r BT BK i)
        = s3.readMem region (offset s s_qk_h DK r BT BK i) := by
    intro region i
    rw [hc4, BlockState.setReg_readMem, hc3, BlockState.setReg_readMem,
        hc2, BlockState.setReg_readMem, hc1, BlockState.setReg_readMem]
  
  have c5peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "_dq" → n ≠ "_dq2" → n ≠ "_dq1" → c5.regs d sh n = s3.regs d sh n := by
    intro d sh n h1 h2 h3
    rw [hc5, bwdScatterRow_regs]; exact c4peel d sh n h1 h2 h3
  have c5read : ∀ (region : RegionName) (i : Fin BK), region ≠ DQInter →
      c5.readMem region (offset s s_qk_h DK r BT BK i)
        = s3.readMem region (offset s s_qk_h DK r BT BK i) := by
    intro region i hreg
    rw [hc5, bwdScatterRow_other_region s DQInter region s_qk_h DK BT BK r _ c4 _ hreg]
    exact c4read region i
  
  have c5mask : c5.regs .bool [BK] "mask" = some maskT := by
    rw [c5peel .bool [BK] "mask" (by decide) (by decide) (by decide)]; exact hs3mask
  have c5pdki : c5.regs .ptr [BK] "p_dk_inner" = some (bwdPtrTileG s DKInner s_qk_h DK BT BK r) := by
    rw [c5peel .ptr [BK] "p_dk_inner" (by decide) (by decide) (by decide)]; exact hs3pdki
  have d8 : stepStmt (Stmt.assign .real [BK] "_dk1"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_dk_inner"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK])))) c5
      = some (c5.setReg "_dk1" .real [BK]
          (bwdGatedG s DK BK (bwdLdRowG s DKInner s_qk_h DK BT BK r))) := by
    rw [bwdEval_assign_load_ptr_maskOther c5 "_dk1" "p_dk_inner" _ _
        (bwdPtrTileG s DKInner s_qk_h DK BT BK r) maskT otherT
        c5pdki (by rw [evalOp_ref]; exact c5mask) (hother c5)]
    rw [bwdLoad_gated s c5 DKInner s_qk_h DK BT BK r
        (fun i => by rw [c5read DKInner i hDKInner_DQInter]; exact rdDKi i)]
  rw [stepStmts.cons_some d8]
  set c6 := c5.setReg "_dk1" .real [BK]
    (bwdGatedG s DK BK (bwdLdRowG s DKInner s_qk_h DK BT BK r)) with hc6
  
  have c6mask : c6.regs .bool [BK] "mask" = some maskT := by
    rw [hc6, BlockState.setReg_ne_name (h := by decide)]; exact c5mask
  have c6pdkt : c6.regs .ptr [BK] "p_dk_inter" = some (bwdPtrTileG s DKInter s_qk_h DK BT BK r) := by
    rw [hc6, BlockState.setReg_ne_name (h := by decide),
        c5peel .ptr [BK] "p_dk_inter" (by decide) (by decide) (by decide)]; exact hs3pdkt
  have c6rdDKt : ∀ i : Fin BK, c6.readMem DKInter (offset s s_qk_h DK r BT BK i)
      = s.readMem DKInter (offset s s_qk_h DK r BT BK i) := by
    intro i; rw [hc6, BlockState.setReg_readMem,
      c5read DKInter i hDKInter_DQInter]; exact rdDKt i
  have d9 : stepStmt (Stmt.assign .real [BK] "_dk2"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_dk_inter"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK])))) c6
      = some (c6.setReg "_dk2" .real [BK]
          (bwdGatedG s DK BK (bwdLdRowG s DKInter s_qk_h DK BT BK r))) := by
    rw [bwdEval_assign_load_ptr_maskOther c6 "_dk2" "p_dk_inter" _ _
        (bwdPtrTileG s DKInter s_qk_h DK BT BK r) maskT otherT
        c6pdkt (by rw [evalOp_ref]; exact c6mask) (hother c6)]
    rw [bwdLoad_gated s c6 DKInter s_qk_h DK BT BK r c6rdDKt]
  rw [stepStmts.cons_some d9]
  set c7 := c6.setReg "_dk2" .real [BK]
    (bwdGatedG s DK BK (bwdLdRowG s DKInter s_qk_h DK BT BK r)) with hc7
  have c7peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "_dk2" → n ≠ "_dk1" → n ≠ "_dq" → n ≠ "_dq2" → n ≠ "_dq1" →
      c7.regs d sh n = s3.regs d sh n := by
    intro d sh n h1 h2 h3 h4 h5
    rw [hc7, BlockState.setReg_ne_name (h := h1), hc6,
        BlockState.setReg_ne_name (h := h2)]
    exact c5peel d sh n h3 h4 h5
  have c7lastg : c7.regs .real [BK] "last_g" = some (bwdGatedG s DK BK lgF) := by
    rw [c7peel .real [BK] "last_g" (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3lastg
  have c7gval : c7.regs .real [BK] "_g" = some
      (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r)) := by
    rw [c7peel .real [BK] "_g" (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3gval
  have c7dk2 : c7.regs .real [BK] "_dk2" = some
      (bwdGatedG s DK BK (bwdLdRowG s DKInter s_qk_h DK BT BK r)) := by
    rw [hc7]; exact BlockState.setReg_same _ _ _ _ _
  
  have hsub10 : evalOp (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BK] "last_g")
      (Op.ref .real [BK] "_g")) c7
      = some (bwdGatedG s DK BK (fun i => lgF i - bwdLdRowG s G s_qk_h DK BT BK r i)) := by
    rw [bwdEval_sub c7 _ _ (bwdGatedG s DK BK lgF)
        (bwdGatedG s DK BK (bwdLdRowG s G s_qk_h DK BT BK r))
        (by rw [evalOp_ref]; exact c7lastg) (by rw [evalOp_ref]; exact c7gval)]
    rw [bwdGatedG_sub]
  have d10 : stepStmt (Stmt.assign .real [BK] "_dk2"
      (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "_dk2")
        (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BK] "last_g")
            (Op.ref .real [BK] "_g")).exp2)) c7
      = some (c7.setReg "_dk2" .real [BK]
          (bwdGatedG s DK BK (fun i => bwdLdRowG s DKInter s_qk_h DK BT BK r i *
            Real.exp ((lgF i - bwdLdRowG s G s_qk_h DK BT BK r i) * Real.log 2)))) := by
    apply stepStmt_assign_eq_some
    rw [bwdEval_mul c7 _ _ (bwdGatedG s DK BK (bwdLdRowG s DKInter s_qk_h DK BT BK r))
        (Tile.uop WithBot.realExp2
          (bwdGatedG s DK BK (fun i => lgF i - bwdLdRowG s G s_qk_h DK BT BK r i)))
        (by rw [evalOp_ref]; exact c7dk2)
        (by rw [bwdEval_exp2 c7 _
            (bwdGatedG s DK BK (fun i => lgF i - bwdLdRowG s G s_qk_h DK BT BK r i)) hsub10])]
    rw [bwdGatedG_mul_exp2]
  rw [stepStmts.cons_some d10]
  set c8 := c7.setReg "_dk2" .real [BK]
    (bwdGatedG s DK BK (fun i => bwdLdRowG s DKInter s_qk_h DK BT BK r i *
      Real.exp ((lgF i - bwdLdRowG s G s_qk_h DK BT BK r i) * Real.log 2))) with hc8
  have c8dk1 : c8.regs .real [BK] "_dk1" = some
      (bwdGatedG s DK BK (bwdLdRowG s DKInner s_qk_h DK BT BK r)) := by
    rw [hc8, BlockState.setReg_ne_name (h := by decide), hc7,
        BlockState.setReg_ne_name (h := by decide), hc6, BlockState.setReg_same]
  
  have d11 : stepStmt (Stmt.assign .real [BK] "_dk"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [BK] "_dk1")
        (Op.ref .real [BK] "_dk2"))) c8
      = some (c8.setReg "_dk" .real [BK]
          (bwdGatedG s DK BK (bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF))) := by
    apply stepStmt_assign_eq_some
    rw [bwdEval_add c8 _ _ (bwdGatedG s DK BK (bwdLdRowG s DKInner s_qk_h DK BT BK r))
        (bwdGatedG s DK BK (fun i => bwdLdRowG s DKInter s_qk_h DK BT BK r i *
          Real.exp ((lgF i - bwdLdRowG s G s_qk_h DK BT BK r i) * Real.log 2)))
        (by rw [evalOp_ref]; exact c8dk1)
        (by rw [evalOp_ref, hc8, BlockState.setReg_same])]
    rw [bwdGatedG_add]; rfl
  rw [stepStmts.cons_some d11]
  set c9 := c8.setReg "_dk" .real [BK]
    (bwdGatedG s DK BK (bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF)) with hc9
  have c9peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "_dk" → n ≠ "_dk2" → n ≠ "_dk1" → n ≠ "_dq" → n ≠ "_dq2" → n ≠ "_dq1" →
      c9.regs d sh n = s3.regs d sh n := by
    intro d sh n h0 h1 h2 h3 h4 h5
    rw [hc9, BlockState.setReg_ne_name (h := h0), hc8,
        BlockState.setReg_ne_name (h := h1)]
    exact c7peel d sh n h1 h2 h3 h4 h5
  have c9mask : c9.regs .bool [BK] "mask" = some maskT := by
    rw [c9peel .bool [BK] "mask" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3mask
  have c9pdkt : c9.regs .ptr [BK] "p_dk_inter" = some (bwdPtrTileG s DKInter s_qk_h DK BT BK r) := by
    rw [c9peel .ptr [BK] "p_dk_inter" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3pdkt
  have c9dk : c9.regs .real [BK] "_dk" = some
      (bwdGatedG s DK BK (bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF)) := by
    rw [hc9]; exact BlockState.setReg_same _ _ _ _ _
  
  have c9read : ∀ (region : RegionName) (i : Fin BK),
      c9.readMem region (offset s s_qk_h DK r BT BK i)
        = c5.readMem region (offset s s_qk_h DK r BT BK i) := by
    intro region i
    rw [hc9, BlockState.setReg_readMem, hc8, BlockState.setReg_readMem,
        hc7, BlockState.setReg_readMem, hc6, BlockState.setReg_readMem]
  
  have d12 : stepStmt (Stmt.store .real [BK] (MemAccess.ptr (Op.ref .ptr [BK] "p_dk_inter"))
      (Op.ref .real [BK] "_dk") (MaskOpt.mask (Op.ref .bool [BK] "mask"))) c9
      = some (bwdScatterRow s DKInter s_qk_h DK BT BK r
          (bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF) c9) := by
    rw [bwdEval_store_ptr_masked c9 "p_dk_inter" _ _
        (bwdPtrTileG s DKInter s_qk_h DK BT BK r)
        (bwdGatedG s DK BK (bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF)) maskT
        c9pdkt (by rw [evalOp_ref]; exact c9dk) (by rw [evalOp_ref]; exact c9mask)]
    rw [bwdStore_scatter]; rfl
  rw [stepStmts.cons_some d12]
  set c10 := bwdScatterRow s DKInter s_qk_h DK BT BK r
    (bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF) c9 with hc10
  have c10peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "_dk" → n ≠ "_dk2" → n ≠ "_dk1" → n ≠ "_dq" → n ≠ "_dq2" → n ≠ "_dq1" →
      c10.regs d sh n = s3.regs d sh n := by
    intro d sh n h0 h1 h2 h3 h4 h5
    rw [hc10, bwdScatterRow_regs]; exact c9peel d sh n h0 h1 h2 h3 h4 h5
  have c10read : ∀ (region : RegionName) (i : Fin BK), region ≠ DKInter → region ≠ DQInter →
      c10.readMem region (offset s s_qk_h DK r BT BK i)
        = s3.readMem region (offset s s_qk_h DK r BT BK i) := by
    intro region i h1 h2
    rw [hc10, bwdScatterRow_other_region s DKInter region s_qk_h DK BT BK r _ c9 _ h1,
        c9read region i]
    exact c5read region i h2
  
  have c10mask : c10.regs .bool [BK] "mask" = some maskT := by
    rw [c10peel .bool [BK] "mask" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3mask
  have c10pq : c10.regs .ptr [BK] "p_q" = some (bwdPtrTileG s Q s_qk_h DK BT BK r) := by
    rw [c10peel .ptr [BK] "p_q" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3pq
  have d13 : stepStmt (Stmt.assign .real [BK] "_q"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_q"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK])))) c10
      = some (c10.setReg "_q" .real [BK]
          (bwdGatedG s DK BK (bwdLdRowG s Q s_qk_h DK BT BK r))) := by
    rw [bwdEval_assign_load_ptr_maskOther c10 "_q" "p_q" _ _
        (bwdPtrTileG s Q s_qk_h DK BT BK r) maskT otherT
        c10pq (by rw [evalOp_ref]; exact c10mask) (hother c10)]
    rw [bwdLoad_gated s c10 Q s_qk_h DK BT BK r
        (fun i => by rw [c10read Q i hQ_DKInter hQ_DQInter]; exact rdQ i)]
  rw [stepStmts.cons_some d13]
  set c11 := c10.setReg "_q" .real [BK]
    (bwdGatedG s DK BK (bwdLdRowG s Q s_qk_h DK BT BK r)) with hc11
  
  have c11mask : c11.regs .bool [BK] "mask" = some maskT := by
    rw [hc11, BlockState.setReg_ne_name (h := by decide)]; exact c10mask
  have c11pk : c11.regs .ptr [BK] "p_k" = some (bwdPtrTileG s K s_qk_h DK BT BK r) := by
    rw [hc11, BlockState.setReg_ne_name (h := by decide),
        c10peel .ptr [BK] "p_k" (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3pk
  have c11rdK : ∀ i : Fin BK, c11.readMem K (offset s s_qk_h DK r BT BK i)
      = s.readMem K (offset s s_qk_h DK r BT BK i) := by
    intro i; rw [hc11, BlockState.setReg_readMem, c10read K i hK_DKInter hK_DQInter]; exact rdK i
  have d14 : stepStmt (Stmt.assign .real [BK] "_k"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_k"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK])))) c11
      = some (c11.setReg "_k" .real [BK]
          (bwdGatedG s DK BK (bwdLdRowG s K s_qk_h DK BT BK r))) := by
    rw [bwdEval_assign_load_ptr_maskOther c11 "_k" "p_k" _ _
        (bwdPtrTileG s K s_qk_h DK BT BK r) maskT otherT
        c11pk (by rw [evalOp_ref]; exact c11mask) (hother c11)]
    rw [bwdLoad_gated s c11 K s_qk_h DK BT BK r c11rdK]
  rw [stepStmts.cons_some d14]
  set c12 := c11.setReg "_k" .real [BK]
    (bwdGatedG s DK BK (bwdLdRowG s K s_qk_h DK BT BK r)) with hc12
  
  have c12dq : c12.regs .real [BK] "_dq" = some
      (bwdGatedG s DK BK (bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r)) := by
    rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10, bwdScatterRow_regs, hc9,
        BlockState.setReg_ne_name (h := by decide), hc8,
        BlockState.setReg_ne_name (h := by decide), hc7,
        BlockState.setReg_ne_name (h := by decide), hc6,
        BlockState.setReg_ne_name (h := by decide), hc5, bwdScatterRow_regs, hc4,
        BlockState.setReg_same]
  have c12dk : c12.regs .real [BK] "_dk" = some
      (bwdGatedG s DK BK (bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF)) := by
    rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10, bwdScatterRow_regs, hc9,
        BlockState.setReg_same]
  have c12qval : c12.regs .real [BK] "_q" = some
      (bwdGatedG s DK BK (bwdLdRowG s Q s_qk_h DK BT BK r)) := by
    rw [hc12, BlockState.setReg_ne_name (h := by decide), hc11, BlockState.setReg_same]
  have c12kval : c12.regs .real [BK] "_k" = some
      (bwdGatedG s DK BK (bwdLdRowG s K s_qk_h DK BT BK r)) := by
    rw [hc12]; exact BlockState.setReg_same _ _ _ _ _
  
  have hmul_dq : evalOp (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "_dq")
      (Op.ref .real [BK] "_q")) c12
      = some (bwdGatedG s DK BK (fun i => bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r i *
          bwdLdRowG s Q s_qk_h DK BT BK r i)) := by
    rw [bwdEval_mul c12 _ _ (bwdGatedG s DK BK (bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r))
        (bwdGatedG s DK BK (bwdLdRowG s Q s_qk_h DK BT BK r))
        (by rw [evalOp_ref]; exact c12dq) (by rw [evalOp_ref]; exact c12qval)]
    rw [bwdGatedG_mul]
  have hmul_dk : evalOp (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "_dk")
      (Op.ref .real [BK] "_k")) c12
      = some (bwdGatedG s DK BK (fun i => bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF i *
          bwdLdRowG s K s_qk_h DK BT BK r i)) := by
    rw [bwdEval_mul c12 _ _ (bwdGatedG s DK BK (bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF))
        (bwdGatedG s DK BK (bwdLdRowG s K s_qk_h DK BT BK r))
        (by rw [evalOp_ref]; exact c12dk) (by rw [evalOp_ref]; exact c12kval)]
    rw [bwdGatedG_mul]
  have d15 : stepStmt (Stmt.assign .real [BK] "_dg"
      (Op.sub .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "_dq")
          (Op.ref .real [BK] "_q"))
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "_dk")
          (Op.ref .real [BK] "_k")))) c12
      = some (c12.setReg "_dg" .real [BK]
          (bwdGatedG s DK BK (bwdDgSumG s DQInner DQInter DKInner DKInter Q K G
            s_qk_h DK BT BK r lgF))) := by
    apply stepStmt_assign_eq_some
    rw [bwdEval_sub c12 _ _
        (bwdGatedG s DK BK (fun i => bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r i *
          bwdLdRowG s Q s_qk_h DK BT BK r i))
        (bwdGatedG s DK BK (fun i => bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF i *
          bwdLdRowG s K s_qk_h DK BT BK r i))
        hmul_dq hmul_dk]
    rw [bwdGatedG_sub]; rfl
  rw [stepStmts.cons_some d15]
  set c13 := c12.setReg "_dg" .real [BK]
    (bwdGatedG s DK BK (bwdDgSumG s DQInner DQInter DKInner DKInter Q K G
      s_qk_h DK BT BK r lgF)) with hc13
  have c13cum : c13.regs .real [BK] "cum_grad_dg" = some (bwdGatedG s DK BK cumF) := by
    rw [hc13, BlockState.setReg_ne_name (h := by decide), hc12,
        BlockState.setReg_ne_name (h := by decide), hc11,
        BlockState.setReg_ne_name (h := by decide), hc10, bwdScatterRow_regs, hc9,
        BlockState.setReg_ne_name (h := by decide), hc8,
        BlockState.setReg_ne_name (h := by decide), hc7,
        BlockState.setReg_ne_name (h := by decide), hc6,
        BlockState.setReg_ne_name (h := by decide), hc5, bwdScatterRow_regs, hc4,
        BlockState.setReg_ne_name (h := by decide), hc3,
        BlockState.setReg_ne_name (h := by decide), hc2,
        BlockState.setReg_ne_name (h := by decide), hc1,
        BlockState.setReg_ne_name (h := by decide)]; exact hs3cum
  
  have d16 : stepStmt (Stmt.assign .real [BK] "cum_grad_dg"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [BK] "cum_grad_dg")
        (Op.ref .real [BK] "_dg"))) c13
      = some (c13.setReg "cum_grad_dg" .real [BK]
          (bwdGatedG s DK BK (fun i => cumF i +
            bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK r lgF i))) := by
    apply stepStmt_assign_eq_some
    rw [bwdEval_add c13 _ _ (bwdGatedG s DK BK cumF)
        (bwdGatedG s DK BK (bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK r lgF))
        (by rw [evalOp_ref]; exact c13cum)
        (by rw [evalOp_ref, hc13, BlockState.setReg_same])]
    rw [bwdGatedG_add]
  rw [stepStmts.cons_some d16]
  set c14 := c13.setReg "cum_grad_dg" .real [BK]
    (bwdGatedG s DK BK (fun i => cumF i +
      bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK r lgF i)) with hc14
  
  have hcumNext : (fun i => cumF i +
      bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK r lgF i)
      = bwdCumPartialG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK (m + 1) := by
    funext i
    rw [bwdCumPartialG_succ s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK m hm i]
  have c14peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "cum_grad_dg" → n ≠ "_dg" → n ≠ "_k" → n ≠ "_q" →
      n ≠ "_dk" → n ≠ "_dk2" → n ≠ "_dk1" → n ≠ "_dq" → n ≠ "_dq2" → n ≠ "_dq1" →
      c14.regs d sh n = s3.regs d sh n := by
    intro d sh n g0 g1 g2 g3 g4 g5 g6 g7 g8 g9
    rw [hc14, BlockState.setReg_ne_name (h := g0), hc13,
        BlockState.setReg_ne_name (h := g1), hc12,
        BlockState.setReg_ne_name (h := g2), hc11,
        BlockState.setReg_ne_name (h := g3), hc10, bwdScatterRow_regs, hc9,
        BlockState.setReg_ne_name (h := g4), hc8,
        BlockState.setReg_ne_name (h := g5), hc7,
        BlockState.setReg_ne_name (h := g5), hc6,
        BlockState.setReg_ne_name (h := g6), hc5, bwdScatterRow_regs, hc4,
        BlockState.setReg_ne_name (h := g7), hc3,
        BlockState.setReg_ne_name (h := g8), hc2,
        BlockState.setReg_ne_name (h := g8), hc1,
        BlockState.setReg_ne_name (h := g9)]
  
  have c14mask : c14.regs .bool [BK] "mask" = some maskT := by
    rw [c14peel .bool [BK] "mask" (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3mask
  have c14pdg : c14.regs .ptr [BK] "p_dg" = some (bwdPtrTileG s DG s_qk_h DK BT BK r) := by
    rw [c14peel .ptr [BK] "p_dg" (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3pdg
  have c14cum : c14.regs .real [BK] "cum_grad_dg" = some
      (bwdGatedG s DK BK (fun i => cumF i +
        bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK r lgF i)) := by
    rw [hc14]; exact BlockState.setReg_same _ _ _ _ _
  have d17 : stepStmt (Stmt.store .real [BK] (MemAccess.ptr (Op.ref .ptr [BK] "p_dg"))
      (Op.ref .real [BK] "cum_grad_dg") (MaskOpt.mask (Op.ref .bool [BK] "mask"))) c14
      = some (bwdScatterRow s DG s_qk_h DK BT BK r
          (fun i => cumF i +
            bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK r lgF i) c14) := by
    rw [bwdEval_store_ptr_masked c14 "p_dg" _ _
        (bwdPtrTileG s DG s_qk_h DK BT BK r)
        (bwdGatedG s DK BK (fun i => cumF i +
          bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK r lgF i)) maskT
        c14pdg (by rw [evalOp_ref]; exact c14cum) (by rw [evalOp_ref]; exact c14mask)]
    rw [bwdStore_scatter]; rfl
  rw [stepStmts.cons_some d17]
  set c15 := bwdScatterRow s DG s_qk_h DK BT BK r
    (fun i => cumF i +
      bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK r lgF i) c14 with hc15
  
  have c15peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "cum_grad_dg" → n ≠ "_dg" → n ≠ "_k" → n ≠ "_q" →
      n ≠ "_dk" → n ≠ "_dk2" → n ≠ "_dk1" → n ≠ "_dq" → n ≠ "_dq2" → n ≠ "_dq1" →
      c15.regs d sh n = s3.regs d sh n := by
    intro d sh n g0 g1 g2 g3 g4 g5 g6 g7 g8 g9
    rw [hc15, bwdScatterRow_regs]; exact c14peel d sh n g0 g1 g2 g3 g4 g5 g6 g7 g8 g9
  have c15p : ∀ (region : RegionName) (n : RegName),
      s3.regs .ptr [BK] n = some (bwdPtrTileG s region s_qk_h DK BT BK r) →
      n ≠ "cum_grad_dg" → n ≠ "_dg" → n ≠ "_k" → n ≠ "_q" →
      n ≠ "_dk" → n ≠ "_dk2" → n ≠ "_dk1" → n ≠ "_dq" → n ≠ "_dq2" → n ≠ "_dq1" →
      c15.regs .ptr [BK] n = some (bwdPtrTileG s region s_qk_h DK BT BK r) := by
    intro region n hn g0 g1 g2 g3 g4 g5 g6 g7 g8 g9
    rw [c15peel .ptr [BK] n g0 g1 g2 g3 g4 g5 g6 g7 g8 g9]; exact hn
  have c15pg := c15p G "p_g" hs3pg (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pk := c15p K "p_k" hs3pk (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pq := c15p Q "p_q" hs3pq (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pdqi := c15p DQInner "p_dq_inner" hs3pdqi (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pdki := c15p DKInner "p_dk_inner" hs3pdki (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pdqt := c15p DQInter "p_dq_inter" hs3pdqt (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pdkt := c15p DKInter "p_dk_inter" hs3pdkt (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  have c15pdg := c15p DG "p_dg" hs3pdg (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  
  have hdec : ∀ region : RegionName,
      (⟨fun idx => (((bwdPtrTileG s region s_qk_h DK BT BK r).data idx).1,
          ((bwdPtrTileG s region s_qk_h DK BT BK r).data idx).2 - DK)⟩ : Tile .ptr [BK])
        = bwdPtrDecTileG s region s_qk_h DK BT BK (m + 1) := by
    intro region
    rw [hr]; simp only [bwdPtrTileG]
    rw [← bwdPtrTileG_dec_succ s region s_qk_h DK BT BK m hm]
  
  have d18 : stepStmt (Stmt.assign .ptr [BK] "p_g"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_g") (Op.constNat DK))) c15
      = some (c15.setReg "p_g" .ptr [BK] (bwdPtrDecTileG s G s_qk_h DK BT BK (m + 1))) := by
    rw [bwdEval_assign_ptrSub c15 "p_g" "p_g" DK _ c15pg, hdec]
  rw [stepStmts.cons_some d18]
  set c16 := c15.setReg "p_g" .ptr [BK] (bwdPtrDecTileG s G s_qk_h DK BT BK (m + 1)) with hc16
  have d19 : stepStmt (Stmt.assign .ptr [BK] "p_k"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_k") (Op.constNat DK))) c16
      = some (c16.setReg "p_k" .ptr [BK] (bwdPtrDecTileG s K s_qk_h DK BT BK (m + 1))) := by
    rw [bwdEval_assign_ptrSub c16 "p_k" "p_k" DK _
        (by rw [hc16, BlockState.setReg_ne_name (h := by decide)]; exact c15pk), hdec]
  rw [stepStmts.cons_some d19]
  set c17 := c16.setReg "p_k" .ptr [BK] (bwdPtrDecTileG s K s_qk_h DK BT BK (m + 1)) with hc17
  have d20 : stepStmt (Stmt.assign .ptr [BK] "p_q"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_q") (Op.constNat DK))) c17
      = some (c17.setReg "p_q" .ptr [BK] (bwdPtrDecTileG s Q s_qk_h DK BT BK (m + 1))) := by
    rw [bwdEval_assign_ptrSub c17 "p_q" "p_q" DK _
        (by rw [hc17, BlockState.setReg_ne_name (h := by decide), hc16,
            BlockState.setReg_ne_name (h := by decide)]; exact c15pq), hdec]
  rw [stepStmts.cons_some d20]
  set c18 := c17.setReg "p_q" .ptr [BK] (bwdPtrDecTileG s Q s_qk_h DK BT BK (m + 1)) with hc18
  have d21 : stepStmt (Stmt.assign .ptr [BK] "p_dq_inner"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dq_inner") (Op.constNat DK))) c18
      = some (c18.setReg "p_dq_inner" .ptr [BK] (bwdPtrDecTileG s DQInner s_qk_h DK BT BK (m + 1))) := by
    rw [bwdEval_assign_ptrSub c18 "p_dq_inner" "p_dq_inner" DK _
        (by rw [hc18, BlockState.setReg_ne_name (h := by decide), hc17,
            BlockState.setReg_ne_name (h := by decide), hc16,
            BlockState.setReg_ne_name (h := by decide)]; exact c15pdqi), hdec]
  rw [stepStmts.cons_some d21]
  set c19 := c18.setReg "p_dq_inner" .ptr [BK] (bwdPtrDecTileG s DQInner s_qk_h DK BT BK (m + 1)) with hc19
  have d22 : stepStmt (Stmt.assign .ptr [BK] "p_dk_inner"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dk_inner") (Op.constNat DK))) c19
      = some (c19.setReg "p_dk_inner" .ptr [BK] (bwdPtrDecTileG s DKInner s_qk_h DK BT BK (m + 1))) := by
    rw [bwdEval_assign_ptrSub c19 "p_dk_inner" "p_dk_inner" DK _
        (by rw [hc19, BlockState.setReg_ne_name (h := by decide), hc18,
            BlockState.setReg_ne_name (h := by decide), hc17,
            BlockState.setReg_ne_name (h := by decide), hc16,
            BlockState.setReg_ne_name (h := by decide)]; exact c15pdki), hdec]
  rw [stepStmts.cons_some d22]
  set c20 := c19.setReg "p_dk_inner" .ptr [BK] (bwdPtrDecTileG s DKInner s_qk_h DK BT BK (m + 1)) with hc20
  have d23 : stepStmt (Stmt.assign .ptr [BK] "p_dq_inter"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dq_inter") (Op.constNat DK))) c20
      = some (c20.setReg "p_dq_inter" .ptr [BK] (bwdPtrDecTileG s DQInter s_qk_h DK BT BK (m + 1))) := by
    rw [bwdEval_assign_ptrSub c20 "p_dq_inter" "p_dq_inter" DK _
        (by rw [hc20, BlockState.setReg_ne_name (h := by decide), hc19,
            BlockState.setReg_ne_name (h := by decide), hc18,
            BlockState.setReg_ne_name (h := by decide), hc17,
            BlockState.setReg_ne_name (h := by decide), hc16,
            BlockState.setReg_ne_name (h := by decide)]; exact c15pdqt), hdec]
  rw [stepStmts.cons_some d23]
  set c21 := c20.setReg "p_dq_inter" .ptr [BK] (bwdPtrDecTileG s DQInter s_qk_h DK BT BK (m + 1)) with hc21
  have d24 : stepStmt (Stmt.assign .ptr [BK] "p_dk_inter"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dk_inter") (Op.constNat DK))) c21
      = some (c21.setReg "p_dk_inter" .ptr [BK] (bwdPtrDecTileG s DKInter s_qk_h DK BT BK (m + 1))) := by
    rw [bwdEval_assign_ptrSub c21 "p_dk_inter" "p_dk_inter" DK _
        (by rw [hc21, BlockState.setReg_ne_name (h := by decide), hc20,
            BlockState.setReg_ne_name (h := by decide), hc19,
            BlockState.setReg_ne_name (h := by decide), hc18,
            BlockState.setReg_ne_name (h := by decide), hc17,
            BlockState.setReg_ne_name (h := by decide), hc16,
            BlockState.setReg_ne_name (h := by decide)]; exact c15pdkt), hdec]
  rw [stepStmts.cons_some d24]
  set c22 := c21.setReg "p_dk_inter" .ptr [BK] (bwdPtrDecTileG s DKInter s_qk_h DK BT BK (m + 1)) with hc22
  have d25 : stepStmt (Stmt.assign .ptr [BK] "p_dg"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dg") (Op.constNat DK))) c22
      = some (c22.setReg "p_dg" .ptr [BK] (bwdPtrDecTileG s DG s_qk_h DK BT BK (m + 1))) := by
    rw [bwdEval_assign_ptrSub c22 "p_dg" "p_dg" DK _
        (by rw [hc22, BlockState.setReg_ne_name (h := by decide), hc21,
            BlockState.setReg_ne_name (h := by decide), hc20,
            BlockState.setReg_ne_name (h := by decide), hc19,
            BlockState.setReg_ne_name (h := by decide), hc18,
            BlockState.setReg_ne_name (h := by decide), hc17,
            BlockState.setReg_ne_name (h := by decide), hc16,
            BlockState.setReg_ne_name (h := by decide)]; exact c15pdg), hdec]
  rw [stepStmts.cons_some d25]
  set c23 := c22.setReg "p_dg" .ptr [BK] (bwdPtrDecTileG s DG s_qk_h DK BT BK (m + 1)) with hc23
  rw [stepStmts.nil]

  have hc23mem : ∀ region a, c23.readMem region a = c15.readMem region a := by
    intro region a
    simp only [hc23, hc22, hc21, hc20, hc19, hc18, hc17, hc16,
      BlockState.setReg_readMem]
  
  have hc14mem : ∀ region a, c14.readMem region a = c10.readMem region a := by
    intro region a
    simp only [hc14, hc13, hc12, hc11, BlockState.setReg_readMem]
  
  have hc9mem : ∀ region a, c9.readMem region a = c5.readMem region a := by
    intro region a
    simp only [hc9, hc8, hc7, hc6, BlockState.setReg_readMem]
  
  have hc4mem : ∀ region a, c4.readMem region a = s3.readMem region a := by
    intro region a
    simp only [hc4, hc3, hc2, hc1, BlockState.setReg_readMem]
  
  have hc23pid : c23.pids = s.pids := by
    simp only [hc23, hc22, hc21, hc20, hc19, hc18, hc17, hc16, BlockState.setReg_pids,
      hc15, bwdScatterRow_pids, hc14, hc13, hc12, hc11, hc10, bwdScatterRow_pids,
      hc9, hc8, hc7, hc6, hc5, bwdScatterRow_pids, hc4, hc3, hc2, hc1]
    exact hs3peelmem.1

  have c23read_other : ∀ (region : RegionName) (a : Nat),
      region ≠ DQInter → region ≠ DKInter → region ≠ DG →
      c23.readMem region a = sc.readMem region a := by
    intro region a h1 h2 h3
    rw [hc23mem, hc15, bwdScatterRow_other_region s DG region s_qk_h DK BT BK r _ c14 _ h3,
        hc14mem, hc10, bwdScatterRow_other_region s DKInter region s_qk_h DK BT BK r _ c9 _ h2,
        hc9mem, hc5, bwdScatterRow_other_region s DQInter region s_qk_h DK BT BK r _ c4 _ h1,
        hc4mem]
    simp only [BlockState.readMem, hs3peelmem.2]
  
  have c23peel : ∀ (d : TileDType) (sh : TileShape) (n : RegName),
      n ≠ "p_dg" → n ≠ "p_dk_inter" → n ≠ "p_dq_inter" → n ≠ "p_dk_inner" →
      n ≠ "p_dq_inner" → n ≠ "p_q" → n ≠ "p_k" → n ≠ "p_g" →
      n ≠ "cum_grad_dg" → n ≠ "_dg" → n ≠ "_k" → n ≠ "_q" →
      n ≠ "_dk" → n ≠ "_dk2" → n ≠ "_dk1" → n ≠ "_dq" → n ≠ "_dq2" → n ≠ "_dq1" →
      c23.regs d sh n = s3.regs d sh n := by
    intro d sh n p0 p1 p2 p3 p4 p5 p6 p7 g0 g1 g2 g3 g4 g5 g6 g7 g8 g9
    rw [hc23, BlockState.setReg_ne_name (h := p0), hc22,
        BlockState.setReg_ne_name (h := p1), hc21,
        BlockState.setReg_ne_name (h := p2), hc20,
        BlockState.setReg_ne_name (h := p3), hc19,
        BlockState.setReg_ne_name (h := p4), hc18,
        BlockState.setReg_ne_name (h := p5), hc17,
        BlockState.setReg_ne_name (h := p6), hc16,
        BlockState.setReg_ne_name (h := p7), hc15, bwdScatterRow_regs]
    exact c14peel d sh n g0 g1 g2 g3 g4 g5 g6 g7 g8 g9

  have hDGr : ∀ i : Fin BK, c23.readMem DG (offset s s_qk_h DK r BT BK i)
      = if active s DK BK i then
          bwdCumPartialG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK (m + 1) i
        else s.readMem DG (offset s s_qk_h DK r BT BK i) := by
    intro i
    rw [hc23mem, hc15, bwdScatterRow_readback]
    by_cases ha : active s DK BK i
    · simp only [ha, if_true]
      rw [bwdCumPartialG_succ s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK m hm i]
    · simp only [ha, if_false]
      rw [hc14mem, hc10, bwdScatterRow_other_region s DKInter DG s_qk_h DK BT BK r _ c9 _
            (Ne.symm hDKInter_DG),
          hc9mem, hc5, bwdScatterRow_other_region s DQInter DG s_qk_h DK BT BK r _ c4 _
            (Ne.symm hDQInter_DG), hc4mem, hs3mem]
      exact hDGunt r hrlt i
  
  have hDQtr : ∀ i : Fin BK, c23.readMem DQInter (offset s s_qk_h DK r BT BK i)
      = if active s DK BK i then bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK r i
        else s.readMem DQInter (offset s s_qk_h DK r BT BK i) := by
    intro i
    rw [hc23mem, hc15, bwdScatterRow_other_region s DG DQInter s_qk_h DK BT BK r _ c14 _ hDQInter_DG,
        hc14mem, hc10, bwdScatterRow_other_region s DKInter DQInter s_qk_h DK BT BK r _ c9 _
          hDQInter_DKInter,
        hc9mem, hc5, bwdScatterRow_readback]
    by_cases ha : active s DK BK i
    · simp only [ha, if_true]
    · simp only [ha, if_false]; rw [hc4mem, hs3mem]
      exact hDQIunt r hrlt i
  
  have hDKtr : ∀ i : Fin BK, c23.readMem DKInter (offset s s_qk_h DK r BT BK i)
      = if active s DK BK i then
          bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK r lgF i
        else s.readMem DKInter (offset s s_qk_h DK r BT BK i) := by
    intro i
    rw [hc23mem, hc15, bwdScatterRow_other_region s DG DKInter s_qk_h DK BT BK r _ c14 _ hDKInter_DG,
        hc14mem, hc10, bwdScatterRow_readback]
    by_cases ha : active s DK BK i
    · simp only [ha, if_true]
    · simp only [ha, if_false]
      rw [hc9mem, hc5, bwdScatterRow_other_region s DQInter DKInter s_qk_h DK BT BK r _ c4 _
            hDKInter_DQInter, hc4mem, hs3mem]
      exact hDKIunt r hrlt i
  
  have hscatter_other_off : ∀ (region : RegionName) (r' : Nat) (i : Fin BK), r' ≠ r →
      c23.readMem region (offset s s_qk_h DK r' BT BK i)
        = sc.readMem region (offset s s_qk_h DK r' BT BK i) := by
    intro region r' i hr'
    have hoff : ∀ j : Fin BK, offset s s_qk_h DK r BT BK j ≠ offset s s_qk_h DK r' BT BK i := by
      intro j
      simp only [offset, baseOffset]
      have hjDK : (j : Nat) < DK := lt_of_lt_of_le j.isLt hBK
      have hiDK : (i : Nat) < DK := lt_of_lt_of_le i.isLt hBK
      have hrr' : r ≠ r' := fun h => hr' h.symm
      have hexpand : ∀ x : Nat, (s.pids 1 * BT + x) * DK = s.pids 1 * BT * DK + x * DK := by
        intro x; rw [Nat.add_mul]
      rw [hexpand r, hexpand r']
      
      rcases Nat.lt_or_ge r r' with hlt | hge
      · have : r * DK + DK ≤ r' * DK := by
          calc r * DK + DK = (r + 1) * DK := by rw [Nat.add_mul, one_mul]
            _ ≤ r' * DK := Nat.mul_le_mul_right DK hlt
        omega
      · have hgt : r' < r := lt_of_le_of_ne hge (Ne.symm hrr')
        have : r' * DK + DK ≤ r * DK := by
          calc r' * DK + DK = (r' + 1) * DK := by rw [Nat.add_mul, one_mul]
            _ ≤ r * DK := Nat.mul_le_mul_right DK hgt
        omega
    
    have hscatter1 : ∀ (X : RegionName) (f : Fin BK → ℝ) (c : BlockState),
        (bwdScatterRow s X s_qk_h DK BT BK r f c).readMem region
            (offset s s_qk_h DK r' BT BK i)
          = c.readMem region (offset s s_qk_h DK r' BT BK i) := by
      intro X f c
      by_cases hX : region = X
      · subst hX; exact bwdScatterRow_other_offset s region s_qk_h DK BT BK r f c _ hoff
      · exact bwdScatterRow_other_region s X region s_qk_h DK BT BK r f c _ hX
    rw [hc23mem, hc15, hscatter1 DG _ c14,
        hc14mem, hc10, hscatter1 DKInter _ c9,
        hc9mem, hc5, hscatter1 DQInter _ c4, hc4mem, hs3mem]
  refine ⟨c23, rfl, ?_⟩
  show bwdInvG DQInner DQInter DKInner DKInter Q K G DG s s_qk_h DK BT BK (m + 1) c23
  simp only [bwdInvG]
  refine ⟨hc23pid, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_⟩
  · 
    intro a; rw [c23read_other G a hG_DQInter hG_DKInter hG_DG]; exact hGrd a
  · intro a; rw [c23read_other Q a hQ_DQInter hQ_DKInter hQ_DG]; exact hQrd a
  · intro a; rw [c23read_other K a hK_DQInter hK_DKInter hK_DG]; exact hKrd a
  · intro a; rw [c23read_other DQInner a hDQInner_DQInter hDQInner_DKInter hDQInner_DG]; exact hDQird a
  · intro a; rw [c23read_other DKInner a hDKInner_DQInter hDKInner_DKInter hDKInner_DG]; exact hDKird a
  · 
    rw [c23peel .bool [BK] "mask" (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3mask
  · 
    rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21,
        BlockState.setReg_ne_name (h := by decide), hc20,
        BlockState.setReg_ne_name (h := by decide), hc19,
        BlockState.setReg_ne_name (h := by decide), hc18,
        BlockState.setReg_ne_name (h := by decide), hc17,
        BlockState.setReg_ne_name (h := by decide), hc16,
        BlockState.setReg_ne_name (h := by decide), hc15, bwdScatterRow_regs, hc14]
    rw [BlockState.setReg_same, hcumNext]
  · 
    rw [c23peel .real [BK] "last_g" (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    rw [hs3lastg]
    apply congrArg; apply congrArg; rw [hlgF]; funext i'; simp only [bwdLastgG]
    rw [if_pos (Nat.succ_pos m)]
  · rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21,
        BlockState.setReg_ne_name (h := by decide), hc20,
        BlockState.setReg_ne_name (h := by decide), hc19,
        BlockState.setReg_ne_name (h := by decide), hc18,
        BlockState.setReg_ne_name (h := by decide), hc17,
        BlockState.setReg_ne_name (h := by decide), hc16]
    exact BlockState.setReg_same _ _ _ _ _
  · rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21,
        BlockState.setReg_ne_name (h := by decide), hc20,
        BlockState.setReg_ne_name (h := by decide), hc19,
        BlockState.setReg_ne_name (h := by decide), hc18,
        BlockState.setReg_ne_name (h := by decide), hc17]
    exact BlockState.setReg_same _ _ _ _ _
  · rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21,
        BlockState.setReg_ne_name (h := by decide), hc20,
        BlockState.setReg_ne_name (h := by decide), hc19,
        BlockState.setReg_ne_name (h := by decide), hc18]
    exact BlockState.setReg_same _ _ _ _ _
  · rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21,
        BlockState.setReg_ne_name (h := by decide), hc20,
        BlockState.setReg_ne_name (h := by decide), hc19]
    exact BlockState.setReg_same _ _ _ _ _
  · rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21,
        BlockState.setReg_ne_name (h := by decide), hc20]
    exact BlockState.setReg_same _ _ _ _ _
  · rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22,
        BlockState.setReg_ne_name (h := by decide), hc21]
    exact BlockState.setReg_same _ _ _ _ _
  · rw [hc23, BlockState.setReg_ne_name (h := by decide), hc22]
    exact BlockState.setReg_same _ _ _ _ _
  · rw [hc23]; exact BlockState.setReg_same _ _ _ _ _
  · 
    intro r' hr' i
    rw [hscatter_other_off DQInter r' i (by rw [hr]; omega)]
    exact hDQIunt r' (by omega) i
  · 
    intro r' hr' i
    rw [hscatter_other_off DKInter r' i (by rw [hr]; omega)]
    exact hDKIunt r' (by omega) i
  · 
    intro r' hr' i
    rw [hscatter_other_off DG r' i (by rw [hr]; omega)]
    exact hDGunt r' (by omega) i
  · 
    intro r' hr'lo hr'hi i
    by_cases heq : r' = r
    · subst heq; rw [hDQtr i]
    · rw [hscatter_other_off DQInter r' i heq]
      have : BT - m ≤ r' := by simp only [hr] at heq; omega
      exact hDQIwr r' this hr'hi i
  · 
    intro r' hr'lo hr'hi i
    by_cases heq : r' = r
    · subst heq; rw [hDKtr i]
    · rw [hscatter_other_off DKInter r' i heq]
      have : BT - m ≤ r' := by simp only [hr] at heq; omega
      exact hDKIwr r' this hr'hi i
  · 
    intro r' hr'lo hr'hi i
    by_cases heq : r' = r
    · subst heq
      rw [hDGr i]
      have hBTr : BT - r = m + 1 := by rw [hr]; omega
      rw [hBTr]
    · rw [hscatter_other_off DG r' i heq]
      have : BT - m ≤ r' := by simp only [hr] at heq; omega
      exact hDGwr r' this hr'hi i

theorem bwdDqOutG_eq_closed (s : BlockState) (DQInner DQInter G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) :
    bwdDqOutG s DQInner DQInter G s_qk_h DK BT BK t_rel.val i
      = bwdDQInterClosed s DQInner DQInter G s_qk_h DK BT BK t_rel i := by
  simp only [bwdDqOutG, bwdDQInterClosed]

theorem bwdDkOutG_eq_closed (s : BlockState) (DKInner DKInter G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) :
    bwdDkOutG s DKInner DKInter G s_qk_h DK BT BK t_rel.val
        (fun i' => s.readMem G (offset s s_qk_h DK (BT - 1) BT BK i')) i
      = bwdDKInterClosed s DKInner DKInter G s_qk_h DK BT BK t_rel i := by
  simp only [bwdDkOutG, bwdDKInterClosed]

theorem bwdDgSumG_eq_summand (s : BlockState)
    (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) :
    bwdDgSumG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK t_rel.val
        (fun i' => s.readMem G (offset s s_qk_h DK (BT - 1) BT BK i')) i
      = bwdDGSummand s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK t_rel i := by
  simp only [bwdDgSumG, bwdDGSummand, bwdDqOutG_eq_closed, bwdDkOutG_eq_closed]

theorem bwdCumPartialG_eq_DGClosed (s : BlockState)
    (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s_qk_h DK BT BK : Nat) (t_rel : Fin BT) (i : Fin BK) :
    bwdCumPartialG s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK
        (BT - t_rel.val) i
      = bwdDGClosed s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK t_rel i := by
  simp only [bwdCumPartialG, bwdDGClosed]

  apply Finset.sum_nbij' (fun d => (⟨BT - t_rel.val - 1 - d.val, by omega⟩ : Fin (BT - t_rel.val)))
    (fun d => (⟨BT - t_rel.val - 1 - d.val, by omega⟩ : Fin (BT - t_rel.val)))
  · intro d _; simp
  · intro d _; simp
  · intro d _; ext; simp; omega
  · intro d _; ext; simp; omega
  · intro d _
    have hidx : BT - 1 - (BT - t_rel.val - 1 - d.val) = t_rel.val + d.val := by omega
    rw [show (⟨t_rel.val + (BT - t_rel.val - 1 - d.val), by omega⟩ : Fin BT)
          = (⟨BT - 1 - d.val, by omega⟩ : Fin BT) by ext; simp; omega]
    rw [← bwdDgSumG_eq_summand s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK
      ⟨BT - 1 - d.val, by omega⟩ i]

set_option maxHeartbeats 4000000 in

theorem bwd_loop_drive_general
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s : BlockState) (s_qk_h DK BT BK : Nat) (hBT : 0 < BT) (hBK : BK ≤ DK)
    (hDKInner_DQInter : DKInner ≠ DQInter) (hDKInter_DQInter : DKInter ≠ DQInter)
    (hQ_DQInter : Q ≠ DQInter) (hQ_DKInter : Q ≠ DKInter)
    (hK_DQInter : K ≠ DQInter) (hK_DKInter : K ≠ DKInter)
    (hDQInter_DKInter : DQInter ≠ DKInter)
    (hDQInter_DG : DQInter ≠ DG) (hDKInter_DG : DKInter ≠ DG)
    (hG_DQInter : G ≠ DQInter) (hG_DKInter : G ≠ DKInter) (hG_DG : G ≠ DG)
    (hQ_DG : Q ≠ DG) (hK_DG : K ≠ DG)
    (hDQInner_DQInter : DQInner ≠ DQInter) (hDQInner_DKInter : DQInner ≠ DKInter)
    (hDQInner_DG : DQInner ≠ DG)
    (hDKInner_DKInter : DKInner ≠ DKInter) (hDKInner_DG : DKInner ≠ DG) :
    ∃ final sfinal,
      exec (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
        s_qk_h DK BT BK) s = some sfinal ∧
      BT ≤ final ∧
      bwdInvG DQInner DQInter DKInner DKInter Q K G DG s s_qk_h DK BT BK final sfinal := by
  
  obtain ⟨s0, hpro, hsh⟩ :=
    bwd_prologue_eval_general DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK hBT s
  have hP0 : bwdInvG DQInner DQInter DKInner DKInter Q K G DG s s_qk_h DK BT BK 0 s0 :=
    bwdInvG_entry DQInner DQInter DKInner DKInter Q K G DG s s0 s_qk_h DK BT BK hsh
  
  obtain ⟨final, sfinal, hloop, hge, hPfinal⟩ :=
    VeriTile.Triton.forRangeDyn_inv (idx := "__rev_j") (start := 0) (stop := BT) (step := 1)
      (P := bwdInvG DQInner DQInter DKInner DKInter Q K G DG s s_qk_h DK BT BK)
      (s_init := s0)
      (evalOp_constNat 0 s0)
      (bwd_stopOp_eval s0 BT hBT)
      (evalOp_constNat 1 s0)
      one_ne_zero
      hP0
      (by
        intro i sc hi hPi
        exact bwd_decay_cumsum_step_general DQInner DQInter DKInner DKInter Q K G DG s
          s_qk_h DK BT BK hBK hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter
          hK_DQInter hK_DKInter hDQInter_DKInter hDQInter_DG hDKInter_DG
          hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG hDQInner_DQInter hDQInner_DKInter
          hDQInner_DG hDKInner_DKInter hDKInner_DG i hi sc hPi)
  refine ⟨final, sfinal, ?_, hge, hPfinal⟩
  
  rw [exec, bwd_body_decomp_general, stepStmts.append_some hpro,
    stepStmts.cons_some hloop, stepStmts.nil]

section GeneralTops

variable (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
  (s : BlockState) (s_qk_h DK BT BK : Nat) (t_rel : Fin BT)
  (hBT : 0 < BT) (hBK : BK ≤ DK)
  (hDKInner_DQInter : DKInner ≠ DQInter) (hDKInter_DQInter : DKInter ≠ DQInter)
  (hQ_DQInter : Q ≠ DQInter) (hQ_DKInter : Q ≠ DKInter)
  (hK_DQInter : K ≠ DQInter) (hK_DKInter : K ≠ DKInter)
  (hDQInter_DKInter : DQInter ≠ DKInter)
  (hDQInter_DG : DQInter ≠ DG) (hDKInter_DG : DKInter ≠ DG)
  (hG_DQInter : G ≠ DQInter) (hG_DKInter : G ≠ DKInter) (hG_DG : G ≠ DG)
  (hQ_DG : Q ≠ DG) (hK_DG : K ≠ DG)
  (hDQInner_DQInter : DQInner ≠ DQInter) (hDQInner_DKInter : DQInner ≠ DKInter)
  (hDQInner_DG : DQInner ≠ DG)
  (hDKInner_DKInter : DKInner ≠ DKInter) (hDKInner_DG : DKInner ≠ DG)

include hBT hBK hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter hK_DQInter
  hK_DKInter hDQInter_DKInter hDQInter_DG hDKInter_DG hG_DQInter hG_DKInter hG_DG
  hQ_DG hK_DG hDQInner_DQInter hDQInner_DKInter hDQInner_DG hDKInner_DKInter hDKInner_DG

set_option maxHeartbeats 1600000 in

theorem bwd_decay_cumsum_dq_inter_closed_compute_correct_general :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s DK BK)
        (fun i => (DQInter, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        bwdDQInterClosed s DQInner DQInter G s_qk_h DK BT BK t_rel i) := by
  obtain ⟨final, sfinal, hexec, hge, hPfinal⟩ :=
    bwd_loop_drive_general DQInner DQInter DKInner DKInter Q K G DG s s_qk_h DK BT BK hBT hBK
      hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter
      hDQInter_DKInter hDQInter_DG hDKInter_DG hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG
      hDQInner_DQInter hDQInner_DKInter hDQInner_DG hDKInner_DKInter hDKInner_DG
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, hDQIwr, _⟩ := hPfinal
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bwd_decay_global_cumsum_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro sa s' hExec hsa
  subst sa
  intro i hActive
  have hs' : s' = sfinal := by
    rw [exec] at hexec; rw [exec] at hExec; rw [hExec] at hexec; exact Option.some.inj hexec
  subst s'
  show sfinal.readMem DQInter (offset s s_qk_h DK t_rel.val BT BK i) = _
  rw [hDQIwr t_rel.val (by omega) t_rel.isLt i, if_pos hActive,
    bwdDqOutG_eq_closed s DQInner DQInter G s_qk_h DK BT BK t_rel i]

set_option maxHeartbeats 1600000 in
                                       
theorem bwd_decay_cumsum_dk_inter_closed_compute_correct_general :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s DK BK)
        (fun i => (DKInter, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        bwdDKInterClosed s DKInner DKInter G s_qk_h DK BT BK t_rel i) := by
  obtain ⟨final, sfinal, hexec, hge, hPfinal⟩ :=
    bwd_loop_drive_general DQInner DQInter DKInner DKInter Q K G DG s s_qk_h DK BT BK hBT hBK
      hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter
      hDQInter_DKInter hDQInter_DG hDKInter_DG hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG
      hDQInner_DQInter hDQInner_DKInter hDQInner_DG hDKInner_DKInter hDKInner_DG
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, hDKIwr, _⟩ := hPfinal
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bwd_decay_global_cumsum_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro sa s' hExec hsa
  subst sa
  intro i hActive
  have hs' : s' = sfinal := by
    rw [exec] at hexec; rw [exec] at hExec; rw [hExec] at hexec; exact Option.some.inj hexec
  subst s'
  show sfinal.readMem DKInter (offset s s_qk_h DK t_rel.val BT BK i) = _
  rw [hDKIwr t_rel.val (by omega) t_rel.isLt i, if_pos hActive,
    bwdDkOutG_eq_closed s DKInner DKInter G s_qk_h DK BT BK t_rel i]

set_option maxHeartbeats 1600000 in

theorem bwd_decay_cumsum_dg_closed_compute_correct_general :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s DK BK)
        (fun i => (DG, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        bwdDGClosed s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK t_rel i) := by
  obtain ⟨final, sfinal, hexec, hge, hPfinal⟩ :=
    bwd_loop_drive_general DQInner DQInter DKInner DKInter Q K G DG s s_qk_h DK BT BK hBT hBK
      hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter
      hDQInter_DKInter hDQInter_DG hDKInter_DG hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG
      hDQInner_DQInter hDQInner_DKInter hDQInner_DG hDKInner_DKInter hDKInner_DG
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, hDGwr⟩ := hPfinal
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [bwd_decay_global_cumsum_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro sa s' hExec hsa
  subst sa
  intro i hActive
  have hs' : s' = sfinal := by
    rw [exec] at hexec; rw [exec] at hExec; rw [hExec] at hexec; exact Option.some.inj hexec
  subst s'
  show sfinal.readMem DG (offset s s_qk_h DK t_rel.val BT BK i) = _
  rw [hDGwr t_rel.val (by omega) t_rel.isLt i, if_pos hActive,
    bwdCumPartialG_eq_DGClosed s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK t_rel i]

theorem decay_cumsum_backward_closed_output_summary_general :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s DK BK)
        (fun i => (DQInter, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        bwdDQInterClosed s DQInner DQInter G s_qk_h DK BT BK t_rel i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s DK BK)
        (fun i => (DKInter, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        bwdDKInterClosed s DKInner DKInter G s_qk_h DK BT BK t_rel i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf (active s DK BK)
        (fun i => (DG, offset s s_qk_h DK t_rel.val BT BK i)))
      (expected := fun i : Fin BK =>
        bwdDGClosed s DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK t_rel i)) := by
  refine ⟨?_, ?_, ?_⟩
  · exact bwd_decay_cumsum_dq_inter_closed_compute_correct_general DQInner DQInter DKInner DKInter
      Q K G DG s s_qk_h DK BT BK t_rel hBT hBK hDKInner_DQInter hDKInter_DQInter
      hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter hDQInter_DKInter hDQInter_DG hDKInter_DG
      hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG hDQInner_DQInter hDQInner_DKInter hDQInner_DG
      hDKInner_DKInter hDKInner_DG
  · exact bwd_decay_cumsum_dk_inter_closed_compute_correct_general DQInner DQInter DKInner DKInter
      Q K G DG s s_qk_h DK BT BK t_rel hBT hBK hDKInner_DQInter hDKInter_DQInter
      hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter hDQInter_DKInter hDQInter_DG hDKInter_DG
      hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG hDQInner_DQInter hDQInner_DKInter hDQInner_DG
      hDKInner_DKInter hDKInner_DG
  · exact bwd_decay_cumsum_dg_closed_compute_correct_general DQInner DQInter DKInner DKInter
      Q K G DG s s_qk_h DK BT BK t_rel hBT hBK hDKInner_DQInter hDKInter_DQInter
      hQ_DQInter hQ_DKInter hK_DQInter hK_DKInter hDQInter_DKInter hDQInter_DG hDKInter_DG
      hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG hDQInner_DQInter hDQInner_DKInter hDQInner_DG
      hDKInner_DKInter hDKInner_DG

end GeneralTops

end BwdAssembly

end Correct_without_Rounding

section IOFace

open scoped VeriTile.Triton.StreamEmitMasked3DKernelIO₁

set_option maxHeartbeats 4000000
set_option linter.unusedVariables false

private def fwdPrologue (G GO : RegionName) (s_qk_h BT BK DK : Nat) : List Stmt :=
  [Stmt.assign TileDType.nat [] "i_k" (Op.programId 0),
   Stmt.assign TileDType.nat [] "i_c" (Op.programId 1),
   Stmt.assign TileDType.nat [] "i_bh" (Op.programId 2),
   Stmt.assign TileDType.ptr [BK] "p_g"
     (Op.ptrAdd Broadcast.scalarL (Op.ptrBase G)
       (Op.add NumericDType.nat Broadcast.scalarL
         (Op.add NumericDType.nat Broadcast.nil
           (Op.add NumericDType.nat Broadcast.nil
             (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh")
               (Op.constNat s_qk_h))
             (Op.mul NumericDType.nat Broadcast.nil
               (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c")
                 (Op.constNat BT))
               (Op.constNat DK)))
           (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k")
             (Op.constNat BK)))
         (Op.arange BK))),
   Stmt.assign TileDType.ptr [BK] "p_go"
     (Op.ptrAdd Broadcast.scalarL (Op.ptrBase GO)
       (Op.add NumericDType.nat Broadcast.scalarL
         (Op.add NumericDType.nat Broadcast.nil
           (Op.add NumericDType.nat Broadcast.nil
             (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh")
               (Op.constNat s_qk_h))
             (Op.mul NumericDType.nat Broadcast.nil
               (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c")
                 (Op.constNat BT))
               (Op.constNat DK)))
           (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k")
             (Op.constNat BK)))
         (Op.arange BK))),
   Stmt.assign TileDType.real [BK] "cum_decay" (Op.full [BK] (Op.const 0)),
   Stmt.assign TileDType.bool [BK] "mask"
     (Op.lt ComparableDType.nat Broadcast.scalarR
       (Op.add NumericDType.nat Broadcast.scalarL
         (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k")
           (Op.constNat BK))
         (Op.arange BK))
       (Op.constNat DK))]

private theorem fwd_body_decomp (G GO : RegionName)
    (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ) (BT BK DK : Nat) :
    (fwd_decay_cumsum_surface G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK).toAlgKernel.body
      = fwdPrologue G GO s_qk_h BT BK DK
        ++ [Stmt.forRange "i" 0 BT 1 (fwdBody G GO s_qk_h BT BK DK)] := by
  rfl

def fwdDecayCumsumKernelIO (G GO : RegionName)
    (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ) (BT BK DK : Nat) :
    StreamEmitMasked3DKernelIO₁ where
  kernel := fwd_decay_cumsum_surface G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK
  inp1 := G
  out := GO
  T := BT
  B1 := BK
  C := BK
  read1 := fun p₀ p₁ p₂ t j => p₂ * s_qk_h + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  write := fun p₀ p₁ p₂ t j => p₂ * s_qk_h + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  mask1 := fun p₀ _ _ _ j => p₀ * BK + j.val < DK
  writeMask := fun p₀ _ _ _ j => p₀ * BK + j.val < DK

noncomputable def fwdDecayStreamSpec (BT BK : Nat)
    (gs : Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  1.44269504 * ∑ u : Fin (t.val + 1), gs (Fin.castLE t.isLt u) j

private theorem fwd_addr_eq (s : BlockState) (s_qk_h BT BK DK m : Nat) (i : Fin BK) :
    s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + i.val + m * DK
      = offset s s_qk_h DK m BT BK i := by
  simp only [offset, baseOffset]
  ring

private theorem fwdDecayStreamSpec_eq_closed (G : RegionName) (s₀ : BlockState)
    (s_qk_h BT BK DK : Nat) (xs : Fin BT → Fin BK → ℝ)
    (hx : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem G
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs t j)
    (t : Fin BT) (j : Fin BK) (hj : active s₀ DK BK j) :
    fwdDecayStreamSpec BT BK xs t j = fwdDecayClosed s₀ G s_qk_h DK BT BK t j := by
  unfold fwdDecayStreamSpec fwdDecayClosed
  congr 1
  refine Finset.sum_congr rfl fun u _ => ?_
  rw [← hx (Fin.castLE t.isLt u) j hj]
  congr 1
  simp only [offset, baseOffset, Fin.val_castLE]
  ring

private theorem Rcast_real_real (R : RoundingModel) :
    R.cast .real .real = FloatDType.cast .real .real := by
  funext v
  simp [RoundingModel.cast, FloatDType.cast]

private theorem fwdPrologue_castFree (R : RoundingModel) (G GO : RegionName)
    (s_qk_h BT BK DK : Nat) (t : BlockState) :
    stepStmtsR R (fwdPrologue G GO s_qk_h BT BK DK) t
      = stepStmts (fwdPrologue G GO s_qk_h BT BK DK) t := by
  simp only [fwdPrologue, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

private theorem fwdBody_castFree (R : RoundingModel) (G GO : RegionName)
    (s_qk_h BT BK DK : Nat) (t : BlockState) :
    stepStmtsR R (fwdBody G GO s_qk_h BT BK DK) t
      = stepStmts (fwdBody G GO s_qk_h BT BK DK) t := by
  simp only [fwdBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real, BlockState.writeMemTypedR]
  rfl

private theorem fwd_flattenOk (G GO : RegionName)
    (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ) (BT BK DK : Nat) :
    ((fwd_decay_cumsum_surface G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [fwd_body_decomp]
  simp [fwdPrologue, fwdBody, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

private theorem fwd_preLoop (G GO : RegionName) (s_qk_h BT BK DK : Nat)
    (s : BlockState) :
    ∃ s6, stepStmts (fwdPrologue G GO s_qk_h BT BK DK) s = some s6
      ∧ fwdInv G GO s s_qk_h DK BT BK 0 s6 ∧ s6.mem = s.mem := by
  set s_ik := s.setReg "i_k" .nat [] (Tile.scalar (s.pids 0)) with hs_ik
  set s_ic := s_ik.setReg "i_c" .nat [] (Tile.scalar (s.pids 1)) with hs_ic
  set s_ibh := s_ic.setReg "i_bh" .nat [] (Tile.scalar (s.pids 2)) with hs_ibh
  have hibh : s_ibh.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by simp [hs_ibh]
  have hic : s_ibh.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by
    rw [hs_ibh, hs_ic]; simp
  have hik : s_ibh.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by
    rw [hs_ibh, hs_ic, hs_ik]; simp
  
  have hpgeval : evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase G)
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.add NumericDType.nat Broadcast.nil
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
            (Op.mul NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
              (Op.constNat DK)))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
        (Op.arange BK))) s_ibh = some (fwdPgTile s G s_qk_h DK BT BK 0) := by
    simp only [evalOp, Option.bind, hibh, hic, hik, evalOp_constNat, evalOp_arange,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
      Tile.bop, Tile.scalar]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [Tile.ptrAdd_data, fwdPgTile, offset, baseOffset,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
    refine Prod.ext rfl ?_
    simp only [Tile.vec]
    ring
  set s_pg := s_ibh.setReg "p_g" .ptr [BK] (fwdPgTile s G s_qk_h DK BT BK 0) with hs_pg
  have hpgostep_ibh : s_pg.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by
    rw [hs_pg]; simp [hibh]
  have hpgostep_ic : s_pg.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by
    rw [hs_pg]; simp [hic]
  have hpgostep_ik : s_pg.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by
    rw [hs_pg]; simp [hik]
  
  have hpgoeval : evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase GO)
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.add NumericDType.nat Broadcast.nil
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
            (Op.mul NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
              (Op.constNat DK)))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
        (Op.arange BK))) s_pg = some (fwdPgoTile s GO s_qk_h DK BT BK 0) := by
    simp only [evalOp, Option.bind, hpgostep_ibh, hpgostep_ic, hpgostep_ik, evalOp_constNat,
      evalOp_arange, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
      Tile.bop, Tile.scalar]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [Tile.ptrAdd_data, fwdPgoTile, offset, baseOffset,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
    refine Prod.ext rfl ?_
    simp only [Tile.vec]
    ring
  set s_pgo := s_pg.setReg "p_go" .ptr [BK] (fwdPgoTile s GO s_qk_h DK BT BK 0) with hs_pgo
  
  have hcumeval : evalOp (Op.full [BK] (Op.const 0)) s_pgo
      = some (fwdCumTile s G s_qk_h DK BT BK 0) := by
    simp only [evalOp_full, evalOp_const, Option.bind]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [fwdCumTile, Tile.scalar, fwdPartial, Finset.univ_eq_empty, Finset.sum_empty,
      mul_zero]
    by_cases ha : active s DK BK i <;> simp [ha]
  set s_cum := s_pgo.setReg "cum_decay" .real [BK] (fwdCumTile s G s_qk_h DK BT BK 0) with hs_cum
  have hcum_ik : s_cum.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by
    rw [hs_cum, hs_pgo, hs_pg]; simp [hik]
  
  have hmaskeval : evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK))
        (Op.arange BK)) (Op.constNat DK)) s_cum = some (fwdMaskTile s DK BK) := by
    simp only [evalOp, Option.bind, hcum_ik, evalOp_constNat, evalOp_arange,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
      ComparableDType.lt, Tile.bop, Tile.cop, Tile.scalar]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [fwdMaskTile, active, elemIndex, Tile.vec,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
    rfl
  
  have hstepK : stepStmt (Stmt.assign TileDType.nat [] "i_k" (Op.programId 0)) s = some s_ik := by
    rw [hs_ik]; exact stepStmt_assign_eq_some (by simp [evalOp])
  have hstepC : stepStmt (Stmt.assign TileDType.nat [] "i_c" (Op.programId 1)) s_ik = some s_ic := by
    rw [hs_ic]; exact stepStmt_assign_eq_some (show evalOp (Op.programId 1) s_ik = _ from by
      simp [evalOp, hs_ik])
  have hstepB : stepStmt (Stmt.assign TileDType.nat [] "i_bh" (Op.programId 2)) s_ic = some s_ibh := by
    rw [hs_ibh]; exact stepStmt_assign_eq_some (show evalOp (Op.programId 2) s_ic = _ from by
      simp [evalOp, hs_ic, hs_ik])
  have hstepPg : stepStmt (Stmt.assign TileDType.ptr [BK] "p_g"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase G)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.arange BK)))) s_ibh = some s_pg := by
    rw [hs_pg]; exact stepStmt_assign_eq_some hpgeval
  have hstepPgo : stepStmt (Stmt.assign TileDType.ptr [BK] "p_go"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase GO)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.arange BK)))) s_pg = some s_pgo := by
    rw [hs_pgo]; exact stepStmt_assign_eq_some hpgoeval
  have hstepCum : stepStmt (Stmt.assign TileDType.real [BK] "cum_decay" (Op.full [BK] (Op.const 0)))
      s_pgo = some s_cum := by
    rw [hs_cum]; exact stepStmt_assign_eq_some hcumeval
  have hstepMask : stepStmt (Stmt.assign TileDType.bool [BK] "mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK))
          (Op.arange BK)) (Op.constNat DK))) s_cum
      = some (s_cum.setReg "mask" .bool [BK] (fwdMaskTile s DK BK)) :=
    stepStmt_assign_eq_some hmaskeval
  unfold fwdPrologue
  rw [stepStmts.cons_some hstepK, stepStmts.cons_some hstepC, stepStmts.cons_some hstepB,
    stepStmts.cons_some hstepPg, stepStmts.cons_some hstepPgo, stepStmts.cons_some hstepCum,
    stepStmts.cons_some hstepMask, stepStmts.nil]
  refine ⟨_, rfl, ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ?_⟩
  · 
    rw [hs_cum, hs_pgo, hs_pg, hs_ibh, hs_ic, hs_ik]
    simp only [BlockState.setReg_pids]
  · 
    intro a
    rw [hs_cum, hs_pgo, hs_pg, hs_ibh, hs_ic, hs_ik]
    simp only [BlockState.setReg_readMem]
  · 
    rw [BlockState.setReg_ne_name (h := by decide), hs_cum, BlockState.setReg_same]
  · 
    rw [BlockState.setReg_ne_name (h := by decide), hs_cum,
      BlockState.setReg_ne_name (h := by decide), hs_pgo,
      BlockState.setReg_ne_name (h := by decide), hs_pg, BlockState.setReg_same]
  · 
    rw [BlockState.setReg_ne_name (h := by decide), hs_cum,
      BlockState.setReg_ne_name (h := by decide), hs_pgo, BlockState.setReg_same]
  · 
    rw [BlockState.setReg_same]
  · 
    intro j hj i
    rw [hs_cum, hs_pgo, hs_pg, hs_ibh, hs_ic, hs_ik]
    simp only [BlockState.setReg_readMem]
  · 
    intro j hj i
    omega
  · 
    rw [hs_cum, hs_pgo, hs_pg, hs_ibh, hs_ic, hs_ik]
    rfl

private theorem stepStmts_cons_assign_inv {dt : TileDType} {sh : TileShape}
    {nm : RegName} {e : Op dt sh} {rest : List Stmt} {s s' : BlockState}
    (h : stepStmts (Stmt.assign dt sh nm e :: rest) s = some s') :
    ∃ v, evalOp e s = some v ∧ stepStmts rest (s.setReg nm dt sh v) = some s' := by
  cases hv : evalOp e s with
  | none => simp [stepStmts, stepStmt, hv] at h
  | some v =>
      rw [stepStmts.cons_some (stepStmt_assign_eq_some hv)] at h
      exact ⟨v, rfl, h⟩

private theorem foldl_writeMem_prop_preserve_cell {α : Type}
    {region : RegionName} (ofn : α → Nat) (vfn : α → ℝ)
    (P : α → Prop) [DecidablePred P]
    (r : RegionName) (oo : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(r = region ∧ oo = ofn k)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (ofn k) (vfn k) else acc) s).mem r oo
      = s.mem r oo := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hm : P hd
      · rw [if_pos hm,
          ih _ (fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk),
          BlockState.writeMem_mem]
        exact if_neg (hnot hd List.mem_cons_self hm)
      · rw [if_neg hm]
        exact ih _ fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk

private theorem fwd_body_step_frame (G GO : RegionName) (s_qk_h BT BK DK : Nat)
    (s sc s' : BlockState) (m : Nat)
    (hpgo : sc.regs .ptr [BK] "p_go" = some (fwdPgoTile s GO s_qk_h DK BT BK m))
    (hmask : sc.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK))
    (hstep : stepStmts (fwdBody G GO s_qk_h BT BK DK)
      (sc.setReg "i" .nat [] (Tile.scalar m)) = some s')
    (r : RegionName) (oo : Nat)
    (hcond : r ≠ GO ∨ ∀ i : Fin BK, active s DK BK i →
      oo ≠ offset s s_qk_h DK m BT BK i) :
    s'.mem r oo = sc.mem r oo := by
  set sloop := sc.setReg "i" .nat [] (Tile.scalar m) with hsloop
  unfold fwdBody at hstep
  obtain ⟨v1, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨v2, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  set s2 := (sloop.setReg "_g" .real [BK] v1).setReg "cum_decay" .real [BK] v2 with hs2
  have hcum2 : s2.regs .real [BK] "cum_decay" = some v2 := by
    rw [hs2]; exact BlockState.setReg_same _ _ _ _ _
  have hpgo2 : s2.regs .ptr [BK] "p_go" = some (fwdPgoTile s GO s_qk_h DK BT BK m) := by
    rw [hs2, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide), hsloop,
      BlockState.setReg_ne_name (h := by decide)]
    exact hpgo
  have hmask2 : s2.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
    rw [hs2, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide), hsloop,
      BlockState.setReg_ne_name (h := by decide)]
    exact hmask
  set vfun : TileIndex [BK] → ℝ := fun k => (v2.data k).unbotD 0 with hvfun
  set sst := (TileShape.allIndices [BK]).foldl
    (fun (acc : BlockState) k =>
      if active s DK BK k.1 then
        acc.writeMem GO (offset s s_qk_h DK m BT BK k.1) (vfun k)
      else acc) s2 with hsst
  have hstore : stepStmt (Stmt.store TileDType.real [BK]
      (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_go"))
      (Op.ref TileDType.real [BK] "cum_decay")
      (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask"))) s2 = some sst := by
    simp only [stepStmt, evalOp, Option.bind, Option.map, hcum2, hpgo2, hmask2]
    apply congrArg some
    rw [hsst]
    congr 1
    funext acc k
    simp only [fwdPgoTile, fwdMaskTile]
    by_cases ha : active s DK BK k.1
    · rw [if_pos (by simpa using ha), if_pos ha, BlockState.writeMemTyped_real]
      rfl
    · rw [if_neg (by simpa using ha), if_neg ha]
  rw [stepStmts.cons_some hstore] at hstep
  obtain ⟨v4, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨v5, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  rw [stepStmts.nil] at hstep
  obtain rfl := Option.some.inj hstep
  simp only [BlockState.setReg_mem]
  rw [hsst]
  refine Eq.trans (foldl_writeMem_prop_preserve_cell _ _ _ r oo _ _ ?_) ?_
  · intro k hk hmk hbad
    rcases hcond with hne' | hno
    · exact hne' hbad.1
    · exact hno k.1 hmk hbad.2
  · rw [hs2, hsloop]
    simp only [BlockState.setReg_mem]

private theorem fwd_bodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (G GO : RegionName) (s_qk_h BT BK DK : Nat)
    (s stt : BlockState) (m : Nat)
    (hpg : stt.regs .ptr [BK] "p_g" = some (fwdPgTile s G s_qk_h DK BT BK m))
    (hpgo : stt.regs .ptr [BK] "p_go" = some (fwdPgoTile s GO s_qk_h DK BT BK m))
    (hmask : stt.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK))
    (hbG : ∀ i : Fin BK, active s DK BK i → offset s s_qk_h DK m BT BK i < bounds G)
    (hbGO : ∀ i : Fin BK, active s DK BK i → offset s s_qk_h DK m BT BK i < bounds GO) :
    Stmt.TraceSafeListR R bounds (fwdBody G GO s_qk_h BT BK DK)
      (stt.setReg "i" .nat [] (Tile.scalar m)) := by
  set sloop := stt.setReg "i" .nat [] (Tile.scalar m) with hsloop
  have hpg' : sloop.regs .ptr [BK] "p_g" = some (fwdPgTile s G s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpg
  have hpgo' : sloop.regs .ptr [BK] "p_go" = some (fwdPgoTile s GO s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpgo
  have hmask' : sloop.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
    rw [hsloop]; simpa using hmask
  unfold fwdBody
  
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t1 ht1 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def], ?_⟩
    intro ptrs hptrs idx hactive
    rw [evalOpR_ref, hpg'] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [evalOpR_ref, hmask'] at hm
    obtain rfl := Option.some.inj hm
    have ha : active s DK BK idx.1 := by simpa [fwdMaskTile] using hmi
    simpa [fwdPgTile] using hbG idx.1 ha
  · obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv ht1
    
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t2 ht2 => ?_)
    obtain ⟨v2, -, rfl⟩ := stepStmtR_assign_inv ht2
    have hpgo3 : ((sloop.setReg "_g" .real [BK] v1).setReg
        "cum_decay" .real [BK] v2).regs .ptr [BK] "p_go"
        = some (fwdPgoTile s GO s_qk_h DK BT BK m) := by
      rw [BlockState.setReg_ne_name (h := by decide),
        BlockState.setReg_ne_name (h := by decide)]
      exact hpgo'
    have hmask3 : ((sloop.setReg "_g" .real [BK] v1).setReg
        "cum_decay" .real [BK] v2).regs .bool [BK] "mask"
        = some (fwdMaskTile s DK BK) := by
      rw [BlockState.setReg_ne_name (h := by decide),
        BlockState.setReg_ne_name (h := by decide)]
      exact hmask'
    
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun t3 ht3 => ?_)
    · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MemAccess.SafeAtR,
        MaskOpt.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
        memAccessActiveAddressSafeR]
      refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def],
        by simp [Op.SafeAtR.eq_def], ?_⟩
      intro ptrs hptrs idx hactive
      rw [evalOpR_ref, hpgo3] at hptrs
      obtain rfl := Option.some.inj hptrs
      obtain ⟨masks, hm, hmi⟩ := hactive
      rw [evalOpR_ref, hmask3] at hm
      obtain rfl := Option.some.inj hm
      have ha : active s DK BK idx.1 := by simpa [fwdMaskTile] using hmi
      simpa [fwdPgoTile] using hbGO idx.1 ha
    · 
      refine Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t4 ht4 => ?_)
      exact Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
        (fun _ _ => Stmt.TraceSafeListR.nil_intro)

private theorem fwd_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (G GO : RegionName) (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ)
    (BT BK DK : Nat) (hne : G ≠ GO) (hBK : BK ≤ DK) (s : BlockState)
    (hbG : ∀ (t : Fin BT) (j : Fin BK), s.pids 0 * BK + j.val < DK →
      s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + j.val + t.val * DK
        < bounds G)
    (hbGO : ∀ (t : Fin BT) (j : Fin BK), s.pids 0 * BK + j.val < DK →
      s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + j.val + t.val * DK
        < bounds GO) :
    ((fwd_decay_cumsum_surface G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK).toAlgKernel).TraceSafeR R bounds s := by
  
  have hbG' : ∀ m : Nat, m < BT → ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK m BT BK i < bounds G := by
    intro m hm i ha
    have h : s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + i.val + m * DK
        < bounds G := hbG ⟨m, hm⟩ i ha
    rwa [fwd_addr_eq] at h
  have hbGO' : ∀ m : Nat, m < BT → ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK m BT BK i < bounds GO := by
    intro m hm i ha
    have h : s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + i.val + m * DK
        < bounds GO := hbGO ⟨m, hm⟩ i ha
    rwa [fwd_addr_eq] at h
  unfold Kernel.TraceSafeR
  rw [fwd_body_decomp]
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · 
    refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro stmt hst s'
    simp only [fwdPrologue, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  · intro s1 hs1
    obtain ⟨s6, hsp, hP0, hmem6⟩ := fwd_preLoop G GO s_qk_h BT BK DK s
    rw [fwdPrologue_castFree R G GO s_qk_h BT BK DK s, hsp] at hs1
    obtain rfl := Option.some.inj hs1
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
    simp only [Stmt.TraceSafeR]
    refine Stmt.forRangeTraceSafeR_inv R bounds "i" BT 1
      (fwdBody G GO s_qk_h BT BK DK) (fwdInv G GO s s_qk_h DK BT BK) ?_ 0 s6 hP0
    intro c stt hc hP
    have hPd := hP
    obtain ⟨hpids, hG, hcum, hpg, hpgo, hmask, hUnt, hRead⟩ := hPd
    refine ⟨fwd_bodySafeR R bounds G GO s_qk_h BT BK DK s stt c hpg hpgo hmask
      (hbG' c hc) (hbGO' c hc), ?_⟩
    obtain ⟨st', hstep, hP'⟩ :=
      fwd_decay_cumsum_step G GO s s_qk_h BT BK DK hne hBK c stt hP
    exact ⟨st', by rw [fwdBody_castFree]; exact hstep, hP'⟩

private theorem fwd_runR (R : RoundingModel) (G GO : RegionName)
    (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ) (BT BK DK : Nat)
    (hne : G ≠ GO) (hBK : BK ≤ DK) (s₀ : BlockState) :
    ∃ sfin,
      execR R (fwd_decay_cumsum_surface G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK).toAlgKernel s₀
        = some sfin
      ∧ (∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
          sfin.readMem GO (offset s₀ s_qk_h DK t.val BT BK i)
            = fwdDecayClosed s₀ G s_qk_h DK BT BK t i)
      ∧ (∀ r oo, (r ≠ GO ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          sfin.mem r oo = s₀.mem r oo) := by
  obtain ⟨s6, hsp, hP0, hmem6⟩ := fwd_preLoop G GO s_qk_h BT BK DK s₀
  
  obtain ⟨final, sfinal, hLoop, hfinal, hPfinal⟩ :=
    forRange_inv (idx := "i") (start := 0) (stop := BT) (step := 1)
      (body := fwdBody G GO s_qk_h BT BK DK)
      (P := fun m stt => fwdInv G GO s₀ s_qk_h DK BT BK m stt
        ∧ ∀ r oo, (r ≠ GO ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          stt.mem r oo = s₀.mem r oo)
      Nat.one_ne_zero
      ⟨hP0, fun r oo _ => by rw [hmem6]⟩
      (fun m stt hm hP => by
        obtain ⟨st', hstep, hP'⟩ :=
          fwd_decay_cumsum_step G GO s₀ s_qk_h BT BK DK hne hBK m stt hP.1
        refine ⟨st', hstep, hP', ?_⟩
        intro r oo hcond
        obtain ⟨hpids, hG, hcum, hpg, hpgo, hmask, hUnt, hRead⟩ := hP.1
        rw [fwd_body_step_frame G GO s_qk_h BT BK DK s₀ stt st' m hpgo hmask hstep r oo ?_]
        · exact hP.2 r oo hcond
        · rcases hcond with hne' | hno
          · exact Or.inl hne'
          · exact Or.inr fun i ha => hno ⟨m, hm⟩ i ha)
  obtain ⟨⟨_, _, _, _, _, _, _, hreadGO⟩, hframe⟩ := hPfinal
  
  have hLoopR : stepStmtR R (Stmt.forRange "i" 0 BT 1 (fwdBody G GO s_qk_h BT BK DK)) s6
      = some sfinal := by
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _ (fwdBody_castFree R G GO s_qk_h BT BK DK) "i",
      ← stepForRangeAux.forRange_unfold]
    exact hLoop
  refine ⟨sfinal, ?_, ?_, ?_⟩
  · unfold execR
    rw [fwd_body_decomp,
      stepStmtsR_append R (fwdPrologue G GO s_qk_h BT BK DK) _ s₀,
      fwdPrologue_castFree R G GO s_qk_h BT BK DK s₀, hsp, Option.bind_some,
      stepStmtsR_cons_some hLoopR, stepStmtsR_nil]
  · intro t i ha
    have ht : t.val < final := lt_of_lt_of_le t.isLt hfinal
    rw [hreadGO t.val ht i, if_pos ha]
    simp only [fwdDecayClosed, fwdPartial]
  · exact hframe

theorem fwd_decay_cumsum_io_correctness (R : RoundingModel)
    (G GO : RegionName) (s_qk_h s_qk_t s_qk_d B H T : Nat) (scale : ℝ)
    (BT BK DK : Nat) (hne : G ≠ GO) (hBK : BK ≤ DK) :
    fwdDecayCumsumKernelIO G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK ⊨[R]
      fun _ _ _ gs t j => fwdDecayStreamSpec BT BK gs t j := by
  refine StreamEmitMasked3DKernelIO₁.ImplementsR.intro _ ?_ ?_ ?_
  · exact fwd_flattenOk G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK
  · 
    intro bounds s xs _hx hbr1 hbw
    simp only [fwdDecayCumsumKernelIO] at hbr1 hbw ⊢
    exact fwd_traceSafeR R bounds G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK
      hne hBK s hbr1 hbw
  · 
    intro s₀ xs _hundef hx
    simp only [fwdDecayCumsumKernelIO] at hx ⊢
    obtain ⟨sfin, hexec, hval, hframe⟩ :=
      fwd_runR R G GO s_qk_h s_qk_t s_qk_d B H T scale BT BK DK hne hBK s₀
    refine ⟨sfin, hexec, ?_, ?_⟩
    · intro t j hj
      have hval' : sfin.readMem GO (offset s₀ s_qk_h DK t.val BT BK j)
          = fwdDecayClosed s₀ G s_qk_h DK BT BK t j := hval t j hj
      rw [BlockState.readMemAs_real, fwd_addr_eq, hval',
        ← fwdDecayStreamSpec_eq_closed G s₀ s_qk_h BT BK DK xs hx t j hj]
      simp [FloatDType.ofReal]
    · intro r oo hcond
      refine hframe r oo ?_
      rcases hcond with hne' | hno
      · exact Or.inl hne'
      · exact Or.inr fun t i ha hoeq =>
          hno t i ha (hoeq.trans (fwd_addr_eq s₀ s_qk_h BT BK DK t.val i).symm)

end IOFace

section PrepareIOFace

open scoped VeriTile.Triton.StreamGroupedEmitMasked3DKernelIO

set_option maxHeartbeats 4000000
set_option linter.unusedVariables false

private def prepPre10 (Q K G QG KG : RegionName)
    (s_qk_h BT BK DK : Nat) : List Stmt :=
  [Stmt.assign TileDType.nat [] "i_k" (Op.programId 0),
   Stmt.assign TileDType.nat [] "i_c" (Op.programId 1),
   Stmt.assign TileDType.nat [] "i_bh" (Op.programId 2),
   Stmt.assign TileDType.nat [BK] "offs" (Op.arange BK),
   Stmt.assign TileDType.ptr [BK] "p_q"
     (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q)
       (Op.add NumericDType.nat Broadcast.scalarL
         (Op.add NumericDType.nat Broadcast.nil
           (Op.add NumericDType.nat Broadcast.nil
             (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh")
               (Op.constNat s_qk_h))
             (Op.mul NumericDType.nat Broadcast.nil
               (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c")
                 (Op.constNat BT))
               (Op.constNat DK)))
           (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k")
             (Op.constNat BK)))
         (Op.ref TileDType.nat [BK] "offs"))),
   Stmt.assign TileDType.ptr [BK] "p_g"
     (Op.ptrAdd Broadcast.scalarL (Op.ptrBase G)
       (Op.add NumericDType.nat Broadcast.scalarL
         (Op.add NumericDType.nat Broadcast.nil
           (Op.add NumericDType.nat Broadcast.nil
             (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh")
               (Op.constNat s_qk_h))
             (Op.mul NumericDType.nat Broadcast.nil
               (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c")
                 (Op.constNat BT))
               (Op.constNat DK)))
           (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k")
             (Op.constNat BK)))
         (Op.ref TileDType.nat [BK] "offs"))),
   Stmt.assign TileDType.ptr [BK] "p_k"
     (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K)
       (Op.add NumericDType.nat Broadcast.scalarL
         (Op.add NumericDType.nat Broadcast.nil
           (Op.add NumericDType.nat Broadcast.nil
             (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh")
               (Op.constNat s_qk_h))
             (Op.mul NumericDType.nat Broadcast.nil
               (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c")
                 (Op.constNat BT))
               (Op.constNat DK)))
           (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k")
             (Op.constNat BK)))
         (Op.ref TileDType.nat [BK] "offs"))),
   Stmt.assign TileDType.ptr [BK] "p_qg"
     (Op.ptrAdd Broadcast.scalarL (Op.ptrBase QG)
       (Op.add NumericDType.nat Broadcast.scalarL
         (Op.add NumericDType.nat Broadcast.nil
           (Op.add NumericDType.nat Broadcast.nil
             (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh")
               (Op.constNat s_qk_h))
             (Op.mul NumericDType.nat Broadcast.nil
               (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c")
                 (Op.constNat BT))
               (Op.constNat DK)))
           (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k")
             (Op.constNat BK)))
         (Op.ref TileDType.nat [BK] "offs"))),
   Stmt.assign TileDType.ptr [BK] "p_kg"
     (Op.ptrAdd Broadcast.scalarL (Op.ptrBase KG)
       (Op.add NumericDType.nat Broadcast.scalarL
         (Op.add NumericDType.nat Broadcast.nil
           (Op.add NumericDType.nat Broadcast.nil
             (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh")
               (Op.constNat s_qk_h))
             (Op.mul NumericDType.nat Broadcast.nil
               (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c")
                 (Op.constNat BT))
               (Op.constNat DK)))
           (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k")
             (Op.constNat BK)))
         (Op.ref TileDType.nat [BK] "offs"))),
   Stmt.assign TileDType.bool [BK] "mask"
     (Op.lt ComparableDType.nat Broadcast.scalarR
       (Op.add NumericDType.nat Broadcast.scalarL
         (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k")
           (Op.constNat BK))
         (Op.ref TileDType.nat [BK] "offs"))
       (Op.constNat DK))]

private def prepLdAddrOp (s_qk_h BT BK DK : Nat) : Op TileDType.nat [BK] :=
  Op.add NumericDType.nat Broadcast.scalarL
    (Op.add NumericDType.nat Broadcast.nil
      (Op.add NumericDType.nat Broadcast.nil
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh")
          (Op.constNat s_qk_h))
        (Op.mul NumericDType.nat Broadcast.nil
          (Op.sub NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c")
                (Op.constNat BT))
              (Op.constNat BT))
            (Op.constNat 1))
          (Op.constNat DK)))
      (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k")
        (Op.constNat BK)))
    (Op.ref TileDType.nat [BK] "offs")

private def prepLdStmt (G : RegionName) (s_qk_h BT BK DK : Nat) : Stmt :=
  Stmt.assign TileDType.real [BK] "last_decay"
    (Op.load TileDType.real
      (MemAccess.region G (prepLdAddrOp s_qk_h BT BK DK))
      MaskOpt.none)

private theorem prep_body_decomp (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) :
    (prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale).toAlgKernel.body
      = prepPre10 Q K G QG KG s_qk_h BT BK DK
        ++ (prepLdStmt G s_qk_h BT BK DK
            :: [Stmt.forRange "i" 0 BT 1 (prepBody Q K G QG KG s_qk_h DK BT BK scale)]) := by
  rfl

def prepareQgKgKernelIO (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) :
    StreamGroupedEmitMasked3DKernelIO where
  kernel := prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale
  nIn := 3
  nOut := 2
  bufs := [Q, K, G, QG, KG]
  inp := fun i => match i with
    | ⟨0, _⟩ => Q
    | ⟨1, _⟩ => K
    | ⟨2, _⟩ => G
    | ⟨n + 3, h⟩ => absurd h (by omega)
  out := fun o => match o with
    | ⟨0, _⟩ => QG
    | ⟨1, _⟩ => KG
    | ⟨n + 2, h⟩ => absurd h (by omega)
  T := BT
  B := BK
  read := fun _ p₀ p₁ p₂ t j =>
    p₂ * s_qk_h + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  readMask := fun i p₀ _ _ t j => match i with
    | ⟨0, _⟩ => p₀ * BK + j.val < DK
    | ⟨1, _⟩ => p₀ * BK + j.val < DK
    | ⟨2, _⟩ => p₀ * BK + j.val < DK ∨ t.val = BT - 1
    | ⟨n + 3, h⟩ => absurd h (by omega)
  write := fun _ p₀ p₁ p₂ t j =>
    p₂ * s_qk_h + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  writeMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK

noncomputable def prepStreamSpec (BT BK : Nat) (scale : ℝ)
    (xs : Fin 3 → Fin BT → Fin BK → ℝ) (o : Fin 2) (t : Fin BT) (j : Fin BK) : ℝ :=
  match o with
  | ⟨0, _⟩ =>
      xs (⟨0, by omega⟩ : Fin 3) t j
        * Real.exp (xs (⟨2, by omega⟩ : Fin 3) t j * Real.log 2) * scale
  | ⟨1, _⟩ =>
      if h : 0 < BT then
        xs (⟨1, by omega⟩ : Fin 3) t j
          * Real.exp ((xs (⟨2, by omega⟩ : Fin 3) ⟨BT - 1, Nat.sub_lt h Nat.one_pos⟩ j
              - xs (⟨2, by omega⟩ : Fin 3) t j) * Real.log 2)
      else 0
  | ⟨n + 2, h2⟩ => absurd h2 (by omega)

private theorem prep_ld_addr_eq (s : BlockState) (s_qk_h BT BK DK : Nat)
    (hBT : 0 < BT) (i : Fin BK) :
    s.pids 2 * s_qk_h + (s.pids 1 * BT + BT - 1) * DK + s.pids 0 * BK + i.val
      = s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + i.val
        + (BT - 1) * DK := by
  rw [show s.pids 1 * BT + BT - 1 = s.pids 1 * BT + (BT - 1) from by omega,
    Nat.add_mul]
  ring

private theorem prepStreamSpec_qg_eq_closed (Q G : RegionName) (s₀ : BlockState)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) (xs : Fin 3 → Fin BT → Fin BK → ℝ)
    (h0 : (0 : Nat) < 2)
    (hxQ : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem Q
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨0, by omega⟩ : Fin 3) t j)
    (hxG : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem G
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨2, by omega⟩ : Fin 3) t j)
    (t : Fin BT) (j : Fin BK) (hj : active s₀ DK BK j) :
    prepStreamSpec BT BK scale xs ⟨0, h0⟩ t j
      = prepareQgClosed s₀ Q G s_qk_h DK BT BK t j scale := by
  have hq := hxQ t j hj
  have hg := hxG t j hj
  rw [fwd_addr_eq] at hq hg
  show xs (⟨0, by omega⟩ : Fin 3) t j
      * Real.exp (xs (⟨2, by omega⟩ : Fin 3) t j * Real.log 2) * scale = _
  rw [← hq, ← hg]
  simp only [prepareQgClosed]

private theorem prepStreamSpec_kg_eq_closed (K G : RegionName) (s₀ : BlockState)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) (xs : Fin 3 → Fin BT → Fin BK → ℝ)
    (hBT : 0 < BT) (h1 : (1 : Nat) < 2)
    (hxK : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem K
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨1, by omega⟩ : Fin 3) t j)
    (hxG : ∀ (t : Fin BT) (j : Fin BK),
      (s₀.pids 0 * BK + j.val < DK ∨ t.val = BT - 1) →
      s₀.readMem G
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨2, by omega⟩ : Fin 3) t j)
    (t : Fin BT) (j : Fin BK) (hj : active s₀ DK BK j) :
    prepStreamSpec BT BK scale xs ⟨1, h1⟩ t j
      = prepareKgClosed s₀ K G s_qk_h DK BT BK t j := by
  have hk := hxK t j hj
  have hg := hxG t j (Or.inl hj)
  rw [fwd_addr_eq] at hk hg
  have hglast : s₀.readMem G (offset s₀ s_qk_h DK (BT - 1) BT BK j)
      = xs (⟨2, by omega⟩ : Fin 3) ⟨BT - 1, Nat.sub_lt hBT Nat.one_pos⟩ j := by
    have h := hxG ⟨BT - 1, Nat.sub_lt hBT Nat.one_pos⟩ j (Or.inr rfl)
    rwa [fwd_addr_eq] at h
  show (if h : 0 < BT then
      xs (⟨1, by omega⟩ : Fin 3) t j
        * Real.exp ((xs (⟨2, by omega⟩ : Fin 3) ⟨BT - 1, Nat.sub_lt h Nat.one_pos⟩ j
            - xs (⟨2, by omega⟩ : Fin 3) t j) * Real.log 2)
    else 0) = _
  rw [dif_pos hBT, ← hk, ← hg, ← hglast]
  simp only [prepareKgClosed]

private theorem prepPre10_castFree (R : RoundingModel) (Q K G QG KG : RegionName)
    (s_qk_h BT BK DK : Nat) (t : BlockState) :
    stepStmtsR R (prepPre10 Q K G QG KG s_qk_h BT BK DK) t
      = stepStmts (prepPre10 Q K G QG KG s_qk_h BT BK DK) t := by
  simp only [prepPre10, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

private theorem prepLd_castFree (R : RoundingModel) (G : RegionName)
    (s_qk_h BT BK DK : Nat) (t : BlockState) :
    stepStmtR R (prepLdStmt G s_qk_h BT BK DK) t
      = stepStmt (prepLdStmt G s_qk_h BT BK DK) t := by
  simp only [prepLdStmt, prepLdAddrOp, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real]

private theorem prepBody_castFree (R : RoundingModel) (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) (t : BlockState) :
    stepStmtsR R (prepBody Q K G QG KG s_qk_h DK BT BK scale) t
      = stepStmts (prepBody Q K G QG KG s_qk_h DK BT BK scale) t := by
  simp only [prepBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real, BlockState.writeMemTypedR]
  rfl

private theorem prep_flattenOk (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) :
    ((prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [prep_body_decomp]
  simp [prepPre10, prepLdStmt, prepLdAddrOp, prepBody, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk.eq_def]

private theorem prep_preLoop (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) (hBT : 0 < BT) (s : BlockState) :
    ∃ s10 s11,
      stepStmts (prepPre10 Q K G QG KG s_qk_h BT BK DK) s = some s10
      ∧ stepStmt (prepLdStmt G s_qk_h BT BK DK) s10 = some s11
      ∧ s10.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2))
      ∧ s10.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1))
      ∧ s10.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0))
      ∧ s10.regs .nat [BK] "offs" = some (prepOffsTile BK)
      ∧ prepInv Q K G QG KG s s_qk_h DK BT BK scale 0 s11
      ∧ s11.mem = s.mem := by
  
  set s_ik := s.setReg "i_k" .nat [] (Tile.scalar (s.pids 0)) with hs_ik
  set s_ic := s_ik.setReg "i_c" .nat [] (Tile.scalar (s.pids 1)) with hs_ic
  set s_ibh := s_ic.setReg "i_bh" .nat [] (Tile.scalar (s.pids 2)) with hs_ibh
  set s_offs := s_ibh.setReg "offs" .nat [BK] (prepOffsTile BK) with hs_offs
  set s_pq := s_offs.setReg "p_q" .ptr [BK] (prepPtrTile s Q s_qk_h DK BT BK 0) with hs_pq
  set s_pg := s_pq.setReg "p_g" .ptr [BK] (prepPtrTile s G s_qk_h DK BT BK 0) with hs_pg
  set s_pk := s_pg.setReg "p_k" .ptr [BK] (prepPtrTile s K s_qk_h DK BT BK 0) with hs_pk
  set s_pqg := s_pk.setReg "p_qg" .ptr [BK] (prepPtrTile s QG s_qk_h DK BT BK 0) with hs_pqg
  set s_pkg := s_pqg.setReg "p_kg" .ptr [BK] (prepPtrTile s KG s_qk_h DK BT BK 0) with hs_pkg
  set s_mask := s_pkg.setReg "mask" .bool [BK] (fwdMaskTile s DK BK) with hs_mask
  set s11 := s_mask.setReg "last_decay" .real [BK] (prepLastDecayTile s G s_qk_h DK BT BK) with hs11
  
  have hibh : s_ibh.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by simp [hs_ibh]
  have hic : s_ibh.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by
    rw [hs_ibh, hs_ic]; simp
  have hik : s_ibh.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by
    rw [hs_ibh, hs_ic, hs_ik]; simp
  
  have hptr : ∀ (R : RegionName) (st : BlockState),
      st.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) →
      st.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) →
      st.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) →
      st.regs .nat [BK] "offs" = some (prepOffsTile BK) →
      evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase R)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.ref TileDType.nat [BK] "offs"))) st = some (prepPtrTile s R s_qk_h DK BT BK 0) := by
    intro R st hbh hc hk ho
    simp only [evalOp, Option.bind, hbh, hc, hk, ho, evalOp_constNat,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
      Tile.bop, Tile.scalar]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [Tile.ptrAdd_data, prepPtrTile, offset, baseOffset, prepOffsTile,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil, Tile.vec]
    refine Prod.ext rfl ?_
    ring
  
  have hstepK : stepStmt (Stmt.assign TileDType.nat [] "i_k" (Op.programId 0)) s = some s_ik := by
    rw [hs_ik]; exact stepStmt_assign_eq_some (by simp [evalOp])
  have hstepC : stepStmt (Stmt.assign TileDType.nat [] "i_c" (Op.programId 1)) s_ik = some s_ic := by
    rw [hs_ic]; exact stepStmt_assign_eq_some (show evalOp (Op.programId 1) s_ik = _ from by simp [evalOp, hs_ik])
  have hstepB : stepStmt (Stmt.assign TileDType.nat [] "i_bh" (Op.programId 2)) s_ic = some s_ibh := by
    rw [hs_ibh]; exact stepStmt_assign_eq_some (show evalOp (Op.programId 2) s_ic = _ from by simp [evalOp, hs_ic, hs_ik])
  have hstepOffs : stepStmt (Stmt.assign TileDType.nat [BK] "offs" (Op.arange BK)) s_ibh = some s_offs := by
    rw [hs_offs]; exact stepStmt_assign_eq_some (by simp only [evalOp_arange, prepOffsTile])
  have hsoffs_bh : s_offs.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by rw [hs_offs]; simp [hibh]
  have hsoffs_c : s_offs.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by rw [hs_offs]; simp [hic]
  have hsoffs_k : s_offs.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by rw [hs_offs]; simp [hik]
  have hsoffs_o : s_offs.regs .nat [BK] "offs" = some (prepOffsTile BK) := by rw [hs_offs]; simp
  have hstepPq : stepStmt (Stmt.assign TileDType.ptr [BK] "p_q"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.ref TileDType.nat [BK] "offs")))) s_offs = some s_pq := by
    rw [hs_pq]; exact stepStmt_assign_eq_some (hptr Q s_offs hsoffs_bh hsoffs_c hsoffs_k hsoffs_o)
  have hspq_bh : s_pq.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by rw [hs_pq]; simp [hsoffs_bh]
  have hspq_c : s_pq.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by rw [hs_pq]; simp [hsoffs_c]
  have hspq_k : s_pq.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by rw [hs_pq]; simp [hsoffs_k]
  have hspq_o : s_pq.regs .nat [BK] "offs" = some (prepOffsTile BK) := by rw [hs_pq]; simp [hsoffs_o]
  have hstepPg : stepStmt (Stmt.assign TileDType.ptr [BK] "p_g"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase G)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.ref TileDType.nat [BK] "offs")))) s_pq = some s_pg := by
    rw [hs_pg]; exact stepStmt_assign_eq_some (hptr G s_pq hspq_bh hspq_c hspq_k hspq_o)
  have hspg_bh : s_pg.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by rw [hs_pg]; simp [hspq_bh]
  have hspg_c : s_pg.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by rw [hs_pg]; simp [hspq_c]
  have hspg_k : s_pg.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by rw [hs_pg]; simp [hspq_k]
  have hspg_o : s_pg.regs .nat [BK] "offs" = some (prepOffsTile BK) := by rw [hs_pg]; simp [hspq_o]
  have hstepPk : stepStmt (Stmt.assign TileDType.ptr [BK] "p_k"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.ref TileDType.nat [BK] "offs")))) s_pg = some s_pk := by
    rw [hs_pk]; exact stepStmt_assign_eq_some (hptr K s_pg hspg_bh hspg_c hspg_k hspg_o)
  have hspk_bh : s_pk.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by rw [hs_pk]; simp [hspg_bh]
  have hspk_c : s_pk.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by rw [hs_pk]; simp [hspg_c]
  have hspk_k : s_pk.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by rw [hs_pk]; simp [hspg_k]
  have hspk_o : s_pk.regs .nat [BK] "offs" = some (prepOffsTile BK) := by rw [hs_pk]; simp [hspg_o]
  have hstepPqg : stepStmt (Stmt.assign TileDType.ptr [BK] "p_qg"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase QG)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.ref TileDType.nat [BK] "offs")))) s_pk = some s_pqg := by
    rw [hs_pqg]; exact stepStmt_assign_eq_some (hptr QG s_pk hspk_bh hspk_c hspk_k hspk_o)
  have hspqg_bh : s_pqg.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by rw [hs_pqg]; simp [hspk_bh]
  have hspqg_c : s_pqg.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by rw [hs_pqg]; simp [hspk_c]
  have hspqg_k : s_pqg.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by rw [hs_pqg]; simp [hspk_k]
  have hspqg_o : s_pqg.regs .nat [BK] "offs" = some (prepOffsTile BK) := by rw [hs_pqg]; simp [hspk_o]
  have hstepPkg : stepStmt (Stmt.assign TileDType.ptr [BK] "p_kg"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase KG)
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.add NumericDType.nat Broadcast.nil
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat s_qk_h))
              (Op.mul NumericDType.nat Broadcast.nil
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_c") (Op.constNat BT))
                (Op.constNat DK)))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK)))
          (Op.ref TileDType.nat [BK] "offs")))) s_pqg = some s_pkg := by
    rw [hs_pkg]; exact stepStmt_assign_eq_some (hptr KG s_pqg hspqg_bh hspqg_c hspqg_k hspqg_o)
  have hspkg_k : s_pkg.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by rw [hs_pkg]; simp [hspqg_k]
  have hspkg_o : s_pkg.regs .nat [BK] "offs" = some (prepOffsTile BK) := by rw [hs_pkg]; simp [hspqg_o]
  
  have hmaskeval : evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK))
        (Op.ref TileDType.nat [BK] "offs")) (Op.constNat DK)) s_pkg = some (fwdMaskTile s DK BK) := by
    simp only [evalOp, Option.bind, hspkg_k, hspkg_o, evalOp_constNat,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
      ComparableDType.lt, Tile.bop, Tile.cop, Tile.scalar]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [fwdMaskTile, active, elemIndex, prepOffsTile, Tile.vec,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
    rfl
  have hstepMask : stepStmt (Stmt.assign TileDType.bool [BK] "mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_k") (Op.constNat BK))
          (Op.ref TileDType.nat [BK] "offs")) (Op.constNat DK))) s_pkg = some s_mask := by
    rw [hs_mask]; exact stepStmt_assign_eq_some hmaskeval
  have hsmask_bh : s_mask.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by
    rw [hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq]; simp [hsoffs_bh]
  have hsmask_c : s_mask.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)) := by
    rw [hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq]; simp [hsoffs_c]
  have hsmask_k : s_mask.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)) := by
    rw [hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq]; simp [hsoffs_k]
  have hsmask_o : s_mask.regs .nat [BK] "offs" = some (prepOffsTile BK) := by
    rw [hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq]; simp [hsoffs_o]
  
  have hldeval : evalOp (Op.load TileDType.real
      (MemAccess.region G (prepLdAddrOp s_qk_h BT BK DK))
      MaskOpt.none) s_mask = some (prepLastDecayTile s G s_qk_h DK BT BK) := by
    simp only [prepLdAddrOp, evalOp, Option.bind, hsmask_bh, hsmask_c, hsmask_k, hsmask_o,
      evalOp_constNat, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add,
      NumericDType.mul, NumericDType.sub, Tile.bop, Tile.scalar]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [prepLastDecayTile, prepOffsTile, Tile.vec,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
    rw [BlockState.readMemValue_real]
    simp only [offset, baseOffset]
    rw [show s.pids 1 * BT + BT - 1 = s.pids 1 * BT + (BT - 1) by omega]
    simp only [if_true]
    rw [hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
    simp only [BlockState.setReg_readMem]
    rfl
  have hstepLd : stepStmt (prepLdStmt G s_qk_h BT BK DK) s_mask = some s11 := by
    rw [hs11]; exact stepStmt_assign_eq_some hldeval
  
  have hP0 : prepInv Q K G QG KG s s_qk_h DK BT BK scale 0 s11 := by
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
      simp only [BlockState.setReg_pids]
    · intro a; rw [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
      simp only [BlockState.setReg_readMem]
    · intro a; rw [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
      simp only [BlockState.setReg_readMem]
    · intro a; rw [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
      simp only [BlockState.setReg_readMem]
    · 
      simp only [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq,
        BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, reduceCtorEq,
        not_false_eq_true, String.reduceEq]
    · 
      simp only [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg,
        BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, reduceCtorEq,
        not_false_eq_true, String.reduceEq]
    · 
      simp only [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk,
        BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, reduceCtorEq,
        not_false_eq_true, String.reduceEq]
    · 
      simp only [hs11, hs_mask, hs_pkg, hs_pqg,
        BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, reduceCtorEq,
        not_false_eq_true, String.reduceEq]
    · 
      simp only [hs11, hs_mask, hs_pkg,
        BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, reduceCtorEq,
        not_false_eq_true, String.reduceEq]
    · 
      simp only [hs11, hs_mask,
        BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, reduceCtorEq,
        not_false_eq_true, String.reduceEq]
    · 
      rw [hs11, BlockState.setReg_same]
    · intro j hj i; rw [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
      simp only [BlockState.setReg_readMem]
    · intro j hj i; rw [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
      simp only [BlockState.setReg_readMem]
    · intro j hj i hjBT; omega
    · intro j hj i hjBT; omega
  refine ⟨s_mask, s11, ?_, hstepLd, hsmask_bh, hsmask_c, hsmask_k, hsmask_o, hP0, ?_⟩
  · unfold prepPre10
    rw [stepStmts.cons_some hstepK, stepStmts.cons_some hstepC, stepStmts.cons_some hstepB,
      stepStmts.cons_some hstepOffs, stepStmts.cons_some hstepPq, stepStmts.cons_some hstepPg,
      stepStmts.cons_some hstepPk, stepStmts.cons_some hstepPqg, stepStmts.cons_some hstepPkg,
      stepStmts.cons_some hstepMask, stepStmts.nil]
  · rw [hs11, hs_mask, hs_pkg, hs_pqg, hs_pk, hs_pg, hs_pq, hs_offs, hs_ibh, hs_ic, hs_ik]
    rfl

private theorem prep_body_step_frame (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ) (s sc s' : BlockState) (m : Nat)
    (hpqg : sc.regs .ptr [BK] "p_qg" = some (prepPtrTile s QG s_qk_h DK BT BK m))
    (hpkg : sc.regs .ptr [BK] "p_kg" = some (prepPtrTile s KG s_qk_h DK BT BK m))
    (hmask : sc.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK))
    (hstep : stepStmts (prepBody Q K G QG KG s_qk_h DK BT BK scale)
      (sc.setReg "i" .nat [] (Tile.scalar m)) = some s')
    (r : RegionName) (oo : Nat)
    (hcQG : r ≠ QG ∨ ∀ i : Fin BK, active s DK BK i →
      oo ≠ offset s s_qk_h DK m BT BK i)
    (hcKG : r ≠ KG ∨ ∀ i : Fin BK, active s DK BK i →
      oo ≠ offset s s_qk_h DK m BT BK i) :
    s'.mem r oo = sc.mem r oo := by
  set sloop := sc.setReg "i" .nat [] (Tile.scalar m) with hsloop
  unfold prepBody at hstep
  obtain ⟨v1, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨v2, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨v3, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨v4, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨v5, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  set s5 := ((((sloop.setReg "_q" .real [BK] v1).setReg "_k" .real [BK] v2).setReg
    "_g" .real [BK] v3).setReg "_q" .real [BK] v4).setReg "_k" .real [BK] v5 with hs5
  have hkval5 : s5.regs .real [BK] "_k" = some v5 := by
    rw [hs5]; exact BlockState.setReg_same _ _ _ _ _
  have hqval5 : s5.regs .real [BK] "_q" = some v4 := by
    rw [hs5, BlockState.setReg_ne_name (h := by decide)]
    exact BlockState.setReg_same _ _ _ _ _
  have hpkg5 : s5.regs .ptr [BK] "p_kg" = some (prepPtrTile s KG s_qk_h DK BT BK m) := by
    rw [hs5, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide), hsloop,
      BlockState.setReg_ne_name (h := by decide)]
    exact hpkg
  have hpqg5 : s5.regs .ptr [BK] "p_qg" = some (prepPtrTile s QG s_qk_h DK BT BK m) := by
    rw [hs5, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide), hsloop,
      BlockState.setReg_ne_name (h := by decide)]
    exact hpqg
  have hmask5 : s5.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
    rw [hs5, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide), hsloop,
      BlockState.setReg_ne_name (h := by decide)]
    exact hmask
  
  set vfun1 : TileIndex [BK] → ℝ := fun k => (v5.data k).unbotD 0 with hvfun1
  set sst1 := (TileShape.allIndices [BK]).foldl
    (fun (acc : BlockState) k =>
      if active s DK BK k.1 then
        acc.writeMem KG (offset s s_qk_h DK m BT BK k.1) (vfun1 k)
      else acc) s5 with hsst1
  have hstore1 : stepStmt (Stmt.store TileDType.real [BK]
      (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_kg"))
      (Op.ref TileDType.real [BK] "_k")
      (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask"))) s5 = some sst1 := by
    simp only [stepStmt, evalOp, Option.bind, Option.map, hkval5, hpkg5, hmask5]
    apply congrArg some
    rw [hsst1]
    congr 1
    funext acc k
    simp only [prepPtrTile, fwdMaskTile]
    by_cases ha : active s DK BK k.1
    · rw [if_pos (by simpa using ha), if_pos ha, BlockState.writeMemTyped_real]
      rfl
    · rw [if_neg (by simpa using ha), if_neg ha]
  rw [stepStmts.cons_some hstore1] at hstep
  
  have hqval1 : sst1.regs .real [BK] "_q" = some v4 := by
    rw [hsst1, BlockState.foldl_writeMem_prop_masked_regs]
    exact hqval5
  have hpqg1 : sst1.regs .ptr [BK] "p_qg" = some (prepPtrTile s QG s_qk_h DK BT BK m) := by
    rw [hsst1, BlockState.foldl_writeMem_prop_masked_regs]
    exact hpqg5
  have hmask1 : sst1.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
    rw [hsst1, BlockState.foldl_writeMem_prop_masked_regs]
    exact hmask5
  set vfun2 : TileIndex [BK] → ℝ := fun k => (v4.data k).unbotD 0 with hvfun2
  set sst2 := (TileShape.allIndices [BK]).foldl
    (fun (acc : BlockState) k =>
      if active s DK BK k.1 then
        acc.writeMem QG (offset s s_qk_h DK m BT BK k.1) (vfun2 k)
      else acc) sst1 with hsst2
  have hstore2 : stepStmt (Stmt.store TileDType.real [BK]
      (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_qg"))
      (Op.ref TileDType.real [BK] "_q")
      (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask"))) sst1 = some sst2 := by
    simp only [stepStmt, evalOp, Option.bind, Option.map, hqval1, hpqg1, hmask1]
    apply congrArg some
    rw [hsst2]
    congr 1
    funext acc k
    simp only [prepPtrTile, fwdMaskTile]
    by_cases ha : active s DK BK k.1
    · rw [if_pos (by simpa using ha), if_pos ha, BlockState.writeMemTyped_real]
      rfl
    · rw [if_neg (by simpa using ha), if_neg ha]
  rw [stepStmts.cons_some hstore2] at hstep
  obtain ⟨w1, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨w2, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨w3, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨w4, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨w5, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  rw [stepStmts.nil] at hstep
  obtain rfl := Option.some.inj hstep
  simp only [BlockState.setReg_mem]
  rw [hsst2]
  refine Eq.trans (foldl_writeMem_prop_preserve_cell _ _ _ r oo _ _ ?_) ?_
  · intro k hk hmk hbad
    rcases hcQG with hne' | hno
    · exact hne' hbad.1
    · exact hno k.1 hmk hbad.2
  · rw [hsst1]
    refine Eq.trans (foldl_writeMem_prop_preserve_cell _ _ _ r oo _ _ ?_) ?_
    · intro k hk hmk hbad
      rcases hcKG with hne' | hno
      · exact hne' hbad.1
      · exact hno k.1 hmk hbad.2
    · rw [hs5, hsloop]
      simp only [BlockState.setReg_mem]

private theorem foldl_regs_of_preserve {α : Type}
    (f : BlockState → α → BlockState)
    (hf : ∀ s a, (f s a).regs = s.regs) :
    ∀ (l : List α) (s : BlockState), (l.foldl f s).regs = s.regs
  | [], _ => rfl
  | a :: l, s => by
      rw [List.foldl_cons, foldl_regs_of_preserve f hf l (f s a), hf]

private theorem stepStmtR_store_real_regs {R : RoundingModel} {sh : TileShape}
    {pe : Op TileDType.ptr sh} {ve : Op TileDType.real sh}
    {me : Op TileDType.bool sh} {s s' : BlockState}
    (h : stepStmtR R (Stmt.store TileDType.real sh (MemAccess.ptr pe) ve
      (MaskOpt.mask me)) s = some s') :
    s'.regs = s.regs := by
  cases hv : evalOpR R ve s with
  | none => simp [stepStmtR, hv] at h
  | some vs =>
      cases hm : evalOpR R me s with
      | none => simp [stepStmtR, hv, hm] at h
      | some ms =>
          cases hp : evalOpR R pe s with
          | none => simp [stepStmtR, hv, hm, hp] at h
          | some ps =>
              simp only [stepStmtR, hv, hm, hp, Option.map_some, bind,
                Option.bind_some, Option.some.injEq] at h
              rw [← h]
              refine foldl_regs_of_preserve _ ?_ _ _
              intro acc i
              by_cases hai : ms.data i = Bool.true
              · rw [if_pos hai]; rfl
              · rw [if_neg hai]

private theorem prepLdAddrR_eq (R : RoundingModel) (s_qk_h BT BK DK : Nat)
    (st : BlockState) :
    evalOpR R (prepLdAddrOp s_qk_h BT BK DK) st
      = evalOp (prepLdAddrOp s_qk_h BT BK DK) st := by
  simp only [prepLdAddrOp, evalOpR.eq_def, evalOp.eq_def]

private theorem prepLdAddr_eval (s_qk_h BT BK DK : Nat) (s st : BlockState)
    (hbh : st.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)))
    (hc : st.regs .nat [] "i_c" = some (Tile.scalar (s.pids 1)))
    (hk : st.regs .nat [] "i_k" = some (Tile.scalar (s.pids 0)))
    (ho : st.regs .nat [BK] "offs" = some (prepOffsTile BK)) :
    evalOp (prepLdAddrOp s_qk_h BT BK DK) st
      = some (Tile.vec (fun i : Fin BK =>
          s.pids 2 * s_qk_h + (s.pids 1 * BT + BT - 1) * DK + s.pids 0 * BK + i.val)) := by
  simp only [prepLdAddrOp, evalOp, Option.bind, hbh, hc, hk, ho, evalOp_constNat,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    NumericDType.sub, Tile.bop, Tile.scalar]
  apply congrArg some
  apply Tile.ext
  intro idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.vec, prepOffsTile,
    Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
    Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]

set_option maxHeartbeats 8000000 in

private theorem prep_bodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (Q K G QG KG : RegionName) (s_qk_h DK BT BK : Nat) (scale : ℝ)
    (s stt : BlockState) (m : Nat)
    (hpq : stt.regs .ptr [BK] "p_q" = some (prepPtrTile s Q s_qk_h DK BT BK m))
    (hpg : stt.regs .ptr [BK] "p_g" = some (prepPtrTile s G s_qk_h DK BT BK m))
    (hpk : stt.regs .ptr [BK] "p_k" = some (prepPtrTile s K s_qk_h DK BT BK m))
    (hpqg : stt.regs .ptr [BK] "p_qg" = some (prepPtrTile s QG s_qk_h DK BT BK m))
    (hpkg : stt.regs .ptr [BK] "p_kg" = some (prepPtrTile s KG s_qk_h DK BT BK m))
    (hmask : stt.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK))
    (hbQ : ∀ i : Fin BK, active s DK BK i → offset s s_qk_h DK m BT BK i < bounds Q)
    (hbK : ∀ i : Fin BK, active s DK BK i → offset s s_qk_h DK m BT BK i < bounds K)
    (hbG : ∀ i : Fin BK, active s DK BK i → offset s s_qk_h DK m BT BK i < bounds G)
    (hbQG : ∀ i : Fin BK, active s DK BK i → offset s s_qk_h DK m BT BK i < bounds QG)
    (hbKG : ∀ i : Fin BK, active s DK BK i → offset s s_qk_h DK m BT BK i < bounds KG) :
    Stmt.TraceSafeListR R bounds (prepBody Q K G QG KG s_qk_h DK BT BK scale)
      (stt.setReg "i" .nat [] (Tile.scalar m)) := by
  set sloop := stt.setReg "i" .nat [] (Tile.scalar m) with hsloop
  have hpq' : sloop.regs .ptr [BK] "p_q" = some (prepPtrTile s Q s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpq
  have hpg' : sloop.regs .ptr [BK] "p_g" = some (prepPtrTile s G s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpg
  have hpk' : sloop.regs .ptr [BK] "p_k" = some (prepPtrTile s K s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpk
  have hpqg' : sloop.regs .ptr [BK] "p_qg" = some (prepPtrTile s QG s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpqg
  have hpkg' : sloop.regs .ptr [BK] "p_kg" = some (prepPtrTile s KG s_qk_h DK BT BK m) := by
    rw [hsloop]; simpa using hpkg
  have hmask' : sloop.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
    rw [hsloop]; simpa using hmask
  unfold prepBody
  
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t1 ht1 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def], ?_⟩
    intro ptrs hptrs idx hactive
    rw [evalOpR_ref, hpq'] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [evalOpR_ref, hmask'] at hm
    obtain rfl := Option.some.inj hm
    have ha : active s DK BK idx.1 := by simpa [fwdMaskTile] using hmi
    simpa [prepPtrTile] using hbQ idx.1 ha
  · obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv ht1
    have hpk1 : (sloop.setReg "_q" .real [BK] v1).regs .ptr [BK] "p_k"
        = some (prepPtrTile s K s_qk_h DK BT BK m) := by
      rw [BlockState.setReg_ne_name (h := by decide)]
      exact hpk'
    have hmask1 : (sloop.setReg "_q" .real [BK] v1).regs .bool [BK] "mask"
        = some (fwdMaskTile s DK BK) := by
      rw [BlockState.setReg_ne_name (h := by decide)]
      exact hmask'
    
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun t2 ht2 => ?_)
    · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
        MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
      refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def], ?_⟩
      intro ptrs hptrs idx hactive
      rw [evalOpR_ref, hpk1] at hptrs
      obtain rfl := Option.some.inj hptrs
      obtain ⟨masks, hm, hmi⟩ := hactive
      rw [evalOpR_ref, hmask1] at hm
      obtain rfl := Option.some.inj hm
      have ha : active s DK BK idx.1 := by simpa [fwdMaskTile] using hmi
      simpa [prepPtrTile] using hbK idx.1 ha
    · obtain ⟨v2, -, rfl⟩ := stepStmtR_assign_inv ht2
      have hpg2 : ((sloop.setReg "_q" .real [BK] v1).setReg
          "_k" .real [BK] v2).regs .ptr [BK] "p_g"
          = some (prepPtrTile s G s_qk_h DK BT BK m) := by
        rw [BlockState.setReg_ne_name (h := by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpg'
      have hmask2 : ((sloop.setReg "_q" .real [BK] v1).setReg
          "_k" .real [BK] v2).regs .bool [BK] "mask"
          = some (fwdMaskTile s DK BK) := by
        rw [BlockState.setReg_ne_name (h := by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hmask'
      
      refine Stmt.TraceSafeListR.cons_intro ?_ (fun t3 ht3 => ?_)
      · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
          MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
        refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def], ?_⟩
        intro ptrs hptrs idx hactive
        rw [evalOpR_ref, hpg2] at hptrs
        obtain rfl := Option.some.inj hptrs
        obtain ⟨masks, hm, hmi⟩ := hactive
        rw [evalOpR_ref, hmask2] at hm
        obtain rfl := Option.some.inj hm
        have ha : active s DK BK idx.1 := by simpa [fwdMaskTile] using hmi
        simpa [prepPtrTile] using hbG idx.1 ha
      · obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv ht3
        
        refine Stmt.TraceSafeListR.cons_intro
          (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t4 ht4 => ?_)
        obtain ⟨v4, -, rfl⟩ := stepStmtR_assign_inv ht4
        
        refine Stmt.TraceSafeListR.cons_intro
          (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t5 ht5 => ?_)
        obtain ⟨v5, -, rfl⟩ := stepStmtR_assign_inv ht5
        have hpkg5 : (((((sloop.setReg "_q" .real [BK] v1).setReg
            "_k" .real [BK] v2).setReg "_g" .real [BK] v3).setReg
            "_q" .real [BK] v4).setReg "_k" .real [BK] v5).regs .ptr [BK] "p_kg"
            = some (prepPtrTile s KG s_qk_h DK BT BK m) := by
          rw [BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide)]
          exact hpkg'
        have hpqg5 : (((((sloop.setReg "_q" .real [BK] v1).setReg
            "_k" .real [BK] v2).setReg "_g" .real [BK] v3).setReg
            "_q" .real [BK] v4).setReg "_k" .real [BK] v5).regs .ptr [BK] "p_qg"
            = some (prepPtrTile s QG s_qk_h DK BT BK m) := by
          rw [BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide)]
          exact hpqg'
        have hmask5 : (((((sloop.setReg "_q" .real [BK] v1).setReg
            "_k" .real [BK] v2).setReg "_g" .real [BK] v3).setReg
            "_q" .real [BK] v4).setReg "_k" .real [BK] v5).regs .bool [BK] "mask"
            = some (fwdMaskTile s DK BK) := by
          rw [BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide),
            BlockState.setReg_ne_name (h := by decide)]
          exact hmask'
        
        refine Stmt.TraceSafeListR.cons_intro ?_ (fun t6 ht6 => ?_)
        · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MemAccess.SafeAtR,
            MaskOpt.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
            memAccessActiveAddressSafeR]
          refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def],
            by simp [Op.SafeAtR.eq_def], ?_⟩
          intro ptrs hptrs idx hactive
          rw [evalOpR_ref, hpkg5] at hptrs
          obtain rfl := Option.some.inj hptrs
          obtain ⟨masks, hm, hmi⟩ := hactive
          rw [evalOpR_ref, hmask5] at hm
          obtain rfl := Option.some.inj hm
          have ha : active s DK BK idx.1 := by simpa [fwdMaskTile] using hmi
          simpa [prepPtrTile] using hbKG idx.1 ha
        · have hregs6 := stepStmtR_store_real_regs ht6
          have hpqg6 : t6.regs .ptr [BK] "p_qg"
              = some (prepPtrTile s QG s_qk_h DK BT BK m) := by
            rw [hregs6]; exact hpqg5
          have hmask6 : t6.regs .bool [BK] "mask" = some (fwdMaskTile s DK BK) := by
            rw [hregs6]; exact hmask5
          
          refine Stmt.TraceSafeListR.cons_intro ?_ (fun t7 ht7 => ?_)
          · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MemAccess.SafeAtR,
              MaskOpt.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
              memAccessActiveAddressSafeR]
            refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def],
              by simp [Op.SafeAtR.eq_def], ?_⟩
            intro ptrs hptrs idx hactive
            rw [evalOpR_ref, hpqg6] at hptrs
            obtain rfl := Option.some.inj hptrs
            obtain ⟨masks, hm, hmi⟩ := hactive
            rw [evalOpR_ref, hmask6] at hm
            obtain rfl := Option.some.inj hm
            have ha : active s DK BK idx.1 := by simpa [fwdMaskTile] using hmi
            simpa [prepPtrTile] using hbQG idx.1 ha
          · 
            refine Stmt.TraceSafeListR.cons_intro
              (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun _ _ => ?_)
            refine Stmt.TraceSafeListR.cons_intro
              (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun _ _ => ?_)
            refine Stmt.TraceSafeListR.cons_intro
              (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun _ _ => ?_)
            refine Stmt.TraceSafeListR.cons_intro
              (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun _ _ => ?_)
            exact Stmt.TraceSafeListR.cons_intro
              (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
              (fun _ _ => Stmt.TraceSafeListR.nil_intro)

set_option maxHeartbeats 8000000 in

private theorem prep_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (Q K G QG KG : RegionName) (s_qk_h DK BT BK : Nat) (scale : ℝ)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hK_QG : K ≠ QG) (hK_KG : K ≠ KG)
    (hG_QG : G ≠ QG) (hG_KG : G ≠ KG) (hQG_KG : QG ≠ KG)
    (hBK : BK ≤ DK) (hBT : 0 < BT) (s : BlockState)
    (hbQ : ∀ (t : Fin BT) (j : Fin BK), s.pids 0 * BK + j.val < DK →
      s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + j.val + t.val * DK
        < bounds Q)
    (hbK : ∀ (t : Fin BT) (j : Fin BK), s.pids 0 * BK + j.val < DK →
      s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + j.val + t.val * DK
        < bounds K)
    (hbG : ∀ (t : Fin BT) (j : Fin BK),
      (s.pids 0 * BK + j.val < DK ∨ t.val = BT - 1) →
      s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + j.val + t.val * DK
        < bounds G)
    (hbQG : ∀ (t : Fin BT) (j : Fin BK), s.pids 0 * BK + j.val < DK →
      s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + j.val + t.val * DK
        < bounds QG)
    (hbKG : ∀ (t : Fin BT) (j : Fin BK), s.pids 0 * BK + j.val < DK →
      s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + j.val + t.val * DK
        < bounds KG) :
    ((prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale).toAlgKernel).TraceSafeR R bounds s := by
  
  have hbQ' : ∀ m : Nat, m < BT → ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK m BT BK i < bounds Q := by
    intro m hm i ha
    have h : s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + i.val + m * DK
        < bounds Q := hbQ ⟨m, hm⟩ i ha
    rwa [fwd_addr_eq] at h
  have hbK' : ∀ m : Nat, m < BT → ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK m BT BK i < bounds K := by
    intro m hm i ha
    have h : s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + i.val + m * DK
        < bounds K := hbK ⟨m, hm⟩ i ha
    rwa [fwd_addr_eq] at h
  have hbG' : ∀ m : Nat, m < BT → ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK m BT BK i < bounds G := by
    intro m hm i ha
    have h : s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + i.val + m * DK
        < bounds G := hbG ⟨m, hm⟩ i (Or.inl ha)
    rwa [fwd_addr_eq] at h
  have hbQG' : ∀ m : Nat, m < BT → ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK m BT BK i < bounds QG := by
    intro m hm i ha
    have h : s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + i.val + m * DK
        < bounds QG := hbQG ⟨m, hm⟩ i ha
    rwa [fwd_addr_eq] at h
  have hbKG' : ∀ m : Nat, m < BT → ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK m BT BK i < bounds KG := by
    intro m hm i ha
    have h : s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + i.val + m * DK
        < bounds KG := hbKG ⟨m, hm⟩ i ha
    rwa [fwd_addr_eq] at h
  unfold Kernel.TraceSafeR
  rw [prep_body_decomp]
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · 
    refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro stmt hst s'
    simp only [prepPre10, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  · intro s1 hs1
    obtain ⟨s10, s11, hsp10, hld, hbh10, hc10, hk10, ho10, hP0, hmem11⟩ :=
      prep_preLoop Q K G QG KG s_qk_h DK BT BK scale hBT s
    rw [prepPre10_castFree R Q K G QG KG s_qk_h BT BK DK s, hsp10] at hs1
    obtain rfl := Option.some.inj hs1
    
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun s11' hs11 => ?_)
    · simp only [prepLdStmt, Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
        MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
      refine ⟨?_, trivial, ?_⟩
      · simp only [prepLdAddrOp, Op.SafeAtR, and_self]
      · intro offsets hoffsets idx _
        rw [prepLdAddrR_eq, prepLdAddr_eval s_qk_h BT BK DK s s10 hbh10 hc10 hk10 ho10]
          at hoffsets
        obtain rfl := Option.some.inj hoffsets
        have hb : s.pids 2 * s_qk_h + s.pids 1 * BT * DK + s.pids 0 * BK + idx.1.val
            + (BT - 1) * DK < bounds G :=
          hbG ⟨BT - 1, Nat.sub_lt hBT Nat.one_pos⟩ idx.1 (Or.inr rfl)
        simpa [Region.cast_id, Tile.vec,
          prep_ld_addr_eq s s_qk_h BT BK DK hBT idx.1] using hb
    · rw [prepLd_castFree R G s_qk_h BT BK DK s10, hld] at hs11
      obtain rfl := Option.some.inj hs11
      
      refine Stmt.TraceSafeListR.cons_intro ?_
        (fun _ _ => Stmt.TraceSafeListR.nil_intro)
      simp only [Stmt.TraceSafeR]
      refine Stmt.forRangeTraceSafeR_inv R bounds "i" BT 1
        (prepBody Q K G QG KG s_qk_h DK BT BK scale)
        (prepInv Q K G QG KG s s_qk_h DK BT BK scale) ?_ 0 s11 hP0
      intro c stt hc hP
      have hPd := hP
      obtain ⟨hpids, hQm, hKm, hGm, hpq, hpg, hpk, hpqg, hpkg, hmask, hldreg,
        hUQG, hUKG, hrQG, hrKG⟩ := hPd
      refine ⟨prep_bodySafeR R bounds Q K G QG KG s_qk_h DK BT BK scale s stt c
        hpq hpg hpk hpqg hpkg hmask (hbQ' c hc) (hbK' c hc) (hbG' c hc)
        (hbQG' c hc) (hbKG' c hc), ?_⟩
      obtain ⟨st', hstep, hP'⟩ := prepare_qg_kg_step Q K G QG KG s s_qk_h DK BT BK scale
        hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK hBT c hc stt hP
      exact ⟨st', by rw [prepBody_castFree]; exact hstep, hP'⟩

set_option maxHeartbeats 8000000 in

private theorem prep_runR (R : RoundingModel) (Q K G QG KG : RegionName)
    (s_qk_h DK BT BK : Nat) (scale : ℝ)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hK_QG : K ≠ QG) (hK_KG : K ≠ KG)
    (hG_QG : G ≠ QG) (hG_KG : G ≠ KG) (hQG_KG : QG ≠ KG)
    (hBK : BK ≤ DK) (hBT : 0 < BT) (s₀ : BlockState) :
    ∃ sfin,
      execR R (prepare_qg_kg_surface Q K G QG KG s_qk_h DK BT BK scale).toAlgKernel s₀
        = some sfin
      ∧ (∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
          sfin.readMem QG (offset s₀ s_qk_h DK t.val BT BK i)
            = prepareQgClosed s₀ Q G s_qk_h DK BT BK t i scale)
      ∧ (∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
          sfin.readMem KG (offset s₀ s_qk_h DK t.val BT BK i)
            = prepareKgClosed s₀ K G s_qk_h DK BT BK t i)
      ∧ (∀ r oo,
          (r ≠ QG ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          (r ≠ KG ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          sfin.mem r oo = s₀.mem r oo) := by
  obtain ⟨s10, s11, hsp10, hld, hbh10, hc10, hk10, ho10, hP0, hmem11⟩ :=
    prep_preLoop Q K G QG KG s_qk_h DK BT BK scale hBT s₀
  
  obtain ⟨final, sfinal, hLoop, hfinal, hPfinal⟩ :=
    forRange_inv (idx := "i") (start := 0) (stop := BT) (step := 1)
      (body := prepBody Q K G QG KG s_qk_h DK BT BK scale)
      (P := fun m stt => prepInv Q K G QG KG s₀ s_qk_h DK BT BK scale m stt
        ∧ ∀ r oo,
          (r ≠ QG ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          (r ≠ KG ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          stt.mem r oo = s₀.mem r oo)
      Nat.one_ne_zero
      ⟨hP0, fun r oo _ _ => by rw [hmem11]⟩
      (fun m stt hm hP => by
        obtain ⟨st', hstep, hP'⟩ := prepare_qg_kg_step Q K G QG KG s₀ s_qk_h DK BT BK scale
          hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK hBT m hm stt hP.1
        refine ⟨st', hstep, hP', ?_⟩
        intro r oo hcQG hcKG
        obtain ⟨hpids, hQm, hKm, hGm, hpq, hpg, hpk, hpqg, hpkg, hmask, hldreg,
          hUQG, hUKG, hrQG, hrKG⟩ := hP.1
        rw [prep_body_step_frame Q K G QG KG s_qk_h DK BT BK scale s₀ stt st' m
          hpqg hpkg hmask hstep r oo ?_ ?_]
        · exact hP.2 r oo hcQG hcKG
        · rcases hcQG with hne' | hno
          · exact Or.inl hne'
          · exact Or.inr fun i ha => hno ⟨m, hm⟩ i ha
        · rcases hcKG with hne' | hno
          · exact Or.inl hne'
          · exact Or.inr fun i ha => hno ⟨m, hm⟩ i ha)
  obtain ⟨⟨_, _, _, _, _, _, _, _, _, _, _, _, _, hreadQG, hreadKG⟩, hframe⟩ := hPfinal
  
  have hLdR : stepStmtR R (prepLdStmt G s_qk_h BT BK DK) s10 = some s11 := by
    rw [prepLd_castFree]; exact hld
  have hLoopR : stepStmtR R
      (Stmt.forRange "i" 0 BT 1 (prepBody Q K G QG KG s_qk_h DK BT BK scale)) s11
      = some sfinal := by
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _
        (prepBody_castFree R Q K G QG KG s_qk_h DK BT BK scale) "i",
      ← stepForRangeAux.forRange_unfold]
    exact hLoop
  refine ⟨sfinal, ?_, ?_, ?_, ?_⟩
  · unfold execR
    rw [prep_body_decomp,
      stepStmtsR_append R (prepPre10 Q K G QG KG s_qk_h BT BK DK) _ s₀,
      prepPre10_castFree R Q K G QG KG s_qk_h BT BK DK s₀, hsp10, Option.bind_some,
      stepStmtsR_cons_some hLdR, stepStmtsR_cons_some hLoopR, stepStmtsR_nil]
  · intro t i ha
    have ht : t.val < final := lt_of_lt_of_le t.isLt hfinal
    rw [hreadQG t.val ht i t.isLt, if_pos ha]
  · intro t i ha
    have ht : t.val < final := lt_of_lt_of_le t.isLt hfinal
    rw [hreadKG t.val ht i t.isLt, if_pos ha]
  · exact hframe

set_option maxHeartbeats 8000000 in

theorem prepare_qg_kg_io_correctness (R : RoundingModel)
    (Q K G QG KG : RegionName) (s_qk_h DK BT BK : Nat) (scale : ℝ)
    (hQ_QG : Q ≠ QG) (hQ_KG : Q ≠ KG) (hK_QG : K ≠ QG) (hK_KG : K ≠ KG)
    (hG_QG : G ≠ QG) (hG_KG : G ≠ KG) (hQG_KG : QG ≠ KG)
    (hBK : BK ≤ DK) (hBT : 0 < BT) :
    prepareQgKgKernelIO Q K G QG KG s_qk_h DK BT BK scale ⊨[R]
      fun _ _ _ xs o t j => prepStreamSpec BT BK scale xs o t j := by
  refine StreamGroupedEmitMasked3DKernelIO.ImplementsR.intro _ ?_ ?_ ?_ ?_
  · 
    intro o
    simp only [prepareQgKgKernelIO]
    match o with
    | ⟨0, _⟩ => show QG ∈ [Q, K, G, QG, KG]; simp
    | ⟨1, _⟩ => show KG ∈ [Q, K, G, QG, KG]; simp
    | ⟨n + 2, h2⟩ => exact absurd h2 (show ¬(n + 2 < 2) by omega)
  · exact prep_flattenOk Q K G QG KG s_qk_h DK BT BK scale
  · 
    intro bounds s xs _hx hbr hbw
    simp only [prepareQgKgKernelIO] at hbr hbw ⊢
    refine prep_traceSafeR R bounds Q K G QG KG s_qk_h DK BT BK scale
      hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK hBT s ?_ ?_ ?_ ?_ ?_
    · exact fun t j hj => hbr ⟨0, by omega⟩ t j hj
    · exact fun t j hj => hbr ⟨1, by omega⟩ t j hj
    · exact fun t j hj => hbr ⟨2, by omega⟩ t j hj
    · exact fun t j hj => hbw ⟨0, by omega⟩ t j hj
    · exact fun t j hj => hbw ⟨1, by omega⟩ t j hj
  · 
    intro s₀ xs _hundef hx
    simp only [prepareQgKgKernelIO] at hx ⊢
    obtain ⟨sfin, hexec, hqg, hkg, hframe⟩ :=
      prep_runR R Q K G QG KG s_qk_h DK BT BK scale
        hQ_QG hQ_KG hK_QG hK_KG hG_QG hG_KG hQG_KG hBK hBT s₀
    refine ⟨sfin, hexec, ?_, ?_⟩
    · intro o t j hj
      match o with
      | ⟨0, h0⟩ =>
          rw [BlockState.readMemAs_real, fwd_addr_eq]
          have hval : sfin.readMem QG (offset s₀ s_qk_h DK t.val BT BK j)
              = prepStreamSpec BT BK scale xs ⟨0, h0⟩ t j := by
            rw [hqg t j hj,
              ← prepStreamSpec_qg_eq_closed Q G s₀ s_qk_h DK BT BK scale xs h0
                (fun t' j' hj' => hx ⟨0, by omega⟩ t' j' hj')
                (fun t' j' hj' => hx ⟨2, by omega⟩ t' j' (Or.inl hj')) t j hj]
          simp only [RoundingModel.round_real_apply]
          exact congrArg some hval
      | ⟨1, h1⟩ =>
          rw [BlockState.readMemAs_real, fwd_addr_eq]
          have hval : sfin.readMem KG (offset s₀ s_qk_h DK t.val BT BK j)
              = prepStreamSpec BT BK scale xs ⟨1, h1⟩ t j := by
            rw [hkg t j hj,
              ← prepStreamSpec_kg_eq_closed K G s₀ s_qk_h DK BT BK scale xs hBT h1
                (fun t' j' hj' => hx ⟨1, by omega⟩ t' j' hj')
                (fun t' j' hj' => hx ⟨2, by omega⟩ t' j' hj') t j hj]
          simp only [RoundingModel.round_real_apply]
          exact congrArg some hval
      | ⟨n + 2, h2⟩ => exact absurd h2 (show ¬(n + 2 < 2) by omega)
    · intro r oo hcond
      refine hframe r oo ?_ ?_
      · by_cases hr : r = QG
        · refine Or.inr fun t i ha hoeq => ?_
          exact hcond ⟨0, by omega⟩ t i ha hr
            (hoeq.trans (fwd_addr_eq s₀ s_qk_h BT BK DK t.val i).symm)
        · exact Or.inl hr
      · by_cases hr : r = KG
        · refine Or.inr fun t i ha hoeq => ?_
          exact hcond ⟨1, by omega⟩ t i ha hr
            (hoeq.trans (fwd_addr_eq s₀ s_qk_h BT BK DK t.val i).symm)
        · exact Or.inl hr

end PrepareIOFace

section BwdIOFace

open scoped VeriTile.Triton.StreamGroupedEmitMasked3DKernelIO

set_option maxHeartbeats 4000000
set_option linter.unusedVariables false

private def bwdStopOp (BT : Nat) : Op TileDType.nat [] :=
  Op.add .nat Broadcast.nil
    (Op.div .nat Broadcast.nil
      (Op.sub .nat Broadcast.nil (Op.constNat BT) (Op.constNat 1)) (Op.constNat 1))
    (Op.constNat 1)

private def bwdCoreG (DK BT BK : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "j"
      (Op.sub .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil (Op.constNat BT) (Op.constNat 1))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "__rev_j") (Op.constNat 1))),
    Stmt.assign .real [BK] "_g"
      (Op.load ComputeDType.fp32.eraseDType (MemAccess.ptr (Op.ref .ptr [BK] "p_g"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.ifThen
      (Op.eq ComparableDType.nat Broadcast.nil (Op.ref .nat [] "j")
        (Op.sub .nat Broadcast.nil (Op.constNat BT) (Op.constNat 1)))
      [Stmt.assign .real [BK] "last_g" (Op.ref .real [BK] "_g")],
    Stmt.assign .real [BK] "_dq1"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_dq_inner"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign .real [BK] "_dq2"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_dq_inter"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign .real [BK] "_dq2"
      (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "_dq2")
        (Op.ref .real [BK] "_g").exp2),
    Stmt.assign .real [BK] "_dq"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [BK] "_dq1")
        (Op.ref .real [BK] "_dq2")),
    Stmt.store .real [BK] (MemAccess.ptr (Op.ref .ptr [BK] "p_dq_inter"))
      (Op.ref .real [BK] "_dq") (MaskOpt.mask (Op.ref .bool [BK] "mask")),
    Stmt.assign .real [BK] "_dk1"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_dk_inner"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign .real [BK] "_dk2"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_dk_inter"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign .real [BK] "_dk2"
      (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "_dk2")
        (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BK] "last_g")
            (Op.ref .real [BK] "_g")).exp2),
    Stmt.assign .real [BK] "_dk"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [BK] "_dk1")
        (Op.ref .real [BK] "_dk2")),
    Stmt.store .real [BK] (MemAccess.ptr (Op.ref .ptr [BK] "p_dk_inter"))
      (Op.ref .real [BK] "_dk") (MaskOpt.mask (Op.ref .bool [BK] "mask")),
    Stmt.assign .real [BK] "_q"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_q"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign .real [BK] "_k"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BK] "p_k"))
        (MaskOpt.maskOther (Op.ref .bool [BK] "mask") ((Op.const 0).broadcast [BK]))),
    Stmt.assign .real [BK] "_dg"
      (Op.sub .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "_dq")
          (Op.ref .real [BK] "_q"))
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BK] "_dk")
          (Op.ref .real [BK] "_k"))),
    Stmt.assign .real [BK] "cum_grad_dg"
      (Op.add .real Broadcast.nil.consSame (Op.ref .real [BK] "cum_grad_dg")
        (Op.ref .real [BK] "_dg")),
    Stmt.store .real [BK] (MemAccess.ptr (Op.ref .ptr [BK] "p_dg"))
      (Op.ref .real [BK] "cum_grad_dg") (MaskOpt.mask (Op.ref .bool [BK] "mask")) ]

private def bwdTailG (DK BK : Nat) : List Stmt :=
  [ Stmt.assign .ptr [BK] "p_g"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_g") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_k"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_k") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_q"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_q") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_dq_inner"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dq_inner") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_dk_inner"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dk_inner") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_dq_inter"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dq_inter") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_dk_inter"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dk_inter") (Op.constNat DK)),
    Stmt.assign .ptr [BK] "p_dg"
      (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] "p_dg") (Op.constNat DK)) ]

private theorem bwdIterBodyG_decomp (DK BT BK : Nat) :
    bwdIterBodyG DK BT BK = bwdCoreG DK BT BK ++ bwdTailG DK BK := rfl

private theorem bwd_body_decomp_io
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) :
    (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
        s_qk_h DK BT BK).toAlgKernel.body
      = bwdPrologueG DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK
        ++ [Stmt.forRangeDyn "__rev_j" (Op.constNat 0) (bwdStopOp BT)
              (Op.constNat 1) (bwdCoreG DK BT BK ++ bwdTailG DK BK)] := by
  rw [bwd_body_decomp_general, bwd_prologue_take_general, ← bwdIterBodyG_decomp]
  rfl

private theorem bwdTailG_ptrDecTail (DK BK : Nat) :
    StmtList.PtrDecTail (bwdTailG DK BK) := by
  intro st hst
  simp only [bwdTailG, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    exact ⟨[BK], _, Broadcast.scalarR, DK, rfl⟩

private theorem bwd_prologue_flattenOkR
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) :
    StmtList.FlattenOkR
      (bwdPrologueG DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK) := by
  simp [bwdPrologueG, StmtList.FlattenOkR, Stmt.FlattenOkR, Op.FlattenOkR.eq_def]

private theorem bwd_core_flattenOkR (DK BT BK : Nat) :
    StmtList.FlattenOkR (bwdCoreG DK BT BK) := by
  simp [bwdCoreG, StmtList.FlattenOkR, Stmt.FlattenOkR, Op.FlattenOkR.eq_def]

private theorem bwd_stopOp_flattenOkR (BT : Nat) : (bwdStopOp BT).FlattenOkR := by
  simp [bwdStopOp, Op.FlattenOkR.eq_def]

private theorem bwdPrologue_castFree (R : RoundingModel)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) (t : BlockState) :
    stepStmtsR R (bwdPrologueG DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK) t
      = stepStmts (bwdPrologueG DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK) t := by
  simp only [bwdPrologueG, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

private theorem bwdIterBody_castFree (R : RoundingModel) (DK BT BK : Nat)
    (t : BlockState) :
    stepStmtsR R (bwdIterBodyG DK BT BK) t = stepStmts (bwdIterBodyG DK BT BK) t := by
  simp only [bwdIterBodyG, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real, BlockState.writeMemTypedR]
  rfl

private theorem bwdCore_castFree (R : RoundingModel) (DK BT BK : Nat)
    (t : BlockState) :
    stepStmtsR R (bwdCoreG DK BT BK) t = stepStmts (bwdCoreG DK BT BK) t := by
  simp only [bwdCoreG, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real, BlockState.writeMemTypedR]
  rfl

private theorem bwd_stopOpR_eval (R : RoundingModel) (c : BlockState)
    (BT : Nat) (hBT : 0 < BT) :
    evalOpR R (bwdStopOp BT) c = some (Tile.scalar BT) := by
  have h : evalOpR R (bwdStopOp BT) c = evalOp (bwdStopOp BT) c := by
    simp only [bwdStopOp, evalOpR.eq_def, evalOp.eq_def]
  rw [h]
  exact bwd_stopOp_eval c BT hBT

private theorem bwd_evalOpR_constNat (R : RoundingModel) (n : Nat) (s : BlockState) :
    evalOpR R (Op.constNat n) s = some (Tile.scalar n) := by
  simp [evalOpR]

def bwdDecayCumsumKernelIO
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) : StreamGroupedEmitMasked3DKernelIO where
  kernel := bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
    s_qk_h DK BT BK
  nIn := 7
  nOut := 3
  bufs := [DQInner, DQInter, DKInner, DKInter, Q, K, G, DG]
  inp := fun i => match i with
    | ⟨0, _⟩ => G
    | ⟨1, _⟩ => DQInner
    | ⟨2, _⟩ => DQInter
    | ⟨3, _⟩ => DKInner
    | ⟨4, _⟩ => DKInter
    | ⟨5, _⟩ => Q
    | ⟨6, _⟩ => K
    | ⟨n + 7, h⟩ => absurd h (by omega)
  out := fun o => match o with
    | ⟨0, _⟩ => DQInter
    | ⟨1, _⟩ => DKInter
    | ⟨2, _⟩ => DG
    | ⟨n + 3, h⟩ => absurd h (by omega)
  T := BT
  B := BK
  read := fun _ p₀ p₁ p₂ t j =>
    p₂ * s_qk_h + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  readMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK
  write := fun _ p₀ p₁ p₂ t j =>
    p₂ * s_qk_h + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  writeMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK

private noncomputable def bwdDqStream (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  xs (⟨1, by omega⟩ : Fin 7) t j
    + xs (⟨2, by omega⟩ : Fin 7) t j
      * Real.exp (xs (⟨0, by omega⟩ : Fin 7) t j * Real.log 2)

private noncomputable def bwdDkStream (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  if h : 0 < BT then
    xs (⟨3, by omega⟩ : Fin 7) t j
      + xs (⟨4, by omega⟩ : Fin 7) t j
        * Real.exp
            ((xs (⟨0, by omega⟩ : Fin 7) ⟨BT - 1, Nat.sub_lt h Nat.one_pos⟩ j
                - xs (⟨0, by omega⟩ : Fin 7) t j) * Real.log 2)
  else 0

private noncomputable def bwdDgStream (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  ∑ d : Fin (BT - t.val),
    (bwdDqStream BT BK xs ⟨t.val + d.val, by omega⟩ j
        * xs (⟨5, by omega⟩ : Fin 7) ⟨t.val + d.val, by omega⟩ j
      - bwdDkStream BT BK xs ⟨t.val + d.val, by omega⟩ j
        * xs (⟨6, by omega⟩ : Fin 7) ⟨t.val + d.val, by omega⟩ j)

noncomputable def bwdDecayStreamSpec (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (o : Fin 3) (t : Fin BT) (j : Fin BK) : ℝ :=
  match o with
  | ⟨0, _⟩ => bwdDqStream BT BK xs t j
  | ⟨1, _⟩ => bwdDkStream BT BK xs t j
  | ⟨2, _⟩ => bwdDgStream BT BK xs t j
  | ⟨n + 3, h3⟩ => absurd h3 (by omega)

private theorem bwdDqStream_eq_closed (DQInner DQInter G : RegionName)
    (s₀ : BlockState) (s_qk_h DK BT BK : Nat) (xs : Fin 7 → Fin BT → Fin BK → ℝ)
    (hxG : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem G
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨0, by omega⟩ : Fin 7) t j)
    (hxDQi : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem DQInner
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨1, by omega⟩ : Fin 7) t j)
    (hxDQt : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem DQInter
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨2, by omega⟩ : Fin 7) t j)
    (t : Fin BT) (j : Fin BK) (hj : active s₀ DK BK j) :
    bwdDqStream BT BK xs t j
      = bwdDQInterClosed s₀ DQInner DQInter G s_qk_h DK BT BK t j := by
  have hg := hxG t j hj
  have h1 := hxDQi t j hj
  have h2 := hxDQt t j hj
  rw [fwd_addr_eq] at hg h1 h2
  show xs (⟨1, by omega⟩ : Fin 7) t j
      + xs (⟨2, by omega⟩ : Fin 7) t j
        * Real.exp (xs (⟨0, by omega⟩ : Fin 7) t j * Real.log 2) = _
  rw [← hg, ← h1, ← h2]
  simp only [bwdDQInterClosed]

private theorem bwdDkStream_eq_closed (DKInner DKInter G : RegionName)
    (s₀ : BlockState) (s_qk_h DK BT BK : Nat) (xs : Fin 7 → Fin BT → Fin BK → ℝ)
    (hBT : 0 < BT)
    (hxG : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem G
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨0, by omega⟩ : Fin 7) t j)
    (hxDKi : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem DKInner
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨3, by omega⟩ : Fin 7) t j)
    (hxDKt : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem DKInter
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨4, by omega⟩ : Fin 7) t j)
    (t : Fin BT) (j : Fin BK) (hj : active s₀ DK BK j) :
    bwdDkStream BT BK xs t j
      = bwdDKInterClosed s₀ DKInner DKInter G s_qk_h DK BT BK t j := by
  have hg := hxG t j hj
  have h3 := hxDKi t j hj
  have h4 := hxDKt t j hj
  have hglast := hxG ⟨BT - 1, Nat.sub_lt hBT Nat.one_pos⟩ j hj
  rw [fwd_addr_eq] at hg h3 h4 hglast
  show (if h : 0 < BT then
      xs (⟨3, by omega⟩ : Fin 7) t j
        + xs (⟨4, by omega⟩ : Fin 7) t j
          * Real.exp
              ((xs (⟨0, by omega⟩ : Fin 7) ⟨BT - 1, Nat.sub_lt h Nat.one_pos⟩ j
                  - xs (⟨0, by omega⟩ : Fin 7) t j) * Real.log 2)
    else 0) = _
  rw [dif_pos hBT, ← hg, ← h3, ← h4, ← hglast]
  simp only [bwdDKInterClosed]

private theorem bwdDgStream_eq_closed
    (DQInner DQInter DKInner DKInter Q K G : RegionName)
    (s₀ : BlockState) (s_qk_h DK BT BK : Nat) (xs : Fin 7 → Fin BT → Fin BK → ℝ)
    (hBT : 0 < BT)
    (hxG : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem G
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨0, by omega⟩ : Fin 7) t j)
    (hxDQi : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem DQInner
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨1, by omega⟩ : Fin 7) t j)
    (hxDQt : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem DQInter
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨2, by omega⟩ : Fin 7) t j)
    (hxDKi : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem DKInner
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨3, by omega⟩ : Fin 7) t j)
    (hxDKt : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem DKInter
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨4, by omega⟩ : Fin 7) t j)
    (hxQ : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem Q
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨5, by omega⟩ : Fin 7) t j)
    (hxK : ∀ (t : Fin BT) (j : Fin BK), s₀.pids 0 * BK + j.val < DK →
      s₀.readMem K
          (s₀.pids 2 * s_qk_h + s₀.pids 1 * BT * DK + s₀.pids 0 * BK + j.val + t.val * DK)
        = xs (⟨6, by omega⟩ : Fin 7) t j)
    (t : Fin BT) (j : Fin BK) (hj : active s₀ DK BK j) :
    bwdDgStream BT BK xs t j
      = bwdDGClosed s₀ DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK t j := by
  simp only [bwdDgStream, bwdDGClosed]
  refine Finset.sum_congr rfl fun d _ => ?_
  have hq := hxQ ⟨t.val + d.val, by omega⟩ j hj
  have hk := hxK ⟨t.val + d.val, by omega⟩ j hj
  rw [fwd_addr_eq] at hq hk
  simp only [bwdDGSummand]
  rw [bwdDqStream_eq_closed DQInner DQInter G s₀ s_qk_h DK BT BK xs hxG hxDQi hxDQt
      ⟨t.val + d.val, by omega⟩ j hj,
    bwdDkStream_eq_closed DKInner DKInter G s₀ s_qk_h DK BT BK xs hBT hxG hxDKi hxDKt
      ⟨t.val + d.val, by omega⟩ j hj,
    ← hq, ← hk]

private theorem stepStmts_cons_ifThen_assign_inv {cond : Op TileDType.bool []}
    {dt : TileDType} {sh : TileShape} {nm : RegName} {e : Op dt sh}
    {rest : List Stmt} {s s' : BlockState}
    (h : stepStmts (Stmt.ifThen cond [Stmt.assign dt sh nm e] :: rest) s = some s') :
    ∃ s₂, stepStmts rest s₂ = some s' ∧ s₂.mem = s.mem ∧
      ∀ (d' : TileDType) (sh' : TileShape) (n' : RegName), n' ≠ nm →
        s₂.regs d' sh' n' = s.regs d' sh' n' := by
  cases hst : stepStmt (Stmt.ifThen cond [Stmt.assign dt sh nm e]) s with
  | none => simp [stepStmts, hst] at h
  | some s₂ =>
      rw [stepStmts.cons_some hst] at h
      refine ⟨s₂, h, ?_⟩
      simp only [stepStmt] at hst
      cases hc : evalOp cond s with
      | none => rw [hc] at hst; exact absurd hst (by simp)
      | some vc =>
          rw [hc] at hst
          replace hst : (if vc.data PUnit.unit = Bool.true then
              stepStmts [Stmt.assign dt sh nm e] s else some s) = some s₂ := hst
          by_cases hb : vc.data PUnit.unit = Bool.true
          · rw [if_pos hb] at hst
            obtain ⟨v, hv, hnil⟩ := stepStmts_cons_assign_inv hst
            rw [stepStmts.nil] at hnil
            obtain rfl := Option.some.inj hnil
            exact ⟨rfl, fun d' sh' n' hne => by
              rw [BlockState.setReg_ne_name (h := hne)]⟩
          · rw [if_neg hb] at hst
            obtain rfl := Option.some.inj hst
            exact ⟨rfl, fun _ _ _ _ => rfl⟩

private theorem stepStmtsR_cons_inv {R : RoundingModel} {st : Stmt}
    {rest : List Stmt} {s s' : BlockState}
    (h : stepStmtsR R (st :: rest) s = some s') :
    ∃ s₂, stepStmtR R st s = some s₂ ∧ stepStmtsR R rest s₂ = some s' := by
  simp only [stepStmtsR] at h
  cases hst : stepStmtR R st s with
  | none => rw [hst] at h; exact absurd h (by simp)
  | some s₂ =>
      rw [hst] at h
      exact ⟨s₂, rfl, h⟩

private theorem stepStmtsR_cons_assign_inv {R : RoundingModel} {dt : TileDType}
    {sh : TileShape} {nm : RegName} {e : Op dt sh} {rest : List Stmt}
    {s s' : BlockState}
    (h : stepStmtsR R (Stmt.assign dt sh nm e :: rest) s = some s') :
    ∃ v, evalOpR R e s = some v ∧
      stepStmtsR R rest (s.setReg nm dt sh v) = some s' := by
  obtain ⟨s₂, hst, h⟩ := stepStmtsR_cons_inv h
  obtain ⟨v, hv, rfl⟩ := stepStmtR_assign_inv hst
  exact ⟨v, hv, h⟩

private theorem stepStmtsR_cons_store_regs {R : RoundingModel} {sh : TileShape}
    {pe : Op TileDType.ptr sh} {ve : Op TileDType.real sh}
    {me : Op TileDType.bool sh} {rest : List Stmt} {s s' : BlockState}
    (h : stepStmtsR R (Stmt.store TileDType.real sh (MemAccess.ptr pe) ve
        (MaskOpt.mask me) :: rest) s = some s') :
    ∃ s₂, stepStmtsR R rest s₂ = some s' ∧ s₂.regs = s.regs := by
  obtain ⟨s₂, hst, h⟩ := stepStmtsR_cons_inv h
  exact ⟨s₂, h, stepStmtR_store_real_regs hst⟩

private theorem stepStmtR_ifThen_assign_regs {R : RoundingModel}
    {cond : Op TileDType.bool []} {dt : TileDType} {sh : TileShape}
    {nm : RegName} {e : Op dt sh} {s s' : BlockState}
    (h : stepStmtR R (Stmt.ifThen cond [Stmt.assign dt sh nm e]) s = some s') :
    ∀ (d' : TileDType) (sh' : TileShape) (n' : RegName), n' ≠ nm →
      s'.regs d' sh' n' = s.regs d' sh' n' := by
  simp only [stepStmtR] at h
  cases hc : evalOpR R cond s with
  | none => rw [hc] at h; exact absurd h (by simp)
  | some vc =>
      rw [hc] at h
      replace h : (if vc.data PUnit.unit = Bool.true then
          stepStmtsR R [Stmt.assign dt sh nm e] s else some s) = some s' := h
      by_cases hb : vc.data PUnit.unit = Bool.true
      · rw [if_pos hb] at h
        obtain ⟨v, hv, hnil⟩ := stepStmtsR_cons_assign_inv h
        rw [stepStmtsR_nil] at hnil
        obtain rfl := Option.some.inj hnil
        exact fun d' sh' n' hne => by rw [BlockState.setReg_ne_name (h := hne)]
      · rw [if_neg hb] at h
        obtain rfl := Option.some.inj h
        exact fun _ _ _ _ => rfl

set_option maxHeartbeats 8000000 in

private theorem bwd_body_step_frame
    (DQInter DKInter DG : RegionName) (s sc s' : BlockState)
    (s_qk_h DK BT BK rr m : Nat)
    (hpdqt : sc.regs .ptr [BK] "p_dq_inter"
      = some (bwdPtrTileG s DQInter s_qk_h DK BT BK rr))
    (hpdkt : sc.regs .ptr [BK] "p_dk_inter"
      = some (bwdPtrTileG s DKInter s_qk_h DK BT BK rr))
    (hpdg : sc.regs .ptr [BK] "p_dg"
      = some (bwdPtrTileG s DG s_qk_h DK BT BK rr))
    (hmask : sc.regs .bool [BK] "mask" = some (bwdMaskTileG s DK BK))
    (hstep : stepStmts (bwdIterBodyG DK BT BK)
      (sc.setReg "__rev_j" .nat [] (Tile.scalar m)) = some s')
    (r : RegionName) (oo : Nat)
    (hcDQ : r ≠ DQInter ∨ ∀ i : Fin BK, active s DK BK i →
      oo ≠ offset s s_qk_h DK rr BT BK i)
    (hcDK : r ≠ DKInter ∨ ∀ i : Fin BK, active s DK BK i →
      oo ≠ offset s s_qk_h DK rr BT BK i)
    (hcDG : r ≠ DG ∨ ∀ i : Fin BK, active s DK BK i →
      oo ≠ offset s s_qk_h DK rr BT BK i) :
    s'.mem r oo = sc.mem r oo := by
  set sloop := sc.setReg "__rev_j" .nat [] (Tile.scalar m) with hsloop
  unfold bwdIterBodyG at hstep
  
  obtain ⟨v0, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨v1, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  
  obtain ⟨s₂, hstep, hmem₂, hregs₂⟩ := stepStmts_cons_ifThen_assign_inv hstep
  
  obtain ⟨v3, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨v4, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨v5, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨v6, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  set s6 := (((s₂.setReg "_dq1" .real [BK] v3).setReg "_dq2" .real [BK] v4).setReg
    "_dq2" .real [BK] v5).setReg "_dq" .real [BK] v6 with hs6
  have hdq6 : s6.regs .real [BK] "_dq" = some v6 := by
    rw [hs6]; exact BlockState.setReg_same _ _ _ _ _
  have hpdqt6 : s6.regs .ptr [BK] "p_dq_inter"
      = some (bwdPtrTileG s DQInter s_qk_h DK BT BK rr) := by
    rw [hs6, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      hregs₂ _ _ _ (by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide), hsloop,
      BlockState.setReg_ne_name (h := by decide)]
    exact hpdqt
  have hmask6 : s6.regs .bool [BK] "mask" = some (bwdMaskTileG s DK BK) := by
    rw [hs6, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      hregs₂ _ _ _ (by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide), hsloop,
      BlockState.setReg_ne_name (h := by decide)]
    exact hmask
  
  set vfun1 : TileIndex [BK] → ℝ := fun k => (v6.data k).unbotD 0 with hvfun1
  set sst1 := (TileShape.allIndices [BK]).foldl
    (fun (acc : BlockState) k =>
      if active s DK BK k.1 then
        acc.writeMem DQInter (offset s s_qk_h DK rr BT BK k.1) (vfun1 k)
      else acc) s6 with hsst1
  have hstore1 : stepStmt (Stmt.store TileDType.real [BK]
      (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_dq_inter"))
      (Op.ref TileDType.real [BK] "_dq")
      (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask"))) s6 = some sst1 := by
    simp only [stepStmt, evalOp, Option.bind, Option.map, hdq6, hpdqt6, hmask6]
    apply congrArg some
    rw [hsst1]
    congr 1
    funext acc k
    simp only [bwdPtrTileG, bwdMaskTileG]
    by_cases ha : active s DK BK k.1
    · rw [if_pos (by simpa using ha), if_pos ha, BlockState.writeMemTyped_real]
      rfl
    · rw [if_neg (by simpa using ha), if_neg ha]
  rw [stepStmts.cons_some hstore1] at hstep
  
  obtain ⟨w1, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨w2, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨w3, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨w4, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  set s11 := (((sst1.setReg "_dk1" .real [BK] w1).setReg "_dk2" .real [BK] w2).setReg
    "_dk2" .real [BK] w3).setReg "_dk" .real [BK] w4 with hs11
  have hdk11 : s11.regs .real [BK] "_dk" = some w4 := by
    rw [hs11]; exact BlockState.setReg_same _ _ _ _ _
  have hpdkt11 : s11.regs .ptr [BK] "p_dk_inter"
      = some (bwdPtrTileG s DKInter s_qk_h DK BT BK rr) := by
    rw [hs11, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      hsst1, BlockState.foldl_writeMem_prop_masked_regs,
      hs6, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      hregs₂ _ _ _ (by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide), hsloop,
      BlockState.setReg_ne_name (h := by decide)]
    exact hpdkt
  have hmask11 : s11.regs .bool [BK] "mask" = some (bwdMaskTileG s DK BK) := by
    rw [hs11, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      hsst1, BlockState.foldl_writeMem_prop_masked_regs]
    exact hmask6
  
  set vfun2 : TileIndex [BK] → ℝ := fun k => (w4.data k).unbotD 0 with hvfun2
  set sst2 := (TileShape.allIndices [BK]).foldl
    (fun (acc : BlockState) k =>
      if active s DK BK k.1 then
        acc.writeMem DKInter (offset s s_qk_h DK rr BT BK k.1) (vfun2 k)
      else acc) s11 with hsst2
  have hstore2 : stepStmt (Stmt.store TileDType.real [BK]
      (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_dk_inter"))
      (Op.ref TileDType.real [BK] "_dk")
      (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask"))) s11 = some sst2 := by
    simp only [stepStmt, evalOp, Option.bind, Option.map, hdk11, hpdkt11, hmask11]
    apply congrArg some
    rw [hsst2]
    congr 1
    funext acc k
    simp only [bwdPtrTileG, bwdMaskTileG]
    by_cases ha : active s DK BK k.1
    · rw [if_pos (by simpa using ha), if_pos ha, BlockState.writeMemTyped_real]
      rfl
    · rw [if_neg (by simpa using ha), if_neg ha]
  rw [stepStmts.cons_some hstore2] at hstep
  
  obtain ⟨x1, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨x2, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨x3, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨x4, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  set s16 := (((sst2.setReg "_q" .real [BK] x1).setReg "_k" .real [BK] x2).setReg
    "_dg" .real [BK] x3).setReg "cum_grad_dg" .real [BK] x4 with hs16
  have hcum16 : s16.regs .real [BK] "cum_grad_dg" = some x4 := by
    rw [hs16]; exact BlockState.setReg_same _ _ _ _ _
  have hpdg16 : s16.regs .ptr [BK] "p_dg"
      = some (bwdPtrTileG s DG s_qk_h DK BT BK rr) := by
    rw [hs16, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      hsst2, BlockState.foldl_writeMem_prop_masked_regs,
      hs11, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      hsst1, BlockState.foldl_writeMem_prop_masked_regs,
      hs6, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      hregs₂ _ _ _ (by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide), hsloop,
      BlockState.setReg_ne_name (h := by decide)]
    exact hpdg
  have hmask16 : s16.regs .bool [BK] "mask" = some (bwdMaskTileG s DK BK) := by
    rw [hs16, BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      hsst2, BlockState.foldl_writeMem_prop_masked_regs]
    exact hmask11
  
  set vfun3 : TileIndex [BK] → ℝ := fun k => (x4.data k).unbotD 0 with hvfun3
  set sst3 := (TileShape.allIndices [BK]).foldl
    (fun (acc : BlockState) k =>
      if active s DK BK k.1 then
        acc.writeMem DG (offset s s_qk_h DK rr BT BK k.1) (vfun3 k)
      else acc) s16 with hsst3
  have hstore3 : stepStmt (Stmt.store TileDType.real [BK]
      (MemAccess.ptr (Op.ref TileDType.ptr [BK] "p_dg"))
      (Op.ref TileDType.real [BK] "cum_grad_dg")
      (MaskOpt.mask (Op.ref TileDType.bool [BK] "mask"))) s16 = some sst3 := by
    simp only [stepStmt, evalOp, Option.bind, Option.map, hcum16, hpdg16, hmask16]
    apply congrArg some
    rw [hsst3]
    congr 1
    funext acc k
    simp only [bwdPtrTileG, bwdMaskTileG]
    by_cases ha : active s DK BK k.1
    · rw [if_pos (by simpa using ha), if_pos ha, BlockState.writeMemTyped_real]
      rfl
    · rw [if_neg (by simpa using ha), if_neg ha]
  rw [stepStmts.cons_some hstore3] at hstep
  
  obtain ⟨u1, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨u2, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨u3, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨u4, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨u5, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨u6, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨u7, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  obtain ⟨u8, -, hstep⟩ := stepStmts_cons_assign_inv hstep
  rw [stepStmts.nil] at hstep
  obtain rfl := Option.some.inj hstep
  simp only [BlockState.setReg_mem]
  rw [hsst3]
  refine Eq.trans (foldl_writeMem_prop_preserve_cell _ _ _ r oo _ _ ?_) ?_
  · intro k hk hmk hbad
    rcases hcDG with hne' | hno
    · exact hne' hbad.1
    · exact hno k.1 hmk hbad.2
  · rw [hs16]
    simp only [BlockState.setReg_mem]
    rw [hsst2]
    refine Eq.trans (foldl_writeMem_prop_preserve_cell _ _ _ r oo _ _ ?_) ?_
    · intro k hk hmk hbad
      rcases hcDK with hne' | hno
      · exact hne' hbad.1
      · exact hno k.1 hmk hbad.2
    · rw [hs11]
      simp only [BlockState.setReg_mem]
      rw [hsst1]
      refine Eq.trans (foldl_writeMem_prop_preserve_cell _ _ _ r oo _ _ ?_) ?_
      · intro k hk hmk hbad
        rcases hcDQ with hne' | hno
        · exact hne' hbad.1
        · exact hno k.1 hmk hbad.2
      · rw [hs6]
        simp only [BlockState.setReg_mem]
        have h₂ : s₂.mem r oo = sloop.mem r oo := by
          rw [hmem₂]
          simp only [BlockState.setReg_mem]
        rw [h₂, hsloop]
        simp only [BlockState.setReg_mem]

private theorem bwd_core_regsR (R : RoundingModel) (DK BT BK : Nat)
    (sin sMid : BlockState)
    (h : stepStmtsR R (bwdCoreG DK BT BK) sin = some sMid)
    (dt : TileDType) (sh : TileShape) (nm : RegName)
    (h1 : nm ≠ "j") (h2 : nm ≠ "_g") (h3 : nm ≠ "last_g") (h4 : nm ≠ "_dq1")
    (h5 : nm ≠ "_dq2") (h6 : nm ≠ "_dq") (h7 : nm ≠ "_dk1") (h8 : nm ≠ "_dk2")
    (h9 : nm ≠ "_dk") (h10 : nm ≠ "_q") (h11 : nm ≠ "_k")
    (h12 : nm ≠ "_dg") (h13 : nm ≠ "cum_grad_dg") :
    sMid.regs dt sh nm = sin.regs dt sh nm := by
  unfold bwdCoreG at h
  obtain ⟨v0, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨v1, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨s₂, hif, h⟩ := stepStmtsR_cons_inv h
  have hregs₂ := stepStmtR_ifThen_assign_regs hif
  obtain ⟨v3, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨v4, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨v5, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨v6, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨s7, h, hregs7⟩ := stepStmtsR_cons_store_regs h
  obtain ⟨w1, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨w2, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨w3, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨w4, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨s12, h, hregs12⟩ := stepStmtsR_cons_store_regs h
  obtain ⟨x1, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨x2, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨x3, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨x4, -, h⟩ := stepStmtsR_cons_assign_inv h
  obtain ⟨s17, h, hregs17⟩ := stepStmtsR_cons_store_regs h
  rw [stepStmtsR_nil] at h
  obtain rfl := Option.some.inj h
  rw [hregs17, BlockState.setReg_ne_name (h := h13),
    BlockState.setReg_ne_name (h := h12), BlockState.setReg_ne_name (h := h11),
    BlockState.setReg_ne_name (h := h10), hregs12,
    BlockState.setReg_ne_name (h := h9), BlockState.setReg_ne_name (h := h8),
    BlockState.setReg_ne_name (h := h8), BlockState.setReg_ne_name (h := h7),
    hregs7, BlockState.setReg_ne_name (h := h6),
    BlockState.setReg_ne_name (h := h5), BlockState.setReg_ne_name (h := h5),
    BlockState.setReg_ne_name (h := h4), hregs₂ _ _ _ h3,
    BlockState.setReg_ne_name (h := h2), BlockState.setReg_ne_name (h := h1)]

private theorem bwd_tailSafeR (R : RoundingModel) (bounds : RegionBounds)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s sMid : BlockState) (s_qk_h DK BT BK rr : Nat) (hrr : 1 ≤ rr)
    (hpg : sMid.regs .ptr [BK] "p_g" = some (bwdPtrTileG s G s_qk_h DK BT BK rr))
    (hpk : sMid.regs .ptr [BK] "p_k" = some (bwdPtrTileG s K s_qk_h DK BT BK rr))
    (hpq : sMid.regs .ptr [BK] "p_q" = some (bwdPtrTileG s Q s_qk_h DK BT BK rr))
    (hpdqi : sMid.regs .ptr [BK] "p_dq_inner"
      = some (bwdPtrTileG s DQInner s_qk_h DK BT BK rr))
    (hpdki : sMid.regs .ptr [BK] "p_dk_inner"
      = some (bwdPtrTileG s DKInner s_qk_h DK BT BK rr))
    (hpdqt : sMid.regs .ptr [BK] "p_dq_inter"
      = some (bwdPtrTileG s DQInter s_qk_h DK BT BK rr))
    (hpdkt : sMid.regs .ptr [BK] "p_dk_inter"
      = some (bwdPtrTileG s DKInter s_qk_h DK BT BK rr))
    (hpdg : sMid.regs .ptr [BK] "p_dg"
      = some (bwdPtrTileG s DG s_qk_h DK BT BK rr)) :
    Stmt.TraceSafeListR R bounds (bwdTailG DK BK) sMid := by
  have hDKle : ∀ i : Fin BK, DK ≤ offset s s_qk_h DK rr BT BK i := by
    intro i
    have h1 : DK ≤ (s.pids 1 * BT + rr) * DK :=
      Nat.le_mul_of_pos_left DK (by omega)
    simp only [offset, baseOffset]
    omega
  have hsafe : ∀ (pname : RegName) (reg : RegionName) (st : BlockState),
      st.regs .ptr [BK] pname = some (bwdPtrTileG s reg s_qk_h DK BT BK rr) →
      Stmt.TraceSafeR R bounds (Stmt.assign .ptr [BK] pname
        (Op.ptrSub Broadcast.scalarR (Op.ref .ptr [BK] pname)
          (Op.constNat DK))) st := by
    intro pname reg st hpin
    simp only [Stmt.TraceSafeR, Op.SafeAtR]
    refine ⟨trivial, trivial, ?_⟩
    intro ptrs offs hp ho i
    rw [evalOpR_ref, hpin] at hp
    obtain rfl := Option.some.inj hp
    rw [bwd_evalOpR_constNat] at ho
    obtain rfl := Option.some.inj ho
    exact hDKle (Broadcast.scalarR.leftIndex i).1
  unfold bwdTailG
  refine Stmt.TraceSafeListR.cons_intro (hsafe "p_g" G sMid hpg)
    (fun t1 ht1 => ?_)
  obtain ⟨q1, -, rfl⟩ := stepStmtR_assign_inv ht1
  refine Stmt.TraceSafeListR.cons_intro (hsafe "p_k" K _ (by
    rw [BlockState.setReg_ne_name (h := by decide)]; exact hpk))
    (fun t2 ht2 => ?_)
  obtain ⟨q2, -, rfl⟩ := stepStmtR_assign_inv ht2
  refine Stmt.TraceSafeListR.cons_intro (hsafe "p_q" Q _ (by
    rw [BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide)]; exact hpq))
    (fun t3 ht3 => ?_)
  obtain ⟨q3, -, rfl⟩ := stepStmtR_assign_inv ht3
  refine Stmt.TraceSafeListR.cons_intro (hsafe "p_dq_inner" DQInner _ (by
    rw [BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide)]; exact hpdqi))
    (fun t4 ht4 => ?_)
  obtain ⟨q4, -, rfl⟩ := stepStmtR_assign_inv ht4
  refine Stmt.TraceSafeListR.cons_intro (hsafe "p_dk_inner" DKInner _ (by
    rw [BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide)]; exact hpdki))
    (fun t5 ht5 => ?_)
  obtain ⟨q5, -, rfl⟩ := stepStmtR_assign_inv ht5
  refine Stmt.TraceSafeListR.cons_intro (hsafe "p_dq_inter" DQInter _ (by
    rw [BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide)]; exact hpdqt))
    (fun t6 ht6 => ?_)
  obtain ⟨q6, -, rfl⟩ := stepStmtR_assign_inv ht6
  refine Stmt.TraceSafeListR.cons_intro (hsafe "p_dk_inter" DKInter _ (by
    rw [BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide)]; exact hpdkt))
    (fun t7 ht7 => ?_)
  obtain ⟨q7, -, rfl⟩ := stepStmtR_assign_inv ht7
  refine Stmt.TraceSafeListR.cons_intro (hsafe "p_dg" DG _ (by
    rw [BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide),
      BlockState.setReg_ne_name (h := by decide)]; exact hpdg))
    (fun _ _ => Stmt.TraceSafeListR.nil_intro)

set_option maxHeartbeats 8000000 in

private theorem bwd_bodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s stt : BlockState) (s_qk_h DK BT BK rr m : Nat)
    (hpg : stt.regs .ptr [BK] "p_g" = some (bwdPtrTileG s G s_qk_h DK BT BK rr))
    (hpk : stt.regs .ptr [BK] "p_k" = some (bwdPtrTileG s K s_qk_h DK BT BK rr))
    (hpq : stt.regs .ptr [BK] "p_q" = some (bwdPtrTileG s Q s_qk_h DK BT BK rr))
    (hpdqi : stt.regs .ptr [BK] "p_dq_inner"
      = some (bwdPtrTileG s DQInner s_qk_h DK BT BK rr))
    (hpdki : stt.regs .ptr [BK] "p_dk_inner"
      = some (bwdPtrTileG s DKInner s_qk_h DK BT BK rr))
    (hpdqt : stt.regs .ptr [BK] "p_dq_inter"
      = some (bwdPtrTileG s DQInter s_qk_h DK BT BK rr))
    (hpdkt : stt.regs .ptr [BK] "p_dk_inter"
      = some (bwdPtrTileG s DKInter s_qk_h DK BT BK rr))
    (hpdg : stt.regs .ptr [BK] "p_dg"
      = some (bwdPtrTileG s DG s_qk_h DK BT BK rr))
    (hmask : stt.regs .bool [BK] "mask" = some (bwdMaskTileG s DK BK))
    (hbG : ∀ i : Fin BK, active s DK BK i → offset s s_qk_h DK rr BT BK i < bounds G)
    (hbQ : ∀ i : Fin BK, active s DK BK i → offset s s_qk_h DK rr BT BK i < bounds Q)
    (hbK : ∀ i : Fin BK, active s DK BK i → offset s s_qk_h DK rr BT BK i < bounds K)
    (hbDQi : ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK rr BT BK i < bounds DQInner)
    (hbDKi : ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK rr BT BK i < bounds DKInner)
    (hbDQt : ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK rr BT BK i < bounds DQInter)
    (hbDKt : ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK rr BT BK i < bounds DKInter)
    (hbDG : ∀ i : Fin BK, active s DK BK i →
      offset s s_qk_h DK rr BT BK i < bounds DG) :
    Stmt.TraceSafeListR R bounds (bwdCoreG DK BT BK)
      (stt.setReg "__rev_j" .nat [] (Tile.scalar m)) := by
  set sloop := stt.setReg "__rev_j" .nat [] (Tile.scalar m) with hsloop
  have hpg' : sloop.regs .ptr [BK] "p_g"
      = some (bwdPtrTileG s G s_qk_h DK BT BK rr) := by
    rw [hsloop]; simpa using hpg
  have hpk' : sloop.regs .ptr [BK] "p_k"
      = some (bwdPtrTileG s K s_qk_h DK BT BK rr) := by
    rw [hsloop]; simpa using hpk
  have hpq' : sloop.regs .ptr [BK] "p_q"
      = some (bwdPtrTileG s Q s_qk_h DK BT BK rr) := by
    rw [hsloop]; simpa using hpq
  have hpdqi' : sloop.regs .ptr [BK] "p_dq_inner"
      = some (bwdPtrTileG s DQInner s_qk_h DK BT BK rr) := by
    rw [hsloop]; simpa using hpdqi
  have hpdki' : sloop.regs .ptr [BK] "p_dk_inner"
      = some (bwdPtrTileG s DKInner s_qk_h DK BT BK rr) := by
    rw [hsloop]; simpa using hpdki
  have hpdqt' : sloop.regs .ptr [BK] "p_dq_inter"
      = some (bwdPtrTileG s DQInter s_qk_h DK BT BK rr) := by
    rw [hsloop]; simpa using hpdqt
  have hpdkt' : sloop.regs .ptr [BK] "p_dk_inter"
      = some (bwdPtrTileG s DKInter s_qk_h DK BT BK rr) := by
    rw [hsloop]; simpa using hpdkt
  have hpdg' : sloop.regs .ptr [BK] "p_dg"
      = some (bwdPtrTileG s DG s_qk_h DK BT BK rr) := by
    rw [hsloop]; simpa using hpdg
  have hmask' : sloop.regs .bool [BK] "mask" = some (bwdMaskTileG s DK BK) := by
    rw [hsloop]; simpa using hmask
  unfold bwdCoreG
  
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t0 ht0 => ?_)
  obtain ⟨v0, -, rfl⟩ := stepStmtR_assign_inv ht0
  have hpg0 : (sloop.setReg "j" .nat [] v0).regs .ptr [BK] "p_g"
      = some (bwdPtrTileG s G s_qk_h DK BT BK rr) := by
    rw [BlockState.setReg_ne_name (h := by decide)]; exact hpg'
  have hmask0 : (sloop.setReg "j" .nat [] v0).regs .bool [BK] "mask"
      = some (bwdMaskTileG s DK BK) := by
    rw [BlockState.setReg_ne_name (h := by decide)]; exact hmask'
  
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t1 ht1 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def], ?_⟩
    intro ptrs hptrs idx hactive
    rw [evalOpR_ref, hpg0] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [evalOpR_ref, hmask0] at hm
    obtain rfl := Option.some.inj hm
    have ha : active s DK BK idx.1 := by simpa [bwdMaskTileG] using hmi
    simpa [bwdPtrTileG] using hbG idx.1 ha
  · obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv ht1
    
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun t2 ht2 => ?_)
    · simp only [Stmt.TraceSafeR]
      refine ⟨by simp [Op.SafeAtR.eq_def], ?_⟩
      split
      · split
        · refine Stmt.TraceSafeListR.of_forall _ _ ?_
          intro st hst s'
          simp only [List.mem_cons, List.not_mem_nil, or_false] at hst
          subst hst
          simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
        · trivial
      · trivial
    · have hregs₂ := stepStmtR_ifThen_assign_regs ht2
      
      have h2pdqi : t2.regs .ptr [BK] "p_dq_inner"
          = some (bwdPtrTileG s DQInner s_qk_h DK BT BK rr) := by
        rw [hregs₂ _ _ _ (by decide), BlockState.setReg_ne_name (h := by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpdqi'
      have h2pdqt : t2.regs .ptr [BK] "p_dq_inter"
          = some (bwdPtrTileG s DQInter s_qk_h DK BT BK rr) := by
        rw [hregs₂ _ _ _ (by decide), BlockState.setReg_ne_name (h := by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpdqt'
      have h2pdki : t2.regs .ptr [BK] "p_dk_inner"
          = some (bwdPtrTileG s DKInner s_qk_h DK BT BK rr) := by
        rw [hregs₂ _ _ _ (by decide), BlockState.setReg_ne_name (h := by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpdki'
      have h2pdkt : t2.regs .ptr [BK] "p_dk_inter"
          = some (bwdPtrTileG s DKInter s_qk_h DK BT BK rr) := by
        rw [hregs₂ _ _ _ (by decide), BlockState.setReg_ne_name (h := by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpdkt'
      have h2pq : t2.regs .ptr [BK] "p_q"
          = some (bwdPtrTileG s Q s_qk_h DK BT BK rr) := by
        rw [hregs₂ _ _ _ (by decide), BlockState.setReg_ne_name (h := by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpq'
      have h2pk : t2.regs .ptr [BK] "p_k"
          = some (bwdPtrTileG s K s_qk_h DK BT BK rr) := by
        rw [hregs₂ _ _ _ (by decide), BlockState.setReg_ne_name (h := by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpk'
      have h2pdg : t2.regs .ptr [BK] "p_dg"
          = some (bwdPtrTileG s DG s_qk_h DK BT BK rr) := by
        rw [hregs₂ _ _ _ (by decide), BlockState.setReg_ne_name (h := by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpdg'
      have h2mask : t2.regs .bool [BK] "mask" = some (bwdMaskTileG s DK BK) := by
        rw [hregs₂ _ _ _ (by decide), BlockState.setReg_ne_name (h := by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hmask'
      
      refine Stmt.TraceSafeListR.cons_intro ?_ (fun t3 ht3 => ?_)
      · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
          MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
        refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def], ?_⟩
        intro ptrs hptrs idx hactive
        rw [evalOpR_ref, h2pdqi] at hptrs
        obtain rfl := Option.some.inj hptrs
        obtain ⟨masks, hm, hmi⟩ := hactive
        rw [evalOpR_ref, h2mask] at hm
        obtain rfl := Option.some.inj hm
        have ha : active s DK BK idx.1 := by simpa [bwdMaskTileG] using hmi
        simpa [bwdPtrTileG] using hbDQi idx.1 ha
      · obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv ht3
        have h3pdqt : (t2.setReg "_dq1" .real [BK] v3).regs .ptr [BK] "p_dq_inter"
            = some (bwdPtrTileG s DQInter s_qk_h DK BT BK rr) := by
          rw [BlockState.setReg_ne_name (h := by decide)]; exact h2pdqt
        have h3mask : (t2.setReg "_dq1" .real [BK] v3).regs .bool [BK] "mask"
            = some (bwdMaskTileG s DK BK) := by
          rw [BlockState.setReg_ne_name (h := by decide)]; exact h2mask
        
        refine Stmt.TraceSafeListR.cons_intro ?_ (fun t4 ht4 => ?_)
        · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
            MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
          refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def], ?_⟩
          intro ptrs hptrs idx hactive
          rw [evalOpR_ref, h3pdqt] at hptrs
          obtain rfl := Option.some.inj hptrs
          obtain ⟨masks, hm, hmi⟩ := hactive
          rw [evalOpR_ref, h3mask] at hm
          obtain rfl := Option.some.inj hm
          have ha : active s DK BK idx.1 := by simpa [bwdMaskTileG] using hmi
          simpa [bwdPtrTileG] using hbDQt idx.1 ha
        · obtain ⟨v4, -, rfl⟩ := stepStmtR_assign_inv ht4
          
          refine Stmt.TraceSafeListR.cons_intro
            (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t5 ht5 => ?_)
          obtain ⟨v5, -, rfl⟩ := stepStmtR_assign_inv ht5
          refine Stmt.TraceSafeListR.cons_intro
            (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun t6 ht6 => ?_)
          obtain ⟨v6, -, rfl⟩ := stepStmtR_assign_inv ht6
          have h6pdqt : ((((t2.setReg "_dq1" .real [BK] v3).setReg
              "_dq2" .real [BK] v4).setReg "_dq2" .real [BK] v5).setReg
              "_dq" .real [BK] v6).regs .ptr [BK] "p_dq_inter"
              = some (bwdPtrTileG s DQInter s_qk_h DK BT BK rr) := by
            rw [BlockState.setReg_ne_name (h := by decide),
              BlockState.setReg_ne_name (h := by decide),
              BlockState.setReg_ne_name (h := by decide),
              BlockState.setReg_ne_name (h := by decide)]
            exact h2pdqt
          have h6mask : ((((t2.setReg "_dq1" .real [BK] v3).setReg
              "_dq2" .real [BK] v4).setReg "_dq2" .real [BK] v5).setReg
              "_dq" .real [BK] v6).regs .bool [BK] "mask"
              = some (bwdMaskTileG s DK BK) := by
            rw [BlockState.setReg_ne_name (h := by decide),
              BlockState.setReg_ne_name (h := by decide),
              BlockState.setReg_ne_name (h := by decide),
              BlockState.setReg_ne_name (h := by decide)]
            exact h2mask
          
          refine Stmt.TraceSafeListR.cons_intro ?_ (fun t7 ht7 => ?_)
          · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MemAccess.SafeAtR,
              MaskOpt.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
              memAccessActiveAddressSafeR]
            refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def],
              by simp [Op.SafeAtR.eq_def], ?_⟩
            intro ptrs hptrs idx hactive
            rw [evalOpR_ref, h6pdqt] at hptrs
            obtain rfl := Option.some.inj hptrs
            obtain ⟨masks, hm, hmi⟩ := hactive
            rw [evalOpR_ref, h6mask] at hm
            obtain rfl := Option.some.inj hm
            have ha : active s DK BK idx.1 := by simpa [bwdMaskTileG] using hmi
            simpa [bwdPtrTileG] using hbDQt idx.1 ha
          · have hregs7 := stepStmtR_store_real_regs ht7
            have h7pdki : t7.regs .ptr [BK] "p_dk_inner"
                = some (bwdPtrTileG s DKInner s_qk_h DK BT BK rr) := by
              rw [hregs7, BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide)]
              exact h2pdki
            have h7pdkt : t7.regs .ptr [BK] "p_dk_inter"
                = some (bwdPtrTileG s DKInter s_qk_h DK BT BK rr) := by
              rw [hregs7, BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide)]
              exact h2pdkt
            have h7pq : t7.regs .ptr [BK] "p_q"
                = some (bwdPtrTileG s Q s_qk_h DK BT BK rr) := by
              rw [hregs7, BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide)]
              exact h2pq
            have h7pk : t7.regs .ptr [BK] "p_k"
                = some (bwdPtrTileG s K s_qk_h DK BT BK rr) := by
              rw [hregs7, BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide)]
              exact h2pk
            have h7pdg : t7.regs .ptr [BK] "p_dg"
                = some (bwdPtrTileG s DG s_qk_h DK BT BK rr) := by
              rw [hregs7, BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide)]
              exact h2pdg
            have h7mask : t7.regs .bool [BK] "mask"
                = some (bwdMaskTileG s DK BK) := by
              rw [hregs7, BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide),
                BlockState.setReg_ne_name (h := by decide)]
              exact h2mask
            
            refine Stmt.TraceSafeListR.cons_intro ?_ (fun t8 ht8 => ?_)
            · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
                MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
              refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def], ?_⟩
              intro ptrs hptrs idx hactive
              rw [evalOpR_ref, h7pdki] at hptrs
              obtain rfl := Option.some.inj hptrs
              obtain ⟨masks, hm, hmi⟩ := hactive
              rw [evalOpR_ref, h7mask] at hm
              obtain rfl := Option.some.inj hm
              have ha : active s DK BK idx.1 := by simpa [bwdMaskTileG] using hmi
              simpa [bwdPtrTileG] using hbDKi idx.1 ha
            · obtain ⟨w1, -, rfl⟩ := stepStmtR_assign_inv ht8
              have h8pdkt : (t7.setReg "_dk1" .real [BK] w1).regs .ptr [BK]
                  "p_dk_inter"
                  = some (bwdPtrTileG s DKInter s_qk_h DK BT BK rr) := by
                rw [BlockState.setReg_ne_name (h := by decide)]; exact h7pdkt
              have h8mask : (t7.setReg "_dk1" .real [BK] w1).regs .bool [BK] "mask"
                  = some (bwdMaskTileG s DK BK) := by
                rw [BlockState.setReg_ne_name (h := by decide)]; exact h7mask
              
              refine Stmt.TraceSafeListR.cons_intro ?_ (fun t9 ht9 => ?_)
              · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
                  MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
                refine ⟨by simp [Op.SafeAtR.eq_def],
                  by simp [Op.SafeAtR.eq_def], ?_⟩
                intro ptrs hptrs idx hactive
                rw [evalOpR_ref, h8pdkt] at hptrs
                obtain rfl := Option.some.inj hptrs
                obtain ⟨masks, hm, hmi⟩ := hactive
                rw [evalOpR_ref, h8mask] at hm
                obtain rfl := Option.some.inj hm
                have ha : active s DK BK idx.1 := by
                  simpa [bwdMaskTileG] using hmi
                simpa [bwdPtrTileG] using hbDKt idx.1 ha
              · obtain ⟨w2, -, rfl⟩ := stepStmtR_assign_inv ht9
                
                refine Stmt.TraceSafeListR.cons_intro
                  (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
                  (fun t10 ht10 => ?_)
                obtain ⟨w3, -, rfl⟩ := stepStmtR_assign_inv ht10
                refine Stmt.TraceSafeListR.cons_intro
                  (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
                  (fun t11 ht11 => ?_)
                obtain ⟨w4, -, rfl⟩ := stepStmtR_assign_inv ht11
                have h11pdkt : ((((t7.setReg "_dk1" .real [BK] w1).setReg
                    "_dk2" .real [BK] w2).setReg "_dk2" .real [BK] w3).setReg
                    "_dk" .real [BK] w4).regs .ptr [BK] "p_dk_inter"
                    = some (bwdPtrTileG s DKInter s_qk_h DK BT BK rr) := by
                  rw [BlockState.setReg_ne_name (h := by decide),
                    BlockState.setReg_ne_name (h := by decide),
                    BlockState.setReg_ne_name (h := by decide),
                    BlockState.setReg_ne_name (h := by decide)]
                  exact h7pdkt
                have h11mask : ((((t7.setReg "_dk1" .real [BK] w1).setReg
                    "_dk2" .real [BK] w2).setReg "_dk2" .real [BK] w3).setReg
                    "_dk" .real [BK] w4).regs .bool [BK] "mask"
                    = some (bwdMaskTileG s DK BK) := by
                  rw [BlockState.setReg_ne_name (h := by decide),
                    BlockState.setReg_ne_name (h := by decide),
                    BlockState.setReg_ne_name (h := by decide),
                    BlockState.setReg_ne_name (h := by decide)]
                  exact h7mask
                
                refine Stmt.TraceSafeListR.cons_intro ?_ (fun t12 ht12 => ?_)
                · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def,
                    MemAccess.SafeAtR, MaskOpt.SafeAtR, MaskOpt.ActiveR,
                    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
                  refine ⟨by simp [Op.SafeAtR.eq_def],
                    by simp [Op.SafeAtR.eq_def],
                    by simp [Op.SafeAtR.eq_def], ?_⟩
                  intro ptrs hptrs idx hactive
                  rw [evalOpR_ref, h11pdkt] at hptrs
                  obtain rfl := Option.some.inj hptrs
                  obtain ⟨masks, hm, hmi⟩ := hactive
                  rw [evalOpR_ref, h11mask] at hm
                  obtain rfl := Option.some.inj hm
                  have ha : active s DK BK idx.1 := by
                    simpa [bwdMaskTileG] using hmi
                  simpa [bwdPtrTileG] using hbDKt idx.1 ha
                · have hregs12 := stepStmtR_store_real_regs ht12
                  have h12pq : t12.regs .ptr [BK] "p_q"
                      = some (bwdPtrTileG s Q s_qk_h DK BT BK rr) := by
                    rw [hregs12, BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide)]
                    exact h7pq
                  have h12pk : t12.regs .ptr [BK] "p_k"
                      = some (bwdPtrTileG s K s_qk_h DK BT BK rr) := by
                    rw [hregs12, BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide)]
                    exact h7pk
                  have h12pdg : t12.regs .ptr [BK] "p_dg"
                      = some (bwdPtrTileG s DG s_qk_h DK BT BK rr) := by
                    rw [hregs12, BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide)]
                    exact h7pdg
                  have h12mask : t12.regs .bool [BK] "mask"
                      = some (bwdMaskTileG s DK BK) := by
                    rw [hregs12, BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide),
                      BlockState.setReg_ne_name (h := by decide)]
                    exact h7mask
                  
                  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t13 ht13 => ?_)
                  · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def,
                      MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
                      memAccessActiveAddressSafeR]
                    refine ⟨by simp [Op.SafeAtR.eq_def],
                      by simp [Op.SafeAtR.eq_def], ?_⟩
                    intro ptrs hptrs idx hactive
                    rw [evalOpR_ref, h12pq] at hptrs
                    obtain rfl := Option.some.inj hptrs
                    obtain ⟨masks, hm, hmi⟩ := hactive
                    rw [evalOpR_ref, h12mask] at hm
                    obtain rfl := Option.some.inj hm
                    have ha : active s DK BK idx.1 := by
                      simpa [bwdMaskTileG] using hmi
                    simpa [bwdPtrTileG] using hbQ idx.1 ha
                  · obtain ⟨x1, -, rfl⟩ := stepStmtR_assign_inv ht13
                    have h13pk : (t12.setReg "_q" .real [BK] x1).regs
                        .ptr [BK] "p_k"
                        = some (bwdPtrTileG s K s_qk_h DK BT BK rr) := by
                      rw [BlockState.setReg_ne_name (h := by decide)]
                      exact h12pk
                    have h13mask : (t12.setReg "_q" .real [BK] x1).regs
                        .bool [BK] "mask" = some (bwdMaskTileG s DK BK) := by
                      rw [BlockState.setReg_ne_name (h := by decide)]
                      exact h12mask
                    
                    refine Stmt.TraceSafeListR.cons_intro ?_
                      (fun t14 ht14 => ?_)
                    · simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def,
                        MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
                        memAccessActiveAddressSafeR]
                      refine ⟨by simp [Op.SafeAtR.eq_def],
                        by simp [Op.SafeAtR.eq_def], ?_⟩
                      intro ptrs hptrs idx hactive
                      rw [evalOpR_ref, h13pk] at hptrs
                      obtain rfl := Option.some.inj hptrs
                      obtain ⟨masks, hm, hmi⟩ := hactive
                      rw [evalOpR_ref, h13mask] at hm
                      obtain rfl := Option.some.inj hm
                      have ha : active s DK BK idx.1 := by
                        simpa [bwdMaskTileG] using hmi
                      simpa [bwdPtrTileG] using hbK idx.1 ha
                    · obtain ⟨x2, -, rfl⟩ := stepStmtR_assign_inv ht14
                      
                      refine Stmt.TraceSafeListR.cons_intro
                        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
                        (fun t15 ht15 => ?_)
                      obtain ⟨x3, -, rfl⟩ := stepStmtR_assign_inv ht15
                      refine Stmt.TraceSafeListR.cons_intro
                        (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
                        (fun t16 ht16 => ?_)
                      obtain ⟨x4, -, rfl⟩ := stepStmtR_assign_inv ht16
                      have h16pdg : ((((t12.setReg "_q" .real [BK] x1).setReg
                          "_k" .real [BK] x2).setReg
                          "_dg" .real [BK] x3).setReg
                          "cum_grad_dg" .real [BK] x4).regs .ptr [BK] "p_dg"
                          = some (bwdPtrTileG s DG s_qk_h DK BT BK rr) := by
                        rw [BlockState.setReg_ne_name (h := by decide),
                          BlockState.setReg_ne_name (h := by decide),
                          BlockState.setReg_ne_name (h := by decide),
                          BlockState.setReg_ne_name (h := by decide)]
                        exact h12pdg
                      have h16mask : ((((t12.setReg
                          "_q" .real [BK] x1).setReg
                          "_k" .real [BK] x2).setReg
                          "_dg" .real [BK] x3).setReg
                          "cum_grad_dg" .real [BK] x4).regs .bool [BK] "mask"
                          = some (bwdMaskTileG s DK BK) := by
                        rw [BlockState.setReg_ne_name (h := by decide),
                          BlockState.setReg_ne_name (h := by decide),
                          BlockState.setReg_ne_name (h := by decide),
                          BlockState.setReg_ne_name (h := by decide)]
                        exact h12mask
                      
                      refine Stmt.TraceSafeListR.cons_intro ?_
                        (fun _ _ => Stmt.TraceSafeListR.nil_intro)
                      simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def,
                        MemAccess.SafeAtR, MaskOpt.SafeAtR, MaskOpt.ActiveR,
                        MemAccess.ActiveAddressSafeR,
                        memAccessActiveAddressSafeR]
                      refine ⟨by simp [Op.SafeAtR.eq_def],
                        by simp [Op.SafeAtR.eq_def],
                        by simp [Op.SafeAtR.eq_def], ?_⟩
                      intro ptrs hptrs idx hactive
                      rw [evalOpR_ref, h16pdg] at hptrs
                      obtain rfl := Option.some.inj hptrs
                      obtain ⟨masks, hm, hmi⟩ := hactive
                      rw [evalOpR_ref, h16mask] at hm
                      obtain rfl := Option.some.inj hm
                      have ha : active s DK BK idx.1 := by
                        simpa [bwdMaskTileG] using hmi
                      simpa [bwdPtrTileG] using hbDG idx.1 ha

set_option maxHeartbeats 8000000 in

private theorem bwd_runR (R : RoundingModel)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat) (hBT : 0 < BT) (hBK : BK ≤ DK)
    (hDKInner_DQInter : DKInner ≠ DQInter) (hDKInter_DQInter : DKInter ≠ DQInter)
    (hQ_DQInter : Q ≠ DQInter) (hQ_DKInter : Q ≠ DKInter)
    (hK_DQInter : K ≠ DQInter) (hK_DKInter : K ≠ DKInter)
    (hDQInter_DKInter : DQInter ≠ DKInter)
    (hDQInter_DG : DQInter ≠ DG) (hDKInter_DG : DKInter ≠ DG)
    (hG_DQInter : G ≠ DQInter) (hG_DKInter : G ≠ DKInter) (hG_DG : G ≠ DG)
    (hQ_DG : Q ≠ DG) (hK_DG : K ≠ DG)
    (hDQInner_DQInter : DQInner ≠ DQInter) (hDQInner_DKInter : DQInner ≠ DKInter)
    (hDQInner_DG : DQInner ≠ DG)
    (hDKInner_DKInter : DKInner ≠ DKInter) (hDKInner_DG : DKInner ≠ DG)
    (s₀ : BlockState) :
    ∃ sfin,
      execR R (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter
        Q K G DG s_qk_h DK BT BK).toAlgKernel s₀ = some sfin
      ∧ (∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
          sfin.readMem DQInter (offset s₀ s_qk_h DK t.val BT BK i)
            = bwdDQInterClosed s₀ DQInner DQInter G s_qk_h DK BT BK t i)
      ∧ (∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
          sfin.readMem DKInter (offset s₀ s_qk_h DK t.val BT BK i)
            = bwdDKInterClosed s₀ DKInner DKInter G s_qk_h DK BT BK t i)
      ∧ (∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
          sfin.readMem DG (offset s₀ s_qk_h DK t.val BT BK i)
            = bwdDGClosed s₀ DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK t i)
      ∧ (∀ r oo,
          (r ≠ DQInter ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          (r ≠ DKInter ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          (r ≠ DG ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          sfin.mem r oo = s₀.mem r oo) := by
  obtain ⟨s0, hpro, hsh⟩ :=
    bwd_prologue_eval_general DQInner DQInter DKInner DKInter Q K G DG
      s_qk_h DK BT BK hBT s₀
  have hP0 : bwdInvG DQInner DQInter DKInner DKInter Q K G DG s₀ s_qk_h DK BT BK 0 s0 :=
    bwdInvG_entry DQInner DQInter DKInner DKInter Q K G DG s₀ s0 s_qk_h DK BT BK hsh
  
  obtain ⟨final, sfinal, haux, hge, hPfinal⟩ :=
    forRangeAux_inv (idx := "__rev_j") (stop := BT) (step := 1)
      (body := bwdIterBodyG DK BT BK)
      (P := fun m sc =>
        bwdInvG DQInner DQInter DKInner DKInter Q K G DG s₀ s_qk_h DK BT BK m sc
        ∧ ∀ r oo,
          (r ≠ DQInter ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          (r ≠ DKInter ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          (r ≠ DG ∨ ∀ (t : Fin BT) (i : Fin BK), active s₀ DK BK i →
            oo ≠ offset s₀ s_qk_h DK t.val BT BK i) →
          sc.mem r oo = s₀.mem r oo)
      one_ne_zero
      (fun m sc hm hP => by
        obtain ⟨s', hstep, hP'⟩ :=
          bwd_decay_cumsum_step_general DQInner DQInter DKInner DKInter Q K G DG s₀
            s_qk_h DK BT BK hBK hDKInner_DQInter hDKInter_DQInter hQ_DQInter
            hQ_DKInter hK_DQInter hK_DKInter hDQInter_DKInter hDQInter_DG
            hDKInter_DG hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG hDQInner_DQInter
            hDQInner_DKInter hDQInner_DG hDKInner_DKInter hDKInner_DG m hm sc hP.1
        refine ⟨s', hstep, hP', ?_⟩
        intro r oo hcDQ hcDK hcDG
        obtain ⟨hpid, hGrd, hQrd, hKrd, hDQird, hDKird, hmaskP, hcumP, hlastgP,
          hpgP, hpkP, hpqP, hpdqiP, hpdkiP, hpdqtP, hpdktP, hpdgP,
          hDQIunt, hDKIunt, hDGunt, hDQIwr, hDKIwr, hDGwr⟩ := hP.1
        rw [bwdPtrDecTileG_row s₀ DQInter s_qk_h DK BT BK m hm] at hpdqtP
        rw [bwdPtrDecTileG_row s₀ DKInter s_qk_h DK BT BK m hm] at hpdktP
        rw [bwdPtrDecTileG_row s₀ DG s_qk_h DK BT BK m hm] at hpdgP
        rw [bwd_body_step_frame DQInter DKInter DG s₀ sc s' s_qk_h DK BT BK
          (BT - 1 - m) m hpdqtP hpdktP hpdgP hmaskP hstep r oo ?_ ?_ ?_]
        · exact hP.2 r oo hcDQ hcDK hcDG
        · rcases hcDQ with hne' | hno
          · exact Or.inl hne'
          · exact Or.inr fun i ha => hno ⟨BT - 1 - m, by omega⟩ i ha
        · rcases hcDK with hne' | hno
          · exact Or.inl hne'
          · exact Or.inr fun i ha => hno ⟨BT - 1 - m, by omega⟩ i ha
        · rcases hcDG with hne' | hno
          · exact Or.inl hne'
          · exact Or.inr fun i ha => hno ⟨BT - 1 - m, by omega⟩ i ha)
      0 s0 ⟨hP0, fun r oo _ _ _ => by rw [hsh.mem]⟩
  obtain ⟨⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    hDQIwr, hDKIwr, hDGwr⟩, hframe⟩ := hPfinal
  
  have hLoopR : stepStmtR R (Stmt.forRangeDyn "__rev_j" (Op.constNat 0)
      (bwdStopOp BT) (Op.constNat 1) (bwdIterBodyG DK BT BK)) s0
      = some sfinal := by
    have h1 : stepStmtR R (Stmt.forRangeDyn "__rev_j" (Op.constNat 0)
        (bwdStopOp BT) (Op.constNat 1) (bwdIterBodyG DK BT BK)) s0
        = stepForRangeAuxR R "__rev_j" 0 BT 1 (bwdIterBodyG DK BT BK) s0 := by
      simp only [stepStmtR, bwd_evalOpR_constNat R 0 s0,
        bwd_stopOpR_eval R s0 BT hBT, bwd_evalOpR_constNat R 1 s0,
        Option.bind_some]
      rfl
    rw [h1, stepForRangeAuxR_castFree R _ (bwdIterBody_castFree R DK BT BK)
      "__rev_j"]
    exact haux
  have hproG : stepStmts (bwdPrologueG DQInner DQInter DKInner DKInter Q K G DG
      s_qk_h DK BT BK) s₀ = some s0 := by
    rw [← bwd_prologue_take_general]
    exact hpro
  refine ⟨sfinal, ?_, ?_, ?_, ?_, ?_⟩
  · unfold execR
    rw [bwd_body_decomp_io DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK,
      stepStmtsR_append R
        (bwdPrologueG DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK) _ s₀,
      bwdPrologue_castFree R DQInner DQInter DKInner DKInter Q K G DG
        s_qk_h DK BT BK s₀,
      hproG, Option.bind_some,
      stepStmtsR_cons_some (show stepStmtR R (Stmt.forRangeDyn "__rev_j"
        (Op.constNat 0) (bwdStopOp BT) (Op.constNat 1)
        (bwdCoreG DK BT BK ++ bwdTailG DK BK)) s0 = some sfinal from by
          rw [← bwdIterBodyG_decomp]; exact hLoopR),
      stepStmtsR_nil]
  · intro t i ha
    rw [hDQIwr t.val (by omega) t.isLt i, if_pos ha]
    exact bwdDqOutG_eq_closed s₀ DQInner DQInter G s_qk_h DK BT BK t i
  · intro t i ha
    rw [hDKIwr t.val (by omega) t.isLt i, if_pos ha]
    exact bwdDkOutG_eq_closed s₀ DKInner DKInter G s_qk_h DK BT BK t i
  · intro t i ha
    rw [hDGwr t.val (by omega) t.isLt i, if_pos ha]
    exact bwdCumPartialG_eq_DGClosed s₀ DQInner DQInter DKInner DKInter Q K G
      s_qk_h DK BT BK t i
  · exact hframe

set_option maxHeartbeats 8000000 in

theorem bwd_decay_global_cumsum_io_correctness (R : RoundingModel)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (s_qk_h DK BT BK : Nat)
    (hDKInner_DQInter : DKInner ≠ DQInter) (hDKInter_DQInter : DKInter ≠ DQInter)
    (hQ_DQInter : Q ≠ DQInter) (hQ_DKInter : Q ≠ DKInter)
    (hK_DQInter : K ≠ DQInter) (hK_DKInter : K ≠ DKInter)
    (hDQInter_DKInter : DQInter ≠ DKInter)
    (hDQInter_DG : DQInter ≠ DG) (hDKInter_DG : DKInter ≠ DG)
    (hG_DQInter : G ≠ DQInter) (hG_DKInter : G ≠ DKInter) (hG_DG : G ≠ DG)
    (hQ_DG : Q ≠ DG) (hK_DG : K ≠ DG)
    (hDQInner_DQInter : DQInner ≠ DQInter) (hDQInner_DKInter : DQInner ≠ DKInter)
    (hDQInner_DG : DQInner ≠ DG)
    (hDKInner_DKInter : DKInner ≠ DKInter) (hDKInner_DG : DKInner ≠ DG)
    (hBK : BK ≤ DK) (hBT : 0 < BT) :
    bwdDecayCumsumKernelIO DQInner DQInter DKInner DKInter Q K G DG
        s_qk_h DK BT BK ⊨[R]
      fun _ _ _ xs o t j => bwdDecayStreamSpec BT BK xs o t j := by
  intro A hd hregs hcov pid₀ pid₁ pid₂ xs s₀ hpid₀ hpid₁ hpid₂ hu hbr hbw hx
  subst hpid₀
  subst hpid₁
  subst hpid₂
  simp only [bwdDecayCumsumKernelIO] at hbr hbw hx ⊢
  
  obtain ⟨s1, hexec, hdqv, hdkv, hdgv, hframe⟩ :=
    bwd_runR R DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK hBT hBK
      hDKInner_DQInter hDKInter_DQInter hQ_DQInter hQ_DKInter hK_DQInter
      hK_DKInter hDQInter_DKInter hDQInter_DG hDKInter_DG hG_DQInter hG_DKInter
      hG_DG hQ_DG hK_DG hDQInner_DQInter hDQInner_DKInter hDQInner_DG
      hDKInner_DKInter hDKInner_DG s₀
  
  have hmsPre : Stmt.TraceSafeListR R A.extent
      (bwdPrologueG DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK)
      s₀ := by
    refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro stmt hst s'
    simp only [bwdPrologueG, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  
  have hloopOb : ∀ sP, stepStmtsR R
      (bwdPrologueG DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK) s₀
        = some sP →
      ((Op.constNat 0).SafeAtR R A.extent sP ∧
        (bwdStopOp BT).SafeAtR R A.extent sP ∧
        (Op.constNat 1).SafeAtR R A.extent sP) ∧
      ∀ bv ev sv, evalOpR R (Op.constNat 0) sP = some bv →
        evalOpR R (bwdStopOp BT) sP = some ev →
        evalOpR R (Op.constNat 1) sP = some sv →
        bwdInvG DQInner DQInter DKInner DKInter Q K G DG s₀ s_qk_h DK BT BK
          (bv.data PUnit.unit) sP ∧
        ∀ (c : Nat) (s' : BlockState), c < ev.data PUnit.unit →
          bwdInvG DQInner DQInter DKInner DKInter Q K G DG s₀ s_qk_h DK BT BK c s' →
          Stmt.TraceSafeListR R A.extent (bwdCoreG DK BT BK)
            (s'.setReg "__rev_j" .nat [] (Tile.scalar c)) ∧
          (∃ s'', stepStmtsR R (bwdCoreG DK BT BK ++ bwdTailG DK BK)
              (s'.setReg "__rev_j" .nat [] (Tile.scalar c)) = some s'' ∧
            bwdInvG DQInner DQInter DKInner DKInter Q K G DG s₀ s_qk_h DK BT BK
              (c + sv.data PUnit.unit) s'') ∧
          (c + sv.data PUnit.unit < ev.data PUnit.unit → ∀ sMid,
            stepStmtsR R (bwdCoreG DK BT BK)
              (s'.setReg "__rev_j" .nat [] (Tile.scalar c)) = some sMid →
            Stmt.TraceSafeListR R A.extent (bwdTailG DK BK) sMid) := by
    intro sP hsP
    refine ⟨⟨by simp [Op.SafeAtR.eq_def], by simp [bwdStopOp, Op.SafeAtR.eq_def],
      by simp [Op.SafeAtR.eq_def]⟩, ?_⟩
    intro bv ev sv hbv hev hsv
    rw [bwd_evalOpR_constNat] at hbv
    obtain rfl := Option.some.inj hbv
    rw [bwd_stopOpR_eval R sP BT hBT] at hev
    obtain rfl := Option.some.inj hev
    rw [bwd_evalOpR_constNat] at hsv
    obtain rfl := Option.some.inj hsv
    obtain ⟨s0, hpro, hsh⟩ :=
      bwd_prologue_eval_general DQInner DQInter DKInner DKInter Q K G DG
        s_qk_h DK BT BK hBT s₀
    rw [bwdPrologue_castFree R DQInner DQInter DKInner DKInter Q K G DG
        s_qk_h DK BT BK s₀,
      ← bwd_prologue_take_general, hpro] at hsP
    obtain rfl := Option.some.inj hsP
    refine ⟨bwdInvG_entry DQInner DQInter DKInner DKInter Q K G DG s₀ s0
      s_qk_h DK BT BK hsh, ?_⟩
    intro c s' hc hP
    have hcBT : c < BT := hc
    have hPd := hP
    obtain ⟨hpid, hGrd, hQrd, hKrd, hDQird, hDKird, hmaskP, hcumP, hlastgP,
      hpgP, hpkP, hpqP, hpdqiP, hpdkiP, hpdqtP, hpdktP, hpdgP,
      hDQIunt, hDKIunt, hDGunt, hDQIwr, hDKIwr, hDGwr⟩ := hPd
    rw [bwdPtrDecTileG_row s₀ G s_qk_h DK BT BK c hcBT] at hpgP
    rw [bwdPtrDecTileG_row s₀ K s_qk_h DK BT BK c hcBT] at hpkP
    rw [bwdPtrDecTileG_row s₀ Q s_qk_h DK BT BK c hcBT] at hpqP
    rw [bwdPtrDecTileG_row s₀ DQInner s_qk_h DK BT BK c hcBT] at hpdqiP
    rw [bwdPtrDecTileG_row s₀ DKInner s_qk_h DK BT BK c hcBT] at hpdkiP
    rw [bwdPtrDecTileG_row s₀ DQInter s_qk_h DK BT BK c hcBT] at hpdqtP
    rw [bwdPtrDecTileG_row s₀ DKInter s_qk_h DK BT BK c hcBT] at hpdktP
    rw [bwdPtrDecTileG_row s₀ DG s_qk_h DK BT BK c hcBT] at hpdgP
    have hrow : BT - 1 - c < BT := by omega
    
    have hbG' : ∀ i : Fin BK, active s₀ DK BK i →
        offset s₀ s_qk_h DK (BT - 1 - c) BT BK i < A.extent G := by
      intro i ha
      have h := hbr ⟨0, by omega⟩ ⟨BT - 1 - c, hrow⟩ i ha
      rw [fwd_addr_eq] at h
      exact h
    have hbDQi' : ∀ i : Fin BK, active s₀ DK BK i →
        offset s₀ s_qk_h DK (BT - 1 - c) BT BK i < A.extent DQInner := by
      intro i ha
      have h := hbr ⟨1, by omega⟩ ⟨BT - 1 - c, hrow⟩ i ha
      rw [fwd_addr_eq] at h
      exact h
    have hbDQt' : ∀ i : Fin BK, active s₀ DK BK i →
        offset s₀ s_qk_h DK (BT - 1 - c) BT BK i < A.extent DQInter := by
      intro i ha
      have h := hbr ⟨2, by omega⟩ ⟨BT - 1 - c, hrow⟩ i ha
      rw [fwd_addr_eq] at h
      exact h
    have hbDKi' : ∀ i : Fin BK, active s₀ DK BK i →
        offset s₀ s_qk_h DK (BT - 1 - c) BT BK i < A.extent DKInner := by
      intro i ha
      have h := hbr ⟨3, by omega⟩ ⟨BT - 1 - c, hrow⟩ i ha
      rw [fwd_addr_eq] at h
      exact h
    have hbDKt' : ∀ i : Fin BK, active s₀ DK BK i →
        offset s₀ s_qk_h DK (BT - 1 - c) BT BK i < A.extent DKInter := by
      intro i ha
      have h := hbr ⟨4, by omega⟩ ⟨BT - 1 - c, hrow⟩ i ha
      rw [fwd_addr_eq] at h
      exact h
    have hbQ' : ∀ i : Fin BK, active s₀ DK BK i →
        offset s₀ s_qk_h DK (BT - 1 - c) BT BK i < A.extent Q := by
      intro i ha
      have h := hbr ⟨5, by omega⟩ ⟨BT - 1 - c, hrow⟩ i ha
      rw [fwd_addr_eq] at h
      exact h
    have hbK' : ∀ i : Fin BK, active s₀ DK BK i →
        offset s₀ s_qk_h DK (BT - 1 - c) BT BK i < A.extent K := by
      intro i ha
      have h := hbr ⟨6, by omega⟩ ⟨BT - 1 - c, hrow⟩ i ha
      rw [fwd_addr_eq] at h
      exact h
    have hbDG' : ∀ i : Fin BK, active s₀ DK BK i →
        offset s₀ s_qk_h DK (BT - 1 - c) BT BK i < A.extent DG := by
      intro i ha
      have h := hbw ⟨2, by omega⟩ ⟨BT - 1 - c, hrow⟩ i ha
      rw [fwd_addr_eq] at h
      exact h
    refine ⟨?_, ?_, ?_⟩
    · exact bwd_bodySafeR R A.extent DQInner DQInter DKInner DKInter Q K G DG
        s₀ s' s_qk_h DK BT BK (BT - 1 - c) c
        hpgP hpkP hpqP hpdqiP hpdkiP hpdqtP hpdktP hpdgP hmaskP
        hbG' hbQ' hbK' hbDQi' hbDKi' hbDQt' hbDKt' hbDG'
    · obtain ⟨s'', hstep, hP'⟩ :=
        bwd_decay_cumsum_step_general DQInner DQInter DKInner DKInter Q K G DG s₀
          s_qk_h DK BT BK hBK hDKInner_DQInter hDKInter_DQInter hQ_DQInter
          hQ_DKInter hK_DQInter hK_DKInter hDQInter_DKInter hDQInter_DG
          hDKInter_DG hG_DQInter hG_DKInter hG_DG hQ_DG hK_DG hDQInner_DQInter
          hDQInner_DKInter hDQInner_DG hDKInner_DKInter hDKInner_DG c hcBT s' hP
      refine ⟨s'', ?_, hP'⟩
      rw [← bwdIterBodyG_decomp, bwdIterBody_castFree]
      exact hstep
    · intro hfin sMid hsMid
      have hfin' : c + 1 < BT := hfin
      have hrr : 1 ≤ BT - 1 - c := by omega
      have hMg : sMid.regs .ptr [BK] "p_g"
          = some (bwdPtrTileG s₀ G s_qk_h DK BT BK (BT - 1 - c)) := by
        rw [bwd_core_regsR R DK BT BK _ sMid hsMid .ptr [BK] "p_g"
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpgP
      have hMk : sMid.regs .ptr [BK] "p_k"
          = some (bwdPtrTileG s₀ K s_qk_h DK BT BK (BT - 1 - c)) := by
        rw [bwd_core_regsR R DK BT BK _ sMid hsMid .ptr [BK] "p_k"
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpkP
      have hMq : sMid.regs .ptr [BK] "p_q"
          = some (bwdPtrTileG s₀ Q s_qk_h DK BT BK (BT - 1 - c)) := by
        rw [bwd_core_regsR R DK BT BK _ sMid hsMid .ptr [BK] "p_q"
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpqP
      have hMdqi : sMid.regs .ptr [BK] "p_dq_inner"
          = some (bwdPtrTileG s₀ DQInner s_qk_h DK BT BK (BT - 1 - c)) := by
        rw [bwd_core_regsR R DK BT BK _ sMid hsMid .ptr [BK] "p_dq_inner"
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpdqiP
      have hMdki : sMid.regs .ptr [BK] "p_dk_inner"
          = some (bwdPtrTileG s₀ DKInner s_qk_h DK BT BK (BT - 1 - c)) := by
        rw [bwd_core_regsR R DK BT BK _ sMid hsMid .ptr [BK] "p_dk_inner"
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpdkiP
      have hMdqt : sMid.regs .ptr [BK] "p_dq_inter"
          = some (bwdPtrTileG s₀ DQInter s_qk_h DK BT BK (BT - 1 - c)) := by
        rw [bwd_core_regsR R DK BT BK _ sMid hsMid .ptr [BK] "p_dq_inter"
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpdqtP
      have hMdkt : sMid.regs .ptr [BK] "p_dk_inter"
          = some (bwdPtrTileG s₀ DKInter s_qk_h DK BT BK (BT - 1 - c)) := by
        rw [bwd_core_regsR R DK BT BK _ sMid hsMid .ptr [BK] "p_dk_inter"
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpdktP
      have hMdg : sMid.regs .ptr [BK] "p_dg"
          = some (bwdPtrTileG s₀ DG s_qk_h DK BT BK (BT - 1 - c)) := by
        rw [bwd_core_regsR R DK BT BK _ sMid hsMid .ptr [BK] "p_dg"
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide),
          BlockState.setReg_ne_name (h := by decide)]
        exact hpdgP
      exact bwd_tailSafeR R A.extent DQInner DQInter DKInner DKInter Q K G DG
        s₀ sMid s_qk_h DK BT BK (BT - 1 - c) hrr
        hMg hMk hMq hMdqi hMdki hMdqt hMdkt hMdg
  
  obtain ⟨s', hexecF, hobs⟩ :=
    A.execR_flatten_deadPtrTail hd hcov R
      (bwd_decay_global_cumsum_surface DQInner DQInter DKInner DKInter Q K G DG
        s_qk_h DK BT BK).toAlgKernel
      (bwdPrologueG DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK)
      "__rev_j" (Op.constNat 0) (bwdStopOp BT) (Op.constNat 1)
      (bwdCoreG DK BT BK) (bwdTailG DK BK)
      (bwd_body_decomp_io DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK)
      (bwdTailG_ptrDecTail DK BK)
      (bwd_prologue_flattenOkR DQInner DQInter DKInner DKInter Q K G DG s_qk_h DK BT BK)
      (bwd_core_flattenOkR DK BT BK)
      (by simp [Op.FlattenOkR.eq_def]) (bwd_stopOp_flattenOkR BT)
      (by simp [Op.FlattenOkR.eq_def])
      s₀ hu hmsPre _ hloopOb s1 hexec
  have hmem0 : DQInter ∈ A.regions := by
    rw [hregs]
    show DQInter ∈ [DQInner, DQInter, DKInner, DKInter, Q, K, G, DG]
    simp
  have hmem1 : DKInter ∈ A.regions := by
    rw [hregs]
    show DKInter ∈ [DQInner, DQInter, DKInner, DKInter, Q, K, G, DG]
    simp
  have hmem2 : DG ∈ A.regions := by
    rw [hregs]
    show DG ∈ [DQInner, DQInter, DKInner, DKInter, Q, K, G, DG]
    simp
  refine ⟨s', hexecF, ?_, ?_⟩
  · 
    
    intro o t j hj
    have hlt := hbw o t j hj
    match o with
    | ⟨0, h0⟩ =>
        rw [hobs.readMemAs_eq, A.flattenState_readMemAs hd s1 hmem0 hlt .real,
          BlockState.readMemAs_real, fwd_addr_eq]
        have hval : s1.readMem DQInter (offset s₀ s_qk_h DK t.val BT BK j)
            = bwdDecayStreamSpec BT BK xs ⟨0, h0⟩ t j := by
          rw [hdqv t j hj]
          show bwdDQInterClosed s₀ DQInner DQInter G s_qk_h DK BT BK t j
            = bwdDqStream BT BK xs t j
          exact (bwdDqStream_eq_closed DQInner DQInter G s₀ s_qk_h DK BT BK xs
            (fun t' j' hj' => hx ⟨0, by omega⟩ t' j' hj')
            (fun t' j' hj' => hx ⟨1, by omega⟩ t' j' hj')
            (fun t' j' hj' => hx ⟨2, by omega⟩ t' j' hj') t j hj).symm
        simp only [RoundingModel.round_real_apply]
        exact congrArg some hval
    | ⟨1, h1⟩ =>
        rw [hobs.readMemAs_eq, A.flattenState_readMemAs hd s1 hmem1 hlt .real,
          BlockState.readMemAs_real, fwd_addr_eq]
        have hval : s1.readMem DKInter (offset s₀ s_qk_h DK t.val BT BK j)
            = bwdDecayStreamSpec BT BK xs ⟨1, h1⟩ t j := by
          rw [hdkv t j hj]
          show bwdDKInterClosed s₀ DKInner DKInter G s_qk_h DK BT BK t j
            = bwdDkStream BT BK xs t j
          exact (bwdDkStream_eq_closed DKInner DKInter G s₀ s_qk_h DK BT BK xs hBT
            (fun t' j' hj' => hx ⟨0, by omega⟩ t' j' hj')
            (fun t' j' hj' => hx ⟨3, by omega⟩ t' j' hj')
            (fun t' j' hj' => hx ⟨4, by omega⟩ t' j' hj') t j hj).symm
        simp only [RoundingModel.round_real_apply]
        exact congrArg some hval
    | ⟨2, h2⟩ =>
        rw [hobs.readMemAs_eq, A.flattenState_readMemAs hd s1 hmem2 hlt .real,
          BlockState.readMemAs_real, fwd_addr_eq]
        have hval : s1.readMem DG (offset s₀ s_qk_h DK t.val BT BK j)
            = bwdDecayStreamSpec BT BK xs ⟨2, h2⟩ t j := by
          rw [hdgv t j hj]
          show bwdDGClosed s₀ DQInner DQInter DKInner DKInter Q K G s_qk_h DK BT BK t j
            = bwdDgStream BT BK xs t j
          exact (bwdDgStream_eq_closed DQInner DQInter DKInner DKInter Q K G s₀
            s_qk_h DK BT BK xs hBT
            (fun t' j' hj' => hx ⟨0, by omega⟩ t' j' hj')
            (fun t' j' hj' => hx ⟨1, by omega⟩ t' j' hj')
            (fun t' j' hj' => hx ⟨2, by omega⟩ t' j' hj')
            (fun t' j' hj' => hx ⟨3, by omega⟩ t' j' hj')
            (fun t' j' hj' => hx ⟨4, by omega⟩ t' j' hj')
            (fun t' j' hj' => hx ⟨5, by omega⟩ t' j' hj')
            (fun t' j' hj' => hx ⟨6, by omega⟩ t' j' hj') t j hj).symm
        simp only [RoundingModel.round_real_apply]
        exact congrArg some hval
    | ⟨n + 3, h3⟩ => exact absurd h3 (show ¬(n + 3 < 3) by omega)
  · 
    intro r' o' hcond
    rw [hobs.mem]
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe r o ?_ ?_ ?_)
          · by_cases hrX : r = DQInter
            · refine Or.inr fun t i ha hoo => ?_
              rcases hcond with hflat | hn
              · exact absurd rfl hflat
              · exact hn ⟨0, by omega⟩ t i ha
                  (by rw [hoeq, hrX, hoo, fwd_addr_eq])
            · exact Or.inl hrX
          · by_cases hrX : r = DKInter
            · refine Or.inr fun t i ha hoo => ?_
              rcases hcond with hflat | hn
              · exact absurd rfl hflat
              · exact hn ⟨1, by omega⟩ t i ha
                  (by rw [hoeq, hrX, hoo, fwd_addr_eq])
            · exact Or.inl hrX
          · by_cases hrX : r = DG
            · refine Or.inr fun t i ha hoo => ?_
              rcases hcond with hflat | hn
              · exact absurd rfl hflat
              · exact hn ⟨2, by omega⟩ t i ha
                  (by rw [hoeq, hrX, hoo, fwd_addr_eq])
            · exact Or.inl hrX
    · simp only [FlatAlloc.flattenState, if_neg hr]

end BwdIOFace

end VeriTile.Bench.TritonBenchG.DecayCumsum.RealReference

#guard_msgs (drop info) in
#auditModuleAxioms
