import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexBackwardLoop
import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexBackwardFinal
import bench.tritonbench_g.diag_ssm_triton.Proofs.ZeroSafety
import VeriTile.Triton.Memory.TraceRewrite
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardSafety
open VeriTile.Triton ComplexForwardStep ComplexBackwardLoop
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Bounds describe exactly the source's active component accesses. A zero
length does not require initial-state or sequence storage; the transition
loads and terminal stores still occur. No block-size/width restriction is
needed, since inactive lanes perform no memory accesses. -/
structure Bounds (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds) : Prop where
  initial : 0<length → ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.s
  transition : ∀ j, mask p u j=Bool.true → (columns p u j%p.dim)*2+1 < bounds p.l
  previous : ∀ t, t+1<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.y
  gradOutput : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.gy
  gradInput : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.gx
  gradInitial : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.gs
  gradTransition : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.gl

theorem loads_safe (bounds : RegionBounds) (gy : RegionName) (length batch dim B k : Nat)
    (u : BlockState) (col : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (hi : u.regs .nat [] "i"=some (Tile.scalar k))
    (hc : u.regs .nat [B] "col_offsets"=some ⟨col⟩)
    (hm : u.regs .bool [B] "mask"=some ⟨mask⟩)
    (hb : ∀ j, mask j=Bool.true → ComplexForwardStep.offset batch dim (length-1-k) col j*2+1 < bounds gy) :
    Stmt.TraceSafeList bounds (ComplexBackwardLoads.source gy length batch dim B) u := by
  simp only [ComplexBackwardLoads.source,ComplexBackwardSource.body,Erasure.complexBackward,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    Except.bind,bind,Bool.false_eq_true,↓reduceIte,List.drop,List.take]
  simp only [Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,
    MemAccess.SafeAt,MaskOpt.SafeAt,MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
    and_true,true_and,and_self,stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
    Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
    String.reduceEq,Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,NumericDType.sub,
    WithBot.realSub,Option.map₂_some_some,Region.cast,
    BlockState.setReg_same,BlockState.setReg_ne_name,BlockState.setReg_pids,
    BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
    Option.bind_some,Option.map_some,Bool.true_eq_false,↓reduceIte,ne_eq,not_false_eq_true,
    hi,hc,hm,Nat.mul_assoc]
  simp only [Option.some.injEq]
  constructor
  · intro offsets hoff j ⟨masks,hmask,hact⟩
    subst offsets; subst masks
    exact Nat.lt_of_le_of_lt (Nat.le_add_right _ 1) (hb j hact)
  · intro offsets hoff j ⟨masks,hmask,hact⟩
    subst offsets; subst masks
    exact hb j hact

theorem update_safe (bounds : RegionBounds) {B : Nat} (u : BlockState) (gx : RegionName)
    (col off : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (g prev carry acc transition : TileIndex [B] → ℂ)
    (h : ComplexBackwardStep.Registers u col mask carry acc transition)
    (ho : u.regs .nat [B] "offsets"=some ⟨fun j => off j*2⟩)
    (hgr : u.regs .real [B] "grad_y_real"=some (realTile g))
    (hgi : u.regs .real [B] "grad_y_imag"=some (imagTile g))
    (hpr : u.regs .real [B] "s_real"=some (realTile prev))
    (hpi : u.regs .real [B] "s_imag"=some (imagTile prev))
    (hb : ∀ j, mask j=Bool.true → off j*2+1 < bounds gx) :
    Stmt.TraceSafeList bounds (ComplexBackwardUpdate.source gx B) u := by
  rcases h with ⟨hc,hm,hcr,hci,har,hai,hlr,hli⟩
  simp only [ComplexBackwardUpdate.source,ComplexBackwardSource.body,Erasure.complexBackward,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    Except.bind,bind,Bool.false_eq_true,↓reduceIte,List.drop]
  simp only [Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,
    MemAccess.SafeAt,MaskOpt.SafeAt,MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
    and_true,true_and,and_self,stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
    Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
    String.reduceEq,Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,NumericDType.sub,
    WithBot.realMul,WithBot.realAdd,WithBot.realSub,Option.map₂_some_some,
    Region.cast,BlockState.writeMemTyped_real,BlockState.foldl_writeMem_masked_regs,
    BlockState.setReg_same,BlockState.setReg_ne_name,BlockState.setReg_pids,
    BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
    Option.bind_some,Option.map_some,Bool.true_eq_false,↓reduceIte,ne_eq,not_false_eq_true,
    hc,hm,hcr,hci,har,hai,hlr,hli,ho,hgr,hgi,hpr,hpi,realTile,imagTile,
    FloatDType.storeValue,FloatDType.toWithBot,WithBot.unbotD_some]
  simp only [Option.some.injEq]
  constructor
  · intro offsets hoff j ⟨masks,hmask,hact⟩
    subst offsets; subst masks
    exact Nat.lt_of_le_of_lt (Nat.le_add_right _ 1) (hb j hact)
  · intro offsets hoff j ⟨masks,hmask,hact⟩
    subst offsets; subst masks
    exact hb j hact

theorem previous_safe (bounds : RegionBounds) (s y : RegionName) (batch dim B t : Nat) (u : BlockState)
    (col : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (ht : u.regs .nat [] "t"=some (Tile.scalar t))
    (hc : u.regs .nat [B] "col_offsets"=some ⟨col⟩)
    (ho : u.regs .nat [B] "offsets"=some ⟨fun j => offset batch dim t col j*2⟩)
    (hm : u.regs .bool [B] "mask"=some ⟨mask⟩)
    (hs : t=0 → ∀ j, mask j=Bool.true → col j*2+1 < bounds s)
    (hy : 0<t → ∀ j, mask j=Bool.true → ComplexForwardStep.offset batch dim (t-1) col j*2+1 < bounds y) :
    Stmt.TraceSafeList bounds (ComplexBackwardPrevious.source s y batch dim B) u := by
  simp only [ComplexBackwardPrevious.source,ComplexBackwardSource.body,Erasure.complexBackward,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    Except.bind,bind,Bool.false_eq_true,↓reduceIte,List.drop,List.take]
  cases t with
  | zero =>
    simp only [Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,
    MemAccess.SafeAt,MaskOpt.SafeAt,MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
    and_true,true_and,and_self,stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
      Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
      String.reduceEq,Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,
      NumericDType.sub,ComparableDType.gt,
      Region.cast,BlockState.setReg_same,BlockState.setReg_ne_name,
      BlockState.setReg_readMemValue,BlockState.readMemValue_real,
      ht,hc,hm,ho,Bool.false_eq_true,decide_eq_true_eq,ite_false,ite_true,↓reduceIte]
    norm_num only
    simp only [ite_false,ite_true,Option.bind,bind,pure,Option.pure_def,
      BlockState.setReg_ne_name,BlockState.setReg_readMem,hc,hm,ho,
      String.reduceEq,ne_eq,not_false_eq_true]
    simp only [Nat.mul_comm 2,Option.some.injEq]
    simp only [and_true,true_and]
    constructor
    · intro offsets hoff j ⟨masks,hmask,hact⟩
      subst offsets; subst masks
      exact Nat.lt_of_le_of_lt (Nat.le_add_right _ 1) (hs rfl j hact)
    · intro offsets hoff j ⟨masks,hmask,hact⟩
      subst offsets; subst masks
      exact hs rfl j hact
  | succ t =>
    have hp : 0<t+1 := by omega
    simp only [Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,
    MemAccess.SafeAt,MaskOpt.SafeAt,MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
    and_true,true_and,and_self,stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
      Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
      String.reduceEq,Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,
      NumericDType.sub,ComparableDType.gt,
      Region.cast,BlockState.setReg_same,BlockState.setReg_ne_name,
      BlockState.setReg_readMemValue,BlockState.readMemValue_real,
      ht,hc,hm,ho,Bool.false_eq_true,decide_eq_true_eq,ite_false,ite_true,↓reduceIte,hp]
    norm_num only
    simp only [ite_false,ite_true,Option.bind,bind,pure,Option.pure_def,
      BlockState.setReg_ne_name,BlockState.setReg_readMem,hc,hm,ho,
      String.reduceEq,ne_eq,not_false_eq_true]
    simp only [ComplexForwardStep.offset,ComplexBackwardStep.previous_address_zero,
      ComplexBackwardPrevious.previous_address_imag,hp,Nat.succ_pos,Option.some.injEq]
    simp only [and_true,true_and]
    constructor
    · intro offsets hoff j ⟨masks,hmask,hact⟩
      subst offsets; subst masks
      exact Nat.lt_of_le_of_lt (Nat.le_add_right _ 1) (hy hp j hact)
    · intro offsets hoff j ⟨masks,hmask,hact⟩
      subst offsets; subst masks
      exact hy hp j hact

theorem body_safe (bounds : RegionBounds) (s y gx gy : RegionName) (length batch dim B k : Nat)
    (u : BlockState) (col : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (carry acc transition : TileIndex [B] → ℂ)
    (hi : u.regs .nat [] "i"=some (Tile.scalar k))
    (h : ComplexBackwardStep.Registers u col mask carry acc transition)
    (hs : length-1-k=0 → ∀ j, mask j=Bool.true → col j*2+1 < bounds s)
    (hy : 0<length-1-k → ∀ j, mask j=Bool.true →
      ComplexForwardStep.offset batch dim (length-1-k-1) col j*2+1 < bounds y)
    (hgy : ∀ j, mask j=Bool.true → ComplexForwardStep.offset batch dim (length-1-k) col j*2+1 < bounds gy)
    (hgx : ∀ j, mask j=Bool.true → ComplexForwardStep.offset batch dim (length-1-k) col j*2+1 < bounds gx) :
    Stmt.TraceSafeList bounds (ComplexBackwardSource.body s y gx gy length batch dim B) u := by
  let t := length-1-k
  let ul := ComplexBackwardLoads.result u gy batch dim t col mask
  let up := ComplexBackwardPrevious.result ul s y batch dim t col mask
  let g := fun j => starRingEnd ℂ (input u gy (ComplexForwardStep.offset batch dim t col) mask j)
  let prev := ComplexBackwardStep.previous u s y batch dim t col mask
  have hl := ComplexBackwardLoads.step gy length batch dim B k u col mask hi h.columns h.active
  have hp := ComplexBackwardPrevious.step s y batch dim B t ul col mask
    (by simp [ul,ComplexBackwardLoads.result])
    (by simpa [ul,ComplexBackwardLoads.result] using h.columns)
    (by simp [ul,ComplexBackwardLoads.result])
    (by simpa [ul,ComplexBackwardLoads.result] using h.active)
  have hr : ComplexBackwardStep.Registers up col mask carry acc transition := by
    constructor
    · simpa [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result] using h.columns
    · simpa [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result] using h.active
    · simpa [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result] using h.carryReal
    · simpa [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result] using h.carryImag
    · simpa [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result] using h.accReal
    · simpa [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result] using h.accImag
    · simpa [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result] using h.transitionReal
    · simpa [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result] using h.transitionImag
  have hprev : ComplexBackwardStep.previous ul s y batch dim t col mask=prev := rfl
  rw [ComplexBackwardStep.source_parts,List.append_assoc]
  apply TraceProofs.append_safe _ u (loads_safe bounds gy length batch dim B k u col mask hi h.columns h.active hgy)
  intro vl he
  rw [hl] at he
  obtain rfl := Option.some.inj he
  change Stmt.TraceSafeList bounds (ComplexBackwardPrevious.source s y batch dim B ++
    ComplexBackwardUpdate.source gx B) ul
  apply TraceProofs.append_safe _ ul (previous_safe bounds s y batch dim B t ul col mask
    (by simp [ul,ComplexBackwardLoads.result])
    (by simpa [ul,ComplexBackwardLoads.result] using h.columns)
    (by simp [ul,ComplexBackwardLoads.result])
    (by simpa [ul,ComplexBackwardLoads.result] using h.active) hs hy)
  intro vp he
  rw [hp] at he
  obtain rfl := Option.some.inj he
  exact update_safe bounds up gx col (ComplexForwardStep.offset batch dim t col) mask g prev carry acc transition hr
    (by simp [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result])
    (by simp [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result,g])
    (by simp [up,ul,ComplexBackwardPrevious.result,ComplexBackwardLoads.result,g])
    (by simp [up,ComplexBackwardPrevious.result,hprev])
    (by simp [up,ComplexBackwardPrevious.result,hprev]) hgx

theorem prelude_safe (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds)
    (hb : Bounds p u length bounds) :
    Stmt.TraceSafeList bounds (ComplexBackwardSource.prelude p.l p.batch p.dim p.B) u := by
  have hstatic : ZeroSafety.complexBackwardBounds bounds u p.l p.gs p.gl p.batch p.dim p.B := by
    refine ⟨?_,?_,?_⟩
    · intro j hj
      exact hb.transition ⟨j,PUnit.unit⟩ (by
        simpa only [mask,ComplexForwardInit.active,ComplexForwardInit.columns,decide_eq_true_eq] using hj)
    · intro j hj
      exact hb.gradInitial ⟨j,PUnit.unit⟩ (by
        simpa only [mask,ComplexForwardInit.active,ComplexForwardInit.columns,decide_eq_true_eq] using hj)
    · intro j hj
      exact hb.gradTransition ⟨j,PUnit.unit⟩ (by
        simpa only [mask,ComplexForwardInit.active,ComplexForwardInit.columns,decide_eq_true_eq] using hj)
  have h := ZeroSafety.complex_backward_safe bounds u p.s p.l p.y p.gs p.gx p.gl p.gy p.batch p.dim p.B hstatic
  unfold Kernel.TraceSafe at h
  rw [ComplexBackwardSource.source_body,List.append_assoc] at h
  exact ((TraceRewrite.append_iff bounds _ _ u).mp h).1

theorem post_safe (bounds : RegionBounds) {B : Nat} (u : BlockState) (gs gl : RegionName) (col : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) (carry acc transition : TileIndex [B] → ℂ)
    (h : ComplexBackwardStep.Registers u col mask carry acc transition)
    (hs : ∀ j, mask j=Bool.true → col j*2+1 < bounds gs)
    (hl : ∀ j, mask j=Bool.true → col j*2+1 < bounds gl) :
    Stmt.TraceSafeList bounds (ComplexBackwardSource.postlude gs gl B) u := by
  rcases h with ⟨hc,hm,hcr,hci,har,hai,hlr,hli⟩
  simp only [ComplexBackwardSource.postlude,Erasure.complexBackward,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    Except.bind,bind,Bool.false_eq_true,↓reduceIte,List.drop]
  simp only [Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,
    MemAccess.SafeAt,MaskOpt.SafeAt,MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
    and_true,true_and,and_self,stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
    Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
    String.reduceEq,Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,NumericDType.sub,
    WithBot.realSub,Option.map₂_some_some,Region.cast,BlockState.writeMemTyped_real,
    BlockState.foldl_writeMem_masked_regs,BlockState.setReg_same,BlockState.setReg_ne_name,
    hc,hm,hcr,hci,har,hai,realTile,imagTile,
    FloatDType.storeValue,FloatDType.toWithBot,WithBot.unbotD_some]
  simp only [Option.some.injEq]
  refine ⟨?_,?_,?_,?_⟩
  · intro offsets hoff j ⟨masks,hmask,hact⟩
    subst offsets; subst masks
    exact Nat.lt_of_le_of_lt (Nat.le_add_right _ 1) (hs j hact)
  · intro offsets hoff j ⟨masks,hmask,hact⟩
    subst offsets; subst masks
    exact hs j hact
  · intro offsets hoff j ⟨masks,hmask,hact⟩
    subst offsets; subst masks
    exact Nat.lt_of_le_of_lt (Nat.le_add_right _ 1) (hl j hact)
  · intro offsets hoff j ⟨masks,hmask,hact⟩
    subst offsets; subst masks
    exact hl j hact

/-- Trace safety follows the actual previous-state branch on each reverse
iteration. Terminal safety uses the loop's proved final register context. -/
theorem trace_safe (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds)
    (hb : Bounds p u length bounds) (sep : ReadSeparation p) :
    (Erasure.complexBackward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B).toAlgKernel.TraceSafe bounds u := by
  unfold Kernel.TraceSafe
  rw [ComplexBackwardSource.source_body,List.append_assoc]
  apply TraceProofs.append_safe _ u (prelude_safe p u length bounds hb)
  intro v he
  rw [ComplexBackwardInit.prefix_step] at he
  obtain rfl := Option.some.inj he
  change Stmt.TraceSafeList bounds (.forLoop "i" length
    (ComplexBackwardSource.body p.s p.y p.gx p.gy length p.batch p.dim p.B) ::
      ComplexBackwardSource.postlude p.gs p.gl p.B) _
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe]
    let P := fun v => ∃ carry acc : Lane p → ℂ, ComplexBackwardStep.Registers v
      (columns p u) (mask p u) carry acc (transition p u)
    apply TraceProofs.loop_safe (P := P) ?_ ?_ length 0 _ (by omega)
      ⟨_,_,ComplexBackwardInit.result_registers u p.l p.batch p.dim p.B⟩
    · intro k v hk hv
      obtain ⟨c,a,hr⟩ := hv
      have ht := hr.setCounter k
      exact body_safe bounds p.s p.y p.gx p.gy length p.batch p.dim p.B k
        (v.setReg "i" .nat [] (Tile.scalar k)) (columns p u) (mask p u) c a (transition p u)
        (by simp) ht
        (fun _ => hb.initial (by omega))
        (fun hp => hb.previous _ (by omega))
        (hb.gradOutput _ (by omega)) (hb.gradInput _ (by omega))
    · intro k v w _ hv he
      obtain ⟨c,a,hr⟩ := hv
      have ht := hr.setCounter k
      have hrun := ComplexBackwardStep.body_step p.s p.y p.gx p.gy length p.batch p.dim p.B k
        (v.setReg "i" .nat [] (Tile.scalar k)) (columns p u) (mask p u) c a (transition p u) (by simp) ht
      rw [hrun] at he
      obtain rfl := Option.some.inj he
      exact ⟨_,_,ComplexBackwardStep.result_registers _ _ _ _ _ _ _ _ _ _ _ _ _ ht⟩
  · intro w he
    obtain ⟨v,hloop,hv⟩ := run_loop p u length sep
    rw [hloop] at he
    obtain rfl := Option.some.inj he
    exact post_safe bounds v p.gs p.gl (columns p u) (mask p u)
      (carry p u length length) (acc p u length length) (transition p u)
      hv.registers hb.gradInitial hb.gradTransition

theorem flatten_ok (p : Layout) (length : Nat) :
    (Erasure.complexBackward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B).toAlgKernel.FlattenOk := by
  simp only [Kernel.FlattenOk,Erasure.complexBackward,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    Except.bind,bind,Bool.false_eq_true,↓reduceIte]
  simp [StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardSafety
