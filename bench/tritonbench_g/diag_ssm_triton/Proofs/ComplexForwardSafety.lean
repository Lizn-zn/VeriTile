import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexForwardLoop
import bench.tritonbench_g.diag_ssm_triton.Proofs.ZeroSafety
import VeriTile.Triton.Memory.TraceRewrite
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexForwardSafety
open VeriTile.Triton ComplexForwardStep ComplexForwardLoop
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Every active component address must fit its allocation. The final
component bound includes the preceding real component. Static reads remain
required for a zero-length sequence; no bound on block versus width is used. -/
structure Bounds (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds) : Prop where
  initial : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.s
  transition : ∀ j, mask p u j=Bool.true → (columns p u j%p.dim)*2+1 < bounds p.l
  input : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.x
  output : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.y

theorem body_safe (bounds : RegionBounds) (x y : RegionName) (batch dim B t : Nat) (u : BlockState)
    (col : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (state transition : TileIndex [B] → ℂ)
    (ht : u.regs .nat [] "t" = some (Tile.scalar t))
    (hc : u.regs .nat [B] "col_offsets" = some ⟨col⟩)
    (hm : u.regs .bool [B] "mask" = some ⟨mask⟩)
    (hsr : u.regs .real [B] "s_real" = some (realTile state))
    (hsi : u.regs .real [B] "s_imag" = some (imagTile state))
    (hlr : u.regs .real [B] "lambda_real" = some (realTile transition))
    (hli : u.regs .real [B] "lambda_imag" = some (imagTile transition))
    (hx : ∀ j, mask j=Bool.true → ComplexForwardStep.offset batch dim t col j*2+1 < bounds x)
    (hy : ∀ j, mask j=Bool.true → ComplexForwardStep.offset batch dim t col j*2+1 < bounds y) :
    Stmt.TraceSafeList bounds (ComplexForwardSource.body x y batch dim B) u := by
  simp only [ComplexForwardSource.body,Erasure.complexForward,
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
    ht,hc,hm,hsr,hsi,hlr,hli,realTile,imagTile,Nat.mul_assoc,
    FloatDType.storeValue,FloatDType.toWithBot,WithBot.unbotD_some]
  simp only [Option.some.injEq]
  refine ⟨?_,?_,?_,?_⟩
  · intro offsets hoff i ⟨masks,hmask,hact⟩
    subst offsets; subst masks
    exact Nat.lt_of_le_of_lt (Nat.le_add_right _ 1) (hx i hact)
  · intro offsets hoff i ⟨masks,hmask,hact⟩
    subst offsets; subst masks
    exact hx i hact
  · intro offsets hoff i ⟨masks,hmask,hact⟩
    subst offsets; subst masks
    exact Nat.lt_of_le_of_lt (Nat.le_add_right _ 1) (hy i hact)
  · intro offsets hoff i ⟨masks,hmask,hact⟩
    subst offsets; subst masks
    exact hy i hact


theorem prelude_safe (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds)
    (hb : Bounds p u length bounds) :
    Stmt.TraceSafeList bounds (ComplexForwardSource.prelude p.s p.l p.batch p.dim p.B) u := by
  have hstatic : ZeroSafety.complexForwardBounds bounds u p.s p.l p.batch p.dim p.B := by
    constructor <;> intro j hj
    · exact hb.initial ⟨j,PUnit.unit⟩ (by
        simpa only [mask,ComplexForwardInit.active,ComplexForwardInit.columns,decide_eq_true_eq] using hj)
    · exact hb.transition ⟨j,PUnit.unit⟩ (by
        simpa only [mask,ComplexForwardInit.active,ComplexForwardInit.columns,decide_eq_true_eq] using hj)
  have h := ZeroSafety.complex_forward_safe bounds u p.s p.x p.y p.l p.batch p.dim p.B hstatic
  unfold Kernel.TraceSafe at h
  rw [ComplexForwardSource.source_body] at h
  exact ((TraceRewrite.append_iff bounds _ _ u).mp h).1

theorem trace_safe (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds)
    (hb : Bounds p u length bounds) :
    (Erasure.complexForward Bool.false p.s p.x p.y p.l length p.batch p.dim p.B).toAlgKernel.TraceSafe bounds u := by
  unfold Kernel.TraceSafe
  rw [ComplexForwardSource.source_body]
  apply TraceProofs.append_safe _ u (prelude_safe p u length bounds hb)
  intro v he
  rw [ComplexForwardInit.prefix_step] at he
  obtain rfl := Option.some.inj he
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
  simp only [Stmt.TraceSafe]
  let P := fun v => ∃ state : Lane p → ℂ, Registers v (columns p u) (mask p u) state (transition p u)
  apply TraceProofs.loop_safe (P := P) ?_ ?_ length 0 _ (by omega)
    ⟨initial p u,ComplexForwardInit.result_registers u p.s p.l p.batch p.dim p.B⟩
  · intro i v hi hv
    obtain ⟨state,hr⟩ := hv
    have ht := hr.setCounter i
    exact body_safe bounds p.x p.y p.batch p.dim p.B i (v.setReg "t" .nat [] (Tile.scalar i))
      (columns p u) (mask p u) state (transition p u) (by simp)
      ht.columns ht.active ht.realState ht.imagState ht.realTransition ht.imagTransition
      (hb.input i hi) (hb.output i hi)
  · intro i v w _ hv he
    obtain ⟨state,hr⟩ := hv
    have ht := hr.setCounter i
    have hrun := body_step p.x p.y p.batch p.dim p.B i (v.setReg "t" .nat [] (Tile.scalar i))
      (columns p u) (mask p u) state (transition p u) (by simp)
      ht.columns ht.active ht.realState ht.imagState ht.realTransition ht.imagTransition
    rw [hrun] at he
    obtain rfl := Option.some.inj he
    exact ⟨_,ComplexForwardStep.result_registers _ _ _ _ _ _ _ _ ht⟩

theorem flatten_ok (p : Layout) (length : Nat) :
    (Erasure.complexForward Bool.false p.s p.x p.y p.l length p.batch p.dim p.B).toAlgKernel.FlattenOk := by
  simp only [Kernel.FlattenOk,Erasure.complexForward,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    Except.bind,bind,Bool.false_eq_true,↓reduceIte]
  simp [StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexForwardSafety
