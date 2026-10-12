import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackwardFlatLoop

/-! Pair the complete region and flat executions, including the DL postlude.
Only unused final pointer registers may differ; every tagged memory cell agrees.
The zero padding oracle here is internal to the flattening bridge. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardFlatExecution
open VeriTile.Triton RealProgram RealBackwardProgram RealBackwardPointers RealBackwardLoop RealBackwardInit
open RealBackwardFlatState
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem init_flatten_ok (p : Layout) (S D DI DG DL DS : RegionName) :
    StmtList.FlattenOk (initCode p S D DI DG DL DS) := by
  rw [RealBackwardInit.code_exact]
  simp [prepareCode,pointerCode,pointerOp,offsetOp,headOp,startOp,gateOp,gradientOp,
    StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk,Op.FlattenOk.eq_def]

theorem post_flatten_ok (p : Layout) (DL : RegionName) :
    StmtList.FlattenOk (RealBackwardPost.code p DL) := by
  simp [RealBackwardPost.code,seedPointerOp,offsetOp,headOp,update,
    StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk,Op.FlattenOk.eq_def]

theorem paired_executes (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Layout) (S D DI DG DL DS : RegionName) (a : Inputs p)
    (storage gate incoming : TypedSource.Storage) (H : Nat) (u : BlockState)
    (hN : 0 < p.N) (sep : Separation S D DI DG DS)
    (hi : NoCollision p (launch u)) (hgi : Function.Injective (gradientAddress p (launch u)))
    (hS : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N-1 →
      u.readMem S (address p (launch u) t e.val j.val) = a.source t e j)
    (hDS : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N →
      u.readMem DS (address p (launch u) t e.val j.val) = a.incoming t e j)
    (hD : ∀ t, t < p.N → u.readMem D (gateAddress p (launch u) t) = a.gate t)
    (hb : RealBackwardSafety.Bounds p (launch u) S D DI DG DL DS A.extent)
    (hu : u.undef = fun _ _ => 0) :
    ∃ v t, exec (TypedSource.backward storage gate incoming S D DI DG DL DS H
        p.N p.K p.V p.BK p.BV).eraseDType.toAlgKernel u = some v ∧
      exec (A.flattenKernel (TypedSource.backward storage gate incoming S D DI DG DL DS H
        p.N p.K p.V p.BK p.BV).eraseDType.toAlgKernel) (A.flattenState u) = some t ∧
      Agree t (A.flattenState v) := by
  let l := launch u
  let start := initial p l S D DI DG DS u
  have heinit := RealBackwardInit.executes p S D DI DG DL DS u
  have hfinit := A.stepStmts_flatten hd hcov (initCode p S D DI DG DL DS) u
    (RealBackwardSafety.init_safe p S D DI DG DL DS A.extent u)
    (init_flatten_ok p S D DI DG DL DS) hu
  rw [heinit,Option.map_some] at hfinit
  obtain ⟨mid,flatMid,he,hflat,h,ha,hmundef⟩ := RealBackwardFlatLoop.executes A hd hcov u p l S D DI DG DL DS a
    sep hi hgi hb start (RealBackward.invariant p l S D DI DG DS a u hS hDS hD)
    (by change u.undef = _; exact hu)
  have hds : mid.regs .ptr [p.BK,p.BV] "DS" = some (pointer p l DS 0) := by
    simpa only [backwardPointer_live p l DS (p.N-1) (p.N-1) (Nat.le_refl _),Nat.sub_self] using h.incoming
  have hgate : mid.regs .ptr [] "D" = some (gatePointer p l D 0) := by
    simpa only [backwardGate_live p l D (p.N-1) (Nat.le_refl _),Nat.sub_self] using h.gate
  have hcarry : mid.regs .real [p.BK,p.BV] "Dacc" = some (tile p (RealMath.reverse a p.N 1)) := by
    have hremain : p.N-(p.N-1) = 1 := by omega
    simpa only [hremain] using h.carry
  have hp := RealBackwardPost.executes p l D DL DS a (RealMath.reverse a p.N 1) mid h.fixed
    hds hgate hcarry (fun e j => h.inputIncoming 0 e j hN) (h.inputGate 0 hN)
  have hfp := A.stepStmts_flatten hd hcov (RealBackwardPost.code p DL) mid
    (RealBackwardSafety.post_safe p l S D DI DG DL DS A.extent hb mid hN h.fixed hds hgate)
    (post_flatten_ok p DL) hmundef
  rw [hp,Option.map_some] at hfp
  obtain ⟨flatFinal,hpost,haFinal⟩ := post_congr A p DL ha hfp
  refine ⟨RealBackwardPost.result p l DL a (RealMath.reverse a p.N 1) mid,flatFinal,?_,?_,haFinal⟩
  · rw [exec,source_body,ProgramRewrite.append,ProgramRewrite.append,heinit]
    change (stepStmts [statement p] start).bind (stepStmts (postCode p S D DI DG DL DS)) = _
    rw [stepStmts.cons_some he,stepStmts.nil]
    simpa only [Option.bind_some,RealBackwardPost.code_exact] using hp
  · change stepStmts (A.flattenStmts _) (A.flattenState u) = _
    rw [source_body,FlatAlloc.flattenStmts_append,FlatAlloc.flattenStmts_append,
      ProgramRewrite.append,ProgramRewrite.append,hfinit]
    change (stepStmts [A.flattenStmt (statement p)] (A.flattenState start)).bind
      (stepStmts (A.flattenStmts (postCode p S D DI DG DL DS))) = _
    rw [stepStmts.cons_some hflat,stepStmts.nil]
    simpa only [Option.bind_some,RealBackwardPost.code_exact] using hpost

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardFlatExecution
