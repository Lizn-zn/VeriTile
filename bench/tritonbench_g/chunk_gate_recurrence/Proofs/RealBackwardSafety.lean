import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealBackward
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealForwardSafety

/-! Bounds for the actual backward accesses. S/DI/DG visit only N-1 chunks;
DS and D also visit chunk zero in the DL postlude. Unused decremented pointer
values introduce no memory-access requirement. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardSafety
open VeriTile.Triton RealProgram RealBackwardProgram RealBackwardPointers RealBackwardLoop
open RealForwardProgram (stride stop)
open RealForwardSafety (load_safe)
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

structure Bounds (p : Layout) (l : Launch) (S D DI DG DL DS : RegionName) (bounds : RegionBounds) : Prop where
  source : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N-1 → address p l t e.val j.val < bounds S
  incoming : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N → address p l t e.val j.val < bounds DS
  gate : ∀ t, t < p.N → gateAddress p l t < bounds D
  outputDI : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N-1 → address p l t e.val j.val < bounds DI
  outputDG : ∀ t, t < p.N-1 → gradientAddress p l t < bounds DG
  outputDL : ∀ (e : Fin p.BK) (j : Fin p.BV), seedAddress p l e.val j.val < bounds DL

theorem store_safe {shape : TileShape} (bounds : RegionBounds) (name value : RegName)
    (ptrs : Tile .ptr shape) (u : BlockState) (hp : u.regs .ptr shape name = some ptrs)
    (hb : ∀ i, (ptrs.data i).2 < bounds (ptrs.data i).1) :
    Stmt.TraceSafe bounds (.store .real shape (.ptr (.ref .ptr shape name)) (.ref .real shape value) .none) u := by
  rw [Stmt.TraceSafe]
  refine ⟨by simp [MemAccess.SafeAt,Op.SafeAt,Op.SafeAt.eq_def],by simp [Op.SafeAt,Op.SafeAt.eq_def],trivial,?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ps he i _
  rw [evalOp_ref,hp] at he
  cases he
  exact hb i

theorem store_regs {shape : TileShape} (ptr : Op .ptr shape) (val : Op .real shape) (s t : BlockState)
    (h : stepStmt (.store .real shape (.ptr ptr) val .none) s = some t) : t.regs = s.regs := by
  simp only [stepStmt] at h
  cases hv : evalOp val s with
  | none => simp [hv] at h
  | some values =>
    cases hp : evalOp ptr s with
    | none => simp [hv,hp] at h
    | some ps =>
      simp only [hv,hp] at h
      cases h
      funext d sh name
      exact BlockState.foldl_writeMemTypedAt_masked_regs .real _ _ _ _ _ s d sh name

theorem iteration_safe (p : Layout) (l : Launch) (S D DI DG DL DS : RegionName)
    (bounds : RegionBounds) (hb : Bounds p l S D DI DG DL DS bounds)
    (k : Nat) (hk : k < p.N-1) (u : BlockState)
    (hs : u.regs .ptr [p.BK,p.BV] "S" = some (backwardPointer p l S (p.N-2) k))
    (hdi : u.regs .ptr [p.BK,p.BV] "DI" = some (backwardPointer p l DI (p.N-2) k))
    (hds : u.regs .ptr [p.BK,p.BV] "DS" = some (backwardPointer p l DS (p.N-1) k))
    (hdg : u.regs .ptr [] "DG" = some (backwardGradient p l DG k))
    (hd : u.regs .ptr [] "D" = some (backwardGate p l D k)) :
    Stmt.TraceSafeList bounds (loopBody p) u := by
  have hs' := hs.trans (congrArg some (backwardPointer_live p l S (p.N-2) k (by omega)))
  have hdi' := hdi.trans (congrArg some (backwardPointer_live p l DI (p.N-2) k (by omega)))
  have hds' := hds.trans (congrArg some (backwardPointer_live p l DS (p.N-1) k (by omega)))
  have hdg' := hdg.trans (congrArg some (backwardGradient_live p l DG k (by omega)))
  have hd' := hd.trans (congrArg some (backwardGate_live p l D k (by omega)))
  unfold loopBody
  apply TraceProofs.assign_cons (load_safe bounds "S" (pointer p l S (p.N-2-k)) u hs'
    (fun i => hb.source _ i.1 i.2.1 (by omega)))
  intro s
  apply TraceProofs.assign_cons (load_safe bounds "DS" (pointer p l DS (p.N-1-k)) _ (by simp [hds'])
    (fun i => hb.incoming _ i.1 i.2.1 (by omega)))
  intro ds
  apply TraceProofs.assign_cons (load_safe bounds "D" (gatePointer p l D (p.N-1-k)) _ (by simp [hd'])
    (fun _ => hb.gate _ (by omega)))
  intro d
  apply TraceProofs.assign_cons
  · simp [update,Op.SafeAt,Op.SafeAt.eq_def]
  intro acc
  apply TraceProofs.assign_cons
  · simp [gradient,Op.SafeAt,Op.SafeAt.eq_def]
  intro grad
  apply Stmt.TraceSafeList.cons_intro (store_safe bounds "DG" "DG_i"
    (Tile.scalar (DG,gradientAddress p l (p.N-2-k))) _ (by simp [hdg'])
    (fun _ => hb.outputDG _ (by omega)))
  intro v he
  have hr := store_regs _ _ _ v he
  apply Stmt.TraceSafeList.cons_intro (store_safe bounds "DI" "Dacc" (pointer p l DI (p.N-2-k)) _
    (by rw [hr]; simp [hdi']) (fun i => hb.outputDI _ i.1 i.2.1 (by omega)))
  intro w _
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def,stride,gradientStride]

theorem loop_safe (initial : BlockState) (p : Layout) (l : Launch) (S D DI DG DL DS : RegionName)
    (a : Inputs p) (sep : Separation S D DI DG DS) (hi : NoCollision p l)
    (hgi : Function.Injective (gradientAddress p l)) (bounds : RegionBounds)
    (hb : Bounds p l S D DI DG DL DS bounds) (u : BlockState)
    (h : Holds initial p l S D DI DG DS a 0 u) : Stmt.TraceSafe bounds (statement p) u := by
  rw [statement,Stmt.TraceSafe]
  have hs : (stop p).SafeAt bounds u := by simp [stop,Op.SafeAt,Op.SafeAt.eq_def]
  simp only [Op.SafeAt,hs,true_and,evalOp_constNat,RealForwardLoop.stop_eval,Tile.scalar]
  apply TraceProofs.range_safe (P := Holds initial p l S D DI DG DS a)
  · intro k v hk hv
    exact iteration_safe p l S D DI DG DL DS bounds hb k hk _
      (by simp [hv.source]) (by simp [hv.target]) (by simp [hv.incoming])
      (by simp [hv.gradient]) (by simp [hv.gate])
  · intro k v w hk hv he
    obtain ⟨w',he',hw⟩ := iteration initial p l S D DI DG DS a sep hi hgi k v hk hv
    rw [he] at he'
    cases he'
    exact hw
  · exact h

theorem init_safe (p : Layout) (S D DI DG DL DS : RegionName) (bounds : RegionBounds) (u : BlockState) :
    Stmt.TraceSafeList bounds (initCode p S D DI DG DL DS) u := by
  rw [RealBackwardInit.code_exact]
  apply TraceProofs.all_safe
  simp [RealBackwardInit.prepareCode,RealBackwardInit.pointerCode,RealBackwardInit.pointerOp,
    RealBackwardInit.offsetOp,RealBackwardInit.headOp,RealBackwardInit.startOp,
    RealBackwardInit.gateOp,RealBackwardInit.gradientOp,Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]

theorem post_safe (p : Layout) (l : Launch) (S D DI DG DL DS : RegionName) (bounds : RegionBounds)
    (hb : Bounds p l S D DI DG DL DS bounds) (u : BlockState) (hN : 0 < p.N)
    (hfixed : Fixed p l u)
    (hds : u.regs .ptr [p.BK,p.BV] "DS" = some (pointer p l DS 0))
    (hd : u.regs .ptr [] "D" = some (gatePointer p l D 0)) :
    Stmt.TraceSafeList bounds (RealBackwardPost.code p DL) u := by
  unfold RealBackwardPost.code
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (RealBackwardInit.seedPointer_eval p l DL u hfixed))]
  refine ⟨by simp [Stmt.TraceSafe,RealBackwardInit.seedPointerOp,RealBackwardInit.offsetOp,
    RealBackwardInit.headOp,Op.SafeAt,Op.SafeAt.eq_def],?_⟩
  apply TraceProofs.assign_cons (load_safe bounds "DS" (pointer p l DS 0) _ (by simp [hds])
    (fun i => hb.incoming 0 i.1 i.2.1 hN))
  intro ds
  apply TraceProofs.assign_cons (load_safe bounds "D" (gatePointer p l D 0) _ (by simp [hd])
    (fun _ => hb.gate 0 hN))
  intro d
  apply TraceProofs.assign_cons
  · simp [update,Op.SafeAt,Op.SafeAt.eq_def]
  intro acc
  apply Stmt.TraceSafeList.cons_intro (store_safe bounds "DL" "Dacc" (seedPointer p l DL) _ (by simp)
    (fun i => hb.outputDL i.1 i.2.1))
  intro _ _
  exact Stmt.TraceSafeList.nil_intro

theorem source_safe (p : Layout) (S D DI DG DL DS : RegionName) (a : Inputs p)
    (storage gate incoming : TypedSource.Storage) (H : Nat) (u : BlockState) (hN : 0 < p.N)
    (sep : Separation S D DI DG DS) (hi : NoCollision p (launch u))
    (hgi : Function.Injective (gradientAddress p (launch u)))
    (hS : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N-1 →
      u.readMem S (address p (launch u) t e.val j.val) = a.source t e j)
    (hDS : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N →
      u.readMem DS (address p (launch u) t e.val j.val) = a.incoming t e j)
    (hD : ∀ t, t < p.N → u.readMem D (gateAddress p (launch u) t) = a.gate t)
    (bounds : RegionBounds) (hb : Bounds p (launch u) S D DI DG DL DS bounds) :
    (TypedSource.backward storage gate incoming S D DI DG DL DS H
      p.N p.K p.V p.BK p.BV).eraseDType.toAlgKernel.TraceSafe bounds u := by
  rw [Kernel.TraceSafe,source_body,List.append_assoc]
  apply TraceProofs.append_safe
  · exact init_safe p S D DI DG DL DS bounds u
  · intro v he
    rw [RealBackwardInit.executes p S D DI DG DL DS u] at he
    cases he
    have hinv := RealBackward.invariant p (launch u) S D DI DG DS a u hS hDS hD
    apply Stmt.TraceSafeList.cons_intro
    · exact loop_safe u p (launch u) S D DI DG DL DS a sep hi hgi bounds hb _ hinv
    · intro w he
      obtain ⟨w',he',hw⟩ := RealBackwardLoop.executes u p (launch u) S D DI DG DS a sep hi hgi _ hinv
      rw [he] at he'
      cases he'
      rw [RealBackwardPost.code_exact]
      exact post_safe p (launch u) S D DI DG DL DS bounds hb w hN hw.fixed
        (by simpa only [backwardPointer_live p (launch u) DS (p.N-1) (p.N-1) (Nat.le_refl _),Nat.sub_self] using hw.incoming)
        (by simpa only [backwardGate_live p (launch u) D (p.N-1) (Nat.le_refl _),Nat.sub_self] using hw.gate)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealBackwardSafety
