import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealForward
import VeriTile.Triton.Memory.TraceProofs

/-! Active-address safety follows the same complete forward execution. The
seed allocation is required only when used. All loads and stores are unmasked,
so every lane of each visited tile must be allocated. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealForwardSafety
open VeriTile.Triton RealProgram RealForwardProgram RealForwardStep RealForwardLoop
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

structure Bounds (p : Layout) (l : Launch) (S D O seed : RegionName)
    (hasSeed : Bool) (bounds : RegionBounds) : Prop where
  source : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N-1 →
    address p l t e.val j.val < bounds S
  gate : ∀ t, t < p.N-1 → gateAddress p l t < bounds D
  output : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N →
    address p l t e.val j.val < bounds O
  seed : hasSeed = Bool.true → ∀ (e : Fin p.BK) (j : Fin p.BV),
    seedAddress p l e.val j.val < bounds seed

theorem load_safe {shape : TileShape} (bounds : RegionBounds) (name : RegName)
    (ptrs : Tile .ptr shape) (u : BlockState) (hp : u.regs .ptr shape name = some ptrs)
    (hb : ∀ i, (ptrs.data i).2 < bounds (ptrs.data i).1) :
    (Op.load .real (.ptr (.ref .ptr shape name)) .none).SafeAt bounds u := by
  rw [Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt,Op.SafeAt.eq_def], trivial, ?_⟩
  refine ⟨by simp [Op.MemorySafe], ?_⟩
  intro ps he i _
  rw [evalOp_ref,hp] at he
  cases he
  exact hb i

theorem store_safe (p : Layout) (bounds : RegionBounds) (ptrs : Tile .ptr [p.BK,p.BV])
    (u : BlockState) (hp : u.regs .ptr [p.BK,p.BV] "O" = some ptrs)
    (hb : ∀ i, (ptrs.data i).2 < bounds (ptrs.data i).1) :
    Stmt.TraceSafe bounds (.store .real [p.BK,p.BV] (.ptr (.ref .ptr [p.BK,p.BV] "O"))
      (.ref .real [p.BK,p.BV] "acc") .none) u := by
  rw [Stmt.TraceSafe]
  refine ⟨by simp [MemAccess.SafeAt,Op.SafeAt,Op.SafeAt.eq_def], by simp [Op.SafeAt,Op.SafeAt.eq_def], trivial, ?_⟩
  refine ⟨by simp [Op.MemorySafe], ?_⟩
  intro ps he i _
  rw [evalOp_ref,hp] at he
  cases he
  exact hb i

theorem iteration_safe (p : Layout) (l : Launch) (S D O seed : RegionName) (hasSeed : Bool)
    (bounds : RegionBounds) (hb : Bounds p l S D O seed hasSeed bounds)
    (c : Nat) (hc : c < p.N-1) (u : BlockState)
    (hs : u.regs .ptr [p.BK,p.BV] "S" = some (pointer p l S c))
    (hd : u.regs .ptr [] "D" = some (gatePointer p l D c))
    (ho : u.regs .ptr [p.BK,p.BV] "O" = some (pointer p l O (c+1))) :
    Stmt.TraceSafeList bounds (loopBody p) u := by
  unfold loopBody
  apply TraceProofs.assign_cons (load_safe bounds "D" (gatePointer p l D c) u hd
    (fun _ => hb.gate c (by omega)))
  intro d
  apply TraceProofs.assign_cons (load_safe bounds "S" (pointer p l S c) _ (by simp [hs])
    (fun i => hb.source c i.1 i.2.1 (by omega)))
  intro s
  apply TraceProofs.assign_cons
  · simp [update,Op.SafeAt,Op.SafeAt.eq_def]
  intro acc
  apply Stmt.TraceSafeList.cons_intro (store_safe p bounds (pointer p l O (c+1)) _
    (by simp [ho]) (fun i => hb.output (c+1) i.1 i.2.1 (by omega)))
  intro v _
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe,Op.SafeAt,stride]

theorem loop_safe (initial : BlockState) (p : Layout) (l : Launch) (S D O seed : RegionName)
    (a : Inputs p) (hasSeed : Bool) (hSO : S ≠ O) (hDO : D ≠ O) (hi : NoCollision p l)
    (bounds : RegionBounds) (hb : Bounds p l S D O seed hasSeed bounds) (u : BlockState)
    (h : Holds initial p l S D O a hasSeed 0 u) :
    Stmt.TraceSafe bounds (statement p) u := by
  rw [statement,Stmt.TraceSafe]
  have hs : (stop p).SafeAt bounds u := by simp [stop,Op.SafeAt,Op.SafeAt.eq_def]
  simp only [Op.SafeAt,hs,true_and,evalOp_constNat,stop_eval,Tile.scalar]
  apply TraceProofs.range_safe (P := Holds initial p l S D O a hasSeed)
  · intro c v hc hv
    exact iteration_safe p l S D O seed hasSeed bounds hb c hc _
      (by simp [hv.source]) (by simp [hv.gate]) (by simp [hv.target])
  · intro c v w hc hv he
    obtain ⟨w',he',hw⟩ := iteration initial p l S D O a hasSeed hSO hDO hi c v hc hv
    rw [he] at he'
    cases he'
    exact hw
  · exact h

theorem seed_safe (p : Layout) (l : Launch) (seed : RegionName) (hasSeed : Bool)
    (bounds : RegionBounds) (u : BlockState)
    (hh : u.regs .nat [] "offset_bh" = some (Tile.scalar l.head))
    (hk : u.regs .nat [] "offset_d" = some (Tile.scalar l.key))
    (hv : u.regs .nat [] "offset_s" = some (Tile.scalar l.value))
    (hb : hasSeed = Bool.true → ∀ (e : Fin p.BK) (j : Fin p.BV),
      seedAddress p l e.val j.val < bounds seed) :
    Stmt.TraceSafe bounds (.ifThenElse (.constBool hasSeed)
      (RealForwardInit.seedCode p seed) (RealForwardInit.zeroCode p)) u := by
  cases hasSeed
  · simp only [Stmt.TraceSafe,Op.SafeAt,evalOp.eq_def,Tile.scalar_data,true_and,↓reduceIte,
      Bool.false_eq_true]
    apply TraceProofs.all_safe
    simp [RealForwardInit.zeroCode,Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]
  · simp only [Stmt.TraceSafe,Op.SafeAt,evalOp.eq_def,Tile.scalar_data,true_and,↓reduceIte,
      RealForwardInit.seedCode]
    rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
      (RealForwardInit.pointer_eval p l seed Bool.true u hh hk hv))]
    simp only [↓reduceIte]
    refine ⟨by simp [Stmt.TraceSafe,RealForwardInit.pointerOp,RealForwardInit.headOp,Op.SafeAt,Op.SafeAt.eq_def], ?_⟩
    apply Stmt.TraceSafeList.cons_intro
    · simpa only [Stmt.TraceSafe] using load_safe bounds "last_kv" (seedPointer p l seed)
        (u.setReg "last_kv" .ptr [p.BK,p.BV] (seedPointer p l seed))
        (by simp) (fun i => hb rfl i.1 i.2.1)
    · intro _ _; exact Stmt.TraceSafeList.nil_intro

theorem init_safe (p : Layout) (S D O seed : RegionName) (a : Inputs p) (hasSeed : Bool)
    (u : BlockState) (hN : 0 < p.N) (bounds : RegionBounds)
    (hb : Bounds p (launch u) S D O seed hasSeed bounds)
    (hSeed : hasSeed = Bool.true → ∀ e j,
      u.readMem seed (seedAddress p (launch u) e.val j.val) = a.seed e j) :
    Stmt.TraceSafeList bounds (initCode p S D O seed hasSeed) u := by
  rw [RealForwardInit.code_exact]
  unfold RealForwardInit.code
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_programId 0 u))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (RealForwardInit.pointer_eval p (launch u) S Bool.false _
    (by simp [launch]) (by simp [launch]) (by simp [launch])))]
  refine ⟨by simp [Stmt.TraceSafe,RealForwardInit.pointerOp,RealForwardInit.headOp,Op.SafeAt,Op.SafeAt.eq_def], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (RealForwardInit.pointer_eval p (launch u) O Bool.false _
    (by simp [launch]) (by simp [launch]) (by simp [launch])))]
  refine ⟨by simp [Stmt.TraceSafe,RealForwardInit.pointerOp,RealForwardInit.headOp,Op.SafeAt,Op.SafeAt.eq_def], ?_⟩
  rw [TraceProofs.cons_of_run_iff (RealForwardInit.seed_executes p (launch u) seed a hasSeed _
    (by simp [launch]) (by simp [launch]) (by simp [launch]) (by simpa using hSeed))]
  refine ⟨seed_safe p (launch u) seed hasSeed bounds _
    (by simp [launch]) (by simp [launch]) (by simp [launch]) hb.seed, ?_⟩
  apply Stmt.TraceSafeList.cons_intro (store_safe p bounds (pointer p (launch u) O 0) _
    (by cases hasSeed <;> simp [RealForwardInit.seeded]) (fun i => hb.output 0 i.1 i.2.1 hN))
  intro v _
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe,stride,RealForwardInit.gateOp,Op.SafeAt,Op.SafeAt.eq_def]

theorem source_safe (p : Layout) (S D O seed : RegionName) (a : Inputs p) (hasSeed : Bool)
    (storage gate seedStorage : TypedSource.Storage) (H : Nat) (u : BlockState)
    (hN : 0 < p.N) (hSO : S ≠ O) (hDO : D ≠ O) (hi : NoCollision p (launch u))
    (hS : ∀ (t : Nat) (e : Fin p.BK) (j : Fin p.BV), t < p.N →
      u.readMem S (address p (launch u) t e.val j.val) = a.source t e j)
    (hD : ∀ t, t < p.N → u.readMem D (gateAddress p (launch u) t) = a.gate t)
    (hSeed : hasSeed = Bool.true → ∀ e j,
      u.readMem seed (seedAddress p (launch u) e.val j.val) = a.seed e j)
    (bounds : RegionBounds) (hb : Bounds p (launch u) S D O seed hasSeed bounds) :
    (TypedSource.forward storage gate seedStorage S D O seed H
      p.N p.K p.V p.BK p.BV hasSeed).eraseDType.toAlgKernel.TraceSafe bounds u := by
  rw [Kernel.TraceSafe,source_body]
  apply TraceProofs.append_safe
  · exact init_safe p S D O seed a hasSeed u hN bounds hb hSeed
  · intro v he
    rw [RealForwardInit.executes p S D O seed a hasSeed u hSeed] at he
    cases he
    apply Stmt.TraceSafeList.cons_intro
    · exact loop_safe u p (launch u) S D O seed a hasSeed hSO hDO hi bounds hb _
        (RealForward.invariant p (launch u) S D O seed a hasSeed u hN hSO hDO hi hS hD)
    · intro _ _; exact Stmt.TraceSafeList.nil_intro

theorem source_flatten_ok (p : Layout) (S D O seed : RegionName) (hasSeed : Bool)
    (storage gate seedStorage : TypedSource.Storage) (H : Nat) :
    (TypedSource.forward storage gate seedStorage S D O seed H
      p.N p.K p.V p.BK p.BV hasSeed).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk,source_body,RealForwardInit.code_exact]
  simp [RealForwardInit.code,RealForwardInit.seedCode,RealForwardInit.zeroCode,
    RealForwardInit.pointerOp,RealForwardInit.headOp,RealForwardInit.gateOp,
    statement,loopBody,update,stop,stride,StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk]
  all_goals simp [Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealForwardSafety
