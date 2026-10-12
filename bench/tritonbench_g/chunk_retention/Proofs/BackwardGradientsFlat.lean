import bench.tritonbench_g.chunk_retention.Proofs.BackwardGradientsExecution
import bench.tritonbench_g.chunk_retention.Proofs.ForwardStateSafety

/-! Access safety of all four loop loads and the dV store. Bounds refer to
actual active addresses, including independent strides and the combined NT*K
history dimension. Arithmetic statements introduce no hidden memory access. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealQKVSafety
open VeriTile.Triton
open RealHMath (Layout)
open RealHProgram (times load2)
open RealHAccessSafety (MatrixBounds load2_safe store2_safe)
open RealQKVMath (Launch Inputs numBlocks)
open RealQKVProgram RealQKVPointers RealQKVStep RealMatrix RealMatrixStore
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def KeyBounds (p : Layout) (l : Launch) (r : RegionName) :=
  MatrixBounds r (l.head*p.keyHead) p.T p.K p.BT p.BK p.keyTime p.keyFeature (l.chunk*p.BT) (l.key*p.BK)
def QueryBounds (p : Layout) (l : Launch) (r : RegionName) :=
  MatrixBounds r (l.head*p.keyHead) p.K p.T p.BK p.BT p.keyFeature p.keyTime (l.key*p.BK) (l.chunk*p.BT)
def ValueBounds (p : Layout) (l : Launch) (r : RegionName) (n : Nat) :=
  MatrixBounds r (l.head*p.valueHead) p.T p.V p.BT p.BV p.valueTime p.valueFeature (l.chunk*p.BT) (n*p.BV)
def HistoryBounds (p : Layout) (l : Launch) (r : RegionName) (n : Nat) :=
  MatrixBounds r (l.head*p.historyHead) p.V (p.NT*p.K) p.BV p.BK 1 p.historyKey (n*p.BV) (l.chunk*p.K+l.key*p.BK)
def GradientBounds (p : Layout) (l : Launch) (r : RegionName) (n : Nat) :=
  MatrixBounds r (l.head*p.historyHead) (p.NT*p.K) p.V p.BK p.BV p.historyKey 1 (l.chunk*p.K+l.key*p.BK) (n*p.BV)
def DVBounds (p : Layout) (l : Launch) (r : RegionName) (n : Nat) :=
  MatrixBounds r ((l.key*l.heads+l.head)*p.valueHead) p.T p.V p.BT p.BV p.valueTime p.valueFeature (l.chunk*p.BT) (n*p.BV)

structure StepBounds (p : Layout) (l : Launch) (r : Regions) (n : Nat) (bounds : RegionBounds) : Prop where
  value : ValueBounds p l r.value n bounds
  incoming : ValueBounds p l r.incoming n bounds
  history : HistoryBounds p l r.history n bounds
  gradient : GradientBounds p l r.historyGradient n bounds
  output : DVBounds p l r.dv n bounds

structure Bounds (p : Layout) (l : Launch) (r : Regions) (bounds : RegionBounds) : Prop where
  query : QueryBounds p l r.query bounds
  key : KeyBounds p l r.key bounds
  loop : ∀ n, n < numBlocks p → StepBounds p l r n bounds
  dq : KeyBounds p l r.dq bounds
  dk : KeyBounds p l r.dk bounds

theorem iteration_safe (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ)
    (n : Nat) (bounds : RegionBounds) (hb : StepBounds p l r n bounds) (u : BlockState)
    (hf : Fixed p l a scale u) (hn : u.regs .nat [] "i_v" = some (Tile.scalar n)) :
    Stmt.TraceSafeList bounds (RealQKVProgram.iteration p r scale) u := by
  unfold RealQKVProgram.iteration
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (value_eval p l r.value n _
    (by simp [hf.head]) (by simp [hf.key]) (by simp [hf.chunk]) (by simp [hn])))]
  refine ⟨by simp [Stmt.TraceSafe,valuePtr,historyOffset,dvBase,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (history_eval p l r.history n _
    (by simp [hf.head]) (by simp [hf.key]) (by simp [hf.chunk]) (by simp [hn])))]
  refine ⟨by simp [Stmt.TraceSafe,historyPtr,historyOffset,dvBase,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (value_eval p l r.incoming n _
    (by simp [hf.head]) (by simp [hf.key]) (by simp [hf.chunk]) (by simp [hn])))]
  refine ⟨by simp [Stmt.TraceSafe,valuePtr,historyOffset,dvBase,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (historyGradient_eval p l r.historyGradient n _
    (by simp [hf.head]) (by simp [hf.key]) (by simp [hf.chunk]) (by simp [hn])))]
  refine ⟨by simp [Stmt.TraceSafe,historyGradientPtr,historyOffset,dvBase,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (dv_eval p l r.dv n _
    (by simp [hf.head]) (by simp [hf.key]) (by simp [hf.chunk]) (by simp [hn]) (by simp [hf.heads])))]
  refine ⟨by simp [Stmt.TraceSafe,dvPtr,historyOffset,dvBase,times,Op.SafeAt],?_⟩
  apply TraceProofs.assign_cons (load2_safe r.value (l.head*p.valueHead) p.T p.V p.BT p.BV p.valueTime p.valueFeature (l.chunk*p.BT) (n*p.BV) "p_v" bounds _ hb.value
    (by simp [valuePointer]))
  intro value
  apply TraceProofs.assign_cons (load2_safe r.incoming (l.head*p.valueHead) p.T p.V p.BT p.BV p.valueTime p.valueFeature (l.chunk*p.BT) (n*p.BV) "p_do" bounds _ hb.incoming
    (by simp [valuePointer]))
  intro incoming
  apply TraceProofs.assign_cons (load2_safe r.history (l.head*p.historyHead) p.V (p.NT*p.K) p.BV p.BK 1 p.historyKey (n*p.BV) (l.chunk*p.K+l.key*p.BK) "p_h" bounds _ hb.history
    (by simp [historyPointer]))
  intro history
  apply TraceProofs.assign_cons (load2_safe r.historyGradient (l.head*p.historyHead) (p.NT*p.K) p.V p.BK p.BV p.historyKey 1 (l.chunk*p.K+l.key*p.BK) (n*p.BV) "p_dh" bounds _ hb.gradient
    (by simp [historyGradientPointer]))
  intro gradient
  apply TraceProofs.assign_cons (by simp [updateDS,add,dot,scaleBy,weighted,ref,Op.SafeAt,Op.SafeAt.eq_def])
  intro resultupdateDS
  apply TraceProofs.assign_cons (by simp [updateDQ,add,dot,scaleBy,weighted,ref,Op.SafeAt,Op.SafeAt.eq_def])
  intro resultupdateDQ
  apply TraceProofs.assign_cons (by simp [updateDK,add,dot,scaleBy,weighted,ref,Op.SafeAt,Op.SafeAt.eq_def])
  intro resultupdateDK
  apply TraceProofs.assign_cons (by simp [outputDV,add,dot,scaleBy,weighted,ref,Op.SafeAt,Op.SafeAt.eq_def])
  intro resultoutputDV
  apply Stmt.TraceSafeList.cons_intro
  · exact store2_safe r.dv ((l.key*l.heads+l.head)*p.valueHead) p.T p.V p.BT p.BV
      p.valueTime p.valueFeature (l.chunk*p.BT) (n*p.BV) "p_dv" "b_dv" bounds _ hb.output
      (by simp [dvPointer])
  · intro _ _; exact Stmt.TraceSafeList.nil_intro

theorem loop_safe (initial : BlockState) (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (scale : ℝ) (hu : initial.undef = fun _ _ => 0) (hs : Separate r) (hinj : RealQKVLoop.NoCollision p l r)
    (bounds : RegionBounds) (hb : ∀ n, n < numBlocks p → StepBounds p l r n bounds) (u : BlockState)
    (h : RealQKVLoop.Holds initial p l r a scale 0 u) :
    Stmt.TraceSafe bounds (loop p r scale) u := by
  rw [loop,Stmt.TraceSafe]
  simp only [Op.SafeAt,RealQKVLoop.bound_eval,evalOp_constNat,Tile.scalar,true_and]
  apply TraceProofs.range_safe (P := RealQKVLoop.Holds initial p l r a scale)
  · intro n v hn hv
    apply iteration_safe p l r a scale n bounds (hb n hn) _ _ (by simp)
    rcases hv.fixed with ⟨hk,ht,hh,hnh,hwq,hwk,hws,hq,hk',hs'⟩
    constructor <;> simp [hk,ht,hh,hnh,hwq,hwk,hws,hq,hk',hs']
  · intro n v w hn hv he
    obtain ⟨w',he',hw⟩ := RealQKVLoop.iteration initial p l r a scale hu hs hinj n v hn hv
    rw [he] at he'
    cases he'
    exact hw
  · exact h

theorem init_safe (p : Layout) (r : Regions) (scale : ℝ) (bounds : RegionBounds) (u : BlockState)
    (hq : QueryBounds p (launch u) r.query bounds) (hk : KeyBounds p (launch u) r.key bounds) :
    Stmt.TraceSafeList bounds (initialization p r scale) u := by
  have hsplit : initialization p r scale = RealQKVDecay.prologue p scale ++
      (initialization p r scale).drop 12 := rfl
  rw [hsplit]
  apply TraceProofs.append_safe _ u
  · apply TraceProofs.all_safe
    simp [RealQKVDecay.prologue,RealHProgram.baseOp,RealHProgram.decayLane,RealDecayTiles.queryOp,
      RealDecayTiles.maskOp,RealDecayTiles.scoreOp,scaleBy,Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]
  intro v he
  rw [RealQKVDecay.executes] at he
  cases he
  unfold initialization
  simp only [List.drop_succ_cons,List.drop_zero]
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (query_eval p (launch u) r.query _
    (by simp [RealQKVDecay.state,launch]) (by simp [RealQKVDecay.state,launch]) (by simp [RealQKVDecay.state,launch])))]
  refine ⟨by simp [Stmt.TraceSafe,queryPtr,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (key_eval p (launch u) r.key _
    (by simp [RealQKVDecay.state,launch]) (by simp [RealQKVDecay.state,launch]) (by simp [RealQKVDecay.state,launch])))]
  refine ⟨by simp [Stmt.TraceSafe,keyPtr,times,Op.SafeAt],?_⟩
  apply TraceProofs.assign_cons (load2_safe r.query ((launch u).head*p.keyHead) p.K p.T p.BK p.BT
    p.keyFeature p.keyTime ((launch u).key*p.BK) ((launch u).chunk*p.BT) "p_q" bounds _ hq
    (by simp [queryPointer]))
  intro query
  apply TraceProofs.assign_cons (load2_safe r.key ((launch u).head*p.keyHead) p.T p.K p.BT p.BK
    p.keyTime p.keyFeature ((launch u).chunk*p.BT) ((launch u).key*p.BK) "p_k" bounds _ hk
    (by simp [keyPointer]))
  intro key
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe,score,scaleBy,dot,ref,pointwise,weighted,Op.SafeAt,Op.SafeAt.eq_def]

theorem final_safe (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ)
    (swapped : Bool) (bounds : RegionBounds) (u : BlockState)
    (hf : Fixed p l a scale u) (hc : Carries p l a scale (numBlocks p) u)
    (hq : KeyBounds p l r.dq bounds) (hk : KeyBounds p l r.dk bounds) :
    Stmt.TraceSafeList bounds (finalization p r scale swapped) u := by
  unfold finalization
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (RealQKVAlgebra.final_ds p l a scale u hf.scoreWeights hc.ds))]
  refine ⟨by simp [Stmt.TraceSafe,finalDS,pointwise,weighted,scaleBy,ref,Op.SafeAt,Op.SafeAt.eq_def],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (RealQKVAlgebra.final_dq p l a scale swapped _
    (by simp [hc.dq]) (by simp) (by simp [hf.keys]) (by simp [hf.queryWeights])))]
  refine ⟨?_,?_⟩
  · cases swapped <;> simp [Stmt.TraceSafe,finalDQ,add,dot,weighted,ref,Op.SafeAt,Op.SafeAt.eq_def]
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (RealQKVAlgebra.final_dk p l a scale _
    (by simp [hc.dk]) (by simp [hf.query]) (by simp) (by simp [hf.keyWeights])))]
  refine ⟨by simp [Stmt.TraceSafe,finalDK,add,dot,weighted,ref,Op.SafeAt,Op.SafeAt.eq_def],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (key_eval p l r.dq _
    (by simp [hf.head]) (by simp [hf.key]) (by simp [hf.chunk])))]
  refine ⟨by simp [Stmt.TraceSafe,keyPtr,times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (key_eval p l r.dk _
    (by simp [hf.head]) (by simp [hf.key]) (by simp [hf.chunk])))]
  refine ⟨by simp [Stmt.TraceSafe,keyPtr,times,Op.SafeAt],?_⟩
  change Stmt.TraceSafeList bounds [store2 p.BT p.BK "p_dq" "b_dq",store2 p.BT p.BK "p_dk" "b_dk"]
    (RealQKVFinal.prepared p l r a scale u)
  rw [TraceProofs.cons_of_run_iff (store_eval (RealQKVFinal.keyWindow p l r.dq) (RealQKVTiles.dq p l a scale) "p_dq" "b_dq" _
    (by simp [RealQKVFinal.prepared]) (by simp [RealQKVFinal.prepared]; rfl))]
  refine ⟨?_,?_⟩
  · exact store2_safe r.dq (l.head*p.keyHead) p.T p.K p.BT p.BK p.keyTime p.keyFeature
      (l.chunk*p.BT) (l.key*p.BK) "p_dq" "b_dq" bounds _ hq (by simp [RealQKVFinal.prepared,keyPointer])
  apply Stmt.TraceSafeList.cons_intro
  · exact store2_safe r.dk (l.head*p.keyHead) p.T p.K p.BT p.BK p.keyTime p.keyFeature
      (l.chunk*p.BT) (l.key*p.BK) "p_dk" "b_dk" bounds _ hk (by simp [RealQKVFinal.prepared,keyPointer])
  · intro _ _; exact Stmt.TraceSafeList.nil_intro

theorem source_safe (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) (swapped : Bool)
    (u : BlockState) (hu : u.undef = fun _ _ => 0) (hs : RealQKVStep.Separate r)
    (hinj : RealQKVLoop.NoCollision p (launch u) r)
    (bounds : RegionBounds) (hb : Bounds p (launch u) r bounds) :
    (source p r storage scale swapped).eraseDType.toAlgKernel.TraceSafe bounds u := by
  let l := launch u
  let a := inputs r u
  let start := RealQKVInit.initial p r a scale u
  have hm : Reads r a u := ⟨fun _ => rfl,fun _ => rfl,fun _ => rfl,fun _ => rfl⟩
  have hi := RealQKVInit.invariant p r a scale u hm
  have hei := RealQKVInit.executes p r a scale u hu (fun _ => rfl) (fun _ => rfl)
  have hloop := loop_safe u p l r a scale hu hs hinj bounds hb.loop start hi
  obtain ⟨mid,hel,hmid⟩ := RealQKVLoop.executes u p l r a scale hu hs hinj start hi
  rw [Kernel.TraceSafe,source_body,body,List.append_assoc]
  apply TraceProofs.append_safe _ u (init_safe p r scale bounds u hb.query hb.key)
  intro v hv
  rw [hei] at hv
  cases hv
  change Stmt.TraceSafeList bounds (loop p r scale :: finalization p r scale swapped) start
  rw [TraceProofs.cons_of_run_iff hel]
  exact ⟨hloop,final_safe p l r a scale swapped bounds mid hmid.fixed hmid.carries hb.dq hb.dk⟩

theorem source_flatten_ok (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) (swapped : Bool) :
    (source p r storage scale swapped).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk,source_body]
  cases swapped <;> simp [StmtList.FlattenOk,Stmt.FlattenOk,body,initialization,loop,finalization,
    RealQKVProgram.iteration,queryPtr,keyPtr,valuePtr,historyPtr,historyGradientPtr,historyOffset,dvPtr,dvBase,
    score,pointwise,weighted,updateDS,updateDQ,updateDK,outputDV,finalDS,finalDQ,finalDK,add,dot,weighted,ref,scaleBy,store2,
    times,load2,RealHProgram.baseOp,RealHProgram.decayLane,RealDecayTiles.queryOp,
    RealDecayTiles.maskOp,RealDecayTiles.scoreOp,Op.FlattenOk]
  all_goals simp [Op.FlattenOk.eq_def]

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealQKVSafety

/-! Complete three-output execution in finite flat allocations. Bounds apply
to actual masked source accesses; the frame preserves every other tagged cell. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealQKVFlat
open VeriTile.Triton
open RealHMath (Layout)
open RealQKVMath (Launch Inputs numBlocks)
open RealQKVProgram
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false

def footprint (A : FlatAlloc) (p : Layout) (l : Launch) (r : Regions) : WriteFootprint := fun cell =>
  cell.1 = A.flat ∧ ∃ rg off, RealQKVExecution.footprint p l r (rg,off) ∧ A.addr rg off = cell.2

theorem flatten_frame (A : FlatAlloc) (p : Layout) (l : Launch) (r : Regions) (u w : BlockState)
    (hf : BlockState.WriteWithin (RealQKVExecution.footprint p l r) u w) :
    BlockState.WriteWithin (footprint A p l r) (A.flattenState u) (A.flattenState w) := by
  intro rg off hmiss
  by_cases hr : rg = A.flat
  · subst rg
    simp only [FlatAlloc.flattenState,if_pos rfl,FlatAlloc.readFlat]
    cases hd : A.decode off with
    | none => rfl
    | some cell =>
      obtain ⟨rg',off'⟩ := cell
      obtain ⟨_,heq,_⟩ := A.decode_sound hd
      apply congrArg A.trCell
      apply hf
      intro hhit
      exact hmiss ⟨rfl,rg',off',hhit,heq.symm⟩
  · simp only [FlatAlloc.flattenState,if_neg hr]

structure Outputs (A : FlatAlloc) (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (scale : ℝ) (w : BlockState) : Prop where
  query : ∀ (i : Fin p.BT) (e : Fin p.BK), RealQKVMath.keyActive p l i.val e.val →
    w.readMem A.flat (A.addr r.dq (RealQKVMath.keyAddress p l i.val e.val)) = RealQKVMath.dq p l a scale i.val e.val
  key : ∀ (i : Fin p.BT) (e : Fin p.BK), RealQKVMath.keyActive p l i.val e.val →
    w.readMem A.flat (A.addr r.dk (RealQKVMath.keyAddress p l i.val e.val)) = RealQKVMath.dk p l a scale i.val e.val
  value : ∀ (c : Nat) (i : Fin p.BT) (j : Fin p.BV), c < numBlocks p → RealQKVMath.valueActive p l c i.val j.val →
    w.readMem A.flat (A.addr r.dv (RealQKVMath.dvAddress p l c i.val j.val)) = RealQKVMath.dv p l a scale c i.val j.val

theorem source_executes (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) (swapped : Bool)
    (a : Inputs) (u : BlockState) (hu : u.undef = fun _ _ => 0) (hs : RealQKVExecution.Separate r)
    (hvi : RealQKVLoop.NoCollision p (launch u) r)
    (hki : (RealQKVFinal.keyWindow p (launch u) r.dq).NoCollision)
    (hq : ∀ off, u.readMem r.query off = a.query off)
    (hk : ∀ off, u.readMem r.key off = a.key off) (hm : RealQKVStep.Reads r a u)
    (hb : RealQKVSafety.Bounds p (launch u) r A.extent) :
    ∃ w, exec (A.flattenKernel (source p r storage scale swapped).eraseDType.toAlgKernel) (A.flattenState u) = some w ∧
      Outputs A p (launch u) r a scale w ∧
      BlockState.WriteWithin (footprint A p (launch u) r) (A.flattenState u) w := by
  obtain ⟨w,he,ho,hf⟩ := RealQKVExecution.executes p r storage scale swapped a u hu hs hvi hki hq hk hm
  have hflat := A.exec_flatten hd hcov _ u
    (RealQKVSafety.source_safe p r storage scale swapped u hu hs.loop hvi A.extent hb)
    (RealQKVSafety.source_flatten_ok p r storage scale swapped) hu
  rw [he,Option.map_some] at hflat
  have hmem (rg : RegionName) (off : Nat) (hbound : off < A.extent rg) :
      (A.flattenState w).readMem A.flat (A.addr rg off) = w.readMem rg off := by
    have hr : rg ∈ A.regions := by
      by_contra hn
      rw [hcov rg hn] at hbound
      exact Nat.not_lt_zero _ hbound
    exact A.flattenState_readMem hd w hr hbound
  refine ⟨A.flattenState w,hflat,⟨?_,?_,?_⟩,flatten_frame A p _ r u w hf⟩
  · intro i e ha
    have hoff : RealQKVMath.keyAddress p (launch u) i.val e.val < A.extent r.dq := hb.dq i e ha
    rw [hmem _ _ hoff]
    exact ho.query i e ha
  · intro i e ha
    have hoff : RealQKVMath.keyAddress p (launch u) i.val e.val < A.extent r.dk := hb.dk i e ha
    rw [hmem _ _ hoff]
    exact ho.key i e ha
  · intro c i j hc ha
    have hoff : RealQKVMath.dvAddress p (launch u) c i.val j.val < A.extent r.dv := (hb.loop c hc).output i j ha
    rw [hmem _ _ hoff]
    exact ho.value c i j hc ha

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealQKVFlat

/-! Three-output IO with visible windows and masks. Complete execution and
untouched tagged-cell preservation are part of the contract. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealQKVIO
open VeriTile.Triton
open RealHMath (Layout)
open RealQKVMath (Launch Inputs numBlocks)
open RealQKVProgram
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false

structure Values (p : Layout) where
  query : Fin p.BT → Fin p.BK → ℝ
  key : Fin p.BT → Fin p.BK → ℝ
  value : Nat → Fin p.BT → Fin p.BV → ℝ

structure IO (p : Layout) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  read : BlockState → Inputs
  valid : BlockState → Prop
  bounds : BlockState → RegionBounds → Prop
  dq : RegionName
  dk : RegionName
  dv : RegionName
  keyWindow : Launch → Fin p.BT → Fin p.BK → Nat
  valueWindow : Launch → Nat → Fin p.BT → Fin p.BV → Nat
  keyMask : Launch → Fin p.BT → Fin p.BK → Prop
  valueMask : Launch → Nat → Fin p.BT → Fin p.BV → Prop

def IO.regionWrites {p : Layout} (io : IO p) (l : Launch) : WriteFootprint := fun cell =>
  (cell.1 = io.dv ∧ ∃ n i j, n < numBlocks p ∧ io.valueMask l n i j ∧ cell.2 = io.valueWindow l n i j) ∨
  ((cell.1 = io.dq ∧ ∃ i e, io.keyMask l i e ∧ cell.2 = io.keyWindow l i e) ∨
   (cell.1 = io.dk ∧ ∃ i e, io.keyMask l i e ∧ cell.2 = io.keyWindow l i e))

def IO.writes {p : Layout} (io : IO p) (A : FlatAlloc) (l : Launch) : WriteFootprint := fun cell =>
  cell.1 = A.flat ∧ ∃ rg off, io.regionWrites l (rg,off) ∧ A.addr rg off = cell.2

/-- Explicit zero oracle for source-unspecified lanes, finite disjoint
allocations and bounds on every active source access. -/
def IO.Implements {p : Layout} (io : IO p) (f : Launch → Inputs → Values p) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ s : BlockState, s.undef = (fun _ _ => 0) → io.valid s → io.bounds s A.extent →
    ∃ w, exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s) = some w ∧
      (∀ i e, io.keyMask (launch s) i e →
        w.readMem A.flat (A.addr io.dq (io.keyWindow (launch s) i e)) = (f (launch s) (io.read s)).query i e) ∧
      (∀ i e, io.keyMask (launch s) i e →
        w.readMem A.flat (A.addr io.dk (io.keyWindow (launch s) i e)) = (f (launch s) (io.read s)).key i e) ∧
      (∀ n i j, n < numBlocks p → io.valueMask (launch s) n i j →
        w.readMem A.flat (A.addr io.dv (io.valueWindow (launch s) n i j)) = (f (launch s) (io.read s)).value n i j) ∧
      BlockState.WriteWithin (io.writes A (launch s)) (A.flattenState s) w

scoped notation:25 io " ⊨ " f => IO.Implements io f

theorem projection (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) (swapped : Bool) :
    (source p r storage scale swapped).eraseDType.toAlgorithm? =
      .ok (source p r storage scale swapped).eraseDType.toAlgKernel := by
  simp only [source,ComputeKernel.eraseDType,RealQKVSource.projects,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

def sourceIO (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) (swapped : Bool) : IO p where
  kernel := (source p r storage scale swapped).eraseDType
  projection := projection p r storage scale swapped
  read := inputs r
  valid := fun s => RealQKVLoop.NoCollision p (launch s) r ∧ (RealQKVFinal.keyWindow p (launch s) r.dq).NoCollision
  bounds := fun s => RealQKVSafety.Bounds p (launch s) r
  dq := r.dq
  dk := r.dk
  dv := r.dv
  keyWindow := fun l i e => RealQKVMath.keyAddress p l i.val e.val
  valueWindow := fun l n i j => RealQKVMath.dvAddress p l n i.val j.val
  keyMask := fun l i e => RealQKVMath.keyActive p l i.val e.val
  valueMask := fun l n i j => RealQKVMath.valueActive p l n i.val j.val

def expected (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) : Values p where
  query := fun i e => RealQKVMath.dq p l a scale i.val e.val
  key := fun i e => RealQKVMath.dk p l a scale i.val e.val
  value := fun n i j => RealQKVMath.dv p l a scale n i.val j.val

theorem implements (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) (swapped : Bool)
    (hs : RealQKVExecution.Separate r) : sourceIO p r storage scale swapped ⊨ expected p scale := by
  intro A hd hcov s hu hv hb
  obtain ⟨w,he,ho,hf⟩ := RealQKVFlat.source_executes A hd hcov p r storage scale swapped (inputs r s) s hu hs
    hv.1 hv.2 (fun _ => rfl) (fun _ => rfl) ⟨fun _ => rfl,fun _ => rfl,fun _ => rfl,fun _ => rfl⟩ hb
  refine ⟨w,he,ho.query,ho.key,ho.value,?_⟩
  intro region off hn
  apply hf region off
  intro hh
  apply hn
  rcases hh with ⟨hr,rg,k,hw,ha⟩
  refine ⟨hr,rg,k,?_,ha⟩
  rcases hw with ⟨n,hb,hwin⟩ | (hwin | hwin)
  · rcases hwin with ⟨hr,i,j,hm,hw⟩
    exact Or.inl ⟨hr,n,i,j,hb,hm,hw⟩
  · exact Or.inr (Or.inl hwin)
  · exact Or.inr (Or.inr hwin)

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealQKVIO

#guard_msgs (drop info) in
#auditModuleAxioms
