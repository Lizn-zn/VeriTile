import bench.tritonbench_g.chunk_retention.Proofs.BackwardStateExecution
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealOutputSafety
import bench.tritonbench_g.chunk_gla_fwd.Proofs.IO

namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealDhSafety
open VeriTile.Triton RealDhMath RealDhProgram RealDhStep RealDhFinal
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputSafety (MatrixBounds pointer load_safe store_safe)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure StepBounds (p : Layout) (l : Launch) (r : Regions) (c : Nat) (bounds : RegionBounds) : Prop where
  incoming : MatrixBounds r.incoming (l.head*p.valueHead) p.T p.V p.BT p.BT p.valueTime p.valueFeature
    (c*p.BT) (l.value*p.BT) bounds
  value : MatrixBounds r.value (l.head*p.valueHead) p.T p.V p.BT p.BT p.valueTime p.valueFeature
    (c*p.BT) (l.value*p.BT) bounds
  history : MatrixBounds r.history (l.head*p.historyHead+c*p.K*p.V) p.K p.V p.BT p.BT p.historyKey 1
    (l.key*p.BT) (l.value*p.BT) bounds
structure Bounds (p : Layout) (l : Launch) (r : Regions) (bounds : RegionBounds) : Prop where
  loop : ∀ c, c < p.NT → StepBounds p l r c bounds
  output : MatrixBounds r.history (l.head*p.historyHead+l.key*p.K*p.V) p.K p.V p.BT p.BT p.historyKey 1
    (l.value*p.BT) 0 bounds

theorem step_safe (p : Layout) (l : Launch) (r : Regions) (c : Nat) (swapped : Bool)
    (bounds : RegionBounds) (hb : StepBounds p l r c bounds) (s : BlockState)
    (hk : s.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hv : s.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hh : s.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : s.regs .nat [] "i_t" = some (Tile.scalar c)) :
    Stmt.TraceSafeList bounds (iterationBody p r swapped) s := by
  unfold iterationBody
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (value_eval p l r.incoming c s hh ht hv))]
  refine ⟨by simp [Stmt.TraceSafe,valuePtr,RealHProgram.times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (value_eval p l r.value c _ (by simp [hh]) (by simp [ht]) (by simp [hv])))]
  refine ⟨by simp [Stmt.TraceSafe,valuePtr,RealHProgram.times,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (history_eval p l r.history c _ (by simp [hh]) (by simp [ht]) (by simp [hk]) (by simp [hv])))]
  refine ⟨by simp [Stmt.TraceSafe,historyPtr,RealHProgram.times,Op.SafeAt],?_⟩
  apply TraceProofs.assign_cons (load_safe r.incoming (l.head*p.valueHead) p.T p.V p.BT p.BT p.valueTime p.valueFeature
    (c*p.BT) (l.value*p.BT) "p_o" bounds _ hb.incoming (by simp [pointer]))
  intro incoming
  apply TraceProofs.assign_cons (load_safe r.value (l.head*p.valueHead) p.T p.V p.BT p.BT p.valueTime p.valueFeature
    (c*p.BT) (l.value*p.BT) "p_v" bounds _ hb.value (by simp [pointer]))
  intro value
  apply TraceProofs.assign_cons (load_safe r.history (l.head*p.historyHead+c*p.K*p.V) p.K p.V p.BT p.BT p.historyKey 1
    (l.key*p.BT) (l.value*p.BT) "p_h" bounds _ hb.history (by simp [pointer]))
  intro history
  apply TraceProofs.all_safe
  cases swapped <;> simp [update,dot,RealDhProgram.carry,Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]

theorem final_safe (p : Layout) (l : Launch) (r : Regions)
    (bounds : RegionBounds) (hb : Bounds p l r bounds) (s : BlockState)
    (hk : s.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hv : s.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hh : s.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : s.regs .nat [] "i_t" = some (Tile.scalar 0)) :
    Stmt.TraceSafeList bounds (finalization p r) s := by
  unfold finalization
  apply TraceProofs.assign_cons (by simp [scaledCarry,RealDhProgram.carry,Op.SafeAt])
  intro output
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (output_eval p l r _ (by simp [hk]) (by simp [hv]) (by simp [hh]) (by simp [ht])))]
  refine ⟨by simp [Stmt.TraceSafe,outputPtr,RealHProgram.times,Op.SafeAt],?_⟩
  apply Stmt.TraceSafeList.cons_intro
    (store_safe r.history (l.head*p.historyHead+l.key*p.K*p.V) p.K p.V p.BT p.BT p.historyKey 1
      (l.value*p.BT) 0 "p_dh" "b_dh" bounds _ hb.output (by simp [outputPointer,pointer]))
  intro _ _
  exact Stmt.TraceSafeList.nil_intro

theorem loop_safe (p : Layout) (r : Regions) (s0 : BlockState) (swapped : Bool)
    (hu : s0.undef = fun _ _ => 0) (bounds : RegionBounds) (hb : Bounds p (launch s0) r bounds)
    (s : BlockState) (hi : RealDhLoop.Holds p r s0 0 s) :
    Stmt.TraceSafe bounds (RealDhLoop.statement p r swapped) s := by
  rw [RealDhLoop.statement,Stmt.TraceSafe]
  apply TraceProofs.range_safe (P := RealDhLoop.Holds p r s0)
  · intro n u hn h
    rw [iteration,TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (RealDhLoop.reverseIndex_eval p n _ (by simp)))]
    refine ⟨by simp [Stmt.TraceSafe,reverseIndex,Op.SafeAt],?_⟩
    apply step_safe p (launch s0) r (p.NT-1-n) swapped bounds (hb.loop _ (by omega))
    · simpa [launch] using h.key
    · simpa [launch] using h.value
    · simpa [launch] using h.head
    · simp
  · intro n u t hn h he
    obtain ⟨t',he',ht⟩ := RealDhLoop.step_executes p r s0 swapped hu n u hn h
    rw [he] at he'
    cases he'
    exact ht
  · exact hi

theorem source_safe (p : Layout) (r : Regions) (dotStorage : Storage) (out : FloatDType) (scale : ℝ)
    (swapped : Bool) (s : BlockState) (hu : s.undef = fun _ _ => 0) (bounds : RegionBounds)
    (hb : Bounds p (launch s) r bounds) :
    (source p r dotStorage out scale swapped).eraseDType.toAlgKernel.TraceSafe bounds s := by
  rw [Kernel.TraceSafe,source_body,body]
  apply TraceProofs.append_safe _ s
  · apply TraceProofs.all_safe
    simp [initialization,RealHProgram.baseOp,RealDecayTiles.queryOp,Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]
  · intro u he
    rw [RealDhLoop.initialization_executes] at he
    cases he
    obtain ⟨v,hl,hv⟩ := RealDhLoop.executes p r s swapped hu (RealDhLoop.start p s) (RealDhLoop.start_holds p r s)
    unfold RealDhLoop.statement at hl
    rw [TraceProofs.cons_of_run_iff hl]
    refine ⟨loop_safe p r s swapped hu bounds hb (RealDhLoop.start p s) (RealDhLoop.start_holds p r s),?_⟩
    exact final_safe p (launch s) r bounds hb v hv.key hv.value hv.head (by simpa using hv.time)

theorem source_flatten_ok (p : Layout) (r : Regions) (dotStorage : Storage) (out : FloatDType) (scale : ℝ)
    (swapped : Bool) : (source p r dotStorage out scale swapped).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk,source_body]
  cases swapped <;> simp [body,initialization,iteration,iterationBody,reverseIndex,valuePtr,historyPtr,
    RealHProgram.times,RealHProgram.load2,RealHProgram.baseOp,RealDecayTiles.queryOp,
    update,dot,RealDhProgram.carry,finalization,scaledCarry,outputPtr,store,
    StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk]
  all_goals simp [Op.FlattenOk.eq_def]

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealDhSafety

namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealDhFlat
open VeriTile.Triton RealDhMath RealDhProgram RealDhStep RealDhFinal
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def footprint (alloc : FlatAlloc) (p : Layout) (l : Launch) (r : Regions) : WriteFootprint :=
  fun cell => cell.1 = alloc.flat ∧ ∃ i : TileIndex [p.BT,p.BT],
    active p l i.1.val i.2.1.val ∧ cell.2 = alloc.addr r.history (outputAddress p l i.1.val i.2.1.val)

theorem flatten_frame (alloc : FlatAlloc) (p : Layout) (l : Launch) (r : Regions) (s t : BlockState)
    (hf : BlockState.WriteWithin (RealDhFinal.footprint p l r) s t) :
    BlockState.WriteWithin (footprint alloc p l r) (alloc.flattenState s) (alloc.flattenState t) := by
  intro region off hn
  by_cases hr : region = alloc.flat
  · subst region
    simp only [FlatAlloc.flattenState,if_pos rfl,FlatAlloc.readFlat]
    cases hd : alloc.decode off with
    | none => rfl
    | some cell =>
      rcases cell with ⟨rg,k⟩
      obtain ⟨_,heq,_⟩ := alloc.decode_sound hd
      apply congrArg alloc.trCell
      apply hf
      rintro ⟨hr,i,hi,haddr⟩
      apply hn
      refine ⟨rfl,i,hi,?_⟩
      rw [← hr,← haddr,← heq]
  · simp only [FlatAlloc.flattenState,if_neg hr]

theorem source_executes (p : Layout) (r : Regions) (dot : Storage) (out : FloatDType) (scale : ℝ)
    (swapped : Bool) (alloc : FlatAlloc) (hd : alloc.Disjoint)
    (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0) (s : BlockState)
    (hu : s.undef = fun _ _ => 0) (hinj : NoCollision p (launch s))
    (hb : RealDhSafety.Bounds p (launch s) r alloc.extent) :
    ∃ sf, exec (alloc.flattenKernel (source p r dot out scale swapped).eraseDType.toAlgKernel)
        (alloc.flattenState s) = some sf ∧
      (∀ i : TileIndex [p.BT,p.BT], active p (launch s) i.1.val i.2.1.val →
        sf.readMem alloc.flat (alloc.addr r.history (outputAddress p (launch s) i.1.val i.2.1.val)) =
          expected p (launch s) (inputs r s) i.1.val i.2.1.val) ∧
      BlockState.WriteWithin (footprint alloc p (launch s) r) (alloc.flattenState s) sf := by
  obtain ⟨t,he,ho,hf⟩ := RealDhExecution.region_executes p r dot out scale swapped s hu hinj
  have hflat := alloc.exec_flatten hd hcov _ s
    (RealDhSafety.source_safe p r dot out scale swapped s hu alloc.extent hb)
    (RealDhSafety.source_flatten_ok p r dot out scale swapped) hu
  rw [he,Option.map_some] at hflat
  refine ⟨alloc.flattenState t,hflat,?_,flatten_frame alloc p (launch s) r s t hf⟩
  intro i hi
  have hoff : outputAddress p (launch s) i.1.val i.2.1.val < alloc.extent r.history := by
    simpa only [outputAddress,Nat.zero_add,Nat.mul_one] using hb.output i.1 i.2.1 (by simpa only [Nat.zero_add] using hi)
  have hr : r.history ∈ alloc.regions := by
    by_contra hn
    rw [hcov r.history hn] at hoff
    exact Nat.not_lt_zero _ hoff
  rw [alloc.flattenState_readMem hd t hr hoff]
  exact ho i hi

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealDhFlat

namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealDhIO
open VeriTile.Triton RealDhMath RealDhProgram RealDhStep RealDhFinal VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIO
open scoped VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIO
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000

theorem projection (p : Layout) (r : Regions) (dot : Storage) (out : FloatDType) (scale : ℝ)
    (swapped : Bool) :
    (source p r dot out scale swapped).eraseDType.toAlgorithm? =
      .ok (source p r dot out scale swapped).eraseDType.toAlgKernel := by
  simp only [source,ComputeKernel.eraseDType,RealDhSource.projects,
    ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody]

def sourceIO (p : Layout) (r : Regions) (dot : Storage) (out : FloatDType) (scale : ℝ) (swapped : Bool) :
    IO Launch Inputs (TileIndex [p.BT,p.BT]) where
  kernel := (source p r dot out scale swapped).eraseDType
  projection := projection p r dot out scale swapped
  grid := launch
  read := inputs r
  valid := fun s => 0 < p.NT ∧ NoCollision p (launch s)
  bounds := fun s => RealDhSafety.Bounds p (launch s) r
  out := r.history
  write := fun l i => outputAddress p l i.1.val i.2.1.val
  mask := fun l i => active p l i.1.val i.2.1.val

theorem implements (p : Layout) (r : Regions) (dot : Storage) (out : FloatDType) (scale : ℝ) (swapped : Bool) :
    sourceIO p r dot out scale swapped ⊨
      (fun l a i => expected p l a i.1.val i.2.1.val) := by
  intro alloc hd hcov s hu hi hb
  exact RealDhFlat.source_executes p r dot out scale swapped alloc hd hcov s hu hi.2 hb

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealDhIO

#guard_msgs (drop info) in
#auditModuleAxioms
