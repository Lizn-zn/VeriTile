import bench.tritonbench_g.chunk_linear_attn.Proofs.ForwardOutputExecution
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealOutputSafety
import bench.tritonbench_g.chunk_gla_fwd.Proofs.IO

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.OutputSafe
open VeriTile.Triton OutputMath RealOutput
open ChunkGlaSimple.Real (mulConst_eval makeBlockPtr_2d_eval)
open ChunkGlaFwd.RealOutputSafety (MatrixBounds pointer load_safe store_safe)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure StepBounds (p : Layout) (l : Launch) (r : Regions) (kb : Nat) (bounds : RegionBounds) : Prop where
  query : MatrixBounds r.query (l.head*p.keyHead) p.T p.K p.BT p.BK p.keyTime p.keyDimension (l.chunk*p.BT) (kb*p.BK) bounds
  key : MatrixBounds r.key (l.head*p.keyHead) p.K p.T p.BK p.BT p.keyDimension p.keyTime (kb*p.BK) (l.chunk*p.BT) bounds
  history : MatrixBounds r.history (l.head*p.historyHead+l.chunk*p.K*p.V) p.K p.V p.BK p.BV p.historyKey 1
    (kb*p.BK) (l.value*p.BV) bounds

structure TailBounds (p : Layout) (l : Launch) (r : Regions) (bounds : RegionBounds) : Prop where
  value : MatrixBounds r.value (l.head*p.valueHead) p.T p.V p.BT p.BV p.valueTime p.valueDimension (l.chunk*p.BT) (l.value*p.BV) bounds
  output : MatrixBounds r.output (l.head*p.valueHead) p.T p.V p.BT p.BV p.valueTime p.valueDimension (l.chunk*p.BT) (l.value*p.BV) bounds

structure Bounds (p : Layout) (l : Launch) (r : Regions) (bounds : RegionBounds) : Prop where
  loop : ∀ kb, kb < numBlocks p → StepBounds p l r kb bounds
  tail : TailBounds p l r bounds

theorem step_safe (p : Layout) (l : Launch) (r : Regions) (kb : Nat)
    (bounds : RegionBounds) (hb : StepBounds p l r kb bounds) (sl : BlockState)
    (hiv : sl.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hit : sl.regs .nat [] "i_t" = some (Tile.scalar l.chunk))
    (hibh : sl.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hik : sl.regs .nat [] "i_k" = some (Tile.scalar kb)) :
    Stmt.TraceSafeList bounds (iteration p r) sl := by
  have hbaseH : ∀ st : BlockState, st.regs .nat [] "i_bh" = some (Tile.scalar l.head) →
      st.regs .nat [] "i_t" = some (Tile.scalar l.chunk) →
      evalOp (Op.add .nat .nil
        (Op.mul .nat .nil (Op.ref .nat [] "i_bh") (Op.constNat p.historyHead))
        (Op.mul .nat .nil (Op.mul .nat .nil (Op.ref .nat [] "i_t") (Op.constNat p.K)) (Op.constNat p.V))) st
      = some (Tile.scalar (l.head*p.historyHead+l.chunk*p.K*p.V)) := by
    intro st hbh ht
    rw [evalOp_add,mulConst_eval st "i_bh" _ _ hbh,evalOp_mul,mulConst_eval st "i_t" _ _ ht]
    simp only [evalOp_constNat,Option.bind_eq_bind,Option.bind_some]
    rfl
  unfold iteration linearOutputLoopBody
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.query sl _ _ _ [p.T,p.K] [p.BT,p.BK] [p.keyTime,p.keyDimension] (l.head*p.keyHead)
      (l.chunk*p.BT) (kb*p.BK) (mulConst_eval sl "i_bh" _ _ hibh)
      (mulConst_eval sl "i_t" _ _ hit) (mulConst_eval sl "i_k" _ _ hik)))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.key _ _ _ _ [p.K,p.T] [p.BK,p.BT] [p.keyDimension,p.keyTime] (l.head*p.keyHead)
      (kb*p.BK) (l.chunk*p.BT)
      (mulConst_eval _ "i_bh" _ _ (by simp [hibh])) (mulConst_eval _ "i_k" _ _ (by simp [hik]))
      (mulConst_eval _ "i_t" _ _ (by simp [hit]))))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.history _ _ _ _ [p.K,p.V] [p.BK,p.BV] [p.historyKey,1]
      (l.head*p.historyHead+l.chunk*p.K*p.V) (kb*p.BK) (l.value*p.BV)
      (hbaseH _ (by simp [hibh]) (by simp [hit])) (mulConst_eval _ "i_k" _ _ (by simp [hik]))
      (mulConst_eval _ "i_v" _ _ (by simp [hiv]))))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  apply TraceProofs.assign_cons (load_safe r.query (l.head*p.keyHead) p.T p.K p.BT p.BK p.keyTime p.keyDimension
    (l.chunk*p.BT) (kb*p.BK) "p_q" bounds _ hb.query (by simp [pointer]))
  intro query
  apply TraceProofs.assign_cons (load_safe r.key (l.head*p.keyHead) p.K p.T p.BK p.BT p.keyDimension p.keyTime
    (kb*p.BK) (l.chunk*p.BT) "p_k" bounds _ hb.key (by simp [pointer]))
  intro key
  apply TraceProofs.assign_cons (load_safe r.history (l.head*p.historyHead+l.chunk*p.K*p.V) p.K p.V p.BK p.BV p.historyKey 1
    (kb*p.BK) (l.value*p.BV) "p_h" bounds _ hb.history (by simp [pointer]))
  intro history
  apply TraceProofs.all_safe
  simp [Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.OutputSafe

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.OutputSafe
open VeriTile.Triton OutputMath RealOutput
open ChunkGlaSimple.Real (mulConst_eval makeBlockPtr_2d_eval)
open ChunkGlaFwd.RealOutputSafety (MatrixBounds pointer load_safe store_safe)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem post_safe (p : Layout) (l : Launch) (r : Regions) (scale : ℝ) (swapped : Bool)
    (bounds : RegionBounds) (hb : TailBounds p l r bounds) (s : BlockState)
    (hiv : s.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hit : s.regs .nat [] "i_t" = some (Tile.scalar l.chunk))
    (hibh : s.regs .nat [] "i_bh" = some (Tile.scalar l.head)) :
    Stmt.TraceSafeList bounds (post p r scale swapped) s := by
  unfold post tail after linearOutputFront
  simp only [List.drop_succ_cons,List.drop_zero,List.take_succ_cons,List.take_zero,
    List.cons_append,List.nil_append]
  apply TraceProofs.assign_cons (by simp [Op.SafeAt,Op.SafeAt.eq_def])
  intro masked
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.value _ _ _ _ [p.T,p.V] [p.BT,p.BV] [p.valueTime,p.valueDimension]
      (l.head*p.valueHead) (l.chunk*p.BT) (l.value*p.BV)
      (mulConst_eval _ "i_bh" _ _ (by simp [hibh])) (mulConst_eval _ "i_t" _ _ (by simp [hit]))
      (mulConst_eval _ "i_v" _ _ (by simp [hiv]))))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval r.output _ _ _ _ [p.T,p.V] [p.BT,p.BV] [p.valueTime,p.valueDimension]
      (l.head*p.valueHead) (l.chunk*p.BT) (l.value*p.BV)
      (mulConst_eval _ "i_bh" _ _ (by simp [hibh])) (mulConst_eval _ "i_t" _ _ (by simp [hit]))
      (mulConst_eval _ "i_v" _ _ (by simp [hiv]))))]
  refine ⟨by simp [Stmt.TraceSafe,Op.SafeAt],?_⟩
  apply TraceProofs.assign_cons (load_safe r.value (l.head*p.valueHead) p.T p.V p.BT p.BV p.valueTime p.valueDimension
    (l.chunk*p.BT) (l.value*p.BV) "p_v" bounds _ hb.value (by simp [pointer]))
  intro value
  apply TraceProofs.assign_cons (by cases swapped <;> simp [totalOp,Op.SafeAt,Op.SafeAt.eq_def])
  intro output
  apply Stmt.TraceSafeList.cons_intro
    (store_safe r.output (l.head*p.valueHead) p.T p.V p.BT p.BV p.valueTime p.valueDimension
      (l.chunk*p.BT) (l.value*p.BV) "p_o" "b_o" bounds _ hb.output (by simp [pointer]))
  intro _ _
  exact Stmt.TraceSafeList.nil_intro

theorem loop_safe (p : Layout) (r : Regions) (s0 : BlockState) (hu : s0.undef = fun _ _ => 0)
    (bounds : RegionBounds) (hb : ∀ kb, kb < numBlocks p → StepBounds p (launch s0) r kb bounds)
    (u : BlockState) (hi : Holds p r s0 0 u) : Stmt.TraceSafe bounds (loop p r) u := by
  rw [loop,Stmt.TraceSafe]
  simp only [tripCount,Op.SafeAt,evalOp,Option.bind,Tile.scalar,Tile.bop_data,
    NumericDType.add,NumericDType.sub,NumericDType.div,Broadcast.leftIndex,Broadcast.rightIndex,true_and]
  apply TraceProofs.range_safe (P := Holds p r s0)
  · intro i s hn hs
    apply step_safe p (launch s0) r i bounds (hb i hn)
    · simpa using hs.valueReg
    · simpa using hs.timeReg
    · simpa using hs.headReg
    · simp
  · intro i s t hn hs he
    obtain ⟨t',he',ht⟩ := loop_step p r s0 hu i s hn hs
    rw [he] at he'
    cases he'
    exact ht
  · exact hi

theorem source_safe (p : Layout) (r : Regions) (dot value : Storage) (out : FloatDType) (scale : ℝ)
    (swapped : Bool) (s : BlockState) (hu : s.undef = fun _ _ => 0) (bounds : RegionBounds)
    (hb : Bounds p (launch s) r bounds) :
    (program p r dot value out scale swapped).eraseDType.toAlgKernel.TraceSafe bounds s := by
  rw [Kernel.TraceSafe,program_body]
  apply TraceProofs.append_safe _ s
  · apply TraceProofs.all_safe
    simp [setup,Stmt.TraceSafe,Op.SafeAt,Op.SafeAt.eq_def]
  · intro u he
    rw [setup_executes] at he
    cases he
    obtain ⟨v,hl,hv⟩ := loop_executes p r s hu (start p s) (start_holds p r s)
    rw [TraceProofs.cons_of_run_iff hl]
    refine ⟨loop_safe p r s hu bounds hb.loop (start p s) (start_holds p r s),?_⟩
    exact post_safe p (launch s) r scale swapped bounds hb.tail v hv.valueReg hv.timeReg hv.headReg

theorem source_flatten_ok (p : Layout) (r : Regions) (dot value : Storage) (out : FloatDType) (scale : ℝ)
    (swapped : Bool) : (program p r dot value out scale swapped).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk,program_body]
  cases swapped <;> simp [setup,loop,tripCount,iteration,linearOutputLoopBody,post,tail,after,linearOutputFront,
    linearOutputStoreStmt,totalOp,StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk]
  all_goals simp [Op.FlattenOk.eq_def]

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.OutputSafe

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.OutputFlat
open VeriTile.Triton OutputMath RealOutput
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def footprint (alloc : FlatAlloc) (p : Layout) (l : Launch) (r : Regions) : WriteFootprint :=
  fun cell => cell.1 = alloc.flat ∧ ∃ i : TileIndex [p.BT,p.BV],
    active p l i.1.val i.2.1.val ∧ cell.2 = alloc.addr r.output (outputAddress p l i.1.val i.2.1.val)

theorem flatten_frame (alloc : FlatAlloc) (p : Layout) (l : Launch) (r : Regions) (s t : BlockState)
    (hf : BlockState.WriteWithin (RealOutput.footprint p l r) s t) :
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

theorem source_executes (p : Layout) (r : Regions) (dot value : Storage) (out : FloatDType) (scale : ℝ)
    (swapped : Bool) (alloc : FlatAlloc) (hd : alloc.Disjoint)
    (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0) (s : BlockState)
    (hu : s.undef = fun _ _ => 0) (hinj : NoCollision p (launch s))
    (hb : OutputSafe.Bounds p (launch s) r alloc.extent) :
    ∃ sf, exec (alloc.flattenKernel (program p r dot value out scale swapped).eraseDType.toAlgKernel)
        (alloc.flattenState s) = some sf ∧
      (∀ i : TileIndex [p.BT,p.BV], active p (launch s) i.1.val i.2.1.val →
        sf.readMem alloc.flat (alloc.addr r.output (outputAddress p (launch s) i.1.val i.2.1.val)) =
          expected p (launch s) (inputs r s) scale i.1.val i.2.1.val) ∧
      BlockState.WriteWithin (footprint alloc p (launch s) r) (alloc.flattenState s) sf := by
  obtain ⟨t,he,ho,hf⟩ := program_executes p r dot value out scale swapped s hu hinj
  have hflat := alloc.exec_flatten hd hcov _ s
    (OutputSafe.source_safe p r dot value out scale swapped s hu alloc.extent hb)
    (OutputSafe.source_flatten_ok p r dot value out scale swapped) hu
  rw [he,Option.map_some] at hflat
  refine ⟨alloc.flattenState t,hflat,?_,flatten_frame alloc p (launch s) r s t hf⟩
  intro i hi
  have hoff : outputAddress p (launch s) i.1.val i.2.1.val < alloc.extent r.output := by
    simpa only [outputAddress,row,column,Nat.mul_one] using hb.tail.output i.1 i.2.1 hi
  have hr : r.output ∈ alloc.regions := by
    by_contra hn
    rw [hcov r.output hn] at hoff
    exact Nat.not_lt_zero _ hoff
  rw [alloc.flattenState_readMem hd t hr hoff]
  exact ho i hi

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.OutputFlat

namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.OutputIO
open VeriTile.Triton OutputMath RealOutput ChunkGlaFwd.RealIO
open scoped ChunkGlaFwd.RealIO
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000

theorem projection (p : Layout) (r : Regions) (dot value : Storage) (out : FloatDType) (scale : ℝ)
    (swapped : Bool) :
    (program p r dot value out scale swapped).eraseDType.toAlgorithm? =
      .ok (program p r dot value out scale swapped).eraseDType.toAlgKernel := by
  cases swapped
  all_goals simp only [program,Bool.false_eq_true,↓reduceIte,kernel,candidate,forwardO,forwardOScaleFirst,
    ComputeKernel.eraseDType,OutputSource.projects,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

def sourceIO (p : Layout) (r : Regions) (dot value : Storage) (out : FloatDType) (scale : ℝ) (swapped : Bool) :
    IO Launch Inputs (TileIndex [p.BT,p.BV]) where
  kernel := (program p r dot value out scale swapped).eraseDType
  projection := projection p r dot value out scale swapped
  grid := launch
  read := inputs r
  valid := fun s => NoCollision p (launch s)
  bounds := fun s => OutputSafe.Bounds p (launch s) r
  out := r.output
  write := fun l i => outputAddress p l i.1.val i.2.1.val
  mask := fun l i => active p l i.1.val i.2.1.val

theorem implements (p : Layout) (r : Regions) (dot value : Storage) (out : FloatDType) (scale : ℝ) (swapped : Bool) :
    sourceIO p r dot value out scale swapped ⊨
      (fun l a i => expected p l a scale i.1.val i.2.1.val) := by
  intro alloc hd hcov s hu hi hb
  exact OutputFlat.source_executes p r dot value out scale swapped alloc hd hcov s hu hi hb

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.OutputIO

#guard_msgs (drop info) in
#auditModuleAxioms
