import bench.tritonbench_g.chunk_retention.Proofs.BackwardStateProgram

namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealDhLoop
open VeriTile.Triton RealDhMath RealDhProgram RealDhStep
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

structure Holds (p : Layout) (r : Regions) (s0 : BlockState) (n : Nat) (s : BlockState) : Prop where
  limit : n ≤ p.NT
  memory : s.mem = s0.mem
  padding : s.undef = s0.undef
  pids : s.pids = s0.pids
  key : s.regs .nat [] "i_k" = some (Tile.scalar (s0.pids 0))
  value : s.regs .nat [] "i_v" = some (Tile.scalar (s0.pids 1))
  head : s.regs .nat [] "i_bh" = some (Tile.scalar (s0.pids 2))
  time : s.regs .nat [] "i_t" = some (Tile.scalar (if n = 0 then 0 else p.NT-n))
  queryDecay : s.regs .real [p.BT] "d_i" = some (queryDecay p (launch s0))
  chunkDecay : s.regs .real [] "d_b" = some (Tile.scalar (Decay.factor (base p (launch s0)) p.BT))
  carry : s.regs .real [p.BT,p.BT] "b_dh" = some (carryTile p (launch s0) (inputs r s0) n)

theorem reverseIndex_eval (p : Layout) (n : Nat) (s : BlockState)
    (hj : s.regs .nat [] "j" = some (Tile.scalar n)) :
    evalOp (reverseIndex p) s = some (Tile.scalar (p.NT-1-n)) := by
  simp only [reverseIndex,evalOp_sub,evalOp_constNat,evalOp_ref,hj,
    Option.bind_eq_bind,Option.bind_some]
  rfl

theorem step_executes (p : Layout) (r : Regions) (s0 : BlockState) (swapped : Bool)
    (hu : s0.undef = fun _ _ => 0) (n : Nat) (s : BlockState)
    (hn : n < p.NT) (hi : Holds p r s0 n s) :
    ∃ t, stepStmts (iteration p r swapped) (s.setReg "j" .nat [] (Tile.scalar n)) = some t ∧
      Holds p r s0 (n+1) t := by
  let u := (s.setReg "j" .nat [] (Tile.scalar n)).setReg "i_t" .nat [] (Tile.scalar (p.NT-1-n))
  have ha : inputs r u = inputs r s0 := by
    change inputs r s = inputs r s0
    unfold inputs
    congr 1 <;> funext off <;> simp only [u,BlockState.setReg_readMem,BlockState.readMem,hi.memory]
  obtain ⟨t,he,h⟩ := RealDhStep.executes p (launch s0) r n swapped u
    (hi.padding.trans hu) (by simpa [u,launch] using hi.key)
    (by simpa [u,launch] using hi.value) (by simpa [u,launch] using hi.head) (by simp [u])
    (by simpa [u] using hi.queryDecay) (by simpa [u,ha] using hi.carry)
  refine ⟨t,?_,Nat.succ_le_of_lt hn,h.memory.trans hi.memory,h.padding.trans hi.padding,
    h.pids.trans hi.pids,?_,?_,?_,?_,?_,?_,?_⟩
  · rw [iteration,stepStmts.cons_some (stepStmt_assign_eq_some (reverseIndex_eval p n _ (by simp)))]
    exact he
  · simpa [u,hi.key] using h.key
  · simpa [u,hi.value] using h.value
  · simpa [u,hi.head] using h.head
  · simpa [u,show p.NT-1-n = p.NT-(n+1) by omega] using h.time
  · simpa [u,hi.queryDecay] using h.queryDecay
  · simpa [u,hi.chunkDecay] using h.chunkDecay
  · simpa only [ha] using h.carry

def statement (p : Layout) (r : Regions) (swapped : Bool) : Stmt :=
  .forRange "j" 0 p.NT 1 (iteration p r swapped)

theorem executes (p : Layout) (r : Regions) (s0 : BlockState) (swapped : Bool)
    (hu : s0.undef = fun _ _ => 0) (s : BlockState) (hi : Holds p r s0 0 s) :
    ∃ t, stepStmt (statement p r swapped) s = some t ∧ Holds p r s0 p.NT t := by
  obtain ⟨n,t,he,hn,ht⟩ := forRange_inv (idx := "j") (start := 0) (stop := p.NT) (step := 1)
    (P := Holds p r s0) (body := iteration p r swapped) (by decide : 1 ≠ 0) hi
    (step_executes p r s0 swapped hu)
  have heq : n = p.NT := Nat.le_antisymm ht.limit hn
  subst n
  exact ⟨t,he,ht⟩

def start (p : Layout) (s : BlockState) : BlockState :=
  let u := s.setReg "i_k" .nat [] (Tile.scalar (s.pids 0))
  let u := u.setReg "i_v" .nat [] (Tile.scalar (s.pids 1))
  let u := u.setReg "i_bh" .nat [] (Tile.scalar (s.pids 2))
  let u := u.setReg "i_h" .nat [] (Tile.scalar (s.pids 2 % p.H))
  let u := u.setReg "b_b" .real [] (Tile.scalar (base p (launch s)))
  let u := u.setReg "o_i" .int [p.BT] (RealDecayExecution.indices p.BT)
  let u := u.setReg "d_b" .real [] (Tile.scalar (Decay.factor (base p (launch s)) p.BT))
  let u := u.setReg "d_i" .real [p.BT] (queryDecay p (launch s))
  let u := u.setReg "b_dh" .real [p.BT,p.BT] (⟨fun _ => some 0⟩ : Tile .real [p.BT,p.BT])
  u.setReg "i_t" .nat [] (Tile.scalar 0)

theorem initialization_executes (p : Layout) (s : BlockState) :
    stepStmts (initialization p) s = some (start p s) := by
  unfold initialization
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s)),
    stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _)),
    stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _)),
    stepStmts.cons_some (stepStmt_assign_eq_some (RealDecayState.head_index_eval p.H (s.pids 2) _ (by simp))),
    stepStmts.cons_some (stepStmt_assign_eq_some (RealDecayExecution.base_eval p.H (s.pids 2) _ (by simp))),
    stepStmts.cons_some (stepStmt_assign_eq_some (RealDecayState.indices_eval p.BT _)),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (RealDecayState.factor_eval p.BT (base p (launch s)) _ (by simp [base,launch]))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (RealDecayTiles.query_eval p.BT (base p (launch s)) _ (by simp) (by simp [base,launch]))),
    stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BT,p.BT] (.const 0)) _ =
      some (⟨fun _ => some 0⟩ : Tile .real [p.BT,p.BT]) from by simp [evalOp])),
    stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_constNat 0 _)),stepStmts.nil]
  rfl

theorem start_holds (p : Layout) (r : Regions) (s : BlockState) : Holds p r s 0 (start p s) := by
  constructor
  · exact Nat.zero_le _
  · rfl
  · rfl
  · rfl
  all_goals simp [start,carryTile,RealDhMath.carry_zero,Tile.mat]

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealDhLoop

namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealDhFinal
open VeriTile.Triton RealDhMath RealDhProgram RealDhStep
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real (makeBlockPtr_2d_eval mulConst_eval)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def resultTile (p : Layout) (l : Launch) (a : Inputs) : Tile .real [p.BT,p.BT] :=
  Tile.mat (fun i j => some (expected p l a i.val j.val))
def outputPointer (p : Layout) (l : Launch) (r : Regions) : Tile .blockPtr [p.BT,p.BT] :=
  ⟨fun _ => BlockPtr.mk r.history (l.head*p.historyHead+l.key*p.K*p.V)
    [p.K,p.V] [p.BT,p.BT] [p.historyKey,1] [l.value*p.BT,0]⟩

theorem scaled_eval (p : Layout) (l : Launch) (a : Inputs) (s : BlockState)
    (hc : s.regs .real [p.BT,p.BT] "b_dh" = some (carryTile p l a p.NT))
    (hd : s.regs .real [] "d_b" = some (Tile.scalar (Decay.factor (base p l) p.BT))) :
    evalOp (scaledCarry p) s = some (resultTile p l a) := by
  simp only [scaledCarry,RealDhProgram.carry,evalOp_mul,evalOp_ref,hc,hd,
    Option.bind_eq_bind,Option.bind_some]
  rfl

theorem output_eval (p : Layout) (l : Launch) (r : Regions) (s : BlockState)
    (hk : s.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hv : s.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hh : s.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : s.regs .nat [] "i_t" = some (Tile.scalar 0)) :
    evalOp (outputPtr p r.history) s = some (outputPointer p l r) := by
  have hb : evalOp (.add .nat .nil (RealHProgram.times "i_bh" p.historyHead)
      (.mul .nat .nil (RealHProgram.times "i_k" p.K) (.constNat p.V))) s =
      some (Tile.scalar (l.head*p.historyHead+l.key*p.K*p.V)) := by
    unfold RealHProgram.times
    rw [evalOp_add,mulConst_eval s "i_bh" _ _ hh,evalOp_mul,mulConst_eval s "i_k" _ _ hk]
    simp only [evalOp_constNat,Option.bind_eq_bind,Option.bind_some]
    rfl
  simpa only [Nat.zero_mul] using makeBlockPtr_2d_eval r.history s _ _ _
    [p.K,p.V] [p.BT,p.BT] [p.historyKey,1] _ (l.value*p.BT) (0*p.BT)
    hb (mulConst_eval s "i_v" _ _ hv) (mulConst_eval s "i_t" _ _ ht)

def scatter (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (s : BlockState) : BlockState :=
  (TileShape.allIndices [p.BT,p.BT]).foldl (fun u i =>
    if active p l i.1.val i.2.1.val then
      u.writeMem r.history (outputAddress p l i.1.val i.2.1.val) (expected p l a i.1.val i.2.1.val) else u) s

def NoCollision (p : Layout) (l : Launch) : Prop :=
  ∀ i j : TileIndex [p.BT,p.BT], active p l i.1.val i.2.1.val → active p l j.1.val j.2.1.val →
    outputAddress p l i.1.val i.2.1.val = outputAddress p l j.1.val j.2.1.val → i = j

def footprint (p : Layout) (l : Launch) (r : Regions) : WriteFootprint :=
  fun cell => cell.1 = r.history ∧ ∃ i : TileIndex [p.BT,p.BT],
    active p l i.1.val i.2.1.val ∧ cell.2 = outputAddress p l i.1.val i.2.1.val

theorem store_executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (s : BlockState)
    (hp : s.regs .blockPtr [p.BT,p.BT] "p_dh" = some (outputPointer p l r))
    (ho : s.regs .real [p.BT,p.BT] "b_dh" = some (resultTile p l a)) :
    stepStmt (store p) s = some (scatter p l r a s) := by
  simp only [store,RealDhProgram.carry,stepStmt,evalOp_ref,hp,ho,Option.bind_eq_bind,Option.bind_some,
    outputPointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets,
    BlockPtr.address_2d_offsets,Bool.true_and,decide_eq_true_eq,Nat.mul_one,Nat.zero_add]
  apply congrArg some
  unfold scatter
  congr 1

theorem scatter_readback (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (s : BlockState)
    (hinj : NoCollision p l) (i : TileIndex [p.BT,p.BT]) (hi : active p l i.1.val i.2.1.val) :
    (scatter p l r a s).readMem r.history (outputAddress p l i.1.val i.2.1.val) =
      expected p l a i.1.val i.2.1.val := by
  exact BlockState.scatter_readback_prop_masked_nd_of_true s _ _ _ i hi
    (fun j hj he => hinj j i hj hi he)

theorem scatter_frame (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (s : BlockState) :
    BlockState.WriteWithin (footprint p l r) s (scatter p l r a s) := by
  apply BlockState.foldl_writeMem_prop_writeWithin
  intro i _ hi
  exact ⟨rfl,i,hi,rfl⟩

theorem executes (p : Layout) (r : Regions) (s0 s : BlockState)
    (h : RealDhLoop.Holds p r s0 p.NT s) (hinj : NoCollision p (launch s0)) :
    ∃ t, stepStmts (finalization p r) s = some t ∧
      (∀ i : TileIndex [p.BT,p.BT], active p (launch s0) i.1.val i.2.1.val →
        t.readMem r.history (outputAddress p (launch s0) i.1.val i.2.1.val) =
          expected p (launch s0) (inputs r s0) i.1.val i.2.1.val) ∧
      BlockState.WriteWithin (footprint p (launch s0) r) s0 t := by
  have ht : s.regs .nat [] "i_t" = some (Tile.scalar 0) := by simpa using h.time
  let u := (s.setReg "b_dh" .real [p.BT,p.BT] (resultTile p (launch s0) (inputs r s0))).setReg
    "p_dh" .blockPtr [p.BT,p.BT] (outputPointer p (launch s0) r)
  have he : stepStmts (finalization p r) s = some (scatter p (launch s0) r (inputs r s0) u) := by
    unfold finalization
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (scaled_eval p (launch s0) (inputs r s0) s h.carry h.chunkDecay)),
      stepStmts.cons_some (stepStmt_assign_eq_some (output_eval p (launch s0) r _
        (by simp [h.key,launch]) (by simp [h.value,launch]) (by simp [h.head,launch]) (by simp [ht]))),
      stepStmts.cons_some (store_executes p (launch s0) r (inputs r s0) u (by simp [u]) (by simp [u])),
      stepStmts.nil]
  refine ⟨_,he,fun i hi => scatter_readback p (launch s0) r (inputs r s0) u hinj i hi,?_⟩
  intro rg off hn
  simpa only [u,BlockState.setReg_mem,h.memory] using scatter_frame p (launch s0) r (inputs r s0) u rg off hn

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealDhFinal

namespace VeriTile.Bench.TritonBenchG.ChunkRetention.RealDhExecution
open VeriTile.Triton RealDhMath RealDhProgram RealDhStep RealDhFinal
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

theorem region_executes (p : Layout) (r : Regions) (dot : Storage) (out : FloatDType) (scale : ℝ)
    (swapped : Bool) (s : BlockState) (hu : s.undef = fun _ _ => 0) (hinj : NoCollision p (launch s)) :
    ∃ t, exec (source p r dot out scale swapped).eraseDType.toAlgKernel s = some t ∧
      (∀ i : TileIndex [p.BT,p.BT], active p (launch s) i.1.val i.2.1.val →
        t.readMem r.history (outputAddress p (launch s) i.1.val i.2.1.val) =
          expected p (launch s) (inputs r s) i.1.val i.2.1.val) ∧
      BlockState.WriteWithin (footprint p (launch s) r) s t := by
  obtain ⟨u,hl,hi⟩ := RealDhLoop.executes p r s swapped hu (RealDhLoop.start p s) (RealDhLoop.start_holds p r s)
  obtain ⟨t,hf,ho,hframe⟩ := RealDhFinal.executes p r s u hi hinj
  unfold RealDhLoop.statement at hl
  refine ⟨t,?_,ho,hframe⟩
  rw [exec,source_body,body,stepStmts.append_some (RealDhLoop.initialization_executes p s),
    stepStmts.cons_some hl,hf]

end
end VeriTile.Bench.TritonBenchG.ChunkRetention.RealDhExecution

#guard_msgs (drop info) in
#auditModuleAxioms
