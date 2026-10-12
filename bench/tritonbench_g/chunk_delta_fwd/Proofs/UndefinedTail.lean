import bench.tritonbench_g.chunk_delta_fwd.Proofs.UndefinedTailStep
import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealExecution
import VeriTile.Meta.Disprove

/-! Removing the zero-unspecified-value premise is false for the original
full real source. T=1, all actual inputs are zero, and K/V time-tail lanes
return one. The final valid state cell is 15 instead of the formula's zero.
This is a Lean real-semantics counterexample, not an observed GPU result. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.UndefinedTail
open VeriTile.Triton RealProgram UndefinedTailStep
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def kernel := (RealExecution.source p r .fp32 .fp32 .fp32 .fp32 Bool.false Bool.true).eraseDType.toAlgKernel

theorem accumulation_value : nextAccum wide l a 0 0 (fun _ _ => 0) (fun _ _ => 0) = fun _ _ => 15 := by
  funext e j
  simp only [nextAccum,zero_add]
  change (∑ c : Fin 16, keyValue wide l a (timeBase wide 0 0 + c.val) e.val *
    correctedValue wide l a (fun _ _ => 0) (timeBase wide 0 0 + c.val) j) = 15
  have ht (c : Fin 16) : keyValue wide l a (timeBase wide 0 0 + c.val) e.val *
      correctedValue wide l a (fun _ _ => 0) (timeBase wide 0 0 + c.val) j =
      if c.val = 0 then 0 else 1 := by
    have he : e.val < 16 := e.isLt
    have hj : j.val < 16 := j.isLt
    have hk : e.val+c.val*16 < 16 ↔ c.val = 0 := by omega
    have hv : c.val*16+j.val < 16 ↔ c.val = 0 := by omega
    simp [correctedValue,keyValue,inputValue,keyIndex,keyAddress,activeValue,valueIndex,
      valueAddress,timeBase,p,wide,l,a,sample,he,hj,c.isLt,hk,hv]
  simp_rw [ht]
  norm_num [Fin.sum_univ_succ]

private theorem initialization :
    stepStmts (cdfPrologue r.initial p.K p.V p.BK p.BV Bool.false) initial =
      some (RealInit.registers p initial) := by
  unfold cdfPrologue
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 initial))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (.full [p.BK,p.BV] (.const 0)) _ =
    some (stateTile p (fun _ _ => 0)) from by simp [evalOp_full,evalOp_const]; rfl))]
  rw [stepStmts.cons_some (cdf_ifThen_false_noop _ _),stepStmts.nil]
  rfl

theorem source_stores_fifteen : ∃ v, exec kernel initial = some v ∧ v.readMem r.final 0 = 15 := by
  let start := RealInit.registers p initial
  let entry := start.setReg "i_t" .nat [] (Tile.scalar 0)
  let pre := RealOuterStep.prepared p l r 0 (fun _ _ => 0) entry
  have hik : entry.regs .nat [] "i_k" = some (Tile.scalar l.key) := by simp [entry,start,RealInit.registers,initial,l]
  have hiv : entry.regs .nat [] "i_v" = some (Tile.scalar l.value) := by simp [entry,start,RealInit.registers,initial,l]
  have hih : entry.regs .nat [] "i_bh" = some (Tile.scalar l.head) := by simp [entry,start,RealInit.registers,initial,l]
  have hit : entry.regs .nat [] "i_t" = some (Tile.scalar 0) := by simp [entry]
  have hh : entry.regs .real [p.BK,p.BV] "b_h" = some (stateTile p (fun _ _ => 0)) := by
    simp [entry,start,RealInit.registers]
  have hepre := RealOuterStep.prefix_executes p l r 0 (fun _ _ => 0) entry hik hiv hih hit hh
  have hi := RealOuterStep.prepared_invariant p l r zero 0 (fun _ _ => 0) entry
    (by constructor <;> decide) hik hiv hih hit
    (by intro off; simp [entry,start,RealInit.registers,initial,zero,BlockState.readMem])
    (by intro off; simp [entry,start,RealInit.registers,initial,zero,BlockState.readMem])
    (by intro off; simp [entry,start,RealInit.registers,initial,zero,BlockState.readMem]) hh
  let innerEntry := pre.setReg "i_c" .nat [] (Tile.scalar 0)
  have heinner : stepStmts (innerBody p r) innerEntry = some (UndefinedTailStep.result innerEntry) := by
    apply UndefinedTailStep.executes
    · simp only [innerEntry,pre,RealOuterStep.prepared,RealStateStore.scatter,cdfWrites_undef,
        entry,start,RealInit.registers,BlockState.setReg]
    · simpa [innerEntry] using hi.key
    · simpa [innerEntry] using hi.value
    · simpa [innerEntry] using hi.head
    · simpa [innerEntry] using hi.time
    · simp [innerEntry]
    · intro off; simpa [innerEntry,zero] using hi.inputKey off
    · intro off; simpa [innerEntry,zero] using hi.inputDelta off
    · intro off; simpa [innerEntry,zero] using hi.inputValue off
    · simpa [innerEntry,wide,p] using hi.state
    · simp [innerEntry,pre,RealOuterStep.prepared,wide,p,stateTile]
  let innerOut := UndefinedTailStep.result innerEntry
  have hel : stepStmt (RealInnerLoop.statement p r) pre = some innerOut := by
    simp only [RealInnerLoop.statement,stepForRangeAux.forRangeDyn_unfold,
      RealInnerLoop.stop_eval,evalOp_constNat,bind,Option.bind_some,Tile.scalar]
    change stepForRangeAux "i_c" 0 1 1 (innerBody p r) pre = _
    change stepStmts (innerBody p r) (pre.setReg "i_c" .nat [] (Tile.scalar 0)) = _ at heinner
    rw [stepForRangeAux]
    simp only [Nat.reduceLT,↓reduceIte,heinner,bind,Option.bind_some]
    rw [stepForRangeAux]
    rfl
  have carry := RealOuterStep.carry_eval p (fun _ _ => 0) (fun _ _ => 15) innerOut
    (by simpa [innerOut,UndefinedTailStep.result,scatter,UndefinedTailStep.beforeStore,innerEntry,wide,p] using hi.state)
    (by
      have ha : innerOut.regs .real [16,16] "b_h_cumsum" = some
          (stateTile wide (nextAccum wide l a 0 0 (fun _ _ => 0) (fun _ _ => 0))) := by
        simp [innerOut,UndefinedTailStep.result]
        rfl
      rw [accumulation_value] at ha
      exact ha)
  let outerOut := innerOut.setReg "b_h" .real [p.BK,p.BV] (stateTile p (fun _ _ => 15))
  have heb : stepStmts (RealOuterStep.body p r) entry = some outerOut := by
    rw [RealOuterStep.body_parts,stepStmts.append_some hepre,stepStmts.cons_some hel]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some carry),stepStmts.nil]
    simp only [outerOut,zero_add]
  have heloop : stepStmt (RealOuterLoop.statement p r) start = some outerOut := by
    simp only [RealOuterLoop.statement,stepStmt]
    change stepForRangeAux "i_t" 0 1 1 (RealOuterStep.body p r) start = _
    change stepStmts (RealOuterStep.body p r) (start.setReg "i_t" .nat [] (Tile.scalar 0)) = _ at heb
    rw [stepForRangeAux]
    simp only [Nat.reduceLT,↓reduceIte,heb,bind,Option.bind_some]
    rw [stepForRangeAux]
    rfl
  have hik' : outerOut.regs .nat [] "i_k" = some (Tile.scalar l.key) := by
    simpa [outerOut,innerOut,UndefinedTailStep.result,scatter,UndefinedTailStep.beforeStore,innerEntry] using hi.key
  have hiv' : outerOut.regs .nat [] "i_v" = some (Tile.scalar l.value) := by
    simpa [outerOut,innerOut,UndefinedTailStep.result,scatter,UndefinedTailStep.beforeStore,innerEntry] using hi.value
  have hih' : outerOut.regs .nat [] "i_bh" = some (Tile.scalar l.head) := by
    simpa [outerOut,innerOut,UndefinedTailStep.result,scatter,UndefinedTailStep.beforeStore,innerEntry] using hi.head
  have hefinal := RealFinal.executes p l r Bool.true (fun _ _ => 15) outerOut hik' hiv' hih' (by simp [outerOut])
  refine ⟨RealFinal.result p l r Bool.true (fun _ _ => 15) outerOut,?_,?_⟩
  · rw [kernel,exec,RealExecution.body_parts,List.append_assoc,stepStmts.append_some initialization]
    change stepStmts (RealOuterLoop.statement p r :: cdfEpilogue r.final p.K p.V p.BK p.BV Bool.true) start = _
    rw [stepStmts.cons_some heloop]
    exact hefinal
  · apply RealFinal.readback p l r (fun _ _ => 15) outerOut ?_ ⟨0,by decide⟩ ⟨0,by decide⟩ (by decide)
    intro e f i j _ _ he
    dsimp [RealStateStore.address,RealInit.base,keyIndex,valueIndex,p,l] at he
    have h : e.val = f.val ∧ i.val = j.val := by omega
    exact ⟨Fin.ext h.1,Fin.ext h.2⟩

refutation final_state_without_zero_padding : ∀ v,
    exec kernel initial = some v → v.readMem r.final 0 =
      RealWindow.expected p l zero (fun _ _ => 0) Bool.false p.NT 0 0 := by
  obtain ⟨v,he,hv⟩ := source_stores_fifteen
  have hzero : RealWindow.expected p l zero (fun _ _ => 0) Bool.false p.NT 0 0 = 0 := by
    norm_num [RealWindow.expected,RealWindow.state,RealWindow.span,RealInnerLoop.count,
      RealMath.corrected,RealValues.arrays,zero,p,l]
  intro h
  have bad := h v he
  rw [hv,hzero] at bad
  norm_num at bad

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.UndefinedTail
