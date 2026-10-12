import bench.tritonbench_g.context_attn_nopad.Proofs.Frames
import bench.tritonbench_g.context_attn_nopad.Proofs.Meaning
import bench.tritonbench_g.context_attn_nopad.Proofs.NumericUpdate
namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.IterationExecution
open VeriTile.Triton Formula Program
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.Frames (then_frame)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

theorem executes (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (initial u : BlockState) (c : Nat) (hBN : 0 < p.BN)
    (hs : Invariant.Static p r initial u)
    (hshift : 0 ≤ (inputs initial r).start (launch initial).batch+(c*p.BN : Nat))
    (mi li : Tile .real [p.BM]) (acc : Tile .real [p.BM,p.D])
    (hmi : u.regs .real [p.BM] "m_i" = some mi)
    (hli : u.regs .real [p.BM] "l_i" = some li)
    (hac : u.regs .real [p.BM,p.D] "acc" = some acc)
    (hi : u.regs .int [] "start_n" = some (Tile.scalar ((c*p.BN : Nat) : Int))) :
    ∃ next maximum,
      stepStmts (LoopProgram.body p r swapped scale) u = some next ∧
      Tile.reduceMaxDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) (LoopState.score p r initial scale c) = some maximum ∧
      Invariant.Static p r initial next ∧
      let updated := NumericUpdate.update p mi li acc
        (LoopState.score p r initial scale c) (LoopState.value p r initial c) maximum
      next.regs .real [p.BM] "m_i" = some updated.1 ∧
      next.regs .real [p.BM] "l_i" = some updated.2.1 ∧
      next.regs .real [p.BM,p.D] "acc" = some updated.2.2 := by
  let l := launch initial
  let a := inputs initial r
  let sn : Int := c*p.BN
  let rows := LoopState.rows p initial
  let start := a.start l.batch
  let last := Formula.sequence p l a
  let group := l.head
  let qt := LoopState.query p r initial
  let kt := LoopState.key p r initial c
  let vt := LoopState.value p r initial c
  let qkt := LoopState.score p r initial scale c
  let s1 := KeysExecution.prepared p r group start sn last u
  have e1 := KeysExecution.executes p r group start sn last u hs.key_ptrs hi hs.start hs.sequence hs.lanes
  have f1 := Frames.keys p r group start sn last u
  have hs1 : Invariant.Static p r initial s1 := by
    apply Frames.preserved p r initial u s1 _ f1 hs
    intro name hn
    simp [Frames.readNames] at hn
    rcases hn with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals decide
  have hk1 : s1.regs .real [p.D,p.BN] "k" = some kt := by
    simp only [s1,KeysExecution.prepared,BlockState.setReg_same]
    exact congrArg some (Meaning.key_matches p r u initial c hs.memory hshift)
  have hn1 : s1.regs .int [] "start_n" = some (Tile.scalar sn) := by simp [s1,KeysExecution.prepared]
  let s2 := ScoresExecution.prepared p swapped scale rows sn qt kt s1
  have e2 := ScoresExecution.executes p swapped scale rows sn qt kt s1
    hs1.rows hs1.lanes hn1 hs1.query hk1
  have f2 := Frames.scores p swapped scale rows sn qt kt s1
  have f12 := then_frame f1 f2
  have hqk2 : s2.regs .real [p.BM,p.BN] "qk" = some qkt := by
    simp only [s2,ScoresExecution.prepared,BlockState.setReg_same]
    exact congrArg some (Meaning.scores_match p r swapped scale initial c)
  obtain ⟨maximum,hrm⟩ := StatsExecution.reduction_exists p qkt hBN
  let center := maximum
  let prob := StatsExecution.exponentials p qkt center
  let lij : Tile .real [p.BM] := StatsExecution.denominator p qkt center
  let s3 := StatsExecution.prepared p qkt maximum s2
  have e3 := StatsExecution.executes p qkt maximum s2 hqk2 hrm
  have f3 := Frames.stats p qkt maximum s2
  have f123 := then_frame f12 f3
  have hmi3 : s3.regs .real [p.BM] "m_i" = some mi :=
    (f123.registers .real [p.BM] "m_i" (by decide)).trans hmi
  have hli3 : s3.regs .real [p.BM] "l_i" = some li :=
    (f123.registers .real [p.BM] "l_i" (by decide)).trans hli
  have hmax3 : s3.regs .real [p.BM] "m_ij" = some center := by simp [s3,StatsExecution.prepared,center]
  have hlij3 : s3.regs .real [p.BM] "l_ij" = some lij := by simp [s3,StatsExecution.prepared,lij,center]
  have hprob3 : s3.regs .real [p.BM,p.BN] "p" = some prob := by simp [s3,StatsExecution.prepared,prob,center]
  let mn := NormalizerExecution.newMaximum p mi center
  let al := NormalizerExecution.alpha p mi center
  let be := NormalizerExecution.beta p mi center
  let ln := NormalizerExecution.newDenominator p mi center li lij
  let s4 := NormalizerExecution.prepared p mi center li lij s3
  have e4 := NormalizerExecution.executes p mi center li lij s3 hmi3 hmax3 hli3 hlij3
  have f4 := Frames.normalizer p mi center li lij s3
  have f1234 := then_frame f123 f4
  have hli4 : s4.regs .real [p.BM] "l_i" = some li :=
    (f4.registers .real [p.BM] "l_i" (by decide)).trans hli3
  have hln4 : s4.regs .real [p.BM] "l_i_new" = some ln := by simp [s4,NormalizerExecution.prepared,ln]
  have hal4 : s4.regs .real [p.BM] "alpha" = some al := by simp [s4,NormalizerExecution.prepared,al]
  have hbe4 : s4.regs .real [p.BM] "beta" = some be := by simp [s4,NormalizerExecution.prepared,be]
  have hmn4 : s4.regs .real [p.BM] "m_i_new" = some mn := by simp [s4,NormalizerExecution.prepared,mn]
  have hprob4 : s4.regs .real [p.BM,p.BN] "p" = some prob :=
    (f4.registers .real [p.BM,p.BN] "p" (by decide)).trans hprob3
  have hac4 : s4.regs .real [p.BM,p.D] "acc" = some acc :=
    (f1234.registers .real [p.BM,p.D] "acc" (by decide)).trans hac
  let prob2 := ScalesExecution.probabilities p prob be ln
  let acc1 := ScalesExecution.accumulator p li ln al acc
  let s5 := ScalesExecution.prepared p li ln al be prob acc s4
  have e5 := ScalesExecution.executes p li ln al be prob acc s4
    hli4 hln4 hal4 hbe4 hprob4 hac4
  have f5 := Frames.scales p li ln al be prob acc s4
  have f12345 := then_frame f1234 f5
  have hn5 : s5.regs .int [] "start_n" = some (Tile.scalar sn) :=
    ((then_frame (then_frame (then_frame f2 f3) f4) f5).registers .int [] "start_n" (by decide)).trans hn1
  let s6 := ValuesExecution.prepared p r group start sn last s5
  have e6 := ValuesExecution.executes p r group start sn last s5
    ((f12345.registers .ptr [p.BN,p.D] "v_ptrs" (by decide)).trans hs.value_ptrs) hn5
    ((f12345.registers .int [] "cur_batch_in_all_start_index" (by decide)).trans hs.start)
    ((f12345.registers .int [] "cur_batch_seq_len" (by decide)).trans hs.sequence)
    ((f12345.registers .nat [p.BN] "offs_n" (by decide)).trans hs.lanes)
  have f6 := Frames.values p r group start sn last s5
  have f123456 := then_frame f12345 f6
  have hv6 : s6.regs .real [p.BN,p.D] "v" = some vt := by
    simp only [s6,ValuesExecution.prepared,BlockState.setReg_same]
    exact congrArg some (Meaning.value_matches p r s5 initial c (f12345.memory.trans hs.memory) hshift)
  have hp5 : s5.regs .real [p.BM,p.BN] "p" = some prob2 := by simp [s5,ScalesExecution.prepared,prob2]
  have ha5 : s5.regs .real [p.BM,p.D] "acc" = some acc1 := by simp [s5,ScalesExecution.prepared,acc1]
  have f56 := then_frame f5 f6
  let s7 := CommitExecution.prepared p prob2 vt acc1 ln mn s6
  have e7 := CommitExecution.executes p prob2 vt acc1 ln mn s6
    ((f6.registers .real [p.BM,p.BN] "p" (by decide)).trans hp5) hv6
    ((f6.registers .real [p.BM,p.D] "acc" (by decide)).trans ha5)
    ((f56.registers .real [p.BM] "l_i_new" (by decide)).trans hln4)
    ((f56.registers .real [p.BM] "m_i_new" (by decide)).trans hmn4)
  have f7 := Frames.commit p prob2 vt acc1 ln mn s6
  have fAll := then_frame f123456 f7
  refine ⟨s7,maximum,?_,hrm,?_,?_⟩
  · rw [LoopProgram.source]
    simp only [List.append_assoc]
    rw [stepStmts.append_some e1]
    change stepStmts _ s1 = _
    rw [stepStmts.append_some e2]
    change stepStmts _ s2 = _
    rw [stepStmts.append_some e3]
    change stepStmts _ s3 = _
    rw [stepStmts.append_some e4]
    change stepStmts _ s4 = _
    rw [stepStmts.append_some e5]
    change stepStmts _ s5 = _
    rw [stepStmts.append_some e6]
    exact e7
  · apply Frames.preserved p r initial u s7 _ fAll hs
    intro name hn
    simp [Frames.readNames] at hn
    rcases hn with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals decide
  · exact CommitExecution.registers p prob2 vt acc1 ln mn s6

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.IterationExecution
