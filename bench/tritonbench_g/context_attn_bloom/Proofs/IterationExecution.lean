import bench.tritonbench_g.context_attn_bloom.Proofs.StaticFrame
import bench.tritonbench_g.context_attn_bloom.Proofs.LoopState
import bench.tritonbench_g.context_attn_bloom.Proofs.KeysMeaning
import bench.tritonbench_g.context_attn_bloom.Proofs.ScoresMeaning
import bench.tritonbench_g.context_attn_bloom.Proofs.ValuesMeaning
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.IterationExecution
open VeriTile.Triton Math Program RealSetup RegisterFrame
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

theorem executes (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ)
    (initial u : BlockState) (c : Nat) (hBN : 0 < p.BN)
    (hs : Invariant.Static p r initial u) (hd : Invariant.Dynamic p r initial scale c u)
    (hi : u.regs .int [] "start_n" = some (Tile.scalar ((c*p.BN : Nat) : Int))) :
    ∃ next maximum,
      stepStmts (LoopProgram.body p r swapped scale) u = some next ∧
      Tile.reduceMaxDrop (⟨1,by simp⟩ : Fin [p.BM,p.BN].length) (LoopState.score p r initial scale c) = some maximum ∧
      Invariant.Static p r initial next ∧
      let updated := NumericUpdate.update p (LoopState.size p r initial) c (LoopState.pairs p r initial scale)
        (LoopState.position p r initial) (LoopState.rows p initial) ((inputs initial r).prompt (initial.pids 0))
        (LoopState.score p r initial scale c) (LoopState.value p r initial c) maximum
      next.regs .real [p.BM] "m_i" = some updated.1 ∧
      next.regs .real [p.BM] "l_i" = some updated.2.1 ∧
      next.regs .real [p.BM,p.D] "acc" = some updated.2.2 := by
  let l := launch initial
  let a := inputs initial r
  let sn : Int := c*p.BN
  let rows := LoopState.rows p initial
  let prompt := a.prompt l.batch
  let req := a.request l.batch
  let last := Math.limit p l a
  let group := l.head/p.kvGroup
  let qt := LoopState.query p r initial
  let kt := LoopState.key p r initial c
  let vt := LoopState.value p r initial c
  let qkt := LoopState.score p r initial scale c
  let mi := Invariant.maxTile p r initial scale c
  let li := Invariant.denominatorTile p r initial scale c
  let acc := Invariant.accumulatorTile p r initial scale c
  let pages : Fin p.BN → Int := fun j => LoopKeysExecution.page p r sn req last u j.val
  let s1 := LoopKeysExecution.prepared p r sn req last group u
  have e1 := LoopKeysExecution.executes p r sn req last group u hi hs.request hs.ending hs.kv_head hs.lanes hs.channels
  have f1 := StageFrames.keys p r sn req last group u
  have hs1 := KeysMeaning.static_preserved p r u initial sn req last group hs
  have hk1 : s1.regs .real [p.D,p.BN] "k" = some kt := KeysMeaning.register_matches p r u initial c hs.memory
  have hn1 : s1.regs .int [] "start_n" = some (Tile.scalar sn) := by simp [s1,LoopKeysExecution.prepared]
  have hg1 : s1.regs .int [p.BN] "kv_loc" = some (Tile.vec pages) := by simp [s1,LoopKeysExecution.prepared,pages]
  let s2 := ScoreTiles.prepared p swapped scale rows prompt sn qt kt s1
  have e2 := ScoresStaged.executes p r swapped scale rows prompt sn qt kt s1
    hs1.rows hs1.lanes hs1.prompt hn1 hs1.query hk1
  have f2 := StageFrames.scores p swapped scale rows prompt sn qt kt s1
  have f12 := then_frame f1 f2
  have hqk2 : s2.regs .real [p.BM,p.BN] "qk" = some qkt := by
    exact ScoresMeaning.register_matches p r swapped scale initial s1 c qt kt (by intros; rfl) (by intros; rfl)
  obtain ⟨maximum,hrm⟩ := StatsExecution.reduction_exists p qkt hBN
  let prob := StatsExecution.exponentials p qkt maximum
  let lij : Tile .real [p.BM] := StatsExecution.denominator p qkt maximum
  let s3 := StatsExecution.prepared p qkt maximum s2
  have e3 := StatsExecution.executes p r swapped scale qkt maximum s2 hqk2 hrm
  have f3 := StageFrames.stats p qkt maximum s2
  have f123 := then_frame f12 f3
  have hmi3 : s3.regs .real [p.BM] "m_i" = some mi :=
    (f123.registers .real [p.BM] "m_i" (by decide)).trans hd.maximum
  have hli3 : s3.regs .real [p.BM] "l_i" = some li :=
    (f123.registers .real [p.BM] "l_i" (by decide)).trans hd.denominator
  have hmax3 : s3.regs .real [p.BM] "m_ij" = some maximum := by simp [s3,StatsExecution.prepared]
  have hlij3 : s3.regs .real [p.BM] "l_ij" = some lij := by simp [s3,StatsExecution.prepared,lij]
  have hprob3 : s3.regs .real [p.BM,p.BN] "p" = some prob := by simp [s3,StatsExecution.prepared,prob]
  let mn := NormalizerExecution.newMaximum p mi maximum
  let al := NormalizerExecution.alpha p mi maximum
  let be := NormalizerExecution.beta p mi maximum
  let ln := NormalizerExecution.newDenominator p mi maximum li lij
  let s4 := NormalizerExecution.prepared p mi maximum li lij s3
  have e4 := NormalizerExecution.executes p r swapped scale mi maximum li lij s3 hmi3 hmax3 hli3 hlij3
  have f4 := StageFrames.normalizer p mi maximum li lij s3
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
    (f1234.registers .real [p.BM,p.D] "acc" (by decide)).trans hd.accumulator
  have hrows4 : s4.regs .nat [p.BM] "offs_m" = some (Tile.vec rows) :=
    (f1234.registers .nat [p.BM] "offs_m" (by decide)).trans hs.rows
  have hprompt4 : s4.regs .int [] "prompt_cache_len" = some (Tile.scalar prompt) :=
    (f1234.registers .int [] "prompt_cache_len" (by decide)).trans hs.prompt
  have f234 := then_frame (then_frame f2 f3) f4
  have hn4 : s4.regs .int [] "start_n" = some (Tile.scalar sn) :=
    (f234.registers .int [] "start_n" (by decide)).trans hn1
  let prob2 := ScalesExecution.probabilities p prob be ln
  let acc1 := ScalesExecution.accumulator p rows prompt sn li ln al acc
  let s5 := ScalesExecution.prepared p rows prompt sn li ln al be prob acc s4
  have e5 := ScalesExecution.executes p r swapped scale rows prompt sn li ln al be prob acc s4
    hrows4 hprompt4 hn4 hli4 hln4 hal4 hbe4 hprob4 hac4
  have f5 := StageFrames.scales p rows prompt sn li ln al be prob acc s4
  have f12345 := then_frame f1234 f5
  have f2345 := then_frame f234 f5
  have hn5 : s5.regs .int [] "start_n" = some (Tile.scalar sn) :=
    (f5.registers .int [] "start_n" (by decide)).trans hn4
  have hg5 : s5.regs .int [p.BN] "kv_loc" = some (Tile.vec pages) :=
    (f2345.registers .int [p.BN] "kv_loc" (by decide)).trans hg1
  let s6 := ValuesExecution.prepared p r sn last group pages s5
  have e6 := ValuesExecution.executes p r sn last group pages s5 hn5
    ((f12345.registers .int [] "block_end_loc" (by decide)).trans hs.ending)
    ((f12345.registers .nat [] "cur_kv_head" (by decide)).trans hs.kv_head)
    ((f12345.registers .nat [p.BN] "offs_n" (by decide)).trans hs.lanes)
    ((f12345.registers .nat [p.D] "offs_d" (by decide)).trans hs.channels) hg5
  have f6 := StageFrames.values p r sn last group pages s5
  have f123456 := then_frame f12345 f6
  have hv6 : s6.regs .real [p.BN,p.D] "v" = some vt :=
    ValuesMeaning.register_matches p r s5 u initial c (f12345.memory.trans hs.memory) hs.memory
  have hp5 : s5.regs .real [p.BM,p.BN] "p" = some prob2 := by simp [s5,ScalesExecution.prepared,prob2]
  have ha5 : s5.regs .real [p.BM,p.D] "acc" = some acc1 := by simp [s5,ScalesExecution.prepared,acc1]
  have f56 := then_frame f5 f6
  let s7 := CommitExecution.prepared p prob2 vt acc1 ln mn s6
  have e7 := CommitExecution.executes p r swapped scale prob2 vt acc1 ln mn s6
    ((f6.registers .real [p.BM,p.BN] "p" (by decide)).trans hp5) hv6
    ((f6.registers .real [p.BM,p.D] "acc" (by decide)).trans ha5)
    ((f56.registers .real [p.BM] "l_i_new" (by decide)).trans hln4)
    ((f56.registers .real [p.BM] "m_i_new" (by decide)).trans hmn4)
  have f7 := StageFrames.commit p prob2 vt acc1 ln mn s6
  have fAll := then_frame f123456 f7
  refine ⟨s7,maximum,?_,hrm,?_,?_⟩
  · rw [LoopProgram.split]
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
  · apply StaticFrame.preserved p r initial u s7 _ fAll hs
    intro name hn
    simp [StaticFrame.readNames] at hn
    rcases hn with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals decide
  · exact CommitExecution.registers p prob2 vt acc1 ln mn s6

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.IterationExecution
