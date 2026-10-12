import bench.tritonbench_g.context_attn_llama.Proofs.Meaning
namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama.IterationExecution
open VeriTile.Triton Formula Program Frames
noncomputable section
set_option maxHeartbeats 2800000
set_option maxRecDepth 7000
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
        (LoopState.score p r initial scale c) (LoopState.value p r initial c) maximum
      next.regs .real [p.BM] "m_i" = some updated.1 ∧
      next.regs .real [p.BM] "l_i" = some updated.2.1 ∧
      next.regs .real [p.BM,p.D] "acc" = some updated.2.2 := by
  let l := launch initial
  let a := inputs initial r
  let sn : Int := c*p.BN
  let rows := LoopState.rows p initial
  let prompt := a.prompt (Formula.batch p l)
  let last := Formula.limit p l a
  let batch := Formula.batch p l
  let request := a.request batch
  let pages := LoopState.pages p r initial c
  let group := Formula.head p l/p.kvGroup
  let qt := LoopState.query p r initial
  let kt := LoopState.key p r initial c
  let vt := LoopState.value p r initial c
  let qkt := LoopState.score p r initial scale c
  let mi := Invariant.maxTile p r initial scale c
  let li := Invariant.denominatorTile p r initial scale c
  let acc := Invariant.accumulatorTile p r initial scale c
  let s1 := KeysExecution.prepared p r group request sn last u
  have e1 := KeysExecution.executes p r group request sn last u hs.request hs.kv_head hi hs.ending hs.lanes hs.channels
  have f1 := Frames.keys p r group request sn last u
  have hs1 : Invariant.Static p r initial s1 := by
    apply Frames.preserved p r initial u s1 _ f1 hs
    intro name hn
    simp [readNames] at hn
    rcases hn with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide
  have hk1 : s1.regs .real [p.D,p.BN] "k" = some kt := by
    simp only [s1,KeysExecution.prepared,BlockState.setReg_same]
    exact congrArg some (Meaning.key_matches p r u initial c hs.memory)
  have hpages1 : s1.regs .int [p.BN] "kv_loc" = some (Tile.vec pages) := by
    simp only [s1,KeysExecution.prepared,BlockState.setReg_ne_name,ne_eq,String.reduceEq,not_false_eq_true,BlockState.setReg_same]
    congr 1
    apply Tile.ext
    rintro ⟨j,⟨⟩⟩
    exact Meaning.pages_match p r u initial c hs.memory j
  have hi1 : s1.regs .int [] "start_n" = some (Tile.scalar sn) := by simp [s1,KeysExecution.prepared]
  let s2 := ScoresExecution.prepared p swapped scale rows prompt sn qt kt s1
  have e2 := ScoresExecution.executes p swapped scale rows prompt sn qt kt s1
    hs1.rows hs1.lanes hs1.prompt hi1 hs1.query hk1
  have f2 := Frames.scores p swapped scale rows prompt sn qt kt s1
  have f12 := then_frame f1 f2
  have hqk2 : s2.regs .real [p.BM,p.BN] "qk" = some qkt := by
    simp only [s2,ScoresExecution.prepared,BlockState.setReg_same]
    exact congrArg some (Meaning.scores_match p r swapped scale initial c)
  obtain ⟨maximum,hrm⟩ := StatsExecution.reduction_exists p qkt hBN
  let mx := StatsExecution.maximum p mi maximum
  let al := StatsExecution.alpha p mi mx
  let prob := StatsExecution.probabilities p qkt mx
  let lij : Tile .real [p.BM] := StatsExecution.denominator p qkt mx
  let ln := StatsExecution.updatedDenominator p li al lij
  let ac := StatsExecution.scaledAcc p acc al
  let s3 := StatsExecution.prepared p qkt mi li maximum acc s2
  have e3 := StatsExecution.executes p qkt mi li maximum acc s2 hqk2
    ((f12.registers .real [p.BM] "m_i" (by decide)).trans hd.maximum)
    ((f12.registers .real [p.BM] "l_i" (by decide)).trans hd.denominator)
    ((f12.registers .real [p.BM,p.D] "acc" (by decide)).trans hd.accumulator) hrm
  have f3 := Frames.stats p qkt mi li maximum acc s2
  have f123 := then_frame f12 f3
  have f23 := then_frame f2 f3
  have hprob3 : s3.regs .real [p.BM,p.BN] "p" = some prob := by simp [s3,StatsExecution.prepared,prob,mx]
  have hmx3 : s3.regs .real [p.BM] "m_ij" = some mx := by simp [s3,StatsExecution.prepared,mx]
  have hac3 : s3.regs .real [p.BM,p.D] "acc" = some ac := by simp [s3,StatsExecution.prepared,ac,al,mx]
  have hln3 : s3.regs .real [p.BM] "l_i" = some ln := by simp [s3,StatsExecution.prepared,ln,al,mx,lij]
  let s4 := ValuesExecution.prepared p r group pages sn last prob mx ac s3
  have e4 := ValuesExecution.executes p r group pages sn last prob mx ac s3
    ((f23.registers .int [p.BN] "kv_loc" (by decide)).trans hpages1)
    ((f123.registers .nat [] "cur_kv_head" (by decide)).trans hs.kv_head)
    ((f23.registers .int [] "start_n" (by decide)).trans hi1)
    ((f123.registers .int [] "block_end_loc" (by decide)).trans hs.ending)
    ((f123.registers .nat [p.BN] "offs_n" (by decide)).trans hs.lanes)
    ((f123.registers .nat [p.D] "offs_d" (by decide)).trans hs.channels) hprob3 hmx3 hac3
  have f4 := Frames.values p r group pages sn last prob mx ac s3
  have fAll := then_frame f123 f4
  have hv3 : ValuesExecution.loaded p r group pages sn last s3 = vt :=
    Meaning.value_matches p r s3 initial c (f123.memory.trans hs.memory)
  refine ⟨s4,maximum,?_,hrm,?_,?_⟩
  · rw [LoopProgram.source]
    simp only [List.append_assoc]
    rw [stepStmts.append_some e1]
    change stepStmts _ s1 = _
    rw [stepStmts.append_some e2]
    change stepStmts _ s2 = _
    rw [stepStmts.append_some e3]
    exact e4
  · apply Frames.preserved p r initial u s4 _ fAll hs
    intro name hn
    simp [readNames] at hn
    rcases hn with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide
  · dsimp only
    change s4.regs .real [p.BM] "m_i" = some mx ∧
      s4.regs .real [p.BM] "l_i" = some ln ∧
      s4.regs .real [p.BM,p.D] "acc" = some (ValuesExecution.updated p prob vt ac)
    refine ⟨?_,(f4.registers .real [p.BM] "l_i" (by decide)).trans hln3,?_⟩
    · simp [s4,ValuesExecution.prepared]
    · simp [s4,ValuesExecution.prepared,hv3]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnLlama.IterationExecution
