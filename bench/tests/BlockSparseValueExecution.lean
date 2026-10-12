import bench.tritonbench_g.block_sparse_attn.Proofs.RealLoopTail

/-! Execute the actual source suffix with nonzero V base/start, stride seven,
two channel blocks and nonzero undefined masked values. The physical key tile
straddles the context bound. These are suffix checks, not a full-kernel proof. -/
namespace BlockSparseValueExecutionTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.BlockSparseAttn
open RealNormalizationValues
noncomputable section

private def inputs : Inputs 2 2 2 where
  score := fun r j => if r.val = 0 ∨ j.val = 1 then ⊥ else (0 : WithBot ℝ)
  maximum := fun r => if r.val = 0 then ⊥ else (0 : WithBot ℝ)
  normalizer := fun r => if r.val = 0 then 0 else 1
  accumulator := fun r _ => if r.val = 0 then 0 else 5
  secondAccumulator := fun r _ => if r.val = 0 then 0 else -3

private def pointers : Tile .ptr [2, 2] := Tile.mat (fun j d => ("V", 41 + j.val * 7 + d.val))

private def initial (two : Bool) : BlockState :=
  let s : BlockState :=
    { mem := fun _ o => MemCell.real
        (if o = 62 then 9 else if o = 63 then 13 else if o = 64 then 11 else if o = 65 then 15 else -77),
      regs := fun _ _ _ => none, pids := fun a => a+3, numPids := fun _ => 19,
      undef := fun _ o => 1000 + (o : ℝ) }
  let s := s.setReg "v_ptrs" .ptr [2, 2] pointers
  let s := s.setReg "start_n" .int [] (Tile.scalar 3)
  let s := s.setReg "offs_n" .nat [2] (Tile.vec Fin.val)
  let s := s.setReg "qk" .real [2, 2] (extendedMatrix inputs.score)
  let s := s.setReg "m_i" .real [2] (extendedRow inputs.maximum)
  let s := s.setReg "l_i" .real [2] (row inputs.normalizer)
  let s := s.setReg "acc" .real [2, 2] (matrix inputs.accumulator)
  if two then s.setReg "acc2" .real [2, 2] (matrix inputs.secondAccumulator) else s

private def config (p : RealProgram.Parameters) (even two : Bool) : RealProgram.Parameters :=
  {p with vRow := 7, context := 4, evenN := even, blocks := if two then 2 else 1}

private def finalState (even two : Bool) :=
  RealLoopTail.result (initial two) inputs pointers 3 7 4 (if two then 2 else 1) even

theorem no_preloaded_value (two : Bool) : (initial two).regs .real [2, 2] "v" = none := by
  cases two <;> simp [initial, BlockState.setReg]

theorem actual_source_suffix (p : RealProgram.Parameters) (swapped even two : Bool) :
    stepStmts (RealProgram.realPart (RealProgram.normalization swapped 2 2 2 (config p even two)) ++
      RealProgram.realPart (RealProgram.accumulation swapped 2 2 2 (config p even two))) (initial two) =
        some (finalState even two) := by
  rw [RealNormalizationSource.matches_source, RealAccumulation.matches_source]
  apply RealLoopTail.executes (by decide)
  all_goals cases two <;> simp [initial, config]

private theorem block_max : inputs.blockMax = fun r => if r.val = 0 then ⊥ else (0 : WithBot ℝ) := by
  funext r
  fin_cases r <;> simp [Inputs.blockMax, inputs, Finset.univ_fin2]

private theorem shifted_zero : StreamingMath.shifted (0 : WithBot ℝ) 0 = 1 := by
  change Real.exp (0 - 0) = 1
  norm_num

private theorem probability : inputs.scaledProbability = fun r j =>
    if r.val = 0 ∨ j.val = 1 then 0 else (1 / 2 : ℝ) := by
  funext r j
  simp only [Inputs.scaledProbability, Inputs.probability, Inputs.probabilityScale,
    Inputs.beta, Inputs.alpha, Inputs.nextNormalizer, Inputs.blockNormalizer,
    Inputs.nextMax, block_max]
  fin_cases r <;> fin_cases j <;> norm_num [inputs, shifted_zero, Fin.sum_univ_two]

private theorem scaled_accumulator : inputs.scaledAccumulator = fun r (_ : Fin 2) =>
    if r.val = 0 then 0 else (5 / 2 : ℝ) := by
  funext r d
  simp only [Inputs.scaledAccumulator, Inputs.accumulatorScale, Inputs.nextNormalizer,
    Inputs.alpha, Inputs.beta, Inputs.nextMax, Inputs.blockNormalizer, Inputs.probability, block_max]
  fin_cases r <;> norm_num [inputs, shifted_zero, Fin.sum_univ_two]

private theorem scaled_second : inputs.scaledSecond = fun r (_ : Fin 2) =>
    if r.val = 0 then 0 else (-3 / 2 : ℝ) := by
  funext r d
  simp only [Inputs.scaledSecond, Inputs.accumulatorScale, Inputs.nextNormalizer,
    Inputs.alpha, Inputs.beta, Inputs.nextMax, Inputs.blockNormalizer, Inputs.probability, block_max]
  fin_cases r <;> norm_num [inputs, shifted_zero, Fin.sum_univ_two]

theorem masked_value_is_not_zero (two : Bool) :
    RealValueLoad.loaded (initial two) pointers 3 7 4 Bool.false Bool.false 1 0 = 1069 := by
  cases two <;> norm_num [RealValueLoad.loaded, RealValueLoad.address, RealValueLoad.active,
    initial, pointers, BlockState.setReg, Int.toNat]

private theorem first_update (even two : Bool) :
    RealAccumulation.update inputs.scaledProbability inputs.scaledAccumulator
      (RealValueLoad.loaded (initial two) pointers 3 7 4 even Bool.false) =
      fun r d => if r.val = 0 then 0 else if d.val = 0 then (7 : ℝ) else 9 := by
  funext r d
  simp only [RealAccumulation.update, probability, scaled_accumulator]
  fin_cases r <;> fin_cases d <;> cases two <;>
    norm_num [RealValueLoad.loaded, RealValueLoad.active, RealValueLoad.address,
      initial, pointers, Fin.sum_univ_two, BlockState.readMem, Int.toNat]

theorem first_channel_output (even two : Bool) :
    (finalState even two).regs .real [2, 2] "acc" =
      some (matrix (fun r d => if r.val = 0 then 0 else if d.val = 0 then (7 : ℝ) else 9)) := by
  simpa only [first_update] using
    (RealLoopTail.outputs (initial two) inputs pointers 3 7 4 (if two then 2 else 1) even).accumulator

theorem second_channel_output (even : Bool) :
    (finalState even Bool.true).regs .real [2, 2] "acc2" =
      some (matrix (fun r d => if r.val = 0 then 0 else if d.val = 0 then (4 : ℝ) else 6)) := by
  have h := (RealLoopTail.outputs (initial Bool.true) inputs pointers 3 7 4 2 even).secondAccumulator (by decide)
  have hu : RealAccumulation.update inputs.scaledProbability inputs.scaledSecond
      (RealValueLoad.loaded (initial Bool.true) pointers 3 7 4 even Bool.true) =
      fun r d => if r.val = 0 then 0 else if d.val = 0 then (4 : ℝ) else 6 := by
    funext r d
    simp only [RealAccumulation.update, probability, scaled_second]
    fin_cases r <;> fin_cases d <;>
      norm_num [RealValueLoad.loaded, RealValueLoad.active, RealValueLoad.address,
        initial, pointers, Fin.sum_univ_two, BlockState.readMem, Int.toNat]
  simpa only [hu] using h

theorem single_channel_keeps_second_absent (even : Bool) :
    (finalState even Bool.false).regs .real [2, 2] "acc2" = none := by
  simp [finalState, RealLoopTail.result, RealAccumulation.result, RealAccumulation.dotResult,
    RealNormalization.result, initial, BlockState.setReg]

theorem complete_state_frame (even two : Bool) :
    { finalState even two with regs := (initial two).regs } = initial two :=
  RealLoopTail.state_frame ..

theorem active_bounds (even second : Bool) :
    RealValueSafety.Bounds (fun _ => if even then 73 else 66) pointers 3 7 4 even second := by
  intro j d ha
  fin_cases j <;> fin_cases d <;> cases even <;> cases second <;>
    simp_all [RealValueLoad.address, RealValueLoad.active, pointers, Int.toNat]

theorem actual_loads_safe (even second two : Bool) :
    Stmt.TraceSafeList (fun _ => if even then 73 else 66)
      (RealValueSource.code 2 2 7 4 even second) (initial two) := by
  apply RealValueSafety.code_safe _ (initial two) pointers 3 7 4 even second
  · cases two <;> simp [initial]
  · cases two <;> simp [initial]
  · cases two <;> simp [initial]
  · exact active_bounds even second

theorem first_channel_fits_its_bound :
    RealValueSafety.Bounds (fun _ => 64) pointers 3 7 4 Bool.false Bool.false := by
  intro j d ha
  fin_cases j <;> fin_cases d <;>
    simp_all [RealValueLoad.address, RealValueLoad.active, pointers, Int.toNat]

/-- A bound covering the first channel does not cover the second channel. -/
theorem second_channel_needs_its_own_bound :
    ¬ RealValueSafety.Bounds (fun _ => 64) pointers 3 7 4 Bool.false Bool.true := by
  intro h
  have hb := h 0 0 (by norm_num [RealValueLoad.active])
  norm_num [RealValueLoad.address, pointers, Int.toNat] at hb

#guard_msgs (drop info) in
#auditModuleAxioms
end
end BlockSparseValueExecutionTests
