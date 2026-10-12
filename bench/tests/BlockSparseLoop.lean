import bench.tritonbench_g.block_sparse_attn.Proofs.RealLoop

/-! A signed CSR range crossing zero, with repeated columns and nonzero
attention outputs. Zero scale makes every visible score zero so the exact
normalizer counts visits. Empty and reversed ranges keep the zero carry. -/
namespace BlockSparseLoopTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.BlockSparseAttn
open RealNormalizationValues
noncomputable section
set_option linter.unusedSimpArgs false

private def config (evenM evenN two : Bool) : RealProgram.Parameters where
  Out := "Out"
  Q := "Queries"
  K := "Keys"
  V := "Values"
  rows := ⟨"Rows"⟩
  cols := ⟨"Cols"⟩
  rowStride := 5
  columnStride := 7
  layouts := 2
  scale := 0
  qBatch := 17
  qHead := 19
  qRow := 3
  kBatch := 23
  kHead := 29
  kRow := 5
  vBatch := 31
  vHead := 37
  vRow := 7
  oBatch := 41
  oHead := 43
  oRow := 11
  heads := 2
  kvHeads := 2
  context := 2
  blocks := if two then 2 else 1
  evenM := evenM
  evenN := evenN

private def initial (lo hi : Int) : BlockState where
  mem := fun region offset => if region = "Rows" then
    MemCell.of .int (if offset = 6 then lo else if offset = 7 then hi else 0)
    else if region = "Cols" then MemCell.of .int (if offset = 6 then 1 else 0)
    else if region = "Values" then MemCell.real (if offset % 7 = 6 then 8 else 7)
    else MemCell.real 3
  undef := fun _ offset => 1000 + offset
  regs := fun _ _ _ => none
  pids := fun gridAxis => if gridAxis = 0 then 1 else if gridAxis = 1 then 3 else 0
  numPids := fun _ => 8

private theorem endpoints (lo hi : Int) (evenM evenN two : Bool) :
    RealLoop.first (initial lo hi) (config evenM evenN two) = lo ∧
    RealLoop.last (initial lo hi) (config evenM evenN two) = hi := by
  norm_num [RealLoop.first, RealLoop.last, RealSetup.head, config, initial]
  constructor <;> rfl

theorem crosses_zero_with_three_visits (evenM evenN two : Bool) :
    RealLoop.count (initial (-2) 1) (config evenM evenN two) = 3 := by
  simp only [RealLoop.count, (endpoints (-2) 1 evenM evenN two).1,
    (endpoints (-2) 1 evenM evenN two).2]
  decide

private theorem key_start (lo hi : Int) (evenM evenN two : Bool) (visit : Nat) :
    RealVisit.start (initial lo hi) (RealLoop.operands (initial lo hi) 1 1 1 (config evenM evenN two) visit)
      (config evenM evenN two) = if (7 + (lo + (visit : Int))).toNat = 6 then 1 else 0 := by
  rw [RealVisit.start]
  simp only [RealLoop.operands, (endpoints lo hi evenM evenN two).1]
  norm_num [RealColumn.column, RealColumn.address, RealSetup.head, config, initial]
  have hcr : ("Cols" : RegionName) ≠ "Rows" := by decide
  simp [BlockState.readMemValue, BlockState.readMemTyped, Region.cast, hcr]

private theorem score_zero (lo hi : Int) (evenM evenN two : Bool) (visit : Nat) (r j : Fin 1) :
    RealLoop.score (initial lo hi) 1 1 1 (config evenM evenN two) visit r j = some 0 := by
  simp only [RealLoop.score, key_start]
  have hr : r.val = 0 := by omega
  have hj : j.val = 0 := by omega
  simp [RealScore.masked, RealLoop.operands, RealSetup.rows, config, initial, hr, hj]
  split_ifs <;> norm_num at *
  all_goals rfl

private theorem value_constant (lo hi : Int) (evenM evenN two second : Bool) (visit : Nat) (j d : Fin 1) :
    RealLoop.value (initial lo hi) 1 1 1 (config evenM evenN two) second visit j d = if second then 8 else 7 := by
  simp only [RealLoop.value, key_start]
  have hj : j.val = 0 := by omega
  have hd : d.val = 0 := by omega
  have hvr : ("Values" : RegionName) ≠ "Rows" := by decide
  have hvc : ("Values" : RegionName) ≠ "Cols" := by decide
  by_cases hk : (7 + (lo + (visit : Int))).toNat = 6
  all_goals cases second
  all_goals
    simp only [hk, ↓reduceIte]
    norm_num [RealValueLoad.loaded, RealValueLoad.active, RealValueLoad.address, RealLoop.operands,
      RealSetup.valuePointers, RealSetup.vBase, RealSetup.batch, RealSetup.keyHead, RealSetup.head,
      config, initial, hj, hd, BlockState.readMem, MemCell.real, hvr, hvc, hk, Int.toNat]

theorem final_normalizer_counts_visits (evenM evenN two : Bool) :
    (RealLoop.carry (initial (-2) 1) 1 1 1 (config evenM evenN two) 3).normalizer 0 = 3 := by
  change StreamingMath.normalizer
    (fun n => RealLoop.score (initial (-2) 1) 1 1 1 (config evenM evenN two) n 0) 3 = 3
  have hs : (fun n => RealLoop.score (initial (-2) 1) 1 1 1 (config evenM evenN two) n 0) =
      fun (_ : Nat) (_ : Fin 1) => (some (0 : ℝ) : WithBot ℝ) := by
    funext n j
    exact score_zero (-2) 1 evenM evenN two n 0 j
  rw [hs]
  norm_num [StreamingMath.normalizer, StreamingMath.maximum, StreamingMath.blockMax,
    StreamingMath.shifted, StreamingMath.weight, WithBot.realSub, WithBot.realExp]

theorem final_accumulators (evenM evenN two : Bool) :
    (RealLoop.carry (initial (-2) 1) 1 1 1 (config evenM evenN two) 3).accumulator 0 0 = 7 ∧
    (RealLoop.carry (initial (-2) 1) 1 1 1 (config evenM evenN two) 3).secondAccumulator 0 0 = 8 := by
  change StreamingMath.accumulated _ _ 3 = 7 ∧ StreamingMath.accumulated _ _ 3 = 8
  constructor <;> rw [StreamingMath.accumulated_eq_formula]
  all_goals norm_num [StreamingMath.moment, StreamingMath.mass, score_zero, value_constant,
    Finset.sum_range_succ, StreamingMath.weight]

theorem actual_prefix_and_loop (swapped evenM evenN two : Bool) :
    ∃ t, stepStmts (RealProgram.realPart (RealProgram.parts swapped 1 1 1 (config evenM evenN two)).before ++
        [.forRangeInt "col_idx_idx" (.ref .int [] "start_l") (.ref .int [] "end_l") (.constNat 1)
          (RealProgram.realPart (RealProgram.parts swapped 1 1 1 (config evenM evenN two)).body)]) (initial (-2) 1) = some t ∧
      t.regs .real [1] "l_i" = some (row (fun _ => 3)) ∧
      t.regs .real [1, 1] "acc" = some (matrix (fun _ _ => 7)) ∧
      (two = Bool.true → t.regs .real [1, 1] "acc2" = some (matrix (fun _ _ => 8))) ∧
      {t with regs := (initial (-2) 1).regs} = initial (-2) 1 := by
  obtain ⟨t, ht, hf, hc⟩ := RealLoop.prologue_loop_executes swapped (initial (-2) 1) 1 1 1
    (config evenM evenN two) (by decide)
  rw [crosses_zero_with_three_visits] at hc
  have hn : (RealLoop.carry (initial (-2) 1) 1 1 1 (config evenM evenN two) 3).normalizer = fun _ => 3 := by
    funext r
    have hr : r = 0 := Subsingleton.elim _ _
    simpa only [hr] using final_normalizer_counts_visits evenM evenN two
  have ha : (RealLoop.carry (initial (-2) 1) 1 1 1 (config evenM evenN two) 3).accumulator = fun _ _ => 7 := by
    funext r d
    have hr : r = 0 := Subsingleton.elim _ _
    have hd : d = 0 := Subsingleton.elim _ _
    simpa only [hr, hd] using (final_accumulators evenM evenN two).1
  have ha2 : (RealLoop.carry (initial (-2) 1) 1 1 1 (config evenM evenN two) 3).secondAccumulator = fun _ _ => 8 := by
    funext r d
    have hr : r = 0 := Subsingleton.elim _ _
    have hd : d = 0 := Subsingleton.elim _ _
    simpa only [hr, hd] using (final_accumulators evenM evenN two).2
  refine ⟨t, ht, by simpa only [hn] using hc.normalizer, by simpa only [ha] using hc.accumulator, ?_, hf.frame⟩
  intro htwo
  simpa only [ha2] using hc.secondAccumulator (by simp [config, htwo])

theorem reversed_range_is_empty (evenM evenN two : Bool) (lo hi : Int) (h : hi ≤ lo) :
    RealLoop.count (initial lo hi) (config evenM evenN two) = 0 := by
  simp only [RealLoop.count, (endpoints lo hi evenM evenN two).1, (endpoints lo hi evenM evenN two).2]
  omega

theorem empty_loop_returns_prefix (swapped evenM evenN two : Bool) (lo hi : Int) (h : hi ≤ lo) :
    stepStmt (.forRangeInt "col_idx_idx" (.ref .int [] "start_l") (.ref .int [] "end_l") (.constNat 1)
      (RealProgram.realPart (RealProgram.parts swapped 1 1 1 (config evenM evenN two)).body))
      (RealPrologue.result (initial lo hi) 1 1 1 (config evenM evenN two)) =
      some (RealPrologue.result (initial lo hi) 1 1 1 (config evenM evenN two)) := by
  have hp := RealPrologue.outputs (initial lo hi) 1 1 1 (config evenM evenN two)
  rw [stepForRangeIntAux.forRangeInt_unfold]
  simp only [evalOp_ref, hp.first, hp.last, evalOp_constNat, Option.bind_some, Tile.scalar_data]
  change stepForRangeIntAux "col_idx_idx" (RealLoop.first (initial lo hi) (config evenM evenN two))
    (RealLoop.last (initial lo hi) (config evenM evenN two)) 1 _ _ = _
  rw [(endpoints lo hi evenM evenN two).1, (endpoints lo hi evenM evenN two).2]
  exact stepForRangeIntAux.step_ge (by decide) h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end BlockSparseLoopTests
