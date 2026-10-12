import bench.tritonbench_g.block_sparse_attn.Proofs.RealVisit

/-! Execute the actual column/score prefix for both source directions, both
K masks and both channel branches. Nonzero signed metadata and pointer bases
produce scores 94 / 320. A query row outside context remains in the tile;
only valid-row scores are compared with the independent formula. -/

namespace BlockSparseScoreTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.BlockSparseAttn
open RealNormalizationValues
noncomputable section
set_option linter.unusedSimpArgs false

private def pointers : Tile .ptr [2, 3] :=
  Tile.mat fun d j => ("Keys", 41 + d.val + j.val * 7)
private def rows : Fin 2 → Nat := fun r => 3 + 2*r.val
private def query : Fin 2 → Fin 2 → ℝ := fun _ d => if d.val = 0 then 1 else 2
private def secondQuery : Fin 2 → Fin 2 → ℝ := fun _ d => if d.val = 0 then 3 else 4
private def initial (two : Bool) : BlockState :=
  let s : BlockState :=
    { mem := fun r o => if r = "Cols" then MemCell.of .int (if o = 11 then 1 else 9)
        else MemCell.real (o : ℝ),
      undef := fun _ o => 1000 + o,
      regs := fun _ _ _ => none, pids := fun a => a + 2, numPids := fun _ => 7 }
  let s := s.setReg "layout_h" .nat [] (Tile.scalar 1)
  let s := s.setReg "col_idx_idx" .int [] (Tile.scalar (-2))
  let s := s.setReg "k_ptrs" .ptr [2, 3] pointers
  let s := s.setReg "offs_m" .nat [2] (Tile.vec rows)
  let s := s.setReg "offs_n" .nat [3] (Tile.vec Fin.val)
  let s := s.setReg "q" .real [2, 2] (matrix query)
  if two then s.setReg "q2" .real [2, 2] (matrix secondQuery) else s
private def config (p : RealProgram.Parameters) (even two : Bool) : RealProgram.Parameters :=
  {p with
    cols := ⟨"Cols"⟩
    columnStride := 13
    kRow := 7
    context := 4
    scale := 1/2
    evenN := even
    blocks := if two then 2 else 1}
private def afterColumn (two : Bool) := RealColumn.result (initial two) 3 ⟨"Cols"⟩ 1 13 (-2)
private def finalState (even two : Bool) :=
  RealScore.result (afterColumn two) pointers 3 7 4 (if two then 2 else 1) (1/2) even rows query secondQuery

private theorem column_value (two : Bool) : RealColumn.column (initial two) ⟨"Cols"⟩ 1 13 (-2) = 1 := by
  cases two <;> rfl

theorem actual_score_prefix (p : RealProgram.Parameters) (swapped even two : Bool) :
    stepStmts (RealProgram.realPart (RealProgram.normalizationPrefix swapped 2 3 2 (config p even two))) (initial two) =
      some (finalState even two) := by
  have h := RealScore.source_executes swapped 2 3 2 (config p even two) (initial two) 1 (-2)
    pointers rows query secondQuery
    (by cases two <;> simp [initial]) (by cases two <;> simp [initial])
    (by cases two <;> simp [initial]) (by cases two <;> simp [initial])
    (by cases two <;> simp [initial]) (by cases two <;> simp [initial])
    (by cases two <;> simp [config, initial])
  simpa only [config, column_value, show (1 : Int)*3=3 by decide] using h

private theorem unscaled_score (even two : Bool) :
    RealScore.unscaled (afterColumn two) pointers 3 7 4 (if two then 2 else 1) even query secondQuery 0 0 =
      if two then 640 else 188 := by
  have hkc : ("Keys" : RegionName) ≠ "Cols" := by decide
  cases two <;> norm_num [RealScore.unscaled, RealScore.dotValues, RealKey.loaded, RealKey.active,
    RealKey.address, afterColumn, RealColumn.result, initial, pointers,
    query, secondQuery, Fin.sum_univ_two, BlockState.readMem, Int.toNat, hkc, MemCell.real]

theorem actual_visible_score (even two : Bool) :
    ((finalState even two).regs .real [2, 3] "qk").map (fun t => t.data (0, 0, PUnit.unit)) =
      some ((if two then (320 : ℝ) else 94 : ℝ) : WithBot ℝ) := by
  simp only [finalState, RealScore.readback, Option.map_some, extendedMatrix]
  norm_num [RealScore.masked, rows, unscaled_score]

theorem causally_masked_scores (even two : Bool) :
    ((finalState even two).regs .real [2, 3] "qk").map (fun t => t.data (0, 1, PUnit.unit)) = some ⊥ ∧
    ((finalState even two).regs .real [2, 3] "qk").map (fun t => t.data (0, 2, PUnit.unit)) = some ⊥ := by
  simp only [finalState, RealScore.readback, Option.map_some, extendedMatrix]
  constructor <;> norm_num [RealScore.masked, rows]

theorem one_invalid_query_row : rows 0 < 4 ∧ ¬ rows 1 < 4 := by decide

theorem complete_state_frame (even two : Bool) :
    {finalState even two with regs := (initial two).regs} = initial two := by
  cases two <;> simp [finalState, RealScore.result, RealScore.dotsResult,
    RealScore.dotResult, afterColumn, RealColumn.result, BlockState.setReg]

theorem single_channel_keeps_second_query_absent (even : Bool) :
    (finalState even Bool.false).regs .real [2, 2] "q2" = none := by
  simp [finalState, RealScore.result, RealScore.dotsResult, RealScore.dotResult,
    afterColumn, RealColumn.result, initial, BlockState.setReg]

private def allocation (two : Bool) : RegionBounds := fun r =>
  if r = "Cols" then 12 else if two then 66 else 64

private theorem key_bounds (two second : Bool) (allowed : second = Bool.true → two = Bool.true) :
    RealKey.Bounds (allocation two) pointers 3 7 4 Bool.false second := by
  have hkc : ("Keys" : RegionName) ≠ "Cols" := by decide
  cases two <;> cases second <;> simp_all only [Bool.false_eq_true, false_implies, true_implies]
  all_goals
    intro d j ha
    have hj : j.val = 0 := by
      simp [RealKey.active] at ha
      omega
    have hd := d.isLt
    simp [RealKey.address, pointers, allocation, Int.toNat, hkc, hj]
    omega

theorem actual_score_prefix_safe (p : RealProgram.Parameters) (swapped two : Bool) :
    Stmt.TraceSafeList (allocation two)
      (RealProgram.realPart (RealProgram.normalizationPrefix swapped 2 3 2 (config p Bool.false two)))
      (initial two) := by
  apply RealScore.source_safe (layout := 1) (visit := -2) (ptrs := pointers) (q := query)
  · cases two <;> simp [initial]
  · cases two <;> simp [initial]
  · cases two <;> simp [initial]
  · cases two <;> simp [initial]
  · cases two <;> simp [initial]
  · norm_num [config, RealColumn.address, allocation, Int.toNat, Region.cast]
  · simpa only [config, column_value, show (1 : Int)*3=3 by decide] using key_bounds two Bool.false (by simp)
  · intro hb
    have htwo : two = Bool.true := by cases two <;> simp [config] at hb ⊢
    simpa only [config, column_value, show (1 : Int)*3=3 by decide] using key_bounds two Bool.true (fun _ => htwo)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end BlockSparseScoreTests

namespace BlockSparseScoreTests
open VeriTile.Triton VeriTile.Bench.TritonBenchG.BlockSparseAttn
open RealNormalizationValues
noncomputable section

private def valuePointers : Tile .ptr [3, 2] :=
  Tile.mat fun j d => ("Values", 41 + j.val * 7 + d.val)
private def carry : Inputs 2 3 2 where
  score := fun _ _ => ⊥
  maximum := fun _ => ⊥
  normalizer := fun _ => 0
  accumulator := fun _ _ => 0
  secondAccumulator := fun _ _ => 0
private def operands : RealVisit.Operands 2 3 2 where
  layout := 1
  visit := -2
  keyPointers := pointers
  valuePointers := valuePointers
  rows := rows
  query := query
  secondQuery := secondQuery
private def loopInitial (two : Bool) : BlockState :=
  let s := (initial two).setReg "v_ptrs" .ptr [3, 2] valuePointers
  let s := s.setReg "m_i" .real [2] (extendedRow carry.maximum)
  let s := s.setReg "l_i" .real [2] (row carry.normalizer)
  let s := s.setReg "acc" .real [2, 2] (matrix carry.accumulator)
  if two then s.setReg "acc2" .real [2, 2] (matrix carry.secondAccumulator) else s
private def visitConfig (p : RealProgram.Parameters) (even two : Bool) : RealProgram.Parameters :=
  {config p even two with vRow := 7}
private def visitResult (p : RealProgram.Parameters) (even two : Bool) : BlockState :=
  RealVisit.result (loopInitial two) operands (visitConfig p even two) carry

/-- The source body executes from Q/pointers and initial carries. K, V, qk,
probabilities and updated carries are all produced by the actual statements. -/
theorem complete_visit_executes (p : RealProgram.Parameters) (swapped even two : Bool) :
    stepStmts (RealProgram.realPart (RealProgram.parts swapped 2 3 2 (visitConfig p even two)).body)
      (loopInitial two) = some (visitResult p even two) := by
  apply RealVisit.source_executes swapped (by decide)
  all_goals cases two <;> simp [loopInitial, initial, operands, carry, visitConfig, config]

theorem complete_visit_state_frame (p : RealProgram.Parameters) (even two : Bool) :
    {visitResult p even two with regs := (loopInitial two).regs} = loopInitial two :=
  RealVisit.state_frame ..

theorem complete_visit_keeps_fixed_operands (p : RealProgram.Parameters) (even two : Bool) :
    (visitResult p even two).regs .ptr [3, 2] "v_ptrs" = some valuePointers ∧
    (visitResult p even two).regs .int [] "col_idx_idx" = some (Tile.scalar (-2)) := by
  constructor
  · rw [visitResult, RealVisit.register_frame _ _ _ _ _ _ _ (by simp [RealVisit.writes, RealNormalization.writes, RealAccumulation.writes])]
    cases two <;> simp [loopInitial, initial]
  · rw [visitResult, RealVisit.register_frame _ _ _ _ _ _ _ (by simp [RealVisit.writes, RealNormalization.writes, RealAccumulation.writes])]
    cases two <;> simp [loopInitial, initial]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end BlockSparseScoreTests
