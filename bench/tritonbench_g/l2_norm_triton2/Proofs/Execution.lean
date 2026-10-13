import bench.tritonbench_g.l2_norm_triton2.Kernels
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Semantics.TileOps

namespace VeriTile.Bench.TritonBenchG.L2NormTriton2.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

/-- Exact arithmetic witness for the operations used by these sources. This
backend is not an IEEE rounding model; casts preserve their exact input. -/
def model : Algebra (WithBot ℝ) where
  literal := fun _ _ r ↦ some r
  negInf := none
  binary := fun _ _ op a b ↦ match op with
    | .add => WithBot.realAdd a b
    | .sub => WithBot.realSub a b
    | .mul => WithBot.realMul a b
    | .div => WithBot.realDiv a b
    | .max => max a b
    | .pow => some 0
  unary := fun _ op a ↦ match op with
    | .sqrt => WithBot.realSqrt a
    | _ => some 0
  cast := fun _ _ _ x ↦ x
  fromNat := fun _ n ↦ some n
  fromInt := fun _ n ↦ some n
  fp32Bits := fun _ ↦ some 0
  fp32Load := id
  reduceMax := fun _ {_} ax keep v ↦
    ((Tile.reduceMax ax keep ⟨v⟩).getD ⟨fun _ ↦ none⟩).data
  reduceSum := fun _ {_} ax keep v ↦ (Tile.reduceSum ax keep ⟨v⟩).data

/-- One active lane and one padded lane at a nonzero row address. -/
def initial : State (WithBot ℝ) where
  mem := fun r o ↦
    if r = "X" then .mk .fp16 (ofFloat .fp16 (some (if o = 3 then 3 else 97)))
    else if r = "DY" then .mk .bf16 (ofFloat .bf16 (some (if o = 3 then 2 else 89)))
    else .mk .fp16 (ofFloat .fp16 (some 7))
  regs := fun _ _ _ ↦ none
  pids := fun ax ↦ if ax = 0 then 1 else 0
  numPids := fun _ ↦ 2
  undef := fun d _ _ ↦ defaultValue d

/-- The actual complete forward source succeeds in a nontrivial numeric backend. -/
def forward_run (v : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (typedFwd .fp16 v "X" "Y" 3 1 7 2).surfaceBody initial = some t} := by
  cases v
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 300000) [typedFwd, ComputeKernel.surfaceBody,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat,
      TileShape.axisDim, Region.cast, State.setReg]
    try rfl

/-- The actual backward source succeeds, retaining both reductions and dtype casts. -/
def backward_run (v : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (typedBwd .fp16 .bf16 v "X" "DY" "DX" 3 1 7 2).surfaceBody initial = some t} := by
  cases v
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 300000) [typedBwd, ComputeKernel.surfaceBody,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat,
      TileShape.axisDim, Region.cast, State.setReg]
    try rfl

/-- The active output is 3/4; the adjacent masked output retains its sentinel. -/
theorem forward_cells (v : Bool) :
    (forward_run v).val.mem "Y" 3 = Cell.mk .fp16 (some (3 / 4 : ℝ)) ∧
    (forward_run v).val.mem "Y" 4 = Cell.mk .fp16 (some 7) := by
  cases v <;>
    simp [forward_run, initial, model, Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.insertAxisIndex, Fin.sum_univ_succ,
      WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv, WithBot.realSqrt,
      FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ, bop,
      Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat]
  all_goals norm_num [WithBot.some_eq_coe, WithBot.none_eq_bot,
    ← WithBot.coe_add, ← WithBot.coe_mul, Option.map₂, Option.map, Option.bind, Fintype.card_fin,
    Fin.sum_univ_two, Fin.sum_univ_succ]
  all_goals repeat' erw [Fin.sum_univ_two]
  all_goals norm_num [WithBot.some_eq_coe, WithBot.none_eq_bot,
    ← WithBot.coe_add, ← WithBot.coe_mul, Option.map₂, Option.map, Option.bind]
  all_goals exact ⟨rfl, rfl⟩

/-- The actual backward result is 7/32, and padding does not overwrite memory. -/
theorem backward_cells (v : Bool) :
    (backward_run v).val.mem "DX" 3 = Cell.mk .fp16 (some (7 / 32 : ℝ)) ∧
    (backward_run v).val.mem "DX" 4 = Cell.mk .fp16 (some 7) := by
  cases v <;>
    simp [backward_run, initial, model, Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.insertAxisIndex, Fin.sum_univ_succ,
      WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv, WithBot.realSqrt,
      FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ, bop,
      Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat]
  all_goals norm_num [WithBot.some_eq_coe, WithBot.none_eq_bot,
    ← WithBot.coe_add, ← WithBot.coe_mul, Option.map₂, Option.map, Option.bind, Fintype.card_fin,
    Fin.sum_univ_two, Fin.sum_univ_succ]
  all_goals repeat' erw [Fin.sum_univ_two]
  all_goals norm_num [WithBot.some_eq_coe, WithBot.none_eq_bot,
    ← WithBot.coe_add, ← WithBot.coe_mul, Option.map₂, Option.map, Option.bind]
  all_goals exact ⟨rfl, rfl⟩

/-- Successful complete typed execution and numerical output, for either version. -/
theorem forward_numerical_execution (v : Bool) :
    ∃ t, FP.Structural.exec (engine model) (typedFwd .fp16 v "X" "Y" 3 1 7 2)
      initial = some t ∧ t.mem "Y" 3 = Cell.mk .fp16 (some (3 / 4 : ℝ)) ∧
      t.mem "Y" 4 = Cell.mk .fp16 (some 7) := by
  exact ⟨(forward_run v).val, (forward_run v).property, forward_cells v⟩

theorem backward_numerical_execution (v : Bool) :
    ∃ t, FP.Structural.exec (engine model) (typedBwd .fp16 .bf16 v "X" "DY" "DX" 3 1 7 2)
      initial = some t ∧ t.mem "DX" 3 = Cell.mk .fp16 (some (7 / 32 : ℝ)) ∧
      t.mem "DX" 4 = Cell.mk .fp16 (some 7) := by
  exact ⟨(backward_run v).val, (backward_run v).property, backward_cells v⟩

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.L2NormTriton2.NumericExecution
