import bench.tritonbench_g.kldiv_triton.Kernels
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KldivTriton.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
noncomputable section
open scoped Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Exact-real backend with actual logarithm, exponential and reductions. Casts
are identities for this numerical witness; this is not an IEEE implementation. -/
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
    | .log => WithBot.realLog a
    | .exp => WithBot.realExp a
    | _ => some 0
  cast := fun _ _ _ x ↦ x
  fromNat := fun _ n ↦ some n
  fromInt := fun _ n ↦ some n
  fp32Bits := fun _ ↦ some 0
  fp32Load := id
  reduceMax := fun _ {_} ax keep v ↦
    ((Tile.reduceMax ax keep ⟨v⟩).getD ⟨fun _ ↦ none⟩).data
  reduceSum := fun _ {_} ax keep v ↦ (Tile.reduceSum ax keep ⟨v⟩).data

/-- Distinct row strides, nonzero program id and tagged output sentinel. -/
def initial (flag : Bool) : State (WithBot ℝ) where
  mem := fun r _ ↦
    if r = "Y" then .mk .real (some 2)
    else if r = "T" then .mk .real (some (if flag then 0 else 1))
    else .mk .bf16 (some 7)
  regs := fun _ _ _ ↦ none
  pids := fun ax ↦ if ax = 0 then 1 else 0
  numPids := fun _ ↦ 2
  undef := fun d _ _ ↦ defaultValue d

/-- Three columns and block size two execute two actual reduced stores. -/
def reduced_run (v flag : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (forward .fp64 .fp32 .bf16 v "Y" "T" "O" 7 9 5 3 2 flag 1).surfaceBody
      (initial flag) = some t} := by
  cases v <;> cases flag
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 500000) [forward, forwardOriginal, commuteOutput,
      ComputeKernel.surfaceBody, ComputeKernel.inputs, ComputeKernel.outputs,
      run, step, FP.Structural.rangeInt, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, numericLt, numericEq, bop, natLt, initial, Cell.read,
      ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg,
      FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ]
    try rfl

private theorem sum_fin2 {α : Type*} [AddCommMonoid α]
    (I : Fintype (Fin 2)) (f : Fin 2 → α) :
    @Finset.sum (Fin 2) α _ (@Finset.univ _ I) f = f 0 + f 1 := by
  have hI : I = Fin.fintype 2 := Subsingleton.elim _ _
  rw [hI]
  simp [Fin.sum_univ_succ]

/-- The two chunk results remain separate; surrounding cells retain the sentinel. -/
theorem reduced_cells (v flag : Bool) :
    (reduced_run v flag).val.mem "O" 5 = Cell.mk .bf16 (some (-4)) ∧
    (reduced_run v flag).val.mem "O" 6 = Cell.mk .bf16 (some (-2)) ∧
    (reduced_run v flag).val.mem "O" 4 = Cell.mk .bf16 (some 7) ∧
    (reduced_run v flag).val.mem "O" 7 = Cell.mk .bf16 (some 7) := by
  cases v <;> cases flag
  all_goals simp [reduced_run, initial, model, Region.cast,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat,
    Tile.reduceSum, Tile.reduceSumDrop, TileShape.insertAxisIndex, TileShape.axisDim,
    sum_fin2, WithBot.realAdd, WithBot.realSub, WithBot.realLog,
    WithBot.realExp, WithBot.realMul]
  all_goals norm_num [WithBot.some_eq_coe, ← WithBot.coe_add, Option.map₂, Option.map, Option.bind,
    bop, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar, -Tile.scalar_eta,
    Function.comp_def, List.finRange_succ, List.foldl, List.flatMap, sum_fin2]
  all_goals first | rfl | (repeat' constructor)

/-- An empty row still performs initialization but no reduced-output store. -/
def empty_run (v flag : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (forward .fp64 .fp32 .bf16 v "Y" "T" "O" 7 9 5 0 2 flag 1).surfaceBody
      (initial flag) = some t} := by
  cases v <;> cases flag
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 500000) [forward, forwardOriginal, commuteOutput,
      ComputeKernel.surfaceBody, ComputeKernel.inputs, ComputeKernel.outputs,
      run, step, FP.Structural.rangeInt, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, numericLt, numericEq, bop, natLt, initial, Cell.read,
      ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg,
      FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ]
    try rfl

theorem empty_frame (v flag : Bool) : (empty_run v flag).val.mem = (initial flag).mem := by
  cases v <;> cases flag <;> rfl

/-- The non-reduced branch executes both blocks and masks its final inactive lane. -/
def element_run (v flag : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (forward .fp64 .fp32 .bf16 v "Y" "T" "O" 7 9 5 3 2 flag 0).surfaceBody
      (initial flag) = some t} := by
  cases v <;> cases flag
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 500000) [forward, forwardOriginal, commuteOutput,
      ComputeKernel.surfaceBody, ComputeKernel.inputs, ComputeKernel.outputs,
      run, step, FP.Structural.rangeInt, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, numericLt, numericEq, bop, natLt, initial, Cell.read,
      ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg,
      FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ]
    try rfl

theorem element_cells (v flag : Bool) :
    (element_run v flag).val.mem "O" 5 = Cell.mk .bf16 (some (-2)) ∧
    (element_run v flag).val.mem "O" 6 = Cell.mk .bf16 (some (-2)) ∧
    (element_run v flag).val.mem "O" 7 = Cell.mk .bf16 (some (-2)) ∧
    (element_run v flag).val.mem "O" 8 = Cell.mk .bf16 (some 7) := by
  cases v <;> cases flag
  all_goals norm_num [element_run, initial, model, Region.cast,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat,
    WithBot.realAdd, WithBot.realSub, WithBot.realLog, WithBot.realExp, WithBot.realMul]

/-- Backward witnesses use distinct row strides and preserve the inactive fourth lane. -/
def backwardInitial (flag : Bool) : State (WithBot ℝ) where
  mem := fun r _ ↦ if r = "T" then .mk .real (some (if flag then 0 else 2))
    else .mk .fp16 (some 7)
  regs := fun _ _ _ ↦ none
  pids := fun ax ↦ if ax = 0 then 1 else 0
  numPids := fun _ ↦ 2
  undef := fun d _ _ ↦ defaultValue d

def backward_run (v flag : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (if flag then (backwardLog .fp64 .fp16 v "T" "O" 9 5 3 2).surfaceBody
        else (backwardDefault .fp64 .fp16 v "T" "O" 9 5 3 2).surfaceBody)
      (backwardInitial flag) = some t} := by
  cases v <;> cases flag
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 500000) [backwardLog, backwardLogOriginal,
      backwardDefault, backwardDefaultOriginal, commuteOutput,
      ComputeKernel.surfaceBody, ComputeKernel.inputs, ComputeKernel.outputs,
      run, step, FP.Structural.rangeInt, FP.Structural.evalExpr, evalComputeOp,
      evalOp_unfold, evalOffsets, evalDeltas, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
      numeric, numericLt, numericEq, bop, natLt, backwardInitial, Cell.read,
      ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg,
      FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ]
    try rfl

theorem backward_cells (v flag : Bool) :
    (backward_run v flag).val.mem "O" 5 = Cell.mk .fp16 (some (if flag then -1 else -2)) ∧
    (backward_run v flag).val.mem "O" 6 = Cell.mk .fp16 (some (if flag then -1 else -2)) ∧
    (backward_run v flag).val.mem "O" 7 = Cell.mk .fp16 (some (if flag then -1 else -2)) ∧
    (backward_run v flag).val.mem "O" 8 = Cell.mk .fp16 (some 7) := by
  cases v <;> cases flag
  all_goals norm_num [backward_run, backwardInitial, model, Region.cast,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat,
    WithBot.realAdd, WithBot.realSub, WithBot.realExp, WithBot.realMul]

/-- A nonempty half-precision multiplication/store witness exercises the literal repair. -/
def halfInitial : State (WithBot ℝ) where
  mem := fun r _ ↦ .mk .fp16 (some (if r = "T" then 2 else 7))
  regs := fun _ _ _ ↦ none
  pids := fun ax ↦ if ax = 0 then 1 else 0
  numPids := fun _ ↦ 2
  undef := fun d _ _ ↦ defaultValue d

def half_run (v : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (backwardDefault .fp16 .fp16 v "T" "O" 9 5 3 2).surfaceBody halfInitial = some t} := by
  cases v
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 500000) [backwardDefault, backwardDefaultOriginal, commuteOutput,
      ComputeKernel.surfaceBody, ComputeKernel.inputs, ComputeKernel.outputs,
      run, step, FP.Structural.rangeInt, FP.Structural.evalExpr, evalComputeOp,
      evalOp_unfold, evalOffsets, evalDeltas, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
      numeric, numericLt, numericEq, bop, natLt, halfInitial, Cell.read,
      ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg,
      FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ]
    try rfl

theorem half_cells (v : Bool) :
    (half_run v).val.mem "O" 5 = Cell.mk .fp16 (some (-2)) ∧
    (half_run v).val.mem "O" 6 = Cell.mk .fp16 (some (-2)) ∧
    (half_run v).val.mem "O" 7 = Cell.mk .fp16 (some (-2)) ∧
    (half_run v).val.mem "O" 8 = Cell.mk .fp16 (some 7) := by
  cases v
  all_goals norm_num [half_run, halfInitial, model, Region.cast,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat,
    WithBot.realAdd, WithBot.realSub, WithBot.realMul]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivTriton.NumericExecution
