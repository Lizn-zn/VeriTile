import bench.tritonbench_g.swiglu_triton.Kernels
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Semantics.TileOps

namespace VeriTile.Bench.TritonBenchG.SwigluTriton.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 8000000
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
    | .sigmoid => WithBot.realSigmoid a
    | _ => some 0
  cast := fun _ _ _ x ↦ x
  fromNat := fun _ n ↦ some n
  fromInt := fun _ n ↦ some n
  fp32Bits := fun _ ↦ some 0
  fp32Load := id
  reduceMax := fun _ {_} ax keep v ↦
    ((Tile.reduceMax ax keep ⟨v⟩).getD ⟨fun _ ↦ none⟩).data
  reduceSum := fun _ {_} ax keep v ↦ (Tile.reduceSum ax keep ⟨v⟩).data

/-- Mixed half/double/bfloat storage at row one, with a masked adjacent lane. -/
def initial : State (WithBot ℝ) where
  mem := fun r o ↦
    if r = "a" then .mk .fp16 (ofFloat .fp16 (some (if o = 3 then 0 else 97)))
    else if r = "b" then .mk .real (some (if o = 3 then 3 else 89))
    else if r = "dc" then .mk .bf16 (ofFloat .bf16 (some (if o = 3 then 4 else 83)))
    else .mk .fp16 (ofFloat .fp16 (some 7))
  regs := fun _ _ _ ↦ none
  pids := fun ax ↦ if ax = 0 then 1 else 0
  numPids := fun _ ↦ 2
  undef := fun d _ _ ↦ defaultValue d

/-- Complete forward execution with actual masked loads, sigmoid and stores. -/
def forward_run (v : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (typedFwd .fp16 .fp64 v "a" "b" "c" 3 1 2).surfaceBody initial = some t} := by
  cases v
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 500000) [typedFwd, ComputeKernel.surfaceBody,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat,
      TileShape.axisDim, Region.cast, State.setReg]
    try rfl

/-- Exact stored numerical values and masked-lane preservation. -/
theorem forward_cells (v : Bool) :
    (forward_run v).val.mem "c" 3 = Cell.mk .fp16 (some 0) ∧
    (forward_run v).val.mem "c" 4 = Cell.mk .fp16 (some 7) := by
  cases v <;>
    simp [forward_run, initial, model,
      WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv, WithBot.realSigmoid,
      FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ, bop,
      Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat]
  all_goals norm_num [WithBot.some_eq_coe, WithBot.none_eq_bot,
    ← WithBot.coe_add, ← WithBot.coe_mul, Option.map₂, Option.map, Option.bind]
  all_goals first | rfl | exact ⟨rfl, rfl⟩ | exact ⟨rfl, rfl, rfl, rfl⟩

/-- Complete backward execution with actual masked loads, sigmoid and stores. -/
def backward_run (v : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (typedBwd .fp16 .fp64 .bf16 v "dc" "a" "b" 3 1 2).surfaceBody initial = some t} := by
  cases v
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 500000) [typedBwd, typedBwd_fp16_fp64, ComputeKernel.surfaceBody,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat,
      TileShape.axisDim, Region.cast, State.setReg]
    try rfl

/-- Exact stored numerical values and masked-lane preservation. -/
theorem backward_cells (v : Bool) :
    (backward_run v).val.mem "a" 3 = Cell.mk .fp16 (some 6) ∧
    (backward_run v).val.mem "b" 3 = Cell.mk .real (some 0) ∧
    (backward_run v).val.mem "a" 4 = Cell.mk .fp16 (some 97) ∧
    (backward_run v).val.mem "b" 4 = Cell.mk .real (some 89) := by
  cases v <;>
    simp [backward_run, initial, model,
      WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv, WithBot.realSigmoid,
      FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ, bop,
      Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat]
  all_goals norm_num [WithBot.some_eq_coe, WithBot.none_eq_bot,
    ← WithBot.coe_add, ← WithBot.coe_mul, Option.map₂, Option.map, Option.bind]
  all_goals first | rfl | exact ⟨rfl, rfl⟩ | exact ⟨rfl, rfl, rfl, rfl⟩

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluTriton.NumericExecution
