import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.swiglu_backward.Kernels
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Semantics.TileOps

namespace VeriTile.Bench.TritonBenchG.SwigluBackward.NumericExecution
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

/-- Two nonzero program axes and six independent row strides. -/
def initial : State (WithBot ℝ) where
  mem := fun r o ↦
    if r = "X" then .mk .fp16 (ofFloat .fp16 (some (if o = 6 then 1 else 97)))
    else if r = "Y" then .mk .fp16 (ofFloat .fp16 (some (if o = 7 then 3 else 89)))
    else if r = "DOUT" then .mk .real (some (if o = 8 then 2 else 83))
    else if r = "OUT" then .mk .f8e5 (ofFloat .f8e5 (some 7))
    else .mk .bf16 (ofFloat .bf16 (some 7))
  regs := fun _ _ _ ↦ none
  pids := fun ax ↦ if ax = 0 ∨ ax = 1 then 1 else 0
  numPids := fun _ ↦ 2
  undef := fun d _ _ ↦ defaultValue d

/-- Both source variants execute both recomputation branches with actual loads,
fp32 casts, sigmoid, gradient stores and optional forward store. -/
def source_run (v R : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (typedKernel .fp16 .fp64 .bf16 .f8e5 v "X" "Y" "DOUT" "OUT" "DX" "DY"
        4 5 6 7 8 9 3 2 R).surfaceBody initial = some t} := by
  cases v <;> cases R
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 500000) [typedKernel, template, typedLoad, typedStore,
      loadAddress, storeAddress, recomputedValue, ComputeKernel.surfaceBody,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat,
      TileShape.axisDim, Region.cast, State.setReg]
    try rfl

/-- Both nonzero gradient values and all masked padding are checked. OUT remains
unchanged with recomputation disabled and holds the forward value when enabled. -/
theorem output_cells (v R : Bool) :
    (source_run v R).val.mem "DX" 10 =
      Cell.mk .bf16 (some (6 * Real.sigmoid 1 * (2 - Real.sigmoid 1))) ∧
    (source_run v R).val.mem "DY" 11 = Cell.mk .bf16 (some (2 * Real.sigmoid 1)) ∧
    (source_run v R).val.mem "OUT" 9 =
      Cell.mk .f8e5 (some (if R then 3 * Real.sigmoid 1 else 7)) ∧
    (source_run v R).val.mem "DX" 11 = Cell.mk .bf16 (some 7) ∧
    (source_run v R).val.mem "DY" 12 = Cell.mk .bf16 (some 7) ∧
    (source_run v R).val.mem "OUT" 10 = Cell.mk .f8e5 (some 7) := by
  cases v <;> cases R <;>
    simp [source_run, initial, model,
      WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv, WithBot.realSigmoid,
      FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ, bop,
      Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat]
  all_goals norm_num [WithBot.some_eq_coe, WithBot.none_eq_bot,
    ← WithBot.coe_add, ← WithBot.coe_mul, Option.map₂, Option.map, Option.bind, mul_comm]
  all_goals try ring_nf
  all_goals exact ⟨rfl,rfl,rfl,rfl,rfl⟩

/-- When OUT aliases DX, recomputation is a later store to the same cells. -/
def alias_run (v R : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (typedKernel .fp16 .fp64 .bf16 .bf16 v "X" "Y" "DOUT" "DX" "DX" "DY"
        4 5 6 8 8 9 3 2 R).surfaceBody initial = some t} := by
  cases v <;> cases R
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 500000) [typedKernel, template, typedLoad, typedStore,
      loadAddress, storeAddress, recomputedValue, ComputeKernel.surfaceBody,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat,
      TileShape.axisDim, Region.cast, State.setReg]
    try rfl


/-- The optional final store wins over DX when those outputs alias. Disabling
recomputation preserves the computed DX gradient instead. -/
theorem aliased_output_order (v R : Bool) :
    (alias_run v R).val.mem "DX" 10 = Cell.mk .bf16 (some
      (if R then 3 * Real.sigmoid 1 else 6 * Real.sigmoid 1 * (2 - Real.sigmoid 1))) ∧
    (alias_run v R).val.mem "DY" 11 = Cell.mk .bf16 (some (2 * Real.sigmoid 1)) ∧
    (alias_run v R).val.mem "DX" 11 = Cell.mk .bf16 (some 7) := by
  cases v <;> cases R <;>
    simp [alias_run, initial, model,
      WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv, WithBot.realSigmoid,
      FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ, bop,
      Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat]
  all_goals norm_num [WithBot.some_eq_coe, WithBot.none_eq_bot,
    ← WithBot.coe_add, ← WithBot.coe_mul, Option.map₂, Option.map, Option.bind, mul_comm]
  all_goals try ring_nf
  all_goals exact ⟨rfl,rfl,rfl⟩

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluBackward.NumericExecution
