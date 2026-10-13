import bench.tritonbench_g.rmsnorm_fused.Kernels
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsnormFused.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

/-- Exact-real numerical witness with actual square root and sum reduction.
Casts preserve values; this algebra is not an IEEE rounding implementation. -/
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


/-- Nonzero row and a padded stride. Every active input is two and weight three. -/
def initial (lateWide : Bool) : State (WithBot ℝ) where
  mem := fun r _ => if r = "X" then
      if lateWide then .mk .fp16 (ofFloat .fp16 (some 2)) else .mk .real (some 2)
    else if r = "W" then
      if lateWide then .mk .real (some 3) else .mk .fp16 (ofFloat .fp16 (some 3))
    else .mk .bf16 (ofFloat .bf16 (some 7))
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 1 else 0
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

/-- Complete executions retain both source loops, including direct-call multi-block runs. -/
def numeric_run (v lateWide multi : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (typedKernel (if lateWide then .fp16 else .fp64) (if lateWide then .fp64 else .fp16) .bf16 v "X" "Y" "W" 5 (if multi then 3 else 1) 2 5).surfaceBody
        (initial lateWide) = some t} := by
  cases v <;> cases lateWide <;> cases multi
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 1000000) [typedKernel, originalTyped, ComputeKernel.surfaceBody,
      ComputeKernel.inputs, ComputeKernel.outputs,
      run, step, FP.Structural.range, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
      evalOffsets, evalDeltas, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, ComputeDType.eraseDType, numeric, bop, natLt,
      initial, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg,
      FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ, Tile.select, Tile.ptrAdd]
    try rfl

private theorem sum_fin2 {α : Type*} [AddCommMonoid α]
    (I : Fintype (Fin 2)) (f : Fin 2 → α) :
    @Finset.sum (Fin 2) α _ (@Finset.univ _ I) f = f 0 + f 1 := by
  have hI : I = Fin.fintype 2 := Subsingleton.elim _ _
  rw [hI]
  simp [Fin.sum_univ_succ]

/-- Both variants cover fp64 input conversion and fp64 weight promotion in the wrapper regime
and in a two-iteration direct call. Masked cells retain their sentinel. -/
theorem numeric_cells (v lateWide multi : Bool) :
    (numeric_run v lateWide multi).val.regs .real [] "rstd" = some (fun _ => some (1 / 3)) ∧
    (numeric_run v lateWide multi).val.mem "Y" 5 = Cell.mk .bf16 (some 2) ∧
    (numeric_run v lateWide multi).val.mem "Y" 6 = Cell.mk .bf16 (some (if multi then 2 else 7)) ∧
    (numeric_run v lateWide multi).val.mem "Y" 7 = Cell.mk .bf16 (some (if multi then 2 else 7)) ∧
    (numeric_run v lateWide multi).val.mem "Y" 8 = Cell.mk .bf16 (some 7) ∧
    (numeric_run v lateWide multi).val.mem "Y" 4 = Cell.mk .bf16 (some 7) := by
  cases v <;> cases lateWide <;> cases multi
  all_goals simp (maxSteps := 1000000) [numeric_run, initial, model,
    FP.Structural.store, State.write, State.setReg, Region.cast, Tile.scalar, -Tile.scalar_eta,
    Function.comp_def, Tile.select, Tile.ptrAdd, TileShape.allIndices, List.finRange_succ, List.foldl, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat,
    Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim, TileShape.insertAxisIndex,
    TileShape.eraseAxis, sum_fin2,
    WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv, WithBot.realSqrt]
  all_goals norm_num [WithBot.some_eq_coe, ← WithBot.coe_add, Option.map₂, Option.map, Option.bind,
    bop, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar, -Tile.scalar_eta, Function.comp_def,
    List.finRange_succ, List.foldl, List.flatMap, sum_fin2]
  all_goals first | rfl | (repeat' constructor)
  all_goals first | rfl | (congr 1; funext i; norm_num)

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormFused.NumericExecution
