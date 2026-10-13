import bench.tritonbench_g.log_softmax.Kernels
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.LogSoftmax.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

/-- Exact-real numerical witness with actual exponential, logarithm and reductions.
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
    | .exp => WithBot.realExp a
    | .log => WithBot.realLog a
    | _ => some 0
  cast := fun _ _ _ x ↦ x
  fromNat := fun _ n ↦ some n
  fromInt := fun _ n ↦ some n
  fp32Bits := fun _ ↦ some 0
  fp32Load := id
  reduceMax := fun _ {_} ax keep v ↦
    ((Tile.reduceMax ax keep ⟨v⟩).getD ⟨fun _ ↦ none⟩).data
  reduceSum := fun _ {_} ax keep v ↦ (Tile.reduceSum ax keep ⟨v⟩).data

/-- Only the first tile row and its first two columns are active. Masked lanes
at addresses 13 and 15 overlap the masked next row; neither may overwrite them. -/
def forwardInitial : State (WithBot ℝ) where
  mem := fun r _ ↦ if r = "X" then .mk .real (some 0)
    else .mk .bf16 (ofFloat .bf16 (some 7))
  regs := fun _ _ _ ↦ none
  pids := fun ax ↦ if ax = 0 ∨ ax = 1 then 1 else 0
  numPids := fun _ ↦ 2
  undef := fun d _ _ ↦ defaultValue d

/-- A complete mixed-storage forward execution, including padded columns and a
fully masked row. Both source variants execute every source statement. -/
def forward_run (v : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (typedForward .fp64 .bf16 v "OUT" "X" 3 2 2 2 4).surfaceBody forwardInitial = some t} := by
  cases v
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 500000) [typedForward, ComputeKernel.surfaceBody,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, bop, natLt, forwardInitial, Cell.read, ofFloat, toFloat,
      TileShape.axisDim, Region.cast, State.setReg]
    try rfl

private theorem univ_fin4 (I : Fintype (Fin 4)) :
    @Finset.univ (Fin 4) I = {0, 1, 2, 3} := by
  ext i
  simp only [Finset.mem_univ, Finset.mem_insert, Finset.mem_singleton, true_iff, Fin.ext_iff]
  omega

private theorem sum_fin4 {α : Type*} [AddCommMonoid α]
    (I : Fintype (Fin 4)) (f : Fin 4 → α) :
    @Finset.sum (Fin 4) α _ (@Finset.univ _ I) f = f 0 + f 1 + f 2 + f 3 := by
  have hI : I = Fin.fintype 4 := Subsingleton.elim _ _
  rw [hI]
  simp [Fin.sum_univ_succ, add_assoc]

/-- The two active values are log(1/2); overlapping masked addresses retain seven. -/
theorem forward_cells (v : Bool) :
    (forward_run v).val.mem "OUT" 9 = Cell.mk .bf16 (some (Real.log (1 / 2))) ∧
    (forward_run v).val.mem "OUT" 11 = Cell.mk .bf16 (some (Real.log (1 / 2))) ∧
    (forward_run v).val.mem "OUT" 13 = Cell.mk .bf16 (some 7) ∧
    (forward_run v).val.mem "OUT" 15 = Cell.mk .bf16 (some 7) := by
  cases v
  all_goals simp (maxSteps := 500000) [forward_run, forwardInitial, model,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.insertAxisIndex, TileShape.eraseAxis,
    univ_fin4, Finset.sup'_eq_sup, Fin.sum_univ_succ,
    WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv,
    WithBot.realExp, WithBot.realLog]
  all_goals norm_num [univ_fin4, Fin.sum_univ_succ, Finset.sup'_eq_sup,
    WithBot.some_eq_coe, ← WithBot.coe_add, Option.map₂, Option.map, Option.bind]
  all_goals simp only [one_div, Real.log_inv]
  all_goals exact ⟨rfl, rfl⟩

/-- The gradient's inactive fp16 cells are deliberately unspecified by the source.
The parameter controls those actual loaded padding values; saved-output padding is zero. -/
def backwardInitial (padding : ℝ) : State (WithBot ℝ) where
  mem := fun r o ↦ if r = "Y" then .mk .real (some 0)
    else if r = "G" then .mk .fp16 (ofFloat .fp16
      (some (if o = 9 then 2 else if o = 11 then 4 else 99)))
    else .mk .bf16 (ofFloat .bf16 (some 7))
  regs := fun _ _ _ ↦ none
  pids := fun ax ↦ if ax = 0 ∨ ax = 1 then 1 else 0
  numPids := fun _ ↦ 2
  undef := fun d _ _ ↦ match d with
    | .real => some 0
    | .fp16 => ofFloat .fp16 (some padding)
    | other => defaultValue other

/-- Full backward execution for arbitrary real fp16 padding and both candidates. -/
def backward_run (v : Bool) (padding : ℝ) :
    {t : State (WithBot ℝ) // run (engine model)
      (typedBackward .fp64 .fp16 .bf16 v "Y" "G" "DX" 3 2 2 2 4).surfaceBody
        (backwardInitial padding) = some t} := by
  cases v
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 500000) [typedBackward, typedBackward_fp64_fp16, ComputeKernel.surfaceBody,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, bop, natLt, backwardInitial, Cell.read, ofFloat, toFloat,
      TileShape.axisDim, Region.cast, State.setReg]
    try rfl

/-- The unmasked reduction includes both unspecified gradient lanes. The formula
exposes their contribution rather than assuming a zero padding result. -/
theorem backward_cells (v : Bool) (padding : ℝ) :
    (backward_run v padding).val.mem "DX" 9 = Cell.mk .bf16 (some (-4 - 2 * padding)) ∧
    (backward_run v padding).val.mem "DX" 11 = Cell.mk .bf16 (some (-2 - 2 * padding)) ∧
    (backward_run v padding).val.mem "DX" 13 = Cell.mk .bf16 (some 7) ∧
    (backward_run v padding).val.mem "DX" 15 = Cell.mk .bf16 (some 7) := by
  cases v
  all_goals simp (maxSteps := 500000) [backward_run, backwardInitial, model,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat,
    Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim, TileShape.insertAxisIndex,
    TileShape.eraseAxis, sum_fin4,
    WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv, WithBot.realExp]
  all_goals norm_num [sum_fin4, WithBot.some_eq_coe, WithBot.none_eq_bot,
    ← WithBot.coe_add, ← WithBot.coe_mul, Option.map₂, Option.map, Option.bind]
  all_goals norm_num [Finset.sum_insert, Finset.sum_singleton,
    WithBot.some_eq_coe, ← WithBot.coe_add, Option.map₂, Option.map, Option.bind]
  all_goals simp only [← WithBot.coe_ofNat, ← WithBot.coe_add]
  all_goals repeat' constructor
  all_goals first | rfl | congr 2 <;> ring

/-- Concrete zero-padding and nonzero-padding executions differ at both active
outputs, while the same fully masked cells remain untouched. -/
theorem backward_zero_and_five (v : Bool) :
    (backward_run v 0).val.mem "DX" 9 = Cell.mk .bf16 (some (-4)) ∧
    (backward_run v 0).val.mem "DX" 11 = Cell.mk .bf16 (some (-2)) ∧
    (backward_run v 5).val.mem "DX" 9 = Cell.mk .bf16 (some (-14)) ∧
    (backward_run v 5).val.mem "DX" 11 = Cell.mk .bf16 (some (-12)) := by
  have h0 := backward_cells v 0
  have h5 := backward_cells v 5
  norm_num at h0 h5
  exact ⟨h0.1, h0.2.1, h5.1, h5.2.1⟩

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LogSoftmax.NumericExecution
