import bench.tritonbench_g.kldiv_compute.Kernels
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KldivCompute.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
noncomputable section
open scoped Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

/-- An exact-real numerical witness, not an IEEE rounding implementation. -/
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
    | _ => some 0
  cast := fun _ _ _ x ↦ x
  fromNat := fun _ n ↦ some n
  fromInt := fun _ n ↦ some n
  fp32Bits := fun _ ↦ some 0
  fp32Load := id
  reduceMax := fun _ {_} ax keep v ↦
    ((Tile.reduceMax ax keep ⟨v⟩).getD ⟨fun _ ↦ none⟩).data
  reduceSum := fun _ {_} ax keep v ↦ (Tile.reduceSum ax keep ⟨v⟩).data

/-- Nonzero program id, active source at offset two, and untouched output padding. -/
def initial : State (WithBot ℝ) where
  mem := fun r _ ↦
    if r = "X" then .mk .real (some 4)
    else if r = "Y" then .mk .fp16 (some 2)
    else .mk .bf16 (some 7)
  regs := fun _ _ _ ↦ none
  pids := fun ax ↦ if ax = 0 then 1 else 0
  numPids := fun _ ↦ 2
  undef := fun d _ _ ↦ defaultValue d

/-- Both complete mixed-storage programs have an actual
successful nonempty numerical execution; no output property is assumed. -/
def source_run (v : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (kernel .fp64 .fp16 .bf16 v "X" "Y" "O" 3 2).surfaceBody initial = some t} := by
  cases v
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 500000) [kernel, kernel_fp64_fp16, ComputeKernel.surfaceBody,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, bop, natLt, initial, Cell.read,
      ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg]
    try rfl

/-- The active lane stores the KL term; the masked neighbor retains its tagged sentinel. -/
theorem output_cells (v : Bool) :
    (source_run v).val.mem "O" 2 = Cell.mk .bf16 (some (4 * Real.log 2)) ∧
    (source_run v).val.mem "O" 3 = Cell.mk .bf16 (some 7) := by
  cases v
  all_goals simp [source_run, initial, model, Region.cast,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat,
    WithBot.realDiv, WithBot.realLog, WithBot.realMul]
  all_goals norm_num
  all_goals exact ⟨rfl, rfl⟩

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivCompute.NumericExecution
