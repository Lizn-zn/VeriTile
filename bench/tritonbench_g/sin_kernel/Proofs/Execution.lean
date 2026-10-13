import bench.tritonbench_g.sin_kernel.Proofs.Precision
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Semantics.TileOps

namespace VeriTile.Bench.TritonBenchG.SinKernel.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
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
    | .sin => WithBot.realSin a
    | _ => some 0
  cast := fun _ _ _ x ↦ x
  fromNat := fun _ n ↦ some n
  fromInt := fun _ n ↦ some n
  fp32Bits := fun _ ↦ some 0
  fp32Load := id
  reduceMax := fun _ {_} ax keep v ↦
    ((Tile.reduceMax ax keep ⟨v⟩).getD ⟨fun _ ↦ none⟩).data
  reduceSum := fun _ {_} ax keep v ↦ (Tile.reduceSum ax keep ⟨v⟩).data

def outputFloat : InputStorage → FloatDType
  | .fp32 | .fp64 => .real

/-- Nonzero program id, active source at offset two, and untouched output padding. -/
def initial (input : InputStorage) : State (WithBot ℝ) where
  mem := fun r _ ↦
    if r = "X" then .mk .real (some 1)
    else .mk (outputFloat input).toTileDType (ofFloat (outputFloat input) (some 7))
  regs := fun _ _ _ ↦ none
  pids := fun ax ↦ if ax = 0 then 1 else 0
  numPids := fun _ ↦ 2
  undef := fun d _ _ ↦ defaultValue d

/-- Every supported storage profile and both complete programs have an actual
successful nonempty numerical execution; no output property is assumed. -/
def source_run (input : InputStorage) (v : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (typedKernel input v "X" "Y" 3 2).surfaceBody (initial input) = some t} := by
  cases input <;> cases v
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 500000) [typedKernel, ComputeKernel.surfaceBody,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, bop, natLt, initial, outputFloat, Cell.read,
      ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg]
    try rfl

/-- The actual sine value is stored and the masked neighboring cell is unchanged. -/
theorem output_cells (input : InputStorage) (v : Bool) :
    (source_run input v).val.mem "Y" 2 =
      Cell.mk (outputFloat input).toTileDType
        (ofFloat (outputFloat input) (some (Real.sin 1))) ∧
    (source_run input v).val.mem "Y" 3 =
      Cell.mk (outputFloat input).toTileDType
        (ofFloat (outputFloat input) (some 7)) := by
  cases input <;> cases v
  all_goals simp [source_run, initial, model, outputFloat, WithBot.realSin,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat]
  all_goals first | rfl | exact ⟨rfl, rfl⟩

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SinKernel.NumericExecution
