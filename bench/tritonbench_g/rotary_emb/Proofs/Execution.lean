import bench.tritonbench_g.rotary_emb.Kernels
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FullSource
noncomputable section
open scoped Classical
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

/-- Exact-real arithmetic witness backend, not an IEEE rounding model. -/
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
  unary := fun _ _ _ ↦ some 0
  cast := fun _ _ _ x ↦ x
  fromNat := fun _ n ↦ some n
  fromInt := fun _ n ↦ some n
  fp32Bits := fun _ ↦ some 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ ↦ fun _ ↦ none
  reduceSum := fun _ {_} _ _ _ ↦ fun _ ↦ some 0

def payload (r : RegionName) (o : Nat) : ℝ :=
  if r="Q" then
    if o∈[220,224] then 1 else if o∈[222,226] then 2 else 7
  else if r="K" then
    if o∈[440,446,460,466] then 4 else if o∈[443,449,463,469] then 5 else 7
  else if r="Cos" then
    if o∈[100,108] then 3 else if o∈[104,112] then 7 else 17
  else if r="Sin" then
    if o∈[100,108] then 5 else if o∈[104,112] then 11 else 19
  else 7

def initial (flip : Bool) : State (WithBot ℝ) where
  mem := fun r o ↦
    if (r="Q" ∧ ¬flip) ∨ (r="K" ∧ flip) then .mk .fp16 (ofFloat .fp16 (some (payload r o)))
    else if (r="Cos" ∧ ¬flip) ∨ (r="Sin" ∧ flip) then .mk .bf16 (ofFloat .bf16 (some (payload r o)))
    else .mk .real (some (payload r o))
  regs := fun _ _ _ ↦ none
  pids := fun ax ↦ if ax=0 ∨ ax=1 then 1 else 0
  numPids := fun _ ↦ 2
  undef := fun d _ _ ↦ defaultValue d

def program (v flip : Bool) : ComputeKernel :=
  kernel (if flip then .fp64 else .fp16) (if flip then .fp16 else .fp64)
    (if flip then .fp32 else .bf16) (if flip then .bf16 else .fp32) v
    "Q" "K" "Cos" "Sin" 100 10 2 200 20 3 50 4 999 983 3 3 4 2 2 4

/-- Two sequences, two heads, two pairs, nonzero program IDs, independent
nonunit strides and both sequence/head padding are present in the full run. -/
def numeric_run (v flip : Bool) :
    {t : State (WithBot ℝ) // run (engine model) (program v flip).surfaceBody
      (initial flip) = some t} := by
  cases v <;> cases flip
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 1000000) [program, kernel, original, template,
      template_fp32_fp32, template_fp32_fp64, template_fp32_fp16, template_fp32_bf16, template_fp32_f8e4, template_fp32_f8e5, template_fp64_fp32, template_fp64_fp64, template_fp64_fp16, template_fp64_bf16, template_fp64_f8e4, template_fp64_f8e5, template_fp16_fp32, template_fp16_fp64, template_fp16_fp16, template_fp16_bf16, template_fp16_f8e4, template_fp16_f8e5, template_bf16_fp32, template_bf16_fp64, template_bf16_fp16, template_bf16_bf16, template_bf16_f8e4, template_bf16_f8e5, template_f8e4_fp32, template_f8e4_fp64, template_f8e4_fp16, template_f8e4_bf16, template_f8e4_f8e4, template_f8e4_f8e5, template_f8e5_fp32, template_f8e5_fp64, template_f8e5_fp16, template_f8e5_bf16, template_f8e5_f8e4, template_f8e5_f8e5,
      ComputeKernel.surfaceBody,
      ComputeKernel.inputs, ComputeKernel.outputs, run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas, Option.bind, engine,
      Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
      numeric, bop, natLt, initial, Cell.read, ofFloat, toFloat, Region.cast, State.setReg,
      FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ]
    try rfl

def qCell (flip : Bool) (x : ℝ) : Cell (WithBot ℝ) :=
  if flip then .mk .real (some x) else .mk .fp16 (ofFloat .fp16 (some x))

def kCell (flip : Bool) (x : ℝ) : Cell (WithBot ℝ) :=
  if flip then .mk .fp16 (ofFloat .fp16 (some x)) else .mk .real (some x)

/-- Odd trigonometric values differ from the even values: the Q odd result is
25, not the 11 produced by reusing the even trigonometric inputs. All four store
families execute, including the second K head; tail and stride-gap cells persist. -/
theorem numeric_cells (v flip : Bool) :
    (numeric_run v flip).val.mem "Q" 220 = qCell flip (-7) ∧
    (numeric_run v flip).val.mem "Q" 222 = qCell flip 25 ∧
    (numeric_run v flip).val.mem "Q" 224 = qCell flip (-7) ∧
    (numeric_run v flip).val.mem "K" 440 = kCell flip (-13) ∧
    (numeric_run v flip).val.mem "K" 443 = kCell flip 79 ∧
    (numeric_run v flip).val.mem "K" 469 = kCell flip 79 ∧
    (numeric_run v flip).val.mem "Q" 230 = qCell flip 7 ∧
    (numeric_run v flip).val.mem "Q" 320 = qCell flip 7 ∧
    (numeric_run v flip).val.mem "Q" 221 = qCell flip 7 ∧
    (numeric_run v flip).val.mem "Sin" 104 = (initial flip).mem "Sin" 104 := by
  cases v <;> cases flip
  all_goals simp (maxSteps := 1000000) [numeric_run, initial, model, payload, qCell, kCell,
    FP.Structural.store, State.write, State.setReg, Region.cast, Tile.scalar, -Tile.scalar_eta,
    Function.comp_def, TileShape.allIndices, List.finRange_succ, List.foldl, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat,
    WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv]
  all_goals norm_num [WithBot.some_eq_coe, ← WithBot.coe_add, Option.map₂, Option.map,
    Option.bind, bop, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar, -Tile.scalar_eta,
    Function.comp_def, List.finRange_succ, List.foldl, List.flatMap]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.NumericExecution
