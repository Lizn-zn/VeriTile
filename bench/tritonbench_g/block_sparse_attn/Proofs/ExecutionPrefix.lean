import bench.tritonbench_g.block_sparse_attn.Proofs.FP
import bench.tritonbench_g.attention_fwd_triton3.Proofs.FPModel

/-! Shared executed prefix for the complete-source FP witness.
The bf16 fixture uses batch/head/layout 1, two CSR visits, independent row/col
strides, one/two channel blocks and both query/key mask flags. The complete loop and output checks are assembled in TypedExecution.
Opaque dots return zero and reductions return one. These are execution/domain
witnesses, not GPU results or the independent real attention formula. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open AttentionFwdTriton3.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def initial : State ℝ where
  mem := fun r o => if r = "Rows" then .mk .int (if o = 6 then 1 else if o = 7 then 3 else -9)
    else if r = "Cols" then .mk .int (if o = 14 then 0 else if o = 15 then 1 else -9)
    else .mk .bf16 (if r = "Out" then 9 else 2)
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 1 else 5
  numPids := fun _ => 8
  undef := fun d _ _ => defaultValue d

def original (em en two : Bool) := FPEquivProofs.original .bf16
  "Out" "Q" "K" "V" ⟨"Rows"⟩ ⟨"Cols"⟩ 5 13 2 3
  100 20 4 80 16 4 80 16 4 200 40 6 4 2 3 2 2 2 (if two then 2 else 1) em en
def candidate (em en two : Bool) := FPEquivProofs.scalarLeft .bf16
  "Out" "Q" "K" "V" ⟨"Rows"⟩ ⟨"Cols"⟩ 5 13 2 3
  100 20 4 80 16 4 80 16 4 200 40 6 4 2 3 2 2 2 (if two then 2 else 1) em en
def program (em en two swapped : Bool) :=
  if swapped then candidate em en two else original em en two

def code (em en two swapped : Bool) := (program em en two swapped).kernel.surfaceBody
def slices (em en two swapped : Bool) : Source.IntRange :=
  (Source.intRange? (code em en two swapped) "col_idx_idx").getD
    ⟨[], .constInt 0, .constInt 0, .constNat 0, [], []⟩

def preloop (em two : Bool) : {t : State ℝ //
    run (engine model) (slices em Bool.false two Bool.false).before initial = some t} := by
  cases em <;> cases two
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [slices, code, program, original, candidate, FPEquivProofs.original,
      FPEquivProofs.scalarLeft, TypedSource.original, TypedSource.scalarLeft,
      TypedSource.specialize, ComputeKernel.surfaceBody, Source.intRange?,
      Option.map, Option.getD, Bool.false_eq_true, ↓reduceIte, beq_self_eq_true]
    simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, ComputeDType.eraseDType, numeric, integralFloorDiv,
      integralMod, numericLt, numericLe, numericEq, natLt, bop, initial,
      State.setReg, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast]
    try rfl

end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.TypedExecution
