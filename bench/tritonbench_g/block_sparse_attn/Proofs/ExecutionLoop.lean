import bench.tritonbench_g.block_sparse_attn.Proofs.ExecutionPrefix

/-! Executed CSR iterations and stores for the abstract FP witness.
Split the actual body at its named row-maximum assignment. Intermediate
states are shared by the next stage; no body execution is assumed. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open AttentionFwdTriton3.FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def score (em en two : Bool) : List ComputeStmt :=
  Source.prefixBefore (slices em en two Bool.false).body "m_ij"

def firstScore (em en two : Bool) : {t : State ℝ //
    run (engine model) (score em en two)
      ((preloop em two).val.setReg "col_idx_idx" .int [] (fun _ => 1)) = some t} := by
  cases em <;> cases en <;> cases two
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [score, slices, code, program, original, candidate, FPEquivProofs.original,
      FPEquivProofs.scalarLeft, TypedSource.original, TypedSource.scalarLeft,
      TypedSource.specialize, ComputeKernel.surfaceBody, Source.intRange?, Source.prefixBefore,
      Option.map, Option.getD, Bool.false_eq_true, ↓reduceIte, beq_self_eq_true]
    simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, ComputeDType.eraseDType, numeric, integralFloorDiv,
      integralMod, numericLt, numericLe, numericEq, natLt, bop, initial, preloop,
      Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast]
    try rfl

def rest (em en two : Bool) : List ComputeStmt :=
  (slices em en two Bool.false).body.drop (score em en two).length

def firstBody (em en two : Bool) : {t : State ℝ //
    run (engine model) (rest em en two) (firstScore em en two).val = some t} := by
  cases em <;> cases en <;> cases two
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [rest, score, slices, code, program, original, candidate, FPEquivProofs.original,
      FPEquivProofs.scalarLeft, TypedSource.original, TypedSource.scalarLeft,
      TypedSource.specialize, ComputeKernel.surfaceBody, Source.intRange?, Source.prefixBefore,
      Option.map, Option.getD, Bool.false_eq_true, ↓reduceIte, beq_self_eq_true,
      List.length, List.drop]
    simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, ComputeDType.eraseDType, numeric, integralFloorDiv,
      integralMod, numericLt, numericLe, numericEq, natLt, bop, initial, preloop,
      Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, firstScore]
    try rfl

def secondScore (em en two : Bool) : {t : State ℝ //
    run (engine model) (score em en two) ((firstBody em en two).val.setReg "col_idx_idx" .int [] (fun _ => 2)) = some t} := by
  cases em <;> cases en <;> cases two
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [rest, score, slices, code, program, original, candidate, FPEquivProofs.original,
      FPEquivProofs.scalarLeft, TypedSource.original, TypedSource.scalarLeft,
      TypedSource.specialize, ComputeKernel.surfaceBody, Source.intRange?, Source.prefixBefore,
      Option.map, Option.getD, Bool.false_eq_true, ↓reduceIte, beq_self_eq_true,
      List.length, List.drop]
    simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, ComputeDType.eraseDType, numeric, integralFloorDiv,
      integralMod, numericLt, numericLe, numericEq, natLt, bop, initial, preloop,
      Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, firstScore, firstBody]
    try rfl

def secondBody (em en two : Bool) : {t : State ℝ //
    run (engine model) (rest em en two) (secondScore em en two).val = some t} := by
  cases em <;> cases en <;> cases two
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [rest, score, slices, code, program, original, candidate, FPEquivProofs.original,
      FPEquivProofs.scalarLeft, TypedSource.original, TypedSource.scalarLeft,
      TypedSource.specialize, ComputeKernel.surfaceBody, Source.intRange?, Source.prefixBefore,
      Option.map, Option.getD, Bool.false_eq_true, ↓reduceIte, beq_self_eq_true,
      List.length, List.drop]
    simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, ComputeDType.eraseDType, numeric, integralFloorDiv,
      integralMod, numericLt, numericLe, numericEq, natLt, bop, initial, preloop,
      Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, firstScore, firstBody, secondScore]
    try rfl

def finish (em en two : Bool) : {t : State ℝ //
    run (engine model) ((slices em en two Bool.false).after) (secondBody em en two).val = some t} := by
  cases em <;> cases en <;> cases two
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp only [rest, score, slices, code, program, original, candidate, FPEquivProofs.original,
      FPEquivProofs.scalarLeft, TypedSource.original, TypedSource.scalarLeft,
      TypedSource.specialize, ComputeKernel.surfaceBody, Source.intRange?, Source.prefixBefore,
      Option.map, Option.getD, Bool.false_eq_true, ↓reduceIte, beq_self_eq_true,
      List.length, List.drop]
    simp (config := { maxSteps := 100000 }) [run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, ComputeDType.eraseDType, numeric, integralFloorDiv,
      integralMod, numericLt, numericLe, numericEq, natLt, bop, initial, preloop,
      Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast, firstScore, firstBody, secondScore, secondBody, FP.Structural.store]
    try rfl

end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.TypedExecution
