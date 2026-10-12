import bench.tritonbench_g.attention_fwd_triton1.Proofs.FPModel

/-! Successful executions of the complete fp32 source, including both loop
iterations and all STORE/IFCOND settings. The same abstract backend satisfies
the admitted scalar rules and all actual iteration guards. This checks that
the equivalence has successful nonempty instances; it is neither a GPU launch
nor evidence for a numerical atom. Correct.lean supplies the independent real
contracts; RealExecution.lean distinguishes the nonzero pre-update snapshots. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.TypedExecution
open VeriTile.Triton FP.Structural
open FPModel
noncomputable section

private def initial : State ℝ where
  mem := fun _ _ => .mk .real 17
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

private def completed (STORE : Bool) (t : State ℝ) : Prop :=
  (t.mem "o" 0).read .real = 0 ∧
  (t.mem "o" 1).read .real = 0 ∧
  (t.mem "h" 0).read .real = (if STORE then 0 else 17) ∧
  (t.mem "h" 1).read .real = (if STORE then 0 else 17) ∧
  (t.mem "q" 0).read .real = 17 ∧
  (t.mem "o" 2).read .real = 17

set_option maxHeartbeats 3200000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false

theorem original_all_flags_execute (STORE IFCOND : Bool) :
    ∃ t, FP.Structural.exec (model.withDefaultPrecision .fp32)
      (TypedSource.original "q" "k" "v" "h" "o" 2 1 1 2 1 2 1 1 1 2 STORE IFCOND)
      initial = some t ∧ completed STORE t := by
  cases STORE <;> cases IFCOND
  all_goals
    simp only [TypedSource.original, FP.Structural.exec, run, step,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets,
      model, Algebra.withDefaultPrecision, resolvePrecision, initial,
      numeric, bind, pure, Option.bind_some, Option.map_some]
    simp (config := { maxSteps := 300000 }) [run, step, range,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets,
      model, Algebra.withDefaultPrecision, resolvePrecision, initial, completed,
      ComputeDType.eraseDType, numeric, numericEq, bop, store, ofFloat, toFloat,
      State.write_same, State.write_other, Cell.read, TileShape.allIndices,
      TileShape.indexToList, List.finRange_succ, BlockPtr.inBounds_nil_checkedAxes]
    simp [State.write, State.setReg, FP.Structural.bop, Cell.read]

theorem candidate_all_flags_execute (STORE IFCOND : Bool) :
    ∃ t, FP.Structural.exec (model.withDefaultPrecision .fp32)
      (TypedSource.scalarLeft "q" "k" "v" "h" "o" 2 1 1 2 1 2 1 1 1 2 STORE IFCOND)
      initial = some t ∧ completed STORE t := by
  cases STORE <;> cases IFCOND
  all_goals
    simp only [TypedSource.scalarLeft, FP.Structural.exec, run, step,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets,
      model, Algebra.withDefaultPrecision, resolvePrecision, initial,
      numeric, bind, pure, Option.bind_some, Option.map_some]
    simp (config := { maxSteps := 300000 }) [run, step, range,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets,
      model, Algebra.withDefaultPrecision, resolvePrecision, initial, completed,
      ComputeDType.eraseDType, numeric, numericEq, bop, store, ofFloat, toFloat,
      State.write_same, State.write_other, Cell.read, TileShape.allIndices,
      TileShape.indexToList, List.finRange_succ, BlockPtr.inBounds_nil_checkedAxes]
    simp [State.write, State.setReg, FP.Structural.bop, Cell.read]

/-- Two active loop iterations under a model of the selected rules, with
nonempty query data, both output writes and the optional H snapshots. -/
theorem nonempty_admitted_execution (R : FP.ScalarArithmetic.Rules) (STORE IFCOND : Bool) :
    FP.Guarded.Models R.assumptions model domain ∧
    (∀ site ∈ FPEquivProofs.requirements "q" "k" "v" "h" "o"
      2 1 1 2 1 2 1 1 1 2 STORE IFCOND, site.Holds model domain initial) ∧
    (∃ t, FP.Structural.exec (model.withDefaultPrecision .fp32)
      (TypedSource.original "q" "k" "v" "h" "o" 2 1 1 2 1 2 1 1 1 2 STORE IFCOND)
      initial = some t ∧ completed STORE t) ∧
    (∃ t, FP.Structural.exec (model.withDefaultPrecision .fp32)
      (TypedSource.scalarLeft "q" "k" "v" "h" "o" 2 1 1 2 1 2 1 1 1 2 STORE IFCOND)
      initial = some t ∧ completed STORE t) :=
  ⟨model_satisfies_rules R,
    requirements_hold "q" "k" "v" "h" "o" 2 1 1 2 1 2 1 1 1 2 STORE IFCOND initial,
    original_all_flags_execute STORE IFCOND, candidate_all_flags_execute STORE IFCOND⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.TypedExecution
