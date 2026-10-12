import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Meta.StatementAudit

/-! Program congruence must respect selected branches, missing operands and
range boundaries; a body that fails must not be silently discarded. -/
namespace ProgramRewriteTests
open VeriTile.Triton

private def initial : BlockState := default
private def missing : List Stmt := [.assign .nat [] "out" (.ref .nat [] "missing")]
private def skipped : List Stmt := [.ifThen (.constBool Bool.false) missing]

/-- An unselected failing arm leaves the state unchanged. -/
theorem skips_unselected_failure : stepStmts skipped initial = some initial := by
  rw [show skipped = [.ifThen (.constBool Bool.false) missing] from rfl,
    ProgramRewrite.constant_branch]
  simp only [Bool.false_eq_true, ↓reduceIte, List.nil_append, stepStmts.nil]

/-- Selecting that same arm really fails. -/
theorem preserves_selected_failure :
    stepStmts [.ifThen (.constBool Bool.true) missing] initial = none := by
  rw [ProgramRewrite.constant_branch]
  simp only [↓reduceIte, List.append_nil, missing, stepStmts, stepStmt, evalOp]
  rfl

/-- Three iterations preserve the skip semantics and still update the index. -/
theorem keeps_range_iterations :
    stepForRangeAux "i" 0 5 2 skipped initial =
      some (initial.setReg "i" .nat [] (Tile.scalar 4)) := by
  have h (s : BlockState) : stepStmts skipped s = stepStmts [] s := by
    simp only [skipped, ProgramRewrite.constant_branch, Bool.false_eq_true,
      ↓reduceIte, List.nil_append]
  rw [ProgramRewrite.range_congr _ _ _ _ _ [] h]
  rw [stepForRangeAux.step_lt (by decide) (by decide), stepStmts.nil, Option.bind_some]
  rw [stepForRangeAux.step_lt (by decide) (by decide), stepStmts.nil, Option.bind_some]
  rw [stepForRangeAux.step_lt (by decide) (by decide), stepStmts.nil, Option.bind_some]
  rw [stepForRangeAux.step_ge (by decide) (by decide)]
  simp only [BlockState.setReg, Option.some.injEq, BlockState.mk.injEq, true_and, and_true]
  funext dtype shape name
  by_cases hn : name = "i" <;> simp [hn]

/-- No body executes when the bounds are empty or the modeled stride is zero. -/
theorem keeps_empty_and_zero_stride :
    stepForRangeAux "i" 3 3 1 missing initial = some initial ∧
      stepForRangeAux "i" 0 3 0 missing initial = some initial := by
  constructor
  · exact stepForRangeAux.step_ge (by decide) (by decide)
  · exact stepForRangeAux.step_zero_step

/-- Even an empty body cannot hide a missing dynamic bound. -/
theorem missing_dynamic_bound_fails :
    stepStmt (.forRangeDyn "i" (.constNat 0) (.ref .nat [] "absent")
      (.constNat 1) skipped) initial = none := by
  simp only [stepForRangeAux.forRangeDyn_unfold, evalOp, Option.bind_some]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end ProgramRewriteTests
