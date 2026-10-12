import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Branch decomposition must retain the guard on the selected arm's actual
operands. An inactive arm needs no numerical guard; an active arm does. -/
namespace FPBranchRewriteTests
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

private def x : Op .real [] := .ref .real [] "x"
private def one : Op .real [] := .const 1
private def padding : ComputeStmt := .assign .nat [] "padding" (.alg (.constNat 17))
private def condition : Op .bool [] := .ref .bool [] "flag"
private def body (swapped : Bool) : List ComputeStmt :=
  [padding, .assign .real [] "x" (.alg
    (if swapped then .add .real .nil one x else .add .real .nil x one))]
private def inner : Site := .add [padding] [] x one

private def program (selected swapped : Bool) : Program where
  kernel := .mk [] [] [padding, .ifThenElse condition
    (if selected then body swapped else [padding])
    (if selected then [padding] else body swapped), padding]
  domain := [.branch [padding] condition selected inner]

theorem then_arm (R : FP.ScalarArithmetic.Rules) :
    program true false ≡[R] program true true := by
  equiv_decompose
  all_goals fp_prove

theorem else_arm (R : FP.ScalarArithmetic.Rules) :
    program false false ≡[R] program false true := by
  equiv_decompose
  all_goals fp_prove

/-- info: FP assumptions used by then_arm:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions then_arm

private def nestedProgram (swapped : Bool) : Program where
  kernel := .mk [] [] [.ifThenElse condition
    [.ifThenElse condition (body swapped) []] []]
  domain := [.branch [] condition true (.branch [] condition true inner)]

theorem nested_arms (R : FP.ScalarArithmetic.Rules) :
    nestedProgram false ≡[R] nestedProgram true := by
  equiv_decompose
  all_goals fp_prove

-- An absent guard, the wrong arm, or a guard before a different prefix cannot
-- justify the rewrite. Keep matching signatures so these test the proof path.
set_option linter.unusedVariables false in
example (R : FP.ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : ({ program true false with domain := [] } : Program) ≡[R]
        ({ program true true with domain := [] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  fail_if_success
    have : ({ program true false with domain := [.branch [padding] condition false inner] } : Program) ≡[R]
        ({ program true true with domain := [.branch [padding] condition false inner] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  fail_if_success
    have : ({ program true false with domain := [.branch [] condition true inner] } : Program) ≡[R]
        ({ program true true with domain := [.branch [] condition true inner] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  trivial

theorem inactive_arm_needs_no_guard {α : Type} [Inhabited α] (M : Algebra α)
    (D : FP.Guarded.Domain α) (s : State α) (site : Site) :
    (Site.branch [] (.constBool false) true site).Holds M D s := by
  intro t ht value hv hselected
  have he : value = fun _ => false := Option.some.inj hv.symm
  simp [he] at hselected

theorem active_arm_checks_operands {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) :
    ¬ (Site.branch [] (.constBool true) true
      (.add [] [] (.const 1) (.const 2))).Holds M (fun _ _ => False) s := by
  intro hd
  have hinner := hd s (by simp [run]) (fun _ => true) rfl rfl
  have hvalues := hinner s (by simp [run])
    (fun _ => (engine M).literal none .real 1)
    (fun _ => (engine M).literal none .real 2) rfl rfl
  exact hvalues.1 PUnit.unit

theorem missing_condition_preserves_failure {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (yes no : List ComputeStmt) (h : s.regs .bool [] "flag" = none) :
    step (engine M) (.ifThenElse condition yes no) s = none := by
  simp [step, condition, FP.Structural.evalOp, h]

theorem inactive_body_is_not_evaluated {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (yes : List ComputeStmt) :
    step (engine M) (.ifThenElse (.constBool false) yes []) s = some s := by
  simp [step, FP.Structural.evalOp, run]

private def singleArmProgram (swapped : Bool) : Program where
  kernel := .mk [] [] [padding, .ifThen condition (body swapped), padding]
  domain := [.branch [padding] condition true inner]

theorem one_arm (R : FP.ScalarArithmetic.Rules) :
    singleArmProgram false ≡[R] singleArmProgram true := by
  equiv_decompose
  all_goals fp_prove

private def loopedSingleArm (swapped : Bool) : Program where
  kernel := .mk [] [] [padding, .ifThen condition
    [.forRangeDyn "i" (.constNat 0) (.constNat 3) (.constNat 1) (body swapped)], padding]
  domain := [.branch [padding] condition true
    (.range [] "i" (.constNat 0) (.constNat 3) (.constNat 1) (body false) inner)]

theorem one_arm_with_loop (R : FP.ScalarArithmetic.Rules) :
    loopedSingleArm false ≡[R] loopedSingleArm true := by
  equiv_decompose
  all_goals fp_prove

set_option linter.unusedVariables false in
example (R : FP.ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : ({ singleArmProgram false with domain := [] } : Program) ≡[R]
        ({ singleArmProgram true with domain := [] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  fail_if_success
    have : ({ singleArmProgram false with domain := [.branch [padding] condition false inner] } : Program) ≡[R]
        ({ singleArmProgram true with domain := [.branch [padding] condition false inner] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  trivial

theorem one_arm_missing_condition_preserves_failure {α : Type} [Inhabited α]
    (M : Algebra α) (s : State α) (yes : List ComputeStmt)
    (h : s.regs .bool [] "flag" = none) :
    step (engine M) (.ifThen condition yes) s = none := by
  simp [step, condition, FP.Structural.evalOp, h]

theorem one_arm_inactive_body_is_not_evaluated {α : Type} [Inhabited α]
    (M : Algebra α) (s : State α) (yes : List ComputeStmt) :
    step (engine M) (.ifThen (.constBool false) yes) s = some s := by
  simp [step, FP.Structural.evalOp]

theorem one_arm_source_found :
    (Source.conditional? (singleArmProgram false).kernel.surfaceBody
      (fun _ => Bool.true)).map (fun b => (b.before,b.condition,b.yes,b.no,b.after)) =
      some ([padding], condition, body false, [], [padding]) := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end FPBranchRewriteTests
