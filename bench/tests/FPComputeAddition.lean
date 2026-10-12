import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Explicit compute precision and intermediate guards are part of the
commutation contract. The reduction stays inside the original fp32 context. -/
namespace FPComputeAdditionTests
open VeriTile Triton FP FP.Structural FP.ProfiledRewrite
open GuardedRewrite (sameShape)
open scoped VeriTile.Spec

private def sum : Op .real [2] :=
  .reduceSum ⟨1, by simp⟩ Bool.false (.ref .real [2, 3] "p")
private def carry : Op .real [2] := .ref .real [2] "carry"
private def before : List ComputeStmt := [
  .assign .real [2, 3] "p" (.compute (.alg .fp32 (.full [2, 3] (.const 2)))),
  .assign .real [2] "carry" (.compute (.alg .fp32 (.full [2] (.const 3))))]
private def expr (swap : Bool) : Op .real [2] :=
  if swap then .add .real (sameShape [2]) carry sum
  else .add .real (sameShape [2]) sum carry
private def assignment (wide swap : Bool) : ComputeStmt :=
  if wide then .assign .real [2] "out" (.compute (.alg .fp64 (expr swap)))
  else .assign .real [2] "out" (.compute (.alg .fp32 (expr swap)))
private def domain : List Site := [.addFP32 before [2] sum carry]
private def program (wide swap : Bool) : Program where
  kernel := .mk [] [] (before ++ [assignment wide swap,
    .assign .real [2] "copied" (.alg (.ref .real [2] "out"))])
  domain := domain

theorem reduce_then_add (R : ScalarArithmetic.Rules) :
    program Bool.false Bool.false ≡[R] program Bool.false Bool.true := by
  equiv_decompose
  all_goals fp_prove

/-- info: FP assumptions used by reduce_then_add:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions reduce_then_add

set_option linter.unusedVariables false
example (R : ScalarArithmetic.Rules) : True := by
  -- Guarding the input tile cannot replace guarding the computed reduction.
  fail_if_success
    have : ({program Bool.false Bool.false with domain := []} : Program) ≡[R]
        ({program Bool.false Bool.true with domain := []} : Program) := by
      equiv_decompose
      all_goals fp_prove
  -- A check before the prefix inspects a different state.
  fail_if_success
    have : ({program Bool.false Bool.false with domain := [.addFP32 [] [2] sum carry]} : Program) ≡[R]
        ({program Bool.false Bool.true with domain := [.addFP32 [] [2] sum carry]} : Program) := by
      equiv_decompose
      all_goals fp_prove
  -- An fp32 atom supplies no law for fp64 addition.
  fail_if_success
    have : program Bool.true Bool.false ≡[R] program Bool.true Bool.true := by
      equiv_decompose
      all_goals fp_prove
  trivial

example : True := by
  fail_if_success
    have : program Bool.false Bool.false ≡[([] : Spec.Assumptions GuardedFragment)]
        program Bool.false Bool.true := by
      equiv_decompose
      all_goals fp_prove
  trivial

#guard_msgs (drop info) in
#auditModuleAxioms
end FPComputeAdditionTests
