import VeriTile.Triton.Float.BroadcastRewrite
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

namespace FPBroadcastRewriteTests
open VeriTile Triton FP FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

private def numerator : Op .real [2, 3, 4] := .ref .real [2, 3, 4] "a"
private def denominator : Op .real [2, 1, 4] := .ref .real [2, 1, 4] "b"
private def bc : Broadcast [2, 3, 4] [2, 1, 4] [2, 3, 4] :=
  .consSame (.consR (.consSame .nil))
private def commonPrefix : List ComputeStmt := [
  .assign .nat [] "padding" (.alg (.constNat 17)),
  .assign .real [2, 3, 4] "a" (.alg (.full [2, 3, 4] (.const 2))),
  .assign .real [2, 1, 4] "b" (.alg (.full [2, 1, 4] (.const 3)))]
private def assignment (precision : ComputeDType) (optimized : Bool) (one : ℝ) : ComputeStmt :=
  -- Both precision alternatives erase to the real carrier.
  match precision with
  | .fp64 => .assign .real [2, 3, 4] "out" (.compute (.alg .fp64
      (if optimized then .mul .real bc numerator (.div .real .scalarL (.const one) denominator)
       else .div .real bc numerator denominator)))
  | _ => .assign .real [2, 3, 4] "out" (.compute (.alg .fp32
      (if optimized then .mul .real bc numerator (.div .real .scalarL (.const one) denominator)
       else .div .real bc numerator denominator)))
private def code (precision : ComputeDType) (optimized : Bool) (one : ℝ) : List ComputeStmt :=
  commonPrefix ++ [assignment precision optimized one,
    .assign .real [2, 3, 4] "copied" (.alg (.ref .real [2, 3, 4] "out"))]
private def site : Site := .divBroadcast
  (Source.prefixBefore (code .fp32 Bool.false 1) "out") [2, 3, 4] 2 [1, 4] numerator denominator
private def program (precision : ComputeDType) (optimized : Bool) (one : ℝ) : Program :=
  ⟨.mk [] [] (code precision optimized one), [site]⟩

-- Rank-three, non-scalar broadcasting retains the prefix and suffix. Both
-- integer and decimal spellings of the reciprocal numerator are supported.
theorem rank_three_broadcast (R : ScalarArithmetic.Rules) :
    program .fp32 Bool.false 1 ≡[R] program .fp32 Bool.true 1 := by
  equiv_decompose
  all_goals fp_prove

theorem decimal_reciprocal (R : ScalarArithmetic.Rules) :
    program .fp32 Bool.false 1 ≡[R] program .fp32 Bool.true 1.0 := by
  equiv_decompose
  all_goals fp_prove

/-- info: FP assumptions used by decimal_reciprocal:
---
info:   div_mul_rcp -/
#guard_msgs in
#print_fp_assumptions decimal_reciprocal
#guard_msgs (drop info) in
#auditModuleAxioms

set_option linter.unusedVariables false

example (R : ScalarArithmetic.Rules) : True := by
  -- Dropping the intermediate guards cannot admit the rewrite.
  fail_if_success
    have : ({ program .fp32 Bool.false 1 with domain := [] } : Program) ≡[R]
        ({ program .fp32 Bool.true 1 with domain := [] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  -- Checks made before the operands are assigned cannot stand in for the site.
  fail_if_success
    have : ({ program .fp32 Bool.false 1 with domain :=
          [.divBroadcast [] [2, 3, 4] 2 [1, 4] numerator denominator] } : Program) ≡[R]
        ({ program .fp32 Bool.true 1 with domain :=
          [.divBroadcast [] [2, 3, 4] 2 [1, 4] numerator denominator] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  -- A fp32 atom cannot authorize fp64 arithmetic or a different numerator.
  fail_if_success
    have : program .fp64 Bool.false 1 ≡[R] program .fp64 Bool.true 1 := by
      equiv_decompose
      all_goals fp_prove
  fail_if_success
    have : program .fp32 Bool.false 1 ≡[R] program .fp32 Bool.true 2 := by
      equiv_decompose
      all_goals fp_prove
  trivial

example : True := by
  fail_if_success
    have : program .fp32 Bool.false 1 ≡[([] : Spec.Assumptions GuardedFragment)]
        program .fp32 Bool.true 1 := by
      equiv_decompose
      all_goals fp_prove
  trivial

end FPBroadcastRewriteTests
