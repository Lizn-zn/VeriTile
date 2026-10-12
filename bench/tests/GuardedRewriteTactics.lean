import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

namespace GuardedRewriteTacticTests
open VeriTile Triton FP FP.Structural FP.GuardedRewrite
open scoped VeriTile.Spec

private def readX : ComputeStmt := .assign .real [4] "x"
  (.alg (.load .real (.region "x" (.arange 4)) .none))
private def readY : ComputeStmt := .assign .real [4] "y"
  (.alg (.load .real (.region "y" (.arange 4)) .none))
private def padding : ComputeStmt := .assign .nat [] "unrelated" (.alg (.constNat 17))
private def operand (name : RegName) : Op .real [4] := .ref .real [4] name
private def add (swapped : Bool) : ComputeStmt :=
  assignment .add "out" (operand (if swapped then "y" else "x"))
    (operand (if swapped then "x" else "y"))
private def mul (swapped : Bool) : ComputeStmt :=
  assignment .mul "out" (operand (if swapped then "y" else "out"))
    (operand (if swapped then "out" else "y"))
private def write : ComputeStmt := .store .real [4] (.region "out" (.arange 4))
  (.alg (operand "out")) .none
private def code (extra : List ComputeStmt) (a b : Bool) : List ComputeStmt :=
  extra ++ [readX, readY, add false, add a, mul b, write]
private def sites (extra : List ComputeStmt) : List Site :=
  [⟨Source.prefixBefore (code extra false false) "out" 1, [4], operand "x", operand "y"⟩,
   ⟨Source.prefixBefore (code extra true false) "out" 2, [4], operand "out", operand "y"⟩]
private def program (extra : List ComputeStmt) (a b : Bool) : Program :=
  ⟨.mk ["x", "y"] ["out"] (code extra a b), sites extra⟩

-- Both rewrites survive an inserted common statement and repeated writes to
-- the same register. The second guard refers to the intermediate program.
theorem two_sites (R : ScalarArithmetic.Rules) :
    program [padding] false false ≡[R] program [padding] true true := by
  equiv_decompose
  all_goals fp_prove

/-- info: FP assumptions used by two_sites:
---
info:   add_commute
---
info:   mul_commute -/
#guard_msgs in
#print_fp_assumptions two_sites

#guard_msgs (drop info) in
#axiomsClean two_sites

set_option linter.unusedVariables false

-- A missing name or the wrong write occurrence cannot supply guards for the
-- real rewrite point. Nor may the guard be checked before loading its inputs.
example (R : ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : ({ program [] false false with domain := [] } : Program) ≡[R]
        ({ program [] true true with domain := [] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  trivial

private def wrongSite (name : RegName) (occurrence : Nat) : Site :=
  ⟨Source.prefixBefore (code [] false false) name occurrence, [4], operand "x", operand "y"⟩

example (R : ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : ({ program [] false false with domain := [wrongSite "out" 0] } : Program) ≡[R]
        ({ program [] true false with domain := [wrongSite "out" 0] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  fail_if_success
    have : ({ program [] false false with domain := [wrongSite "missing" 0] } : Program) ≡[R]
        ({ program [] true false with domain := [wrongSite "missing" 0] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  fail_if_success
    have : ({ program [] false false with domain := [wrongSite "x" 0] } : Program) ≡[R]
        ({ program [] true false with domain := [wrongSite "x" 0] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  trivial

-- A fp32 atom supplies neither fp64 commutation nor a changed store.
example (R : ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : (⟨.mk ["x", "y"] [] [add false], sites []⟩ : Program) ≡[R]
        (⟨.mk ["x", "y"] [] [.assign .real [4] "out" (.compute (.alg .fp64
          (.add .real (.consSame .nil) (operand "y") (operand "x"))))], sites []⟩ : Program) := by
      equiv_decompose
      all_goals fp_prove
  fail_if_success
    have : (⟨.mk ["x", "y"] ["out"] [write], []⟩ : Program) ≡[R]
        (⟨.mk ["x", "y"] ["out"] [.store .real [4] (.region "out" (.arange 4))
          (.alg (operand "x")) .none], []⟩ : Program) := by
      equiv_decompose
      all_goals fp_prove
  trivial

-- Signatures, list alignment and atom admission remain proof obligations.
example (R : ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : program [] false false ≡[R]
        ({ program [] true true with kernel := .mk ["different"] ["out"] (code [] true true) } : Program) := by
      equiv_decompose
      all_goals fp_prove
  fail_if_success
    have : program [] false false ≡[R] program [padding] true true := by
      equiv_decompose
      all_goals fp_prove
  trivial

example : True := by
  fail_if_success
    have : program [] false false ≡[([] : Spec.Assumptions GuardedFragment)] program [] true true := by
      equiv_decompose
      all_goals fp_prove
  trivial

end GuardedRewriteTacticTests
