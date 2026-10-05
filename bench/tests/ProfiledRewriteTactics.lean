import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

namespace ProfiledRewriteTacticTests
open VeriTile Triton FP FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

private def readX : ComputeStmt := .assign .real [4] "x"
  (.alg (.load .real (.region "x" (.arange 4)) .none))
private def readY : ComputeStmt := .assign .real [4] "y"
  (.alg (.load .real (.region "y" (.arange 4)) .none))
private def padding : ComputeStmt := .assign .nat [] "unrelated" (.alg (.constNat 17))
private def add (swapped : Bool) : ComputeStmt := .assign .real [4] "sum" (.alg
  (.add .real (.consSame .nil)
    (.ref .real [4] (if swapped then "y" else "x"))
    (.ref .real [4] (if swapped then "x" else "y"))))
private def divide (optimized : Bool) : ComputeStmt := .assign .real [4] "out" (.alg
  (if optimized then .mul .real .scalarR (.ref .real [4] "sum") (.div .real .nil (.const 1) (.const 2))
   else .div .real .scalarR (.ref .real [4] "sum") (.const 2)))
private def write : ComputeStmt := .store .real [4] (.region "out" (.arange 4))
  (.alg (.ref .real [4] "out")) .none
private def code (extra : List ComputeStmt) (a b : Bool) : List ComputeStmt :=
  extra ++ [readX, readY, add a, divide b, write]
private def sites (extra : List ComputeStmt) : List Site :=
  [.add (prefixBefore (code extra false false) "sum") [4] (.ref .real [4] "x") (.ref .real [4] "y"),
   .div (prefixBefore (code extra true false) "out") 4 [] (.ref .real [4] "sum") 2]
private def program (extra : List ComputeStmt) (a b : Bool) : Program :=
  ⟨.mk ["x", "y"] ["out"] (code extra a b), sites extra⟩

-- Two differences compose through the actual intermediate prefix. The extra
-- common statement shifts both sites without changing their definitions.
theorem two_sites (R : ScalarArithmetic.Rules) :
    program [padding] false false ≡[R] program [padding] true true := by
  equiv_decompose
  all_goals fp_prove

#guard_msgs (drop info) in
#axiomsClean two_sites

-- Failed proof attempts intentionally leave their hypotheses unused.
set_option linter.unusedVariables false

-- The domain is mandatory: removing the finite and nonzero obligations must
-- not turn the same source transformation into a proved equivalence.
example (R : ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : ({ program [] false false with domain := [] } : Program) ≡[R]
        ({ program [] true true with domain := [] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  trivial

-- A different actual operand cannot borrow the first site's guard.
example (R : ScalarArithmetic.Rules) (M : Algebra Nat) (D : Guarded.Domain Nat)
    (hM : Guarded.Models R.assumptions M D) (s : State Nat)
    (hd : (Site.add [] [4] (.ref .real [4] "x") (.ref .real [4] "y")).Holds M D s) : True := by
  fail_if_success
    have : Contextual M s [] []
        (.assign .real [4] "out" (.alg (.add .real (.consSame .nil) (.ref .real [4] "other") (.ref .real [4] "y"))))
        (.assign .real [4] "out" (.alg (.add .real (.consSame .nil) (.ref .real [4] "y") (.ref .real [4] "other")))) := by
      fp_prove
  trivial

-- The adapter decomposes a changed memory statement but cannot justify it
-- using an arithmetic atom. No store congruence is assumed.
example (R : ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : (⟨.mk ["x", "y"] ["out"] [write], []⟩ : Program) ≡[R]
        (⟨.mk ["x", "y"] ["out"]
          [.store .real [4] (.region "out" (.arange 4)) (.alg (.ref .real [4] "x")) .none], []⟩ : Program) := by
      equiv_decompose
      all_goals fp_prove
  trivial

-- Changing compute precision is an observable source difference. A fp32
-- scalar atom cannot discharge a statement evaluated at fp64.
example (R : ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : (⟨.mk ["x", "y"] [] [add false], (sites [])⟩ : Program) ≡[R]
        (⟨.mk ["x", "y"] []
          [.assign .real [4] "sum" (.compute (.alg .fp64
            (.add .real (.consSame .nil) (.ref .real [4] "y") (.ref .real [4] "x"))))],
          (sites [])⟩ : Program) := by
      equiv_decompose
      all_goals fp_prove
  trivial

-- An empty assumption table supplies no numerical rewrite evidence.
example : True := by
  fail_if_success
    have : program [] false false ≡[([] : Spec.Assumptions GuardedFragment)] program [] true true := by
      equiv_decompose
      all_goals fp_prove
  trivial

end ProfiledRewriteTacticTests
