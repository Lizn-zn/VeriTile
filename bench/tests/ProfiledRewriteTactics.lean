import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Triton.Float.ExpressionRewrite
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

private def originalSites (extra : List ComputeStmt) : List Site :=
  [.add (prefixBefore (code extra false false) "sum") [4]
      (.ref .real [4] "x") (.ref .real [4] "y"),
   .div (prefixBefore (code extra false false) "out") 4 [] (.ref .real [4] "sum") 2]

-- Later sites can refer to the original prefix, including earlier changed
-- assignments. Composing the later rewrite first keeps these guards intact.
theorem two_original_sites (R : ScalarArithmetic.Rules) :
    ({ program [padding] false false with domain := originalSites [padding] } : Program) ≡[R]
    ({ program [padding] true true with domain := originalSites [padding] } : Program) := by
  equiv_decompose (reverse := true)
  all_goals fp_prove

#guard_msgs (drop info) in
#axiomsClean two_original_sites

-- Proof order supplies no missing guard for the later computed operand.
example (R : ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : ({ program [] false false with domain := (originalSites []).take 1 } : Program) ≡[R]
        ({ program [] true true with domain := (originalSites []).take 1 } : Program) := by
      equiv_decompose (reverse := true)
      all_goals fp_prove
  trivial

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

-- Arithmetic rules cannot replace the stored register with an unrelated
-- register merely because the address and declared IO signature match.
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

namespace ComputedDenominatorTests
open VeriTile Triton FP FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

private def numerator : Op .real [4] := .ref .real [4] "x"
private def denominator : Op .real [] := .ref .real [] "sum"
private def commonPrefix : List ComputeStmt := [
  .assign .real [4] "x" (.alg (.load .real (.region "x" (.arange 4)) .none)),
  .assign .real [] "sum" (.alg (.reduceSum ⟨0, by simp⟩ false numerator))]
private def assignment (optimized : Bool) : ComputeStmt := .assign .real [4] "out" (.alg
  (if optimized then .mul .real .scalarR numerator (.div .real .nil (.const 1) denominator)
   else .div .real .scalarR numerator denominator))
private def site : Site := .divScalar commonPrefix 4 [] numerator denominator
private def program (optimized : Bool) : Program :=
  ⟨.mk ["x"] [] (commonPrefix ++ [assignment optimized]), [site]⟩

theorem computed_denominator (R : ScalarArithmetic.Rules) :
    program false ≡[R] program true := by
  equiv_decompose
  all_goals fp_prove

/-- info: FP assumptions used by computed_denominator:
---
info:   div_mul_rcp -/
#guard_msgs in
#print_fp_assumptions computed_denominator
#guard_msgs (drop info) in
#axiomsClean computed_denominator

set_option linter.unusedVariables false

-- Checking finite operands alone cannot replace the nonzero requirement.
example (R : ScalarArithmetic.Rules) : True := by
  let finiteOnly : List Site := [.add commonPrefix [4] numerator numerator,
    .add commonPrefix [] denominator denominator]
  fail_if_success
    have : ({ program false with domain := finiteOnly } : Program) ≡[R]
        ({ program true with domain := finiteOnly } : Program) := by
      equiv_decompose
      all_goals fp_prove
  -- A check at the wrong commonPrefix does not establish the computed sum's domain.
  fail_if_success
    have : ({ program false with domain := [.divScalar [] 4 [] numerator denominator] } : Program) ≡[R]
        ({ program true with domain := [.divScalar [] 4 [] numerator denominator] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  trivial

end ComputedDenominatorTests

namespace ScalarBroadcastTests
open VeriTile Triton FP FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

private def tile : Op .real [4] := .ref .real [4] "x"
private def scalar : Op .real [] := .ref .real [] "value"
private def commonPrefix : List ComputeStmt := [
  .assign .nat [] "padding" (.alg (.constNat 17)),
  .assign .real [4] "x" (.alg (.load .real (.region "x" (.arange 4)) .none)),
  .assign .real [] "value" (.alg (.reduceSum ⟨0, by simp⟩ false tile))]
private def expression (swapped : Bool) : Op .real [4] :=
  if swapped then .add .real .scalarL scalar tile else .add .real .scalarR tile scalar
private def site : Site := .addScalar commonPrefix 4 [] tile scalar
private def program (swapped : Bool) : Program :=
  ⟨.mk ["x"] [] (commonPrefix ++ [.assign .real [4] "out" (.alg (expression swapped))]),
    [site]⟩

-- The scalar is a computed reduction result, not a literal whose finiteness
-- can be assumed by its spelling. The extra prefix statement is retained.
theorem computed_scalar (R : ScalarArithmetic.Rules) :
    program false ≡[R] program true := by
  equiv_decompose
  all_goals fp_prove

/-- info: FP assumptions used by computed_scalar:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions computed_scalar
#guard_msgs (drop info) in
#axiomsClean computed_scalar

set_option linter.unusedVariables false

example (R : ScalarArithmetic.Rules) : True := by
  -- Finiteness of the tile says nothing about the computed scalar.
  fail_if_success
    have : ({ program false with domain := [.add commonPrefix [4] tile tile] } : Program) ≡[R]
        ({ program true with domain := [.add commonPrefix [4] tile tile] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  -- Neither another scalar nor the scalar before its computation can lend a guard.
  fail_if_success
    have : ({ program false with domain := [.addScalar commonPrefix 4 [] tile
          (.ref .real [] "other")] } : Program) ≡[R]
        ({ program true with domain := [.addScalar commonPrefix 4 [] tile
          (.ref .real [] "other")] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  fail_if_success
    have : ({ program false with domain := [.addScalar [] 4 [] tile scalar] } : Program) ≡[R]
        ({ program true with domain := [.addScalar [] 4 [] tile scalar] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  -- An explicit fp64 operation does not become a profiled fp32 operation.
  fail_if_success
    have : program false ≡[R]
        (⟨.mk ["x"] [] (commonPrefix ++ [.assign .real [4] "out"
          (.compute (.alg .fp64 (expression true)))]), [site]⟩ : Program) := by
      equiv_decompose
      all_goals fp_prove
  trivial

end ScalarBroadcastTests

namespace ScalarMultiplicationTests
open VeriTile Triton FP FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

private def tile : Op .real [4] := .ref .real [4] "x"
private def scalar : Op .real [] := .ref .real [] "scale"
private def commonPrefix : List ComputeStmt := [
  .assign .real [4] "x" (.alg (.load .real (.region "x" (.arange 4)) .none)),
  .assign .real [] "scale" (.alg (.reduceSum ⟨0, by simp⟩ false tile))]
private def expression (swapped : Bool) : Op .real [4] :=
  if swapped then .mul .real .scalarR tile scalar else .mul .real .scalarL scalar tile
private def assignment (swapped : Bool) : ComputeStmt :=
  .assign .real [4] "out" (.alg (expression swapped))
private def write : ComputeStmt :=
  .store .real [4] (.region "x" (.arange 4)) (.alg (.ref .real [4] "out")) .none
private def site : Site := .mulScalar commonPrefix 4 [] tile scalar
private def program (swapped : Bool) : Program :=
  ⟨.mk ["x"] ["x"] (commonPrefix ++ [assignment swapped, write]), [site]⟩

-- The scalar is computed, and the result overwrites the input region. The
-- rewrite must preserve both the actual scalar domain and the suffix store.
theorem inplace_computed_multiplier (R : ScalarArithmetic.Rules) :
    program false ≡[R] program true := by
  equiv_decompose
  all_goals fp_prove

/-- info: FP assumptions used by inplace_computed_multiplier:
---
info:   mul_commute -/
#guard_msgs in
#print_fp_assumptions inplace_computed_multiplier
#guard_msgs (drop info) in
#axiomsClean inplace_computed_multiplier

set_option linter.unusedVariables false

example (R : ScalarArithmetic.Rules) : True := by
  -- Finiteness of a tile does not establish finiteness of its reduced multiplier.
  fail_if_success
    have : ({ program false with domain := [.add commonPrefix [4] tile tile] } : Program) ≡[R]
        ({ program true with domain := [.add commonPrefix [4] tile tile] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  -- Keeping the declared IO names cannot justify storing the old input instead.
  fail_if_success
    have : program false ≡[R]
        (⟨.mk ["x"] ["x"] (commonPrefix ++ [assignment true,
          .store .real [4] (.region "x" (.arange 4)) (.alg tile) .none]), [site]⟩ : Program) := by
      equiv_decompose
      all_goals fp_prove
  -- The new broadcast rule cannot erase an explicit compute precision.
  fail_if_success
    have : program false ≡[R]
        (⟨.mk ["x"] ["x"] (commonPrefix ++ [.assign .real [4] "out"
          (.compute (.alg .fp64 (expression true))), write]), [site]⟩ : Program) := by
      equiv_decompose
      all_goals fp_prove
  trivial

end ScalarMultiplicationTests

namespace StoreValueTests
open VeriTile Triton FP FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

private def left : Op .real [4] := .ref .real [4] "x"
private def right : Op .real [4] := .ref .real [4] "y"
private def expression (swapped : Bool) : Op .real [4] :=
  if swapped then .add .real (.consSame .nil) right left
  else .add .real (.consSame .nil) left right
private def tailMask : MaskOpt .real [4] :=
  .mask (.lt .nat .scalarR (.arange 4) (.constNat 3))
private def commonPrefix : List ComputeStmt := [
  .assign .real [4] "x" (.alg (.load .real (.region "x" (.arange 4)) .none)),
  .assign .real [4] "y" (.alg (.load .real (.region "y" (.arange 4)) .none)),
  .alg (.store .real [4] (.region "scratch" (.arange 4)) left .none),
  .assign .real [4] "x" (.alg (.exp left)),
  .assign .nat [] "padding" (.alg (.constNat 17))]
private def write (swapped : Bool) : ComputeStmt :=
  .store .real [4] (.region "out" (.arange 4)) (.alg (expression swapped)) tailMask
private def code (swapped : Bool) : List ComputeStmt := commonPrefix ++ [write swapped]
private def sites : List Site :=
  [.add (Source.prefixBeforeStore (code false) 1) [4] left right]
private def program (swapped : Bool) : Program :=
  ⟨.mk ["x", "y"] ["scratch", "out"] (code swapped), sites⟩

-- The selected store follows an algorithm-wrapped store, an operand update
-- and an unrelated assignment. Its value changes; the tail mask stays intact.
theorem masked_store_value (R : ScalarArithmetic.Rules) :
    program false ≡[R] program true := by
  equiv_decompose
  all_goals fp_prove

/-- info: FP assumptions used by masked_store_value:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions masked_store_value
#guard_msgs (drop info) in
#axiomsClean masked_store_value

set_option linter.unusedVariables false

example (R : ScalarArithmetic.Rules) : True := by
  -- Arithmetic evidence cannot change the store's mask or destination, even
  -- when the declared IO names and all numerical guards are unchanged.
  fail_if_success
    have : program false ≡[R]
        (⟨.mk ["x", "y"] ["scratch", "out"] (commonPrefix ++ [
          .store .real [4] (.region "out" (.arange 4)) (.alg (expression true)) .none]),
          sites⟩ : Program) := by
      equiv_decompose
      all_goals fp_prove
  fail_if_success
    have : program false ≡[R]
        (⟨.mk ["x", "y"] ["scratch", "out"] (commonPrefix ++ [
          .store .real [4] (.region "other" (.arange 4)) (.alg (expression true)) tailMask]),
          sites⟩ : Program) := by
      equiv_decompose
      all_goals fp_prove
  fail_if_success
    have : program false ≡[R]
        (⟨.mk ["x", "y"] ["scratch", "out"] (commonPrefix ++ [
          .store .real [4] (.region "out" (.arange 4))
            (.compute (.alg .fp64 (expression true))) tailMask]), sites⟩ : Program) := by
      equiv_decompose
      all_goals fp_prove
  -- The earlier store's guards do not establish finiteness after exp updates x.
  fail_if_success
    have : ({ program false with domain :=
          [.add (Source.prefixBeforeStore (code false)) [4] left right] } : Program) ≡[R]
        ({ program true with domain :=
          [.add (Source.prefixBeforeStore (code false)) [4] left right] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  trivial

end StoreValueTests

namespace ExplicitFP32ScalarTests
open VeriTile Triton FP FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

private def tile : Op .real [4] := .ref .real [4] "x"
-- A computed scalar forces the rule and its guards to retain the fp32
-- evaluation context, rather than treating a register lookup as a special case.
private def scalar : Op .real [] := .reduceSum ⟨0, by simp⟩ Bool.false tile
private def assignment (wide swapped : Bool) : ComputeStmt :=
  let op := if swapped then .mul .real .scalarL scalar tile else .mul .real .scalarR tile scalar
  .assign .real [4] "out" (if wide then .compute (.alg .fp64 op) else .compute (.alg .fp32 op))
private def site : Site := .mulScalarFP32 [] 4 [] tile scalar
private def program (wide swapped : Bool) (sites : List Site) : Program :=
  ⟨.mk ["x"] [] [assignment wide swapped], sites⟩

theorem computed_fp32_scalar (R : ScalarArithmetic.Rules) :
    program Bool.false Bool.false [site] ≡[R] program Bool.false Bool.true [site] := by
  equiv_decompose
  all_goals fp_prove

/-- info: FP assumptions used by computed_fp32_scalar:
---
info:   mul_commute -/
#guard_msgs in
#print_fp_assumptions computed_fp32_scalar
#guard_msgs (drop info) in
#axiomsClean computed_fp32_scalar

set_option linter.unusedVariables false
example (R : ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : program Bool.false Bool.false [] ≡[R] program Bool.false Bool.true [] := by
      equiv_decompose
      all_goals fp_prove
  trivial

example (R : ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : program Bool.true Bool.false [site] ≡[R] program Bool.true Bool.true [site] := by
      equiv_decompose
      all_goals fp_prove
  trivial
end ExplicitFP32ScalarTests
