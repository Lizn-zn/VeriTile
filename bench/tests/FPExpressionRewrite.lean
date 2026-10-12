import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Triton.Float.ExpressionRewrite
import bench.tritonbench_g.apply_penalty.FPEquiv
import VeriTile.Meta.StatementAudit

namespace FPExpressionRewriteTests
open VeriTile Triton FP FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option linter.unusedVariables false

private def x : Op .real [4] := .ref .real [4] "x"
private def b : Op .real [] := .ref .real [] "b"
private def condition : Op .bool [4] := .ref .bool [4] "positive"
private def expressionWith (c : Op .bool [4]) (optimized : Bool) : Op .real [4] :=
  .add .real (.consSame .nil)
    (.sub .real (.consSame .nil)
      (.where c
        (if optimized then .mul .real .scalarR x (.div .real .nil (.const 1) b)
         else .div .real .scalarR x b)
        (.mul .real .scalarR x b)) x) x
private def expression := expressionWith condition
private def commonPrefix : List ComputeStmt := [
  .assign .real [4] "x" (.alg (.load .real (.region "x" (.arange 4)) .none)),
  .assign .real [] "b" (.alg (.load .real (.region "b" (.constNat 0)) .none)),
  .assign .bool [4] "positive" (.alg (.gt .real .scalarR x (.const 0)))]
private def tailMask : MaskOpt .real [4] :=
  .mask (.lt .nat .scalarR (.arange 4) (.constNat 3))
private def store (optimized : Bool) : ComputeStmt :=
  .store .real [4] (.region "x" (.arange 4)) (.alg (expression optimized)) tailMask
private def program (optimized : Bool) : Program where
  kernel := .mk ["x", "b"] ["x"] (commonPrefix ++ [store optimized])
  domain := [.divScalar commonPrefix 4 [] x b]

-- A division nested under where, subtraction and addition is lifted through an
-- in-place masked store. The condition and the unselected arm stay unchanged.
theorem nested_masked_store (R : ScalarArithmetic.Rules) :
    program Bool.false ≡[R] program Bool.true := by
  equiv_decompose
  all_goals fp_prove

#print_fp_assumptions nested_masked_store
#guard_msgs (drop info) in
#axiomsClean nested_masked_store

-- Congruence cannot create missing admission-domain premises.
example (R : ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : ({ program Bool.false with domain := [] } : Program) ≡[R]
        ({ program Bool.true with domain := [] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  trivial

-- Finite operands alone do not supply the reciprocal rule's nonzero condition.
example (R : ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : ({ program Bool.false with domain := [.addScalar commonPrefix 4 [] x b] } : Program) ≡[R]
        ({ program Bool.true with domain := [.addScalar commonPrefix 4 [] x b] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  trivial

-- Replacing the condition changes which value is selected, even with the same
-- division guards. The where-congruence lemma must check the condition too.
example (R : ScalarArithmetic.Rules) : True := by
  let altered : ComputeKernel := .mk ["x", "b"] ["x"] (commonPrefix ++ [
    .store .real [4] (.region "x" (.arange 4))
      (.alg (expressionWith (.full [4] (.constBool Bool.true)) Bool.true)) tailMask])
  fail_if_success
    have : program Bool.false ≡[R] ({ program Bool.true with kernel := altered } : Program) := by
      equiv_decompose
      all_goals fp_prove
  trivial

-- Keeping the numerical expression does not justify an unmasked store.
example (R : ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : program Bool.false ≡[R] ({ program Bool.true with kernel := .mk ["x", "b"] ["x"] (commonPrefix ++ [
        .store .real [4] (.region "x" (.arange 4)) (.alg (expression Bool.true)) .none]) } : Program) := by
      equiv_decompose
      all_goals fp_prove
  trivial

-- The fp32 leaf cannot be lifted into an explicitly fp64 expression.
example (R : ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : program Bool.false ≡[R] ({ program Bool.true with kernel := .mk ["x", "b"] ["x"] (commonPrefix ++ [
        .store .real [4] (.region "x" (.arange 4))
          (.compute (.alg .fp64 (expression Bool.true))) tailMask]) } : Program) := by
      equiv_decompose
      all_goals fp_prove
  trivial

-- Abstract execution witnesses use a total backend and a genuine nonzero
-- denominator value. These are not additional numerical admissions.
private def model : Algebra Nat where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun _ _ _ _ _ => 0
  unary := fun _ _ _ => 0
  compareEq := fun _ _ => some (fun _ _ => Bool.true)
  compareLt := fun _ _ => some (fun _ _ => Bool.false)
  compareLe := fun _ _ => some (fun _ _ => Bool.true)
  cast := fun _ _ _ _ => 0
  fromNat := fun _ _ => 0
  fromInt := fun _ _ => 0
  fp32Bits := fun _ => 0
  fp32Load := fun _ => 0
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0

private def initial : State Nat where
  mem := fun r o =>
    if r = "meta" then .mk .nat (if o = 1 then 1 else 0)
    else if r = "ids" ∨ r = "counts" then .mk .nat 0
    else .mk .real (if r = "repetition" then 1 else 0)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

private def domain : Guarded.Domain Nat
  | .finite, n => n ≤ 1
  | .nonzero, n => n ≠ 0
  | _, n => 0 < n

-- Even the unselected where arm is evaluated: a missing operand causes failure.
theorem where_is_eager : Structural.evalOp (engine model) none
    (.where (.constBool Bool.false) (.ref .real [] "missing") (.const 0)) initial = none := rfl

open VeriTile.Bench.TritonBenchG.ApplyPenalty
set_option maxHeartbeats 1600000
set_option linter.unusedSimpArgs false

theorem penalty_executes : ∃ t, Structural.exec (engine model)
    (apply_penalty "logits" "presence" "frequency" "repetition" "ids" "counts" "meta" 1 1 1)
    initial = some t := by
  simp [apply_penalty, Structural.exec, run, step, Structural.evalExpr,
    evalComputeOp, evalOp_unfold, engine, Algebra.withDefaultPrecision, model,
    initial, ComputeDType.eraseDType, numeric, numericEq, numericLt, natLt,
    bop, Structural.store, State.setReg, TileShape.allIndices, List.finRange_succ]

theorem penalty_domain_inhabited :
    (FPEquiv.divisionSite "logits" "presence" "frequency" "repetition"
      "ids" "counts" "meta" 1 1 1).Holds model domain initial := by
  intro t ht a b ha hb
  simp [FPEquiv.divisionSite, Source.prefixBefore, apply_penalty,
    ComputeKernel.surfaceBody, run, step, Structural.evalExpr, evalComputeOp,
    evalOp_unfold, engine, Algebra.withDefaultPrecision, model, initial,
    ComputeDType.eraseDType, numeric, numericEq, numericLt, natLt, bop, State.setReg] at ht
  subst t
  simp [evalOp_unfold, State.setReg] at ha hb
  subst a
  subst b
  exact ⟨fun _ => Nat.zero_le 1, Nat.le_refl 1, Nat.one_ne_zero⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end FPExpressionRewriteTests
