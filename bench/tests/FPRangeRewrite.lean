import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! A loop rewrite must keep the guard at every reachable iteration. The
negative witness below passes the first iteration but fails the next one. -/
namespace FPRangeRewriteTests
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

private def x : Op .real [] := .ref .real [] "x"
private def one : Op .real [] := .const 1
private def body (swapped : Bool) : List ComputeStmt :=
  [.assign .real [] "x" (.alg
    (if swapped then .add .real .nil one x else .add .real .nil x one))]
private def inner : Site := .add [] [] x one
private def loopSite (stop : Nat) : Site :=
  .range [] "i" (.constNat 0) (.constNat stop) (.constNat 1) (body false) inner

theorem loop_contextual {α : Type} [Inhabited α] (R : FP.ScalarArithmetic.Rules)
    (M : Algebra α) (D : FP.Guarded.Domain α)
    (hM : FP.Guarded.Models R.assumptions M D) (s : State α) (stop : Nat)
    (hd : (loopSite stop).Holds M D s) :
    Contextual M s [] []
      (.forRangeDyn "i" (.constNat 0) (.constNat stop) (.constNat 1) (body false))
      (.forRangeDyn "i" (.constNat 0) (.constNat stop) (.constNat 1) (body true)) := by
  apply contextual_range M D s [] [] "i" (.constNat 0) (.constNat stop)
    (.constNat 1) (body false) (body true) inner hd
  intro t ht
  exact add_commute R M D hM t [] [] "x" x one ht

private def program (swapped : Bool) (stop : Nat) : Program where
  kernel := .mk [] [] [.forRangeDyn "i" (.constNat 0) (.constNat stop)
    (.constNat 1) (body swapped)]
  domain := [loopSite stop]

theorem loop_add_commute (R : FP.ScalarArithmetic.Rules) (stop : Nat) :
    program false stop ≡[R] program true stop := by
  equiv_decompose
  all_goals fp_prove

/-- info: FP assumptions used by loop_add_commute:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions loop_add_commute

private def padding : ComputeStmt := .assign .nat [] "padding" (.alg (.constNat 17))
private def paddedBody (swapped : Bool) := [padding] ++ body swapped
private def paddedCode (swapped : Bool) (stop : Nat) : List ComputeStmt :=
  [padding, .forRangeDyn "i" (.constNat 0) (.constNat stop) (.constNat 1)
    (paddedBody swapped), padding]
private def paddedSite (stop : Nat) : List Site :=
  match Source.dynamicRange? (paddedCode false stop) "i" with
  | none => []
  | some loop => [.range loop.before "i" loop.start loop.stop loop.stride loop.body
      (.add (Source.prefixBefore loop.body "x") [] x one)]
private def paddedProgram (swapped : Bool) (stop : Nat) : Program :=
  ⟨.mk [] [] (paddedCode swapped stop), paddedSite stop⟩

-- Common insertions before, inside and after the loop do not require source
-- positions in the proof or the domain. The actual prefix remains checked.
theorem loop_with_inserted_statements (R : FP.ScalarArithmetic.Rules) (stop : Nat) :
    paddedProgram false stop ≡[R] paddedProgram true stop := by
  equiv_decompose
  all_goals fp_prove

private def nestedProgram (swapped : Bool) (outer innerStop : Nat) : Program where
  kernel := .mk [] [] [.forRangeDyn "j" (.constNat 0) (.constNat outer)
    (.constNat 1) (paddedCode swapped innerStop)]
  domain := [.range [] "j" (.constNat 0) (.constNat outer) (.constNat 1)
    (paddedCode false innerStop)
    (.range [padding] "i" (.constNat 0) (.constNat innerStop) (.constNat 1)
      (paddedBody false) (.add [padding] [] x one))]

theorem nested_ranges (R : FP.ScalarArithmetic.Rules) (outer innerStop : Nat) :
    nestedProgram false outer innerStop ≡[R] nestedProgram true outer innerStop := by
  equiv_decompose
  all_goals fp_prove

/-- info: FP assumptions used by nested_ranges:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions nested_ranges

-- These negative checks exercise automation, not a numerical counterexample.
set_option linter.unusedVariables false in
example (R : FP.ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : ({ program false 2 with domain := [] } : Program) ≡[R]
        ({ program true 2 with domain := [] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  -- An entry-state check is insufficient for later iterations.
  fail_if_success
    have : ({ program false 2 with domain := [inner] } : Program) ≡[R]
        ({ program true 2 with domain := [inner] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  -- Retain the same signature so failure must come from the different bound.
  fail_if_success
    have : program false 2 ≡[R]
        ({ program true 3 with domain := [loopSite 2] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  -- A guard with the wrong per-iteration prefix cannot be borrowed.
  fail_if_success
    have : ({ paddedProgram false 2 with domain := [loopSite 2] } : Program) ≡[R]
        ({ paddedProgram true 2 with domain := [loopSite 2] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  trivial

example : True := by
  fail_if_success
    have : program false 2 ≡[([] : Spec.Assumptions FP.GuardedFragment)]
        program true 2 := by
      equiv_decompose
      all_goals fp_prove
  trivial

private def model : Algebra Nat where
  literal := fun _ _ _ => 1
  negInf := 0
  binary := fun _ _ _ a b => a + b
  unary := fun _ _ x => x
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n.toNat
  fp32Bits := fun _ => 1
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0

private def initial : State Nat :=
  ({ mem := fun _ _ => .mk .real 0, regs := fun _ _ _ => none,
     pids := fun _ => 0, numPids := fun _ => 1,
     undef := fun d _ _ => defaultValue d } : State Nat).setReg "x" .real [] (fun _ => 1)
private def next : State Nat :=
  (initial.setReg "i" .nat [] (fun _ => 0)).setReg "x" .real [] (fun _ => 2)
private def bounded : FP.Guarded.Domain Nat
  | .finite, a => a ≤ 1
  | _, _ => True

theorem first_operands_are_finite : inner.Holds model bounded
    (initial.setReg "i" .nat [] (fun _ => 0)) := by
  intro t ht a b ha hb
  simp only [run, Option.some.injEq] at ht
  subst t
  simp [x, one, FP.Structural.evalOp, initial] at ha hb
  subst a
  subst b
  simp [engine, model, Algebra.withDefaultPrecision, bounded]

theorem first_iteration_reaches_second :
    RangeReachable (engine model) "i" 2 1 (body false) 0 initial 1 next := by
  apply RangeReachable.next .initial (by decide) (by decide)
  simp [run, step, body, FP.Structural.evalExpr, FP.Structural.evalOp, x, one,
    engine, model, Algebra.withDefaultPrecision, numeric, initial, next]
  rfl

theorem initial_finite_does_not_cover_later_iterations :
    ¬ (loopSite 2).Holds model bounded initial := by
  intro hd
  have hh := hd initial (by simp [run]) (fun _ => 0) (fun _ => 2) (fun _ => 1)
    (by rfl) (by rfl) (by rfl) 1 next first_iteration_reaches_second
    (by decide) (by decide)
  have bad := hh (next.setReg "i" .nat [] (fun _ => 1)) (by simp [run])
    (fun _ => 2) (fun _ => 1)
    (by simp [x, FP.Structural.evalOp, next])
    (by rfl)
  exact (by decide : ¬ (2 ≤ 1)) (bad.1 PUnit.unit)

-- Zero stride and empty intervals have no body visits and need no operands.
theorem zero_stride_needs_no_body_guard {α : Type} [Inhabited α]
    (M : Algebra α) (D : FP.Guarded.Domain α) (s : State α) :
    (Site.range [] "i" (.constNat 0) (.constNat 2) (.constNat 0)
      (body false) inner).Holds M D s := by
  intro t ht a b d ha hb hd cur u hr hz hc
  have he : d = fun _ => 0 := Option.some.inj hd.symm
  exact False.elim (hz (by rw [he]))

private def scaledBody (fp64 keepCast swapped : Bool) :
    List ComputeStmt :=
  let value : Op .real [2, 3] := if swapped then
    .mul .real .scalarL (.const 2) (.ref .real [2, 3] "tile")
    else .mul .real .scalarR (.ref .real [2, 3] "tile") (.const 2)
  let casted := if keepCast then .castFloat .real .real value else value
  [.assign .real [2, 3] "tile" (if fp64 then .compute (.alg .fp64 casted)
    else .compute (.alg .fp32 casted))]
private def scaledProgram (fp64 keepCast swapped : Bool) : Program where
  kernel := .mk [] [] [.forRangeDyn "i" (.constNat 0) (.constNat 2) (.constNat 1)
    (scaledBody fp64 keepCast swapped)]
  domain := [.range [] "i" (.constNat 0) (.constNat 2) (.constNat 1)
    (scaledBody false true false) (.mulScalar [] 2 [3] (.ref .real [2, 3] "tile") (.const 2))]

theorem explicit_cast_preserved (R : FP.ScalarArithmetic.Rules) :
    scaledProgram false true false ≡[R] scaledProgram false true true := by
  equiv_decompose
  all_goals fp_prove

/-- info: FP assumptions used by explicit_cast_preserved:
---
info:   mul_commute -/
#guard_msgs in
#print_fp_assumptions explicit_cast_preserved

set_option linter.unusedVariables false in
example (R : FP.ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : scaledProgram false true false ≡[R] scaledProgram false false true := by
      equiv_decompose
      all_goals fp_prove
  fail_if_success
    have : scaledProgram false true false ≡[R] scaledProgram true true true := by
      equiv_decompose
      all_goals fp_prove
  trivial

#guard_msgs (drop info) in
#auditModuleAxioms
end FPRangeRewriteTests
