import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Static natural bounds retain reached-iteration guards, complete prefixes
and the original range constructor. An entry-only or misplaced guard fails. -/
namespace FPStaticRangeRewriteTests
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

private def x : Op .real [] := .ref .real [] "x"
private def one : Op .real [] := .const 1
private def padding : ComputeStmt := .assign .nat [] "unrelated" (.alg (.constNat 17))
private def body (swapped : Bool) : List ComputeStmt :=
  [padding, .assign .real [] "x" (.alg
    (if swapped then .add .real .nil one x else .add .real .nil x one))]
private def code (swapped : Bool) (start stop stride : Nat) : List ComputeStmt :=
  [padding, .forRange "i" start stop stride (body swapped), padding]
private def sites (start stop stride : Nat) : List Site :=
  (do
    let loop ← Source.dynamicRange? (code false start stop stride) "i"
    return Site.range loop.before "i" loop.start loop.stop loop.stride loop.body
      (.add (Source.prefixBefore loop.body "x") [] x one)).toList
private def program (swapped : Bool) (start stop stride : Nat) : Program :=
  ⟨.mk [] [] (code swapped start stop stride), sites start stop stride⟩

theorem static_range_with_insertions (R : FP.ScalarArithmetic.Rules)
    (start stop stride : Nat) :
    program false start stop stride ≡[R] program true start stop stride := by
  equiv_decompose
  fp_prove

private def nested (swapped : Bool) (outer inner : Nat) : Program where
  kernel := .mk [] [] [.forRange "j" 0 outer 1 (code swapped 0 inner 1)]
  domain := [.range [] "j" (.constNat 0) (.constNat outer) (.constNat 1)
    (code false 0 inner 1)
    (.range [padding] "i" (.constNat 0) (.constNat inner) (.constNat 1)
      (body false) (.add [padding] [] x one))]

theorem nested_static_ranges (R : FP.ScalarArithmetic.Rules) (outer inner : Nat) :
    nested false outer inner ≡[R] nested true outer inner := by
  equiv_decompose
  fp_prove

/-- A common inner loop before the changed statement stays in its guard. -/
private def precedingLoop : ComputeStmt :=
  .forRangeDyn "j" (.constNat 0) (.constNat 3) (.constNat 1) [padding]
private def afterInner (swapped : Bool) : Program where
  kernel := .mk [] [] [.forRange "i" 0 2 1 (precedingLoop :: body swapped)]
  domain := [.range [] "i" (.constNat 0) (.constNat 2) (.constNat 1)
    (precedingLoop :: body false) (.add [precedingLoop, padding] [] x one)]

theorem static_rewrite_after_inner_loop (R : FP.ScalarArithmetic.Rules) :
    afterInner false ≡[R] afterInner true := by
  equiv_decompose
  fp_prove

/-- Occurrences count matching static and dynamic natural ranges together. -/
theorem mixed_range_lookup :
    ((Source.dynamicRange?
      [.forRange "i" 1 2 1 [], padding,
       .forRangeDyn "i" (.constNat 4) (.constNat 9) (.constNat 2) []]
      "i" 1).map fun p => (p.before, p.start, p.stop, p.stride)) =
      some ([.forRange "i" 1 2 1 [], padding], .constNat 4, .constNat 9, .constNat 2) := rfl

set_option linter.unusedVariables false in
example (R : FP.ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : ({ program false 0 2 1 with domain := [] } : Program) ≡[R]
        ({ program true 0 2 1 with domain := [] } : Program) := by
      equiv_decompose
      fp_prove
  fail_if_success
    have : ({ program false 0 2 1 with domain := [.add [] [] x one] } : Program) ≡[R]
        ({ program true 0 2 1 with domain := [.add [] [] x one] } : Program) := by
      equiv_decompose
      fp_prove
  fail_if_success
    have : program false 0 2 1 ≡[R]
        ({ program true 0 3 1 with domain := sites 0 2 1 } : Program) := by
      equiv_decompose
      fp_prove
  fail_if_success
    have : ({ afterInner false with domain := sites 0 2 1 } : Program) ≡[R]
        ({ afterInner true with domain := sites 0 2 1 } : Program) := by
      equiv_decompose
      fp_prove
  trivial

example : True := by
  fail_if_success
    have : program false 0 2 1 ≡[([] : Spec.Assumptions FP.GuardedFragment)]
        program true 0 2 1 := by
      equiv_decompose
      fp_prove
  trivial

/-- info: FP assumptions used by static_range_with_insertions:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions static_range_with_insertions
#guard_msgs (drop info) in
#auditModuleAxioms
end FPStaticRangeRewriteTests
