import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Signed-loop decomposition must retain negative visits, actual prefixes
and guards at later iterations. Natural-loop guards cannot stand in for it. -/
namespace FPIntRangeRewriteTests
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

private def x : Op .real [] := .ref .real [] "x"
private def one : Op .real [] := .const 1
private def body (swapped : Bool) : List ComputeStmt :=
  [.assign .real [] "x" (.alg
    (if swapped then .add .real .nil one x else .add .real .nil x one))]
private def inner : Site := .add [] [] x one
private def site (start stop : Int) (stride : Nat) : Site :=
  .intRange [] "i" (.constInt start) (.constInt stop) (.constNat stride) (body false) inner
private def program (swapped : Bool) (start stop : Int) (stride : Nat) : Program where
  kernel := .mk [] [] [.forRangeInt "i" (.constInt start) (.constInt stop)
    (.constNat stride) (body swapped)]
  domain := [site start stop stride]

theorem signed_loop_equiv (R : FP.ScalarArithmetic.Rules) (start stop : Int) (stride : Nat) :
    program false start stop stride ≡[R] program true start stop stride := by
  equiv_decompose
  all_goals fp_prove

/-- info: FP assumptions used by signed_loop_equiv:
---
info:   add_commute -/
#guard_msgs in
#print_fp_assumptions signed_loop_equiv

private def padding : ComputeStmt := .assign .int [] "padding" (.alg (.constInt (-17)))
private def code (swapped : Bool) : List ComputeStmt :=
  [padding, .forRangeInt "i" (.constInt (-2)) (.constInt 1) (.constNat 1)
    ([padding] ++ body swapped), padding]
private def located : List Site :=
  match Source.intRange? (code false) "i" with
  | none => []
  | some loop => [.intRange loop.before "i" loop.start loop.stop loop.stride loop.body
      (.add (Source.prefixBefore loop.body "x") [] x one)]
private def padded (swapped : Bool) : Program := ⟨.mk [] [] (code swapped), located⟩

theorem insertions_keep_signed_context (R : FP.ScalarArithmetic.Rules) :
    padded false ≡[R] padded true := by
  equiv_decompose
  all_goals fp_prove

private def nested (swapped : Bool) (selected : Bool) : Program where
  kernel := .mk [] [] [.ifThen (.constBool selected)
    [.forRangeDyn "j" (.constNat 0) (.constNat 2) (.constNat 1) (code swapped)]]
  domain := [.branch [] (.constBool selected) true
    (.range [] "j" (.constNat 0) (.constNat 2) (.constNat 1) (code false)
      (.intRange [padding] "i" (.constInt (-2)) (.constInt 1) (.constNat 1)
        ([padding] ++ body false) (.add [padding] [] x one)))]

theorem nested_signed_loop (R : FP.ScalarArithmetic.Rules) (selected : Bool) :
    nested false selected ≡[R] nested true selected := by
  equiv_decompose
  all_goals fp_prove

set_option linter.unusedVariables false in
example (R : FP.ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : ({ program false (-2) 1 1 with domain := [] } : Program) ≡[R]
        ({ program true (-2) 1 1 with domain := [] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  fail_if_success
    have : ({ program false (-2) 1 1 with domain := [inner] } : Program) ≡[R]
        ({ program true (-2) 1 1 with domain := [inner] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  fail_if_success
    have : program false (-2) 1 1 ≡[R]
        ({ program true 0 1 1 with domain := [site (-2) 1 1] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  fail_if_success
    have : ({ padded false with domain := [site (-2) 1 1] } : Program) ≡[R]
        ({ padded true with domain := [site (-2) 1 1] } : Program) := by
      equiv_decompose
      all_goals fp_prove
  let natural := Site.range [] "i" (.constNat 0) (.constNat 1) (.constNat 1)
    (body false) inner
  fail_if_success
    have : ({ program false (-2) 1 1 with domain := [natural] } : Program) ≡[R]
        ({ program true (-2) 1 1 with domain := [natural] } : Program) := by
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
  (initial.setReg "i" .int [] (fun _ => -2)).setReg "x" .real [] (fun _ => 2)
private def bounded : FP.Guarded.Domain Nat
  | .finite, a => a ≤ 1
  | _, _ => True

theorem first_operands_are_finite : inner.Holds model bounded
    (initial.setReg "i" .int [] (fun _ => -2)) := by
  intro t ht a b ha hb
  simp only [run, Option.some.injEq] at ht
  subst t
  simp [x, one, FP.Structural.evalOp, initial] at ha hb
  subst a
  subst b
  simp [engine, model, Algebra.withDefaultPrecision, bounded]

theorem negative_iteration_reaches_next :
    IntRangeReachable (engine model) "i" 1 1 (body false) (-2) initial (-1) next := by
  apply IntRangeReachable.next .initial (by decide) (by decide)
  simp [run, step, body, FP.Structural.evalExpr, FP.Structural.evalOp, x, one,
    engine, model, Algebra.withDefaultPrecision, numeric, initial, next]
  rfl

theorem entry_guard_does_not_cover_later_negative_index :
    ¬ (site (-2) 1 1).Holds model bounded initial := by
  intro hd
  have hh := hd initial (by simp [run]) (fun _ => -2) (fun _ => 1) (fun _ => 1)
    (by rfl) (by rfl) (by rfl) (-1) next negative_iteration_reaches_next
    (by decide) (by decide)
  have bad := hh (next.setReg "i" .int [] (fun _ => -1)) (by simp [run])
    (fun _ => 2) (fun _ => 1)
    (by simp [x, FP.Structural.evalOp, next]) (by rfl)
  exact (by decide : ¬ (2 ≤ 1)) (bad.1 PUnit.unit)

theorem zero_stride_needs_no_operands {α : Type} [Inhabited α]
    (M : Algebra α) (D : FP.Guarded.Domain α) (s : State α) (start stop : Int) :
    (site start stop 0).Holds M D s := by
  intro t ht a b d ha hb hd cur u hr hz hc
  have he : d = fun _ => 0 := Option.some.inj hd.symm
  exact False.elim (hz (by rw [he]))

#guard_msgs (drop info) in
#auditModuleAxioms
end FPIntRangeRewriteTests
