import VeriTile.Meta.Disprove
import VeriTile.Meta.StatementAudit

/-! Refutation declarations and model-independent search: polarity, scopes,
bounds, registration and rollback. FP kernel checks are in FPDisprove.lean. -/
namespace VeriTile.Tests.Disprove

refutation literal_difference : (1 = 2) := by disprove
example : ¬ (1 = 2) := literal_difference
refutation universal_counterexample : (∀ n : Nat, n = 0) := by disprove
example : ¬ (∀ n : Nat, n = 0) := universal_counterexample
refutation dependent_counterexample : (∀ n : Nat, 0 < n → n = 1) := by disprove
example : ¬ (∀ n : Nat, 0 < n → n = 1) := dependent_counterexample
theorem finite_counterexample : ¬ (∀ i : Fin 3, i.val < 2) := by disprove
theorem bool_counterexample : ¬ (∀ b : Bool, b = Bool.false) := by disprove

-- Values beyond the default candidate range can be supplied explicitly.
theorem supplied_witness : ¬ (∀ n : Nat, n < 7) := by disprove [7]
theorem larger_search : ¬ (∀ n : Nat, n < 5) := by disprove (maxWitnesses := 6)

structure Agreement (a b : Nat → Nat) : Prop where
  equal : ∀ n, a n = b n

@[disprove_rule]
theorem refute_agreement {a b : Nat → Nat} (n : Nat) (h : a n ≠ b n) : ¬ Agreement a b :=
  fun agreement => h (agreement.equal n)

theorem custom_semantics : ¬ Agreement (fun n => n) (fun _ => 0) := by disprove

-- An unsuccessful branch must restore all unification and introduced facts.
theorem failure_is_inconclusive : True := by
  fail_if_success
    have : ¬ (1 = 1) := by disprove
  fail_if_success
    have : ¬ (∀ n : Nat, n = 0) := by disprove (maxWitnesses := 1)
  fail_if_success
    have : ¬ (∀ n : Nat, n < 5) := by disprove
  fail_if_success
    have : ¬ (1 = 2) := by disprove (maxNodes := 0)
  fail_if_success
    have : True := by disprove
  trivial

private theorem local_fact : (8 : Nat) ≠ 0 := by decide
attribute [disprove_hint] local_fact

theorem registered_fact : ¬ (8 = 0) := by disprove

@[disprove_hint] def largeWitness : Nat := 12

theorem registered_witness : ¬ (∀ n : Nat, n < 12) := by disprove

theorem false_from_context (h : False) : False := by disprove

#axiomsClean universal_counterexample
#axiomsClean dependent_counterexample
#axiomsClean finite_counterexample
#axiomsClean custom_semantics
#axiomsClean registered_witness

end VeriTile.Tests.Disprove

namespace VeriTile.Tests.Refutation

-- Parameters are retained outside the negation, including local hypotheses.
refutation with_parameters (P : Prop) (h : ¬ P) : P := by exact h
example : ∀ (P : Prop), ¬ P → ¬ P := with_parameters

-- The command accepts ordinary proof terms as well as the disprove tactic.
refutation with_term : (1 : Nat) = 0 := Nat.one_ne_zero
example : (1 : Nat) ≠ 0 := with_term

refutation polymorphic.{u} {α : Type u} (a : α) : ∀ b : α, b ≠ a := by
  intro h
  exact h a rfl
example {α : Type u} (a : α) : ¬ (∀ b : α, b ≠ a) := polymorphic a

section
variable {α : Type u} [Inhabited α]

refutation from_section : ∀ a : α, a ≠ default := by
  intro h
  exact h default rfl
example : ¬ (∀ a : α, a ≠ default) := from_section
end

-- Always negate exactly once, even if the displayed claim is already negative.
refutation negative_claim : ¬ True := by disprove
example : ¬ ¬ True := negative_claim

/-- Refutation documentation is retained by the generated theorem. -/
@[simp] refutation documented_claim : (2 : Nat) = 3 := by disprove

private refutation private_claim : (3 : Nat) = 4 := by disprove
example : (3 : Nat) ≠ 4 := private_claim

open Lean Elab Command in
run_cmd do
  let env ← getEnv
  let name := ``documented_claim
  let .thmInfo _ ← getConstInfo name |
    throwError "refutation must create a theorem"
  unless (← findDocString? env name).isSome do
    throwError "refutation lost its documentation"

#axiomsClean with_parameters
#axiomsClean polymorphic
#axiomsClean from_section
#axiomsClean negative_claim

end VeriTile.Tests.Refutation
