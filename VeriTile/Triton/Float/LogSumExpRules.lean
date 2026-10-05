import VeriTile.Triton.Float.ExponentialLaws
import VeriTile.Triton.Float.LogExp

/-! Combine existing scalar theories without introducing an application atom. -/
noncomputable section
namespace VeriTile.Triton.FP.LogSumExp
open Structural Guarded

structure Rules where
  exponential : Exponential.Rules
  logarithm : LogExp.Rules

def Rules.assumptions (R : Rules) : Spec.Assumptions GuardedFragment :=
  R.exponential.assumptions ++ R.logarithm.assumptions

instance : CoeOut Rules (Spec.Assumptions GuardedFragment) := ⟨Rules.assumptions⟩

/-- Every step remains a derivation from the same atomic entry. -/
theorem include_derivation {A B : Spec.Assumptions GuardedFragment}
    (subset : ∀ e ∈ A, e ∈ B) {lhs rhs : List GuardedFragment}
    (h : Spec.Derivation A lhs rhs) : Spec.Derivation B lhs rhs := by
  induction h with
  | refl code => exact .refl code
  | atom e he ha => exact .atom e (subset e he) ha
  | symm _ ih => exact .symm ih
  | trans _ _ ih₁ ih₂ => exact .trans ih₁ ih₂
  | frame beforeCode afterCode _ ih => exact .frame beforeCode afterCode ih

theorem exponential_models {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D) :
    Models R.exponential.assumptions M D :=
  fun lhs rhs h => hM lhs rhs (include_derivation (fun _ he => List.mem_append_left _ he) h)

theorem logarithm_models {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D) :
    Models R.logarithm.assumptions M D :=
  fun lhs rhs h => hM lhs rhs (include_derivation (fun _ he => List.mem_append_right _ he) h)

end VeriTile.Triton.FP.LogSumExp
