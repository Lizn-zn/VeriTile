import VeriTile.Triton.Float.ScheduledIO

/-! Scheduled kernel contracts with explicit comparison support.
A comparison requirement states only that the interpreter implements that
operation at the requested precision. It constrains neither its Boolean result
nor any numeric equality. Finite/positive operand checks remain in the domain.
-/
namespace VeriTile.Triton.FP.Scheduled
open Structural Guarded
open Equational (Schedules)

inductive Comparison where
  | lt (precision : ComputeDType) (dtype : FloatDType)
  | le (precision : ComputeDType) (dtype : FloatDType)
  deriving DecidableEq, Repr

def Comparison.Supported {α : Type} (M : Algebra α) : Comparison → Prop
  | .lt p d => (M.compareLt (some p) d).isSome = true
  | .le p d => (M.compareLe (some p) d).isSome = true

/-- An execution interface for kernels that branch as well as reduce.
Comparison requirements belong to the signature shared by both programs. -/
structure ConditionalIO₁ extends IO₁ where
  comparisons : List Comparison

def ConditionalEquivalent₁ (R : Spec.Assumptions GuardedFragment)
    (lhs rhs : ConditionalIO₁) : Prop :=
  IO₁PrivateScratch lhs.io ∧ IO₁PrivateScratch rhs.io ∧
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (D : Domain α), Models R M D →
    (∀ c ∈ lhs.comparisons, c.Supported M) →
    ∀ (plans : Schedules) (s : State α), (lhs.domain plans).Holds M D s →
      ∃ a b, Structural.exec (lhs.profile.algebra M plans) lhs.io.kernel s = some a ∧
        Structural.exec (rhs.profile.algebra M plans) rhs.io.kernel s = some b ∧
        (∀ i : Fin lhs.io.Bout,
          a.mem lhs.io.out (lhs.io.write (s.pids 0) + i.val) =
            b.mem rhs.io.out (rhs.io.write (s.pids 0) + i.val)) ∧
        IO₁Frame lhs.io s a ∧ IO₁Frame rhs.io s b

instance : Spec.ProgramSyntax ConditionalIO₁ where
  Statement := GuardedFragment
  Signature := IO₁Signature × Profile × (Schedules → GuardExpression.Condition) × List Comparison
  signature p := (io₁Signature p.io, p.profile, p.domain, p.comparisons)
  body p := [⟨[], p.io.kernel.surfaceBody⟩]
  structural := some (fun _ _ => False)
  numerical := ConditionalEquivalent₁
  sameContext := fun lhs rhs => lhs = rhs

end VeriTile.Triton.FP.Scheduled
