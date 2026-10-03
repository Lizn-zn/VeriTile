/- Public FP IO contracts with an explicit execution profile and syntactic
domain checks. Numerical kernels still use the existing equivalence notation. -/
import VeriTile.Triton.Float.GuardExpression
import VeriTile.Triton.Float.ScalarReduction
import VeriTile.Triton.Float.ExecutionProfile

namespace VeriTile.Triton.FP.Scheduled
open Structural Guarded
open Equational (Schedules)

/-- Preserve whether sums are opaque or expanded as fp32 addition trees,
and the default precision of operations without an explicit tag. -/
structure Profile where
  defaultPrecision : ComputeDType
  fp32SumTrees : Bool

def Profile.algebra {α : Type} (profile : Profile) (M : Algebra α) (plans : Schedules) : Algebra α :=
  let base := if profile.fp32SumTrees then ScalarReduction.algebra M plans else M
  base.withDefaultPrecision profile.defaultPrecision

def fp32 : Profile := ⟨.fp32, true⟩

structure IO₁ₓ₂ where
  io : KernelIO₁ₓ₂
  profile : Profile
  domain : Schedules → GuardExpression.Condition

/-- Domain expressions use the base numerical algebra. The domain generator
receives schedules as data and can explicitly expand their intermediate trees.
It receives no floating values, operation interpretation or equality oracle. -/
def Equivalent₁ₓ₂ (R : Spec.Assumptions GuardedFragment) (lhs rhs : IO₁ₓ₂) : Prop :=
  IO₁ₓ₂PrivateScratch lhs.io ∧ IO₁ₓ₂PrivateScratch rhs.io ∧
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (D : Domain α), Models R M D →
    ∀ (plans : Schedules) (s : State α), (lhs.domain plans).Holds M D s →
      ∃ a b, Structural.exec (lhs.profile.algebra M plans) lhs.io.kernel s = some a ∧
        Structural.exec (rhs.profile.algebra M plans) rhs.io.kernel s = some b ∧
        IO₁ₓ₂Outputs lhs.io rhs.io s a b ∧ IO₁ₓ₂Frame lhs.io s a ∧ IO₁ₓ₂Frame rhs.io s b

/-- Structural execution proofs work for any shared profile and introduce
no numerical assumptions. The public signature also retains that profile. -/
theorem Equivalent₁ₓ₂.ofStructural (R : Spec.Assumptions GuardedFragment) {lhs rhs : IO₁ₓ₂}
    (hp : lhs.profile = rhs.profile) (h : Structural.IO₁ₓ₂Equiv lhs.io rhs.io) :
    Equivalent₁ₓ₂ R lhs rhs := by
  refine ⟨h.1, h.2.1, ?_⟩
  intro α _ M _ _ plans s _
  simpa only [← hp] using h.2.2 α (lhs.profile.algebra M plans) s

instance : Spec.ProgramSyntax IO₁ₓ₂ where
  Statement := GuardedFragment
  Signature := IO₁ₓ₂Signature × Profile × (Schedules → GuardExpression.Condition)
  signature p := (io₁ₓ₂Signature p.io, p.profile, p.domain)
  body p := [⟨[], p.io.kernel.surfaceBody⟩]
  structural := some (fun _ _ => False)
  numerical := Equivalent₁ₓ₂
  sameContext := fun lhs rhs => lhs = rhs

end VeriTile.Triton.FP.Scheduled
