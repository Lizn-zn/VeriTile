/- Read-only row observations after successful kernel execution. An observation
may load a stored row or evaluate an expression using the final registers.
The relation compares these values, with each kernel's memory frame stated
separately; it does not equate their final memories or add an output store. -/
import VeriTile.Triton.Float.ScheduledIO

namespace VeriTile.Triton.FP.ObservedRow
open Structural Guarded
open Equational (Schedules)

structure Program where
  input : RegionName
  output : RegionName
  size : Nat
  offset : Nat → Nat
  kernel : ComputeKernel
  /-- The first argument is the original program ID. Each observation returns
  a real-typed scalar, with numerical calls interpreted at `profile` precision. -/
  observe : Nat → Fin size → Op .real []
  writesOutput : Bool
  profile : Scheduled.Profile
  domain : Schedules → GuardExpression.Condition

def Frame {α : Type} [Inhabited α] (p : Program) (s t : State α) : Prop :=
  ∀ r o, (p.writesOutput = false ∨ r ≠ p.output ∨
    ∀ i : Fin p.size, o ≠ p.offset (s.pids 0) + i.val) → t.mem r o = s.mem r o

/-- Both kernels and every readback must succeed. A missing register cannot
turn a failed readback into an equal pair of `none` results. -/
def Equivalent (R : Spec.Assumptions GuardedFragment) (lhs rhs : Program) : Prop :=
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (D : Domain α), Models R M D →
    ∀ (plans : Schedules) (s : State α), (lhs.domain plans).Holds M D s →
      ∃ a b, ∃ values : Nat → α,
        Structural.exec (lhs.profile.algebra M plans) lhs.kernel s = some a ∧
        Structural.exec (rhs.profile.algebra M plans) rhs.kernel s = some b ∧
        (∀ i : Fin lhs.size, Structural.evalOp (lhs.profile.algebra M plans) none
          (lhs.observe (s.pids 0) i) a = some (fun _ => values i.val)) ∧
        (∀ i : Fin rhs.size, Structural.evalOp (rhs.profile.algebra M plans) none
          (rhs.observe (s.pids 0) i) b = some (fun _ => values i.val)) ∧
        Frame lhs s a ∧ Frame rhs s b

instance : Spec.ProgramSyntax Program where
  Statement := GuardedFragment
  Signature := RegionName × RegionName × Nat × (Nat → Nat) × Scheduled.Profile ×
    (Schedules → GuardExpression.Condition)
  signature p := (p.input, p.output, p.size, p.offset, p.profile, p.domain)
  body p := [⟨[], p.kernel.surfaceBody⟩]
  structural := some (fun _ _ => False)
  numerical := Equivalent
  -- Kernel-body syntax alone cannot change an observation or memory frame.
  sameContext := fun lhs rhs => lhs = rhs

end VeriTile.Triton.FP.ObservedRow
