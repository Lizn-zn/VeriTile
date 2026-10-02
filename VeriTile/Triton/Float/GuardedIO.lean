/- Conditional scalar theories and successful kernel execution.
Numerical premises apply only to admitted scalar fragments in their declared
operand domain. Program signatures retain the input contract and IO windows. -/
import VeriTile.Triton.Float.GuardedRules
import VeriTile.Triton.Float.StructuralIO

namespace VeriTile.Triton.FP.Guarded
open Structural

/-- Domain predicates are interpreted alongside the opaque FP operations. -/
abbrev Domain (α : Type) := GuardKind → α → Prop

def ScalarDomain {α : Type} (D : Domain α) (guards : List OperandGuard)
    (s : State α) : Prop :=
  ∀ g ∈ guards, ∃ v, s.regs .real [] g.operand = some v ∧ D g.kind (v PUnit.unit)

/-- A model of the admitted scalar theory, observed at the scalar `out` register.
This does not assert that the theory is an exact IEEE identity. Both executions
must exist, and both operand domains must hold, before a law can be used. -/
def Models {α : Type} [Inhabited α] (R : Spec.Assumptions GuardedFragment)
    (M : Algebra α) (D : Domain α) : Prop :=
  ∀ lhs rhs, Spec.Derivation R [lhs] [rhs] →
    ∀ s, ScalarDomain D lhs.guards s → ScalarDomain D rhs.guards s →
      ∀ a b, run M lhs.code s = some a → run M rhs.code s = some b →
        a.regs .real [] "out" = b.regs .real [] "out"

structure RegisterGuard where
  name : RegName
  shape : TileShape
  kind : GuardKind

/-- A domain contract checks intermediate operands of a common prefix. It is
part of the public signature; changing it is not an equivalence step. -/
structure Precondition where
  code : List ComputeStmt
  guards : List RegisterGuard

def Precondition.Holds {α : Type} [Inhabited α] (P : Precondition)
    (M : Algebra α) (D : Domain α) (s : State α) : Prop :=
  ∃ t, run M P.code s = some t ∧
    ∀ g ∈ P.guards, ∃ v, t.regs .real g.shape g.name = some v ∧ ∀ i, D g.kind (v i)

structure IO where
  io : KernelIO₁
  domain : Precondition

def Equivalent (R : Spec.Assumptions GuardedFragment) (lhs rhs : IO) : Prop :=
  IO₁PrivateScratch lhs.io ∧ IO₁PrivateScratch rhs.io ∧
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (D : Domain α), Models R M D →
    ∀ s, lhs.domain.Holds M D s →
      ∃ a b, Structural.exec M lhs.io.kernel s = some a ∧ Structural.exec M rhs.io.kernel s = some b ∧
        (∀ i : Fin lhs.io.Bout,
          a.mem lhs.io.out (lhs.io.write (s.pids 0) + i.val) =
            b.mem rhs.io.out (rhs.io.write (s.pids 0) + i.val)) ∧
        IO₁Frame lhs.io s a ∧ IO₁Frame rhs.io s b

instance : Spec.ProgramSyntax IO where
  Statement := GuardedFragment
  Signature := IO₁Signature × Precondition
  signature p := (io₁Signature p.io, p.domain)
  body p := [⟨[], p.io.kernel.surfaceBody⟩]
  structural := some (fun _ _ => False)
  numerical := Equivalent
  -- These IO proofs use successful execution, never unchecked syntax rewrites.
  sameContext := fun lhs rhs => lhs = rhs

end VeriTile.Triton.FP.Guarded
