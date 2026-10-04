/- Conditional scalar theories and successful kernel execution.
Numerical premises apply only to admitted scalar fragments in their declared
operand domain. Program signatures retain the input contract and IO windows. -/
import VeriTile.Triton.Float.GuardedRules
import VeriTile.Triton.Float.StructuralIO
import VeriTile.Triton.Float.ExecutionProfile

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
  defaultPrecision : Option ComputeDType := none

/-- Keep explicit precision tags; optionally supply the precision of ordinary
Triton expressions, including comparisons and `where` temporaries. -/
def IO.algebra {α : Type} (io : IO) (M : Algebra α) : Algebra α :=
  match io.defaultPrecision with
  | none => M
  | some p => M.withDefaultPrecision p

def Equivalent (R : Spec.Assumptions GuardedFragment) (lhs rhs : IO) : Prop :=
  IO₁PrivateScratch lhs.io ∧ IO₁PrivateScratch rhs.io ∧
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (D : Domain α), Models R M D →
    ∀ s, lhs.domain.Holds (lhs.algebra M) D s →
      ∃ a b, Structural.exec (lhs.algebra M) lhs.io.kernel s = some a ∧
        Structural.exec (rhs.algebra M) rhs.io.kernel s = some b ∧
        (∀ i : Fin lhs.io.Bout,
          a.mem lhs.io.out (lhs.io.write (s.pids 0) + i.val) =
            b.mem rhs.io.out (rhs.io.write (s.pids 0) + i.val)) ∧
        IO₁Frame lhs.io s a ∧ IO₁Frame rhs.io s b

instance : Spec.ProgramSyntax IO where
  Statement := GuardedFragment
  Signature := IO₁Signature × Precondition × Option ComputeDType
  signature p := (io₁Signature p.io, p.domain, p.defaultPrecision)
  body p := [⟨[], p.io.kernel.surfaceBody⟩]
  structural := some (fun _ _ => False)
  numerical := Equivalent
  -- These IO proofs use successful execution, never unchecked syntax rewrites.
  sameContext := fun lhs rhs => lhs = rhs

/-- A guarded one-input/two-output contract, such as mean and variance.
Its public FP specification uses the same equivalence notation as `IO`. -/
structure IO₁ₓ₂ where
  io : KernelIO₁ₓ₂
  domain : Precondition

def Equivalent₁ₓ₂ (R : Spec.Assumptions GuardedFragment) (lhs rhs : IO₁ₓ₂) : Prop :=
  IO₁ₓ₂PrivateScratch lhs.io ∧ IO₁ₓ₂PrivateScratch rhs.io ∧
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (D : Domain α), Models R M D →
    ∀ s, lhs.domain.Holds M D s →
      ∃ a b, Structural.exec M lhs.io.kernel s = some a ∧ Structural.exec M rhs.io.kernel s = some b ∧
        IO₁ₓ₂Outputs lhs.io rhs.io s a b ∧ IO₁ₓ₂Frame lhs.io s a ∧ IO₁ₓ₂Frame rhs.io s b

/-- Structural proofs remain usable under a guarded contract without adding
any numerical assumption. The specification separately checks signatures. -/
theorem Equivalent₁ₓ₂.ofStructural (R : Spec.Assumptions GuardedFragment) {lhs rhs : IO₁ₓ₂}
    (h : Structural.IO₁ₓ₂Equiv lhs.io rhs.io) : Equivalent₁ₓ₂ R lhs rhs :=
  ⟨h.1, h.2.1, fun α _ M _ _ s _ => h.2.2 α M s⟩

instance : Spec.ProgramSyntax IO₁ₓ₂ where
  Statement := GuardedFragment
  Signature := IO₁ₓ₂Signature × Precondition
  signature p := (io₁ₓ₂Signature p.io, p.domain)
  body p := [⟨[], p.io.kernel.surfaceBody⟩]
  structural := some (fun _ _ => False)
  numerical := Equivalent₁ₓ₂
  sameContext := fun lhs rhs => lhs = rhs

end VeriTile.Triton.FP.Guarded
