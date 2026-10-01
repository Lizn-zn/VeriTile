/- Structural FP steps on an IO contract. The public signature includes the
kernel's declared ports, tile lengths and address functions. Private scratch
windows are explicit and can change only through a successful structural
execution proof with a cell-level frame on each side. -/
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.Equivalence
import VeriTile.Triton.Memory.KernelSpec.Basic

namespace VeriTile.Triton.FP.Structural

structure IO₃Signature where
  ports : List RegionName × List RegionName
  in1 : RegionName
  in2 : RegionName
  in3 : RegionName
  out : RegionName
  B1 : Nat
  B2 : Nat
  B3 : Nat
  Bout : Nat
  read1 : Nat → Nat
  read2 : Nat → Nat
  read3 : Nat → Nat
  write : Nat → Nat

def ioSignature (io : KernelIO₃) : IO₃Signature where
  ports := match io.kernel with | .mk ins outs _ => (ins, outs)
  in1 := io.in1
  in2 := io.in2
  in3 := io.in3
  out := io.out
  B1 := io.B1
  B2 := io.B2
  B3 := io.B3
  Bout := io.Bout
  read1 := io.read1
  read2 := io.read2
  read3 := io.read3
  write := io.write

def PrivateScratch (io : KernelIO₃) : Prop :=
  ∀ p ∈ io.scratch, p.buf ≠ io.in1 ∧ p.buf ≠ io.in2 ∧ p.buf ≠ io.in3 ∧ p.buf ≠ io.out

/-- Preserve every cell outside this implementation's output and scratch
windows. In particular, another cell in a scratch region is not hidden. -/
def Frame {α : Type} (io : KernelIO₃) (before after : State α) : Prop :=
  ∀ (r : RegionName) o,
    (r ≠ io.out ∨ ∀ i : Fin io.Bout, o ≠ io.write (before.pids 0) + i.val) →
    (∀ p ∈ io.scratch, r = p.buf → ∀ i : Fin p.len, o ≠ p.win (before.pids 0) + i.val) →
    after.mem r o = before.mem r o

/-- A structural equality must work for every floating carrier, every
interpretation of the numerical operations and every initial state. Both runs
must succeed. Only the explicitly private scratch windows may differ. This
supplies structural steps of the FP theory, not an IEEE execution theorem. -/
def IO₃Equiv (lhs rhs : KernelIO₃) : Prop :=
  PrivateScratch lhs ∧ PrivateScratch rhs ∧
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (s : State α),
    ∃ a b, exec M lhs.kernel s = some a ∧ exec M rhs.kernel s = some b ∧
      (∀ i : Fin lhs.Bout,
        a.mem lhs.out (lhs.write (s.pids 0) + i.val) =
          b.mem rhs.out (rhs.write (s.pids 0) + i.val)) ∧
      Frame lhs s a ∧ Frame rhs s b

end VeriTile.Triton.FP.Structural

namespace VeriTile.Triton

/-- The same `≡[R]` surface now also supports structural transformations on an
explicit IO contract. Numeric atoms still use the existing statement calculus. -/
instance kernelIO₃FPProgramSyntax : Spec.ProgramSyntax KernelIO₃ where
  Statement := ComputeStmt
  Signature := FP.Structural.IO₃Signature
  signature := FP.Structural.ioSignature
  body := fun io => io.kernel.surfaceBody
  structural := some FP.Structural.IO₃Equiv
  sameContext := fun lhs rhs => lhs.scratch = rhs.scratch

end VeriTile.Triton
