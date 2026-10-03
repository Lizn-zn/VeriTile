/- Structural FP steps on an IO contract. The public signature includes the
kernel's declared ports, tile lengths and address functions. Private scratch
windows are explicit and can change only through a successful structural
execution proof with a cell-level frame on each side. -/
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.TermModel
import VeriTile.Triton.Float.Equivalence
import VeriTile.Triton.Memory.KernelSpec.Basic
import VeriTile.Triton.Memory.KernelSpec.Masked

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

structure IO₁Signature where
  ports : List RegionName × List RegionName
  inp : RegionName
  out : RegionName
  Bin : Nat
  Bout : Nat
  read : Nat → Nat
  write : Nat → Nat

def io₁Signature (io : KernelIO₁) : IO₁Signature where
  ports := match io.kernel with | .mk ins outs _ => (ins, outs)
  inp := io.inp
  out := io.out
  Bin := io.Bin
  Bout := io.Bout
  read := io.read
  write := io.write

def IO₁PrivateScratch (io : KernelIO₁) : Prop :=
  ∀ p ∈ io.scratch, p.buf ≠ io.inp ∧ p.buf ≠ io.out

def IO₁Frame {α : Type} (io : KernelIO₁) (before after : State α) : Prop :=
  ∀ (r : RegionName) o,
    (r ≠ io.out ∨ ∀ i : Fin io.Bout, o ≠ io.write (before.pids 0) + i.val) →
    (∀ p ∈ io.scratch, r = p.buf → ∀ i : Fin p.len, o ≠ p.win (before.pids 0) + i.val) →
    after.mem r o = before.mem r o

def IO₁Equiv (lhs rhs : KernelIO₁) : Prop :=
  IO₁PrivateScratch lhs ∧ IO₁PrivateScratch rhs ∧
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (s : State α),
    ∃ a b, exec M lhs.kernel s = some a ∧ exec M rhs.kernel s = some b ∧
      (∀ i : Fin lhs.Bout,
        a.mem lhs.out (lhs.write (s.pids 0) + i.val) =
          b.mem rhs.out (rhs.write (s.pids 0) + i.val)) ∧
      IO₁Frame lhs s a ∧ IO₁Frame rhs s b

/-- Both output identities, sizes and address functions belong to the public
signature. A proof cannot hide one output by shortening its window. -/
structure IO₁ₓ₂Signature where
  ports : List RegionName × List RegionName
  inp : RegionName
  out1 : RegionName
  out2 : RegionName
  Bin : Nat
  Bout1 : Nat
  Bout2 : Nat
  read : Nat → Nat
  write1 : Nat → Nat
  write2 : Nat → Nat

def io₁ₓ₂Signature (io : KernelIO₁ₓ₂) : IO₁ₓ₂Signature where
  ports := match io.kernel with | .mk ins outs _ => (ins, outs)
  inp := io.inp
  out1 := io.out1
  out2 := io.out2
  Bin := io.Bin
  Bout1 := io.Bout1
  Bout2 := io.Bout2
  read := io.read
  write1 := io.write1
  write2 := io.write2

def IO₁ₓ₂PrivateScratch (io : KernelIO₁ₓ₂) : Prop :=
  ∀ p ∈ io.scratch, p.buf ≠ io.inp ∧ p.buf ≠ io.out1 ∧ p.buf ≠ io.out2

/-- Only the union of the two output windows and declared private scratch
windows may change. Other offsets in any of those regions stay observable. -/
def IO₁ₓ₂Frame {α : Type} (io : KernelIO₁ₓ₂) (before after : State α) : Prop :=
  ∀ (r : RegionName) o,
    (r ≠ io.out1 ∨ ∀ i : Fin io.Bout1, o ≠ io.write1 (before.pids 0) + i.val) →
    (r ≠ io.out2 ∨ ∀ i : Fin io.Bout2, o ≠ io.write2 (before.pids 0) + i.val) →
    (∀ p ∈ io.scratch, r = p.buf → ∀ i : Fin p.len, o ≠ p.win (before.pids 0) + i.val) →
    after.mem r o = before.mem r o

/-- Both complete output windows are observed with their dtype tags. -/
def IO₁ₓ₂Outputs {α : Type} (lhs rhs : KernelIO₁ₓ₂) (before a b : State α) : Prop :=
  (∀ i : Fin lhs.Bout1,
    a.mem lhs.out1 (lhs.write1 (before.pids 0) + i.val) =
      b.mem rhs.out1 (rhs.write1 (before.pids 0) + i.val)) ∧
  (∀ i : Fin lhs.Bout2,
    a.mem lhs.out2 (lhs.write2 (before.pids 0) + i.val) =
      b.mem rhs.out2 (rhs.write2 (before.pids 0) + i.val))

def IO₁ₓ₂Equiv (lhs rhs : KernelIO₁ₓ₂) : Prop :=
  IO₁ₓ₂PrivateScratch lhs ∧ IO₁ₓ₂PrivateScratch rhs ∧
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (s : State α),
    ∃ a b, exec M lhs.kernel s = some a ∧ exec M rhs.kernel s = some b ∧
      IO₁ₓ₂Outputs lhs rhs s a b ∧ IO₁ₓ₂Frame lhs s a ∧ IO₁ₓ₂Frame rhs s b

/-- Floating values are related by the generated term theory; discrete values
and dtype tags still have to agree exactly. -/
def ValueRelated (R : Spec.Assumptions ComputeStmt) : (d : TileDType) →
    Value (Equational.Term α) d → Value (Equational.Term α) d → Prop
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => Equational.TermEq R
  | .nat | .int | .bool | .ptr | .blockPtr => Eq

def CellRelated (R : Spec.Assumptions ComputeStmt)
    (a b : Cell (Equational.Term α)) : Prop :=
  ∃ d va vb, a = .mk d va ∧ b = .mk d vb ∧ ValueRelated R d va vb

/-- Successful executions with outputs derived equivalent from scalar atoms.
All addition schedules are quantified over; padding leaves, precision and
non-addition primitives remain in the terms. Frames retain exact cells. -/
def IO₁NumericalEquiv (R : Spec.Assumptions ComputeStmt) (lhs rhs : KernelIO₁) : Prop :=
  IO₁PrivateScratch lhs ∧ IO₁PrivateScratch rhs ∧
  ∀ (α : Type) [Inhabited α] (plans : Equational.Schedules) (s : State (Equational.Term α)),
    ∃ a b, exec (Equational.algebra plans) lhs.kernel s = some a ∧
      exec (Equational.algebra plans) rhs.kernel s = some b ∧
      (∀ i : Fin lhs.Bout,
        CellRelated R (a.mem lhs.out (lhs.write (s.pids 0) + i.val))
          (b.mem rhs.out (rhs.write (s.pids 0) + i.val))) ∧
      IO₁Frame lhs s a ∧ IO₁Frame rhs s b

/-- Mask and tile length belong to the public signature. A derivation cannot
weaken the output obligations by changing its active lanes. -/
structure MaskedIO₂Signature where
  ports : List RegionName × List RegionName
  in1 : RegionName
  in2 : RegionName
  out : RegionName
  B : Nat
  read1 : Nat → Nat
  read2 : Nat → Nat
  write : Nat → Nat
  mask : Nat → Fin B → Prop

def maskedIOSignature (io : MaskedKernelIO₂) : MaskedIO₂Signature where
  ports := match io.kernel with | .mk ins outs _ => (ins, outs)
  in1 := io.in1
  in2 := io.in2
  out := io.out
  B := io.B
  read1 := io.read1
  read2 := io.read2
  write := io.write
  mask := io.mask

def MaskedPrivateScratch (io : MaskedKernelIO₂) : Prop :=
  ∀ p ∈ io.scratch, p.1 ≠ io.in1 ∧ p.1 ≠ io.in2 ∧ p.1 ≠ io.out

/-- Inactive lanes are framed, even inside a declared output or scratch tile. -/
def MaskedFrame {α : Type} (io : MaskedKernelIO₂) (before after : State α) : Prop :=
  ∀ (r : RegionName) o,
    (r ≠ io.out ∨ ∀ i : Fin io.B, io.mask (before.pids 0) i →
      o ≠ io.write (before.pids 0) + i.val) →
    (∀ p ∈ io.scratch, r = p.1 → ∀ i : Fin io.B, io.mask (before.pids 0) i →
      o ≠ p.2 (before.pids 0) + i.val) →
    after.mem r o = before.mem r o

def MaskedIO₂Equiv (lhs rhs : MaskedKernelIO₂) : Prop :=
  MaskedPrivateScratch lhs ∧ MaskedPrivateScratch rhs ∧
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (s : State α),
    ∃ a b, exec M lhs.kernel s = some a ∧ exec M rhs.kernel s = some b ∧
      (∀ i : Fin lhs.B, lhs.mask (s.pids 0) i →
        a.mem lhs.out (lhs.write (s.pids 0) + i.val) =
          b.mem rhs.out (rhs.write (s.pids 0) + i.val)) ∧
      MaskedFrame lhs s a ∧ MaskedFrame rhs s b

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

instance kernelIO₁FPProgramSyntax : Spec.ProgramSyntax KernelIO₁ where
  Statement := ComputeStmt
  Signature := FP.Structural.IO₁Signature
  signature := FP.Structural.io₁Signature
  body := fun io => io.kernel.surfaceBody
  structural := some FP.Structural.IO₁Equiv
  numerical := FP.Structural.IO₁NumericalEquiv
  sameContext := fun lhs rhs => lhs.scratch = rhs.scratch

instance kernelIO₁ₓ₂FPProgramSyntax : Spec.ProgramSyntax KernelIO₁ₓ₂ where
  Statement := ComputeStmt
  Signature := FP.Structural.IO₁ₓ₂Signature
  signature := FP.Structural.io₁ₓ₂Signature
  body := fun io => io.kernel.surfaceBody
  structural := some FP.Structural.IO₁ₓ₂Equiv
  sameContext := fun lhs rhs => lhs.scratch = rhs.scratch

instance maskedKernelIO₂FPProgramSyntax : Spec.ProgramSyntax MaskedKernelIO₂ where
  Statement := ComputeStmt
  Signature := FP.Structural.MaskedIO₂Signature
  signature := FP.Structural.maskedIOSignature
  body := fun io => io.kernel.surfaceBody
  structural := some FP.Structural.MaskedIO₂Equiv
  sameContext := fun lhs rhs => lhs.scratch = rhs.scratch

end VeriTile.Triton
