import VeriTile.Triton.Memory.KernelSpec.StreamAttentionReal

/-! Real correctness over explicitly encoded input cells. The existing IO
structure supplies addresses, masks and launch conditions; this relation also
pins each input cell's representation. It supports mixed real/integer streams
without changing the executed kernel or preparing replacement memory. -/
namespace VeriTile.Triton.StreamMasked3DKernelIO₃

/-- Exact execution and the full memory frame for three streams of encoded
input cells. Pinning complete cells retains integer storage and distinguishes
it from a real readback that may silently default on dtype mismatch. -/
def ImplementsCells {α β γ : Type} (io : StreamMasked3DKernelIO₃)
    (cell1 : α → MemCell) (cell2 : β → MemCell) (cell3 : γ → MemCell)
    (f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → α) →
      (Fin io.T → Fin io.B2 → β) → (Fin io.T → Fin io.B3 → γ) →
      Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp1, io.inp2, io.inp3, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀ (xs : Fin io.T → Fin io.B1 → α) (ys : Fin io.T → Fin io.B2 → β)
    (zs : Fin io.T → Fin io.B3 → γ)
    (s₀ : BlockState),
    io.pre pid₀ pid₁ pid₂ →
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      io.read1 pid₀ pid₁ pid₂ t j < A.extent io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ t j →
      io.read2 pid₀ pid₁ pid₂ t j < A.extent io.inp2) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ t j →
      io.read3 pid₀ pid₁ pid₂ t j < A.extent io.inp3) →
    (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ j →
      io.write pid₀ pid₁ pid₂ j < A.extent io.out) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      s₀.mem io.inp1 (io.read1 pid₀ pid₁ pid₂ t j) = cell1 (xs t j)) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ t j →
      s₀.mem io.inp2 (io.read2 pid₀ pid₁ pid₂ t j) = cell2 (ys t j)) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ t j →
      s₀.mem io.inp3 (io.read3 pid₀ pid₁ pid₂ t j) = cell3 (zs t j)) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ j →
          s'.readMem A.flat
              (A.addr io.out (io.write pid₀ pid₁ pid₂ j))
            = f pid₀ pid₁ pid₂ xs ys zs j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ pid₂ j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')


end VeriTile.Triton.StreamMasked3DKernelIO₃
