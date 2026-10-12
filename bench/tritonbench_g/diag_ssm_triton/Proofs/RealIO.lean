import VeriTile.Triton.Memory.KernelSpec.StreamEmit
import VeriTile.Meta.StatementAudit
/-! Exact real IO for three streams and emitted time windows. The contract
requires finite active accesses and concludes successful flattened execution,
independent output values and preservation of every other tagged cell. -/
namespace VeriTile.Triton.StreamEmitMasked2DKernelIO₃
/-- Finite real execution, initial-array outputs and preservation outside the write footprint. -/
def Implements (io : StreamEmitMasked2DKernelIO₃)
    (f : Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      Fin io.T → Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp1, io.inp2, io.inp3, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
    (zs : Fin io.T → Fin io.B3 → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ t j →
      io.read1 pid₀ pid₁ t j < A.extent io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ t j →
      io.read2 pid₀ pid₁ t j < A.extent io.inp2) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ t j →
      io.read3 pid₀ pid₁ t j < A.extent io.inp3) →
    (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ t j →
      io.write pid₀ pid₁ t j < A.extent io.out) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ t j) = xs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ t j) = ys t j) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ t j →
      s₀.readMem io.inp3 (io.read3 pid₀ pid₁ t j) = zs t j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ t j →
          s'.readMem A.flat
              (A.addr io.out (io.write pid₀ pid₁ t j))
            = f pid₀ pid₁ xs ys zs t j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ t j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ t j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨ " f => Implements io f

theorem Implements.of_identityRounding (io : StreamEmitMasked2DKernelIO₃)
    (hreal : io.outDType = .real)
    (f : Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      Fin io.T → Fin io.C → ℝ)
    (h : ImplementsR io .triv f) : Implements io f := by
  cases io
  cases hreal
  simp only [ImplementsR, execR_triv, BlockState.readMemAs_real] at h
  simpa [Implements, RoundingModel.triv, FloatDType.ofReal, FloatDType.ofWithBot,
    WithBot.some_eq_coe] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Triton.StreamEmitMasked2DKernelIO₃
