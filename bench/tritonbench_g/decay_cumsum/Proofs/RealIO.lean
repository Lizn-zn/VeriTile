import VeriTile.Triton.Memory.KernelSpec.StreamEmit
import VeriTile.Meta.StatementAudit

/-! Exact real IO for one input stream with in-loop output on a three-dimensional
program grid. The contract includes successful flat-memory execution, an
independent formula and preservation of every cell outside active output
writes. Identity rounding is only an internal bridge to existing proofs. -/
namespace VeriTile.Triton.StreamEmitMasked3DKernelIO₁

/-- Exact execution, active output readback and the memory frame for one input stream and each emitted window. -/
def Implements (io : StreamEmitMasked3DKernelIO₁)
    (f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      Fin io.T → Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp1, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀ (xs : Fin io.T → Fin io.B1 → ℝ)
    (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      io.read1 pid₀ pid₁ pid₂ t j < A.extent io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ pid₂ t j →
      io.write pid₀ pid₁ pid₂ t j < A.extent io.out) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ pid₂ t j) = xs t j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ pid₂ t j →
          s'.readMem A.flat
              (A.addr io.out (io.write pid₀ pid₁ pid₂ t j))
            = f pid₀ pid₁ pid₂ xs t j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ pid₂ t j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ pid₂ t j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')


@[inherit_doc] scoped notation:25 io " ⊨ " f => Implements io f

/-- Reuse the streamed IO transport proof at identity rounding and real storage.
The resulting contract contains exact execution and real output readback. -/
theorem Implements.of_identityRounding (io : StreamEmitMasked3DKernelIO₁)
    (hreal : io.outDType = .real)
    (f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      Fin io.T → Fin io.C → ℝ)
    (h : ImplementsR io .triv f) : Implements io f := by
  cases io
  cases hreal
  simp only [ImplementsR, execR_triv, BlockState.readMemAs_real] at h
  simpa [Implements, RoundingModel.triv, FloatDType.ofReal, FloatDType.ofWithBot,
    WithBot.some_eq_coe] using h

end VeriTile.Triton.StreamEmitMasked3DKernelIO₁

#guard_msgs (drop info) in
#auditModuleAxioms
