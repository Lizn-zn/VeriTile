import VeriTile.Triton.Memory.KernelSpec.Stream
import VeriTile.Meta.StatementAudit

/-! Exact real IO for two input streams and one output on a two-dimensional
program grid. The contract includes successful flat-memory execution, an
independent formula and preservation of every cell outside active output
writes. Identity rounding is only an internal bridge to existing proofs. -/
namespace VeriTile.Triton.StreamMasked2DKernelIO₂

/-- Exact execution, active output readback and the memory frame for two input streams. -/
def Implements (io : StreamMasked2DKernelIO₂)
    (f : Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) →
      Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp1, io.inp2, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
    (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ t j →
      io.read1 pid₀ pid₁ t j < A.extent io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ t j →
      io.read2 pid₀ pid₁ t j < A.extent io.inp2) →
    (∀ j : Fin io.C, io.writeMask pid₀ pid₁ j →
      io.write pid₀ pid₁ j < A.extent io.out) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ t j) = xs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ t j) = ys t j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.C, io.writeMask pid₀ pid₁ j →
          s'.readMem A.flat
              (A.addr io.out (io.write pid₀ pid₁ j))
            = f pid₀ pid₁ xs ys j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.C, io.writeMask pid₀ pid₁ j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')


@[inherit_doc] scoped notation:25 io " ⊨ " f => Implements io f

/-- Reuse the streamed IO transport proof at identity rounding and real storage.
The resulting contract contains exact execution and real output readback. -/
theorem Implements.of_identityRounding (io : StreamMasked2DKernelIO₂)
    (hreal : io.outDType = .real)
    (f : Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) →
      Fin io.C → ℝ)
    (h : ImplementsR io .triv f) : Implements io f := by
  cases io
  cases hreal
  simp only [ImplementsR, execR_triv, BlockState.readMemAs_real] at h
  simpa [Implements, RoundingModel.triv, FloatDType.ofReal, FloatDType.ofWithBot,
    WithBot.some_eq_coe] using h

end VeriTile.Triton.StreamMasked2DKernelIO₂

#guard_msgs (drop info) in
#auditModuleAxioms
