import VeriTile.Triton.Memory.KernelSpec.StreamEmit
import VeriTile.Meta.StatementAudit
namespace VeriTile.Triton.StreamGroupedEmitMasked3DKernelIO
/-! Exact grouped real IO: successful finite flat execution, independent
per-channel formulas and the complete tagged-cell frame. The shared adapter's
zero unspecified-value condition is explicit in this contract. -/
/-- Successful exact real execution, output formulas and the complete memory frame. -/
def Implements (io : StreamGroupedEmitMasked3DKernelIO)
    (f : Nat → Nat → Nat → (Fin io.nIn → Fin io.T → Fin io.B → ℝ) →
      Fin io.nOut → Fin io.T → Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = io.bufs →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀ (xs : Fin io.nIn → Fin io.T → Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (i : Fin io.nIn) (t : Fin io.T) (j : Fin io.B),
      io.readMask i pid₀ pid₁ pid₂ t j →
      io.read i pid₀ pid₁ pid₂ t j < A.extent (io.inp i)) →
    (∀ (o : Fin io.nOut) (t : Fin io.T) (j : Fin io.B),
      io.writeMask o pid₀ pid₁ pid₂ t j →
      io.write o pid₀ pid₁ pid₂ t j < A.extent (io.out o)) →
    (∀ (i : Fin io.nIn) (t : Fin io.T) (j : Fin io.B),
      io.readMask i pid₀ pid₁ pid₂ t j →
      s₀.readMem (io.inp i) (io.read i pid₀ pid₁ pid₂ t j) = xs i t j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ (o : Fin io.nOut) (t : Fin io.T) (j : Fin io.B),
          io.writeMask o pid₀ pid₁ pid₂ t j →
          s'.readMem A.flat
              (A.addr (io.out o) (io.write o pid₀ pid₁ pid₂ t j))
            = f pid₀ pid₁ pid₂ xs o t j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ (o : Fin io.nOut) (t : Fin io.T) (j : Fin io.B),
              io.writeMask o pid₀ pid₁ pid₂ t j →
              o' ≠ A.addr (io.out o) (io.write o pid₀ pid₁ pid₂ t j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')


@[inherit_doc] scoped notation:25 io " ⊨ " f => Implements io f

theorem Implements.of_identityRounding (io : StreamGroupedEmitMasked3DKernelIO)
    (hreal : io.outDType = .real)
    (f : Nat → Nat → Nat → (Fin io.nIn → Fin io.T → Fin io.B → ℝ) →
      Fin io.nOut → Fin io.T → Fin io.B → ℝ)
    (h : ImplementsR io .triv f) : Implements io f := by
  cases io
  cases hreal
  simp only [ImplementsR,execR_triv,BlockState.readMemAs_real] at h
  simpa [Implements,RoundingModel.triv,FloatDType.ofReal,FloatDType.ofWithBot,
    WithBot.some_eq_coe] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Triton.StreamGroupedEmitMasked3DKernelIO
