import VeriTile.Triton.Memory.KernelSpec.StreamAttention

/-! Exact IO correctness for real kernel projections: three, four or five input streams with
one output, or three input streams with two outputs. Contracts include
successful flat-memory execution, independent output formulas and preservation
of every cell outside the active output windows. -/
namespace VeriTile.Triton.StreamMasked3DKernelIO₃

/-- Exact execution, active output readback and the memory frame for three input streams. -/
def Implements (io : StreamMasked3DKernelIO₃)
    (f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp1, io.inp2, io.inp3, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀ (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
    (zs : Fin io.T → Fin io.B3 → ℝ)
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
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ pid₂ t j) = xs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ pid₂ t j) = ys t j) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp3 (io.read3 pid₀ pid₁ pid₂ t j) = zs t j) →
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


@[inherit_doc] scoped notation:25 io " ⊨ " f => Implements io f

/-- Reuse the streamed IO transport proof at identity rounding and real storage.
The resulting contract contains exact execution and real output readback. -/
theorem Implements.of_identityRounding (io : StreamMasked3DKernelIO₃)
    (hreal : io.outDType = .real)
    (f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      Fin io.C → ℝ)
    (h : ImplementsR io .triv f) : Implements io f := by
  cases io
  cases hreal
  simp only [ImplementsR, execR_triv, BlockState.readMemAs_real] at h
  simpa [Implements, RoundingModel.triv, FloatDType.ofReal, FloatDType.ofWithBot,
    WithBot.some_eq_coe] using h

end VeriTile.Triton.StreamMasked3DKernelIO₃

namespace VeriTile.Triton.StreamMasked3DKernelIO₄

/-- Exact execution, active output readback and the memory frame for four input streams. -/
def Implements (io : StreamMasked3DKernelIO₄)
    (f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      (Fin io.T → Fin io.B4 → ℝ) →
      Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp1, io.inp2, io.inp3, io.inp4, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀ (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
    (zs : Fin io.T → Fin io.B3 → ℝ) (ws : Fin io.T → Fin io.B4 → ℝ)
    (s₀ : BlockState),
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
    (∀ (t : Fin io.T) (j : Fin io.B4), io.mask4 pid₀ pid₁ pid₂ t j →
      io.read4 pid₀ pid₁ pid₂ t j < A.extent io.inp4) →
    (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ j →
      io.write pid₀ pid₁ pid₂ j < A.extent io.out) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ pid₂ t j) = xs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ pid₂ t j) = ys t j) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp3 (io.read3 pid₀ pid₁ pid₂ t j) = zs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B4), io.mask4 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp4 (io.read4 pid₀ pid₁ pid₂ t j) = ws t j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ j →
          s'.readMem A.flat
              (A.addr io.out (io.write pid₀ pid₁ pid₂ j))
            = f pid₀ pid₁ pid₂ xs ys zs ws j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ pid₂ j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')


@[inherit_doc] scoped notation:25 io " ⊨ " f => Implements io f

/-- Reuse the streamed IO transport proof at identity rounding and real storage.
The resulting contract contains exact execution and real output readback. -/
theorem Implements.of_identityRounding (io : StreamMasked3DKernelIO₄)
    (hreal : io.outDType = .real)
    (f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      (Fin io.T → Fin io.B4 → ℝ) →
      Fin io.C → ℝ)
    (h : ImplementsR io .triv f) : Implements io f := by
  cases io
  cases hreal
  simp only [ImplementsR, execR_triv, BlockState.readMemAs_real] at h
  simpa [Implements, RoundingModel.triv, FloatDType.ofReal, FloatDType.ofWithBot,
    WithBot.some_eq_coe] using h

end VeriTile.Triton.StreamMasked3DKernelIO₄

namespace VeriTile.Triton.StreamMasked3DKernelIO₅

/-- Exact execution, active output readback and the memory frame for five input streams. -/
def Implements (io : StreamMasked3DKernelIO₅)
    (f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      (Fin io.T → Fin io.B4 → ℝ) → (Fin io.T → Fin io.B5 → ℝ) →
      Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp1, io.inp2, io.inp3, io.inp4, io.inp5, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀ (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
    (zs : Fin io.T → Fin io.B3 → ℝ) (ws : Fin io.T → Fin io.B4 → ℝ)
    (vs : Fin io.T → Fin io.B5 → ℝ) (s₀ : BlockState),
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
    (∀ (t : Fin io.T) (j : Fin io.B4), io.mask4 pid₀ pid₁ pid₂ t j →
      io.read4 pid₀ pid₁ pid₂ t j < A.extent io.inp4) →
    (∀ (t : Fin io.T) (j : Fin io.B5), io.mask5 pid₀ pid₁ pid₂ t j →
      io.read5 pid₀ pid₁ pid₂ t j < A.extent io.inp5) →
    (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ j →
      io.write pid₀ pid₁ pid₂ j < A.extent io.out) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ pid₂ t j) = xs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ pid₂ t j) = ys t j) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp3 (io.read3 pid₀ pid₁ pid₂ t j) = zs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B4), io.mask4 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp4 (io.read4 pid₀ pid₁ pid₂ t j) = ws t j) →
    (∀ (t : Fin io.T) (j : Fin io.B5), io.mask5 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp5 (io.read5 pid₀ pid₁ pid₂ t j) = vs t j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ j →
          s'.readMem A.flat
              (A.addr io.out (io.write pid₀ pid₁ pid₂ j))
            = f pid₀ pid₁ pid₂ xs ys zs ws vs j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ pid₂ j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')


@[inherit_doc] scoped notation:25 io " ⊨ " f => Implements io f

/-- Reuse the streamed IO transport proof at identity rounding and real storage.
The resulting contract contains exact execution and real output readback. -/
theorem Implements.of_identityRounding (io : StreamMasked3DKernelIO₅)
    (hreal : io.outDType = .real)
    (f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      (Fin io.T → Fin io.B4 → ℝ) → (Fin io.T → Fin io.B5 → ℝ) →
      Fin io.C → ℝ)
    (h : ImplementsR io .triv f) : Implements io f := by
  cases io
  cases hreal
  simp only [ImplementsR, execR_triv, BlockState.readMemAs_real] at h
  simpa [Implements, RoundingModel.triv, FloatDType.ofReal, FloatDType.ofWithBot,
    WithBot.some_eq_coe] using h

end VeriTile.Triton.StreamMasked3DKernelIO₅

namespace VeriTile.Triton.StreamMasked3DKernelIO₃ₓ₂

/-- Exact three-stream, two-output correctness. The two windows may describe
all values emitted during a loop; neither is restricted to a single terminal
store. Output masks retain the frame of a disabled optional output. -/
def Implements (io : StreamMasked3DKernelIO₃ₓ₂)
    (f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      (Fin io.C1 → ℝ) × (Fin io.C2 → ℝ)) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp1, io.inp2, io.inp3, io.out1, io.out2] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀ (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
    (zs : Fin io.T → Fin io.B3 → ℝ) (s₀ : BlockState),
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
    (∀ j : Fin io.C1, io.writeMask1 pid₀ pid₁ pid₂ j →
      io.write1 pid₀ pid₁ pid₂ j < A.extent io.out1) →
    (∀ j : Fin io.C2, io.writeMask2 pid₀ pid₁ pid₂ j →
      io.write2 pid₀ pid₁ pid₂ j < A.extent io.out2) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ pid₂ t j) = xs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ pid₂ t j) = ys t j) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp3 (io.read3 pid₀ pid₁ pid₂ t j) = zs t j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.C1, io.writeMask1 pid₀ pid₁ pid₂ j →
          s'.readMem A.flat
              (A.addr io.out1 (io.write1 pid₀ pid₁ pid₂ j))
            = (f pid₀ pid₁ pid₂ xs ys zs).1 j)
      ∧ (∀ j : Fin io.C2, io.writeMask2 pid₀ pid₁ pid₂ j →
          s'.readMem A.flat
              (A.addr io.out2 (io.write2 pid₀ pid₁ pid₂ j))
            = (f pid₀ pid₁ pid₂ xs ys zs).2 j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.C1, io.writeMask1 pid₀ pid₁ pid₂ j →
                o' ≠ A.addr io.out1 (io.write1 pid₀ pid₁ pid₂ j)) ∧
             (∀ j : Fin io.C2, io.writeMask2 pid₀ pid₁ pid₂ j →
                o' ≠ A.addr io.out2 (io.write2 pid₀ pid₁ pid₂ j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')


@[inherit_doc] scoped notation:25 io " ⊨ " f => Implements io f

/-- Assemble real execution, both output formulas and their memory frame
using exact address safety and flat-memory transport. -/
theorem Implements.intro (io : StreamMasked3DKernelIO₃ₓ₂)
    {f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      (Fin io.C1 → ℝ) × (Fin io.C2 → ℝ)}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState)
        (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
        (zs : Fin io.T → Fin io.B3 → ℝ),
      io.pre (s.pids 0) (s.pids 1) (s.pids 2) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp1
            (io.read1 (s.pids 0) (s.pids 1) (s.pids 2) t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp2
            (io.read2 (s.pids 0) (s.pids 1) (s.pids 2) t j) = ys t j) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp3
            (io.read3 (s.pids 0) (s.pids 1) (s.pids 2) t j) = zs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read1 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp1) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read2 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp2) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read3 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp3) →
      (∀ j : Fin io.C1,
        io.writeMask1 (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.write1 (s.pids 0) (s.pids 1) (s.pids 2) j < bounds io.out1) →
      (∀ j : Fin io.C2,
        io.writeMask2 (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.write2 (s.pids 0) (s.pids 1) (s.pids 2) j < bounds io.out2) →
      (io.kernel.toAlgKernel).TraceSafe bounds s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.T → Fin io.B1 → ℝ)
        (ys : Fin io.T → Fin io.B2 → ℝ) (zs : Fin io.T → Fin io.B3 → ℝ),
      io.pre (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) →
      s₀.undef = (fun _ _ => 0) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp1
            (io.read1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp2
            (io.read2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = ys t j) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp3
            (io.read3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = zs t j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.C1,
            io.writeMask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
            s1.readMem io.out1
                (io.write1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)
              = (f (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) xs ys zs).1 j)
        ∧ (∀ j : Fin io.C2,
            io.writeMask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
            s1.readMem io.out2
                (io.write2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)
              = (f (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) xs ys zs).2 j)
        ∧ (∀ r oo,
            ((∀ j : Fin io.C1,
                io.writeMask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
                r = io.out1 →
                oo ≠ io.write1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j) ∧
             (∀ j : Fin io.C2,
                io.writeMask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
                r = io.out2 →
                oo ≠ io.write2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)) →
            s1.mem r oo = s₀.mem r oo)) :
    io.Implements f := by
  intro A hd hregs hcov pid₀ pid₁ pid₂ xs ys zs s₀ hpre hpid₀ hpid₁ hpid₂ hu
    hbr1 hbr2 hbr3 hbw1 hbw2 hx hy hz
  subst hpid₀
  subst hpid₁
  subst hpid₂
  obtain ⟨s1, hexec, hval1, hval2, hframe⟩ := hrun s₀ xs ys zs hpre hu hx hy hz
  have hts' : (io.kernel.toAlgKernel).TraceSafe A.extent s₀ :=
    hts A.extent s₀ xs ys zs hpre hx hy hz hbr1 hbr2 hbr3 hbw1 hbw2
  have hbridge := A.exec_flatten hd hcov _ s₀ hts' hok hu
  have hmem1 : io.out1 ∈ A.regions := by rw [hregs]; simp
  have hmem2 : io.out2 ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, ?_, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j hj
    have hlt : io.write1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j
        < A.extent io.out1 := hbw1 j hj
    rw [A.flattenState_readMem hd s1 hmem1 hlt]
    exact hval1 j hj
  · intro j hj
    have hlt : io.write2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j
        < A.extent io.out2 := hbw2 j hj
    rw [A.flattenState_readMem hd s1 hmem2 hlt]
    exact hval2 j hj
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe r o ⟨?_, ?_⟩)
          · intro j hj hro hoj
            rcases hcond with hflat | ⟨hn1, _⟩
            · exact absurd rfl hflat
            · exact hn1 j hj (by rw [hoeq, hro, hoj])
          · intro j hj hro hoj
            rcases hcond with hflat | ⟨_, hn2⟩
            · exact absurd rfl hflat
            · exact hn2 j hj (by rw [hoeq, hro, hoj])
    · simp only [FlatAlloc.flattenState, if_neg hr]


end VeriTile.Triton.StreamMasked3DKernelIO₃ₓ₂
