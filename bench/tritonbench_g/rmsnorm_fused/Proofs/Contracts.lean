import bench.tritonbench_g.rmsnorm_fused.Proofs.Real
import bench.tritonbench_g.rmsnorm_fused.Proofs.Candidate

namespace VeriTile.Bench.TritonBenchG.RmsnormFused
open VeriTile.Triton

/-- Successful exact-real execution, all active streamed outputs, and every other memory cell. -/
def ExactIO (io : StreamEmitMasked2DKernelIO₂)
    (f : Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → Fin io.T → Fin io.C → ℝ) : Prop :=
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
    (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ t j →
      io.write pid₀ pid₁ t j < A.extent io.out) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ t j) = xs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ t j) = ys t j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ t j →
          s'.readMem A.flat
              (A.addr io.out (io.write pid₀ pid₁ t j))
            = f pid₀ pid₁ xs ys t j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ t j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ t j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

def typedIO (input weight output : Storage) (v : Bool)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) :
    StreamEmitMasked2DKernelIO₂ :=
  { rmsnormFusedKernelIO X Y W stride N B eps with
    kernel := (typedKernel input weight output v X Y W stride N B eps).eraseDType
    projection := typed_supported input weight output v X Y W stride N B eps }

/-- The independent formula sums every active input in both-dimensional stream
coordinates; it does not depend on the output program or an output assumption. -/
theorem typed_exact_io (input weight output : Storage) (v : Bool)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ)
    (hNpos : 0 < N) (hNle : N ≤ B) :
    ExactIO (typedIO input weight output v X Y W stride N B eps)
      (fun _ _ xs ws t j => rmsFusedStreamSpec N B eps xs ws t j) := by
  intro A hd hr hc p0 p1 xs ws s hp0 hp1 hu hbx hbw hbo hx hw
  obtain ⟨t, he, hv, hf⟩ := rms_norm_fwd_fused_io_correctness_bridge .triv
    X Y W stride N B eps hNpos hNle
    A hd hr hc p0 p1 xs ws s hp0 hp1 hu hbx hbw hbo hx hw
  refine ⟨t, ?_, ?_, hf⟩
  · rw [execR_triv] at he
    exact (typed_flat_exec A input weight output v X Y W stride N B eps
      (A.flattenState s)).trans he
  · simpa [typedIO, rmsnormFusedKernelIO, BlockState.readMemAs_real,
      RoundingModel.triv, FloatDType.ofReal, FloatDType.ofWithBot,
      WithBot.some_eq_coe] using hv

end VeriTile.Bench.TritonBenchG.RmsnormFused
