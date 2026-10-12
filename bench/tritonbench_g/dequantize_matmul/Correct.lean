import bench.tritonbench_g.dequantize_matmul.Kernels
import bench.tritonbench_g.dequantize_matmul.Proofs.RealIO
/-! Real correctness of the complete dequantization JIT and its candidate.
The signed-byte payload uses the mathematical Int channel, scales are real
values after erasure, and output storage is erased only for this real theorem.
Contracts require a finite disjoint allocation, bounds for the actual reads
and writes and zero unspecified values. They conclude successful full flat
execution, the independent product and preservation of all other tagged cells.
The host torch.mm call is outside this JIT specification. -/
namespace VeriTile.Bench.TritonBenchG.DequantizeMatmul.Correctness
open VeriTile VeriTile.Triton TypedSource
open scoped VeriTile.Triton.Masked2DKernelIO₂

/-- The scale mask is column-only: the source reads these cells even if no
weight or output lane in a K row is active. -/
def originalIO (output : Storage) (weights scales out : RegionName)
    (K N weightRow weightCol outputRow outputCol BN BK : Nat) : Masked2DKernelIO₂ where
  kernel := (kernel output Bool.false weights scales out K N weightRow weightCol outputRow outputCol BN BK).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType,Projection.source_projects,
      ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody]
  in1 := weights
  in2 := scales
  out := out
  B := BK*BN
  read1 := fun pk pn j => (pk*BK+j.val/BN)*weightRow+(pn*BN+j.val%BN)*weightCol
  read2 := fun _ pn j => pn*BN+j.val%BN
  write := fun pk pn j => (pk*BK+j.val/BN)*outputRow+(pn*BN+j.val%BN)*outputCol
  mask := fun pk pn j => pk*BK+j.val/BN<K ∧ pn*BN+j.val%BN<N
  read2Mask := fun _ pn j => pn*BN+j.val%BN<N

def indexCommutedIO (output : Storage) (weights scales out : RegionName)
    (K N weightRow weightCol outputRow outputCol BN BK : Nat) : Masked2DKernelIO₂ :=
  { originalIO output weights scales out K N weightRow weightCol outputRow outputCol BN BK with
    kernel := (kernel output Bool.true weights scales out K N weightRow weightCol outputRow outputCol BN BK).eraseDType
    projection := by
      simp only [ComputeKernel.eraseDType,Projection.source_projects,
        ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody] }

/-- Distinct tile lanes must target distinct output cells. -/
def outputInjective (outputRow outputCol BN BK : Nat) : Prop :=
  ∀ pk pn : Nat, Function.Injective fun j : Fin (BK*BN) =>
    (pk*BK+j.val/BN)*outputRow+(pn*BN+j.val%BN)*outputCol

specification original_correct (output : Storage) (weights scales out : RegionName)
    (K N weightRow weightCol outputRow outputCol BN BK : Nat)
    (positiveKBlock : 0<BK) (distinctOutputs : outputInjective outputRow outputCol BN BK) :
    Spec.Real (originalIO output weights scales out K N weightRow weightCol outputRow outputCol BN BK
      ⊨ᵢ fun _ _ xs ys j => (xs j : ℝ)*ys j) := by
  exact RealIO.original_correct output weights scales out K N weightRow weightCol outputRow outputCol BN BK
    positiveKBlock distinctOutputs

specification index_commuted_correct (output : Storage) (weights scales out : RegionName)
    (K N weightRow weightCol outputRow outputCol BN BK : Nat)
    (positiveKBlock : 0<BK) (distinctOutputs : outputInjective outputRow outputCol BN BK) :
    Spec.Real (indexCommutedIO output weights scales out K N weightRow weightCol outputRow outputCol BN BK
      ⊨ᵢ fun _ _ xs ys j => (xs j : ℝ)*ys j) := by
  exact RealIO.index_commuted_correct output weights scales out K N weightRow weightCol outputRow outputCol BN BK
    positiveKBlock distinctOutputs

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DequantizeMatmul.Correctness
