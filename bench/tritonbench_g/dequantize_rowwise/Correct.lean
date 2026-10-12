import bench.tritonbench_g.dequantize_rowwise.Kernels
import bench.tritonbench_g.dequantize_rowwise.Proofs.RealIO
/-! Complete real correctness for rowwise dequantization: the initial row
scale times the signed integer weight converted to real, times inv. The source
has int8 weights, fp32 scales and fp16 output; precision is erased here only.
The IO contract proves successful finite flat execution and the full tagged
memory frame, under actual-access bounds and a zero unspecified-value oracle.
Positive block/lane counts ensure the unmasked scalar scale read is included
in the lane-wise bounds. The host launch is outside these per-program specs. -/
namespace VeriTile.Bench.TritonBenchG.DequantizeRowwise.Correctness
open VeriTile VeriTile.Triton TypedSource
open scoped VeriTile.Triton.Masked2DKernelIO₂

/-- The row uses its lane mask; all active lanes refer to the same unmasked
scale cell. The positive-size conditions below guarantee such a lane exists. -/
def originalIO (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) : Masked2DKernelIO₂ where
  kernel := (original x scale out inv n block lanes).eraseDType
  projection := by
    have h : (original x scale out inv n block lanes).toAlgorithm? =
        .ok (original x scale out inv n block lanes).toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType,h,ComputeKernel.toAlgorithm?_fromKernelBody,
      ComputeKernel.toAlgKernel_fromKernelBody]
  in1 := x
  in2 := scale
  out := out
  B := lanes
  read1 := fun pid _ j => pid*block+j.val
  read2 := fun pid _ _ => pid
  write := fun pid _ j => pid*block+j.val
  mask := fun _ _ j => j.val<block

def indexCommutedIO (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) : Masked2DKernelIO₂ :=
  { originalIO x scale out inv n block lanes with
    kernel := (indexCommuted x scale out inv n block lanes).eraseDType
    projection := by
      have h : (indexCommuted x scale out inv n block lanes).toAlgorithm? =
          .ok (indexCommuted x scale out inv n block lanes).toAlgKernel := rfl
      simp only [ComputeKernel.eraseDType,h,ComputeKernel.toAlgorithm?_fromKernelBody,
        ComputeKernel.toAlgKernel_fromKernelBody] }


specification original_correct (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat)
    (positiveBlock : 0<block) (positiveLanes : 0<lanes) :
    Spec.Real (originalIO x scale out inv n block lanes ⊨ᵢ fun _ _ xs ys j => ys j*(xs j : ℝ)*inv) := by
  exact RealIO.original_correct x scale out inv n block lanes positiveBlock positiveLanes

specification index_commuted_correct (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat)
    (positiveBlock : 0<block) (positiveLanes : 0<lanes) :
    Spec.Real (indexCommutedIO x scale out inv n block lanes ⊨ᵢ fun _ _ xs ys j => ys j*(xs j : ℝ)*inv) := by
  exact RealIO.index_commuted_correct x scale out inv n block lanes positiveBlock positiveLanes

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DequantizeRowwise.Correctness
