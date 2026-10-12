import bench.tritonbench_g.dequantize_rowwise.Proofs.Projection
namespace VeriTile.Bench.TritonBenchG.DequantizeRowwise.RealIO
open VeriTile.Triton TypedSource
open scoped VeriTile.Triton.Masked2DKernelIO₂

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

theorem original_correct (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat)
    (hb : 0<block) (hl : 0<lanes) :
    originalIO x scale out inv n block lanes ⊨ᵢ fun _ _ xs ys j => ys j*(xs j : ℝ)*inv := by
  have h := RealReference.reference_correct x scale out inv n block lanes hb hl
  simpa only [Masked2DKernelIO₂.ImplementsInt,originalIO,RealReference.dequantizeRowwiseIO,
    Projection.original_exec] using h

theorem index_commuted_correct (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat)
    (hb : 0<block) (hl : 0<lanes) :
    indexCommutedIO x scale out inv n block lanes ⊨ᵢ fun _ _ xs ys j => ys j*(xs j : ℝ)*inv := by
  have h := original_correct x scale out inv n block lanes hb hl
  simpa only [Masked2DKernelIO₂.ImplementsInt,indexCommutedIO,originalIO,
    Projection.candidate_exec] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DequantizeRowwise.RealIO
