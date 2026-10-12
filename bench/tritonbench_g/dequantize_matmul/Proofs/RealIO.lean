import bench.tritonbench_g.dequantize_matmul.Proofs.Projection
import bench.tritonbench_g.dequantize_matmul.Proofs.Candidate
import bench.tritonbench_g.dequantize_matmul.Proofs.ReferenceIO
namespace VeriTile.Bench.TritonBenchG.DequantizeMatmul.RealIO
open VeriTile VeriTile.Triton TypedSource RealReference
open scoped VeriTile.Triton.Masked2DKernelIO₂
def originalIO (output : Storage) (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    Masked2DKernelIO₂ where
  kernel := (kernel output Bool.false b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
    stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType,Projection.source_projects,
      ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody]
  in1 := b_ptr
  in2 := b_scale_ptr
  out := fpb_ptr
  B := BLOCK_SIZE_K * BLOCK_SIZE_N
  read1 := fun p₀ p₁ j =>
    bOffset p₀ p₁ stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K
      (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j)
  read2 := fun _ p₁ j =>
    nOffset p₁ BLOCK_SIZE_N (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j).2.1
  write := fun p₀ p₁ j =>
    fpbOffset p₀ p₁ stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K
      (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j)
  mask := fun p₀ p₁ j =>
    dequantizeActive p₀ p₁ K N BLOCK_SIZE_N BLOCK_SIZE_K
      (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j)
  read2Mask := fun _ p₁ j =>
    nOffset p₁ BLOCK_SIZE_N (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j).2.1 < N


def indexCommutedIO (output : Storage) (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) : Masked2DKernelIO₂ :=
  { originalIO output b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K with
    kernel := (kernel output Bool.true b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K).eraseDType
    projection := by
      simp only [ComputeKernel.eraseDType,Projection.source_projects,
        ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody] }

theorem original_correct (output : Storage)
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (hBK : 0 < BLOCK_SIZE_K)
    (hOutInj : ∀ p₀ p₁ : Nat, Function.Injective
      (fun j : Fin (BLOCK_SIZE_K * BLOCK_SIZE_N) =>
        fpbOffset p₀ p₁ stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K
          (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j))) :
    originalIO output b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
        stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K
      ⊨ᵢ fun _ _ xs ys j => (xs j : ℝ) * ys j := by
  have h := ReferenceIO.reference_correct b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K hBK hOutInj
  simpa only [Masked2DKernelIO₂.ImplementsInt,originalIO,ReferenceIO.dequantizeIO,
    Projection.original_exec] using h

theorem index_commuted_correct (output : Storage)
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (hBK : 0 < BLOCK_SIZE_K)
    (hOutInj : ∀ p₀ p₁ : Nat, Function.Injective
      (fun j : Fin (BLOCK_SIZE_K * BLOCK_SIZE_N) =>
        fpbOffset p₀ p₁ stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K
          (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j))) :
    indexCommutedIO output b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
        stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K
      ⊨ᵢ fun _ _ xs ys j => (xs j : ℝ) * ys j := by
  have h := original_correct output b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K hBK hOutInj
  simpa only [Masked2DKernelIO₂.ImplementsInt,indexCommutedIO,originalIO,
    Candidate.flat_exec_equal] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DequantizeMatmul.RealIO
