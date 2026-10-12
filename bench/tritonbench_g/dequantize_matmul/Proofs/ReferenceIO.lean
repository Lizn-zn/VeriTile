import bench.tritonbench_g.dequantize_matmul.Proofs.RealReference
import bench.tritonbench_g.dequantize_matmul.Proofs.IntegerIO
namespace VeriTile.Bench.TritonBenchG.DequantizeMatmul.ReferenceIO
open VeriTile.Triton RealReference
open scoped VeriTile.Triton.Masked2DKernelIO₂
def dequantizeIO (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    Masked2DKernelIO₂ where
  kernel := dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
    stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K
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

/-- Signed integer weight values are lifted to the real product. Wider
scale-load bounds and output-address injectivity are retained. -/
theorem reference_correct
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (hBK : 0 < BLOCK_SIZE_K)
    (hOutInj : ∀ p₀ p₁ : Nat, Function.Injective
      (fun j : Fin (BLOCK_SIZE_K * BLOCK_SIZE_N) =>
        fpbOffset p₀ p₁ stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K
          (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j))) :
    dequantizeIO b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
        stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K
      ⊨ᵢ fun _ _ xs ys j => (xs j : ℝ) * ys j := by
  refine Masked2DKernelIO₂.ImplementsInt.intro _ ?_ ?_ ?_
  · exact dequantize_kernel_flattenOk b_ptr b_scale_ptr fpb_ptr K N stride_bk
      stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K
  · intro bounds s hb hs hout _
    exact dequantize_kernel_traceSafe b_ptr b_scale_ptr fpb_ptr K N stride_bk
      stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K hBK bounds s
      hb hs hout
  · intro s₀ xs ys hx hy
    obtain ⟨s1, hs1, hval, hframe⟩ :=
      dequantize_kernel_region_run b_ptr b_scale_ptr fpb_ptr K N stride_bk
        stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K s₀ xs ys
        (inj_of_lane_inj (hOutInj (s₀.pids 0) (s₀.pids 1))) hx hy
    exact ⟨s1, hs1, hval, fun r o hcond _ => hframe r o hcond⟩


#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DequantizeMatmul.ReferenceIO
