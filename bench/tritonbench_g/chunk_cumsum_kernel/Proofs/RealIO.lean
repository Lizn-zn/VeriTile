import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.RealCandidate
import VeriTile.Triton.Memory.KernelSpec.StreamEmitReal

/-! Connect the independent global prefix-sum formula, successful flat
execution and complete memory frame to both typed sources after erasure. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumKernel.RealIO
open VeriTile.Triton
open scoped StreamEmitMasked2DKernelIO₁
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000

/-- Internal view; the public entry repeats these windows and masks. -/
def io (swapped : Bool) (dtype out_dtype : FloatDType) (s o : RegionName)
    (T BT : Nat) : StreamEmitMasked2DKernelIO₁ where
  kernel := (RealSource.source swapped dtype out_dtype s o T BT).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType, RealSource.projects]
    simp only [ComputeKernel.fromKernelBody, ComputeKernel.toAlgorithm?_mk_map_alg,
      ComputeKernel.toAlgKernel]
  inp1 := s
  out := o
  T := ccNumChunks T BT
  B1 := BT
  C := BT
  read1 := fun pid _ t j => pid * T + (t.val * BT + j.val)
  write := fun pid _ t j => pid * T + (t.val * BT + j.val)
  mask1 := fun _ _ t j => t.val * BT + j.val < T
  writeMask := fun _ _ t j => t.val * BT + j.val < T

theorem original_implements (dtype out_dtype : FloatDType) (s o : RegionName)
    (T BT : Nat) (hSO : o ≠ s) (hBT : 0 < BT) :
    io Bool.false dtype out_dtype s o T BT ⊨ fun _ _ xs t j => ccStreamSpec T BT xs t j := by
  have old := StreamEmitMasked2DKernelIO₁.Implements.of_identityRounding
    (chunkCumsumKernelIO s o T BT) rfl _
    (chunk_cumsum_scalar_io_correctness .triv s o T BT hSO hBT)
  intro A hd hr hc p₀ p₁ xs u hp₀ hp₁ hu hb hw hx
  obtain ⟨v, he, hv, hf⟩ := old A hd hr hc p₀ p₁ xs u hp₀ hp₁ hu hb hw hx
  refine ⟨v, ?_, hv, hf⟩
  simpa only [io, exec, FlatAlloc.flattenKernel, RealSource.original_body,
    chunkCumsumKernelIO] using he

theorem candidate_implements (dtype out_dtype : FloatDType) (s o : RegionName)
    (T BT : Nat) (hSO : o ≠ s) (hBT : 0 < BT) :
    io Bool.true dtype out_dtype s o T BT ⊨ fun _ _ xs t j => ccStreamSpec T BT xs t j := by
  intro A hd hr hc p₀ p₁ xs u hp₀ hp₁ hu hb hw hx
  obtain ⟨v, he, hv, hf⟩ := original_implements dtype out_dtype s o T BT hSO hBT
    A hd hr hc p₀ p₁ xs u hp₀ hp₁ hu hb hw hx
  refine ⟨v, ?_, hv, hf⟩
  exact (RealCandidate.flat_exec_equal A dtype out_dtype s o T BT (A.flattenState u)).symm.trans he

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkCumsumKernel.RealIO
