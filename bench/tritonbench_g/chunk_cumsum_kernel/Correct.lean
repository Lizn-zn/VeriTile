import bench.tritonbench_g.chunk_cumsum_kernel.Kernels
import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.RealIO
import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.RealPadding
import VeriTile.Meta.StatementAudit

/-! Both complete sources in Kernels implement a global prefix sum after dtype
erasure. Positive block width, disjoint allocations, active-address bounds and
zero unspecified load values are explicit flat IO adapter conditions.
`RealPadding.source_outputs` additionally proves valid region outputs for
arbitrary unspecified values. All chunks,
including the final partial one, are covered; no benchmark shape is fixed.
The contract guarantees successful flat execution, every active output and
preservation of all tagged cells outside the output row. Host launch composition
and concrete IEEE numerical accuracy are separate claims. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumKernel.Correctness
open VeriTile.Triton
open scoped BigOperators StreamEmitMasked2DKernelIO₁
noncomputable section

/-- Each program scans one row, loading and emitting BT-lane windows. Both
read and write masks retain the source boundary check. The real IO contract
quantifies over disjoint allocations with sufficient space for active accesses. -/
def originalIO (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) :
    StreamEmitMasked2DKernelIO₁ where
  kernel := (TypedSource.original dtype out_dtype s o T BT).eraseDType
  projection := (RealIO.io Bool.false dtype out_dtype s o T BT).projection
  inp1 := s
  out := o
  T := ccNumChunks T BT
  B1 := BT
  C := BT
  read1 := fun pid _ t j => pid * T + (t.val * BT + j.val)
  write := fun pid _ t j => pid * T + (t.val * BT + j.val)
  mask1 := fun _ _ t j => t.val * BT + j.val < T
  writeMask := fun _ _ t j => t.val * BT + j.val < T

/-- The candidate changes only the order of each carry addition. -/
def sumLeftIO (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) :
    StreamEmitMasked2DKernelIO₁ :=
  { originalIO dtype out_dtype s o T BT with
    kernel := (TypedSource.sumLeft dtype out_dtype s o T BT).eraseDType
    projection := (RealIO.io Bool.true dtype out_dtype s o T BT).projection }

/-- The sum of the independent initial input cells up to the output's flat
row index. Masked tail cells never enter the formula. -/
def expected (T BT : Nat) (_pid₀ _pid₁ : Nat)
    (xs : Fin (ccNumChunks T BT) → Fin BT → ℝ)
    (t : Fin (ccNumChunks T BT)) (j : Fin BT) : ℝ :=
  ∑ u : Fin (ccNumChunks T BT), ∑ e : Fin BT,
    if u.val * BT + e.val ≤ t.val * BT + j.val ∧ u.val * BT + e.val < T
    then xs u e else 0

specification chunk_cumsum_correctness (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) (hSO : o ≠ s) (hBT : 0 < BT) :
    Spec.Real (originalIO dtype out_dtype s o T BT ⊨ expected T BT) := by
  exact RealIO.original_implements dtype out_dtype s o T BT hSO hBT

specification chunk_sum_left_correctness (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) (hSO : o ≠ s) (hBT : 0 < BT) :
    Spec.Real (sumLeftIO dtype out_dtype s o T BT ⊨ expected T BT) := by
  exact RealIO.candidate_implements dtype out_dtype s o T BT hSO hBT

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkCumsumKernel.Correctness
