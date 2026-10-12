import bench.tritonbench_g.cache_transform_triton.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite

/-! The complete source differs only in exact token-index multiplication.
The floating carrier remains opaque. No numerical rewrite atom or numerical
domain is required for copying the same values at the same addresses. -/
namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton.FPEquivProofs
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

section
variable (cosType sinType : TypedSource.Storage) (cos sin : RegionName) (metadata : Region .int)
    (cosOut sinOut : RegionName) (rowStride hiddenStride total H N B : Nat)

def prefillOriginal : Program where
  kernel := TypedSource.prefill cosType sinType Bool.false cos sin metadata cosOut sinOut
    rowStride hiddenStride total H N B
  domain := []

def prefillCommuted : Program where
  kernel := TypedSource.prefill cosType sinType Bool.true cos sin metadata cosOut sinOut
    rowStride hiddenStride total H N B
  domain := []

theorem prefill_equiv :
    prefillOriginal cosType sinType cos sin metadata cosOut sinOut rowStride hiddenStride total H N B ≡[assumptions]
    prefillCommuted cosType sinType cos sin metadata cosOut sinOut rowStride hiddenStride total H N B := by
  cases cosType <;> cases sinType <;> equiv_decompose <;> fp_prove

def decodeOriginal : Program where
  kernel := TypedSource.decode cosType sinType Bool.false cos sin metadata cosOut sinOut
    rowStride hiddenStride H N B
  domain := []

def decodeCommuted : Program where
  kernel := TypedSource.decode cosType sinType Bool.true cos sin metadata cosOut sinOut
    rowStride hiddenStride H N B
  domain := []

theorem decode_equiv :
    decodeOriginal cosType sinType cos sin metadata cosOut sinOut rowStride hiddenStride H N B ≡[assumptions]
    decodeCommuted cosType sinType cos sin metadata cosOut sinOut rowStride hiddenStride H N B := by
  cases cosType <;> cases sinType <;> equiv_decompose <;> fp_prove
end

#print_fp_assumptions prefill_equiv
#print_fp_assumptions decode_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.CacheTransformTriton.FPEquivProofs
