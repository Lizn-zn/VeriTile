import bench.tritonbench_g.mean_reduction.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite

namespace VeriTile.Bench.TritonBenchG.MeanReduction.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (input output : Storage) (commuted : Bool) (X Mean : RegionName)
    (M N BM BN : Nat) : Program where
  kernel := typedKernel input output commuted X Mean M N BM BN
  domain := []

/-- The complete accumulation loop and its precision conversions are unchanged.
Only exact natural-number address arithmetic is commuted; every execution result,
including failure, all registers and all memory, is preserved. -/
specification mean_index_equiv (input output : Storage) (X Mean : RegionName)
    (M N BM BN : Nat) (_hM : 0 < BM) (_hN : 0 < BN) :
    program input output Bool.false X Mean M N BM BN ≡[assumptions]
      program input output Bool.true X Mean M N BM BN := by
  cases input <;> cases output <;> equiv_decompose <;> fp_prove

#print_fp_assumptions mean_index_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MeanReduction.FPEquiv
