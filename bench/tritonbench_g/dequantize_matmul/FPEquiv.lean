import bench.tritonbench_g.dequantize_matmul.Kernels
import bench.tritonbench_g.dequantize_matmul.Proofs.FP
/-! Complete FP program equivalence for signed int8 weights, fp32 column
scales and fp16/bf16/fp32 output. Only the exact integer column-mask product
is commuted; numerical computations, conversions, memory and failures are
preserved. No numerical atom or numerical-domain condition is needed. -/
namespace VeriTile.Bench.TritonBenchG.DequantizeMatmul.FPEquivalence
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def original (output : Storage) (weights scales out : RegionName)
    (K N weightRow weightCol outputRow outputCol BN BK : Nat) : Program where
  kernel := TypedSource.kernel output Bool.false weights scales out K N weightRow weightCol outputRow outputCol BN BK
  domain := []

def indexCommuted (output : Storage) (weights scales out : RegionName)
    (K N weightRow weightCol outputRow outputCol BN BK : Nat) : Program where
  kernel := TypedSource.kernel output Bool.true weights scales out K N weightRow weightCol outputRow outputCol BN BK
  domain := []

specification index_commuted_equiv (output : Storage) (weights scales out : RegionName)
    (K N weightRow weightCol outputRow outputCol BN BK : Nat) :
    original output weights scales out K N weightRow weightCol outputRow outputCol BN BK ≡[assumptions]
      indexCommuted output weights scales out K N weightRow weightCol outputRow outputCol BN BK := by
  exact FPProof.equivalent output weights scales out K N weightRow weightCol outputRow outputCol BN BK

#print_fp_assumptions index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DequantizeMatmul.FPEquivalence
