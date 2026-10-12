import bench.tritonbench_g.dequantize_matmul.Kernels
import bench.tritonbench_g.decay_cumsum.Proofs.MaskEquivalence
namespace VeriTile.Bench.TritonBenchG.DequantizeMatmul.FPProof
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (output : Storage) (swapped : Bool) (b scale out : RegionName)
    (K N sbk sbn sok son BN BK : Nat) : Program where
  kernel := TypedSource.kernel output swapped b scale out K N sbk sbn sok son BN BK
  domain := []

theorem equivalent (output : Storage) (b scale out : RegionName)
    (K N sbk sbn sok son BN BK : Nat) :
    program output Bool.false b scale out K N sbk sbn sok son BN BK ≡[assumptions]
      program output Bool.true b scale out K N sbk sbn sok son BN BK := by
  cases output <;> equiv_decompose <;> fp_prove

#print_fp_assumptions equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DequantizeMatmul.FPProof
