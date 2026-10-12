import bench.tritonbench_g.dequantize_rowwise.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.DequantizeRowwise.FPProof
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) : Program where
  kernel := TypedSource.original x scale out inv n block lanes
  domain := []
def indexCommuted (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) : Program where
  kernel := TypedSource.indexCommuted x scale out inv n block lanes
  domain := []
theorem equivalent (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) :
    original x scale out inv n block lanes ≡[assumptions] indexCommuted x scale out inv n block lanes := by
  equiv_decompose
  fp_prove
#print_fp_assumptions equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DequantizeRowwise.FPProof
