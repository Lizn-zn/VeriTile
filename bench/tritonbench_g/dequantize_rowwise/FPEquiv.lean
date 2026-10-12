import bench.tritonbench_g.dequantize_rowwise.Kernels
import bench.tritonbench_g.dequantize_rowwise.Proofs.FP
import VeriTile.Meta.StatementAudit
/-! Whole-program FP equivalence. Only the exact integer block product is
commuted. Integer loads/conversion, fp32 computation, fp16 output, both load
masks and failure are retained. No numerical atom or domain is required. -/
namespace VeriTile.Bench.TritonBenchG.DequantizeRowwise.FPEquivalence
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
specification index_commuted_equiv (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) :
    original x scale out inv n block lanes ≡[assumptions] indexCommuted x scale out inv n block lanes := by
  exact FPProof.equivalent x scale out inv n block lanes
#print_fp_assumptions index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DequantizeRowwise.FPEquivalence
