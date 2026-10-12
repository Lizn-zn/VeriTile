import bench.tritonbench_g.cosine_compute.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite
import VeriTile.Meta.StatementAudit

/-! Exact integer-index commutation inside the complete typed program. The FP
cosine, casts and unspecified masked loads are opaque and remain identical.
There is no numerical atom or numerical-domain premise. -/
namespace VeriTile.Bench.TritonBenchG.CosineCompute.FPProof
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def original (storage : Storage) (a b : RegionName) (n B : Nat) : Program where
  kernel := CosineCompute.original storage a b n B
  domain := []

def indexCommuted (storage : Storage) (a b : RegionName) (n B : Nat) : Program :=
  { original storage a b n B with kernel := CosineCompute.indexCommuted storage a b n B }

theorem equivalent (storage : Storage) (a b : RegionName) (n B : Nat) :
    original storage a b n B ≡[assumptions] indexCommuted storage a b n B := by
  cases storage <;> equiv_decompose <;> fp_prove

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.CosineCompute.FPProof
