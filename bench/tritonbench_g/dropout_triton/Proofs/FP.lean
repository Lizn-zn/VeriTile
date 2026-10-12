import bench.tritonbench_g.dropout_triton.Kernels
import bench.tritonbench_g.destindex_copy.Proofs.IndexRewrite
namespace VeriTile.Bench.TritonBenchG.DropoutTriton.FPProof
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (storage : Storage) (keep : KeepStorage) (swapped : Bool)
    (x mask out : RegionName) (n B : Nat) (p : ℝ) : Program where
  kernel := source storage keep swapped x mask out n B p
  domain := []

theorem equivalent (storage : Storage) (keep : KeepStorage)
    (x mask out : RegionName) (n B : Nat) (p : ℝ) :
    program storage keep Bool.false x mask out n B p ≡[assumptions]
      program storage keep Bool.true x mask out n B p := by
  cases storage <;> cases keep <;> equiv_decompose <;> fp_prove

#print_fp_assumptions equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DropoutTriton.FPProof
