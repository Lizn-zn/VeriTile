import bench.tritonbench_g.index_select_bwd.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite
import bench.tritonbench_g.index_select_bwd.Proofs.Execution

namespace VeriTile.Bench.TritonBenchG.IndexSelectBwd.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (input output : Storage) (index : IndexStorage) (v : Bool)
    (out idx inp : RegionName) (rows n c s0 s1 BI BC : Nat) : Program where
  kernel := source input output index v out idx inp rows n c s0 s1 BI BC
  domain := []

/-- Complete typed scatter executions agree after commuting only the natural
column-block address product. Signed index loads, masks, explicit conversions,
ordered duplicate writes and buffer aliasing are all preserved. -/
specification index_equiv (input output : Storage) (index : IndexStorage)
    (out idx inp : RegionName) (rows n c s0 s1 BI BC : Nat) :
    program input output index Bool.false out idx inp rows n c s0 s1 BI BC ≡[assumptions]
      program input output index Bool.true out idx inp rows n c s0 s1 BI BC := by
  cases input <;> cases output <;> cases index <;> equiv_decompose <;> fp_prove

#print_fp_assumptions index_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.IndexSelectBwd.FPEquiv
