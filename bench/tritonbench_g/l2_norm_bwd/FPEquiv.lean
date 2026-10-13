import bench.tritonbench_g.l2_norm_bwd.Proofs.FP
import bench.tritonbench_g.l2_norm_bwd.Proofs.SourceChecks
import bench.tritonbench_g.l2_norm_bwd.Proofs.Execution

/-! Complete typed-state/failure equivalence under the default fp32 profile.
X/DX storage and DY storage independently range over fp16/bf16/fp32/fp64.
Both loads narrow to fp32, both runtime-epsilon uses convert to fp32, and the
store converts to X/DX storage. The candidate only commutes natural-number
address products; reductions, sqrt, arithmetic and masks are unchanged. -/
namespace VeriTile.Bench.TritonBenchG.L2NormBwd.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 2000000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (storage gradient : Storage) (v : Bool) (X DY DX : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) : Program where
  kernel := typedBwd storage gradient v X DY DX stride N eps B
  domain := []

specification source_equivalence (storage gradient : Storage) (X DY DX : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) :
    program storage gradient Bool.false X DY DX stride N eps B ≡[assumptions]
      program storage gradient Bool.true X DY DX stride N eps B := by
  cases storage <;> cases gradient <;> equiv_decompose <;> fp_prove

#print_fp_assumptions source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.L2NormBwd.FPEquiv
