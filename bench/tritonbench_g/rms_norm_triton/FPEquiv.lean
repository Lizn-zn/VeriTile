import bench.tritonbench_g.rms_norm_triton.Proofs.FP
import bench.tritonbench_g.rms_norm_triton.Proofs.SourceChecks
import bench.tritonbench_g.rms_norm_triton.Proofs.Execution

/-! Full typed state/failure equality, including all registers and memory.
All arithmetic retains source precision under the default fp32 profile. The
candidate only commutes two natural row-base products; no floating algebraic
law or change to the pre-weight cast is assumed. -/
namespace VeriTile.Bench.TritonBenchG.RmsNormTriton.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 2000000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (storage weight : Storage) (v : Bool) (Y X W : RegionName)
    (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat) : Program where
  kernel := typed storage weight v Y X W ysr ysc xsr xsc N eps B
  domain := []

specification source_equivalence (storage weight : Storage) (Y X W : RegionName)
    (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat) :
    program storage weight Bool.false Y X W ysr ysc xsr xsc N eps B ≡[assumptions]
      program storage weight Bool.true Y X W ysr ysc xsr xsc N eps B := by
  cases storage <;> cases weight <;> equiv_decompose <;> fp_prove

#print_fp_assumptions source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsNormTriton.FPEquiv
