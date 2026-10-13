import bench.tritonbench_g.rmsnorm_triton.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsnormTriton
open VeriTile.Triton
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

/-- The full mathematical projection retains every source statement and both
loops. This canonical floating profile is only an erasure representative;
no input values are precomputed or substituted. -/
def mathKernel (v : Bool) (X W O : RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : Kernel :=
  (typedKernel .fp32 .fp32 .fp32 v X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgKernel

/-- Every floating storage profile has the same complete real projection. -/
theorem typed_projection (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgKernel =
      mathKernel v X W O sxb sxm sxk sw sob som sok N B eps := by
  cases input <;> cases weight <;> cases output <;> cases v <;> rfl

theorem typed_supported (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgorithm? =
      Except.ok (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgKernel := by
  cases input <;> cases weight <;> cases output <;> cases v <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormTriton
