import bench.tritonbench_g.context_attn_bloom.Proofs.PreludeAlignment
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.RealSetup
open VeriTile.Triton Math Program
noncomputable section
/-- The complete source prologue computes masked queries and signed loop bounds
from the initial arrays. Explicit masked-load alternatives remove any need
for a global undefined-memory value assumption. -/
theorem prepares (p : Layout) (r : Regions) (u : BlockState) :
    stepStmts (prelude p r) u = some (initialized p r u) := by
  rw [PreludeComplete.executes,PreludeComplete.query_agrees]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.RealSetup
