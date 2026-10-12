import bench.tritonbench_g.context_attn_bloom.Proofs.PreludeComplete
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.PreludeComplete
open VeriTile.Triton Math Program RealSetup
noncomputable section
set_option maxHeartbeats 700000
set_option maxRecDepth 2000
set_option linter.unusedSimpArgs false

theorem scalar_agrees (p : Layout) (r : Regions) (u : BlockState) :
    offsetState p r u = setupScalar p r u := by
  simp only [offsetState,metadataState,PreludeOffsets.prepared,PreludeMetadata.prepared,launch_inputs]
  simp only [launchState,PreludeLaunch.launched,setupScalar,launch,Math.sequence,Math.queryRow]

theorem offset_reads (p : Layout) (r : Regions) (u : BlockState) (region : RegionName) (off : Nat) :
    (offsetState p r u).readMem region off = u.readMem region off := by
  simp only [offsetState,metadataState,launchState,PreludeOffsets.prepared,PreludeMetadata.prepared,
    PreludeLaunch.launched,BlockState.setReg_readMem]

theorem query_agrees (p : Layout) (r : Regions) (u : BlockState) :
    queryState p r u = initialized p r u := by
  simp only [queryState,PreludeQuery.queried,offset_reads]
  simp only [scalar_agrees,initialized,
    Math.query,Math.sequence,Math.queryRow,Math.limit,launch,inputs,
    Nat.cast_add,Nat.cast_mul,Nat.cast_one,Int.sub_add_cancel]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.PreludeComplete
