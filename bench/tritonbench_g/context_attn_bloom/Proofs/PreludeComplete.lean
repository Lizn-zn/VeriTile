import bench.tritonbench_g.context_attn_bloom.Proofs.PreludeQuery
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.PreludeComplete
open VeriTile.Triton Math Program RealSetup PreludeProgram
noncomputable section
set_option maxHeartbeats 700000
set_option maxRecDepth 2000
set_option linter.unusedSimpArgs false

def launchState (p : Layout) (u : BlockState) := PreludeLaunch.launched p u
def metadataState (p : Layout) (r : Regions) (u : BlockState) :=
  PreludeMetadata.prepared r (u.pids 0) (launchState p u)
def offsetState (p : Layout) (r : Regions) (u : BlockState) :=
  PreludeOffsets.prepared p (u.pids 1) (u.pids 2) ((inputs u r).start (u.pids 0))
    (metadataState p r u)
def queryState (p : Layout) (r : Regions) (u : BlockState) :=
  PreludeQuery.queried p r (u.pids 1) (u.pids 2) ((inputs u r).start (u.pids 0))
    (Math.sequence (inputs u r) (launch u)) ((inputs u r).prompt (u.pids 0))
    (offsetState p r u)

theorem launch_inputs (p : Layout) (r : Regions) (u : BlockState) :
    inputs (launchState p u) r = inputs u r := by rfl

theorem metadata_executes (p : Layout) (r : Regions) (u : BlockState) :
    stepStmts (metadataCode r) (launchState p u) = some (metadataState p r u) := by
  exact PreludeMetadata.executes r (u.pids 0) (launchState p u) (PreludeLaunch.batch p u)

theorem offsets_execute (p : Layout) (r : Regions) (u : BlockState) :
    stepStmts (offsetCode p) (metadataState p r u) = some (offsetState p r u) := by
  apply PreludeOffsets.executes
  · simp only [metadataState,PreludeMetadata.unchanged _ _ _ _ _ "cur_head" (by decide) (by decide) (by decide) (by decide)]
    exact PreludeLaunch.head p u
  · simp only [metadataState,PreludeMetadata.unchanged _ _ _ _ _ "start_m" (by decide) (by decide) (by decide) (by decide)]
    exact PreludeLaunch.block p u
  · simp only [metadataState,PreludeMetadata.start,launch_inputs]

theorem query_executes (p : Layout) (r : Regions) (u : BlockState) :
    stepStmts (queryCode p r) (offsetState p r u) = some (queryState p r u) := by
  apply PreludeQuery.executes
  · exact PreludeOffsets.off_q _ _ _ _ _
  · exact PreludeOffsets.rows _ _ _ _ _
  · exact PreludeOffsets.dims _ _ _ _ _
  · exact PreludeOffsets.block_start _ _ _ _ _
  · simp only [offsetState,PreludeOffsets.unchanged _ _ _ _ _ _ _ "start_m" (by decide) (by decide) (by decide) (by decide) (by decide),
      metadataState,PreludeMetadata.unchanged _ _ _ _ _ "start_m" (by decide) (by decide) (by decide) (by decide)]
    exact PreludeLaunch.block p u
  · simp only [offsetState,PreludeOffsets.unchanged _ _ _ _ _ _ _ "cur_batch_seq_len" (by decide) (by decide) (by decide) (by decide) (by decide),
      metadataState,PreludeMetadata.sequence,launch_inputs,Math.sequence,launch]
  · simp only [offsetState,PreludeOffsets.unchanged _ _ _ _ _ _ _ "prompt_cache_len" (by decide) (by decide) (by decide) (by decide) (by decide),
      metadataState,PreludeMetadata.prompt,launch_inputs]

theorem executes (p : Layout) (r : Regions) (u : BlockState) :
    stepStmts (prelude p r) u = some (queryState p r u) := by
  rw [PreludeProgram.source,List.append_assoc,List.append_assoc,
    stepStmts.append_some (PreludeLaunch.executes p u)]
  change stepStmts (metadataCode r ++ (offsetCode p ++ queryCode p r)) (launchState p u) = _
  rw [stepStmts.append_some (metadata_executes p r u),
    stepStmts.append_some (offsets_execute p r u),query_executes]
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.PreludeComplete
