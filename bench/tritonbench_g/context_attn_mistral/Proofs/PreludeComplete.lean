import bench.tritonbench_g.context_attn_mistral.Proofs.PreludeExecution
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.PreludeComplete
open VeriTile.Triton Formula Program PreludeProgram
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

def launchState (p : Layout) (u : BlockState) := PreludeLaunch.launched p u
def metadataState (p : Layout) (r : Regions) (u : BlockState) :=
  PreludeMetadata.prepared r (u.pids 0) (launchState p u)
def offsetState (p : Layout) (r : Regions) (u : BlockState) :=
  PreludeOffsets.prepared p (u.pids 1) (u.pids 2)
    ((inputs u r).start (u.pids 0)) (metadataState p r u)
def initialized (p : Layout) (r : Regions) (u : BlockState) :=
  PreludeQuery.queried p r (u.pids 1) (u.pids 2)
    ((inputs u r).start (u.pids 0))
    (Formula.sequence p (launch u) (inputs u r))
    (offsetState p r u)

theorem launch_inputs (p : Layout) (r : Regions) (u : BlockState) :
    inputs (launchState p u) r = inputs u r := by rfl

theorem metadata_executes (p : Layout) (r : Regions) (u : BlockState) :
    stepStmts (metadataCode r) (launchState p u) = some (metadataState p r u) := by
  exact PreludeMetadata.executes r (u.pids 0) (launchState p u) (PreludeLaunch.batch p u)

theorem offsets_execute (p : Layout) (r : Regions) (u : BlockState) :
    stepStmts (offsetCode p) (metadataState p r u) = some (offsetState p r u) := by
  apply PreludeOffsets.executes
  · simp [metadataState,launchState,PreludeMetadata.prepared,PreludeLaunch.launched,launch]
  · simp [metadataState,launchState,PreludeMetadata.prepared,PreludeLaunch.launched,launch]
  · simp [metadataState,launchState,PreludeMetadata.prepared,PreludeLaunch.launched,launch]
  · simp only [metadataState,PreludeMetadata.start,launch_inputs]

theorem query_executes (p : Layout) (r : Regions) (u : BlockState) :
    stepStmts (queryCode p r) (offsetState p r u) = some (initialized p r u) := by
  apply PreludeQuery.executes
  · exact PreludeOffsets.off_q _ _ _ _ _
  · exact PreludeOffsets.rows _ _ _ _ _
  · exact PreludeOffsets.block_start _ _ _ _ _
  · simp [offsetState,PreludeOffsets.prepared,metadataState,PreludeMetadata.prepared,
      launchState,PreludeLaunch.launched,inputs,Formula.sequence,launch]
  · simp [offsetState,PreludeOffsets.prepared]
  · simp [offsetState,PreludeOffsets.prepared]

theorem executes (p : Layout) (r : Regions) (u : BlockState) :
    stepStmts (prelude p r) u = some (initialized p r u) := by
  rw [PreludeProgram.source,List.append_assoc,List.append_assoc,
    stepStmts.append_some (PreludeLaunch.executes p u)]
  change stepStmts (metadataCode r ++ (offsetCode p ++ queryCode p r)) (launchState p u) = _
  rw [stepStmts.append_some (metadata_executes p r u),
    stepStmts.append_some (offsets_execute p r u),query_executes]

theorem memory (p : Layout) (r : Regions) (u : BlockState) :
    (initialized p r u).mem = u.mem := by
  funext region off
  simp only [initialized,offsetState,metadataState,launchState,PreludeQuery.queried,
    PreludeOffsets.prepared,PreludeMetadata.prepared,PreludeLaunch.launched,BlockState.setReg_mem]

theorem pids (p : Layout) (r : Regions) (u : BlockState) :
    (initialized p r u).pids = u.pids := by
  simp only [initialized,offsetState,metadataState,launchState,PreludeQuery.queried,
    PreludeOffsets.prepared,PreludeMetadata.prepared,PreludeLaunch.launched,BlockState.setReg_pids]

theorem offset_reads (p : Layout) (r : Regions) (u : BlockState) (rg : RegionName) (off : Nat) :
    (offsetState p r u).readMem rg off = u.readMem rg off := by
  simp only [offsetState,metadataState,launchState,PreludeOffsets.prepared,
    PreludeMetadata.prepared,PreludeLaunch.launched,BlockState.setReg_readMem]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.PreludeComplete
