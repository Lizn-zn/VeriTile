import bench.tritonbench_g.context_attn_fwd.Proofs.SafeOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.FlattenShape
open VeriTile.Triton Formula Program PreludeProgram
set_option maxHeartbeats 1400000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

theorem supported (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) (scale : ℝ) :
    (kernel p r storage swapped scale).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk,Program.storage_erasure,WholeProgram.source,PreludeProgram.source,
    EpilogueProgram.source]
  unfold WholeProgram.loop
  rw [LoopProgram.source]
  cases swapped
  all_goals simp [launchCode,metadataCode,offsetCode,queryCode,loadMeta,n,z,rows,dimensions,
    rowView,dimView,offset,queryMask,loadQuery,endCandidate,fullLength,ending,
    EpilogueProgram.code,EpilogueProgram.offsets,
    LoopProgram.keys,LoopProgram.keyOffsets,LoopProgram.keyMask,LoopProgram.keyView,LoopProgram.keyChannel,LoopProgram.lanes,
    LoopProgram.scores,LoopProgram.causalMask,LoopProgram.scaling,LoopProgram.stats,
    LoopProgram.vec,LoopProgram.mat,LoopProgram.column,
    LoopProgram.values,LoopProgram.valueOffsets,LoopProgram.valueMask,LoopProgram.valueView,LoopProgram.valueChannel,
    List.cons_append,List.nil_append,StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.FlattenShape
