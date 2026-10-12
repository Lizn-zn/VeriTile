import bench.tritonbench_g.context_attn_mistral.Proofs.SafeOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.FlattenShape
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
    rowView,dimView,offset,queryMask,loadQuery,keyOffset,valueOffset,keyView,valueView,keyChannel,lanes,
    EpilogueProgram.code,EpilogueProgram.offsets,
    LoopProgram.keys,LoopProgram.keyAdvance,LoopProgram.keyMask,
    LoopProgram.scores,LoopProgram.causalMask,LoopProgram.windowMask,LoopProgram.sentinel,LoopProgram.scaling,
    LoopProgram.statsCode,LoopProgram.normalizerCode,LoopProgram.scalesCode,LoopProgram.commitCode,
    LoopProgram.vec,LoopProgram.mat,LoopProgram.column,
    LoopProgram.values,LoopProgram.valueAdvance,LoopProgram.valueMask,
    List.cons_append,List.nil_append,StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.FlattenShape
