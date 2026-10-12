import bench.tritonbench_g.context_attn_bloom.Proofs.SafeOps
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.FlattenShape
open VeriTile.Triton Math Program RealSetup PreludeProgram
set_option maxHeartbeats 1400000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

theorem supported (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) (scale : ℝ) :
    (kernel p r storage swapped scale).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk,Program.storage_erasure,WholeProgram.source,PreludeProgram.source,
    EpilogueProgram.source]
  unfold WholeProgram.loop
  rw [LoopProgram.split,LoopKeysProgram.source,ScoresProgram.source,
    (UpdateProgram.sources p r swapped scale).1,(UpdateProgram.sources p r swapped scale).2.1,
    (UpdateProgram.sources p r swapped scale).2.2.1,(UpdateProgram.sources p r swapped scale).2.2.2,
    ValuesProgram.source]
  cases swapped
  all_goals simp [launchCode,metadataCode,offsetCode,queryCode,loadMeta,n,z,rows,dimensions,
    rowView,dimView,offset,queryMask,loadQuery,endCandidate,fullLength,ending,
    EpilogueProgram.code,EpilogueProgram.offsets,
    LoopKeysProgram.code,LoopKeysProgram.gather,LoopKeysProgram.tokenIndices,LoopKeysProgram.lanes,
    LoopKeysProgram.keyOffsets,LoopKeysProgram.keyMask,LoopKeysProgram.tokenView,LoopKeysProgram.channelView,
    ScoresProgram.code,ScoresProgram.causalMask,ScoresProgram.scaling,
    UpdateProgram.statsCode,UpdateProgram.normalizerCode,UpdateProgram.scalesCode,UpdateProgram.commitCode,
    UpdateProgram.vec,UpdateProgram.mat,UpdateProgram.column,UpdateProgram.rescaleGuard,
    ValuesProgram.code,ValuesProgram.offsets,ValuesProgram.mask,ValuesProgram.tokenView,ValuesProgram.channelView,
    List.cons_append,List.nil_append,StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.FlattenShape
