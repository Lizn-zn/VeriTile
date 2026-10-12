import bench.tritonbench_g.context_attn_bloom.Proofs.ScoresStaged
import bench.tritonbench_g.context_attn_bloom.Proofs.StatsExecution
import bench.tritonbench_g.context_attn_bloom.Proofs.NormalizerExecution
import bench.tritonbench_g.context_attn_bloom.Proofs.ScalesExecution
import bench.tritonbench_g.context_attn_bloom.Proofs.ValuesExecution
import bench.tritonbench_g.context_attn_bloom.Proofs.CommitExecution
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.LoopProgram
open VeriTile.Triton Math Program RealSetup PreludeProgram
set_option maxHeartbeats 2200000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

def body (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) : List Stmt :=
  lower (parts p r swapped scale).body

/-- Named source segments cover every instruction of the complete signed loop. -/
theorem split (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) :
    body p r swapped scale =
      LoopKeysProgram.keys p r ++ ScoresProgram.scores p r swapped scale ++
      UpdateProgram.stats p r swapped scale ++ UpdateProgram.normalizer p r swapped scale ++
      UpdateProgram.scales p r swapped scale ++ ValuesProgram.values p r ++
      UpdateProgram.commit p r swapped scale := by
  rw [LoopKeysProgram.source,ScoresProgram.source,(UpdateProgram.sources p r swapped scale).1,
    (UpdateProgram.sources p r swapped scale).2.1,(UpdateProgram.sources p r swapped scale).2.2.1,
    ValuesProgram.source,(UpdateProgram.sources p r swapped scale).2.2.2]
  cases swapped
  all_goals simp only [body,lower,parts,mathematical,ComputeKernel.surfaceBody,
    Bool.false_eq_true,↓reduceIte,String.reduceEq,Option.map,Option.getD,Except.toOption,
    Source.intRange?,Source.prefixBefore,List.length_cons,List.length_nil,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Except.bind,bind,List.map_cons,List.map_nil,
    Stmt.eraseDType,Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    LoopKeysProgram.code,LoopKeysProgram.gather,LoopKeysProgram.tokenIndices,LoopKeysProgram.lanes,
    LoopKeysProgram.keyOffsets,LoopKeysProgram.keyMask,LoopKeysProgram.tokenView,LoopKeysProgram.channelView,
    ScoresProgram.code,ScoresProgram.causalMask,ScoresProgram.scaling,
    UpdateProgram.statsCode,UpdateProgram.normalizerCode,UpdateProgram.scalesCode,UpdateProgram.commitCode,
    UpdateProgram.vec,UpdateProgram.mat,UpdateProgram.column,UpdateProgram.rescaleGuard,
    ValuesProgram.code,ValuesProgram.offsets,ValuesProgram.mask,ValuesProgram.tokenView,ValuesProgram.channelView,
    n,z,List.cons_append,List.nil_append]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    Region.cast_cast,Region.cast_id,eraseDType_float]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.LoopProgram
