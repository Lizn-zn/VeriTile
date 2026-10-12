import bench.tritonbench_g.context_attn_llama.Proofs.EpilogueProgram
namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama.WholeProgram
open VeriTile.Triton Formula Program PreludeProgram
set_option maxHeartbeats 2500000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

def loop (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) : Stmt :=
  .forRangeInt "start_n" (.constInt 0)
    (.mul .int .nil (n "block_mask").castNatToInt (z "block_end_loc"))
    (.constNat p.BN) (LoopProgram.body p r swapped scale)

theorem source (p : Layout) (r : Regions) (swapped : Bool) (scale : ℝ) :
    (mathematical p r swapped scale).eraseDType.toAlgKernel.body =
      prelude p r ++ [loop p r swapped scale] ++ EpilogueProgram.post p r := by
  rw [← prelude_shared p r swapped scale,← EpilogueProgram.shared p r swapped scale]
  cases swapped
  all_goals simp only [lower,parts,mathematical,kernel,VeriTile.Bench.TritonBenchG.ContextAttnLlama.original,VeriTile.Bench.TritonBenchG.ContextAttnLlama.scaleFirst,ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,Kernel.eraseDType,Stmt.eraseDTypeList,
    Bool.false_eq_true,↓reduceIte,String.reduceEq,Option.map,Option.getD,Except.toOption,
    Source.intRange?,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Except.bind,bind,List.map_cons,List.map_nil,
    Stmt.eraseDType,Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    loop,LoopProgram.body,n,z,List.cons_append,List.nil_append]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    Region.cast_cast,Region.cast_id,eraseDType_float,FloatDType.toTileDType]
  all_goals rfl
end VeriTile.Bench.TritonBenchG.ContextAttnLlama.WholeProgram
