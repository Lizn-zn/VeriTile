import bench.tritonbench_g.flash_attn.Proofs.NumericTypedSecond
namespace VeriTile.Bench.TritonBenchG.FlashAttn.NumericTypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
def afterLoop (causal : Bool) : State (WithBot ℝ) :=
  if causal then (first causal).val else (second).val

def post (causal : Bool) :
    {t : State (WithBot ℝ) // run (engine model) (parts causal).after
      (afterLoop causal)=some t} := by
  cases causal
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [parts,original,FPProof.program,kernel,kernel_fp16_fp16,kernel_fp16_bf16,kernel_fp16_fp32,kernel_bf16_fp16,kernel_bf16_bf16,kernel_bf16_fp32,kernel_fp32_fp16,kernel_fp32_bf16,kernel_fp32_fp32,
      ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
      String.reduceEq,beq_self_eq_true,Bool.false_eq_true,↓reduceIte]
    simp (maxSteps := 300000) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
      Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,model,
      ComputeDType.eraseDType,numeric,bop,numericLe,numericLt,numericEq,natLt,
      initial,storageType,Cell.read,ofFloat,toFloat,TileShape.axisDim,Region.cast,afterLoop,entry,first,second,
      FP.Structural.store,masked_scatter_regs]
    try rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FlashAttn.NumericTypedExecution
