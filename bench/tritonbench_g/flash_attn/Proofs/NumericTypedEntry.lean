import bench.tritonbench_g.flash_attn.Proofs.NumericTypedModel
namespace VeriTile.Bench.TritonBenchG.FlashAttn.NumericTypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
/-- A strided, nonzero-head witness; noncausal mode loads an unaligned final KV tile. -/
def original (causal : Bool) :=
  FPProof.program Bool.false .fp16 .fp16 "Q" "K" "V" "L" "O" 1
    50 20 5 2  50 20 7 2  50 20 99 3  50 20 5 2  1 2 3 2 2 2 causal

def parts (causal : Bool) : Source.DynamicRange :=
  (Source.dynamicRange? (original causal).kernel.surfaceBody "start_n").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

def entry (causal : Bool) :
    {t : State (WithBot ℝ) // run (engine model) (parts causal).before
      (initial .fp16 .fp16)=some t} := by
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
      initial,storageType,Cell.read,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg]
    try rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FlashAttn.NumericTypedExecution
