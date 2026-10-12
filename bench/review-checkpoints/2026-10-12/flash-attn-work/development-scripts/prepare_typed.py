from pathlib import Path
w=Path(__file__).resolve().parent;p=w/'FlashAttnDraft'
head='''namespace VeriTile.Bench.TritonBenchG.FlashAttn.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
'''
names=','.join(f'kernel_{a}_{b}' for a in ['fp16','bf16','fp32'] for b in ['fp16','bf16','fp32'])
source=f'''    simp only [parts,original,FPProof.program,kernel,{names},
      ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
      String.reduceEq,beq_self_eq_true,Bool.false_eq_true,↓reduceIte]
'''
semantics='''run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
      Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,model,
      ComputeDType.eraseDType,numeric,bop,numericLe,numericLt,numericEq,natLt,
      initial,storageType,Cell.read,ofFloat,toFloat,TileShape.axisDim,Region.cast'''
def body(extra=''):
 return '''  cases input <;> cases output <;> cases causal
  all_goals
    refine ⟨?_,?_⟩
    swap
'''+source+f'''    simp (maxSteps := 300000) [{semantics}{extra}]
    try rfl
'''
ending='''
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FlashAttn.TypedExecution
'''
s='import FlashAttnDraft.TypedModel\n'+head+'''/-- A strided, nonzero-head witness; noncausal mode loads an unaligned final KV tile. -/
def original (input output : Storage) (causal : Bool) :=
  FPProof.program Bool.false input output "Q" "K" "V" "L" "O" 1
    50 20 5 2  50 20 7 2  50 20 99 3  50 20 5 2  1 2 3 2 2 2 causal

def parts (input output : Storage) (causal : Bool) : Source.DynamicRange :=
  (Source.dynamicRange? (original input output causal).kernel.surfaceBody "start_n").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

def entry (input output : Storage) (causal : Bool) :
    {t : State PUnit // run (engine model) (parts input output causal).before
      (initial input output)=some t} := by
'''+body(',State.setReg')+ending
(p/'TypedEntry.lean').write_text(s)
for stage,previous,index in [('first','entry',0),('second','first',2)]:
 s=f'import FlashAttnDraft.Typed{ "Entry" if stage=="first" else "First"}\n'+head+f'''def {stage} (input output : Storage) (causal : Bool) :
    {{t : State PUnit // run (engine model) (parts input output causal).body
      (({previous} input output causal).val.setReg "start_n" .nat [] (fun _ => {index}))=some t}} := by
'''+body(',entry'+(',first' if stage=='second' else ''))+ending
 (p/f'Typed{stage.title()}.lean').write_text(s)
