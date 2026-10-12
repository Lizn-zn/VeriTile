from pathlib import Path
import itertools
D=Path('/tmp/FifthSphDraft');s='''import FifthSphDraft.FP
import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel (model)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def storageType : Storage → FloatDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .real

def initial (input destination upstream : Storage) : State ℝ where
  mem := fun r _ => if r="x" then .mk (storageType input).toTileDType (ofFloat (storageType input) 2)
    else if r="g" then .mk (storageType upstream).toTileDType (ofFloat (storageType upstream) 1)
    else .mk (storageType destination).toTileDType (ofFloat (storageType destination) 5)
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then 1 else 0
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

/-! Empty-register execution for every independent storage choice, with a
nonempty partial tile. This algebra does not model physical GPU rounding. -/
'''
for kind,n,labels,ptrs in [('forward',2,'input destination','"x" "y"'),('backward',3,'input destination upstream','"x" "dx" "g"')]:
 initial='initial input destination '+('.fp32' if n==2 else 'upstream');params=labels+' Bool.false '+ptrs+' 2 9 36 1 12';aliases=','.join(kind+'_'+'_'.join(c) for c in itertools.product(['fp16','bf16','fp32'],repeat=n))
 s+=f'''theorem {kind}_executes ({labels} : Storage) :
    ∃ v,FP.Structural.exec (engine model)
      ({kind} {params}) ({initial})=some v := by
  '''+' <;> '.join('cases '+x for x in labels.split())+f'''
  all_goals
    simp (maxSteps := 3000000) [FP.Structural.exec,{kind},{aliases},ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,initial,storageType,ofFloat,toFloat,Region.cast,
      State.setReg,FP.Structural.store,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      TileShape.allIndices,List.finRange_succ,State.write]

'''
 s+=f'''theorem {kind}_candidate_executes ({labels} : Storage) :
    ∃ v,FP.Structural.exec (engine model)
      ({kind} {params.replace('Bool.false','Bool.true')}) ({initial})=some v := by
  obtain ⟨v,hv⟩ := {kind}_executes {labels}
  have h := FP.ProfiledRewrite.sound
    (FPProof.{kind}_equivalent {labels} {ptrs} 2 9 36 1 12)
    ℝ model (fun _ _ => True) (by apply FP.models_of_atoms;intro e he;cases he)
    ({initial}) (by simp [FPProof.{kind}Program])
  exact ⟨v,h.symm.trans hv⟩

'''
s+='''#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedExecution
''';(D/'TypedExecution.lean').write_text(s)
p=D/'Execution.lean';p.write_text(p.read_text().replace('MemCell.toReal','MemCell.readAs'))
