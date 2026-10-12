from pathlib import Path
src=Path('/tmp/FifthSphCanonical/TypedPre.lean').read_text()
p=Path('/tmp/FifthSphDraft')
a=src.index('theorem one_forward');header=src[:a]
footer='\n#guard_msgs (drop info) in\n#auditModuleAxioms\nend\nend VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedExecution\n'
(p/'TypedModel.lean').write_text(header+footer)
import re
body=src[a:src.index('#guard_msgs',a)];proof=body[body.index('  simp only [FP.Structural.exec'):]
proof=re.sub(r'  run_tac do IO.FS.writeFile[^\n]+\n','',proof)
proof=re.sub(r'  intro t\d+','  intro',proof)
assignstart=proof.index('  apply assigns_exists');end=proof.rindex('  simp (maxSteps')
assignchunk=proof[assignstart:proof.index('\n  apply assigns_exists',assignstart+3)]+'\n'
final=proof[end:].rstrip()+'\n'
module_header='''import FifthSphDraft.TypedModel
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
'''
storages=['fp16','bf16','fp32']
for direction,n,nparams in [('forward',62,2),('backward',89,3)]:
 from itertools import product
 variants=','.join(direction+'_'+'_'.join(xs) for xs in product(storages,repeat=nparams))
 params='input destination'+(' upstream' if nparams==3 else '')
 kernelargs='input destination'+(' upstream' if nparams==3 else '')+' Bool.false '+ ('"x" "y"' if nparams==2 else '"x" "dx" "g"')+' 2 9 36 1 12'
 initialargs='input destination '+('upstream' if nparams==3 else '.fp32')
 s=module_header+f'''\n/-- Successful nonempty typed execution for every independent storage choice.
Only the abstract singleton algebra is used; memory tags and masks are retained. -/
theorem {direction}_executes ({params} : Storage) :
    ∃ v,FP.Structural.exec (engine model)
      ({direction} {kernelargs}) (initial {initialargs})=some v := by
  cases input <;> cases destination'''+(' <;> cases upstream' if nparams==3 else '')+'\n  all_goals\n'
 block=f'  simp only [FP.Structural.exec,{direction},{variants},ComputeKernel.surfaceBody,Bool.false_eq_true,↓reduceIte]\n'+assignchunk*n+final
 s+='\n'.join('  '+l if l else '' for l in block.splitlines())+'\n'+footer
 (p/('Typed'+direction.title()+'Execution.lean')).write_text(s)
old=(p/'TypedExecution.lean').read_text()
parts=[]
for name,nextname in [('forward_candidate_executes','backward_executes'),('backward_candidate_executes',None)]:
 a=old.index('theorem '+name);b=old.index('theorem '+nextname,a) if nextname else old.index('#guard_msgs',a);parts.append(old[a:b])
(p/'TypedExecution.lean').write_text(module_header.replace('import FifthSphDraft.TypedModel','import FifthSphDraft.TypedForwardExecution\nimport FifthSphDraft.TypedBackwardExecution')+'\n'.join(parts)+footer)
print('Prepared shared unit model and all 9+27 typed execution cases')
