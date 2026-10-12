from pathlib import Path
import re
p=Path('/tmp/FifthSphDraft')
base=Path('/tmp/FifthSphCanonical/ReadbackMemoryPre.lean').read_text()
old=(p/'BackwardReadback.lean').read_text()
header=base[:base.index('theorem component0')].replace('import FifthSphCanonical.FrameProbe\nimport FifthSphCanonical.NormalizedSource','import FifthSphDraft.BackwardBase')
footer='\n#guard_msgs (drop info) in\n#auditModuleAxioms\nend\nend VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardReadback\n'
(p/'BackwardReadbackCommon.lean').write_text(header+footer)
modhead='''import FifthSphDraft.BackwardReadbackCommon
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardReadback
open VeriTile.Triton BackwardBase
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
'''
inv=base[base.index('  change exec (NormalizedSource.original'):base.index('  simp only [coord,Nat.add_zero,BlockState.pid]')]
he_tail=base[base.index('  have hc0 :='):base.index('  simp only [↓evalOp.eq_def',base.index('  have hc0 :='))]
lanes=base[base.index('  simp only [↓evalOp.eq_def',base.index('  have hc0 :=')):base.index('  rw [←h86]')]
last=lanes[lanes.index('  simp only [↓evalOp.eq_def',lanes.index('lane 85')):]
pre=lanes[:lanes.index('  simp only [↓evalOp.eq_def',lanes.index('lane 85'))]
for c in range(3):
 a=old.index(f'theorem component{c}');b=old.index(' := by',a)+len(' := by')
 statement=old[a:b]+'\n'
 stores_a=old.index('  simp only [coord,Nat.add_zero,BlockState.pid]',b);stores_b=old.index('  have hc0 :=',stores_a)
 stores=old[stores_a:stores_b]
 thisinv=inv
 for i in range(86,89):
  thisinv=thisinv.replace(f'  clear ht{i}\n','')
  if i!=86+c:thisinv=thisinv.replace(f'  obtain ⟨t{i},ht{i},he⟩ := assigns_inv he\n',f'  obtain ⟨t{i},ht{i},he⟩ := assigns_inv he\n  clear ht{i}\n')
 thislast=re.sub(r'\b(h|ht)86\b',lambda m:m[1]+str(86+c),last)
 part=statement+thisinv+stores+he_tail+pre+thislast+f'''  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-{c}-progress.txt" "polynomial equation derived"
  rw [←h{86+c}]
  simp only [WithBot.unbotD,WithBot.recBotCoe]
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-{c}-progress.txt" "readback substituted"
  norm_num [RealMath.gradient,RealMath.backwardWeight,Fin.sum_univ_succ,coord,output,BlockState.pid]
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-{c}-progress.txt" "normalization complete"
  ring
'''
 part=part.replace('"/tmp/fifth-sph-readback-memory-pre-progress.txt"',f'"/tmp/fifth-sph-readback-{c}-progress.txt"')
 (p/f'BackwardReadback{c}.lean').write_text(modhead+part+footer)
agg='\n'.join(f'import FifthSphDraft.BackwardReadback{c}' for c in range(3))+'\n'+modhead[modhead.index('namespace'):]+old[old.index('theorem all_components'):old.index('#guard_msgs',old.index('theorem all_components'))]+footer
(p/'BackwardReadback.lean').write_text(agg)
print('Prepared separate full-source readbacks for all three components')
