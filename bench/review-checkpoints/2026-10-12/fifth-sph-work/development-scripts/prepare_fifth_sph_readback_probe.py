from pathlib import Path
original=Path('/tmp/FifthSphDraft/BackwardReadback.lean').read_text()
header=original[:original.index('theorem component0')].replace('import FifthSphDraft.BackwardBase','import FifthSphCanonical.FrameProbe')
a=original.index('theorem component0');b=original.index('theorem component1',a);part=original[a:b]
a=part.index('  simp (maxSteps');b=part.index('\n  subst v',a)
frame=Path('/tmp/FifthSphCanonical/FrameProbe.lean').read_text();x=frame.index('  simp (maxSteps := 3000000) only [source,');y=frame.index('\n  subst v',x);inv=frame[x:y]
inv=inv.replace('Except.bind,bind,↓reduceIte]','Except.bind,bind,↓reduceIte,Nat.sub_zero]')
for i in range(89):
 if i not in [0,1,2,3,7,8,87,88]:inv=inv.replace(f'  clear ht{i}\n','')
part=part[:a]+inv+part[b:]
a=part.index('  simp (maxSteps := 3000000) only [hc0');part=part[:a]
known=[]
for i in list(range(4,7))+list(range(9,87)):
 idx='PUnit.unit' if 20<=i<74 else '(j,PUnit.unit)'
 part+=f'  have h{i} := congrArg (fun r => r.map (fun v => v.data {idx})) ht{i}\n'
 decls='evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,'+','.join('hg'+str(d) for d in range(11))
 if known:decls+=','+','.join('←h'+str(k) for k in known)
 part+=f'  simp (maxSteps := 3000000) [{decls}] at h{i}\n'
 known.append(i)
part+='''  rw [←h86]
  simp only [WithBot.unbotD,WithBot.recBotCoe]
  norm_num [RealMath.gradient,RealMath.backwardWeight,Fin.sum_univ_succ,coord,output,BlockState.pid]
  ring

'''
Path('/tmp/FifthSphCanonical/ReadbackProbe.lean').write_text(header+part+'\n#guard_msgs (drop info) in\n#auditModuleAxioms\nend\nend VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardReadback\n')
