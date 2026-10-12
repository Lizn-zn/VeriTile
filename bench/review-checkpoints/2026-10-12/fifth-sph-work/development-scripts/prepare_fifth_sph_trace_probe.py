from pathlib import Path
s=Path('/tmp/FifthSphCanonical/ExecuteComposed.lean').read_text();header=s[:s.index('theorem assigns_exists')].replace('import VeriTile.Triton.Float.Erasure','import VeriTile.Triton.Float.Erasure\nimport VeriTile.Triton.Memory.TraceProofs')
base=Path('/tmp/FifthSphDraft/BackwardBase.lean').read_text();a=base.index('structure Bounds');b=base.index('theorem trace_safe',a);bounds=base[a:b]
a=base.index('theorem trace_safe');b=base.index(' := by',a);sig=base[a:b]+' := by\n'
a=s.index('  simp (maxSteps := 3000000) only [source,');b=s.index('\n  unfold exec',a);expansion=s[a:b]
foot='''Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt'''
finish='''    all_goals intro j hj
    all_goals first
'''
for group,n in [('coordinates',3),('gradient',3),('upstream',11)]:
 for i in range(n):finish+=f'      | simpa [coord,output] using hb.{group} {i} j hj\n'
safe='simp (maxSteps := 3000000) ['+foot+']\n'+finish
proof=expansion+'\n  unfold Kernel.TraceSafe\n'
for i in range(89):
 name='assign_eval_cons' if i in [0,1,2,3,7,8] else 'assign_cons'
 proof+='  refine TraceProofs.'+name+' (by\n    '+safe.rstrip()+'\n  ) ?_\n'
 if name=='assign_eval_cons':proof+=f'  intro t{i} ht{i}\n  simp [evalOp.eq_def,tile_elementwise] at ht{i}\n  subst t{i}\n'
 else:proof+=f'  intro t{i}\n'
proof+='  simp (maxSteps := 3000000) [Stmt.TraceSafeList,Stmt.TraceSafe,stepStmts,stepStmt,'+foot+']\n  repeat\' apply And.intro\n'+finish.replace('    all_goals','  all_goals').replace('      |','    |')
Path('/tmp/FifthSphCanonical/TraceProbe.lean').write_text(header+bounds+sig+proof+'\nend\nend VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardBase\n')
# The validated frame proof retains opaque numerical values between assignments.
p=Path('/tmp/FifthSphDraft/BackwardBase.lean');base=p.read_text();frame=Path('/tmp/FifthSphCanonical/FrameProbe.lean').read_text();a=frame.index('theorem assigns_inv');b=frame.index('\nend\n',a);fragment=frame[a:b]
a=base.index('theorem frame');b=base.index('theorem flatten_ok',a);base=base[:a]+fragment+'\n\n'+base[b:];p.write_text(base)
