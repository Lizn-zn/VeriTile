from pathlib import Path
s=Path('/tmp/FifthSphCanonical/ExecuteComposed.lean').read_text();a=s.index('theorem assigns_exists');header=s[:a]
helper='''theorem assigns_inv {d sh name} {e : Op d sh} {s t : BlockState} {rest : List Stmt}
    (h : stepStmts (.assign d sh name e :: rest) s=some t) :
    ∃ v,evalOp e s=some v ∧ stepStmts rest (s.setReg name d sh v)=some t := by
  simp only [stepStmts,stepStmt] at h
  cases hv : evalOp e s with
  | none => simp [hv] at h
  | some v => exact ⟨v,rfl,by simpa [hv] using h⟩

'''
base=Path('/tmp/FifthSphCanonical/BackwardBase.lean').read_text();a=base.index('theorem frame');b=base.index('theorem flatten_ok',a);frame=base[a:b];a=frame.index('  simp (maxSteps');b=frame.index('\n  subst v',a)
expansion=s[s.index('  simp (maxSteps := 3000000) only [source,'):s.index('\n  unfold exec',s.index('theorem executes'))]
new=expansion.rstrip()+' at he\n  unfold exec at he\n'
for i in range(89):
 new+=f'  obtain ⟨t{i},ht{i},he⟩ := assigns_inv he\n'
 if i in [0,1,2,3,7,8]:new+=f'  simp [evalOp.eq_def,tile_elementwise] at ht{i}\n  subst t{i}\n'
 else:new+=f'  clear ht{i}\n'
new+='  simp (maxSteps := 3000000) [stepStmts,stepStmt,evalOp.eq_def,tile_elementwise] at he'
frame=frame[:a]+new+frame[b:]
Path('/tmp/FifthSphCanonical/FrameProbe.lean').write_text(header+helper+frame+'\nend\nend VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardBase\n')
