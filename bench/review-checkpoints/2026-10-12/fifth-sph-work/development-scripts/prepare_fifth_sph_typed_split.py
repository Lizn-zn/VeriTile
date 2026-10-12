from pathlib import Path
s=Path('/tmp/FifthSphCanonical/TypedKernelProbe.lean').read_text();a=s.index('theorem one_forward');s=s[:a]
s+='''theorem assigns_exists {d sh name} {e : ComputeExpr d sh} {s : State PUnit} {rest : List ComputeStmt}
    (he : ∃ v,evalExpr (engine model) e s=some v)
    (hr : ∀ v,∃ t,run (engine model) rest (s.setReg name d sh v)=some t) :
    ∃ t,run (engine model) (.assign d sh name e :: rest) s=some t := by
  obtain ⟨v,hv⟩ := he
  obtain ⟨t,ht⟩ := hr v
  refine ⟨t,?_⟩
  simpa only [run,step,hv,Option.bind_some] using ht

theorem one_forward :
    ∃ v,FP.Structural.exec (engine model)
      (forward .fp32 .fp32 Bool.false "x" "y" 2 9 36 1 12) (initial .fp32 .fp32 .fp32)=some v := by
  simp only [FP.Structural.exec,forward,forward_fp32_fp32,ComputeKernel.surfaceBody,Bool.false_eq_true,↓reduceIte]
'''
lemmas='evalExpr,evalComputeOp,evalOp_unfold,State.setReg,Option.bind_some,String.reduceEq,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast'
for i in range(62):
 s+='  apply assigns_exists\n  case he =>\n    simp only ['+lemmas+']\n    exact ⟨_,rfl⟩\n'
 s+=f'  intro t{i}\n  run_tac do IO.FS.writeFile "/tmp/fifth-sph-typed-split-progress.txt" "assignment {i}"\n'
s+='  simp (maxSteps := 3000000) [run,step,FP.Structural.store,'+lemmas+',initial,TileShape.allIndices,List.finRange_succ,State.write]\n'
s+='\n#guard_msgs (drop info) in\n#auditModuleAxioms\nend\nend VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedExecution\n'
Path('/tmp/FifthSphCanonical/TypedSplit.lean').write_text(s)
