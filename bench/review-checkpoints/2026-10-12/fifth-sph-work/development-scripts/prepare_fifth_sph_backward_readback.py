from pathlib import Path
s='''import FifthSphDraft.BackwardBase
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardReadback
open VeriTile.Triton BackwardBase
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem coord_injective (pid B c : Nat) :
    Function.Injective (fun j : TileIndex [B] => j.1.val*3+B*3*pid+c) := by
  intro a b h
  have hv : a.1.val=b.1.val := by omega
  obtain ⟨a,⟨⟩⟩ := a
  obtain ⟨b,⟨⟩⟩ := b
  have hab : a=b := Fin.ext hv
  subst b
  rfl

theorem coord_disjoint (pid B : Nat) (j : Fin B) (k : TileIndex [B]) (c d : Fin 3)
    (hcd : c≠d) : coord pid B j.val c.val≠coord pid B k.1.val d.val := by
  unfold coord
  have hc := c.isLt
  have hd := d.isLt
  intro h
  have he : c.val=d.val := by omega
  exact hcd (Fin.ext he)

'''
expand='''source,TypedSource.backward,TypedSource.backward_fp32_fp32_fp32,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,Except.bind,bind,
    exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,tile_elementwise'''
for c in range(3):
 s+=f'''theorem component{c} (X dX g : RegionName) (B N M off stride : Nat) (u v : BlockState)
    (he : exec (source X dX g B N M off stride) u=some v) (j : Fin B)
    (hcoords : ∀ d : Fin 3,coord u.pid B j.val d.val<N)
    (hg : ∀ d : Fin 11,output u.pid B stride off j.val d.val<M) :
    v.readMem dX (coord u.pid B j.val {c})=
      RealMath.gradient {c} (u.readMem X (coord u.pid B j.val 0))
        (u.readMem X (coord u.pid B j.val 1)) (u.readMem X (coord u.pid B j.val 2))
        (u.readMem dX (coord u.pid B j.val {c}))
        (fun d => u.readMem g (output u.pid B stride off j.val d.val)) := by
  simp (maxSteps := 3000000) [{expand}] at he
  subst v
  simp only [coord,Nat.add_zero]
'''
 for d in reversed(range(c+1,3)):
  s+=f'''  rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
    (P := fun idx : TileIndex [B] => idx.1.val*3+B*3*u.pids 0+{d}<N)
    (offsetFn := fun idx : TileIndex [B] => idx.1.val*3+B*3*u.pids 0+{d})
    (hOff := fun k _ _ => coord_disjoint u.pid B j k {c} {d} (by decide))]
'''
 s+=f'''  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ (coord_injective u.pid B {c}) (j,PUnit.unit)]
'''
 for d in range(3):s+=f'  have hc{d} := hcoords {d}\n'
 for d in range(11):s+=f'  have hg{d} := hg {d}\n'
 s+='  simp only [coord,output,Nat.add_zero] at hc0 hc1 hc2 '+' '.join('hg'+str(d) for d in range(11))+'\n'
 s+='''  simp (maxSteps := 3000000) only [hc0,hc1,hc2,'''+','.join('hg'+str(d) for d in range(11))+''',if_true,
    NumericDType.mul,NumericDType.add,NumericDType.sub,
    WithBot.realSub,WithBot.realAdd,WithBot.realMul,
    Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,
    RealMath.gradient,RealMath.backwardWeight,Fin.sum_univ_succ,Fin.val_zero,Fin.val_succ,
    output,coord,Nat.add_zero]
  norm_num
  ring

'''
s+='''theorem all_components (X dX g : RegionName) (B N M off stride : Nat) (u v : BlockState)
    (he : exec (source X dX g B N M off stride) u=some v) (j : Fin B)
    (hcoords : ∀ d : Fin 3,coord u.pid B j.val d.val<N)
    (hg : ∀ d : Fin 11,output u.pid B stride off j.val d.val<M) (c : Fin 3) :
    v.readMem dX (coord u.pid B j.val c.val)=
      RealMath.gradient c (u.readMem X (coord u.pid B j.val 0))
        (u.readMem X (coord u.pid B j.val 1)) (u.readMem X (coord u.pid B j.val 2))
        (u.readMem dX (coord u.pid B j.val c.val))
        (fun d => u.readMem g (output u.pid B stride off j.val d.val)) := by
  fin_cases c
  · exact component0 X dX g B N M off stride u v he j hcoords hg
  · exact component1 X dX g B N M off stride u v he j hcoords hg
  · exact component2 X dX g B N M off stride u v he j hcoords hg

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardReadback
'''
Path('/tmp/FifthSphDraft/BackwardReadback.lean').write_text(s)
