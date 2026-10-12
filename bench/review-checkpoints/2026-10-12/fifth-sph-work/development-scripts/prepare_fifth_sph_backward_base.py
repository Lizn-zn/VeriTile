from pathlib import Path
expand='''source,TypedSource.backward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,Except.bind,bind,
    exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,tile_elementwise'''
s='''import FifthSphDraft.Kernels
import FifthSphDraft.RealMath
import VeriTile.Triton.Float.Erasure
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardBase
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def source (X dX g : RegionName) (B N M off stride : Nat) : Kernel :=
  (TypedSource.backward .fp32 .fp32 .fp32 Bool.false X dX g B N M off stride).eraseDType.toAlgKernel

def coord (pid B j c : Nat) : Nat := j*3+B*3*pid+c
def output (pid B stride off j c : Nat) : Nat := j*stride+B*stride*pid+off+c

theorem executes (X dX g : RegionName) (B N M off stride : Nat) (u : BlockState) :
    ∃ v,exec (source X dX g B N M off stride) u=some v := by
  simp (maxSteps := 3000000) ['''+expand+''']

theorem frame (X dX g : RegionName) (B N M off stride : Nat) (u v : BlockState)
    (he : exec (source X dX g B N M off stride) u=some v) (r : RegionName) (o : Nat)
    (hmiss : ∀ c : Fin 3,∀ j : Fin B,coord u.pid B j.val c.val<N → ¬(dX=r ∧ coord u.pid B j.val c.val=o)) :
    v.mem r o=u.mem r o := by
  simp (maxSteps := 3000000) ['''+expand+'''] at he
  subst v
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
    (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
      (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl))
  · intro j _ hj ho
    exact hmiss 2 j.1 (by simpa [coord] using hj) (by simpa [coord] using ho)
  · intro j _ hj ho
    exact hmiss 1 j.1 (by simpa [coord] using hj) (by simpa [coord] using ho)
  · intro j _ hj ho
    exact hmiss 0 j.1 (by simpa [coord] using hj) (by simpa [coord] using ho)

theorem flatten_ok (X dX g : RegionName) (B N M off stride : Nat) :
    (source X dX g B N M off stride).FlattenOk := by
  unfold Kernel.FlattenOk
  simp (maxSteps := 3000000) ['''+expand+''',StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

structure Bounds (X dX g : RegionName) (B N M off stride : Nat) (u : BlockState) (bounds : RegionBounds) : Prop where
  coordinates : ∀ c : Fin 3,∀ j : Fin B,coord u.pid B j.val c.val<N → coord u.pid B j.val c.val<bounds X
  gradient : ∀ c : Fin 3,∀ j : Fin B,coord u.pid B j.val c.val<N → coord u.pid B j.val c.val<bounds dX
  upstream : ∀ c : Fin 11,∀ j : Fin B,output u.pid B stride off j.val c.val<M → output u.pid B stride off j.val c.val<bounds g

theorem trace_safe (X dX g : RegionName) (B N M off stride : Nat) (u : BlockState) (bounds : RegionBounds)
    (hb : Bounds X dX g B N M off stride u bounds) :
    Kernel.TraceSafe bounds (source X dX g B N M off stride) u := by
  unfold Kernel.TraceSafe
  simp (maxSteps := 3000000) ['''+expand+''',
    Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
    MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
    Op.PointerAddressesSafeOn,Op.MemorySafe]
  repeat' apply And.intro
  all_goals intro j hj
  all_goals first
'''
for field,n in [('coordinates',3),('upstream',11),('gradient',3)]:
 for c in range(n):s+=f'    | simpa [coord,output] using hb.{field} {c} j hj\n'
s+='''
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardBase
''';Path('/tmp/FifthSphDraft/BackwardBase.lean').write_text(s)
