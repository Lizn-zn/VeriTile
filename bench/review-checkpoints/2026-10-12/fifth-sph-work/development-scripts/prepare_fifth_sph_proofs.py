from pathlib import Path
prefix='''import FifthSphDraft.Kernels
import bench.tritonbench_g.destindex_copy.Proofs.IndexRewrite
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.FPProof
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

'''
s=prefix
for name,tys,regions in [('forward',['input','output'],['X','Y']),('backward',['input','gradient','upstream'],['X','dX','g'])]:
 ts=' '.join(tys);rs=' '.join(regions)
 s+=f'''def {name}Program ({ts} : Storage) (swapped : Bool) ({rs} : RegionName) (B N M off stride : Nat) : Program where
  kernel := {name} {ts} swapped {rs} B N M off stride
  domain := []

theorem {name}_equivalent ({ts} : Storage) ({rs} : RegionName) (B N M off stride : Nat) :
    {name}Program {ts} Bool.false {rs} B N M off stride ≡[assumptions]
    {name}Program {ts} Bool.true {rs} B N M off stride := by
  '''+' <;> '.join('cases '+t for t in tys)+''' <;> equiv_decompose <;> fp_prove

#print_fp_assumptions '''+name+'_equivalent\n\n'
s+='''#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.FPProof
''';Path('/tmp/FifthSphDraft/FP.lean').write_text(s)
s='''import FifthSphDraft.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.Erasure
open VeriTile.Triton TypedSource
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

'''
for name,tys,regions in [('forward',['input','output'],['X','Y']),('backward',['input','gradient','upstream'],['X','dX','g'])]:
 ts=' '.join(tys);rs=' '.join(regions)
 s+=f'''theorem {name}_body ({ts} : Storage) (swapped : Bool) ({rs} : RegionName) (B N M off stride : Nat) :
    ({name} {ts} swapped {rs} B N M off stride).eraseDType.toAlgKernel=
    ({name} '''+' '.join('.fp32' for _ in tys)+f''' swapped {rs} B N M off stride).eraseDType.toAlgKernel := by
  '''+' <;> '.join('cases '+t for t in tys)+''' <;> cases swapped
  all_goals simp only ['''+name+''',ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind]
  all_goals rfl

'''
s+='''#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.Erasure
''';Path('/tmp/FifthSphDraft/Erasure.lean').write_text(s)
