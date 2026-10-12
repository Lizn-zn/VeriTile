from pathlib import Path
import sys,json,re
root=Path('/home/argustest/lizn/VeriTile');sys.path.insert(0,str(root/'scripts'));from spec_sheet import split_decls
old=root/'bench/tritonbench_g/fifth_order_sph_harmonics/FifthOrderSphHarmonics.lean';decls=split_decls(old.read_text());chosen=set(json.loads(Path('/tmp/FifthSphDraft/forward-legacy-closure.json').read_text()))
s='''import FifthSphDraft.Kernels
import FifthSphDraft.RealMath
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.ForwardReal
open VeriTile.Triton
open scoped VeriTile.Triton.GroupedMasked2DKernelIO
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

'''
extras='''TypedSource.forward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
      ComputeDType.eraseDType,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
      Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      IntegralDType.eraseDType,Region.cast,Except.bind,bind,'''
for d in decls:
 if d['name'] not in chosen:continue
 if d['name']=='fifth_order_fwd_surface':
  s+='''def fifth_order_fwd_surface (coord_ptr output_ptr : RegionName)
    (block_size coord_numel output_numel col_offset output_stride : Nat) : ComputeKernel :=
  (TypedSource.forward .fp32 .fp32 Bool.false coord_ptr output_ptr block_size coord_numel
    output_numel col_offset output_stride).eraseDType

''';continue
 t=d['text'];t=t.replace('fifth_order_fwd_surface,','fifth_order_fwd_surface,'+extras)
 s+=t+'\n\n'
s+='''#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.ForwardReal
''';Path('/tmp/FifthSphDraft/ForwardReal.lean').write_text(s)
