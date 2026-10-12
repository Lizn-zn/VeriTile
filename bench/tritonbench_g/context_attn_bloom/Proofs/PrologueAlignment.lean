import bench.tritonbench_g.context_attn_bloom.Proofs.PrologueDefinition
namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.RealSetup
open VeriTile.Triton Math Program
set_option maxHeartbeats 1500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem prelude_source (p : Layout) (r : Regions) : prelude p r = explicitPrelude p r := by
  simp only [prelude,explicitPrelude,lower,parts,mathematical,prologueSource,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,Except.bind,bind,Except.toOption,Option.getD,
    Kernel.eraseDType,Stmt.eraseDTypeList,List.map_cons,List.map_nil,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,VeriTile.Triton.eraseDType,
    NumericDType.eraseDType,IntegralDType.eraseDType,ComparableDType.eraseDType,
    Source.intRange?,String.reduceEq,Bool.false_eq_true,↓reduceIte,Option.map_some]
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.RealSetup
