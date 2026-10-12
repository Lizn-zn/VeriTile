import bench.tritonbench_g.dequantize_matmul.Kernels
import bench.tritonbench_g.dequantize_matmul.Proofs.RealReference
import VeriTile.Triton.Float.Erasure
namespace VeriTile.Bench.TritonBenchG.DequantizeMatmul.Projection
open VeriTile.Triton TypedSource
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem source_projects (output : Storage) (swapped : Bool) (b scale out : RegionName)
    (K N sbk sbn sok son BN BK : Nat) :
    (kernel output swapped b scale out K N sbk sbn sok son BN BK).toAlgorithm? =
      .ok (kernel output swapped b scale out K N sbk sbn sok son BN BK).toAlgKernel := by
  cases output <;> cases swapped <;> rfl

theorem original_body (output : Storage) (b scale out : RegionName)
    (K N sbk sbn sok son BN BK : Nat) :
    (kernel output Bool.false b scale out K N sbk sbn sok son BN BK).eraseDType.toAlgKernel.body =
      (RealReference.dequantize_kernel b scale out K N sbk sbn sok son BN BK).toAlgKernel.body := by
  cases output
  all_goals simp only [kernel,RealReference.dequantize_kernel,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,Except.bind,bind,eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType]
  all_goals rfl

theorem original_exec (A : FlatAlloc) (output : Storage) (b scale out : RegionName)
    (K N sbk sbn sok son BN BK : Nat) (s : BlockState) :
    exec (A.flattenKernel (kernel output Bool.false b scale out K N sbk sbn sok son BN BK).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (RealReference.dequantize_kernel b scale out K N sbk sbn sok son BN BK).toAlgKernel) s := by
  simp only [exec,FlatAlloc.flattenKernel,original_body]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DequantizeMatmul.Projection
