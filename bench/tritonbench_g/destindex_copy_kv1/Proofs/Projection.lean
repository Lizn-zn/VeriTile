import bench.tritonbench_g.destindex_copy_kv1.Kernels
import bench.tritonbench_g.destindex_copy_kv1.Proofs.RealReference
import VeriTile.Triton.Float.Erasure
namespace VeriTile.Bench.TritonBenchG.DestindexCopyKv1.Projection
open VeriTile.Triton TypedSource
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
section
variable (K : RegionName) (Dest_loc : Region .int) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num head_dim BLOCK_DMODEL BLOCK_HEAD : Nat)

theorem source_projects (input output : Storage) (swapped : Bool) :
    (kernel input output swapped K Dest_loc Out stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num head_dim BLOCK_DMODEL BLOCK_HEAD).toAlgorithm? =
      .ok (kernel input output swapped K Dest_loc Out stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num head_dim BLOCK_DMODEL BLOCK_HEAD).toAlgKernel := by
  cases input <;> cases output <;> cases swapped <;> rfl

theorem original_body (input output : Storage) :
    (kernel input output Bool.false K Dest_loc Out stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num head_dim BLOCK_DMODEL BLOCK_HEAD).eraseDType.toAlgKernel.body =
      (RealReference.fwd_kernel_destindex_copy_kv K Dest_loc Out stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num head_dim BLOCK_DMODEL BLOCK_HEAD).toAlgKernel.body := by
  cases input <;> cases output
  all_goals simp only [kernel,RealReference.fwd_kernel_destindex_copy_kv,
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

theorem original_exec (A : FlatAlloc) (input output : Storage) (s : BlockState) :
    exec (A.flattenKernel (kernel input output Bool.false K Dest_loc Out stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num head_dim BLOCK_DMODEL BLOCK_HEAD).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (RealReference.fwd_kernel_destindex_copy_kv K Dest_loc Out stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num head_dim BLOCK_DMODEL BLOCK_HEAD).toAlgKernel) s := by
  simp only [exec,FlatAlloc.flattenKernel,original_body]
end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DestindexCopyKv1.Projection
