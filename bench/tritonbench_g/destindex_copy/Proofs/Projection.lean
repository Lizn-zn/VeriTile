import bench.tritonbench_g.destindex_copy.Kernels
import bench.tritonbench_g.destindex_copy.Proofs.RealReference
import VeriTile.Triton.Float.Erasure
namespace VeriTile.Bench.TritonBenchG.DestindexCopy.Projection
open VeriTile.Triton TypedSource
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
section
variable (KV_nope KV_rope : RegionName) (Dest_loc : Region .int) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat)

theorem source_projects (nopeType ropeType : Storage) (swapped : Bool) :
    (kernel nopeType ropeType swapped KV_nope KV_rope Dest_loc O_nope O_rope
    stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
    stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
    stride_o_nope_bs stride_o_nope_h stride_o_nope_d
    stride_o_rope_bs stride_o_rope_h stride_o_rope_d
    kv_nope_head_num kv_rope_head_num BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE).toAlgorithm? =
      .ok (kernel nopeType ropeType swapped KV_nope KV_rope Dest_loc O_nope O_rope
    stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
    stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
    stride_o_nope_bs stride_o_nope_h stride_o_nope_d
    stride_o_rope_bs stride_o_rope_h stride_o_rope_d
    kv_nope_head_num kv_rope_head_num BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE).toAlgKernel := by
  cases nopeType <;> cases ropeType <;> cases swapped <;> rfl

theorem original_body (nopeType ropeType : Storage) :
    (kernel nopeType ropeType Bool.false KV_nope KV_rope Dest_loc O_nope O_rope
    stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
    stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
    stride_o_nope_bs stride_o_nope_h stride_o_nope_d
    stride_o_rope_bs stride_o_rope_h stride_o_rope_d
    kv_nope_head_num kv_rope_head_num BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE).eraseDType.toAlgKernel.body =
      (RealReference.fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
    stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
    stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
    stride_o_nope_bs stride_o_nope_h stride_o_nope_d
    stride_o_rope_bs stride_o_rope_h stride_o_rope_d
    kv_nope_head_num kv_rope_head_num BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE).toAlgKernel.body := by
  cases nopeType <;> cases ropeType
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

theorem original_exec (A : FlatAlloc) (nopeType ropeType : Storage) (s : BlockState) :
    exec (A.flattenKernel (kernel nopeType ropeType Bool.false KV_nope KV_rope Dest_loc O_nope O_rope
    stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
    stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
    stride_o_nope_bs stride_o_nope_h stride_o_nope_d
    stride_o_rope_bs stride_o_rope_h stride_o_rope_d
    kv_nope_head_num kv_rope_head_num BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (RealReference.fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
    stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
    stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
    stride_o_nope_bs stride_o_nope_h stride_o_nope_d
    stride_o_rope_bs stride_o_rope_h stride_o_rope_d
    kv_nope_head_num kv_rope_head_num BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE).toAlgKernel) s := by
  simp only [exec,FlatAlloc.flattenKernel,original_body]
end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DestindexCopy.Projection
