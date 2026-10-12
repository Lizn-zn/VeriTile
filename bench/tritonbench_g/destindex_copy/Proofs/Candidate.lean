import bench.tritonbench_g.destindex_copy.Proofs.Projection
import VeriTile.Triton.Semantics.ProgramRewrite
namespace VeriTile.Bench.TritonBenchG.DestindexCopy.Candidate
open VeriTile.Triton TypedSource
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def pointer (swapped : Bool) (KV_nope : RegionName) (bs ds B : Nat) : Stmt :=
  .assign .ptr [1,B] "kv_nope_ptrs" (.ptrAdd .scalarL (.ptrBase KV_nope)
    (.add .nat .scalarL
      (if swapped then .mul .nat .nil (.constNat bs) (.ref .nat [] "cur_index")
       else .mul .nat .nil (.ref .nat [] "cur_index") (.constNat bs))
      (.mul .nat .scalarL (.constNat ds) (.expandDim (shape := [B]) ⟨0,by simp⟩ (.ref .nat [B] "offs_d_nope")))))

section
variable (KV_nope KV_rope : RegionName) (Dest_loc : Region .int) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat)

def body (swapped : Bool) : List Stmt :=
  let old := (RealReference.fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
    stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
    stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
    stride_o_nope_bs stride_o_nope_h stride_o_nope_d
    stride_o_rope_bs stride_o_rope_h stride_o_rope_d
    kv_nope_head_num kv_rope_head_num BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE).toAlgKernel.body
  old.take 4 ++ [pointer swapped KV_nope stride_kv_nope_bs stride_kv_nope_d BLOCK_DMODEL_NOPE] ++ old.drop 5

theorem source_body (nopeType ropeType : Storage) (swapped : Bool) :
    (kernel nopeType ropeType swapped KV_nope KV_rope Dest_loc O_nope O_rope
    stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
    stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
    stride_o_nope_bs stride_o_nope_h stride_o_nope_d
    stride_o_rope_bs stride_o_rope_h stride_o_rope_d
    kv_nope_head_num kv_rope_head_num BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE).eraseDType.toAlgKernel.body =
      body KV_nope KV_rope Dest_loc O_nope O_rope
    stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
    stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
    stride_o_nope_bs stride_o_nope_h stride_o_nope_d
    stride_o_rope_bs stride_o_rope_h stride_o_rope_d
    kv_nope_head_num kv_rope_head_num BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE swapped := by
  cases nopeType <;> cases ropeType <;> cases swapped
  all_goals simp only [kernel,body,pointer,RealReference.fwd_kernel_destindex_copy_kv,
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
end

theorem flat_pointer_equal (A : FlatAlloc) (r : RegionName) (bs ds B : Nat) (s : BlockState) :
    stepStmts (A.flattenStmts [pointer Bool.false r bs ds B]) s =
      stepStmts (A.flattenStmts [pointer Bool.true r bs ds B]) s := by
  have hmul : evalOp (.mul .nat .nil (.ref .nat [] "cur_index") (.constNat bs)) s =
      evalOp (.mul .nat .nil (.constNat bs) (.ref .nat [] "cur_index")) s := by
    simp only [evalOp_mul,evalOp_constNat]
    cases h : evalOp (.ref .nat [] "cur_index") s
    · rfl
    · simp only [bind,Option.bind_some]
      congr 1
      ext i
      exact Nat.mul_comm _ _
  simp only [pointer,Bool.false_eq_true,↓reduceIte,FlatAlloc.flattenStmts,
    FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenOp.eq_def,stepStmts,stepStmt,
    evalOp_ptrAdd,evalOp_add,hmul]

theorem flat_exec_equal (A : FlatAlloc) (nopeType ropeType : Storage)
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .int) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat) (s : BlockState) :
    exec (A.flattenKernel (kernel nopeType ropeType Bool.false KV_nope KV_rope Dest_loc O_nope O_rope
    stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
    stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
    stride_o_nope_bs stride_o_nope_h stride_o_nope_d
    stride_o_rope_bs stride_o_rope_h stride_o_rope_d
    kv_nope_head_num kv_rope_head_num BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (kernel nopeType ropeType Bool.true KV_nope KV_rope Dest_loc O_nope O_rope
    stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
    stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
    stride_o_nope_bs stride_o_nope_h stride_o_nope_d
    stride_o_rope_bs stride_o_rope_h stride_o_rope_d
    kv_nope_head_num kv_rope_head_num BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE).eraseDType.toAlgKernel) s := by
  simp only [exec,FlatAlloc.flattenKernel,source_body,body,FlatAlloc.flattenStmts_append]
  exact ProgramRewrite.context _ _ _ _ (flat_pointer_equal A KV_nope stride_kv_nope_bs stride_kv_nope_d BLOCK_DMODEL_NOPE) s

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DestindexCopy.Candidate
