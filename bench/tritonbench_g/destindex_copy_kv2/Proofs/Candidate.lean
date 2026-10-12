import bench.tritonbench_g.destindex_copy_kv2.Proofs.Projection
import VeriTile.Triton.Semantics.ProgramRewrite
namespace VeriTile.Bench.TritonBenchG.DestindexCopyKv2.Candidate
open VeriTile.Triton TypedSource
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def pointer (swapped : Bool) (K : RegionName) (bs hs ds BH BD : Nat) : Stmt :=
  .assign .ptr [BH,BD] "k_ptrs" (.ptrAdd .scalarL (.ptrBase K)
    (.add .nat (.consR (.consL .nil))
      (.add .nat .scalarL
        (if swapped then .mul .nat .nil (.constNat bs) (.ref .nat [] "cur_index")
         else .mul .nat .nil (.ref .nat [] "cur_index") (.constNat bs))
        (.mul .nat .scalarL (.constNat hs) (.expandDim (shape := [BH]) ⟨1,by simp⟩ (.ref .nat [BH] "offs_h"))))
      (.mul .nat .scalarL (.constNat ds) (.expandDim (shape := [BD]) ⟨0,by simp⟩ (.ref .nat [BD] "offs_d")))))

section
variable (K : RegionName) (Dest_loc : Region .int) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat)

def body (swapped : Bool) : List Stmt :=
  let old := (RealReference.fwd_kernel_destindex_copy_kv K Dest_loc Out stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num BLOCK_DMODEL BLOCK_HEAD).toAlgKernel.body
  old.take 4 ++ [pointer swapped K stride_k_bs stride_k_h stride_k_d BLOCK_HEAD BLOCK_DMODEL] ++ old.drop 5

theorem source_body (input output : Storage) (swapped : Bool) :
    (kernel input output swapped K Dest_loc Out stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num BLOCK_DMODEL BLOCK_HEAD).eraseDType.toAlgKernel.body =
      body K Dest_loc Out stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num BLOCK_DMODEL BLOCK_HEAD swapped := by
  cases input <;> cases output <;> cases swapped
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

theorem flat_pointer_equal (A : FlatAlloc) (r : RegionName) (bs hs ds BH BD : Nat) (s : BlockState) :
    stepStmts (A.flattenStmts [pointer Bool.false r bs hs ds BH BD]) s =
      stepStmts (A.flattenStmts [pointer Bool.true r bs hs ds BH BD]) s := by
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

theorem flat_exec_equal (A : FlatAlloc) (input output : Storage)
    (K : RegionName) (Dest_loc : Region .int) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat) (s : BlockState) :
    exec (A.flattenKernel (kernel input output Bool.false K Dest_loc Out stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num BLOCK_DMODEL BLOCK_HEAD).eraseDType.toAlgKernel) s =
      exec (A.flattenKernel (kernel input output Bool.true K Dest_loc Out stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num BLOCK_DMODEL BLOCK_HEAD).eraseDType.toAlgKernel) s := by
  simp only [exec,FlatAlloc.flattenKernel,source_body,body,FlatAlloc.flattenStmts_append]
  exact ProgramRewrite.context _ _ _ _ (flat_pointer_equal A K stride_k_bs stride_k_h stride_k_d BLOCK_HEAD BLOCK_DMODEL) s

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DestindexCopyKv2.Candidate
