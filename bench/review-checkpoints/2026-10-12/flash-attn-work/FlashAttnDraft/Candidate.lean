import FlashAttnDraft.Erasure
import FlashAttnDraft.NormalizedSource
import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Triton.Memory.FlattenTolerant
namespace VeriTile.Bench.TritonBenchG.FlashAttn.Candidate
open VeriTile.Triton TypedSource Formula
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def source (swapped : Bool) (Q K V L O : RegionName) (p : Layout) : Kernel :=
  (kernel .fp32 .fp32 swapped Q K V L O p.sm_scale
    0 p.stride_q_head p.stride_q_seqlen p.stride_q_dim
    0 0 p.stride_k_seqlen p.stride_k_dim 0 0 0 p.stride_v_dim
    0 0 p.stride_o_seqlen p.stride_o_dim 0 0 p.SEQLEN p.BLOCK_M p.DIM p.BLOCK_N p.causal).eraseDType.toAlgKernel

def indexStatement (swapped : Bool) (stride_q_head : Nat) : Stmt :=
  .assign .nat [] "qkv_base_offset"
    (if swapped then
      .mul .nat .nil (.constNat stride_q_head) (.ref .nat [] "off_bs_head")
    else .mul .nat .nil (.ref .nat [] "off_bs_head") (.constNat stride_q_head))

theorem index_flat (A : FlatAlloc) (stride_q_head : Nat) (u : BlockState) :
    stepStmts (A.flattenStmts [indexStatement Bool.false stride_q_head]) u =
    stepStmts (A.flattenStmts [indexStatement Bool.true stride_q_head]) u := by
  cases hh : u.regs .nat [] "off_bs_head"
  all_goals simp [indexStatement,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
    stepStmts,stepStmt,evalOp.eq_def,hh,Tile.bop,NumericDType.mul,Nat.mul_comm]

theorem decomposition (swapped : Bool) (Q K V L O : RegionName) (p : Layout) :
    (source swapped Q K V L O p).body =
      (source Bool.false Q K V L O p).body.take 2 ++
      [indexStatement swapped p.stride_q_head] ++
      (source Bool.false Q K V L O p).body.drop 3 := by
  cases swapped
  all_goals simp only [source,indexStatement,kernel,kernel_fp32_fp32,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,VeriTile.Triton.eraseDType,
    NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType,ComputeDType.eraseDType,
    Bool.false_eq_true,↓reduceIte,Except.bind,bind,List.map_cons,List.map_nil]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,List.map_cons,List.map_nil]
  all_goals rfl

theorem flat_exec (A : FlatAlloc) (Q K V L O : RegionName) (p : Layout) (u : BlockState) :
    exec (A.flattenKernel (source Bool.false Q K V L O p)) u =
    exec (A.flattenKernel (source Bool.true Q K V L O p)) u := by
  unfold exec FlatAlloc.flattenKernel
  rw [decomposition Bool.false,decomposition Bool.true]
  simp only [FlatAlloc.flattenStmts_append,ProgramRewrite.append,index_flat]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FlashAttn.Candidate
