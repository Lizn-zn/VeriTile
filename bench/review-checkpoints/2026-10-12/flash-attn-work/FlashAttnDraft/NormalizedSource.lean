import FlashAttnDraft.Kernels
import FlashAttnDraft.Formula
import VeriTile.Triton.Float.Erasure
namespace VeriTile.Bench.TritonBenchG.FlashAttn.NormalizedSource
open VeriTile.Triton TypedSource Formula
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

/-- Unused BS/head/stride arguments are zero here; every argument affecting
source execution is kept in `p`. The public source bridge checks this reduction. -/
def original (Q K V L O : RegionName) (p : Layout) : Kernel :=
  (kernel .fp32 .fp32 Bool.false Q K V L O p.sm_scale
    0 p.stride_q_head p.stride_q_seqlen p.stride_q_dim
    0 0 p.stride_k_seqlen p.stride_k_dim
    0 0 0 p.stride_v_dim
    0 0 p.stride_o_seqlen p.stride_o_dim
    0 0 p.SEQLEN p.BLOCK_M p.DIM p.BLOCK_N p.causal).eraseDType.toAlgKernel

/-- Cache a normalized AST with a kernel-checked equality to the full typed
source. This equality is used only for execution, never as an output formula. -/
def reduced (Q K V L O : RegionName) (p : Layout) :
    {k : Kernel // original Q K V L O p=k} := by
  refine ⟨?_,?_⟩
  swap
  conv_lhs =>
    simp (maxSteps := 3000000) only [original,kernel,kernel_fp32_fp32,
      ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
      Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,
      VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      IntegralDType.eraseDType,ComputeDType.eraseDType,Region.cast,Bool.false_eq_true,
      Except.bind,bind,↓reduceIte,Nat.sub_zero,List.map_cons,List.map_nil]
    simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,
      ComparableDType.eraseDType,IntegralDType.eraseDType,List.map_cons,List.map_nil]

def algorithm (Q K V L O : RegionName) (p : Layout) : Kernel := (reduced Q K V L O p).val

theorem original_eq_algorithm (Q K V L O : RegionName) (p : Layout) :
    original Q K V L O p=algorithm Q K V L O p := (reduced Q K V L O p).property

theorem statement_count (Q K V L O : RegionName) (p : Layout) :
    (algorithm Q K V L O p).body.length=22 := by rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FlashAttn.NormalizedSource
