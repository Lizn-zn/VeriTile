import FifthSphDraft.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.Erasure
open VeriTile.Triton TypedSource
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem forward_body (input output : Storage) (swapped : Bool) (X Y : RegionName) (B N M off stride : Nat) :
    (forward input output swapped X Y B N M off stride).eraseDType.toAlgKernel=
    (forward .fp32 .fp32 swapped X Y B N M off stride).eraseDType.toAlgKernel := by
  cases input <;> cases output <;> cases swapped
  all_goals simp only [forward,forward_fp16_fp16,forward_fp16_bf16,forward_fp16_fp32,forward_bf16_fp16,forward_bf16_bf16,forward_bf16_fp32,forward_fp32_fp16,forward_fp32_bf16,forward_fp32_fp32,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind]
  all_goals rfl

theorem backward_body (input gradient upstream : Storage) (swapped : Bool) (X dX g : RegionName) (B N M off stride : Nat) :
    (backward input gradient upstream swapped X dX g B N M off stride).eraseDType.toAlgKernel=
    (backward .fp32 .fp32 .fp32 swapped X dX g B N M off stride).eraseDType.toAlgKernel := by
  cases input <;> cases gradient <;> cases upstream <;> cases swapped
  all_goals simp only [backward,backward_fp16_fp16_fp16,backward_fp16_fp16_bf16,backward_fp16_fp16_fp32,backward_fp16_bf16_fp16,backward_fp16_bf16_bf16,backward_fp16_bf16_fp32,backward_fp16_fp32_fp16,backward_fp16_fp32_bf16,backward_fp16_fp32_fp32,backward_bf16_fp16_fp16,backward_bf16_fp16_bf16,backward_bf16_fp16_fp32,backward_bf16_bf16_fp16,backward_bf16_bf16_bf16,backward_bf16_bf16_fp32,backward_bf16_fp32_fp16,backward_bf16_fp32_bf16,backward_bf16_fp32_fp32,backward_fp32_fp16_fp16,backward_fp32_fp16_bf16,backward_fp32_fp16_fp32,backward_fp32_bf16_fp16,backward_fp32_bf16_bf16,backward_fp32_bf16_fp32,backward_fp32_fp32_fp16,backward_fp32_fp32_bf16,backward_fp32_fp32_fp32,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.Erasure
