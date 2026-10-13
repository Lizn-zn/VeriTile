import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.log_softmax.Kernels

namespace VeriTile.Bench.TritonBenchG.LogSoftmax
open VeriTile.Triton
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

def commutedMathForward
    (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  m_offset = $(BLOCK_M) * pid_m + tl.arange(0, $(BLOCK_M))
  n_offset = tl.arange(0, $(BLOCK_N))
  offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  input_ptrs = input_ptr + offset
  inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to(tl.float32)
  row_minus_max = inp - tl.max(inp, axis=1)[:, None]
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=1)[:, None]
  softmax_output = tl.log(numerator / denominator)
  output_ptrs = output_ptr + offset
  tl.store(output_ptrs, softmax_output, mask=mask)
}

def mathForward (v : Bool) (OUT X : RegionName) (M N K R C : Nat) : Kernel :=
  Kernel.mk [X] [OUT]
    (if v then (commutedMathForward OUT X M N K R C).toAlgKernel.eraseDType.body
     else (log_softmax_kernel OUT X M N K R C).toAlgKernel.eraseDType.body)

theorem typedForward_projects (input output : Storage) (v : Bool)
    (OUT X : RegionName) (M N K R C : Nat) :
    (typedForward input output v OUT X M N K R C).toAlgorithm? = .ok (typedForward input output v OUT X M N K R C).toAlgKernel := by
  cases input <;> cases output <;> cases v <;> rfl

theorem typedForward_projection (input output : Storage) (v : Bool)
    (OUT X : RegionName) (M N K R C : Nat) :
    (typedForward input output v OUT X M N K R C).eraseDType.toAlgKernel = mathForward v OUT X M N K R C := by
  simp only [ComputeKernel.eraseDType, typedForward_projects,
    ComputeKernel.toAlgKernel_fromKernelBody]
  cases input <;> cases output <;> cases v
  all_goals simp only [typedForward,
    mathForward, commutedMathForward, log_softmax_kernel, Bool.false_eq_true, ↓reduceIte,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals rfl

theorem typedForward_supported (input output : Storage) (v : Bool)
    (OUT X : RegionName) (M N K R C : Nat) :
    (typedForward input output v OUT X M N K R C).eraseDType.toAlgorithm? = .ok (typedForward input output v OUT X M N K R C).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typedForward_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

def commutedMathBackward
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  m_offset = $(BLOCK_M) * pid_m + tl.arange(0, $(BLOCK_M))
  n_offset = tl.arange(0, $(BLOCK_N))
  offsets = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  out_ptrs = out_ptr + offsets
  out = tl.load(out_ptrs, mask=mask).to(tl.float32)
  out_grad_ptrs = out_grad_ptr + offsets
  out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)
  scale = tl.sum(out_grad, 1)
  in_grad = out_grad - tl.exp((out).to(tl.float32)) * scale[:, None]
  in_grad_ptrs = in_grad_ptr + offsets
  tl.store(in_grad_ptrs, in_grad, mask=mask)
}

def mathBackward (v : Bool) (Y G DX : RegionName) (M N K R C : Nat) : Kernel :=
  Kernel.mk [Y,G] [DX]
    (if v then (commutedMathBackward Y G DX M N K R C).toAlgKernel.eraseDType.body
     else (log_softmax_backward_kernel Y G DX M N K R C).toAlgKernel.eraseDType.body)

theorem typedBackward_projects (input upstream output : Storage) (v : Bool)
    (Y G DX : RegionName) (M N K R C : Nat) :
    (typedBackward input upstream output v Y G DX M N K R C).toAlgorithm? = .ok (typedBackward input upstream output v Y G DX M N K R C).toAlgKernel := by
  cases input <;> cases upstream <;> cases output <;> cases v <;> rfl

theorem typedBackward_projection (input upstream output : Storage) (v : Bool)
    (Y G DX : RegionName) (M N K R C : Nat) :
    (typedBackward input upstream output v Y G DX M N K R C).eraseDType.toAlgKernel = mathBackward v Y G DX M N K R C := by
  simp only [ComputeKernel.eraseDType, typedBackward_projects,
    ComputeKernel.toAlgKernel_fromKernelBody]
  cases input <;> cases upstream <;> cases output <;> cases v
  all_goals simp only [typedBackward,
    typedBackward_fp32_fp32, typedBackward_fp32_fp16, typedBackward_fp32_bf16, typedBackward_fp32_fp64, typedBackward_fp32_f8e4, typedBackward_fp32_f8e5, typedBackward_fp16_fp32, typedBackward_fp16_fp16, typedBackward_fp16_bf16, typedBackward_fp16_fp64, typedBackward_fp16_f8e4, typedBackward_fp16_f8e5, typedBackward_bf16_fp32, typedBackward_bf16_fp16, typedBackward_bf16_bf16, typedBackward_bf16_fp64, typedBackward_bf16_f8e4, typedBackward_bf16_f8e5, typedBackward_fp64_fp32, typedBackward_fp64_fp16, typedBackward_fp64_bf16, typedBackward_fp64_fp64, typedBackward_fp64_f8e4, typedBackward_fp64_f8e5, typedBackward_f8e4_fp32, typedBackward_f8e4_fp16, typedBackward_f8e4_bf16, typedBackward_f8e4_fp64, typedBackward_f8e4_f8e4, typedBackward_f8e4_f8e5, typedBackward_f8e5_fp32, typedBackward_f8e5_fp16, typedBackward_f8e5_bf16, typedBackward_f8e5_fp64, typedBackward_f8e5_f8e4, typedBackward_f8e5_f8e5,
    mathBackward, commutedMathBackward, log_softmax_backward_kernel, Bool.false_eq_true, ↓reduceIte,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals rfl

theorem typedBackward_supported (input upstream output : Storage) (v : Bool)
    (Y G DX : RegionName) (M N K R C : Nat) :
    (typedBackward input upstream output v Y G DX M N K R C).eraseDType.toAlgorithm? = .ok (typedBackward input upstream output v Y G DX M N K R C).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typedBackward_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LogSoftmax
