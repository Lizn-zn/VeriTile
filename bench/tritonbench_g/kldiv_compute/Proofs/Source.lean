import bench.tritonbench_g.kldiv_compute.Kernels
import bench.tritonbench_g.kldiv_compute.Proofs.LegacyReal
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KldivCompute
open VeriTile.Triton TypedSource
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

def commutedMathKernel (X Y O : RegionName) (N B : Nat) : ComputeKernel := triton {
  pid = tl.program_id(0)
  block_start = $(B) * pid
  offsets = block_start + tl.arange(0, $(B))
  mask = offsets < $(N)
  x = tl.load(X + offsets, mask=mask)
  y = tl.load(Y + offsets, mask=mask)
  output = x * tl.log(x / y)
  tl.store(O + offsets, output, mask=mask)
}

def mathKernel (v : Bool) (X Y O : RegionName) (N B : Nat) : Kernel :=
  if v then (commutedMathKernel X Y O N B).toAlgKernel
  else (kldivergence_kernel X Y O N B).toAlgKernel

theorem typed_projects (x y output : Storage) (v : Bool)
    (X Y O : RegionName) (N B : Nat) :
    (kernel x y output v X Y O N B).toAlgorithm? =
      .ok (kernel x y output v X Y O N B).toAlgKernel := by
  cases x <;> cases y <;> cases output <;> cases v <;> rfl

theorem typed_projection (x y output : Storage) (v : Bool)
    (X Y O : RegionName) (N B : Nat) :
    (kernel x y output v X Y O N B).eraseDType.toAlgKernel =
      mathKernel v X Y O N B := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgKernel_fromKernelBody]
  cases x <;> cases y <;> cases output <;> cases v
  all_goals simp only [kernel, kernel_fp32_fp32, kernel_fp32_fp16, kernel_fp32_bf16, kernel_fp32_fp64, kernel_fp32_f8e4, kernel_fp32_f8e5,
    kernel_fp16_fp32, kernel_fp16_fp16, kernel_fp16_bf16, kernel_fp16_fp64, kernel_fp16_f8e4, kernel_fp16_f8e5,
    kernel_bf16_fp32, kernel_bf16_fp16, kernel_bf16_bf16, kernel_bf16_fp64, kernel_bf16_f8e4, kernel_bf16_f8e5,
    kernel_fp64_fp32, kernel_fp64_fp16, kernel_fp64_bf16, kernel_fp64_fp64, kernel_fp64_f8e4, kernel_fp64_f8e5,
    kernel_f8e4_fp32, kernel_f8e4_fp16, kernel_f8e4_bf16, kernel_f8e4_fp64, kernel_f8e4_f8e4, kernel_f8e4_f8e5,
    kernel_f8e5_fp32, kernel_f8e5_fp16, kernel_f8e5_bf16, kernel_f8e5_fp64, kernel_f8e5_f8e4, kernel_f8e5_f8e5,
    mathKernel, commutedMathKernel, kldivergence_kernel,
    Bool.false_eq_true, ↓reduceIte,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals rfl

theorem typed_supported (x y output : Storage) (v : Bool)
    (X Y O : RegionName) (N B : Nat) :
    (kernel x y output v X Y O N B).eraseDType.toAlgorithm? =
      .ok (kernel x y output v X Y O N B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivCompute
