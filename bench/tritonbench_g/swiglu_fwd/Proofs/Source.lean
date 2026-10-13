import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.swiglu_fwd.Kernels

namespace VeriTile.Bench.TritonBenchG.SwigluFwd
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

def commutedMathKernel
    (X Y OUT : RegionName)
    (stride_x_row stride_y_row stride_out_row ncols BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  start_col = tl.program_id(1) * $(BLOCK_N)
  X += $(stride_x_row) * row
  Y += $(stride_y_row) * row
  OUT += $(stride_out_row) * row
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  y = tl.load(Y + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  out = x * tl.sigmoid(x) * y
  tl.store(OUT + cols, out, mask=cols < $(ncols))
}


def mathKernel (v : Bool) (X Y OUT : RegionName) (sx sy so N B : Nat) : Kernel :=
  Kernel.mk [X,Y] [OUT]
    (if v then (commutedMathKernel X Y OUT sx sy so N B).toAlgKernel.body
     else (swiglu_fwd_kernel X Y OUT sx sy so N B).toAlgKernel.body)

theorem typed_projects (input partner output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) :
    (typedKernel input partner output v X Y OUT sx sy so N B).toAlgorithm? =
      .ok (typedKernel input partner output v X Y OUT sx sy so N B).toAlgKernel := by
  cases input <;> cases partner <;> cases output <;> cases v <;> rfl

theorem typed_projection (input partner output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) :
    (typedKernel input partner output v X Y OUT sx sy so N B).eraseDType.toAlgKernel =
      mathKernel v X Y OUT sx sy so N B := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgKernel_fromKernelBody]
  cases input <;> cases partner <;> cases output <;> cases v
  all_goals simp only [typedKernel,
    typed_fp32_fp32, typed_fp32_fp16, typed_fp32_bf16,
    typed_fp32_fp64, typed_fp32_f8e4, typed_fp32_f8e5,
    typed_fp16_fp32, typed_fp16_fp16, typed_fp16_bf16,
    typed_fp16_fp64, typed_fp16_f8e4, typed_fp16_f8e5,
    typed_bf16_fp32, typed_bf16_fp16, typed_bf16_bf16,
    typed_bf16_fp64, typed_bf16_f8e4, typed_bf16_f8e5,
    typed_fp64_fp32, typed_fp64_fp16, typed_fp64_bf16,
    typed_fp64_fp64, typed_fp64_f8e4, typed_fp64_f8e5,
    typed_f8e4_fp32, typed_f8e4_fp16, typed_f8e4_bf16,
    typed_f8e4_fp64, typed_f8e4_f8e4, typed_f8e4_f8e5,
    typed_f8e5_fp32, typed_f8e5_fp16, typed_f8e5_bf16,
    typed_f8e5_fp64, typed_f8e5_f8e4, typed_f8e5_f8e5,
    mathKernel, commutedMathKernel, swiglu_fwd_kernel, Bool.false_eq_true, ↓reduceIte,
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

theorem typed_supported (input partner output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) :
    (typedKernel input partner output v X Y OUT sx sy so N B).eraseDType.toAlgorithm? =
      .ok (typedKernel input partner output v X Y OUT sx sy so N B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluFwd
