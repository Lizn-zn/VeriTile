import bench.tritonbench_g.geglu_tanh_triton.Kernels
import VeriTile.Triton

namespace VeriTile.Bench.TritonBenchG.GegluTanhTriton
open VeriTile.Triton
set_option maxHeartbeats 12000000
set_option maxRecDepth 16000

def sourceMathFwd
    (a b c : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
  ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  a += program_id * $(stride)
  b += program_id * $(stride)
  c += program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0)
  sqrt_2_over_pi = 0.7978845608028654
  a_cubed = a_row * a_row * a_row
  tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
  tanh_result = libdevice.tanh(tanh_arg)
  geglu_a = 0.5 * a_row * (1 + tanh_result)
  c_row = geglu_a * b_row
  tl.store(c + col_offsets, c_row, mask=mask)
}

def commutedMathFwd
    (a b c : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
  ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  a += $(stride) * program_id
  b += $(stride) * program_id
  c += $(stride) * program_id
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0)
  sqrt_2_over_pi = 0.7978845608028654
  a_cubed = a_row * a_row * a_row
  tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
  tanh_result = libdevice.tanh(tanh_arg)
  geglu_a = 0.5 * a_row * (1 + tanh_result)
  c_row = geglu_a * b_row
  tl.store(c + col_offsets, c_row, mask=mask)
}

def mathFwd (v : Bool) (a b c : RegionName) (stride n_cols BLOCK_SIZE : Nat) : Kernel :=
  .mk [a, b] [c]
    (if v then (commutedMathFwd a b c stride n_cols BLOCK_SIZE).toAlgKernel.body
     else (sourceMathFwd a b c stride n_cols BLOCK_SIZE).toAlgKernel.body)

def sourceMathBwd
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
  ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  dc += program_id * $(stride)
  a += program_id * $(stride)
  b += program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
  a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0)
  sqrt_2_over_pi = 0.7978845608028654
  a_cubed = a_row * a_row * a_row
  tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
  tanh_result = libdevice.tanh(tanh_arg)
  geglu_a = 0.5 * a_row * (1 + tanh_result)
  db_row = dc_row * geglu_a
  term1 = 0.5 * (1 + tanh_result)
  tanh_sq = tanh_result * tanh_result
  term2 = 0.5 * a_row * (1 - tanh_sq) *
    (sqrt_2_over_pi * (1 + 0.134145 * a_row * a_row))
  da_row = dc_row * b_row * (term1 + term2)
  tl.store(a + col_offsets, da_row, mask=mask)
  tl.store(b + col_offsets, db_row, mask=mask)
}

def commutedMathBwd
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
  ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  dc += $(stride) * program_id
  a += $(stride) * program_id
  b += $(stride) * program_id
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
  a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0)
  sqrt_2_over_pi = 0.7978845608028654
  a_cubed = a_row * a_row * a_row
  tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
  tanh_result = libdevice.tanh(tanh_arg)
  geglu_a = 0.5 * a_row * (1 + tanh_result)
  db_row = dc_row * geglu_a
  term1 = 0.5 * (1 + tanh_result)
  tanh_sq = tanh_result * tanh_result
  term2 = 0.5 * a_row * (1 - tanh_sq) *
    (sqrt_2_over_pi * (1 + 0.134145 * a_row * a_row))
  da_row = dc_row * b_row * (term1 + term2)
  tl.store(a + col_offsets, da_row, mask=mask)
  tl.store(b + col_offsets, db_row, mask=mask)
}

def mathBwd (v : Bool) (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : Kernel :=
  .mk [dc, a, b] [a, b]
    (if v then (commutedMathBwd dc a b stride n_cols BLOCK_SIZE).toAlgKernel.body
     else (sourceMathBwd dc a b stride n_cols BLOCK_SIZE).toAlgKernel.body)

theorem fwd_projects (storage partner : Storage) (v : Bool)
    (a b c : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    (typedFwd storage partner v a b c stride n_cols BLOCK_SIZE).toAlgorithm? =
      .ok (typedFwd storage partner v a b c stride n_cols BLOCK_SIZE).toAlgKernel := by
  cases storage <;> cases partner <;> cases v <;> rfl

theorem fwd_projection (storage partner : Storage) (v : Bool)
    (a b c : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    (typedFwd storage partner v a b c stride n_cols BLOCK_SIZE).eraseDType.toAlgKernel =
      mathFwd v a b c stride n_cols BLOCK_SIZE := by
  simp only [ComputeKernel.eraseDType, fwd_projects, ComputeKernel.toAlgKernel_fromKernelBody]
  cases storage <;> cases partner <;> cases v
  all_goals simp only [typedFwd, mathFwd, commutedMathFwd,
    sourceMathFwd, Bool.false_eq_true, ↓reduceIte,
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

theorem bwd_projects (storage partner gradient : Storage) (v : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    (typedBwd storage partner gradient v dc a b stride n_cols BLOCK_SIZE).toAlgorithm? =
      .ok (typedBwd storage partner gradient v dc a b stride n_cols BLOCK_SIZE).toAlgKernel := by
  cases storage <;> cases partner <;> cases gradient <;> cases v <;> rfl

theorem bwd_projection (storage partner gradient : Storage) (v : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    (typedBwd storage partner gradient v dc a b stride n_cols BLOCK_SIZE).eraseDType.toAlgKernel =
      mathBwd v dc a b stride n_cols BLOCK_SIZE := by
  simp only [ComputeKernel.eraseDType, bwd_projects, ComputeKernel.toAlgKernel_fromKernelBody]
  cases storage <;> cases partner <;> cases gradient <;> cases v
  all_goals simp only [typedBwd,
    typedBwd_fp32_fp32, typedBwd_fp32_fp16, typedBwd_fp32_bf16, typedBwd_fp32_fp64,
    typedBwd_fp32_f8e4, typedBwd_fp32_f8e5, typedBwd_fp16_fp32, typedBwd_fp16_fp16,
    typedBwd_fp16_bf16, typedBwd_fp16_fp64, typedBwd_fp16_f8e4, typedBwd_fp16_f8e5,
    typedBwd_bf16_fp32, typedBwd_bf16_fp16, typedBwd_bf16_bf16, typedBwd_bf16_fp64,
    typedBwd_bf16_f8e4, typedBwd_bf16_f8e5, typedBwd_fp64_fp32, typedBwd_fp64_fp16,
    typedBwd_fp64_bf16, typedBwd_fp64_fp64, typedBwd_fp64_f8e4, typedBwd_fp64_f8e5,
    typedBwd_f8e4_fp32, typedBwd_f8e4_fp16, typedBwd_f8e4_bf16, typedBwd_f8e4_fp64,
    typedBwd_f8e4_f8e4, typedBwd_f8e4_f8e5, typedBwd_f8e5_fp32, typedBwd_f8e5_fp16,
    typedBwd_f8e5_bf16, typedBwd_f8e5_fp64, typedBwd_f8e5_f8e4, typedBwd_f8e5_f8e5, mathBwd,
    commutedMathBwd,
    sourceMathBwd, Bool.false_eq_true, ↓reduceIte,
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

theorem fwd_supported (storage partner : Storage) (v : Bool)
    (a b c : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    (typedFwd storage partner v a b c stride n_cols BLOCK_SIZE).eraseDType.toAlgorithm? =
      .ok (typedFwd storage partner v a b c stride n_cols BLOCK_SIZE).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, fwd_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

theorem bwd_supported (storage partner gradient : Storage) (v : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    (typedBwd storage partner gradient v dc a b stride n_cols BLOCK_SIZE).eraseDType.toAlgorithm? =
      .ok (typedBwd storage partner gradient v dc a b stride n_cols BLOCK_SIZE).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, bwd_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.GegluTanhTriton
