import bench.tritonbench_g.swiglu_triton.Kernels
import VeriTile.Triton

namespace VeriTile.Bench.TritonBenchG.SwigluTriton
open VeriTile.Triton
set_option maxHeartbeats 12000000
set_option maxRecDepth 16000

def sourceMathFwd
    (a_ptr b_ptr c_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
  ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  a_ptr += program_id * $(stride)
  b_ptr += program_id * $(stride)
  c_ptr += program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
  c_row = silu(a_row) * b_row
  tl.store(c_ptr + col_offsets, c_row, mask=mask)
}

def commutedMathFwd
    (a_ptr b_ptr c_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
  ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  a_ptr += $(stride) * program_id
  b_ptr += $(stride) * program_id
  c_ptr += $(stride) * program_id
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
  c_row = silu(a_row) * b_row
  tl.store(c_ptr + col_offsets, c_row, mask=mask)
}

def mathFwd (v : Bool) (a_ptr b_ptr c_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : Kernel :=
  .mk [a_ptr, b_ptr] [c_ptr]
    (if v then (commutedMathFwd a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE).toAlgKernel.body
     else (sourceMathFwd a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE).toAlgKernel.body)

def sourceMathBwd
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
  ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  dc_ptr += program_id * $(stride)
  a_ptr += program_id * $(stride)
  b_ptr += program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
  a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
  sig_a = tl.sigmoid(a_row)
  silu_a = a_row * sig_a
  db_row = dc_row * silu_a
  da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row
  tl.store(a_ptr + col_offsets, da_row, mask=mask)
  tl.store(b_ptr + col_offsets, db_row, mask=mask)
}

def commutedMathBwd
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
  ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  dc_ptr += $(stride) * program_id
  a_ptr += $(stride) * program_id
  b_ptr += $(stride) * program_id
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
  a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
  sig_a = tl.sigmoid(a_row)
  silu_a = a_row * sig_a
  db_row = dc_row * silu_a
  da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row
  tl.store(a_ptr + col_offsets, da_row, mask=mask)
  tl.store(b_ptr + col_offsets, db_row, mask=mask)
}

def mathBwd (v : Bool) (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : Kernel :=
  .mk [dc_ptr, a_ptr, b_ptr] [a_ptr, b_ptr]
    (if v then (commutedMathBwd dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE).toAlgKernel.body
     else (sourceMathBwd dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE).toAlgKernel.body)

theorem fwd_projects (storage partner : Storage) (v : Bool)
    (a_ptr b_ptr c_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    (typedFwd storage partner v a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE).toAlgorithm? =
      .ok (typedFwd storage partner v a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE).toAlgKernel := by
  cases storage <;> cases partner <;> cases v <;> rfl

theorem fwd_projection (storage partner : Storage) (v : Bool)
    (a_ptr b_ptr c_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    (typedFwd storage partner v a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE).eraseDType.toAlgKernel =
      mathFwd v a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE := by
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
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    (typedBwd storage partner gradient v dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE).toAlgorithm? =
      .ok (typedBwd storage partner gradient v dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE).toAlgKernel := by
  cases storage <;> cases partner <;> cases gradient <;> cases v <;> rfl

theorem bwd_projection (storage partner gradient : Storage) (v : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    (typedBwd storage partner gradient v dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE).eraseDType.toAlgKernel =
      mathBwd v dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE := by
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
    (a_ptr b_ptr c_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    (typedFwd storage partner v a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE).eraseDType.toAlgorithm? =
      .ok (typedFwd storage partner v a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, fwd_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

theorem bwd_supported (storage partner gradient : Storage) (v : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    (typedBwd storage partner gradient v dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE).eraseDType.toAlgorithm? =
      .ok (typedBwd storage partner gradient v dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, bwd_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluTriton
