import bench.tritonbench_g.rmsnorm_implementation.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsnormImplementation
open VeriTile.Triton
set_option maxHeartbeats 12000000
set_option maxRecDepth 16000

def uncastMathKernel
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k : Nat)
    (N_SIZE BLOCK_N_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := ComputeKernel.mk [x_ptr, rms_w_ptr] [out_ptr] (triton {
  pid_batch = tl.program_id(0)
  pid_m = tl.program_id(1)
  offset_m = pid_batch * $(stride_x_batch) + pid_m * $(stride_x_m)
  block_n_size = tl.arange(0, $(BLOCK_N_SIZE))
  var = tl.zeros([$(BLOCK_N_SIZE)], tl.float32)
  for block_n_strart_ptr in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offset_n = block_n_strart_ptr + block_n_size
    x_ptr_mask = offset_n < $(N_SIZE)
    x = tl.load(x_ptr + offset_m + offset_n * $(stride_x_k), mask=x_ptr_mask, other=0.0)
    xf = x
    var += xf * xf
  }
  var = tl.sum(var, axis=0) / $(N_SIZE)
  std = tl.sqrt(var + $(eps))
  for block_n_strart_ptr in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offset_n = block_n_strart_ptr + block_n_size
    x_ptr_mask = offset_n < $(N_SIZE)
    rms_w_offset = tl.load(rms_w_ptr + offset_n * $(stride_rms_w), mask=x_ptr_mask)
    x = tl.load(x_ptr + offset_m + offset_n * $(stride_x_k), mask=x_ptr_mask, other=0.0)
    x_new = x / std
    out = x_new * rms_w_offset
    out_offset = pid_batch * $(stride_out_batch) + pid_m * $(stride_out_m) +
      offset_n * $(stride_out_k)
    tl.store(out_ptr + out_offset, out, mask=x_ptr_mask)
  }
}).surfaceBody

def commutedMathKernel
    (x_ptr rms_w_ptr out_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k : Nat)
    (N_SIZE BLOCK_N_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := ComputeKernel.mk [x_ptr, rms_w_ptr] [out_ptr] (triton {
  pid_batch = tl.program_id(0)
  pid_m = tl.program_id(1)
  offset_m = $(stride_x_batch) * pid_batch + pid_m * $(stride_x_m)
  block_n_size = tl.arange(0, $(BLOCK_N_SIZE))
  var = tl.zeros([$(BLOCK_N_SIZE)], tl.float32)
  for block_n_strart_ptr in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offset_n = block_n_strart_ptr + block_n_size
    x_ptr_mask = offset_n < $(N_SIZE)
    x = tl.load(x_ptr + offset_m + offset_n * $(stride_x_k), mask=x_ptr_mask, other=0.0)
    xf = x
    var += xf * xf
  }
  var = tl.sum(var, axis=0) / $(N_SIZE)
  std = tl.sqrt(var + $(eps))
  for block_n_strart_ptr in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offset_n = block_n_strart_ptr + block_n_size
    x_ptr_mask = offset_n < $(N_SIZE)
    rms_w_offset = tl.load(rms_w_ptr + offset_n * $(stride_rms_w), mask=x_ptr_mask)
    x = tl.load(x_ptr + offset_m + offset_n * $(stride_x_k), mask=x_ptr_mask, other=0.0)
    x_new = x / std
    out = x_new * rms_w_offset
    out_offset = pid_batch * $(stride_out_batch) + pid_m * $(stride_out_m) +
      offset_n * $(stride_out_k)
    tl.store(out_ptr + out_offset, out, mask=x_ptr_mask)
  }
}).surfaceBody

def mathKernel (v : Bool) (X W O : RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : Kernel :=
  if v then (commutedMathKernel X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel
  else (uncastMathKernel X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel

theorem typed_projects (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).toAlgorithm? =
      .ok (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel := by
  cases input <;> cases weight <;> cases output <;> cases v <;> rfl

theorem typed_projection (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgKernel =
      mathKernel v X W O sxb sxm sxk sw sob som sok N B eps := by
  simp only [ComputeKernel.eraseDType, typed_projects, ComputeKernel.toAlgKernel_fromKernelBody]
  cases input <;> cases weight <;> cases output <;> cases v
  all_goals simp only [typedKernel, originalTyped, mathKernel, commutedMathKernel,
    uncastMathKernel, Bool.false_eq_true, ↓reduceIte,
    ComputeKernel.inputs, ComputeKernel.outputs,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType, List.take, List.drop, List.cons_append, List.nil_append]
  all_goals rfl

theorem typed_supported (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgorithm? =
      .ok (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormImplementation
