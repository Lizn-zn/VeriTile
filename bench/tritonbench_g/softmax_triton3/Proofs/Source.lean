import bench.tritonbench_g.softmax_triton3.Kernels
import VeriTile.Triton

namespace VeriTile.Bench.TritonBenchG.SoftmaxTriton3
open VeriTile.Triton

/-- Cast-erased mathematical source used only to establish the checked real projection. -/
def math_kernel
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))
  row_minus_max = row - tl.max(row, axis=0)
  if HAS_MASK {
    mask_ptrs = (mask_ptr + (row_idx * $(row_stride))) + col_offsets
    mask = tl.load(mask_ptrs, mask=col_offsets < $(n_cols), other=0)
    row_minus_max = row_minus_max + mask
  }
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  output_row_start_ptr = output_ptr + row_idx * $(row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, softmax_output, mask=col_offsets < $(n_cols))
}

/-- Real-projection companion for the typed reciprocal candidate. -/
def math_reciprocal_kernel
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))
  row_minus_max = row - tl.max(row, axis=0)
  if HAS_MASK {
    mask_ptrs = (mask_ptr + (row_idx * $(row_stride))) + col_offsets
    mask = tl.load(mask_ptrs, mask=col_offsets < $(n_cols), other=0)
    row_minus_max = row_minus_max + mask
  }
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator * (1 / denominator)
  output_row_start_ptr = output_ptr + row_idx * $(row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, softmax_output, mask=col_offsets < $(n_cols))
}

set_option maxRecDepth 16000
set_option maxHeartbeats 3000000

private theorem erased_kernel (k : ComputeKernel)
    (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    k.eraseDType.toAlgKernel = k.toAlgKernel.eraseDType := by
  simp only [ComputeKernel.eraseDType, h, ComputeKernel.toAlgKernel_fromKernelBody]

private theorem typed_projects (inputType maskType : Storage) (v : Bool)
    (out inp mask : RegionName) (stride n B : Nat) (hasMask : Bool) :
    (typedKernel inputType maskType v out inp mask stride n B hasMask).toAlgorithm? =
      .ok (typedKernel inputType maskType v out inp mask stride n B hasMask).toAlgKernel := by
  cases inputType <;> cases maskType <;> cases v <;> cases hasMask <;> rfl

theorem typed_projection (inputType maskType : Storage) (out inp mask : RegionName)
    (stride n B : Nat) (hasMask : Bool) :
    (typedKernel inputType maskType Bool.false out inp mask stride n B hasMask).eraseDType.toAlgKernel =
      (math_kernel out inp mask stride n B hasMask).toAlgKernel := by
  rw [erased_kernel _ (typed_projects ..)]
  cases inputType <;> cases maskType <;> cases hasMask
  all_goals simp only [typedKernel, math_kernel,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType]
  all_goals simp only [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType]
  all_goals simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals rfl

theorem reciprocal_projection (inputType maskType : Storage) (out inp mask : RegionName)
    (stride n B : Nat) (hasMask : Bool) :
    (typedKernel inputType maskType Bool.true out inp mask stride n B hasMask).eraseDType.toAlgKernel =
      (math_reciprocal_kernel out inp mask stride n B hasMask).toAlgKernel := by
  rw [erased_kernel _ (typed_projects ..)]
  cases inputType <;> cases maskType <;> cases hasMask
  all_goals simp only [typedKernel, math_reciprocal_kernel,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType]
  all_goals simp only [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType]
  all_goals simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals rfl

theorem typed_supported (inputType maskType : Storage) (v : Bool) (out inp mask : RegionName)
    (stride n B : Nat) (hasMask : Bool) :
    (typedKernel inputType maskType v out inp mask stride n B hasMask).eraseDType.toAlgorithm? =
      .ok (typedKernel inputType maskType v out inp mask stride n B hasMask).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

/-- Exact output-store AST expected from the source's implicit conversion.
fp32 adds no conversion. Half/bfloat16 each have exactly one fp32-to-storage cast. -/
def expectedStore (storage : Storage) (n B : Nat) : ComputeStmt :=
  let gate : MaskOpt .real [B] := .mask
    (.lt .nat .scalarR (.ref .nat [B] "col_offsets") (.constNat n))
  match storage with
  | .fp32 => .store .real [B] (.ptr (.ref .ptr [B] "output_ptrs"))
      (.compute (.alg .fp32 (.ref .real [B] "softmax_output"))) gate
  | .fp16 => .store .fp16 [B] (.ptr (.ref .ptr [B] "output_ptrs"))
      (.alg (.castFloat .real .fp16 (.ref .real [B] "softmax_output")))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "col_offsets") (.constNat n)))
  | .bf16 => .store .bf16 [B] (.ptr (.ref .ptr [B] "output_ptrs"))
      (.alg (.castFloat .real .bf16 (.ref .real [B] "softmax_output")))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "col_offsets") (.constNat n)))

/-- Precision regression: all nine storage pairs, both source versions and both
mask branches retain the exact destination carrier and a single required conversion. -/
theorem output_store_precision (inputType maskType : Storage) (v : Bool)
    (out inp mask : RegionName) (stride n B : Nat) (hasMask : Bool) :
    (typedKernel inputType maskType v out inp mask stride n B hasMask).surfaceBody.getLast? =
      some (expectedStore inputType n B) := by
  cases inputType <;> cases maskType <;> cases v <;> cases hasMask <;> rfl

/-- The candidate changes no instruction before normalization. -/
theorem normalization_prefix_preserved (inputType maskType : Storage)
    (out inp mask : RegionName) (stride n B : Nat) (hasMask : Bool) :
    (typedKernel inputType maskType Bool.false out inp mask stride n B hasMask).surfaceBody.take 9 =
      (typedKernel inputType maskType Bool.true out inp mask stride n B hasMask).surfaceBody.take 9 := by
  cases inputType <;> cases maskType <;> cases hasMask <;> rfl

/-- Numerator exponentiation remains the fp32 Triton intrinsic for every storage pair. -/
theorem numerator_precision (inputType maskType : Storage) (v : Bool)
    (out inp mask : RegionName) (stride n B : Nat) (hasMask : Bool) :
    (typedKernel inputType maskType v out inp mask stride n B hasMask).surfaceBody[7]? =
      some (.assign .real [B] "numerator"
        (.compute (.alg .fp32 (.exp (.ref .real [B] "row_minus_max"))))) := by
  cases inputType <;> cases maskType <;> cases v <;> cases hasMask <;> rfl

end VeriTile.Bench.TritonBenchG.SoftmaxTriton3
