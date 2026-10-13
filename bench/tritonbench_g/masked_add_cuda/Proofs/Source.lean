import bench.tritonbench_g.masked_add_cuda.Kernels

namespace VeriTile.Bench.TritonBenchG.MaskedAddCuda
open VeriTile.Triton

/-- Cast-erased mathematical companion, retaining the signed integer mask load. -/
def integer_math_kernel (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) : ComputeKernel := triton {
  tl.region p_mask_ptr = tl.int64
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
  mask = mask & ~p_mask
  p = tl.load(p_ptr + offsets, mask=mask)
  grad = tl.load(grad_ptr + offsets, mask=mask)
  grad += p * $(alpha)
  tl.store(grad_ptr + offsets, grad, mask=mask)
}

/-- Cast-erased mathematical companion, retaining the signed integer mask load. -/
def integer_commuted_math_kernel (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) : ComputeKernel := triton {
  tl.region p_mask_ptr = tl.int64
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
  mask = mask & ~p_mask
  p = tl.load(p_ptr + offsets, mask=mask)
  grad = tl.load(grad_ptr + offsets, mask=mask)
  grad = p * $(alpha) + grad
  tl.store(grad_ptr + offsets, grad, mask=mask)
}

/-- Signed source read and comparison to exactly zero, never a Bool-channel load. -/
def integerMaskRead : ComputeStmt → Bool
  | .assign .bool _ _ (.alg (.ne .int _ (.load .int _ _ _) (.broadcast (.constInt 0) _))) => Bool.true
  | _ => Bool.false

example (storage : MaskStorage) (v : Bool) (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) :
    (integer_kernel storage v grad p mask n alpha B).surfaceBody[4]?.map integerMaskRead =
      some Bool.true := by
  cases storage <;> cases v <;> rfl

/-- Inspect the precision wrapper of the final arithmetic assignment. -/
def arithmeticPrecision (precision : ComputeDType) : ComputeStmt → Bool
  | .assign _ _ _ (.compute (.alg actual _)) => decide (actual = precision)
  | _ => Bool.false

example (storage : MaskStorage) (v : Bool) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) :
    (integer_kernel storage v g p m n a B).surfaceBody[8]?.map
      (arithmeticPrecision .fp32) = some Bool.true := by
  cases storage <;> cases v <;> rfl

example (storage : MaskStorage) (v : Bool) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) :
    (integer_fp64_kernel storage v g p m n a B).surfaceBody[4]?.map integerMaskRead =
      some Bool.true := by
  cases storage <;> cases v <;> rfl

/-- The candidate keeps every original operation before the final addition. -/
example (storage : MaskStorage) (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) :
    (integer_kernel storage Bool.false g p m n a B).surfaceBody.take 8 =
      (integer_kernel storage Bool.true g p m n a B).surfaceBody.take 8 := by
  cases storage <;> rfl

private theorem typed_projects (storage : MaskStorage) (v : Bool)
    (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) :
    (integer_kernel storage v g p m n a B).toAlgorithm? =
      .ok (integer_kernel storage v g p m n a B).toAlgKernel := by
  cases storage <;> cases v <;> rfl

private theorem erased_kernel (k : ComputeKernel)
    (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    k.eraseDType.toAlgKernel = k.toAlgKernel.eraseDType := by
  simp only [ComputeKernel.eraseDType, h, ComputeKernel.toAlgKernel_fromKernelBody]

theorem integer_projection (storage : MaskStorage) (v : Bool)
    (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) :
    (integer_kernel storage v g p m n a B).eraseDType.toAlgKernel =
      (if v then integer_commuted_math_kernel g p m n a B else
        integer_math_kernel g p m n a B).toAlgKernel := by
  rw [erased_kernel _ (typed_projects ..)]
  cases storage <;> cases v <;> simp only [Bool.false_eq_true, if_true, if_false]
  all_goals simp only [integer_kernel, integer_math_kernel, integer_commuted_math_kernel,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType]
  all_goals simp only [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    ComputeDType.eraseDType]
  all_goals rfl

theorem integer_supported (storage : MaskStorage) (v : Bool)
    (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) :
    (integer_kernel storage v g p m n a B).eraseDType.toAlgorithm? =
      .ok (integer_kernel storage v g p m n a B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

private theorem fp64_typed_projects (storage : MaskStorage) (v : Bool)
    (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) :
    (integer_fp64_kernel storage v g p m n a B).toAlgorithm? =
      .ok (integer_fp64_kernel storage v g p m n a B).toAlgKernel := by
  cases storage <;> cases v <;> rfl

private theorem fp64_erased_kernel (k : ComputeKernel)
    (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    k.eraseDType.toAlgKernel = k.toAlgKernel.eraseDType := by
  simp only [ComputeKernel.eraseDType, h, ComputeKernel.toAlgKernel_fromKernelBody]

theorem fp64_integer_projection (storage : MaskStorage) (v : Bool)
    (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) :
    (integer_fp64_kernel storage v g p m n a B).eraseDType.toAlgKernel =
      (if v then integer_commuted_math_kernel g p m n a B else
        integer_math_kernel g p m n a B).toAlgKernel := by
  rw [fp64_erased_kernel _ (fp64_typed_projects ..)]
  cases storage <;> cases v <;> simp only [Bool.false_eq_true, if_true, if_false]
  all_goals simp only [integer_fp64_kernel, integer_math_kernel, integer_commuted_math_kernel,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType]
  all_goals simp only [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    ComputeDType.eraseDType]
  all_goals rfl

theorem fp64_integer_supported (storage : MaskStorage) (v : Bool)
    (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) :
    (integer_fp64_kernel storage v g p m n a B).eraseDType.toAlgorithm? =
      .ok (integer_fp64_kernel storage v g p m n a B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, fp64_typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

/-- Exact store carrier and single implicit-to-explicit output conversion. -/
def expectedTypedStore (storage : FloatStorage) (g : RegionName) (B : Nat) : ComputeStmt :=
  match storage with
  | .fp32 => .store .real [B] (.region g (.ref .nat [B] "offsets"))
      (.compute (.alg .fp32 (.ref .real [B] "grad"))) (.mask (.ref .bool [B] "mask"))
  | .fp16 => .store .fp16 [B] (.region (.cast g) (.ref .nat [B] "offsets"))
      (.alg (.castFloat .real .fp16 (.ref .real [B] "grad"))) (.mask (.ref .bool [B] "mask"))
  | .bf16 => .store .bf16 [B] (.region (.cast g) (.ref .nat [B] "offsets"))
      (.alg (.castFloat .real .bf16 (.ref .real [B] "grad"))) (.mask (.ref .bool [B] "mask"))

theorem typed_store_precision (gt pt : FloatStorage) (mt : MaskStorage) (v : Bool)
    (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) :
    (typed_integer_kernel gt pt mt v g p m n a B).surfaceBody.getLast? =
      some (expectedTypedStore gt g B) := by
  cases gt <;> cases pt <;> cases mt <;> cases v <;> rfl

theorem typed_mask_carrier (gt pt : FloatStorage) (mt : MaskStorage) (v : Bool)
    (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) :
    (typed_integer_kernel gt pt mt v g p m n a B).surfaceBody[4]?.map integerMaskRead =
      some Bool.true := by
  cases gt <;> cases pt <;> cases mt <;> cases v <;> rfl

theorem typed_arithmetic_precision (gt pt : FloatStorage) (mt : MaskStorage) (v : Bool)
    (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) :
    (typed_integer_kernel gt pt mt v g p m n a B).surfaceBody[8]?.map
      (arithmeticPrecision .fp32) = some Bool.true := by
  cases gt <;> cases pt <;> cases mt <;> cases v <;> rfl

private theorem full_typed_projects (gradType pType : FloatStorage) (storage : MaskStorage) (v : Bool)
    (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) :
    (typed_integer_kernel gradType pType storage v g p m n a B).toAlgorithm? =
      .ok (typed_integer_kernel gradType pType storage v g p m n a B).toAlgKernel := by
  cases gradType <;> cases pType <;> cases storage <;> cases v <;> rfl

private theorem full_erased_kernel (k : ComputeKernel)
    (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    k.eraseDType.toAlgKernel = k.toAlgKernel.eraseDType := by
  simp only [ComputeKernel.eraseDType, h, ComputeKernel.toAlgKernel_fromKernelBody]

theorem full_integer_projection (gradType pType : FloatStorage) (storage : MaskStorage) (v : Bool)
    (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) :
    (typed_integer_kernel gradType pType storage v g p m n a B).eraseDType.toAlgKernel =
      (if v then integer_commuted_math_kernel g p m n a B else
        integer_math_kernel g p m n a B).toAlgKernel := by
  rw [full_erased_kernel _ (full_typed_projects ..)]
  cases gradType <;> cases pType <;> cases storage <;> cases v <;> simp only [Bool.false_eq_true, if_true, if_false]
  all_goals simp only [typed_integer_kernel, integer_math_kernel, integer_commuted_math_kernel,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType]
  all_goals simp only [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    ComputeDType.eraseDType]
  all_goals rfl

theorem full_integer_supported (gradType pType : FloatStorage) (storage : MaskStorage) (v : Bool)
    (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) :
    (typed_integer_kernel gradType pType storage v g p m n a B).eraseDType.toAlgorithm? =
      .ok (typed_integer_kernel gradType pType storage v g p m n a B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, full_typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

end VeriTile.Bench.TritonBenchG.MaskedAddCuda
