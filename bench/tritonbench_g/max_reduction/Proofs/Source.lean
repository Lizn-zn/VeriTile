import bench.tritonbench_g.max_reduction.Kernels
import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.MaxReduction
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

def mathStage1 (v : Bool) : ValueSource :=
  if v then fun (inp mid : RegionName) (M BLOCK_SIZE : Nat) => ComputeKernel.mk [inp] [mid] (triton {
  pid = tl.program_id(0)
  offset = $(BLOCK_SIZE) * pid + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(inp_val)
  mid_ptr = mid + pid
  tl.store(mid_ptr, max_val)
}).surfaceBody
  else fun inp mid M B => ComputeKernel.mk [inp] [mid] (max_kernel_1 inp mid M B).surfaceBody

def mathStage2 (v : Bool) : ValueSource :=
  if v then fun (mid out : RegionName) (mid_size BLOCK_MID : Nat) => ComputeKernel.mk [mid] [out] (triton {
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid + offset
  mask = not ($(mid_size) <= offset)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(mid_val)
  tl.store(out, max_val)
}).surfaceBody
  else fun mid out M B => ComputeKernel.mk [mid] [out] (max_kernel_2 mid out M B).surfaceBody

theorem stage1_projects (storage : Storage) (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage1 storage v inp out M B).toAlgorithm? =
      .ok (stage1 storage v inp out M B).toAlgKernel := by
  cases storage <;> cases v <;> rfl

theorem stage1_supported (storage : Storage) (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage1 storage v inp out M B).eraseDType.toAlgorithm? =
      .ok (stage1 storage v inp out M B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, stage1_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

theorem stage1_projection (storage : Storage) (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage1 storage v inp out M B).eraseDType.toAlgKernel =
      (mathStage1 v inp out M B).toAlgKernel := by
  simp only [ComputeKernel.eraseDType, stage1_projects, ComputeKernel.toAlgKernel_fromKernelBody]
  cases storage <;> cases v
  all_goals simp only [stage1, mathStage1, Bool.false_eq_true, ↓reduceIte, max_kernel_1,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals rfl

theorem stage2_projects (storage : Storage) (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage2 storage v inp out M B).toAlgorithm? =
      .ok (stage2 storage v inp out M B).toAlgKernel := by
  cases storage <;> cases v <;> rfl

theorem stage2_supported (storage : Storage) (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage2 storage v inp out M B).eraseDType.toAlgorithm? =
      .ok (stage2 storage v inp out M B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, stage2_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

theorem stage2_projection (storage : Storage) (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage2 storage v inp out M B).eraseDType.toAlgKernel =
      (mathStage2 v inp out M B).toAlgKernel := by
  simp only [ComputeKernel.eraseDType, stage2_projects, ComputeKernel.toAlgKernel_fromKernelBody]
  cases storage <;> cases v
  all_goals simp only [stage2, mathStage2, Bool.false_eq_true, ↓reduceIte, max_kernel_2,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals rfl

abbrev DimensionSource := RegionName → RegionName → RegionName → Nat → Nat → Nat → Nat → Nat → ComputeKernel

def mathDimension (v : Bool) : DimensionSource :=
  fun inp out_value out_index M N K BLOCK_M BLOCK_N =>
  ComputeKernel.mk [inp] [out_value, out_index] (if v then (triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  m_offset = $(BLOCK_M) * pid_m + tl.arange(0, $(BLOCK_M))
  n_offset = tl.arange(0, $(BLOCK_N))
  offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  offset_index = m_offset * $(K) + pid_k
  mask1 = m_offset < $(M)
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  inp_ptrs = inp + offset
  inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  result_value, result_index := tl.max(inp_vals, axis=1, return_indices=True)
  out_value_ptrs = out_value + offset_index
  out_index_ptrs = out_index + offset_index
  tl.store(out_value_ptrs, result_value, mask=mask1)
  tl.store(out_index_ptrs, result_index.to(tl.int64), mask=mask1)
}).surfaceBody
  else (triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  n_offset = tl.arange(0, $(BLOCK_N))
  offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  offset_index = m_offset * $(K) + pid_k
  mask1 = m_offset < $(M)
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  inp_ptrs = inp + offset
  inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  result_value, result_index := tl.max(inp_vals, axis=1, return_indices=True)
  out_value_ptrs = out_value + offset_index
  out_index_ptrs = out_index + offset_index
  tl.store(out_value_ptrs, result_value, mask=mask1)
  tl.store(out_index_ptrs, result_index.to(tl.int64), mask=mask1)
}).surfaceBody)

theorem dimension_projects (storage : Storage) (v : Bool) (inp ov oi : RegionName)
    (M N K BM BN : Nat) :
    (dimensionSource storage v inp ov oi M N K BM BN).toAlgorithm? =
      .ok (dimensionSource storage v inp ov oi M N K BM BN).toAlgKernel := by
  cases storage <;> cases v <;> rfl

theorem dimension_supported (storage : Storage) (v : Bool) (inp ov oi : RegionName)
    (M N K BM BN : Nat) :
    (dimensionSource storage v inp ov oi M N K BM BN).eraseDType.toAlgorithm? =
      .ok (dimensionSource storage v inp ov oi M N K BM BN).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, dimension_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

theorem dimension_projection (storage : Storage) (v : Bool) (inp ov oi : RegionName) (M N K BM BN : Nat) :
    (dimensionSource storage v inp ov oi M N K BM BN).eraseDType.toAlgKernel =
      (mathDimension v inp ov oi M N K BM BN).toAlgKernel := by
  simp only [ComputeKernel.eraseDType, dimension_projects, ComputeKernel.toAlgKernel_fromKernelBody]
  cases storage <;> cases v
  all_goals simp only [dimensionSource, mathDimension, Bool.false_eq_true, ↓reduceIte,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals rfl

/-- Exact fp16 value-reduction path: typed load and padding, fp32 maximum,
then one explicit narrowing store. Candidate address changes cannot alter it. -/
theorem half_stage1_precision (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage1 .fp16 v inp out M B).surfaceBody[4]? = some
      (.assign .fp16 [B] "inp_val" (.alg (.load .fp16
        (.ptr (.ref .ptr [B] "inp_ptrs"))
        (.maskOther (.ref .bool [B] "mask")
          (.broadcast (.castFloat .real .fp16 .negInf) [B]))))) ∧
    (stage1 .fp16 v inp out M B).surfaceBody[5]? = some
      (.assign .real [] "max_val" (.compute (.alg .fp32
        (.reduceMax ⟨0, by simp⟩ Bool.false
          (.castFloat .fp16 .real (.ref .fp16 [B] "inp_val")))))) ∧
    (stage1 .fp16 v inp out M B).surfaceBody[7]? = some
      (.store .fp16 [] (.ptr (.ref .ptr [] "mid_ptr"))
        (.alg (.castFloat .real .fp16 (.ref .real [] "max_val"))) .none) := by
  cases v <;> exact ⟨rfl, rfl, rfl⟩

/-- Bfloat values use the same fp32 max and exactly one output narrowing. -/
theorem bfloat_stage2_precision (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage2 .bf16 v inp out M B).surfaceBody[3]? = some
      (.assign .bf16 [B] "mid_val" (.alg (.load .bf16
        (.ptr (.ref .ptr [B] "mid_ptrs"))
        (.maskOther (.ref .bool [B] "mask")
          (.broadcast (.castFloat .real .bf16 .negInf) [B]))))) ∧
    (stage2 .bf16 v inp out M B).surfaceBody[4]? = some
      (.assign .real [] "max_val" (.compute (.alg .fp32
        (.reduceMax ⟨0, by simp⟩ Bool.false
          (.castFloat .bf16 .real (.ref .bf16 [B] "mid_val")))))) ∧
    (stage2 .bf16 v inp out M B).surfaceBody[5]? = some
      (.store .bf16 [] (.region (Region.cast out) (.constNat 0))
        (.alg (.castFloat .real .bf16 (.ref .real [] "max_val"))) .none) := by
  cases v <;> exact ⟨rfl, rfl, rfl⟩

/-- Exact bf16 value-reduction path: typed load and padding, fp32 maximum,
then one explicit narrowing store. Candidate address changes cannot alter it. -/
theorem bfloat_stage1_precision (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage1 .bf16 v inp out M B).surfaceBody[4]? = some
      (.assign .bf16 [B] "inp_val" (.alg (.load .bf16
        (.ptr (.ref .ptr [B] "inp_ptrs"))
        (.maskOther (.ref .bool [B] "mask")
          (.broadcast (.castFloat .real .bf16 .negInf) [B]))))) ∧
    (stage1 .bf16 v inp out M B).surfaceBody[5]? = some
      (.assign .real [] "max_val" (.compute (.alg .fp32
        (.reduceMax ⟨0, by simp⟩ Bool.false
          (.castFloat .bf16 .real (.ref .bf16 [B] "inp_val")))))) ∧
    (stage1 .bf16 v inp out M B).surfaceBody[7]? = some
      (.store .bf16 [] (.ptr (.ref .ptr [] "mid_ptr"))
        (.alg (.castFloat .real .bf16 (.ref .real [] "max_val"))) .none) := by
  cases v <;> exact ⟨rfl, rfl, rfl⟩

/-- Half values use the same fp32 max and exactly one output narrowing. -/
theorem half_stage2_precision (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage2 .fp16 v inp out M B).surfaceBody[3]? = some
      (.assign .fp16 [B] "mid_val" (.alg (.load .fp16
        (.ptr (.ref .ptr [B] "mid_ptrs"))
        (.maskOther (.ref .bool [B] "mask")
          (.broadcast (.castFloat .real .fp16 .negInf) [B]))))) ∧
    (stage2 .fp16 v inp out M B).surfaceBody[4]? = some
      (.assign .real [] "max_val" (.compute (.alg .fp32
        (.reduceMax ⟨0, by simp⟩ Bool.false
          (.castFloat .fp16 .real (.ref .fp16 [B] "mid_val")))))) ∧
    (stage2 .fp16 v inp out M B).surfaceBody[5]? = some
      (.store .fp16 [] (.region (Region.cast out) (.constNat 0))
        (.alg (.castFloat .real .fp16 (.ref .real [] "max_val"))) .none) := by
  cases v <;> exact ⟨rfl, rfl, rfl⟩

/-- The dimension wrapper stores signed int64 indices; natural index arithmetic
is explicitly converted before the masked store, never substituted as storage. -/
theorem dimension_index_store (storage : Storage) (v : Bool)
    (inp ov oi : RegionName) (M N K BM BN : Nat) :
    (dimensionSource storage v inp ov oi M N K BM BN).surfaceBody[14]? = some
      (.store .int [BM] (.ptr (.ref .ptr [BM] "out_index_ptrs"))
        (.alg (.castNatToInt (.ref .nat [BM] "result_index")))
        (.mask (.ref .bool [BM] "mask1"))) := by
  cases storage <;> cases v <;> rfl

/-- Indexed half maximum retains its channel and uses the paired backend. -/
theorem dimension_half_pair (v : Bool) (inp ov oi : RegionName) (M N K BM BN : Nat) :
    (dimensionSource .fp16 v inp ov oi M N K BM BN).surfaceBody[10]? = some
      (.ifThen (.constBool Bool.true)
        [.assign .fp16 [BM] "result_value" (.alg (.indexedMaxValue .fp16 ⟨1, by simp⟩
          (.ref .fp16 [BM,BN] "inp_vals"))),
         .assign .nat [BM] "result_index" (.alg (.argMaxTyped .fp16 ⟨1, by simp⟩
          (.ref .fp16 [BM,BN] "inp_vals")))]) := by
  cases v <;> rfl

/-- Indexed bfloat maximum promotes both paired observations identically and
narrows the value once at its actual bfloat output store. -/
theorem dimension_bfloat_pair (v : Bool) (inp ov oi : RegionName) (M N K BM BN : Nat) :
    (dimensionSource .bf16 v inp ov oi M N K BM BN).surfaceBody[10]? = some
      (.ifThen (.constBool Bool.true)
        [.assign .real [BM] "result_value" (.compute (.alg .fp32 (.indexedMaxValue .real ⟨1, by simp⟩
          (.castFloat .bf16 .real (.ref .bf16 [BM,BN] "inp_vals"))))),
         .assign .nat [BM] "result_index" (.alg (.argMaxTyped .real ⟨1, by simp⟩
          (.castFloat .bf16 .real (.ref .bf16 [BM,BN] "inp_vals")) (some .fp32)))]) ∧
    (dimensionSource .bf16 v inp ov oi M N K BM BN).surfaceBody[13]? = some
      (.store .bf16 [BM] (.ptr (.ref .ptr [BM] "out_value_ptrs"))
        (.alg (.castFloat .real .bf16 (.ref .real [BM] "result_value")))
        (.mask (.ref .bool [BM] "mask1"))) := by
  cases v <;> exact ⟨rfl,rfl⟩

/-- fp32 indexed values retain their explicit fp32 load and paired profile. -/
theorem dimension_fp32_pair (v : Bool) (inp ov oi : RegionName) (M N K BM BN : Nat) :
    (dimensionSource .fp32 v inp ov oi M N K BM BN).surfaceBody[9]? = some
      (.assign .real [BM,BN] "inp_vals" (.compute (.load .fp32
        (.ptr (.ref .ptr [BM,BN] "inp_ptrs"))
        (.maskOther (.ref .bool [BM,BN] "mask") (.broadcast .negInf [BM,BN]))))) ∧
    (dimensionSource .fp32 v inp ov oi M N K BM BN).surfaceBody[10]? = some
      (.ifThen (.constBool Bool.true)
        [.assign .real [BM] "result_value" (.compute (.alg .fp32 (.indexedMaxValue .real ⟨1, by simp⟩
          (.ref .real [BM,BN] "inp_vals")))),
         .assign .nat [BM] "result_index" (.alg (.argMaxTyped .real ⟨1, by simp⟩
          (.ref .real [BM,BN] "inp_vals") (some .fp32)))]) := by
  cases v <;> exact ⟨rfl,rfl⟩

/-- Half and fp32 dimension outputs need no narrowing or extra conversion. -/
theorem dimension_value_stores (v : Bool) (inp ov oi : RegionName) (M N K BM BN : Nat) :
    (dimensionSource .fp16 v inp ov oi M N K BM BN).surfaceBody[13]? = some
      (.store .fp16 [BM] (.ptr (.ref .ptr [BM] "out_value_ptrs"))
        (.alg (.ref .fp16 [BM] "result_value")) (.mask (.ref .bool [BM] "mask1"))) ∧
    (dimensionSource .fp32 v inp ov oi M N K BM BN).surfaceBody[13]? = some
      (.store .real [BM] (.ptr (.ref .ptr [BM] "out_value_ptrs"))
        (.compute (.alg .fp32 (.ref .real [BM] "result_value")))
        (.mask (.ref .bool [BM] "mask1"))) := by
  cases v <;> exact ⟨rfl,rfl⟩

theorem f8e4_stage1_precision (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage1 .f8e4 v inp out M B).surfaceBody[4]? = some
      (.assign .f8e4 [B] "inp_val" (.alg (.load .f8e4
        (.ptr (.ref .ptr [B] "inp_ptrs"))
        (.maskOther (.ref .bool [B] "mask")
          (.broadcast (.castFloat .real .f8e4 .negInf) [B]))))) ∧
    (stage1 .f8e4 v inp out M B).surfaceBody[5]? = some
      (.assign .real [] "max_val" (.compute (.alg .fp32
        (.reduceMax ⟨0, by simp⟩ Bool.false
          (.castFloat .f8e4 .real (.ref .f8e4 [B] "inp_val")))))) ∧
    (stage1 .f8e4 v inp out M B).surfaceBody[7]? = some
      (.store .f8e4 [] (.ptr (.ref .ptr [] "mid_ptr"))
        (.alg (.castFloat .real .f8e4 (.ref .real [] "max_val"))) .none) := by
  cases v <;> exact ⟨rfl, rfl, rfl⟩

theorem f8e4_stage2_precision (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage2 .f8e4 v inp out M B).surfaceBody[3]? = some
      (.assign .f8e4 [B] "mid_val" (.alg (.load .f8e4
        (.ptr (.ref .ptr [B] "mid_ptrs"))
        (.maskOther (.ref .bool [B] "mask")
          (.broadcast (.castFloat .real .f8e4 .negInf) [B]))))) ∧
    (stage2 .f8e4 v inp out M B).surfaceBody[4]? = some
      (.assign .real [] "max_val" (.compute (.alg .fp32
        (.reduceMax ⟨0, by simp⟩ Bool.false
          (.castFloat .f8e4 .real (.ref .f8e4 [B] "mid_val")))))) ∧
    (stage2 .f8e4 v inp out M B).surfaceBody[5]? = some
      (.store .f8e4 [] (.region (Region.cast out) (.constNat 0))
        (.alg (.castFloat .real .f8e4 (.ref .real [] "max_val"))) .none) := by
  cases v <;> exact ⟨rfl, rfl, rfl⟩

theorem dimension_f8e4_pair (v : Bool) (inp ov oi : RegionName) (M N K BM BN : Nat) :
    (dimensionSource .f8e4 v inp ov oi M N K BM BN).surfaceBody[10]? = some
      (.ifThen (.constBool Bool.true)
        [.assign .f8e4 [BM] "result_value" (.alg (.indexedMaxValue .f8e4 ⟨1, by simp⟩
          (.ref .f8e4 [BM,BN] "inp_vals"))),
         .assign .nat [BM] "result_index" (.alg (.argMaxTyped .f8e4 ⟨1, by simp⟩
          (.ref .f8e4 [BM,BN] "inp_vals")))]) := by
  cases v <;> rfl

theorem dimension_f8e4_store (v : Bool) (inp ov oi : RegionName) (M N K BM BN : Nat) :
    (dimensionSource .f8e4 v inp ov oi M N K BM BN).surfaceBody[13]? = some
      (.store .f8e4 [BM] (.ptr (.ref .ptr [BM] "out_value_ptrs"))
        (.alg (.ref .f8e4 [BM] "result_value"))
        (.mask (.ref .bool [BM] "mask1"))) := by
  cases v <;> rfl

theorem f8e5_stage1_precision (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage1 .f8e5 v inp out M B).surfaceBody[4]? = some
      (.assign .f8e5 [B] "inp_val" (.alg (.load .f8e5
        (.ptr (.ref .ptr [B] "inp_ptrs"))
        (.maskOther (.ref .bool [B] "mask")
          (.broadcast (.castFloat .real .f8e5 .negInf) [B]))))) ∧
    (stage1 .f8e5 v inp out M B).surfaceBody[5]? = some
      (.assign .real [] "max_val" (.compute (.alg .fp32
        (.reduceMax ⟨0, by simp⟩ Bool.false
          (.castFloat .f8e5 .real (.ref .f8e5 [B] "inp_val")))))) ∧
    (stage1 .f8e5 v inp out M B).surfaceBody[7]? = some
      (.store .f8e5 [] (.ptr (.ref .ptr [] "mid_ptr"))
        (.alg (.castFloat .real .f8e5 (.ref .real [] "max_val"))) .none) := by
  cases v <;> exact ⟨rfl, rfl, rfl⟩

theorem f8e5_stage2_precision (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage2 .f8e5 v inp out M B).surfaceBody[3]? = some
      (.assign .f8e5 [B] "mid_val" (.alg (.load .f8e5
        (.ptr (.ref .ptr [B] "mid_ptrs"))
        (.maskOther (.ref .bool [B] "mask")
          (.broadcast (.castFloat .real .f8e5 .negInf) [B]))))) ∧
    (stage2 .f8e5 v inp out M B).surfaceBody[4]? = some
      (.assign .real [] "max_val" (.compute (.alg .fp32
        (.reduceMax ⟨0, by simp⟩ Bool.false
          (.castFloat .f8e5 .real (.ref .f8e5 [B] "mid_val")))))) ∧
    (stage2 .f8e5 v inp out M B).surfaceBody[5]? = some
      (.store .f8e5 [] (.region (Region.cast out) (.constNat 0))
        (.alg (.castFloat .real .f8e5 (.ref .real [] "max_val"))) .none) := by
  cases v <;> exact ⟨rfl, rfl, rfl⟩

theorem dimension_f8e5_pair (v : Bool) (inp ov oi : RegionName) (M N K BM BN : Nat) :
    (dimensionSource .f8e5 v inp ov oi M N K BM BN).surfaceBody[10]? = some
      (.ifThen (.constBool Bool.true)
        [.assign .f8e5 [BM] "result_value" (.alg (.indexedMaxValue .f8e5 ⟨1, by simp⟩
          (.ref .f8e5 [BM,BN] "inp_vals"))),
         .assign .nat [BM] "result_index" (.alg (.argMaxTyped .f8e5 ⟨1, by simp⟩
          (.ref .f8e5 [BM,BN] "inp_vals")))]) := by
  cases v <;> rfl

theorem dimension_f8e5_store (v : Bool) (inp ov oi : RegionName) (M N K BM BN : Nat) :
    (dimensionSource .f8e5 v inp ov oi M N K BM BN).surfaceBody[13]? = some
      (.store .f8e5 [BM] (.ptr (.ref .ptr [BM] "out_value_ptrs"))
        (.alg (.ref .f8e5 [BM] "result_value"))
        (.mask (.ref .bool [BM] "mask1"))) := by
  cases v <;> rfl

theorem dimension_fp16_load (v : Bool) (inp ov oi : RegionName) (M N K BM BN : Nat) :
    (dimensionSource .fp16 v inp ov oi M N K BM BN).surfaceBody[9]? = some
      (.assign .fp16 [BM,BN] "inp_vals" (.alg (.load .fp16
        (.ptr (.ref .ptr [BM,BN] "inp_ptrs"))
        (.maskOther (.ref .bool [BM,BN] "mask")
          (.broadcast (.castFloat .real .fp16 .negInf) [BM,BN]))))) := by
  cases v <;> rfl

theorem dimension_bf16_load (v : Bool) (inp ov oi : RegionName) (M N K BM BN : Nat) :
    (dimensionSource .bf16 v inp ov oi M N K BM BN).surfaceBody[9]? = some
      (.assign .bf16 [BM,BN] "inp_vals" (.alg (.load .bf16
        (.ptr (.ref .ptr [BM,BN] "inp_ptrs"))
        (.maskOther (.ref .bool [BM,BN] "mask")
          (.broadcast (.castFloat .real .bf16 .negInf) [BM,BN]))))) := by
  cases v <;> rfl

theorem dimension_f8e4_load (v : Bool) (inp ov oi : RegionName) (M N K BM BN : Nat) :
    (dimensionSource .f8e4 v inp ov oi M N K BM BN).surfaceBody[9]? = some
      (.assign .f8e4 [BM,BN] "inp_vals" (.alg (.load .f8e4
        (.ptr (.ref .ptr [BM,BN] "inp_ptrs"))
        (.maskOther (.ref .bool [BM,BN] "mask")
          (.broadcast (.castFloat .real .f8e4 .negInf) [BM,BN]))))) := by
  cases v <;> rfl

theorem dimension_f8e5_load (v : Bool) (inp ov oi : RegionName) (M N K BM BN : Nat) :
    (dimensionSource .f8e5 v inp ov oi M N K BM BN).surfaceBody[9]? = some
      (.assign .f8e5 [BM,BN] "inp_vals" (.alg (.load .f8e5
        (.ptr (.ref .ptr [BM,BN] "inp_ptrs"))
        (.maskOther (.ref .bool [BM,BN] "mask")
          (.broadcast (.castFloat .real .f8e5 .negInf) [BM,BN]))))) := by
  cases v <;> rfl

/-- Explicit fp64 storage keeps raw load, reduction, and store at fp64 precision. -/
theorem fp64_stage1_precision (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage1 .fp64 v inp out M B).surfaceBody[4]? = some
      (.assign .real [B] "inp_val" (.compute (.alg .fp64 (.load .real
        (.ptr (.ref .ptr [B] "inp_ptrs"))
        (.maskOther (.ref .bool [B] "mask") (.broadcast .negInf [B])))))) ∧
    (stage1 .fp64 v inp out M B).surfaceBody[5]? = some
      (.assign .real [] "max_val" (.compute (.alg .fp64
        (.reduceMax ⟨0, by simp⟩ Bool.false (.ref .real [B] "inp_val"))))) ∧
    (stage1 .fp64 v inp out M B).surfaceBody[7]? = some
      (.store .real [] (.ptr (.ref .ptr [] "mid_ptr"))
        (.compute (.alg .fp64 (.ref .real [] "max_val"))) .none) := by
  cases v <;> exact ⟨rfl, rfl, rfl⟩

/-- Explicit fp64 storage keeps raw load, reduction, and store at fp64 precision. -/
theorem fp64_stage2_precision (v : Bool) (inp out : RegionName) (M B : Nat) :
    (stage2 .fp64 v inp out M B).surfaceBody[3]? = some
      (.assign .real [B] "mid_val" (.compute (.alg .fp64 (.load .real
        (.ptr (.ref .ptr [B] "mid_ptrs"))
        (.maskOther (.ref .bool [B] "mask") (.broadcast .negInf [B])))))) ∧
    (stage2 .fp64 v inp out M B).surfaceBody[4]? = some
      (.assign .real [] "max_val" (.compute (.alg .fp64
        (.reduceMax ⟨0, by simp⟩ Bool.false (.ref .real [B] "mid_val"))))) ∧
    (stage2 .fp64 v inp out M B).surfaceBody[5]? = some
      (.store .real [] (.region (Region.cast out) (.constNat 0))
        (.compute (.alg .fp64 (.ref .real [] "max_val"))) .none) := by
  cases v <;> exact ⟨rfl, rfl, rfl⟩

/-- Both indexed observations use fp64 on the actual raw-loaded tile. -/
theorem dimension_fp64_precision (v : Bool) (inp ov oi : RegionName) (M N K BM BN : Nat) :
    (dimensionSource .fp64 v inp ov oi M N K BM BN).surfaceBody[9]? = some
      (.assign .real [BM,BN] "inp_vals" (.compute (.alg .fp64 (.load .real
        (.ptr (.ref .ptr [BM,BN] "inp_ptrs"))
        (.maskOther (.ref .bool [BM,BN] "mask") (.broadcast .negInf [BM,BN])))))) ∧
    (dimensionSource .fp64 v inp ov oi M N K BM BN).surfaceBody[10]? = some
      (.ifThen (.constBool Bool.true)
        [.assign .real [BM] "result_value" (.compute (.alg .fp64
          (.indexedMaxValue .real ⟨1, by simp⟩ (.ref .real [BM,BN] "inp_vals")))),
         .assign .nat [BM] "result_index" (.alg
          (.argMaxTyped .real ⟨1, by simp⟩ (.ref .real [BM,BN] "inp_vals") (some .fp64)))]) ∧
    (dimensionSource .fp64 v inp ov oi M N K BM BN).surfaceBody[13]? = some
      (.store .real [BM] (.ptr (.ref .ptr [BM] "out_value_ptrs"))
        (.compute (.alg .fp64 (.ref .real [BM] "result_value")))
        (.mask (.ref .bool [BM] "mask1"))) := by
  cases v <;> exact ⟨rfl, rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MaxReduction
