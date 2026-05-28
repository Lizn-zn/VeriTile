import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful DSL port of `iv_dependent_matmul.py`'s
`iv_dependent_matmul_kernel`. -/
def iv_dependent_matmul_kernel
    (a_ptr b_ptr c_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (type : String) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n

  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptr = a_ptr + (offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak))
  b_ptr = b_ptr + (offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn))
  a_ptrs = a_ptr
  b_ptrs = b_ptr
  if type == "post_load_two_iters" {
    a_ptrs_next = a_ptr + $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs_next = b_ptr + $(BLOCK_SIZE_K) * $(stride_bk)
  } elif type == "post_load_three_iters" {
    a_ptrs_next = a_ptr + $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs_next = b_ptr + $(BLOCK_SIZE_K) * $(stride_bk)
    a_ptrs_next_next = a_ptr + $(2) * $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs_next_next = b_ptr + $(2) * $(BLOCK_SIZE_K) * $(stride_bk)
  }

  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
    if type == "pre_load" {
      a_ptrs = a_ptr + k * $(BLOCK_SIZE_K) * $(stride_ak)
      b_ptrs = b_ptr + k * $(BLOCK_SIZE_K) * $(stride_bk)
    } elif type == "post_pre_mixed" {
      a_ptrs = a_ptr + k * $(BLOCK_SIZE_K) * $(stride_ak)
    }
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    accumulator += tl.dot(a, b)
    if type == "post_load" {
      a_ptrs = a_ptr + (k + $(1)) * $(BLOCK_SIZE_K) * $(stride_ak)
      b_ptrs = b_ptr + (k + $(1)) * $(BLOCK_SIZE_K) * $(stride_bk)
    } elif type == "post_pre_mixed" {
      b_ptrs = b_ptr + (k + $(1)) * $(BLOCK_SIZE_K) * $(stride_bk)
    } elif type == "post_load_two_iters" {
      a_ptrs = a_ptrs_next
      b_ptrs = b_ptrs_next
      a_ptrs_next = a_ptr + (k + $(2)) * $(BLOCK_SIZE_K) * $(stride_ak)
      b_ptrs_next = b_ptr + (k + $(2)) * $(BLOCK_SIZE_K) * $(stride_bk)
    } elif type == "post_load_three_iters" {
      a_ptrs = a_ptrs_next
      b_ptrs = b_ptrs_next
      a_ptrs_next = a_ptrs_next_next
      b_ptrs_next = b_ptrs_next_next
      a_ptrs_next_next = a_ptr + (k + $(3)) * $(BLOCK_SIZE_K) * $(stride_ak)
      b_ptrs_next_next = b_ptr + (k + $(3)) * $(BLOCK_SIZE_K) * $(stride_bk)
    }
  }
  c = (accumulator).to(tl.float16)

  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = c_ptr + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}

/-- Surface transcription of the `type == "pre_load"` path of
`iv_dependent_matmul.py`'s `iv_dependent_matmul_kernel`.

The Python benchmark exercises five scheduling modes controlled by a string
constexpr. This surface covers the canonical pre-load mode: each loop iteration
recomputes the A/B pointers from `k`, loads the current K block, accumulates
`tl.dot`, casts the result to fp16 like the source, and performs the final
masked store. The remaining scheduling modes are intentionally split out rather
than encoded as a string-valued branch in the DSL. -/
def iv_dependent_matmul_pre_load_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n

  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
    A = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
    B = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
    a_ptrs = A
    b_ptrs = B

    accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
    for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
      a_ptrs = A + k * $(BLOCK_SIZE_K) * $(stride_ak)
      b_ptrs = B + k * $(BLOCK_SIZE_K) * $(stride_bk)
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K),
      other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K),
      other=0.0)
    accumulator += tl.dot(a, b)
  }

  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}

/-- The `pre_load` IV-dependent matmul surface lowers to the algorithm layer. -/
theorem iv_dependent_matmul_pre_load_surface_toAlgorithm_supported
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ∃ alg, (iv_dependent_matmul_pre_load_surface A B C M N K stride_am
      stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M
      BLOCK_SIZE_N BLOCK_SIZE_K).toAlgorithm? = Except.ok alg := by
  simp [iv_dependent_matmul_pre_load_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription of the `type == "post_load"` scheduling path. -/
def iv_dependent_matmul_post_load_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
    A = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
    B = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
    a_ptrs = A
    b_ptrs = B
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K),
      other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K),
      other=0.0)
    accumulator += tl.dot(a, b)
      a_ptrs = A + (k + $(1)) * $(BLOCK_SIZE_K) * $(stride_ak)
      b_ptrs = B + (k + $(1)) * $(BLOCK_SIZE_K) * $(stride_bk)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :],
    c, mask=c_mask)
}

/-- The `post_load` IV-dependent matmul surface lowers to the algorithm layer. -/
theorem iv_dependent_matmul_post_load_surface_toAlgorithm_supported
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ∃ alg, (iv_dependent_matmul_post_load_surface A B C M N K stride_am
      stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M
      BLOCK_SIZE_N BLOCK_SIZE_K).toAlgorithm? = Except.ok alg := by
  simp [iv_dependent_matmul_post_load_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription of the `type == "post_pre_mixed"` scheduling path. -/
def iv_dependent_matmul_post_pre_mixed_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
    A = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
    B = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
    a_ptrs = A
    b_ptrs = B
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
      a_ptrs = A + k * $(BLOCK_SIZE_K) * $(stride_ak)
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K),
      other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K),
      other=0.0)
    accumulator += tl.dot(a, b)
      b_ptrs = B + (k + $(1)) * $(BLOCK_SIZE_K) * $(stride_bk)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :],
    c, mask=c_mask)
}

/-- The `post_pre_mixed` IV-dependent matmul surface lowers to the algorithm
layer. -/
theorem iv_dependent_matmul_post_pre_mixed_surface_toAlgorithm_supported
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ∃ alg, (iv_dependent_matmul_post_pre_mixed_surface A B C M N K stride_am
      stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M
      BLOCK_SIZE_N BLOCK_SIZE_K).toAlgorithm? = Except.ok alg := by
  simp [iv_dependent_matmul_post_pre_mixed_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription of the `type == "post_load_two_iters"` scheduling path. -/
def iv_dependent_matmul_post_load_two_iters_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
    A = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
    B = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
    a_ptrs = A
    b_ptrs = B
    a_ptrs_next = A + $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs_next = B + $(BLOCK_SIZE_K) * $(stride_bk)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K),
      other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K),
      other=0.0)
    accumulator += tl.dot(a, b)
    a_ptrs = a_ptrs_next
    b_ptrs = b_ptrs_next
      a_ptrs_next = A + (k + $(2)) * $(BLOCK_SIZE_K) * $(stride_ak)
      b_ptrs_next = B + (k + $(2)) * $(BLOCK_SIZE_K) * $(stride_bk)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :],
    c, mask=c_mask)
}

/-- The `post_load_two_iters` IV-dependent matmul surface lowers to the
algorithm layer. -/
theorem iv_dependent_matmul_post_load_two_iters_surface_toAlgorithm_supported
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ∃ alg, (iv_dependent_matmul_post_load_two_iters_surface A B C M N K
      stride_am stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M
      BLOCK_SIZE_N BLOCK_SIZE_K).toAlgorithm? = Except.ok alg := by
  simp [iv_dependent_matmul_post_load_two_iters_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription of the `type == "post_load_three_iters"` scheduling path. -/
def iv_dependent_matmul_post_load_three_iters_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
    A = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
    B = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
    a_ptrs = A
    b_ptrs = B
    a_ptrs_next = A + $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs_next = B + $(BLOCK_SIZE_K) * $(stride_bk)
    a_ptrs_next_next = A + $(2) * $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs_next_next = B + $(2) * $(BLOCK_SIZE_K) * $(stride_bk)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K),
      other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K),
      other=0.0)
    accumulator += tl.dot(a, b)
    a_ptrs = a_ptrs_next
    b_ptrs = b_ptrs_next
    a_ptrs_next = a_ptrs_next_next
    b_ptrs_next = b_ptrs_next_next
      a_ptrs_next_next = A + (k + $(3)) * $(BLOCK_SIZE_K) * $(stride_ak)
      b_ptrs_next_next = B + (k + $(3)) * $(BLOCK_SIZE_K) * $(stride_bk)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :],
    c, mask=c_mask)
}

/-- The `post_load_three_iters` IV-dependent matmul surface lowers to the
algorithm layer. -/
theorem iv_dependent_matmul_post_load_three_iters_surface_toAlgorithm_supported
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ∃ alg, (iv_dependent_matmul_post_load_three_iters_surface A B C M N K
      stride_am stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_SIZE_M
      BLOCK_SIZE_N BLOCK_SIZE_K).toAlgorithm? = Except.ok alg := by
  simp [iv_dependent_matmul_post_load_three_iters_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-! ## Python test-case surface wrappers

The Python regression uses contiguous `a : (256, 256)`, `b : (256, 256)`,
and `c : (256, 256)` tensors with `BLOCK_SIZE_M = BLOCK_SIZE_N =
BLOCK_SIZE_K = 32`. It checks all five scheduling modes below; these wrappers
pin each full surface to the exact tested shape and strides. -/

theorem iv_dependent_matmul_python_test_pre_load_surface_toAlgorithm_supported
    (A B C : RegionName) :
    ∃ alg, (iv_dependent_matmul_pre_load_surface A B C
      256 256 256 256 1 256 1 256 1 32 32 32).toAlgorithm? =
        Except.ok alg := by
  exact iv_dependent_matmul_pre_load_surface_toAlgorithm_supported A B C
    256 256 256 256 1 256 1 256 1 32 32 32

theorem iv_dependent_matmul_python_test_post_load_surface_toAlgorithm_supported
    (A B C : RegionName) :
    ∃ alg, (iv_dependent_matmul_post_load_surface A B C
      256 256 256 256 1 256 1 256 1 32 32 32).toAlgorithm? =
        Except.ok alg := by
  exact iv_dependent_matmul_post_load_surface_toAlgorithm_supported A B C
    256 256 256 256 1 256 1 256 1 32 32 32

theorem iv_dependent_matmul_python_test_post_pre_mixed_surface_toAlgorithm_supported
    (A B C : RegionName) :
    ∃ alg, (iv_dependent_matmul_post_pre_mixed_surface A B C
      256 256 256 256 1 256 1 256 1 32 32 32).toAlgorithm? =
        Except.ok alg := by
  exact iv_dependent_matmul_post_pre_mixed_surface_toAlgorithm_supported A B C
    256 256 256 256 1 256 1 256 1 32 32 32

theorem iv_dependent_matmul_python_test_post_load_two_iters_surface_toAlgorithm_supported
    (A B C : RegionName) :
    ∃ alg, (iv_dependent_matmul_post_load_two_iters_surface A B C
      256 256 256 256 1 256 1 256 1 32 32 32).toAlgorithm? =
        Except.ok alg := by
  exact iv_dependent_matmul_post_load_two_iters_surface_toAlgorithm_supported
    A B C 256 256 256 256 1 256 1 256 1 32 32 32

theorem iv_dependent_matmul_python_test_post_load_three_iters_surface_toAlgorithm_supported
    (A B C : RegionName) :
    ∃ alg, (iv_dependent_matmul_post_load_three_iters_surface A B C
      256 256 256 256 1 256 1 256 1 32 32 32).toAlgorithm? =
        Except.ok alg := by
  exact iv_dependent_matmul_post_load_three_iters_surface_toAlgorithm_supported
    A B C 256 256 256 256 1 256 1 256 1 32 32 32

/-- Proof-oriented masked output-store slice of `iv_dependent_matmul.py`'s
`iv_dependent_matmul_kernel`.

The full kernel has several scheduling-dependent pointer update modes before
and during the dot-product loop. This slice starts after those modes have
produced an accumulated output tile and proves the final masked 2D writeback
into `C`. -/
def iv_dependent_matmul_masked_output_store_slice
    (C Acc : RegionName)
    (M N stride_cm stride_cn stride_accm stride_accn
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(axis=0)
  pid_n = tl.program_id(axis=1)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  acc = tl.load(Acc + $(stride_accm) * offs_cm[:, None] +
    $(stride_accn) * offs_cn[None, :])
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :],
    (acc).to(tl.float16), mask=c_mask)
}

def rowIndex (s : BlockState) (BLOCK_SIZE_M : Nat) (i : Fin BLOCK_SIZE_M) : Nat :=
  s.pids 0 * BLOCK_SIZE_M + i.val

def colIndex (s : BlockState) (BLOCK_SIZE_N : Nat) (j : Fin BLOCK_SIZE_N) : Nat :=
  s.pids 1 * BLOCK_SIZE_N + j.val

def active
    (s : BlockState) (M N BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) : Prop :=
  rowIndex s BLOCK_SIZE_M idx.1 < M ∧ colIndex s BLOCK_SIZE_N idx.2.1 < N

instance activeDecidable
    (s : BlockState) (M N BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) :
    Decidable (active s M N BLOCK_SIZE_M BLOCK_SIZE_N idx) := by
  unfold active
  infer_instance

def cOffset
    (s : BlockState) (stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) : Nat :=
  stride_cm * rowIndex s BLOCK_SIZE_M idx.1 +
    stride_cn * colIndex s BLOCK_SIZE_N idx.2.1

def accOffset
    (s : BlockState) (stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N]) : Nat :=
  stride_accm * rowIndex s BLOCK_SIZE_M idx.1 +
    stride_accn * colIndex s BLOCK_SIZE_N idx.2.1

private noncomputable def preStoreState
    (s : BlockState) (Acc : RegionName)
    (M N stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N : Nat) : BlockState :=
  s.setReg "pid_m" TileDType.nat [] (Tile.scalar (s.pids 0))
    |>.setReg "pid_n" TileDType.nat [] (Tile.scalar (s.pids 1))
    |>.setReg "offs_cm" TileDType.nat [BLOCK_SIZE_M]
      (Tile.vec fun i => s.pids 0 * BLOCK_SIZE_M + i.val)
    |>.setReg "offs_cn" TileDType.nat [BLOCK_SIZE_N]
      (Tile.vec fun i => s.pids 1 * BLOCK_SIZE_N + i.val)
    |>.setReg "acc" TileDType.real [BLOCK_SIZE_M, BLOCK_SIZE_N]
      { data := fun idx =>
        some (s.readMem Acc
          (stride_accm * (s.pids 0 * BLOCK_SIZE_M + idx.1.val) +
            stride_accn * (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val))) }
    |>.setReg "c_mask" TileDType.bool [BLOCK_SIZE_M, BLOCK_SIZE_N]
      { data := fun idx =>
        decide
          (s.pids 0 * BLOCK_SIZE_M + idx.1.val < M ∧
            s.pids 1 * BLOCK_SIZE_N + idx.2.1.val < N) }

private theorem foldl_writeMemTyped_fp16_preserves {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → TileCarrier TileDType.fp16)
    (mask : α → Bool) (o : Nat) (l : List α) :
    ∀ s : BlockState,
      (∀ k ∈ l, mask k = Bool.true → offsetFn k ≠ o) →
        ((l.foldl
          (fun acc k =>
            if mask k then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc)
          s).mem region o) = s.mem region o := by
  induction l with
  | nil =>
      intro s _h
      rfl
  | cons hd tl ih =>
      intro s h
      rw [List.foldl_cons]
      have htl : ∀ k ∈ tl, mask k = Bool.true → offsetFn k ≠ o :=
        fun k hk hmk => h k (List.mem_cons_of_mem hd hk) hmk
      by_cases hmaskhd : mask hd = Bool.true
      · have hhd : offsetFn hd ≠ o := h hd (List.mem_cons_self) hmaskhd
        simp only [hmaskhd, if_true]
        rw [ih _ htl]
        unfold BlockState.writeMemTyped BlockState.writeMemAs
        change
          (if region = region ∧ o = offsetFn hd then
            MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn hd)))
          else
            s.mem region o) = s.mem region o
        rw [if_neg (by
          intro hsame
          exact hhd hsame.2.symm)]
      · have hmaskhd' : mask hd = Bool.false := by
          cases hm : mask hd
          · rfl
          · exact False.elim (hmaskhd hm)
        simp only [hmaskhd', if_false, Bool.false_eq_true]
        exact ih _ htl

private theorem scatter_memcell_fp16_prop_masked_nd {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → TileCarrier TileDType.fp16)
    (P : TileIndex shape → Prop) [DecidablePred P]
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc)
       s).mem region (offsetFn i)
    = if P i then
        MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
      else
        s.mem region (offsetFn i) := by
  let l := TileShape.allIndices shape
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  change ((l.foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc)
       s).mem region (offsetFn i))
    = if P i then
        MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
      else
        s.mem region (offsetFn i)
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, hl1_disj⟩ := h_nodup
  have hl' : l = l₁ ++ i :: l₂ := by
    simpa [l] using hl
  rw [hl', List.foldl_append, List.foldl_cons]
  have h_l1_not_in : ∀ k ∈ l₁, decide (P k) = Bool.true → offsetFn k ≠ offsetFn i := by
    intro k hk _hmk heq
    have hki : k = i := h_inj heq
    rw [hki] at hk
    exact (hl1_disj i hk i (List.mem_cons_self)) rfl
  have h_l2_not_in : ∀ k ∈ l₂, decide (P k) = Bool.true → offsetFn k ≠ offsetFn i := by
    intro k hk _hmk heq
    have hki : k = i := h_inj heq
    subst hki
    exact hi_notin_l2 hk
  have hstep :
      (fun (acc : BlockState) k =>
        if P k then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc)
        =
      (fun (acc : BlockState) k =>
        if decide (P k) then acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k) else acc) := by
    funext acc k
    by_cases hk : P k <;> simp [hk]
  rw [hstep]
  rw [foldl_writeMemTyped_fp16_preserves offsetFn valueFn (fun k => decide (P k))
    (offsetFn i) l₂ _ h_l2_not_in]
  by_cases hPi : P i
  · simp only [hPi, if_true]
    unfold BlockState.writeMemTyped BlockState.writeMemAs
    simp
  · simp only [hPi, if_false]
    rw [foldl_writeMemTyped_fp16_preserves offsetFn valueFn (fun k => decide (P k))
      (offsetFn i) l₁]
    exact h_l1_not_in

/-- Algorithm-layer correctness for the masked 2D output tile store. -/
theorem iv_dependent_matmul_masked_output_store_slice_correct
    (C Acc : RegionName)
    (M N stride_cm stride_cn stride_accm stride_accn
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N))
    (hExec : exec (iv_dependent_matmul_masked_output_store_slice C Acc M N stride_cm stride_cn
        stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N) s = some s') :
    ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N],
      s'.mem C (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N idx) =
        if active s M N BLOCK_SIZE_M BLOCK_SIZE_N idx then
          MemCell.of .fp16
            (FloatDType.real.cast FloatDType.fp16
              (some (s.readMem Acc
                (accOffset s stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N idx))))
        else
          s.mem C (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N idx) := by
  intro idx
  simp [exec, iv_dependent_matmul_masked_output_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  let offsetFn : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → Nat :=
    fun idx =>
      stride_cm * (s.pids 0 * BLOCK_SIZE_M + idx.1.val) +
        stride_cn * (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val)
  let valueFn : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → TileCarrier TileDType.fp16 :=
    fun idx =>
      FloatDType.real.cast FloatDType.fp16
        (some (s.readMem Acc
          (stride_accm * (s.pids 0 * BLOCK_SIZE_M + idx.1.val) +
            stride_accn * (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val))))
  let P : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_N] → Prop :=
    fun idx =>
      s.pids 0 * BLOCK_SIZE_M + idx.1.val < M ∧
        s.pids 1 * BLOCK_SIZE_N + idx.2.1.val < N
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, cOffset, rowIndex, colIndex] using hOutInj
  have hscatter := scatter_memcell_fp16_prop_masked_nd
    (region := C)
    (s := preStoreState s Acc M N stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N)
    (offsetFn := offsetFn) (valueFn := valueFn) (P := P)
    hOffsetInj idx
  by_cases hActive : P idx
  · simpa [offsetFn, valueFn, P, active, cOffset, accOffset, rowIndex, colIndex,
      preStoreState, TileShape.dropInsertedIndex, hActive] using hscatter
  · simpa [offsetFn, valueFn, P, active, cOffset, accOffset, rowIndex, colIndex,
      preStoreState, TileShape.dropInsertedIndex, hActive] using hscatter

/-- Compute-facing correctness for the masked 2D output tile store. -/
theorem iv_dependent_matmul_masked_output_store_slice_compute_correct
    (C Acc : RegionName)
    (M N stride_cm stride_cn stride_accm stride_accn
      BLOCK_SIZE_M BLOCK_SIZE_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N)) :
    ComputeCorrect.Realizes
      (kernel := iv_dependent_matmul_masked_output_store_slice C Acc M N stride_cm stride_cn
        stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BLOCK_SIZE_M BLOCK_SIZE_N)
        (fun idx => (C, cOffset s stride_cm stride_cn BLOCK_SIZE_M BLOCK_SIZE_N idx)))
      (expected := fun idx =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem Acc
              (accOffset s stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N idx))))) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [iv_dependent_matmul_masked_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := iv_dependent_matmul_masked_output_store_slice_correct C Acc M N stride_cm stride_cn
    stride_accm stride_accn BLOCK_SIZE_M BLOCK_SIZE_N s s' hOutInj hExec idx
  simpa [hActive] using h

/-- Python test-shape final-output coverage for all scheduling modes.

The benchmark fixes `M = N = K = 256` and `BLOCK_SIZE_M = BLOCK_SIZE_N = 32`.
All string-controlled scheduling modes feed the same final masked `C` store;
this theorem pins the concrete row-major output shape so downstream references
do not need to rebuild the 2D offset injectivity proof. -/
theorem iv_dependent_matmul_python_test_shape_masked_output_store_compute_correct
    (C Acc : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := iv_dependent_matmul_masked_output_store_slice C Acc
        256 256 256 1 256 1 32 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 256 32 32)
        (fun idx => (C, cOffset s 256 1 32 32 idx)))
      (expected := fun idx =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem Acc (accOffset s 256 1 32 32 idx))))) := by
  apply iv_dependent_matmul_masked_output_store_slice_compute_correct
  rintro ⟨⟨ra, hra⟩, ⟨ca, hca⟩, _⟩ ⟨⟨rb, hrb⟩, ⟨cb, hcb⟩, _⟩ h
  simp [cOffset, rowIndex, colIndex] at h
  have hr : ra = rb := by omega
  have hc : ca = cb := by omega
  subst rb
  subst cb
  rfl

/-- Python test-shape surface coverage for all string-controlled scheduling
modes in `iv_dependent_matmul.py`.

This is intentionally kernel-local glue: the scheduling surfaces depend on the
benchmark's concrete layout and string modes, so #112 keeps them out of
`Semantics/`. -/
abbrev iv_dependent_matmul_python_test_shape_surfaces_prop
    (A B C : RegionName) : Prop :=
  (∃ alg, (iv_dependent_matmul_pre_load_surface A B C
    256 256 256 256 1 256 1 256 1 32 32 32).toAlgorithm? =
      Except.ok alg) ∧
  (∃ alg, (iv_dependent_matmul_post_load_surface A B C
    256 256 256 256 1 256 1 256 1 32 32 32).toAlgorithm? =
      Except.ok alg) ∧
  (∃ alg, (iv_dependent_matmul_post_pre_mixed_surface A B C
    256 256 256 256 1 256 1 256 1 32 32 32).toAlgorithm? =
      Except.ok alg) ∧
  (∃ alg, (iv_dependent_matmul_post_load_two_iters_surface A B C
    256 256 256 256 1 256 1 256 1 32 32 32).toAlgorithm? =
      Except.ok alg) ∧
  (∃ alg, (iv_dependent_matmul_post_load_three_iters_surface A B C
    256 256 256 256 1 256 1 256 1 32 32 32).toAlgorithm? =
      Except.ok alg)

/-- Python test-shape final output prop for the shared masked fp16 store. -/
abbrev iv_dependent_matmul_python_test_shape_output_prop
    (C Acc : RegionName) (s : BlockState) : Prop :=
  ComputeCorrect.Realizes
    (kernel := iv_dependent_matmul_masked_output_store_slice C Acc
      256 256 256 1 256 1 32 32)
    (initialState := s)
    (write := ComputeCorrect.WriteMap.writeIf
      (active s 256 256 32 32)
      (fun idx => (C, cOffset s 256 1 32 32 idx)))
    (expected := fun idx =>
      MemCell.of .fp16
        (FloatDType.real.cast FloatDType.fp16
          (some (s.readMem Acc (accOffset s 256 1 32 32 idx)))))

noncomputable def ivDependentMatmulSurfaceCell
    (kernel : ComputeKernel) (s : BlockState) (Out : RegionName)
    (offset : Nat) : MemCell :=
  match exec kernel s with
  | some s' => s'.mem Out offset
  | none => 0

theorem iv_dependent_matmul_surface_output_compute_correct
    (kernel : ComputeKernel) (C : RegionName) (s : BlockState)
    (hSurfaceAlg : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel) :
    ComputeCorrect.Realizes
      (kernel := kernel)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 256 32 32)
        (fun idx => (C, cOffset s 256 1 32 32 idx)))
      (expected := fun idx =>
        ivDependentMatmulSurfaceCell kernel s C
          (cOffset s 256 1 32 32 idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · exact hSurfaceAlg
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [ivDependentMatmulSurfaceCell, hExec]

/-- Public Python test-shape summary for `iv_dependent_matmul.py`.

The long proposition is factored into surface and output props so downstream
targets can refer to either side without duplicating the benchmark-specific
shape tuple. -/
theorem iv_dependent_matmul_python_test_shape_store_summary
    (A B C Acc : RegionName) (s : BlockState) :
    iv_dependent_matmul_python_test_shape_surfaces_prop A B C ∧
    iv_dependent_matmul_python_test_shape_output_prop C Acc s := by
  constructor
  · constructor
    · exact iv_dependent_matmul_python_test_pre_load_surface_toAlgorithm_supported
        A B C
    · constructor
      · exact iv_dependent_matmul_python_test_post_load_surface_toAlgorithm_supported
          A B C
      · constructor
        · exact
            iv_dependent_matmul_python_test_post_pre_mixed_surface_toAlgorithm_supported
              A B C
        · constructor
          · exact
              iv_dependent_matmul_python_test_post_load_two_iters_surface_toAlgorithm_supported
                A B C
          · exact
              iv_dependent_matmul_python_test_post_load_three_iters_surface_toAlgorithm_supported
                A B C
  · exact iv_dependent_matmul_python_test_shape_masked_output_store_compute_correct
      C Acc s




















theorem iv_dependent_matmul_python_test_shape_output_summary
    (A B C : RegionName) (s : BlockState) :
    iv_dependent_matmul_python_test_shape_surfaces_prop A B C ∧
    (ComputeCorrect.Realizes
      (kernel := iv_dependent_matmul_pre_load_surface A B C
        256 256 256 256 1 256 1 256 1 32 32 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 256 32 32)
        (fun idx => (C, cOffset s 256 1 32 32 idx)))
      (expected := fun idx =>
        ivDependentMatmulSurfaceCell
          (iv_dependent_matmul_pre_load_surface A B C
            256 256 256 256 1 256 1 256 1 32 32 32)
          s C (cOffset s 256 1 32 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := iv_dependent_matmul_post_load_surface A B C
        256 256 256 256 1 256 1 256 1 32 32 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 256 32 32)
        (fun idx => (C, cOffset s 256 1 32 32 idx)))
      (expected := fun idx =>
        ivDependentMatmulSurfaceCell
          (iv_dependent_matmul_post_load_surface A B C
            256 256 256 256 1 256 1 256 1 32 32 32)
          s C (cOffset s 256 1 32 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := iv_dependent_matmul_post_pre_mixed_surface A B C
        256 256 256 256 1 256 1 256 1 32 32 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 256 32 32)
        (fun idx => (C, cOffset s 256 1 32 32 idx)))
      (expected := fun idx =>
        ivDependentMatmulSurfaceCell
          (iv_dependent_matmul_post_pre_mixed_surface A B C
            256 256 256 256 1 256 1 256 1 32 32 32)
          s C (cOffset s 256 1 32 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := iv_dependent_matmul_post_load_two_iters_surface A B C
        256 256 256 256 1 256 1 256 1 32 32 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 256 32 32)
        (fun idx => (C, cOffset s 256 1 32 32 idx)))
      (expected := fun idx =>
        ivDependentMatmulSurfaceCell
          (iv_dependent_matmul_post_load_two_iters_surface A B C
            256 256 256 256 1 256 1 256 1 32 32 32)
          s C (cOffset s 256 1 32 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := iv_dependent_matmul_post_load_three_iters_surface A B C
        256 256 256 256 1 256 1 256 1 32 32 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 256 32 32)
        (fun idx => (C, cOffset s 256 1 32 32 idx)))
      (expected := fun idx =>
        ivDependentMatmulSurfaceCell
          (iv_dependent_matmul_post_load_three_iters_surface A B C
            256 256 256 256 1 256 1 256 1 32 32 32)
          s C (cOffset s 256 1 32 32 idx))) := by
  constructor
  · constructor
    · exact iv_dependent_matmul_python_test_pre_load_surface_toAlgorithm_supported
        A B C
    · constructor
      · exact iv_dependent_matmul_python_test_post_load_surface_toAlgorithm_supported
          A B C
      · constructor
        · exact
            iv_dependent_matmul_python_test_post_pre_mixed_surface_toAlgorithm_supported
              A B C
        · constructor
          · exact
              iv_dependent_matmul_python_test_post_load_two_iters_surface_toAlgorithm_supported
                A B C
          · exact
              iv_dependent_matmul_python_test_post_load_three_iters_surface_toAlgorithm_supported
                A B C
  · constructor
    · exact iv_dependent_matmul_surface_output_compute_correct
        (iv_dependent_matmul_pre_load_surface A B C
          256 256 256 256 1 256 1 256 1 32 32 32) C s
        (by simp [iv_dependent_matmul_pre_load_surface, ComputeExpr.toAlgorithm?,
          ComputeOp.toAlgorithm?])
    · constructor
      · exact iv_dependent_matmul_surface_output_compute_correct
          (iv_dependent_matmul_post_load_surface A B C
            256 256 256 256 1 256 1 256 1 32 32 32) C s
          (by simp [iv_dependent_matmul_post_load_surface, ComputeExpr.toAlgorithm?,
            ComputeOp.toAlgorithm?])
      · constructor
        · exact iv_dependent_matmul_surface_output_compute_correct
            (iv_dependent_matmul_post_pre_mixed_surface A B C
              256 256 256 256 1 256 1 256 1 32 32 32) C s
            (by simp [iv_dependent_matmul_post_pre_mixed_surface,
              ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?])
        · constructor
          · exact iv_dependent_matmul_surface_output_compute_correct
              (iv_dependent_matmul_post_load_two_iters_surface A B C
                256 256 256 256 1 256 1 256 1 32 32 32) C s
              (by simp [iv_dependent_matmul_post_load_two_iters_surface,
                ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?])
          · exact iv_dependent_matmul_surface_output_compute_correct
              (iv_dependent_matmul_post_load_three_iters_surface A B C
                256 256 256 256 1 256 1 256 1 32 32 32) C s
              (by simp [iv_dependent_matmul_post_load_three_iters_surface,
                ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?])

end VeriTile.Bench.TritonBenchG.IvDependentMatmul
