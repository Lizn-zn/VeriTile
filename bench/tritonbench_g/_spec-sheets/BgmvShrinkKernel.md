# Spec sheet — `bench/tritonbench_g/bgmv_shrink_kernel/BgmvShrinkKernel.lean`

**Python source:** `bench/tritonbench_g/bgmv_shrink_kernel/bgmv_shrink_kernel.py`

## Public theorem: `bgmv_shrink_kernel_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general** correctness summary for `bgmv_shrink_kernel.py`'s
`_bgmv_shrink_kernel`, against the **genuine closed form**

```
shrinkSpec n = scaling · Σ_{c<⌈K/(BLOCK_K·SPLIT_K)⌉} Σ_{e<BLOCK_K}
  [k(c,e) < K] · input[cur_batch·xm_stride + k(c,e)]
             · loraA[l0_stride·lora_index + n·lora_k_stride + k(c,e)·lora_n_stride]
```

(`k(c,e) = c·BLOCK_K·SPLIT_K + pid_sk·BLOCK_K + e`, this program's rank
slice) — a pure function of INPUT memory, never a read-back of the kernel's
own output — for arbitrary `N`, `K`, `BLOCK_N`, `BLOCK_K`, `SPLIT_K`, strides,
`scaling`, program ids, and data-dependent `lora_index`. It packages:

* all three surfaces lower to the algorithm layer (the faithful guarded
  surface with both constexpr tail branches, and the two per-branch proof
  surfaces);
* the sentinel path: `lora_indices[cur_batch] = -1` (Python's early `return`)
  leaves memory untouched;
* the `SPLIT_K = 1` branch (`tl.store`): every active lane `n < N` of
  `out[cur_batch]` holds `shrinkSpec n`;
* the `SPLIT_K > 1` branch (`tl.atomic_add`): every active lane holds
  `out-before + shrinkSpec n` — the per-program atomic accumulation
  obligation; the cross-program sum over `pid_sk` (and the sentinel skip
  itself for the proof surfaces) is the host launch's trusted composition.

Honest side-conditions: `0 < BLOCK_K`, `0 < SPLIT_K` (a launched constexpr
tile is nonempty — also the K-loop stride), and `0 < cn_stride` (output-lane
footprint injectivity; torch strides of a non-degenerate output are ≥ 1). -/
```
</details>

**Statement:**
```lean
theorem bgmv_shrink_kernel_output_summary_general
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices_int : Region .int) (lora_indices : Region .nat)
    (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) (SPLIT_K_ONE : Bool)
    (s : BlockState)
    (hBK : 0 < BLOCK_K) (hSK : 0 < SPLIT_K) (hcn : 0 < cn_stride) :
    -- (1) the faithful guarded surface (sentinel guard + both constexpr tail
    --     branches) lowers to the algorithm layer
    (∃ alg, (bgmv_shrink_surface input_ptr lora_ptr out_ptr N K
      lora_indices_int scaling xm_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K SPLIT_K_ONE).toAlgorithm?
        = Except.ok alg) ∧
    -- (2) both per-branch proof surfaces lower to the algorithm layer
    (∃ alg, (bgmv_shrink_store_surface input_ptr lora_ptr out_ptr N K
      lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K).toAlgorithm?
        = Except.ok alg) ∧
    (∃ alg, (bgmv_shrink_atomic_surface input_ptr lora_ptr out_ptr N K
      lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K).toAlgorithm?
        = Except.ok alg) ∧
    -- (3) the sentinel early-return path writes nothing
    (∀ s', s.readMemValue .int lora_indices_int (s.pids 1) = (-1 : Int) →
      exec (bgmv_shrink_surface input_ptr lora_ptr out_ptr N K lora_indices_int
        scaling xm_stride l0_stride lora_k_stride lora_n_stride cm_stride
        cn_stride BLOCK_N BLOCK_K SPLIT_K SPLIT_K_ONE) s = some s' →
      s'.mem = s.mem) ∧
    -- (4) SPLIT_K = 1: masked store of the genuine contraction
    ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Assumptions / layout contracts:**
- `hBK : 0 < BLOCK_K`
- `hSK : 0 < SPLIT_K`
- `hcn : 0 < cn_stride`

**Closed-form spec defs (transitive):** `bgmv_shrink_surface`, `bgmv_shrink_store_surface`, `bgmv_shrink_atomic_surface`

<details><summary><code>bgmv_shrink_surface</code></summary>

```
/-- Faithful transcription of `bgmv_shrink_kernel.py`'s `_bgmv_shrink_kernel`.

Python's signed `lora_index == -1` early return is represented as a guard
around the active body; the constexpr `SPLIT_K == 1` store-vs-atomic tail is
gated by `SPLIT_K_ONE` (`= decide (SPLIT_K = 1)` at the trusted host boundary).
`tl.max_contiguous` is a layout hint (erased to its value argument by the
DSL). -/
```
```lean
def bgmv_shrink_surface
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .int) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) (SPLIT_K_ONE : Bool) :
    ComputeKernel := triton {
  pid_sk = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
  if lora_index != $((-1 : Int)) {
    offset_n = tl.arange(0, $(BLOCK_N))
    offset_k = tl.arange(0, $(BLOCK_K)) + pid_sk * $(BLOCK_K)
    a_ptr = input_ptr + cur_batch * $(xm_stride)
    b_ptr = lora_ptr + $(l0_stride) * lora_index
    accumulator = tl.zeros([$(BLOCK_N)], dtype=tl.float32)
    for k in range($(0), $(K), $(BLOCK_K * SPLIT_K)) {
      current_k = k + offset_k
      current_k_c = tl.max_contiguous(current_k, $(BLOCK_K))
      tiled_a = tl.load(a_ptr + current_k_c, mask=current_k < $(K), other=0.0)
      b_ptr_mask = (offset_n[:, None] < $(N)) & (current_k[None, :] < $(K))
      tiled_b = tl.load(
        b_ptr + offset_n[:, None] * $(lora_k_stride) +
          current_k[None, :] * $(lora_n_stride),
        mask=b_ptr_mask, other=0.0)
      accumulator += tl.sum(tiled_a * tiled_b, 1)
    }
    accumulator *= $(scaling)
    offset_cn = tl.arange(0, $(BLOCK_N))
    c_ptr = out_ptr + cur_batch * $(cm_stride) + offset_cn * $(cn_stride)
    c_mask = offset_cn < $(N)
    if SPLIT_K_ONE {
      tl.store(c_ptr, accumulator, mask=c_mask)
    } else {
      tl.atomic_add(c_ptr, accumulator, mask=c_mask)
    }
  }
}
```
</details>

<details><summary><code>bgmv_shrink_store_surface</code></summary>

```
/-- The `SPLIT_K = 1` proof surface: sentinel guard elided (trusted host
boundary; the host only launches the active body when `lora_index ≠ -1`),
`Region .nat`-typed `lora_indices`, and the `tl.store` tail branch. The loop
stride is kept symbolic in `SPLIT_K` (the branch is taken at `SPLIT_K = 1`). -/
```
```lean
def bgmv_shrink_store_surface
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .nat) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) :
    ComputeKernel := triton {
  pid_sk = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
  offset_n = tl.arange(0, $(BLOCK_N))
  offset_k = tl.arange(0, $(BLOCK_K)) + pid_sk * $(BLOCK_K)
  a_ptr = input_ptr + cur_batch * $(xm_stride)
  b_ptr = lora_ptr + $(l0_stride) * lora_index
  accumulator = tl.zeros([$(BLOCK_N)], dtype=tl.float32)
  for k in range($(0), $(K), $(BLOCK_K * SPLIT_K)) {
    current_k = k + offset_k
    current_k_c = tl.max_contiguous(current_k, $(BLOCK_K))
    tiled_a = tl.load(a_ptr + current_k_c, mask=current_k < $(K), other=0.0)
    b_ptr_mask = (offset_n[:, None] < $(N)) & (current_k[None, :] < $(K))
    tiled_b = tl.load(
      b_ptr + offset_n[:, None] * $(lora_k_stride) +
        current_k[None, :] * $(lora_n_stride),
      mask=b_ptr_mask, other=0.0)
    accumulator += tl.sum(tiled_a * tiled_b, 1)
  }
  accumulator *= $(scaling)
  offset_cn = tl.arange(0, $(BLOCK_N))
  c_ptr = out_ptr + cur_batch * $(cm_stride) + offset_cn * $(cn_stride)
  c_mask = offset_cn < $(N)
  tl.store(c_ptr, accumulator, mask=c_mask)
}
```
</details>

<details><summary><code>bgmv_shrink_atomic_surface</code></summary>

```
/-- The `SPLIT_K > 1` proof surface: identical to the store surface except the
tail is the upstream `tl.atomic_add(c_ptr, accumulator, mask=c_mask)`. -/
```
```lean
def bgmv_shrink_atomic_surface
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .nat) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) :
    ComputeKernel := triton {
  pid_sk = tl.program_id(axis=0)
  cur_batch = tl.program_id(axis=1)
  lora_index = tl.load(lora_indices + cur_batch)
  offset_n = tl.arange(0, $(BLOCK_N))
  offset_k = tl.arange(0, $(BLOCK_K)) + pid_sk * $(BLOCK_K)
  a_ptr = input_ptr + cur_batch * $(xm_stride)
  b_ptr = lora_ptr + $(l0_stride) * lora_index
  accumulator = tl.zeros([$(BLOCK_N)], dtype=tl.float32)
  for k in range($(0), $(K), $(BLOCK_K * SPLIT_K)) {
    current_k = k + offset_k
    current_k_c = tl.max_contiguous(current_k, $(BLOCK_K))
    tiled_a = tl.load(a_ptr + current_k_c, mask=current_k < $(K), other=0.0)
    b_ptr_mask = (offset_n[:, None] < $(N)) & (current_k[None, :] < $(K))
    tiled_b = tl.load(
      b_ptr + offset_n[:, None] * $(lora_k_stride) +
        current_k[None, :] * $(lora_n_stride),
      mask=b_ptr_mask, other=0.0)
    accumulator += tl.sum(tiled_a * tiled_b, 1)
  }
  accumulator *= $(scaling)
  offset_cn = tl.arange(0, $(BLOCK_N))
  c_ptr = out_ptr + cur_batch * $(cm_stride) + offset_cn * $(cn_stride)
  c_mask = offset_cn < $(N)
  tl.atomic_add(c_ptr, accumulator, mask=c_mask)
}
```
</details>

## Also present (pinned special-case summaries)
- `store_compute_correct`
- `atomic_compute_correct`
