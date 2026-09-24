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
specification bgmv_shrink_kernel_output_summary_general
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

## Public theorem: `bgmv_shrink_store_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming metadata headline (wave-5, store face;
`SPLIT_K = 1` branch).** For every rounding model `R`, the store proof
surface implements, on its `StreamMetaMasked3DKernelIO₂` signature, the
**ideal ℝ scaled rank-slice fold** over the streamed tiles: for every
write-active output lane `j < N`,

```
out[cur_batch·cm_stride + j·cn_stride] =
  scaling · Σ_{t<⌈K/(BLOCK_K·SPLIT_K)⌉} Σ_{e<BLOCK_K}
    [current_k(t,e) < K] · x-tile[t](e) · loraA-tile[t](j,e)
```

with `current_k(t,e) = t·BLOCK_K·SPLIT_K + (e + pid_sk·BLOCK_K)` — the spec
`f` is exact real arithmetic over the pinned streams; the loaded metadata
slot `lora_indices[cur_batch]` enters only the `read2` base-address geometry
(the adapter-matrix select), never the value contract. The K-tail is genuine:
the streamed masks reproduce the kernel's `current_k < K` windows verbatim,
so `K` need not be a multiple of the split stride, and the masked spec sum
runs exactly over the kernel's live lanes.

This kernel has **zero rounding events** (every load, the K-loop and the
terminal masked store are `.real`), so with `outDType := .real` the readback
contract's `R.round .real` is the identity — the `⊨[R]` face at the `.real`
grid is the exact streaming contract stated once for every `R` (`R := .triv`
is the literal exact degeneration). Layer map: the prologue, K-loop and
common tail are cast-free and collapse onto the proven exact
`shrink_preLoop` / `shrink_step` / `forRange_inv` / `tail_common_steps`
stack; only the masked store statement is re-proved on the `R` side
(`shrink_postLoopR`).

Both hypotheses are truth-forced, inherited verbatim from
`store_exec_correct`:

* `hS : 0 < BLOCK_K * SPLIT_K` — the K-loop stride is positive (a launched
  constexpr tile is nonempty; at stride `0` the trip-count closed form
  `⌈K/(BLOCK_K·SPLIT_K)⌉` is meaningless and the loop would not advance).
* `hcn : 0 < cn_stride` — output-lane footprint injectivity
  (`affine1D_inj`): with `cn_stride = 0` all output lanes collide on one
  cell, the per-lane readback would be last-writer-wins and the statement
  false. Torch strides of a non-degenerate output are ≥ 1.

Division of labor: the `SPLIT_K > 1` **atomic face** is a read-modify-write
(`tl.atomic_add`), not a terminal single-store fold, hence outside this
skin's S1 genre — its per-program `out-before + contribution` obligation
remains conjunct (5) of the exact five-way summary
`bgmv_shrink_kernel_output_summary_general`, which is retained unchanged
(as are the faithful `.int` sentinel surface and its write-free skip path,
conjuncts (1)–(4)). -/
```
</details>

**Statement:**
```lean
specification bgmv_shrink_store_io_correctness (R : RoundingModel)
    (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .nat) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat)
    (hS : 0 < BLOCK_K * SPLIT_K) (hcn : 0 < cn_stride) :
    bgmvShrinkStoreIO input_ptr lora_ptr out_ptr N K lora_indices scaling
        xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
        BLOCK_N BLOCK_K SPLIT_K
      ⊨[R] fun pid₀ _ _ _ xs ys j =>
        scaling * ∑ t : Fin (numKIters K (BLOCK_K * SPLIT_K)),
          ∑ e : Fin BLOCK_K,
            if t.val * (BLOCK_K * SPLIT_K) + (e.val + pid₀ * BLOCK_K) < K
            then xs t e * ys t (bStreamLane BLOCK_N BLOCK_K j e)
            else 0
```

**Assumptions / layout contracts:**
- `hS : 0 < BLOCK_K * SPLIT_K`
- `hcn : 0 < cn_stride`

**Closed-form spec defs (transitive):** `bgmvShrinkStoreIO`, `numKIters`, `bStreamLane`, `bgmv_shrink_store_surface`

<details><summary><code>bgmvShrinkStoreIO</code></summary>

```
/-- **Streaming metadata IO signature** of the `SPLIT_K = 1` store face on the
metadata-slot two-stream fold skin (S1: fold + terminal store). One `.nat`
slot: `lora_indices[cur_batch]`, read at the pid-only cell `pid₁`
(`mwin`; the store surface's `Region .nat` metadata convention — the `-1`
skip sentinel is the faithful `.int` surface's separate obligation). Step `t`
of the K-loop reads the `[BLOCK_K]` input row tile and the
`[BLOCK_N, BLOCK_K]` LoRA-A weight tile; after the loop one `[BLOCK_N]`
output tile is masked-stored at the exact `.real` grid (`outDType := .real` —
the kernel's store is untyped/fp32-register, no rounding event). The windows
transcribe the kernel's pointer arithmetic verbatim:

* `read1` step `t`, lane `e`: `cur_batch·xm_stride + current_k` with
  `current_k = t·(BLOCK_K·SPLIT_K) + (e + pid_sk·BLOCK_K)` — the running
  `a_ptr + current_k_c` cell (`tl.max_contiguous` erased).
* `read2` step `t`, lane `l = (n, e)` (row-major over `[BLOCK_N, BLOCK_K]`):
  `l0_stride·slot + n·lora_k_stride + current_k·lora_n_stride` — the
  `b_ptr` cell; **the slot value enters this base address** (the loaded
  adapter index selects the LoRA-A matrix).
* `write` lane `j`: `cur_batch·cm_stride + j·cn_stride` — the `c_ptr` cell
  (pure pid geometry; the slot never reaches the write address).
* `mask1`/`mask2` transcribe the loads' genuine `current_k < K` K-tail
  windows (`mask2` also the `offset_n < N` leg); `writeMask` transcribes
  the store's `offset_cn < N` boundary mask verbatim.

The kernel launches on a 2-D grid `(pid_sk, cur_batch)`; the skin's `pid₂`
slot is unused — every window is constant in it, and the headline still
quantifies over all three pids. -/
```
```lean
def bgmvShrinkStoreIO (input_ptr lora_ptr out_ptr : RegionName)
    (N K : Nat) (lora_indices : Region .nat) (scaling : ℝ)
    (xm_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) : StreamMetaMasked3DKernelIO₂ where
  kernel := bgmv_shrink_store_surface input_ptr lora_ptr out_ptr N K
    lora_indices scaling xm_stride l0_stride lora_k_stride lora_n_stride
    cm_stride cn_stride BLOCK_N BLOCK_K SPLIT_K
  inp1 := input_ptr
  inp2 := lora_ptr
  out := out_ptr
  nMeta := 1
  sty := fun _ => .nat
  mbuf := fun _ => Region.cast lora_indices
  mwin := fun _ _ pid₁ _ => pid₁
  T := numKIters K (BLOCK_K * SPLIT_K)
  B1 := BLOCK_K
  B2 := BLOCK_N * BLOCK_K
  C := BLOCK_N
  outDType := .real
  read1 := fun pid₀ pid₁ _ _ t e =>
    pid₁ * xm_stride + (t.val * (BLOCK_K * SPLIT_K) + (e.val + pid₀ * BLOCK_K))
  read2 := fun pid₀ _ _ m t l =>
    l0_stride * m 0 + l.val / BLOCK_K * lora_k_stride
      + (t.val * (BLOCK_K * SPLIT_K) + (l.val % BLOCK_K + pid₀ * BLOCK_K))
        * lora_n_stride
  write := fun _ pid₁ _ _ j => pid₁ * cm_stride + j.val * cn_stride
  mask1 := fun pid₀ _ _ _ t e =>
    t.val * (BLOCK_K * SPLIT_K) + (e.val + pid₀ * BLOCK_K) < K
  mask2 := fun pid₀ _ _ _ t l =>
    l.val / BLOCK_K < N ∧
      t.val * (BLOCK_K * SPLIT_K) + (l.val % BLOCK_K + pid₀ * BLOCK_K) < K
  writeMask := fun _ _ _ _ j => j.val < N
```
</details>

<details><summary><code>numKIters</code></summary>

```
/-- Trip count of `for k in range(0, K, S)`: `⌈K/S⌉`. -/
```
```lean
def numKIters (K S : Nat) : Nat := (K + S - 1) / S
```
</details>

<details><summary><code>bStreamLane</code></summary>

```
/-- The `tiled_b`-stream lane feeding output lane `j` at rank key `e`: `(j, e)`
row-major over the `[BLOCK_N, BLOCK_K]` per-step weight tile, via the shared
`Lane2D` bridge. -/
```
```lean
def bStreamLane (BLOCK_N BLOCK_K : Nat) (j : Fin BLOCK_N) (e : Fin BLOCK_K) :
    Fin (BLOCK_N * BLOCK_K) :=
  Lane2D.encode (j, e, PUnit.unit)
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

## Also present (pinned special-case summaries)
- `store_compute_correct`
- `atomic_compute_correct`
