# Spec sheet — `bench/tritonbench_g/relu_strided_buffer/ReluStridedBuffer.lean`

**Python source:** `bench/tritonbench_g/relu_strided_buffer/relu_strided_buffer.py`

## Public theorem: `relu_strided_buffer_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general** correctness summary for `relu_strided_buffer.py`'s
`relu_forward_kernel_rank_1`, against the **genuine closed form**
`reluSpec = relu(in0[t·in0_stride0]) = max 0 (…)` — a pure function of INPUT
memory, never a read-back of the kernel's own output — for arbitrary `s0`,
strides, `tile_size0` and `tiles_per_cta`. It packages, for **both**
`one_tile_per_cta` constexpr branches:

* both surfaces lower to the algorithm layer;
* the `one_tile_per_cta = true` branch: every active lane
  (`pid·tile_size0 + i < s0`) of the program's single tile holds
  `relu(in0[t·in0_stride0])` at `out0[t·out0_stride0]`;
* the `one_tile_per_cta = false` grid-stride branch: for **every** iteration
  `j < tiles_per_cta` and lane `i` of tile `pid + j·num_ctas`, the active
  cells hold the genuine ReLU value — the full multi-iteration loop is
  verified end-to-end (loop invariant `gs_loop_readback`).

Honest side-conditions: `0 < out0_stride0` (store-footprint injectivity —
torch strides of a non-degenerate rank-1 buffer are ≥ 1), and for the
grid-stride branch `in0_ptr ≠ out0_ptr` (later iterations load after earlier
stores) and `0 < s.numPids 0` (a launched grid has at least one program). -/
```
</details>

**Statement:**
```lean
theorem relu_strided_buffer_output_summary_general
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (s : BlockState)
    (hStride : 0 < out0_stride0)
    (hDisj : in0_ptr ≠ out0_ptr)
    (hGrid : 0 < s.numPids 0) :
    -- (1) both branch surfaces lower to the algorithm layer
    (∃ alg, (relu_forward_kernel_rank_1_one_tile_surface in0_ptr out0_ptr
      in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
      tile_size0).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (relu_forward_kernel_rank_1_grid_stride_surface in0_ptr out0_ptr
      in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
      tile_size0).toAlgorithm? = Except.ok alg) ∧
    -- (2) one_tile_per_cta = true: genuine elementwise ReLU
    ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Assumptions / layout contracts:**
- `hStride : 0 < out0_stride0`
- `hDisj : in0_ptr ≠ out0_ptr`
- `hGrid : 0 < s.numPids 0`

**Closed-form spec defs (transitive):** `relu_forward_kernel_rank_1_one_tile_surface`, `relu_forward_kernel_rank_1_grid_stride_surface`

<details><summary><code>relu_forward_kernel_rank_1_one_tile_surface</code></summary>

```
/-- Faithful transcription of `relu_strided_buffer.py`'s
`relu_forward_kernel_rank_1`, specialized to the `one_tile_per_cta = true`
(monolithic) branch: one `tile_size0`-wide tile per program, block-pointer
load/store with `boundary_check` on axis 0, `relu_forward` inlined as
`tl.where(in0 > 0, in0, 0)`. -/
```
```lean
def relu_forward_kernel_rank_1_one_tile_surface
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  num_tiles0 = tl.cdiv($(s0), $(tile_size0))
  tile_id = pid
  tile_id0 = tile_id
  offset0 = tile_id0 * $(tile_size0)
  in0_bptr = tl.make_block_ptr(base=in0_ptr, shape=($(s0)), strides=($(in0_stride0)),
    offsets=(offset0), block_shape=($(tile_size0)), order=(0))
  in0 = (tl.load(in0_bptr, boundary_check=([0] : List Nat))).to(in0_ptr.type.element_ty)
  out0 = tl.where(in0 > 0, in0, 0)
  out0_bptr = tl.make_block_ptr(base=out0_ptr, shape=($(s0)), strides=($(out0_stride0)),
    offsets=(offset0), block_shape=($(tile_size0)), order=(0))
  tl.store(out0_bptr, (out0).to(out0_bptr.type.element_ty), boundary_check=([0] : List Nat))
}
```
</details>

<details><summary><code>relu_forward_kernel_rank_1_grid_stride_surface</code></summary>

```
/-- Faithful transcription of `relu_forward_kernel_rank_1`, specialized to the
`one_tile_per_cta = false` (grid-stride-loop) branch: program `pid` covers
tiles `pid + j·num_ctas` for `j < tiles_per_cta`, with
`num_ctas = tl.num_programs(0)`. -/
```
```lean
def relu_forward_kernel_rank_1_grid_stride_surface
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  num_tiles0 = tl.cdiv($(s0), $(tile_size0))
  num_ctas = tl.num_programs(0)
  for j in range($(0), $(tiles_per_cta)) {
    tile_id = pid + j * num_ctas
    tile_id0 = tile_id
    offset0 = tile_id0 * $(tile_size0)
    in0_bptr = tl.make_block_ptr(base=in0_ptr, shape=($(s0)), strides=($(in0_stride0)),
      offsets=(offset0), block_shape=($(tile_size0)), order=(0))
    in0 = (tl.load(in0_bptr, boundary_check=([0] : List Nat))).to(in0_ptr.type.element_ty)
    out0 = tl.where(in0 > 0, in0, 0)
    out0_bptr = tl.make_block_ptr(base=out0_ptr, shape=($(s0)), strides=($(out0_stride0)),
      offsets=(offset0), block_shape=($(tile_size0)), order=(0))
    tl.store(out0_bptr, (out0).to(out0_bptr.type.element_ty), boundary_check=([0] : List Nat))
  }
}
```
</details>

## Also present (pinned special-case summaries)
- `relu_one_tile_compute_correct`
- `relu_grid_stride_compute_correct`
