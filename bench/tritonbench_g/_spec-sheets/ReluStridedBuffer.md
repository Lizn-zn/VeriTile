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
specification relu_strided_buffer_output_summary_general
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

## Public theorem: `relu_strided_buffer_one_tile_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for `relu_strided_buffer.py`'s
`relu_forward_kernel_rank_1`, `one_tile_per_cta = true` branch: for every
disjoint flat placement of the two buffers, every program id whose active lanes
are in bounds, and every launch state whose input window holds `xs`, the
translated pointer kernel terminates, every active lane of the output tile holds
`relu (xs i) = max 0 (xs i)`, and every other memory cell is unchanged.

Dimension-general in `s0`, both strides, `tile_size0`; the only side-condition
is `0 < out0_stride0` (store-footprint injectivity — a torch stride of a
non-degenerate rank-1 buffer is ≥ 1). The strided, two-stride footprint is
carried verbatim by `MaskedTileKernelIO₁`'s address functions. -/
```
</details>

**Statement:**
```lean
specification relu_strided_buffer_one_tile_io_correctness
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (hStride : 0 < out0_stride0) :
    reluOneTileIO in0_ptr out0_ptr in0_stride0 out0_stride0 s0 num_tasks
        tiles_per_cta tile_size0
      ⊨ fun _pid xs i => TiledActivation.relu (xs i)
```

**Assumptions / layout contracts:**
- `hStride : 0 < out0_stride0`

**Closed-form spec defs (transitive):** `reluOneTileIO`, `relu_forward_kernel_rank_1_one_tile_surface`, `taskIndex`

<details><summary><code>reluOneTileIO</code></summary>

```
/-- IO signature of the `one_tile_per_cta = true` branch on the **tile-indexed**
surface: lane `i` of program `pid` covers task `t = pid·tile_size0 + i`, reads
`in0_ptr` at `t·in0_stride0`, writes `out0_ptr` at `t·out0_stride0`, and is
active exactly on the `boundary_check=(0,)` guard `t < s0`. -/
```
```lean
def reluOneTileIO (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    MaskedTileKernelIO₁ where
  kernel := relu_forward_kernel_rank_1_one_tile_surface in0_ptr out0_ptr
    in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0
  inp := in0_ptr
  out := out0_ptr
  shape := [tile_size0]
  read := fun pid i => taskIndex pid tile_size0 i.1 * in0_stride0
  write := fun pid i => taskIndex pid tile_size0 i.1 * out0_stride0
  mask := fun pid i => taskIndex pid tile_size0 i.1 < s0
```
</details>

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

<details><summary><code>taskIndex</code></summary>

```
/-- Flat task index covered by lane `i` of tile `tile_id0`. -/
```
```lean
def taskIndex (tile_id0 tile_size0 : Nat) (i : Fin tile_size0) : Nat :=
  tile_id0 * tile_size0 + i.val
```
</details>

## Public theorem: `relu_strided_buffer_grid_stride_io_correctnessR`

<details><summary>docstring</summary>

```
/-- **The headline on the grid-width-aware IO surface** for
`relu_strided_buffer.py`'s `relu_forward_kernel_rank_1`,
`one_tile_per_cta = false` (grid-stride) branch: for every disjoint flat
placement of the two buffers, every program id and **launch grid width**, and
every launch state whose per-step input windows hold `xs`, the translated
pointer kernel terminates and every write-active lane of every grid-stride step
holds `relu (xs t i) = max 0 (xs t i)`, with every other memory **cell**
unchanged.

Dimension-general in `s0`, both strides, `tile_size0` and `tiles_per_cta`, and
**universally quantified over the launch grid width** — the point of the skin:
the windows stride by `nCtas = tl.num_programs(0)`, pinned inside the relation
to `s₀.numPids 0`.

Honest side-conditions: `0 < out0_stride0` (store-footprint injectivity — a
torch stride of a non-degenerate rank-1 buffer is ≥ 1); `in0_ptr ≠ out0_ptr`
(later grid-stride steps load after earlier steps stored, and the two buffers
carry *different* strides, so an aliased pair genuinely breaks the contract —
the IO surface's placement disjointness does not supply this, since the core
allows a skin to name one region twice for in-place kernels); and the skin's
`pre` `pid₀ < nCtas` (launch legality, which also supplies the `0 < nCtas` the
loop arithmetic needs).

`R` is threaded through the whole kernel but rounds nothing — with
`outDType := .real` every per-step store is exact under `execR R`, so this is
the exact grid-stride streaming contract. That is faithful rather than vacuous
because ReLU is a *selection* (`tl.where`), so no arithmetic manufactures a real
that would need quantizing. -/
```
</details>

**Statement:**
```lean
specification relu_strided_buffer_grid_stride_io_correctnessR (R : RoundingModel)
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (hStride : 0 < out0_stride0) (hDisj : in0_ptr ≠ out0_ptr) :
    reluGridStrideIO in0_ptr out0_ptr in0_stride0 out0_stride0 s0 num_tasks
        tiles_per_cta tile_size0
      ⊨[R] fun _pid₀ _pid₁ _nCtas xs t i => TiledActivation.relu (xs t i)
```

**Assumptions / layout contracts:**
- `hStride : 0 < out0_stride0`
- `hDisj : in0_ptr ≠ out0_ptr`

**Closed-form spec defs (transitive):** `reluGridStrideIO`, `relu_forward_kernel_rank_1_grid_stride_surface`, `taskIndex`

<details><summary><code>reluGridStrideIO</code></summary>

```
/-- IO signature of the `one_tile_per_cta = false` **grid-stride** branch on the
grid-width-aware streaming surface: step `j` of program `(pid₀, _)` in a launch
grid of width `nCtas` covers the flat tile `pid₀ + j·nCtas`, whose lane `i` reads
`in0_ptr` at `t·in0_stride0` and writes `out0_ptr` at `t·out0_stride0` for the
task index `t`, active exactly on the `boundary_check=(0,)` guard `t < s0`.

`pre` is the grid-stride idiom's own launch legality `pid₀ < nCtas` — a program's
id is below its grid's width. `BlockState` carries no invariant tying `pids` to
`numPids`, and the kernel's loop genuinely needs `0 < nCtas` (at `nCtas = 0`
every step would revisit tile `pid₀`), so it is assumed here rather than
pretended free. `outDType` stays at the default `.real`: both source casts are
`.to(...element_ty)`, which the DSL erases. -/
```
```lean
def reluGridStrideIO (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    StreamGridStrideEmitMasked2DKernelIO₁ where
  kernel := relu_forward_kernel_rank_1_grid_stride_surface in0_ptr out0_ptr
    in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0
  inp1 := in0_ptr
  out := out0_ptr
  T := tiles_per_cta
  B1 := tile_size0
  C := tile_size0
  pre := fun pid₀ _ nCtas => pid₀ < nCtas
  read1 := fun p₀ _ nCtas t j =>
    taskIndex (p₀ + t.val * nCtas) tile_size0 j * in0_stride0
  write := fun p₀ _ nCtas t j =>
    taskIndex (p₀ + t.val * nCtas) tile_size0 j * out0_stride0
  mask1 := fun p₀ _ nCtas t j => taskIndex (p₀ + t.val * nCtas) tile_size0 j < s0
  writeMask := fun p₀ _ nCtas t j =>
    taskIndex (p₀ + t.val * nCtas) tile_size0 j < s0
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

<details><summary><code>taskIndex</code></summary>

```
/-- Flat task index covered by lane `i` of tile `tile_id0`. -/
```
```lean
def taskIndex (tile_id0 tile_size0 : Nat) (i : Fin tile_size0) : Nat :=
  tile_id0 * tile_size0 + i.val
```
</details>

## Also present (pinned special-case summaries)
- `relu_one_tile_compute_correct`
- `relu_grid_stride_compute_correct`
