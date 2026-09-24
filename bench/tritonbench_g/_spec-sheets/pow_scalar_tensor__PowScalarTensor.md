# Spec sheet — `bench/tritonbench_g/pow_scalar_tensor/PowScalarTensor.lean`

**Python source:** `bench/tritonbench_g/pow_scalar_tensor/pow_scalar_tensor.py`

## Public theorem: `pow_scalar_tensor_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general** correctness summary for `pow_scalar_tensor.py`'s
`pow_func_scalar_tensor_kernel_rank_1`, against the **genuine closed form**
`powSpec = Real.rpow val0 (in0[t·in0_stride0])` — a pure function of INPUT
memory and the runtime scalar base `val0`, never a read-back of the kernel's
own output — for arbitrary `val0 : ℝ`, `s0`, strides, `tile_size0` and
`tiles_per_cta`. It packages, for **both** `one_tile_per_cta` constexpr
branches:

* both surfaces lower to the algorithm layer;
* the `one_tile_per_cta = true` branch: every active lane
  (`pid·tile_size0 + i < s0`) of the program's single tile holds
  `val0 ** in0[t·in0_stride0]` at `out0[t·out0_stride0]`;
* the `one_tile_per_cta = false` grid-stride branch: for **every** iteration
  `j < tiles_per_cta` and lane `i` of tile `pid + j·num_ctas`, the active
  cells hold the genuine power value — the full multi-iteration loop is
  verified end-to-end (loop invariant `gs_loop_readback`).

Honest side-conditions: `0 < out0_stride0` (store-footprint injectivity —
torch strides of a non-degenerate rank-1 buffer are ≥ 1), and for the
grid-stride branch `in0_ptr ≠ out0_ptr` (later iterations load after earlier
stores) and `0 < s.numPids 0` (a launched grid has at least one program).
Spec-strength caveat (see the Modeling boundary section and the `Op.pow` doc
comment in `VeriTile/Triton/Core/Ast.lean`): `_pow` is `Real.rpow`, which
matches CUDA `pow` for `val0 > 0` but returns Mathlib's junk-value convention
where CUDA returns NaN for a negative base with non-integer exponent. -/
```
</details>

**Statement:**
```lean
specification pow_scalar_tensor_output_summary_general
    (val0 : ℝ) (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (s : BlockState)
    (hStride : 0 < out0_stride0)
    (hDisj : in0_ptr ≠ out0_ptr)
    (hGrid : 0 < s.numPids 0) :
    -- (1) both branch surfaces lower to the algorithm layer
    (∃ alg, (pow_func_scalar_tensor_kernel_rank_1_one_tile_surface val0
      in0_ptr out0_ptr in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
      tile_size0).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (pow_func_scalar_tensor_kernel_rank_1_grid_stride_surface val0
      in0_ptr out0_ptr in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
      tile_size0).toAlgorithm? = Except.ok alg) ∧
    -- (2) one_tile_per_cta = true: genuine elementwise scalar-base power
    ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Assumptions / layout contracts:**
- `hStride : 0 < out0_stride0`
- `hDisj : in0_ptr ≠ out0_ptr`
- `hGrid : 0 < s.numPids 0`

**Closed-form spec defs (transitive):** `pow_func_scalar_tensor_kernel_rank_1_one_tile_surface`, `pow_func_scalar_tensor_kernel_rank_1_grid_stride_surface`

<details><summary><code>pow_func_scalar_tensor_kernel_rank_1_one_tile_surface</code></summary>

```
/-- Faithful transcription of `pow_scalar_tensor.py`'s
`pow_func_scalar_tensor_kernel_rank_1`, specialized to the
`one_tile_per_cta = true` (monolithic) branch: one `tile_size0`-wide tile per
program, block-pointer load/store with `boundary_check` on axis 0,
`pow_func_scalar_tensor` inlined as
`tl.extra.cuda.libdevice.pow(val0.to(tl.float32), in0)`. -/
```
```lean
def pow_func_scalar_tensor_kernel_rank_1_one_tile_surface
    (val0 : ℝ) (in0_ptr out0_ptr : RegionName)
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
  out0 = tl.extra.cuda.libdevice.pow($((val0 : ℝ)).to(tl.float32), in0)
  out0_bptr = tl.make_block_ptr(base=out0_ptr, shape=($(s0)), strides=($(out0_stride0)),
    offsets=(offset0), block_shape=($(tile_size0)), order=(0))
  tl.store(out0_bptr, (out0).to(out0_bptr.type.element_ty), boundary_check=([0] : List Nat))
}
```
</details>

<details><summary><code>pow_func_scalar_tensor_kernel_rank_1_grid_stride_surface</code></summary>

```
/-- Faithful transcription of `pow_func_scalar_tensor_kernel_rank_1`,
specialized to the `one_tile_per_cta = false` (grid-stride-loop) branch:
program `pid` covers tiles `pid + j·num_ctas` for `j < tiles_per_cta`, with
`num_ctas = tl.num_programs(0)`. -/
```
```lean
def pow_func_scalar_tensor_kernel_rank_1_grid_stride_surface
    (val0 : ℝ) (in0_ptr out0_ptr : RegionName)
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
    out0 = tl.extra.cuda.libdevice.pow($((val0 : ℝ)).to(tl.float32), in0)
    out0_bptr = tl.make_block_ptr(base=out0_ptr, shape=($(s0)), strides=($(out0_stride0)),
      offsets=(offset0), block_shape=($(tile_size0)), order=(0))
    tl.store(out0_bptr, (out0).to(out0_bptr.type.element_ty), boundary_check=([0] : List Nat))
  }
}
```
</details>

## Public theorem: `pow_scalar_tensor_one_tile_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for `pow_scalar_tensor.py`'s
`pow_func_scalar_tensor_kernel_rank_1`, `one_tile_per_cta = true` branch: for
every disjoint flat placement of the two buffers, every program id whose active
lanes are in bounds, and every launch state whose input window holds `xs`, the
translated pointer kernel terminates, every active lane of the output tile holds
the genuine scalar-base power `val0 ^ xs i` (`Real.rpow`), and every other
memory cell is unchanged.

Dimension-general in `s0`, both strides, `tile_size0`, and the scalar base
`val0`; the only side-condition is `0 < out0_stride0` (store-footprint
injectivity — a torch stride of a non-degenerate rank-1 buffer is ≥ 1). The
strided, two-stride footprint is carried verbatim by `MaskedTileKernelIO₁`'s
address functions. -/
```
</details>

**Statement:**
```lean
specification pow_scalar_tensor_one_tile_io_correctness (val0 : ℝ)
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (hStride : 0 < out0_stride0) :
    powOneTileIO val0 in0_ptr out0_ptr in0_stride0 out0_stride0 s0 num_tasks
        tiles_per_cta tile_size0
      ⊨ fun _pid xs i => Real.rpow val0 (xs i)
```

**Assumptions / layout contracts:**
- `hStride : 0 < out0_stride0`

**Closed-form spec defs (transitive):** `powOneTileIO`, `pow_func_scalar_tensor_kernel_rank_1_one_tile_surface`, `taskIndex`

<details><summary><code>powOneTileIO</code></summary>

```
/-- IO signature of the `one_tile_per_cta = true` branch on the **tile-indexed**
surface: lane `i` of program `pid` covers task `t = pid·tile_size0 + i`, reads
`in0_ptr` at `t·in0_stride0`, writes `out0_ptr` at `t·out0_stride0`, and is
active exactly on the `boundary_check=(0,)` guard `t < s0`. -/
```
```lean
def powOneTileIO (val0 : ℝ) (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    MaskedTileKernelIO₁ where
  kernel := pow_func_scalar_tensor_kernel_rank_1_one_tile_surface val0 in0_ptr
    out0_ptr in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0
  inp := in0_ptr
  out := out0_ptr
  shape := [tile_size0]
  read := fun pid i => taskIndex pid tile_size0 i.1 * in0_stride0
  write := fun pid i => taskIndex pid tile_size0 i.1 * out0_stride0
  mask := fun pid i => taskIndex pid tile_size0 i.1 < s0
```
</details>

<details><summary><code>pow_func_scalar_tensor_kernel_rank_1_one_tile_surface</code></summary>

```
/-- Faithful transcription of `pow_scalar_tensor.py`'s
`pow_func_scalar_tensor_kernel_rank_1`, specialized to the
`one_tile_per_cta = true` (monolithic) branch: one `tile_size0`-wide tile per
program, block-pointer load/store with `boundary_check` on axis 0,
`pow_func_scalar_tensor` inlined as
`tl.extra.cuda.libdevice.pow(val0.to(tl.float32), in0)`. -/
```
```lean
def pow_func_scalar_tensor_kernel_rank_1_one_tile_surface
    (val0 : ℝ) (in0_ptr out0_ptr : RegionName)
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
  out0 = tl.extra.cuda.libdevice.pow($((val0 : ℝ)).to(tl.float32), in0)
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

## Also present (pinned special-case summaries)
- `pow_one_tile_compute_correct`
- `pow_grid_stride_compute_correct`
