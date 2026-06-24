# Spec sheet — `bench/tritonbench_g/destindex_copy/DestindexCopy.lean`

**Python source:** `bench/tritonbench_g/destindex_copy/destindex_copy.py`

## Public theorem: `fwd_kernel_destindex_copy_kv_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `_fwd_kernel_destindex_copy_kv`: the DSL
surface lowers to the algorithm layer, and both dest-indexed scatters are
compute-correct — every `O_nope` cell holds the matching `KV_nope` cell and
every `O_rope` cell holds the matching `KV_rope` cell. -/
```
</details>

**Statement:**
```lean
theorem fwd_kernel_destindex_copy_kv_output_summary
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .nat) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat)
    (s : BlockState)
    (hRegion : O_nope ≠ O_rope)
    (hOutNopeInj : Function.Injective
      (fun idx : TileIndex [1, BLOCK_DMODEL_NOPE] =>
        outNopeAddr s Dest_loc stride_o_nope_bs stride_o_nope_d idx))
    (hOutRopeInj : Function.Injective
      (fun idx : TileIndex [1, BLOCK_DMODEL_ROPE] =>
        outRopeAddr s Dest_loc stride_o_rope_bs stride_o_rope_d idx)) :
    (∃ alg, (fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
        stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
        stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
        stride_o_nope_bs stride_o_nope_h stride_o_nope_d
        stride_o_rope_bs stride_o_rope_h stride_o_rope_d
        kv_nope_head_num kv_rope_head_num
        BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
        stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
        stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
        stride_o_nope_bs stride_o_nope_h stride_o_nope_d
        stride_o_rope_bs stride_o_rope_h stride_o_rope_d
        kv_nope_head_num kv_rope_head_num
        BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE)
      (initialState := s)
      (write := fun idx : TileIndex [1, BLOCK_DMODEL_NOPE] =>
        some (O_nope, outNopeAddr s Dest_loc stride_o_nope_bs stride_o_nope_d idx))
      (expected := fun idx : TileIndex [1, BLOCK_DMODEL_NOPE] =>
        s.readMem KV_nope (sourceNopeAddr s stride_kv_nope_bs stride_kv_nope_d idx)) ∧
    ComputeCorrect.Realizes
      (kernel := fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
        stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
        stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
        stride_o_nope_bs stride_o_nope_h stride_o_nope_d
        stride_o_rope_bs stride_o_rope_h stride_o_rope_d
        kv_nope_head_num kv_rope_head_num
        BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE)
      (initialState := s)
      (write := fun idx : TileIndex [1, BLOCK_DMODEL_ROPE] =>
        some (O_rope, outRopeAddr s Dest_loc stride_o_rope_bs stride_o_rope_d idx))
      (expected := fun idx : TileIndex [1, BLOCK_DMODEL_ROPE] =>
        s.readMem KV_rope (sourceRopeAddr s stride_kv_rope_bs stride_kv_rope_d idx))
```

**Assumptions / layout contracts:**
- `hRegion : O_nope ≠ O_rope`
- `hOutNopeInj : Function.Injective
      (fun idx : TileIndex [1, BLOCK_DMODEL_NOPE] =>
        outNopeAddr s Dest_loc stride_o_nope_bs stride_o_nope_d idx)`
- `hOutRopeInj : Function.Injective
      (fun idx : TileIndex [1, BLOCK_DMODEL_ROPE] =>
        outRopeAddr s Dest_loc stride_o_rope_bs stride_o_rope_d idx)`
- `kernel : = fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
        stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
        stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
        stride_o_nope_bs stride_o_nope_h stride_o_nope_d
        stride_o_rope_bs stride_o_rope_h stride_o_rope_d
        kv_nope_head_num kv_rope_head_num
        BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE`
- `initialState : = s`
- `write : = fun idx : TileIndex [1, BLOCK_DMODEL_NOPE] =>
        some (O_nope, outNopeAddr s Dest_loc stride_o_nope_bs stride_o_nope_d idx)`
- `expected : = fun idx : TileIndex [1, BLOCK_DMODEL_NOPE] =>
        s.readMem KV_nope (sourceNopeAddr s stride_kv_nope_bs stride_kv_nope_d idx)`
- `kernel : = fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
        stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
        stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
        stride_o_nope_bs stride_o_nope_h stride_o_nope_d
        stride_o_rope_bs stride_o_rope_h stride_o_rope_d
        kv_nope_head_num kv_rope_head_num
        BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE`
- `initialState : = s`
- `write : = fun idx : TileIndex [1, BLOCK_DMODEL_ROPE] =>
        some (O_rope, outRopeAddr s Dest_loc stride_o_rope_bs stride_o_rope_d idx)`
- `expected : = fun idx : TileIndex [1, BLOCK_DMODEL_ROPE] =>
        s.readMem KV_rope (sourceRopeAddr s stride_kv_rope_bs stride_kv_rope_d idx)`

**Closed-form spec defs (transitive):** `outNopeAddr`, `outRopeAddr`, `fwd_kernel_destindex_copy_kv`, `sourceNopeAddr`, `sourceRopeAddr`, `destBase`, `dimNope`, `dimRope`

<details><summary><code>outNopeAddr</code></summary>

```lean
def outNopeAddr
    (s : BlockState) (Dest_loc : RegionName)
    (stride_o_nope_bs stride_o_nope_d : Nat)
    (idx : TileIndex [1, BLOCK_DMODEL_NOPE]) : Nat :=
  destBase s Dest_loc * stride_o_nope_bs + stride_o_nope_d * dimNope idx
```
</details>

<details><summary><code>outRopeAddr</code></summary>

```lean
def outRopeAddr
    (s : BlockState) (Dest_loc : RegionName)
    (stride_o_rope_bs stride_o_rope_d : Nat)
    (idx : TileIndex [1, BLOCK_DMODEL_ROPE]) : Nat :=
  destBase s Dest_loc * stride_o_rope_bs + stride_o_rope_d * dimRope idx
```
</details>

<details><summary><code>fwd_kernel_destindex_copy_kv</code></summary>

```
/-- Faithful transcription of `destindex_copy.py`'s
`_fwd_kernel_destindex_copy_kv`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_DMODEL_*: tl.constexpr` -> Lean `Nat` parameters.
- The unused Python head-count and head-stride arguments are retained at the
  theorem boundary, matching the original kernel signature. -/
```
```lean
def fwd_kernel_destindex_copy_kv
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .nat) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs _stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs _stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs _stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs _stride_o_rope_h stride_o_rope_d
      _kv_nope_head_num _kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_d_nope = tl.arange(0, $(BLOCK_DMODEL_NOPE))
  offs_d_rope = tl.arange(0, $(BLOCK_DMODEL_ROPE))
  dest_index = tl.load(Dest_loc + cur_index)

  kv_nope_ptrs = KV_nope + cur_index * $(stride_kv_nope_bs) +
    $(stride_kv_nope_d) * offs_d_nope[None, :]
  kv_rope_ptrs = KV_rope + cur_index * $(stride_kv_rope_bs) +
    $(stride_kv_rope_d) * offs_d_rope[None, :]

  o_nope_ptrs = O_nope + dest_index * $(stride_o_nope_bs) +
    $(stride_o_nope_d) * offs_d_nope[None, :]
  o_rope_ptrs = O_rope + dest_index * $(stride_o_rope_bs) +
    $(stride_o_rope_d) * offs_d_rope[None, :]

  kv_nope = tl.load(kv_nope_ptrs)
  kv_rope = tl.load(kv_rope_ptrs)

  tl.store(o_nope_ptrs, kv_nope)
  tl.store(o_rope_ptrs, kv_rope)
}
```
</details>

<details><summary><code>sourceNopeAddr</code></summary>

```lean
def sourceNopeAddr
    (s : BlockState) (stride_kv_nope_bs stride_kv_nope_d : Nat)
    (idx : TileIndex [1, BLOCK_DMODEL_NOPE]) : Nat :=
  s.pid * stride_kv_nope_bs + stride_kv_nope_d * dimNope idx
```
</details>

<details><summary><code>sourceRopeAddr</code></summary>

```lean
def sourceRopeAddr
    (s : BlockState) (stride_kv_rope_bs stride_kv_rope_d : Nat)
    (idx : TileIndex [1, BLOCK_DMODEL_ROPE]) : Nat :=
  s.pid * stride_kv_rope_bs + stride_kv_rope_d * dimRope idx
```
</details>

<details><summary><code>destBase</code></summary>

```lean
def destBase (s : BlockState) (Dest_loc : RegionName) : Nat :=
  s.readMemValue .nat Dest_loc s.pid
```
</details>

<details><summary><code>dimNope</code></summary>

```lean
def dimNope (idx : TileIndex [1, BLOCK_DMODEL_NOPE]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>dimRope</code></summary>

```lean
def dimRope (idx : TileIndex [1, BLOCK_DMODEL_ROPE]) : Nat :=
  idx.2.1.val
```
</details>

## Also present (pinned special-case summaries)
- `fwd_kernel_destindex_copy_kv_nope_compute_correct`
- `fwd_kernel_destindex_copy_kv_rope_compute_correct`
