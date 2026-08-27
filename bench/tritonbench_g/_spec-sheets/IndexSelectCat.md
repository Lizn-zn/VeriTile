# Spec sheet — `bench/tritonbench_g/index_select_cat/IndexSelectCat.lean`

**Python source:** `bench/tritonbench_g/index_select_cat/index_select_cat.py`

## Public theorem: `index_select_cat_fwd_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `index_select_cat_fwd_kernel` implements the pure gather
`output[i, col] = source[index[i], col]` on its gather IO signature — for every
disjoint flat placement of `source_ptr`/`index_ptr`/`output_ptr`, every pair of
program ids whose active lanes are in bounds, and every launch state whose
windows hold the row-index tile `ids` (`.nat` channel) and the gathered data
tile `xs`, the translated pointer kernel terminates; every write-active lane
`j` (`indices < num_indices ∧ cols < num_cols`, the kernel's 2D `mask`) has
`xs j` at its contiguous destination cell; and every memory cell off the store
window is unchanged — unconditionally.

The skin's readback leg carries a per-context `WriteInj` antecedent because a
`GatherMasked2DKernelIO₁` `write` window may in general eat the loaded index
tile. On this pure-gather side the `write` window is *static*, so `WriteInj` is
just the old `hOutInj` no-duplicate-destination side condition on
`i·stride0 + col·stride1` at the write-active lanes — a fact about the strides,
not about the index data, and it stays inside `⊨` rather than narrowing the
headline.

`hBC : 0 < BLOCK_SIZE_COL` is genuinely forced: the index-tile load runs over
the `[BLOCK_SIZE_INDEX]` tile regardless of `BLOCK_SIZE_COL`, so when
`BLOCK_SIZE_COL = 0` the flattened lane space `Fin (BI * BC)` is empty and
carries no in-bounds witness for it.

Proof: `GatherMasked2DKernelIO₁.Implements.intro` assembles the region-model
gather triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification index_select_cat_fwd_kernel_correctness
    (output_ptr source_ptr index_ptr : RegionName)
    (num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (hBC : 0 < BLOCK_SIZE_COL) :
    indexSelectCatIO output_ptr source_ptr index_ptr num_indices num_cols
      stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL ⊨
        fun _ _ _ xs j => xs j
```

**Assumptions / layout contracts:**
- `hBC : 0 < BLOCK_SIZE_COL`

**Closed-form spec defs (transitive):** `indexSelectCatIO`, `index_select_cat_fwd_kernel`

<details><summary><code>indexSelectCatIO</code></summary>

```
/-- `index_select_cat_fwd_kernel`'s gather **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `inp`/`idxbuf`/`out` — which buffer is which argument (data `source_ptr`,
  `.nat` row-index channel `index_ptr`, destination `output_ptr`);
* `B = BLOCK_SIZE_INDEX * BLOCK_SIZE_COL` — the kernel's `[BI, BC]` tile,
  flattened: lane `j` is tile cell `(j / BC, j % BC)`;
* `readx` — the **static** index-tile window `pid0 * BI + j / BC`, i.e. the
  kernel's `indices` (row block of the 2D grid's first axis);
* `read` — the **data-dependent** gather source
  `ids j * stride0 + (pid1 * BC + j % BC) * stride1`: the loaded row times
  `stride0` plus the column offset, exactly the kernel's `source_offsets`;
* `write` — the **static** contiguous destination
  `(pid0 * BI + j / BC) * stride0 + (pid1 * BC + j % BC) * stride1`: the
  kernel's `output_offsets`, which ignores the loaded index tile;
* `mask` — the index-tile gate `pid0 * BI + j / BC < num_indices`;
* `readMask`/`writeMask` — the kernel's 2D `mask`, `indices < num_indices ∧
  cols < num_cols`.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing, masking and index-channel plumbing
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the active lanes. -/
```
```lean
def indexSelectCatIO (output_ptr source_ptr index_ptr : RegionName)
    (num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat) : GatherMasked2DKernelIO₁ where
  kernel := index_select_cat_fwd_kernel output_ptr source_ptr index_ptr
    num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL
  inp := source_ptr
  idxbuf := index_ptr
  out := output_ptr
  B := BLOCK_SIZE_INDEX * BLOCK_SIZE_COL
  readx := fun p₀ _ j => p₀ * BLOCK_SIZE_INDEX + j.val / BLOCK_SIZE_COL
  read := fun _ p₁ ids j =>
    ids j * stride0 + (p₁ * BLOCK_SIZE_COL + j.val % BLOCK_SIZE_COL) * stride1
  write := fun p₀ p₁ _ j =>
    (p₀ * BLOCK_SIZE_INDEX + j.val / BLOCK_SIZE_COL) * stride0 +
      (p₁ * BLOCK_SIZE_COL + j.val % BLOCK_SIZE_COL) * stride1
  mask := fun p₀ _ j =>
    p₀ * BLOCK_SIZE_INDEX + j.val / BLOCK_SIZE_COL < num_indices
  readMask := fun p₀ p₁ _ j =>
    p₀ * BLOCK_SIZE_INDEX + j.val / BLOCK_SIZE_COL < num_indices ∧
      p₁ * BLOCK_SIZE_COL + j.val % BLOCK_SIZE_COL < num_cols
  writeMask := fun p₀ p₁ _ j =>
    p₀ * BLOCK_SIZE_INDEX + j.val / BLOCK_SIZE_COL < num_indices ∧
      p₁ * BLOCK_SIZE_COL + j.val % BLOCK_SIZE_COL < num_cols
```
</details>

<details><summary><code>index_select_cat_fwd_kernel</code></summary>

```
/-- Faithful transcription of `index_select_cat.py`'s
`index_select_cat_fwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE_INDEX: tl.constexpr` / `BLOCK_SIZE_COL: tl.constexpr`
  → Lean `Nat` parameters.
- `index_ptr` is a typed Lean region so its `tl.load` needs no extra `dtype=`
  kwarg. -/
```
```lean
def index_select_cat_fwd_kernel
    (output_ptr source_ptr : RegionName) (index_ptr : Region .nat)
    (num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat) :
    ComputeKernel := triton {
  pid0 = tl.program_id(axis=0)
  pid1 = tl.program_id(axis=1)
  indices = pid0 * $(BLOCK_SIZE_INDEX) + tl.arange(0, $(BLOCK_SIZE_INDEX))
  rows = tl.load(index_ptr + indices, mask=indices < $(num_indices))
  cols = pid1 * $(BLOCK_SIZE_COL) + tl.arange(0, $(BLOCK_SIZE_COL))
  source_offsets = source_ptr + rows[:, None] * $(stride0) + cols[None, :] * $(stride1)
  mask = (indices[:, None] < $(num_indices)) & (cols[None, :] < $(num_cols))
  output = tl.load(source_offsets, mask=mask)
  output_offsets = output_ptr + indices[:, None] * $(stride0) + cols[None, :] * $(stride1)
  tl.store(output_offsets, output, mask=mask)
}
```
</details>
