# Spec sheet — `bench/tritonbench_g/index_select_bwd/IndexSelectBwd.lean`

**Python source:** `bench/tritonbench_g/index_select_bwd/index_select_bwd.py`

## Public theorem: `index_select_cat_bwd_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `index_select_cat_bwd_kernel` implements the indexed
gradient scatter `grad_source[index[i], col] = grad_output[i, col]` on its
scatter IO signature — for every disjoint flat placement of
`grad_output_ptr`/`index_ptr`/`grad_source_ptr`, every pair of program ids
whose active lanes are in bounds (the write bound at the *ungated* scatter
cells), and every launch state whose windows hold the row-index tile `ids`
(`.nat` channel) and the gradient tile `xs`, the translated pointer kernel
terminates; under the per-context `WriteInj` antecedent every write-active lane
`j` (`grad_output_indices < num_indices ∧ cols < num_cols`, the kernel's
`grad_output_mask`) has `xs j` at its scatter cell
`ids j * stride0 + col * stride1`; and every memory cell off the raw scatter
cells is unchanged — unconditionally.

`WriteInj` here is **genuinely non-vacuous and genuinely required**: the
benchmark's own test cases pass repeated indices (`[0, 0, 0, 0, 0]`,
`[9, 9, 9, 9, 9]`), two lanes of one column block then scatter to the same cell,
and no pointwise readback can hold. It is the honest no-duplicate-destination
precondition and it lives inside `⊨`, so the headline itself stays free of
scope-narrowing hypotheses.

`hBC : 0 < BLOCK_SIZE_COL` is genuinely forced: the index-tile load runs over
the `[BLOCK_SIZE_INDEX]` tile regardless of `BLOCK_SIZE_COL`, so when
`BLOCK_SIZE_COL = 0` the flattened lane space is empty and carries no in-bounds
witness for it.

Proof: `GatherMasked2DKernelIO₁.Implements.intro` assembles the region-model
scatter triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification index_select_cat_bwd_kernel_correctness
    (grad_source_ptr index_ptr grad_output_ptr : RegionName)
    (num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (hBC : 0 < BLOCK_SIZE_COL) :
    indexSelectBwdIO grad_source_ptr index_ptr grad_output_ptr num_rows
      num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL ⊨
        fun _ _ _ xs j => xs j
```

**Assumptions / layout contracts:**
- `hBC : 0 < BLOCK_SIZE_COL`

**Closed-form spec defs (transitive):** `indexSelectBwdIO`, `index_select_cat_bwd_kernel`

<details><summary><code>indexSelectBwdIO</code></summary>

```
/-- `index_select_cat_bwd_kernel`'s scatter **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `inp`/`idxbuf`/`out` — which buffer is which argument (incoming gradient
  `grad_output_ptr`, `.nat` row-index channel `index_ptr`, scatter target
  `grad_source_ptr`);
* `B = BLOCK_SIZE_INDEX * BLOCK_SIZE_COL` — the kernel's `[BI, BC]` tile,
  flattened: lane `j` is tile cell `(j / BC, j % BC)`;
* `readx` — the **static** index-tile window `pid0 * BI + j / BC`, i.e. the
  kernel's `grad_output_indices`;
* `read` — the **static** data window
  `(pid0 * BI + j / BC) * stride0 + (pid1 * BC + j % BC) * stride1`: the
  kernel's `grad_output_offsets`, which ignores the loaded index tile;
* `write` — the **data-dependent** scatter destination
  `ids j * stride0 + (pid1 * BC + j % BC) * stride1`: the loaded row times
  `stride0` plus the column offset, exactly the kernel's
  `grad_source_offsets`;
* `mask` — the index-tile gate `pid0 * BI + j / BC < num_indices`;
* `readMask`/`writeMask` — the kernel's `grad_output_mask`,
  `grad_output_indices < num_indices ∧ cols < num_cols`.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing, masking and index-channel plumbing
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the active lanes and the
write-active scatter cells. -/
```
```lean
def indexSelectBwdIO (grad_source_ptr index_ptr grad_output_ptr : RegionName)
    (num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat) : GatherMasked2DKernelIO₁ where
  kernel := index_select_cat_bwd_kernel grad_source_ptr index_ptr grad_output_ptr
    num_rows num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL
  inp := grad_output_ptr
  idxbuf := index_ptr
  out := grad_source_ptr
  B := BLOCK_SIZE_INDEX * BLOCK_SIZE_COL
  readx := fun p₀ _ j => p₀ * BLOCK_SIZE_INDEX + j.val / BLOCK_SIZE_COL
  read := fun p₀ p₁ _ j =>
    (p₀ * BLOCK_SIZE_INDEX + j.val / BLOCK_SIZE_COL) * stride0 +
      (p₁ * BLOCK_SIZE_COL + j.val % BLOCK_SIZE_COL) * stride1
  write := fun _ p₁ ids j =>
    ids j * stride0 + (p₁ * BLOCK_SIZE_COL + j.val % BLOCK_SIZE_COL) * stride1
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

<details><summary><code>index_select_cat_bwd_kernel</code></summary>

```
/-- Faithful transcription of `index_select_bwd.py`'s
`index_select_cat_bwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE_INDEX: tl.constexpr` / `BLOCK_SIZE_COL: tl.constexpr`
  -> Lean `Nat` parameters.
- Python `.to(tl.float32)` on `grad_output` is represented explicitly in the
  Compute layer; the algorithm-layer theorem observes its Real projection. -/
```
```lean
def index_select_cat_bwd_kernel
    (grad_source_ptr : RegionName) (index_ptr : Region .nat) (grad_output_ptr : RegionName)
    (_num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat) :
    ComputeKernel := triton {
  pid0 = tl.program_id(axis=0)
  pid1 = tl.program_id(axis=1)
  cols = pid1 * $(BLOCK_SIZE_COL) + tl.arange(0, $(BLOCK_SIZE_COL))
  grad_output_indices = pid0 * $(BLOCK_SIZE_INDEX) + tl.arange(0, $(BLOCK_SIZE_INDEX))
  grad_output_offsets =
    grad_output_ptr + grad_output_indices[:, None] * $(stride0) +
      cols[None, :] * $(stride1)
  grad_output_mask =
    (grad_output_indices[:, None] < $(num_indices)) &
      (cols[None, :] < $(num_cols))
  grad_output = (tl.load(grad_output_offsets, mask=grad_output_mask)).to(tl.float32)
  grad_source_indices =
    tl.load(index_ptr + grad_output_indices,
      mask=grad_output_indices < $(num_indices))
  grad_source_offsets =
    grad_source_ptr + grad_source_indices[:, None] * $(stride0) +
      cols[None, :] * $(stride1)
  tl.store(grad_source_offsets, grad_output, mask=grad_output_mask)
}
```
</details>
