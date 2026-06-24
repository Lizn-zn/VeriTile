# Spec sheet — `bench/tritonbench_g/index_select_cat/IndexSelectCat.lean`

**Python source:** `bench/tritonbench_g/index_select_cat/index_select_cat.py`

## Public theorem: `index_select_cat_fwd_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `index_select_cat_fwd_kernel`: the DSL surface
lowers to the algorithm layer, and the contiguous store to `output_ptr` is
compute-correct — under the injectivity hypothesis `hOutInj`, every active cell
holds the gathered value `source[index[i], col]`. -/
```
</details>

**Statement:**
```lean
theorem index_select_cat_fwd_kernel_output_summary
    (output_ptr source_ptr index_ptr : RegionName)
    (num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL] =>
        outputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)) :
    (∃ alg, (index_select_cat_fwd_kernel output_ptr source_ptr index_ptr
        num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX
        BLOCK_SIZE_COL).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := index_select_cat_fwd_kernel output_ptr source_ptr index_ptr
        num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL)
        (fun idx => (output_ptr,
          outputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)))
      (expected := fun idx =>
        s.readMem source_ptr
          (sourceAddr s index_ptr stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx))
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL] =>
        outputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)`
- `kernel : = index_select_cat_fwd_kernel output_ptr source_ptr index_ptr
        num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL`
- `initialState : = s`
- `expected : = fun idx =>
        s.readMem source_ptr
          (sourceAddr s index_ptr stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)`

**Closed-form spec defs (transitive):** `outputAddr`, `index_select_cat_fwd_kernel`, `active`, `sourceAddr`, `indexBase`, `colBase`

<details><summary><code>outputAddr</code></summary>

```lean
def outputAddr (s : BlockState) (stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL]) : Nat :=
  indexBase s BLOCK_SIZE_INDEX idx.1 * stride0 +
    colBase s BLOCK_SIZE_COL idx.2.1 * stride1
```
</details>

<details><summary><code>index_select_cat_fwd_kernel</code></summary>

```
/-- Faithful transcription of `index_select_cat.py`'s
`index_select_cat_fwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE_INDEX: tl.constexpr` / `BLOCK_SIZE_COL: tl.constexpr`
  → Lean `Nat` parameters. -/
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

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL]) : Prop :=
  indexBase s BLOCK_SIZE_INDEX idx.1 < num_indices ∧
    colBase s BLOCK_SIZE_COL idx.2.1 < num_cols
```
</details>

<details><summary><code>sourceAddr</code></summary>

```lean
def sourceAddr (s : BlockState) (index_ptr : RegionName)
    (stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL]) : Nat :=
  s.readMemValue .nat index_ptr (indexBase s BLOCK_SIZE_INDEX idx.1) * stride0 +
    colBase s BLOCK_SIZE_COL idx.2.1 * stride1
```
</details>

<details><summary><code>indexBase</code></summary>

```lean
def indexBase (s : BlockState) (BLOCK_SIZE_INDEX : Nat) (i : Fin BLOCK_SIZE_INDEX) : Nat :=
  s.pids 0 * BLOCK_SIZE_INDEX + i.val
```
</details>

<details><summary><code>colBase</code></summary>

```lean
def colBase (s : BlockState) (BLOCK_SIZE_COL : Nat) (j : Fin BLOCK_SIZE_COL) : Nat :=
  s.pids 1 * BLOCK_SIZE_COL + j.val
```
</details>

## Also present (pinned special-case summaries)
- `index_select_cat_fwd_kernel_compute_correct`
