# Spec sheet — `bench/tritonbench_g/index_select_bwd/IndexSelectBwd.lean`

**Python source:** `bench/tritonbench_g/index_select_bwd/index_select_bwd.py`

## Public theorem: `index_select_cat_bwd_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `index_select_cat_bwd_kernel`: the DSL surface
lowers to the algorithm layer, and the indexed scatter into `grad_source` is
compute-correct — under the no-duplicate-destination hypothesis `hStoreInj`, every
active cell writes `grad_output[row, col]` to `grad_source[index[row], col]`. -/
```
</details>

**Statement:**
```lean
theorem index_select_cat_bwd_kernel_output_summary
    (grad_source_ptr : RegionName) (index_ptr : Region .nat) (grad_output_ptr : RegionName)
    (num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (s : BlockState)
    (hStoreInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL] =>
        gradSourceStoreAddr s index_ptr num_indices stride0 stride1
          BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)) :
    (∃ alg, (index_select_cat_bwd_kernel grad_source_ptr index_ptr grad_output_ptr
        num_rows num_indices num_cols stride0 stride1
        BLOCK_SIZE_INDEX BLOCK_SIZE_COL).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := index_select_cat_bwd_kernel grad_source_ptr index_ptr grad_output_ptr
        num_rows num_indices num_cols stride0 stride1
        BLOCK_SIZE_INDEX BLOCK_SIZE_COL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL)
        (fun idx => (grad_source_ptr,
          gradSourceStoreAddr s index_ptr num_indices stride0 stride1
            BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)))
      (expected := fun idx =>
        s.readMem grad_output_ptr
          (gradOutputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx))
```

**Assumptions / layout contracts:**
- `hStoreInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL] =>
        gradSourceStoreAddr s index_ptr num_indices stride0 stride1
          BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)`

**Closed-form spec defs (transitive):** `gradSourceStoreAddr`, `index_select_cat_bwd_kernel`, `active`, `gradOutputAddr`, `outputIndex`, `colIndex`

<details><summary><code>gradSourceStoreAddr</code></summary>

```lean
def gradSourceStoreAddr (s : BlockState) (index_ptr : RegionName)
    (num_indices stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL]) : Nat :=
  (if outputIndex s BLOCK_SIZE_INDEX idx.1 < num_indices then
      s.readMemValue .nat index_ptr (outputIndex s BLOCK_SIZE_INDEX idx.1) * stride0
    else
      BlockState.defaultCarrier TileDType.nat * stride0) +
    colIndex s BLOCK_SIZE_COL idx.2.1 * stride1
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

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL]) : Prop :=
  outputIndex s BLOCK_SIZE_INDEX idx.1 < num_indices ∧
    colIndex s BLOCK_SIZE_COL idx.2.1 < num_cols
```
</details>

<details><summary><code>gradOutputAddr</code></summary>

```lean
def gradOutputAddr (s : BlockState)
    (stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL]) : Nat :=
  outputIndex s BLOCK_SIZE_INDEX idx.1 * stride0 +
    colIndex s BLOCK_SIZE_COL idx.2.1 * stride1
```
</details>

<details><summary><code>outputIndex</code></summary>

```lean
def outputIndex (s : BlockState) (BLOCK_SIZE_INDEX : Nat)
    (i : Fin BLOCK_SIZE_INDEX) : Nat :=
  s.pids 0 * BLOCK_SIZE_INDEX + i.val
```
</details>

<details><summary><code>colIndex</code></summary>

```lean
def colIndex (s : BlockState) (BLOCK_SIZE_COL : Nat)
    (j : Fin BLOCK_SIZE_COL) : Nat :=
  s.pids 1 * BLOCK_SIZE_COL + j.val
```
</details>

## Also present (pinned special-case summaries)
- `index_select_cat_bwd_kernel_compute_correct`
