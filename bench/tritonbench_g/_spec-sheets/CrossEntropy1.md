# Spec sheet — `bench/tritonbench_g/cross_entropy1/CrossEntropy1.lean`

**Python source:** `bench/tritonbench_g/cross_entropy1/cross_entropy1.py`

## Public theorem: `cross_entropy_fwd_output_summary`

<details><summary>docstring</summary>

```
/-- **Per-kernel forward output summary for `cross_entropy_fwd_surface`
(genuine, end-to-end).**

For any execution `exec ... s = some s'` (with `lse_ptr ≠ loss_ptr`,
`lse_ptr ≠ logits_ptr`, and at least one valid lane), bundles:
1. the full forward surface lowers to the algorithm layer (`toAlgorithm? = ok`);
2. **genuine LSE side output**: `lse_ptr[col_block·n_rows + row]` holds exactly the
   masked-lane stable log-sum-exp `partialLSE_full` of the INPUT block logits;
3. **genuine loss output**: `loss_ptr[col_block·n_rows + row]` holds exactly the
   faithful five-way cross-entropy `crossEntropyLossSpec`
   (`label==ignored` / label-in-block / `HAS_SMOOTHING` / `SPLIT` / `lse²`), with
   every sub-term (`sum_logits`, the data-dependent `logits_label`, `lse`) read
   from INPUT memory.

Both value specs read INPUT memory, never `exec(...).readMem`, so this summary is
non-self-referential. The region-distinctness hypotheses are the only framing
side-conditions (the LSE store must not overwrite the logits it just read, and
the two side outputs are distinct buffers). -/
```
</details>

**Statement:**
```lean
specification cross_entropy_fwd_output_summary
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride : Nat) (n : Nat)
    (HAS_SMOOTHING SPLIT : Bool)
    (s : BlockState)
    (h_tail : s.pids 1 * (n+1) < n_cols)
    (hne : lse_ptr ≠ loss_ptr)
    (hLL : lse_ptr ≠ logits_ptr) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := cross_entropy_fwd_surface loss_ptr lse_ptr logits_ptr labels_ptr
        smoothing lse_square_scale ignored_index total_classes class_start_idx
        n_cols n_rows logits_row_stride (n+1) HAS_SMOOTHING SPLIT)
      (initialState := s)
      (write := fun _ : PUnit => some (lse_ptr, lseOutOffset s n_rows))
      (expected := fun _ =>
        partialLSE_full (n := n) (rowLogits s logits_ptr logits_row_stride n_cols)
          (s.pids 1) h_tail Bool.false 0)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := cross_entropy_fwd_surface loss_ptr lse_ptr logits_ptr labels_ptr
        smoothing lse_square_scale ignored_index total_classes class_start_idx
        n_cols n_rows logits_row_stride (n+1) HAS_SMOOTHING SPLIT)
      (initialState := s)
      (write := fun _ : PUnit => some (loss_ptr, lseOutOffset s n_rows))
      (expected := fun _ =>
        crossEntropyLossSpec s logits_ptr (labelValue s labels_ptr) smoothing
          lse_square_scale ignored_index total_classes class_start_idx n_cols
          logits_row_stride n HAS_SMOOTHING SPLIT
          (partialLSE_full (n := n) (rowLogits s logits_ptr logits_row_stride n_cols)
            (s.pids 1) h_tail Bool.false 0)))
```

**Assumptions / layout contracts:**
- `h_tail : s.pids 1 * (n+1) < n_cols`
- `hne : lse_ptr ≠ loss_ptr`
- `hLL : lse_ptr ≠ logits_ptr`

**Closed-form spec defs (transitive):** `cross_entropy_fwd_surface`, `lseOutOffset`, `rowLogits`, `crossEntropyLossSpec`, `labelValue`, `blockSumLogits`, `labelLogit`

<details><summary><code>cross_entropy_fwd_surface</code></summary>

```
/-- Faithful transcription of `cross_entropy1.py`'s
`cross_entropy_fwd_kernel`.

This preserves the block logits load, optional smoothing sum, LSE side store,
label-in-block loss selection, optional split behavior, and LSE-square term. -/
```
```lean
def cross_entropy_fwd_surface
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ)
    (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_block_idx = tl.program_id(1)
  logits_ptr = logits_ptr + row_idx * ($(logits_row_stride)).to(tl.int64)
  col_offsets = col_block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  label_idx = tl.load(labels_ptr + row_idx)
  logits = tl.load(logits_ptr + col_offsets,
    mask=col_offsets < $(n_cols), other=-float("inf")).to(tl.float32)
  max_logits = tl.max(logits, 0)
  if HAS_SMOOTHING {
    sum_logits = tl.sum(tl.where(col_offsets < $(n_cols), logits, 0.0), 0)
  }
  lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
  tl.store(lse_ptr + col_block_idx * $(n_rows) + row_idx, lse)
  if label_idx == $((ignored_index : Int)) {
    loss = 0.0
  } else {
    label_idx -= $((class_start_idx : Int))
    if (label_idx >= col_block_idx * $(BLOCK_SIZE)) and
        (label_idx < min($(n_cols), (col_block_idx + $(1)) * $(BLOCK_SIZE))) {
      logits_label = tl.load(logits_ptr + label_idx)
      if HAS_SMOOTHING {
        loss = (lse if not SPLIT else 0.0) -
          $(smoothing) * sum_logits / $(total_classes) -
          (1.0 - $(smoothing)) * logits_label
      } else {
        loss = (lse if not SPLIT else 0.0) - logits_label
      }
    } else {
      if HAS_SMOOTHING {
        loss = $(smoothing) *
          ((lse if not SPLIT else 0.0) - sum_logits / $(total_classes))
      } else {
        loss = 0.0
      }
    }
    if not SPLIT {
      loss += $(lse_square_scale) * lse * lse
    }
  }
  tl.store(loss_ptr + col_block_idx * $(n_rows) + row_idx, loss)
}
```
</details>

<details><summary><code>lseOutOffset</code></summary>

```lean
def lseOutOffset (s : BlockState) (n_rows : Nat) : Nat :=
  s.pids 1 * n_rows + s.pids 0
```
</details>

<details><summary><code>rowLogits</code></summary>

```
/-- Row-logits function for program `row_idx`: position `j` reads INPUT memory
`logits_ptr` at `row_idx * logits_row_stride + j`. The forward kernel's
`col_block_idx`-th tile views the contiguous block `[col_block_idx·BLOCK_SIZE,
(col_block_idx+1)·BLOCK_SIZE)` of this row. -/
```
```lean
noncomputable def rowLogits
    (s : BlockState) (logits_ptr : RegionName)
    (logits_row_stride n_cols : Nat) (j : Fin n_cols) : ℝ :=
  s.readMem logits_ptr (s.pids 0 * logits_row_stride + j.val)
```
</details>

<details><summary><code>crossEntropyLossSpec</code></summary>

```
/-- The genuine `loss` value computed by `cross_entropy_fwd_surface` for program
`(row_idx, col_block_idx)`, a faithful Lean transcription of the kernel's
five-way branch over `label_idx`/in-block/`HAS_SMOOTHING`/`SPLIT`/`lse²`. All
sub-terms read INPUT memory; `lse` is the genuine `partialLSE_full`. -/
```
```lean
noncomputable def crossEntropyLossSpec
    (s : BlockState) (logits_ptr : RegionName) (labelVal : Int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols logits_row_stride : Nat) (n : Nat)
    (HAS_SMOOTHING SPLIT : Bool)
    (lse : ℝ) : ℝ :=
  if labelVal = ignored_index then 0 else
    let lblShift : Int := labelVal - class_start_idx
    let lseTerm : ℝ := if SPLIT then 0 else lse
    let sq : ℝ := if SPLIT then 0 else lse_square_scale * lse * lse
    let core : ℝ :=
      if (lblShift ≥ (s.pids 1 * (n+1) : Nat)) ∧
         (lblShift < (min n_cols ((s.pids 1 + 1) * (n+1)) : Nat)) then
        if HAS_SMOOTHING then
          lseTerm - smoothing * blockSumLogits s logits_ptr logits_row_stride n_cols n
            / total_classes
            - (1 - smoothing) * labelLogit s logits_ptr logits_row_stride lblShift
        else
          lseTerm - labelLogit s logits_ptr logits_row_stride lblShift
      else
        if HAS_SMOOTHING then
          smoothing * (lseTerm - blockSumLogits s logits_ptr logits_row_stride n_cols n
            / total_classes)
        else 0
    core + sq
```
</details>

<details><summary><code>labelValue</code></summary>

```
/-- The label value loaded by the kernel: `label_idx = tl.load(labels_ptr +
row_idx)` from INPUT memory. -/
```
```lean
noncomputable def labelValue (s : BlockState) (labels_ptr : Region .int) : Int :=
  s.readMemValue .int (Region.cast labels_ptr) (s.pids 0)
```
</details>

<details><summary><code>blockSumLogits</code></summary>

```
/-- The kernel's `sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0))`:
the sum of in-range block logits, read from INPUT memory. Out-of-range lanes
contribute `0`. -/
```
```lean
noncomputable def blockSumLogits
    (s : BlockState) (logits_ptr : RegionName)
    (logits_row_stride n_cols : Nat) (n : Nat) : ℝ :=
  ∑ i : Fin (n+1),
    if h : s.pids 1 * (n+1) + i.val < n_cols then
      rowLogits s logits_ptr logits_row_stride n_cols ⟨s.pids 1 * (n+1) + i.val, h⟩
    else 0
```
</details>

<details><summary><code>labelLogit</code></summary>

```
/-- The label logit `logits_label = tl.load(logits_ptr + (label_idx -
class_start_idx))`, read from INPUT memory at the shifted label position. -/
```
```lean
noncomputable def labelLogit
    (s : BlockState) (logits_ptr : RegionName)
    (logits_row_stride : Nat) (lblShift : Int) : ℝ :=
  s.readMem logits_ptr (s.pids 0 * logits_row_stride + lblShift.toNat)
```
</details>

## Also present (pinned special-case summaries)
- `cross_entropy_bwd_store_slice_compute_correct`
