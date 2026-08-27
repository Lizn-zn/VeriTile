# Spec sheet — `bench/tritonbench_g/cross_entropy1/CrossEntropy1.lean`

**Python source:** `bench/tritonbench_g/cross_entropy1/cross_entropy1.py`

## Public theorem: `cross_entropy_fwd_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `cross_entropy_fwd_kernel` implements the pure
per-program cross-entropy pair on its metadata-genre IO signature — for
every disjoint flat placement of the four buffers, every program
`(row, col_block)` whose declared cells/lanes are in bounds, and every launch
state pinning the label `lab` at the slot cell, the block logits `xs` on the
active lanes, and the gather cell `g` under its gate, the translated pointer
kernel terminates and writes

* `loss_ptr[col_block·n_rows + row] = ceLossLocal … lab xs g` — the faithful
  five-way loss (ignored label / label-in-block / `HAS_SMOOTHING` / `SPLIT` /
  `lse²` term) over the pinned inputs, and
* `lse_ptr[col_block·n_rows + row] = ceBlockLSE … xs` — the log-sum-exp of
  the active block lanes,

for in-grid programs (`col_block·B < n_cols`); programs whose block lies past
the row end write the IEEE-faithful `⊥`-path fallback `0` to both cells.
Every other memory cell is unchanged. `0 < BLOCK_SIZE` is required (the `max`
reduce needs a lane); the two output buffers must differ (`lse_ptr ≠
loss_ptr`) so each readback sees through the other store, and `lse_ptr ≠
logits_ptr` so the LSE store does not clobber the gather cell it reads.
Proof: `MetaGatherMasked2DKernelIO₂ₓ₂.Implements.intro` assembles the region-model
masked triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification cross_entropy_fwd_correctness
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool)
    (hB : 0 < BLOCK_SIZE)
    (hne : lse_ptr ≠ loss_ptr) (hLL : lse_ptr ≠ logits_ptr) :
    crossEntropyFwdIO loss_ptr lse_ptr logits_ptr labels_ptr smoothing
        lse_square_scale ignored_index total_classes class_start_idx n_cols
        n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT ⊨
      fun _ pid₁ lab xs g =>
        if pid₁ * BLOCK_SIZE < n_cols then
          (ceLossLocal n_cols total_classes BLOCK_SIZE smoothing
             lse_square_scale ignored_index class_start_idx HAS_SMOOTHING SPLIT
             pid₁ lab xs g,
           ceBlockLSE n_cols BLOCK_SIZE pid₁ xs)
        else (0, 0)
```

**Assumptions / layout contracts:**
- `hB : 0 < BLOCK_SIZE`
- `hne : lse_ptr ≠ loss_ptr`
- `hLL : lse_ptr ≠ logits_ptr`

**Closed-form spec defs (transitive):** `crossEntropyFwdIO`, `ceLossLocal`, `ceBlockLSE`, `cross_entropy_fwd_surface`, `ceBlockSum`

<details><summary><code>crossEntropyFwdIO</code></summary>

```
/-- `cross_entropy_fwd_surface`'s metadata-genre **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `mbufL` — the `.int` label slot: program `(row, col_block)` loads
  `labels_ptr[row]` (`mwinL`);
* `inp` — the logits matrix, read twice: the masked row block
  (`read`/`mask`: lane `j` at `row·stride + col_block·B + j`, active while
  `col_block·B + j < n_cols`) and the label-gated single-cell gather
  (`gwin`/`gmask`: cell `row·stride + (lab − class_start_idx)`, read exactly
  when the label is live and its shifted position falls in this block);
* `out1`/`out2` — the loss and LSE cells, both at `col_block·n_rows + row`
  (the host's `(n_splits, n_rows)` layout), written unconditionally
  (`writeMask` defaults).

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing, gating, and masking match them. -/
```
```lean
def crossEntropyFwdIO
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool) : MetaGatherMasked2DKernelIO₂ₓ₂ where
  kernel := cross_entropy_fwd_surface loss_ptr lse_ptr logits_ptr labels_ptr
    smoothing lse_square_scale ignored_index total_classes class_start_idx
    n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT
  mbufL := Region.cast labels_ptr
  inp := logits_ptr
  out1 := loss_ptr
  out2 := lse_ptr
  B := BLOCK_SIZE
  mwinL := fun pid₀ _ => pid₀
  read := fun pid₀ pid₁ _ j =>
    pid₀ * logits_row_stride + (pid₁ * BLOCK_SIZE + j.val)
  mask := fun _ pid₁ _ j => pid₁ * BLOCK_SIZE + j.val < n_cols
  gwin := fun pid₀ _ lab =>
    pid₀ * logits_row_stride + (lab - class_start_idx).toNat
  gmask := fun _ pid₁ lab =>
    lab ≠ ignored_index ∧
    lab - class_start_idx ≥ (↑(pid₁ * BLOCK_SIZE) : Int) ∧
    lab - class_start_idx < (↑(min n_cols ((pid₁ + 1) * BLOCK_SIZE)) : Int)
  write1 := fun pid₀ pid₁ _ => pid₁ * n_rows + pid₀
  write2 := fun pid₀ pid₁ _ => pid₁ * n_rows + pid₀
```
</details>

<details><summary><code>ceLossLocal</code></summary>

```
/-- The kernel's five-way loss, as a pure function of the pinned inputs: the
loaded label `lab`, the block values `xs`, and the gather cell `g` (the label
logit, meaningful exactly on the in-block branch that reads it). Mirrors
`crossEntropyLossSpec` with every memory read replaced by its pinned value. -/
```
```lean
noncomputable def ceLossLocal (n_cols total_classes B : Nat)
    (smoothing lse_square_scale : ℝ) (ignored_index class_start_idx : Int)
    (HAS_SMOOTHING SPLIT : Bool)
    (pid₁ : Nat) (lab : Int) (xs : Fin B → ℝ) (g : ℝ) : ℝ :=
  if lab = ignored_index then 0 else
    let lblShift : Int := lab - class_start_idx
    let lse : ℝ := ceBlockLSE n_cols B pid₁ xs
    let lseTerm : ℝ := if SPLIT then 0 else lse
    let sq : ℝ := if SPLIT then 0 else lse_square_scale * lse * lse
    let core : ℝ :=
      if (lblShift ≥ (pid₁ * B : Nat)) ∧
         (lblShift < (min n_cols ((pid₁ + 1) * B) : Nat)) then
        if HAS_SMOOTHING then
          lseTerm - smoothing * ceBlockSum n_cols B pid₁ xs / total_classes
            - (1 - smoothing) * g
        else
          lseTerm - g
      else
        if HAS_SMOOTHING then
          smoothing * (lseTerm - ceBlockSum n_cols B pid₁ xs / total_classes)
        else 0
    core + sq
```
</details>

<details><summary><code>ceBlockLSE</code></summary>

```
/-- Pure block log-sum-exp over the active lanes (`pid₁·B + i < n_cols`) of a
`B`-lane block: the plain (shift-free) form `log (∑ exp xᵢ)`; the stable
kernel form `partialLSE_full` collapses to it via
`partialLSE_full_eq_blockLSE`. -/
```
```lean
noncomputable def ceBlockLSE (n_cols B pid₁ : Nat) (xs : Fin B → ℝ) : ℝ :=
  Real.log (∑ i ∈ Finset.univ.filter (fun i : Fin B => pid₁ * B + i.val < n_cols),
    Real.exp (xs i))
```
</details>

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

<details><summary><code>ceBlockSum</code></summary>

```
/-- Pure masked block sum: the kernel's
`sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0.0))` over the
pinned block values. -/
```
```lean
noncomputable def ceBlockSum (n_cols B pid₁ : Nat) (xs : Fin B → ℝ) : ℝ :=
  ∑ i : Fin B, if pid₁ * B + i.val < n_cols then xs i else 0
```
</details>

## Also present (pinned special-case summaries)
- `cross_entropy_bwd_store_slice_compute_correct`
