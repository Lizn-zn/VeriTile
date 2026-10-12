# Spec sheet — `bench/tritonbench_g/cross_entropy_ops/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/cross_entropy_ops/cross_entropy_ops.py`

## Public theorem: `forward_index_commuted_equiv`

**Statement:**
```lean
specification forward_index_commuted_equiv :
    forwardOriginal storage loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT ≡[assumptions] forwardIndexCommuted storage loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT
```

**Closed-form spec defs (transitive):** `forwardOriginal`, `assumptions`, `forwardIndexCommuted`, `forward`

<details><summary><code>forwardOriginal</code></summary>

```lean
def forwardOriginal : Program where
  kernel := forward storage Bool.false loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []

section
variable (storage : Storage) (loss_ptr lse_ptr z_loss_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing logit_scale lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat) (HAS_SMOOTHING SPLIT : Bool)
```
</details>

<details><summary><code>forwardIndexCommuted</code></summary>

```lean
def forwardIndexCommuted : Program where
  kernel := forward storage Bool.true loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT
  domain := []
```
</details>

<details><summary><code>forward</code></summary>

```lean
def forward (storage : Storage) (swapped : Bool) := match storage with
  | .fp16 => if swapped then crossEntropyOpsFwdSource%[tl.float16,indexCommuted] else crossEntropyOpsFwdSource%[tl.float16,original]
  | .bf16 => if swapped then crossEntropyOpsFwdSource%[tl.bfloat16,indexCommuted] else crossEntropyOpsFwdSource%[tl.bfloat16,original]
  | .fp32 => if swapped then crossEntropyOpsFwdSource%[tl.float32,indexCommuted] else crossEntropyOpsFwdSource%[tl.float32,original]

/-! Source: cross_entropy_ops.py, cross_entropy_bwd_kernel. -/
syntax "crossEntropyOpsBwdSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(crossEntropyOpsBwdSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let colIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then
        `(tritonStmt| col_offsets = $(BLOCK_SIZE) * col_block_idx + tl.arange(0, $(BLOCK_SIZE)))
      else `(tritonStmt| col_offsets = col_block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE)))
    `(fun (dlogits_ptr dloss_ptr logits_ptr lse_ptr : RegionName) (labels_ptr : Region .int)
        (smoothing logit_scale lse_square_scale : ℝ) (ignored_index : Int)
        (total_classes : Nat) (class_start_idx : Int)
        (n_cols logits_row_stride dlogits_row_stride dloss_row_stride BLOCK_SIZE : Nat)
        (HAS_SMOOTHING : Bool) => ComputeKernel.mk
      [dloss_ptr,logits_ptr,lse_ptr,Region.cast labels_ptr] [dlogits_ptr] (triton {
  tl.region logits_ptr = $storage, dlogits_ptr = $storage,
    dloss_ptr = tl.float32, lse_ptr = tl.float32, z_loss_ptr = tl.float32
  row_idx = tl.program_id(0)
  col_block_idx = tl.program_id(1)
  logits_ptr = logits_ptr + row_idx * ($(logits_row_stride)).to(tl.int64)
  dlogits_ptr = dlogits_ptr + row_idx * ($(dlogits_row_stride)).to(tl.int64)
  $colIndex
  label_idx = tl.load(labels_ptr + row_idx)
  if label_idx != $((ignored_index : Int)) {
    dloss = tl.load(dloss_ptr + row_idx * $(dloss_row_stride))
  } else {
    dloss = 0.0
  }
  logits = tl.load(logits_ptr + col_offsets,
    mask=col_offsets < $(n_cols), other=-float("inf")).to(tl.float32) * $((logit_scale : ℝ))
  lse = tl.load(lse_ptr + row_idx)
  probs = tl.exp(logits - lse)
  probs += 2.0 * $((lse_square_scale : ℝ)) * lse * probs
  label_idx -= $((class_start_idx : Int))
  if HAS_SMOOTHING {
    smooth_positive = 1.0 - $((smoothing : ℝ))
    smooth_negative = $((smoothing : ℝ)) / $(total_classes)
    probs = tl.where(col_offsets == label_idx, probs - (1 - $((smoothing : ℝ))), probs) - smooth_negative
  } else {
    probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
  }
  tl.store(dlogits_ptr + col_offsets, ((dloss * $(logit_scale)) * probs).to($storage:tritonDType), mask=col_offsets < $(n_cols))
}).surfaceBody)
```
</details>

## Public theorem: `backward_index_commuted_equiv`

**Statement:**
```lean
specification backward_index_commuted_equiv :
    backwardOriginal storage dlogits_ptr dloss_ptr logits_ptr lse_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols logits_row_stride dlogits_row_stride dloss_row_stride BLOCK_SIZE HAS_SMOOTHING ≡[assumptions] backwardIndexCommuted storage dlogits_ptr dloss_ptr logits_ptr lse_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols logits_row_stride dlogits_row_stride dloss_row_stride BLOCK_SIZE HAS_SMOOTHING
```

**Closed-form spec defs (transitive):** `backwardOriginal`, `assumptions`, `backwardIndexCommuted`, `backward`

<details><summary><code>backwardOriginal</code></summary>

```lean
def backwardOriginal : Program where
  kernel := backward storage Bool.false dlogits_ptr dloss_ptr logits_ptr lse_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols logits_row_stride dlogits_row_stride dloss_row_stride BLOCK_SIZE HAS_SMOOTHING
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []

section
variable (storage : Storage) (loss_ptr lse_ptr z_loss_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing logit_scale lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat) (HAS_SMOOTHING SPLIT : Bool)
```
</details>

<details><summary><code>backwardIndexCommuted</code></summary>

```lean
def backwardIndexCommuted : Program where
  kernel := backward storage Bool.true dlogits_ptr dloss_ptr logits_ptr lse_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols logits_row_stride dlogits_row_stride dloss_row_stride BLOCK_SIZE HAS_SMOOTHING
  domain := []
```
</details>

<details><summary><code>backward</code></summary>

```lean
def backward (storage : Storage) (swapped : Bool) := match storage with
  | .fp16 => if swapped then crossEntropyOpsBwdSource%[tl.float16,indexCommuted] else crossEntropyOpsBwdSource%[tl.float16,original]
  | .bf16 => if swapped then crossEntropyOpsBwdSource%[tl.bfloat16,indexCommuted] else crossEntropyOpsBwdSource%[tl.bfloat16,original]
  | .fp32 => if swapped then crossEntropyOpsBwdSource%[tl.float32,indexCommuted] else crossEntropyOpsBwdSource%[tl.float32,original]
```
</details>
