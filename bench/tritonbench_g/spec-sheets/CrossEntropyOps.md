# Spec sheet — `bench/tritonbench_g/cross_entropy_ops/CrossEntropyOps.lean`

**Python source:** `bench/tritonbench_g/cross_entropy_ops/cross_entropy_ops.py`

## Public theorem: `cross_entropy_bwd_store_slice_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the final masked `dlogits` store. -/
```
</details>

**Statement:**
```lean
theorem cross_entropy_bwd_store_slice_compute_correct
    (dlogits_ptr dloss_ptr Probs : RegionName)
    (n_cols dlogits_row_stride dloss_row_stride probs_row_stride BLOCK_SIZE : Nat)
    (logit_scale : ℝ)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := cross_entropy_bwd_store_slice dlogits_ptr dloss_ptr Probs
        n_cols dlogits_row_stride dloss_row_stride probs_row_stride BLOCK_SIZE
        logit_scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active s n_cols BLOCK_SIZE i)
        (fun i => (dlogits_ptr, outOffset s dlogits_row_stride BLOCK_SIZE i)))
      (expected := fun i =>
        expectedGrad s dloss_ptr Probs dloss_row_stride probs_row_stride
          BLOCK_SIZE logit_scale i)
```

**Assumptions / layout contracts:**
- `fun i : Fin BLOCK_SIZE => active s n_cols BLOCK_SIZE i`

**Closed-form spec defs (transitive):** `cross_entropy_bwd_store_slice`, `active`, `outOffset`, `expectedGrad`, `colOffset`, `probsOffset`

<details><summary><code>cross_entropy_bwd_store_slice</code></summary>

```
/-- Proof-oriented final-store slice of `cross_entropy_ops.py`'s
`cross_entropy_bwd_kernel`.

The full kernel computes `probs` from logits/LSE/labels/smoothing. This slice
starts from a precomputed `Probs` row and proves the masked
`dlogits = (dloss * logit_scale) * probs` writeback. -/
```
```lean
def cross_entropy_bwd_store_slice
    (dlogits_ptr dloss_ptr Probs : RegionName)
    (n_cols dlogits_row_stride dloss_row_stride probs_row_stride BLOCK_SIZE : Nat)
    (logit_scale : ℝ) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_block_idx = tl.program_id(1)
  col_offsets = col_block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  dloss = tl.load(dloss_ptr + row_idx * $(dloss_row_stride))
  probs = tl.load(Probs + row_idx * $(probs_row_stride) + col_offsets,
    mask=col_offsets < $(n_cols), other=0.0)
  tl.store(dlogits_ptr + row_idx * $(dlogits_row_stride) + col_offsets,
    (dloss * $(logit_scale)) * probs, mask=col_offsets < $(n_cols))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (n_cols BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Prop :=
  colOffset s BLOCK_SIZE i < n_cols
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (dlogits_row_stride BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * dlogits_row_stride + colOffset s BLOCK_SIZE i
```
</details>

<details><summary><code>expectedGrad</code></summary>

```lean
noncomputable def expectedGrad
    (s : BlockState) (dloss_ptr Probs : RegionName)
    (dloss_row_stride probs_row_stride BLOCK_SIZE : Nat)
    (logit_scale : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  (s.readMem dloss_ptr (s.pids 0 * dloss_row_stride) * logit_scale) *
    s.readMem Probs (probsOffset s probs_row_stride BLOCK_SIZE i)
```
</details>

<details><summary><code>colOffset</code></summary>

```lean
def colOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 1 * BLOCK_SIZE + i.val
```
</details>

<details><summary><code>probsOffset</code></summary>

```lean
def probsOffset
    (s : BlockState) (probs_row_stride BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * probs_row_stride + colOffset s BLOCK_SIZE i
```
</details>

## Public theorem: `cross_entropy_lse_store_slice_compute_correct`

**Statement:**
```lean
theorem cross_entropy_lse_store_slice_compute_correct
    (LsePre lse_ptr : RegionName) (n_rows : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := cross_entropy_lse_store_slice LsePre lse_ptr n_rows)
      (initialState := s)
      (write := fun _ : PUnit => some (lse_ptr, lseOutOffset s n_rows))
      (expected := fun _ => lseStoreSpec s LsePre n_rows)
```

**Closed-form spec defs (transitive):** `cross_entropy_lse_store_slice`, `lseOutOffset`, `lseStoreSpec`

<details><summary><code>cross_entropy_lse_store_slice</code></summary>

```
/-- Proof-oriented LSE-store slice of `cross_entropy_ops.py`'s forward kernel.
Companion to the bwd_store_slice: takes a precomputed `LsePre` scalar and
proves the writeback into `lse_ptr`. -/
```
```lean
def cross_entropy_lse_store_slice
    (LsePre lse_ptr : RegionName) (n_rows : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_block_idx = tl.program_id(1)
  lse = tl.load(LsePre + col_block_idx * $(n_rows) + row_idx)
  tl.store(lse_ptr + col_block_idx * $(n_rows) + row_idx, lse)
}
```
</details>

<details><summary><code>lseOutOffset</code></summary>

```lean
def lseOutOffset (s : BlockState) (n_rows : Nat) : Nat :=
  s.pids 1 * n_rows + s.pids 0
```
</details>

<details><summary><code>lseStoreSpec</code></summary>

```lean
noncomputable def lseStoreSpec (s : BlockState) (LsePre : RegionName)
    (n_rows : Nat) : ℝ :=
  s.readMem LsePre (lseOutOffset s n_rows)
```
</details>

## Public theorem: `cross_entropy_loss_store_slice_compute_correct`

**Statement:**
```lean
theorem cross_entropy_loss_store_slice_compute_correct
    (LossPre loss_ptr : RegionName) (n_rows : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := cross_entropy_loss_store_slice LossPre loss_ptr n_rows)
      (initialState := s)
      (write := fun _ : PUnit => some (loss_ptr, lseOutOffset s n_rows))
      (expected := fun _ => lossStoreSpec s LossPre n_rows)
```

**Closed-form spec defs (transitive):** `cross_entropy_loss_store_slice`, `lseOutOffset`, `lossStoreSpec`

<details><summary><code>cross_entropy_loss_store_slice</code></summary>

```
/-- Proof-oriented loss-store slice of `cross_entropy_ops.py`'s forward kernel.
Same scalar-copy pattern as the LSE store slice. -/
```
```lean
def cross_entropy_loss_store_slice
    (LossPre loss_ptr : RegionName) (n_rows : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_block_idx = tl.program_id(1)
  loss = tl.load(LossPre + col_block_idx * $(n_rows) + row_idx)
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

<details><summary><code>lossStoreSpec</code></summary>

```lean
noncomputable def lossStoreSpec (s : BlockState) (LossPre : RegionName)
    (n_rows : Nat) : ℝ :=
  s.readMem LossPre (lseOutOffset s n_rows)
```
</details>

## Public theorem: `cross_entropy_z_loss_store_slice_compute_correct`

**Statement:**
```lean
theorem cross_entropy_z_loss_store_slice_compute_correct
    (ZLossPre z_loss_ptr : RegionName) (n_rows : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := cross_entropy_z_loss_store_slice ZLossPre z_loss_ptr n_rows)
      (initialState := s)
      (write := fun _ : PUnit => some (z_loss_ptr, lseOutOffset s n_rows))
      (expected := fun _ => zLossStoreSpec s ZLossPre n_rows)
```

**Closed-form spec defs (transitive):** `cross_entropy_z_loss_store_slice`, `lseOutOffset`, `zLossStoreSpec`

<details><summary><code>cross_entropy_z_loss_store_slice</code></summary>

```
/-- Proof-oriented z_loss-store slice of `cross_entropy_ops.py`'s forward kernel. -/
```
```lean
def cross_entropy_z_loss_store_slice
    (ZLossPre z_loss_ptr : RegionName) (n_rows : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_block_idx = tl.program_id(1)
  z_loss = tl.load(ZLossPre + col_block_idx * $(n_rows) + row_idx)
  tl.store(z_loss_ptr + col_block_idx * $(n_rows) + row_idx, z_loss)
}
```
</details>

<details><summary><code>lseOutOffset</code></summary>

```lean
def lseOutOffset (s : BlockState) (n_rows : Nat) : Nat :=
  s.pids 1 * n_rows + s.pids 0
```
</details>

<details><summary><code>zLossStoreSpec</code></summary>

```lean
noncomputable def zLossStoreSpec (s : BlockState) (ZLossPre : RegionName)
    (n_rows : Nat) : ℝ :=
  s.readMem ZLossPre (lseOutOffset s n_rows)
```
</details>
