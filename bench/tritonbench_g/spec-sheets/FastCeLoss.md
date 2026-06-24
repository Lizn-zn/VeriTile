# Spec sheet — `bench/tritonbench_g/fast_ce_loss/FastCeLoss.lean`

**Python source:** `bench/tritonbench_g/fast_ce_loss/fast_ce_loss.py`

## Public theorem: `cross_entropy_backward_store_slice_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the masked backward writeback. -/
```
</details>

**Statement:**
```lean
theorem cross_entropy_backward_store_slice_compute_correct
    (logits_ptr dloss_ptr Grad : RegionName)
    (VOCAB_SIZE logits_row_stride dloss_row_stride grad_row_stride
      BLOCK_SIZE : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := cross_entropy_backward_store_slice logits_ptr dloss_ptr Grad
        VOCAB_SIZE logits_row_stride dloss_row_stride grad_row_stride BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active s VOCAB_SIZE BLOCK_SIZE i)
        (fun i => (logits_ptr, logitsOffset s logits_row_stride BLOCK_SIZE i)))
      (expected := fun i =>
        expectedBackward s dloss_ptr Grad dloss_row_stride grad_row_stride
          BLOCK_SIZE i)
```

**Assumptions / layout contracts:**
- `fun i : Fin BLOCK_SIZE => active s VOCAB_SIZE BLOCK_SIZE i`

**Closed-form spec defs (transitive):** `cross_entropy_backward_store_slice`, `active`, `logitsOffset`, `expectedBackward`, `colOffset`, `gradOffset`

<details><summary><code>cross_entropy_backward_store_slice</code></summary>

```
/-- Proof-oriented backward final-store slice of `fast_ce_loss.py`'s
`_cross_entropy_backward`.

The full kernel builds `y` from logits, logsumexp, labels, optional logit
scaling, and optional softcapping. This slice starts from a precomputed gradient
tile `Grad` and proves the final masked in-place writeback
`logits[col_offsets] = dloss * Grad[col_offsets]`. -/
```
```lean
def cross_entropy_backward_store_slice
    (logits_ptr dloss_ptr Grad : RegionName)
    (VOCAB_SIZE logits_row_stride dloss_row_stride grad_row_stride
      BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  block_idx = tl.program_id(1)
  col_offsets = block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(VOCAB_SIZE)
  dloss = tl.load(dloss_ptr + row_idx * $(dloss_row_stride))
  y = tl.load(Grad + row_idx * $(grad_row_stride) + col_offsets,
    mask=mask, other=0.0)
  tl.store(logits_ptr + row_idx * $(logits_row_stride) + col_offsets,
    dloss * y, mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (VOCAB_SIZE BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Prop :=
  colOffset s BLOCK_SIZE i < VOCAB_SIZE
```
</details>

<details><summary><code>logitsOffset</code></summary>

```lean
def logitsOffset
    (s : BlockState) (logits_row_stride BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * logits_row_stride + colOffset s BLOCK_SIZE i
```
</details>

<details><summary><code>expectedBackward</code></summary>

```lean
noncomputable def expectedBackward
    (s : BlockState) (dloss_ptr Grad : RegionName)
    (dloss_row_stride grad_row_stride BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem dloss_ptr (s.pids 0 * dloss_row_stride) *
    s.readMem Grad (gradOffset s grad_row_stride BLOCK_SIZE i)
```
</details>

<details><summary><code>colOffset</code></summary>

```lean
def colOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 1 * BLOCK_SIZE + i.val
```
</details>

<details><summary><code>gradOffset</code></summary>

```lean
def gradOffset
    (s : BlockState) (grad_row_stride BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * grad_row_stride + colOffset s BLOCK_SIZE i
```
</details>

## Public theorem: `cross_entropy_lse_store_slice_compute_correct`

**Statement:**
```lean
theorem cross_entropy_lse_store_slice_compute_correct
    (LsumPre logsumexp_ptr : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := cross_entropy_lse_store_slice LsumPre logsumexp_ptr)
      (initialState := s)
      (write := fun _ : PUnit => some (logsumexp_ptr, lseOutOffset s))
      (expected := fun _ => lseStoreSpec s LsumPre)
```

**Closed-form spec defs (transitive):** `cross_entropy_lse_store_slice`, `lseOutOffset`, `lseStoreSpec`

<details><summary><code>cross_entropy_lse_store_slice</code></summary>

```
/-- Proof-oriented LSE store slice of `fast_ce_loss.py`'s
`_cross_entropy_forward`. Companion to the bwd_store_slice: takes a precomputed
`LsumPre` scalar (per row) and proves the unmasked writeback into
`logsumexp_ptr` at offset `row_idx`. -/
```
```lean
def cross_entropy_lse_store_slice
    (LsumPre logsumexp_ptr : RegionName) : ComputeKernel := triton {
  row_idx = tl.program_id(0)
  lsum = tl.load(LsumPre + row_idx)
  tl.store(logsumexp_ptr + row_idx, lsum)
}
```
</details>

<details><summary><code>lseOutOffset</code></summary>

```lean
def lseOutOffset (s : BlockState) : Nat := s.pid
```
</details>

<details><summary><code>lseStoreSpec</code></summary>

```lean
noncomputable def lseStoreSpec (s : BlockState) (LsumPre : RegionName) : ℝ :=
  s.readMem LsumPre s.pid
```
</details>

## Public theorem: `cross_entropy_loss_store_slice_compute_correct`

**Statement:**
```lean
theorem cross_entropy_loss_store_slice_compute_correct
    (LossPre loss_ptr : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := cross_entropy_loss_store_slice LossPre loss_ptr)
      (initialState := s)
      (write := fun _ : PUnit => some (loss_ptr, lseOutOffset s))
      (expected := fun _ => lossStoreSpec s LossPre)
```

**Closed-form spec defs (transitive):** `cross_entropy_loss_store_slice`, `lseOutOffset`, `lossStoreSpec`

<details><summary><code>cross_entropy_loss_store_slice</code></summary>

```
/-- Proof-oriented loss-store slice of `fast_ce_loss.py`'s forward kernel.
Same scalar-copy pattern as the LSE store slice: takes a precomputed
LossPre scalar (per row) and proves the writeback into loss_ptr. -/
```
```lean
def cross_entropy_loss_store_slice
    (LossPre loss_ptr : RegionName) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  loss = tl.load(LossPre + row_idx)
  tl.store(loss_ptr + row_idx, loss)
}
```
</details>

<details><summary><code>lseOutOffset</code></summary>

```lean
def lseOutOffset (s : BlockState) : Nat := s.pid
```
</details>

<details><summary><code>lossStoreSpec</code></summary>

```lean
noncomputable def lossStoreSpec (s : BlockState) (LossPre : RegionName) : ℝ :=
  s.readMem LossPre s.pid
```
</details>
