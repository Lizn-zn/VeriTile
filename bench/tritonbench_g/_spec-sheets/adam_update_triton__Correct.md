# Spec sheet — `bench/tritonbench_g/adam_update_triton/Correct.lean`

**Python source:** `bench/tritonbench_g/adam_update_triton/adam_update_triton.py`

## Public theorem: `update_fn_kernel_correctness`

<details><summary>docstring</summary>

```
/-- Both original outputs satisfy the Lion formulas over real arithmetic. -/
```
</details>

**Statement:**
```lean
specification update_fn_kernel_correctness
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (hRegions : p_ptr ≠ exp_avg_ptr) :
    Spec.Real (adamIO p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n_elements BLOCK_SIZE
      ⊨ fun p grad expAvg =>
        (fun i => TiledOptimizer.lionParam (p i) (expAvg i) (grad i) lr wd beta1,
         fun i => TiledOptimizer.lionMomentum (expAvg i) (grad i) beta2))
```

**Assumptions / layout contracts:**
- `hRegions : p_ptr ≠ exp_avg_ptr`

**Closed-form spec defs (transitive):** `adamIO`, `update_fn_kernel`

<details><summary><code>adamIO</code></summary>

```
/-- All three reads and both writes use the same symbolic block and tail mask.
The allocation list is separate from the in-place input/output roles. -/
```
```lean
def adamIO (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) :
    MaskedKernelIO₃ₓ₂ where
  kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr
    lr wd beta1 beta2 n_elements BLOCK_SIZE
  bufs := [p_ptr, grad_ptr, exp_avg_ptr]  -- p and exp_avg are updated in place
  in1 := p_ptr
  in2 := grad_ptr
  in3 := exp_avg_ptr
  out1 := p_ptr          -- = in1: in-place parameter update
  out2 := exp_avg_ptr    -- = in3: in-place momentum update
  B := BLOCK_SIZE
  read1 := fun pid => pid * BLOCK_SIZE
  read2 := fun pid => pid * BLOCK_SIZE
  read3 := fun pid => pid * BLOCK_SIZE
  write1 := fun pid => pid * BLOCK_SIZE
  write2 := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements
```
</details>

<details><summary><code>update_fn_kernel</code></summary>

```
/-- Faithful transcription of `adam_update_triton.py`'s `update_fn_kernel`. -/
```
```lean
def update_fn_kernel
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  offset_p_ptr = p_ptr + offsets
  offset_grad_ptr = grad_ptr + offsets
  offset_exp_avg_ptr = exp_avg_ptr + offsets
  p = tl.load(offset_p_ptr, mask=mask)
  grad = tl.load(offset_grad_ptr, mask=mask)
  exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)
  p = p * (1 - $((lr : ℝ)) * $((wd : ℝ)))
  diff = exp_avg - grad
  update = diff * $(beta1) + grad
  can_update = update != 0
  update_sign = tl.where(update > 0, -$((lr : ℝ)), $((lr : ℝ)))
  p = p + update_sign * can_update
  exp_avg = diff * $(beta2) + grad
  tl.store(offset_p_ptr, p, mask=mask)
  tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)
}
```
</details>

## Public theorem: `update_commuted_correctness`

<details><summary>docstring</summary>

```
/-- Both candidate outputs satisfy the same independent real formulas. -/
```
</details>

**Statement:**
```lean
specification update_commuted_correctness (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n B : Nat) (hRegions : p_ptr ≠ exp_avg_ptr) :
    Spec.Real (commutedIO p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B
      ⊨ fun p grad expAvg =>
        (fun i => TiledOptimizer.lionParam (p i) (expAvg i) (grad i) lr wd beta1,
         fun i => TiledOptimizer.lionMomentum (expAvg i) (grad i) beta2))
```

**Assumptions / layout contracts:**
- `hRegions : p_ptr ≠ exp_avg_ptr`

**Closed-form spec defs (transitive):** `commutedIO`, `adamIO`, `update_commuted_kernel`, `update_fn_kernel`

<details><summary><code>commutedIO</code></summary>

```
/-- The candidate retains the allocation list, in-place wiring and tail mask. -/
```
```lean
def commutedIO (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n B : Nat) : MaskedKernelIO₃ₓ₂ :=
  { adamIO p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B with
    kernel := update_commuted_kernel p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B
    projection := by rfl }
```
</details>

<details><summary><code>adamIO</code></summary>

```
/-- All three reads and both writes use the same symbolic block and tail mask.
The allocation list is separate from the in-place input/output roles. -/
```
```lean
def adamIO (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) :
    MaskedKernelIO₃ₓ₂ where
  kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr
    lr wd beta1 beta2 n_elements BLOCK_SIZE
  bufs := [p_ptr, grad_ptr, exp_avg_ptr]  -- p and exp_avg are updated in place
  in1 := p_ptr
  in2 := grad_ptr
  in3 := exp_avg_ptr
  out1 := p_ptr          -- = in1: in-place parameter update
  out2 := exp_avg_ptr    -- = in3: in-place momentum update
  B := BLOCK_SIZE
  read1 := fun pid => pid * BLOCK_SIZE
  read2 := fun pid => pid * BLOCK_SIZE
  read3 := fun pid => pid * BLOCK_SIZE
  write1 := fun pid => pid * BLOCK_SIZE
  write2 := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements
```
</details>

<details><summary><code>update_commuted_kernel</code></summary>

```
/-- Same update with `grad + diff * beta2` in the final momentum assignment. -/
```
```lean
def update_commuted_kernel
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  offset_p_ptr = p_ptr + offsets
  offset_grad_ptr = grad_ptr + offsets
  offset_exp_avg_ptr = exp_avg_ptr + offsets
  p = tl.load(offset_p_ptr, mask=mask)
  grad = tl.load(offset_grad_ptr, mask=mask)
  exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)
  p = p * (1 - $((lr : ℝ)) * $((wd : ℝ)))
  diff = exp_avg - grad
  update = diff * $(beta1) + grad
  can_update = update != 0
  update_sign = tl.where(update > 0, -$((lr : ℝ)), $((lr : ℝ)))
  p = p + update_sign * can_update
  exp_avg = grad + diff * $(beta2)
  tl.store(offset_p_ptr, p, mask=mask)
  tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)
}
```
</details>

<details><summary><code>update_fn_kernel</code></summary>

```
/-- Faithful transcription of `adam_update_triton.py`'s `update_fn_kernel`. -/
```
```lean
def update_fn_kernel
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  offset_p_ptr = p_ptr + offsets
  offset_grad_ptr = grad_ptr + offsets
  offset_exp_avg_ptr = exp_avg_ptr + offsets
  p = tl.load(offset_p_ptr, mask=mask)
  grad = tl.load(offset_grad_ptr, mask=mask)
  exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)
  p = p * (1 - $((lr : ℝ)) * $((wd : ℝ)))
  diff = exp_avg - grad
  update = diff * $(beta1) + grad
  can_update = update != 0
  update_sign = tl.where(update > 0, -$((lr : ℝ)), $((lr : ℝ)))
  p = p + update_sign * can_update
  exp_avg = diff * $(beta2) + grad
  tl.store(offset_p_ptr, p, mask=mask)
  tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)
}
```
</details>
