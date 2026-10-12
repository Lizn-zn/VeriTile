# Spec sheet — `bench/tritonbench_g/adam_update_triton/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/adam_update_triton/adam_update_triton.py`

## Public theorem: `update_commuted_equiv`

<details><summary>docstring</summary>

```
/-- Admitted add_commute preserves both outputs and all surrounding execution.
Unlike the independent output formulas in Correct.lean, this contextual result
also allows output-region aliasing: both programs perform the same two stores. -/
```
</details>

**Statement:**
```lean
specification update_commuted_equiv (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n B : Nat) (R : Rules) :
    original p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B ≡[R]
      commuted p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B
```

**Closed-form spec defs (transitive):** `Rules`, `original`, `commuted`, `update_fn_kernel`, `additionSite`, `update_commuted_kernel`

<details><summary><code>Rules</code></summary>

```lean
abbrev Rules := FP.ScalarArithmetic.Rules
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n B : Nat) : Program where
  kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B
  domain := [additionSite p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B]
```
</details>

<details><summary><code>commuted</code></summary>

```lean
def commuted (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n B : Nat) : Program :=
  { original p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B with
    kernel := update_commuted_kernel p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B }
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

<details><summary><code>additionSite</code></summary>

```
/-- Check the actual computed diff * beta2 and loaded grad for finiteness,
including evaluated masked lanes. The first exp_avg assignment loads momentum;
the second is this update. Positivity and nonzero are not required. -/
```
```lean
def additionSite (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n B : Nat) : Site :=
  .add (Source.prefixBefore
    (update_fn_kernel p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B).surfaceBody
    "exp_avg" 1) [B]
    (.mul .real .scalarR (.ref .real [B] "diff") (.const beta2))
    (.ref .real [B] "grad")
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
