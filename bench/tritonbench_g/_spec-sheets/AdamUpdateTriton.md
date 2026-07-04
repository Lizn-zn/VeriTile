# Spec sheet — `bench/tritonbench_g/adam_update_triton/AdamUpdateTriton.lean`

**Python source:** `bench/tritonbench_g/adam_update_triton/adam_update_triton.py`

## Public theorem: `update_fn_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Single-program output summary for Python `update_fn`.

This is a local, per-`BlockState` correctness statement, not the whole-grid
launch theorem.  It says three things about executing one Triton program/block:

* the DSL surface for `update_fn_kernel` successfully lowers to the algorithm
  layer (`toAlgorithm? = Except.ok alg`);
* for every active lane `i : Fin BLOCK_SIZE`, where
  `linearOffset s BLOCK_SIZE i < n_elements` is the kernel mask
  `offsets < n_elements`, the store to
  `(p_ptr, linearOffset s BLOCK_SIZE i)` realizes `pFullSpec`, i.e. the reusable
  Lion parameter-update oracle applied to the values loaded by that lane;
* for the same active lanes, the store to
  `(exp_avg_ptr, linearOffset s BLOCK_SIZE i)` realizes `expAvgFullSpec`, i.e.
  the reusable Lion momentum oracle.

The side condition `p_ptr ≠ exp_avg_ptr` rules out aliasing between the two
output regions, so the second masked store cannot overwrite the first output.
Grid coverage, `cdiv n_elements BLOCK_SIZE`, and cross-program disjointness are
separate whole-grid obligations handled by the worked example
`VeriTile.Examples.AdamUpdateGridLaunch`, outside the scope of this per-kernel
theorem. -/
```
</details>

**Statement:**
```lean
theorem update_fn_kernel_output_summary
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (s : BlockState)
    (hRegions : p_ptr ≠ exp_avg_ptr) :
    (∃ alg, (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => linearOffset s BLOCK_SIZE i < n_elements)
        (fun i => (p_ptr, linearOffset s BLOCK_SIZE i)))
      (expected := fun i =>
        pFullSpec s p_ptr grad_ptr exp_avg_ptr lr wd beta1 BLOCK_SIZE i)) ∧
    (ComputeCorrect.Realizes
      (kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => linearOffset s BLOCK_SIZE i < n_elements)
        (fun i => (exp_avg_ptr, linearOffset s BLOCK_SIZE i)))
      (expected := fun i =>
        expAvgFullSpec s grad_ptr exp_avg_ptr beta2 BLOCK_SIZE i)))
```

**Assumptions / layout contracts:**
- `hRegions : p_ptr ≠ exp_avg_ptr`
- `fun i : Fin BLOCK_SIZE => linearOffset s BLOCK_SIZE i < n_elements`
- `fun i : Fin BLOCK_SIZE => linearOffset s BLOCK_SIZE i < n_elements`

**Closed-form spec defs (transitive):** `update_fn_kernel`, `pFullSpec`, `expAvgFullSpec`

<details><summary><code>update_fn_kernel</code></summary>

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

<details><summary><code>pFullSpec</code></summary>

```
/-- Per-lane `p` output spec: the reusable Lion parameter-update oracle applied
to the values this lane loads. -/
```
```lean
noncomputable def pFullSpec
    (s : BlockState) (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  TiledOptimizer.lionParam
    (s.readMem p_ptr (linearOffset s BLOCK_SIZE i))
    (s.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i))
    (s.readMem grad_ptr (linearOffset s BLOCK_SIZE i)) lr wd beta1
```
</details>

<details><summary><code>expAvgFullSpec</code></summary>

```
/-- Per-lane `exp_avg` output spec: the reusable Lion momentum oracle applied
to the values this lane loads. The math lives once in `Math.Optimizer`; this
only names the memory reads. -/
```
```lean
noncomputable def expAvgFullSpec
    (s : BlockState) (grad_ptr exp_avg_ptr : RegionName)
    (beta2 : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  TiledOptimizer.lionMomentum
    (s.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i))
    (s.readMem grad_ptr (linearOffset s BLOCK_SIZE i)) beta2
```
</details>

## Also present (pinned special-case summaries)
- `update_fn_kernel_exp_avg_compute_correct`
- `update_fn_kernel_p_compute_correct`
- `update_fn_kernel_all_outputs_compute_correct`
