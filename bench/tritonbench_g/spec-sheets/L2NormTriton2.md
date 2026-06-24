# Spec sheet — `bench/tritonbench_g/l2_norm_triton2/L2NormTriton2.lean`

**Python source:** `bench/tritonbench_g/l2_norm_triton2/l2_norm_triton2.py`

## Public theorem: `l2_norm_fwd_1pass_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `_l2_norm_fwd_1pass_kernel`: the DSL surface
lowers to the algorithm layer, and the masked store to `Y` is compute-correct —
every active lane holds `l2FwdSpec` (the oracle L2-norm value), out-of-bounds
lanes are preserved. -/
```
</details>

**Statement:**
```lean
theorem l2_norm_fwd_1pass_kernel_output_summary
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (s : BlockState) :
    (∃ alg, (l2_norm_fwd_1pass_kernel X Y stride_x_row N eps BLOCK_N).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := l2_norm_fwd_1pass_kernel X Y stride_x_row N eps BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, l2OutOffset s stride_x_row i)))
      (expected := fun i => l2FwdSpec s X stride_x_row N BLOCK_N eps i)
```

**Assumptions / layout contracts:**
- `kernel : = l2_norm_fwd_1pass_kernel X Y stride_x_row N eps BLOCK_N`
- `initialState : = s`
- `fun i : Fin BLOCK_N => i.val < N`
- `expected : = fun i => l2FwdSpec s X stride_x_row N BLOCK_N eps i`

**Closed-form spec defs (transitive):** `l2_norm_fwd_1pass_kernel`, `l2OutOffset`, `l2FwdSpec`, `l2Load`

<details><summary><code>l2_norm_fwd_1pass_kernel</code></summary>

```
/-- Faithful transcription of `l2_norm_triton2.py`'s `_l2_norm_fwd_1pass_kernel`. -/
```
```lean
def l2_norm_fwd_1pass_kernel
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_x_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0)
  rstd = 1 / tl.sqrt(var + $(eps))
  mask = cols < $(N)
  y = x * rstd
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>l2OutOffset</code></summary>

```lean
def l2OutOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * stride_x_row + i.val
```
</details>

<details><summary><code>l2FwdSpec</code></summary>

```lean
noncomputable def l2FwdSpec
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) (eps : ℝ)
    (idx : Fin BLOCK_N) : ℝ :=
  l2Norm (l2Load s X stride_x_row N BLOCK_N) eps idx
```
</details>

<details><summary><code>l2Load</code></summary>

```lean
noncomputable def l2Load
    (s : BlockState) (R : RegionName) (stride_x_row N BLOCK_N : Nat)
    (idx : Fin BLOCK_N) : ℝ :=
  if idx.val < N then
    s.readMem R (s.pids 0 * stride_x_row + idx.val)
  else
    0
```
</details>

## Public theorem: `l2_norm_bwd_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `_l2_norm_bwd_kernel`: the DSL surface lowers to
the algorithm layer, and the masked store to `DX` is compute-correct — every
active lane holds `l2BwdSpec` (the oracle L2-norm backward value), out-of-bounds
lanes are preserved. -/
```
</details>

**Statement:**
```lean
theorem l2_norm_bwd_kernel_output_summary
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (s : BlockState) :
    (∃ alg, (l2_norm_bwd_kernel X DY DX stride_x_row N eps BLOCK_N).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := l2_norm_bwd_kernel X DY DX stride_x_row N eps BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DX, l2OutOffset s stride_x_row i)))
      (expected := fun i => l2BwdSpec s X DY stride_x_row N BLOCK_N eps i)
```

**Assumptions / layout contracts:**
- `kernel : = l2_norm_bwd_kernel X DY DX stride_x_row N eps BLOCK_N`
- `initialState : = s`
- `fun i : Fin BLOCK_N => i.val < N`
- `expected : = fun i => l2BwdSpec s X DY stride_x_row N BLOCK_N eps i`

**Closed-form spec defs (transitive):** `l2_norm_bwd_kernel`, `l2OutOffset`, `l2BwdSpec`, `l2Load`

<details><summary><code>l2_norm_bwd_kernel</code></summary>

```
/-- Faithful transcription of `l2_norm_triton2.py`'s `_l2_norm_bwd_kernel`. -/
```
```lean
def l2_norm_bwd_kernel
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  DX += row * $(stride_x_row)
  DY += row * $(stride_x_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(x * x)
  rstd = 1 / tl.sqrt(var + $(eps))
  mask = cols < $(N)
  dy = tl.load(DY + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  dy = tl.where(cols < $(N), dy, 0.0)
  dx = dy * rstd - tl.sum(dy * x) * (1 / (var + $(eps))) * rstd * x
  tl.store(DX + cols, dx, mask=mask)
}
```
</details>

<details><summary><code>l2OutOffset</code></summary>

```lean
def l2OutOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * stride_x_row + i.val
```
</details>

<details><summary><code>l2BwdSpec</code></summary>

```lean
noncomputable def l2BwdSpec
    (s : BlockState) (X DY : RegionName)
    (stride_x_row N BLOCK_N : Nat) (eps : ℝ) (idx : Fin BLOCK_N) : ℝ :=
  l2NormBwd
    (l2Load s X stride_x_row N BLOCK_N)
    (l2Load s DY stride_x_row N BLOCK_N)
    eps idx
```
</details>

<details><summary><code>l2Load</code></summary>

```lean
noncomputable def l2Load
    (s : BlockState) (R : RegionName) (stride_x_row N BLOCK_N : Nat)
    (idx : Fin BLOCK_N) : ℝ :=
  if idx.val < N then
    s.readMem R (s.pids 0 * stride_x_row + idx.val)
  else
    0
```
</details>

## Also present (pinned special-case summaries)
- `l2_norm_fwd_1pass_kernel_compute_correct`
- `l2_norm_bwd_kernel_compute_correct`
