# Spec sheet — `bench/tritonbench_g/l2_norm_triton1/L2NormTriton1.lean`

**Python source:** `bench/tritonbench_g/l2_norm_triton1/l2_norm_triton1.py`

## Public theorem: `l2_norm_fwd_1pass_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `_l2_norm_fwd_1pass_kernel`: the DSL surface
lowers to the algorithm layer, and the masked store to `Y` is compute-correct —
every active lane holds `l2Spec` (the oracle L2-norm value), out-of-bounds lanes
are preserved. -/
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
          (fun i => (Y, s.pid * stride_x_row + i.val)))
      (expected := fun i => l2Spec s X stride_x_row N BLOCK_N eps i)
```

**Assumptions / layout contracts:**
- `kernel : = l2_norm_fwd_1pass_kernel X Y stride_x_row N eps BLOCK_N`
- `initialState : = s`
- `fun i : Fin BLOCK_N => i.val < N`
- `expected : = fun i => l2Spec s X stride_x_row N BLOCK_N eps i`

**Closed-form spec defs (transitive):** `l2_norm_fwd_1pass_kernel`, `l2Spec`, `l2Load`

<details><summary><code>l2_norm_fwd_1pass_kernel</code></summary>

```
/-- Faithful transcription of `l2_norm_triton1.py`'s `_l2_norm_fwd_1pass_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N: tl.constexpr` → Lean `Nat` parameter. -/
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

<details><summary><code>l2Spec</code></summary>

```lean
noncomputable def l2Spec
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) (eps : ℝ)
    (idx : Fin BLOCK_N) : ℝ :=
  l2Norm (l2Load s X stride_x_row N BLOCK_N) eps idx
```
</details>

<details><summary><code>l2Load</code></summary>

```lean
noncomputable def l2Load
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat)
    (idx : Fin BLOCK_N) : ℝ :=
  if idx.val < N then
    s.readMem X (s.pid * stride_x_row + idx.val)
  else
    0
```
</details>

## Also present (pinned special-case summaries)
- `l2_norm_fwd_1pass_kernel_compute_correct`
