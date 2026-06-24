# Spec sheet — `bench/tritonbench_g/layernorm_fwd_triton/LayernormFwdTriton.lean`

**Python source:** `bench/tritonbench_g/layernorm_fwd_triton/layernorm_fwd_triton.py`

## Public theorem: `layernorm_fwd_triton_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `_layer_norm_fwd_kernel`: the DSL surface
lowers to the algorithm layer, and the masked store to `Y` is compute-correct
for arbitrary `N` — every output column holds the full-`N` LayerNorm spec
`layernormYFullNSpec`. Built on the multi-block `*_compute_fullN_correct`
result; requires only `0 < BLOCK_SIZE` and output/input disjointness. -/
```
</details>

**Statement:**
```lean
theorem layernorm_fwd_triton_output_summary
    (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hBlockPos : 0 < BLOCK_SIZE)
    (hXYNe : X ≠ Y)
    (hWYNe : W ≠ Y) :
    (∃ alg, (layernorm_fwd_triton X W Y
        stride_x_N stride_x_hn stride_x_hd
        stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
        N BLOCK_SIZE eps).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := layernorm_fwd_triton X W Y
        stride_x_N stride_x_hn stride_x_hd
        stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
        N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin N => True)
        (fun i => (Y, yColOffset s stride_y_N stride_y_hn i.val)))
      (expected := fun i =>
        layernormYFullNSpec s X W stride_x_N stride_x_hn stride_w_hn
          N BLOCK_SIZE eps i)
```

**Assumptions / layout contracts:**
- `hBlockPos : 0 < BLOCK_SIZE`
- `hXYNe : X ≠ Y`
- `hWYNe : W ≠ Y`
- `kernel : = layernorm_fwd_triton X W Y
        stride_x_N stride_x_hn stride_x_hd
        stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
        N BLOCK_SIZE eps`
- `initialState : = s`
- `fun _ : Fin N => True`
- `expected : = fun i =>
        layernormYFullNSpec s X W stride_x_N stride_x_hn stride_w_hn
          N BLOCK_SIZE eps i`

**Closed-form spec defs (transitive):** `layernorm_fwd_triton`, `yColOffset`, `layernormYFullNSpec`, `xColOffset`, `layernormMeanFullNSpec`, `layernormRstdFullNSpec`, `wColOffset`, `layernormVarFullNSpec`

<details><summary><code>layernorm_fwd_triton</code></summary>

```
/-- Documented transcription of `layernorm_fwd_triton.py`'s
`_layer_norm_fwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `N` / `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameters.
- The Python `stride_x_hd`, `stride_y_hd`, and `stride_w_hd` parameters are kept
  as unused Lean parameters because the source kernel body does not use them. -/
```
```lean
def layernorm_fwd_triton
    (X W Y : RegionName)
    (stride_x_N stride_x_hn _stride_x_hd
      stride_y_N stride_y_hn _stride_y_hd
      stride_w_hn _stride_w_hd : Nat)
    (N BLOCK_SIZE : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  Seq = tl.program_id(0)
  H = tl.program_id(1)
  X += Seq * $(stride_x_N) + H * $(stride_x_hn)
  Y += Seq * $(stride_y_N) + H * $(stride_y_hn)
  W += H * $(stride_w_hn)
  _mean = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    a = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    _mean += a
  }
  mean = tl.sum(_mean, axis=0) / $(N)
  _var = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    x = tl.where(cols < $(N), x - mean, 0.0)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = (x - mean) * rstd
    y = x_hat * w
    tl.store(Y + cols, (y).to(X.dtype.element_ty), mask=mask)
  }
}
```
</details>

<details><summary><code>yColOffset</code></summary>

```lean
def yColOffset
    (s : BlockState) (stride_y_N stride_y_hn : Nat) (col : Nat) : Nat :=
  s.pids 0 * stride_y_N + s.pids 1 * stride_y_hn + col
```
</details>

<details><summary><code>layernormYFullNSpec</code></summary>

```
/-- Full-N output spec for every Python-observable output column. -/
```
```lean
noncomputable def layernormYFullNSpec
    (s : BlockState) (X W : RegionName)
    (stride_x_N stride_x_hn stride_w_hn N BLOCK_SIZE : Nat) (eps : ℝ)
    (i : Fin N) : ℝ :=
  ((s.readMem X (xColOffset s stride_x_N stride_x_hn i.val) -
      layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE) *
    layernormRstdFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE eps) *
    s.readMem W (wColOffset s stride_w_hn i.val)
```
</details>

<details><summary><code>xColOffset</code></summary>

```lean
def xColOffset
    (s : BlockState) (stride_x_N stride_x_hn : Nat) (col : Nat) : Nat :=
  s.pids 0 * stride_x_N + s.pids 1 * stride_x_hn + col
```
</details>

<details><summary><code>layernormMeanFullNSpec</code></summary>

```
/-- Full-N mean used by the Python `for off in range(0, N, BLOCK_SIZE)` path. -/
```
```lean
noncomputable def layernormMeanFullNSpec
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N _BLOCK_SIZE : Nat) : ℝ :=
  (∑ j : Fin N, s.readMem X (xColOffset s stride_x_N stride_x_hn j.val)) /
    (N : ℝ)
```
</details>

<details><summary><code>layernormRstdFullNSpec</code></summary>

```lean
noncomputable def layernormRstdFullNSpec
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) (eps : ℝ) : ℝ :=
  (Real.sqrt
    (layernormVarFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE + eps))⁻¹
```
</details>

<details><summary><code>wColOffset</code></summary>

```lean
def wColOffset (s : BlockState) (stride_w_hn : Nat) (col : Nat) : Nat :=
  s.pids 1 * stride_w_hn + col
```
</details>

<details><summary><code>layernormVarFullNSpec</code></summary>

```
/-- Full-N variance after subtracting the full-N mean. -/
```
```lean
noncomputable def layernormVarFullNSpec
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) : ℝ :=
  (∑ j : Fin N,
      (s.readMem X (xColOffset s stride_x_N stride_x_hn j.val) -
        layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE)^2) /
    (N : ℝ)
```
</details>

## Also present (pinned special-case summaries)
- `layernorm_fwd_triton_compute_correct`
