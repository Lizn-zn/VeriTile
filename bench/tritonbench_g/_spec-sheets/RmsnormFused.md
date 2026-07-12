# Spec sheet — `bench/tritonbench_g/rmsnorm_fused/RmsnormFused.lean`

**Python source:** `bench/tritonbench_g/rmsnorm_fused/rmsnorm_fused.py`

## Public theorem: `rms_norm_fwd_fused_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `rms_norm_fwd_fused`: the DSL surface lowers to
the algorithm layer, and the masked store to `Y` is compute-correct — every
active lane (`i.val < N`) holds the RMSNorm spec `rmsnormSpec`, out-of-bounds
lanes are preserved. Stated under the `0 < N ≤ BLOCK_SIZE` single-block launch
precondition chosen by the Python wrapper. -/
```
</details>

**Statement:**
```lean
specification rms_norm_fwd_fused_output_summary
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i)) :
    (∃ alg, (rms_norm_fwd_fused X Y W stride N BLOCK_SIZE eps).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rms_norm_fwd_fused X Y W stride N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < N)
        (fun i => (Y, yOffset s stride i)))
      (expected := fun i => rmsnormSpec s X W stride N BLOCK_SIZE eps i)
```

**Assumptions / layout contracts:**
- `hNpos : 0 < N`
- `hNle : N ≤ BLOCK_SIZE`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i)`
- `fun i : Fin BLOCK_SIZE => i.val < N`

**Closed-form spec defs (transitive):** `yOffset`, `rms_norm_fwd_fused`, `rmsnormSpec`, `rmsLoad`, `rmsWeight`, `xOffset`

<details><summary><code>yOffset</code></summary>

```lean
def yOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * stride + i.val
```
</details>

<details><summary><code>rms_norm_fwd_fused</code></summary>

```
/-- Faithful `forRange` transcription of `rmsnorm_fused.py`'s
`rms_norm_fwd_fused`.

The Python wrapper chooses `BLOCK_SIZE >= N` and raises otherwise, so the
correctness theorem below proves the full loop-shaped kernel under that
runtime precondition. Under the precondition both `range(0, N, BLOCK_SIZE)`
loops execute exactly the `off = 0` iteration.

Allowed mechanical Lean-syntax-only changes:
- Python `N` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameters. -/
```
```lean
def rms_norm_fwd_fused
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  Y += row * $(stride)
  X += row * $(stride)
  _var = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    x = tl.where(cols < $(N), x, 0.0)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y + cols, y, mask=mask)
  }
}
```
</details>

<details><summary><code>rmsnormSpec</code></summary>

```lean
noncomputable def rmsnormSpec
    (s : BlockState) (X W : RegionName)
    (stride N BLOCK_SIZE : Nat) (eps : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  TiledRMSNorm.rmsAffine
    (rmsLoad s X stride N BLOCK_SIZE)
    (rmsWeight s W)
    N eps i
```
</details>

<details><summary><code>rmsLoad</code></summary>

```lean
noncomputable def rmsLoad
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  Tile.maskedRowLoad
    (fun k : Fin BLOCK_SIZE => s.readMem X (xOffset s stride k))
    (fun k : Fin BLOCK_SIZE => k.val < N)
    0
    i
```
</details>

<details><summary><code>rmsWeight</code></summary>

```lean
noncomputable def rmsWeight
    (s : BlockState) (W : RegionName) (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem W i.val
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * stride + i.val
```
</details>

## Also present (pinned special-case summaries)
- `rms_norm_fwd_fused_compute_correct`
