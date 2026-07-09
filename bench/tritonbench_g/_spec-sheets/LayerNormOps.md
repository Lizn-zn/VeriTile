# Spec sheet — `bench/tritonbench_g/layer_norm_ops/LayerNormOps.lean`

**Python source:** `bench/tritonbench_g/layer_norm_ops/layer_norm_ops.py`

## Public theorem: `layer_norm_fwd_rms_one_block_y_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the `Y` output of the RMS slice. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_fwd_rms_one_block_y_compute_correct
    (X Y W Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hYRstd : Y ≠ Rstd)
    (hWRstd : W ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_rms_one_block X Y W Rstd
        stride_x_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, yOffset s stride_y_row i)))
      (expected := fun i => rmsYSpec s X W stride_x_row N BLOCK_N eps i)
```

**Assumptions / layout contracts:**
- `hYRstd : Y ≠ Rstd`
- `hWRstd : W ≠ Rstd`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `yOffset`, `layer_norm_fwd_rms_one_block`, `rmsYSpec`, `xOffset`, `rmsInvCarrier`, `rmsVarCarrier`, `rmsInputTile`

<details><summary><code>yOffset</code></summary>

```lean
def yOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_y_row + i.val
```
</details>

<details><summary><code>layer_norm_fwd_rms_one_block</code></summary>

```
/-- Proof-oriented RMS/no-residual/no-bias slice of `layer_norm_ops.py`'s
`_layer_norm_fwd_1pass_kernel`.

This specializes the constexpr branches to:
- `IS_RMS_NORM = true`
- `HAS_RESIDUAL = false`
- `STORE_RESIDUAL_OUT = false`
- `HAS_BIAS = false`

It captures the row pointer arithmetic, masked load, RMS variance, `Rstd`
side-output store, weight multiply, and masked `Y` output store. -/
```
```lean
def layer_norm_fwd_rms_one_block
    (X Y W Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  x_hat = x * rstd
  y = x_hat * w
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>rmsYSpec</code></summary>

```lean
noncomputable def rmsYSpec
    (s : BlockState) (X W : RegionName)
    (stride_x_row N BLOCK_N : Nat) (eps : ℝ) (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun x inv => x * inv)
        (some (s.readMem X (xOffset s stride_x_row i)))
        (rmsInvCarrier s X stride_x_row N BLOCK_N eps))
      (some (s.readMem W i.val)))
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

<details><summary><code>rmsInvCarrier</code></summary>

```lean
noncomputable def rmsInvCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (rmsVarCarrier s X stride_x_row N BLOCK_N)))
```
</details>

<details><summary><code>rmsVarCarrier</code></summary>

```lean
noncomputable def rmsVarCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsInputTile s X stride_x_row N BLOCK_N)
        (rmsInputTile s X stride_x_row N BLOCK_N))).data PUnit.unit)
```
</details>

<details><summary><code>rmsInputTile</code></summary>

```lean
noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        if idx.1.val < N then
          some (s.readMem X (xOffset s stride_x_row idx.1))
        else some (0.0 : ℝ)
      else some (0.0 : ℝ) }
```
</details>

## Public theorem: `layer_norm_fwd_rms_bias_one_block_y_compute_correct`

**Statement:**
```lean
theorem layer_norm_fwd_rms_bias_one_block_y_compute_correct
    (X Y W B Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hYRstd : Y ≠ Rstd)
    (hWRstd : W ≠ Rstd)
    (hBRstd : B ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_rms_bias_one_block X Y W B Rstd
        stride_x_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, yOffset s stride_y_row i)))
      (expected := fun i =>
        rmsBiasYSpec s X W B stride_x_row N BLOCK_N eps i)
```

**Assumptions / layout contracts:**
- `hYRstd : Y ≠ Rstd`
- `hWRstd : W ≠ Rstd`
- `hBRstd : B ≠ Rstd`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `yOffset`, `layer_norm_fwd_rms_bias_one_block`, `rmsBiasYSpec`, `xOffset`, `rmsInvCarrier`, `rmsVarCarrier`, `rmsInputTile`

<details><summary><code>yOffset</code></summary>

```lean
def yOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_y_row + i.val
```
</details>

<details><summary><code>layer_norm_fwd_rms_bias_one_block</code></summary>

```
/-! ## Forward RMS bias arithmetic slice

This covers `IS_RMS_NORM = true`, `HAS_RESIDUAL = false`, and `HAS_BIAS = true`
for the Python forward kernel. -/
```
```lean
def layer_norm_fwd_rms_bias_one_block
    (X Y W B Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
  x_hat = x * rstd
  y = x_hat * w + b
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>rmsBiasYSpec</code></summary>

```lean
noncomputable def rmsBiasYSpec
    (s : BlockState) (X W B : RegionName)
    (stride_x_row N BLOCK_N : Nat) (eps : ℝ) (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun weighted bias => weighted + bias)
      (Option.map₂ (fun scaled w => scaled * w)
        (Option.map₂ (fun x inv => x * inv)
          (some (s.readMem X (xOffset s stride_x_row i)))
          (rmsInvCarrier s X stride_x_row N BLOCK_N eps))
        (some (s.readMem W i.val)))
      (some (s.readMem B i.val)))
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

<details><summary><code>rmsInvCarrier</code></summary>

```lean
noncomputable def rmsInvCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (rmsVarCarrier s X stride_x_row N BLOCK_N)))
```
</details>

<details><summary><code>rmsVarCarrier</code></summary>

```lean
noncomputable def rmsVarCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsInputTile s X stride_x_row N BLOCK_N)
        (rmsInputTile s X stride_x_row N BLOCK_N))).data PUnit.unit)
```
</details>

<details><summary><code>rmsInputTile</code></summary>

```lean
noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        if idx.1.val < N then
          some (s.readMem X (xOffset s stride_x_row idx.1))
        else some (0.0 : ℝ)
      else some (0.0 : ℝ) }
```
</details>

## Public theorem: `layer_norm_fwd_rms_one_block_rstd_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the `Rstd` scalar store of
`layer_norm_fwd_rms_one_block`. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_fwd_rms_one_block_rstd_compute_correct
    (X Y W Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hRstdY : Rstd ≠ Y) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_rms_one_block X Y W Rstd
        stride_x_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, s.pid))
      (expected := fun _ =>
        rmsInvVarFullSpec s X stride_x_row N BLOCK_N eps)
```

**Assumptions / layout contracts:**
- `hRstdY : Rstd ≠ Y`

**Closed-form spec defs (transitive):** `layer_norm_fwd_rms_one_block`, `rmsInvVarFullSpec`, `rmsInvCarrier`, `rmsVarCarrier`, `rmsInputTile`, `xOffset`

<details><summary><code>layer_norm_fwd_rms_one_block</code></summary>

```
/-- Proof-oriented RMS/no-residual/no-bias slice of `layer_norm_ops.py`'s
`_layer_norm_fwd_1pass_kernel`.

This specializes the constexpr branches to:
- `IS_RMS_NORM = true`
- `HAS_RESIDUAL = false`
- `STORE_RESIDUAL_OUT = false`
- `HAS_BIAS = false`

It captures the row pointer arithmetic, masked load, RMS variance, `Rstd`
side-output store, weight multiply, and masked `Y` output store. -/
```
```lean
def layer_norm_fwd_rms_one_block
    (X Y W Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  x_hat = x * rstd
  y = x_hat * w
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>rmsInvVarFullSpec</code></summary>

```
/-- Full-kernel spec for the `Rstd` scalar store of `layer_norm_fwd_rms_one_block`.
Wraps `rmsInvCarrier` with `WithBot.unbotD 0` to match the kernel's
post-execution scalar readback. -/
```
```lean
noncomputable def rmsInvVarFullSpec
    (s : BlockState) (X : RegionName)
    (stride_x_row N BLOCK_N : Nat) (eps : ℝ) : ℝ :=
  WithBot.unbotD 0
    (rmsInvCarrier s X stride_x_row N BLOCK_N eps)
```
</details>

<details><summary><code>rmsInvCarrier</code></summary>

```lean
noncomputable def rmsInvCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (rmsVarCarrier s X stride_x_row N BLOCK_N)))
```
</details>

<details><summary><code>rmsVarCarrier</code></summary>

```lean
noncomputable def rmsVarCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsInputTile s X stride_x_row N BLOCK_N)
        (rmsInputTile s X stride_x_row N BLOCK_N))).data PUnit.unit)
```
</details>

<details><summary><code>rmsInputTile</code></summary>

```lean
noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        if idx.1.val < N then
          some (s.readMem X (xOffset s stride_x_row idx.1))
        else some (0.0 : ℝ)
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

## Public theorem: `layer_norm_fwd_rms_bias_one_block_rstd_compute_correct`

**Statement:**
```lean
theorem layer_norm_fwd_rms_bias_one_block_rstd_compute_correct
    (X Y W B Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hRstdY : Rstd ≠ Y) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_rms_bias_one_block X Y W B Rstd
        stride_x_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, s.pid))
      (expected := fun _ =>
        rmsInvVarFullSpec s X stride_x_row N BLOCK_N eps)
```

**Assumptions / layout contracts:**
- `hRstdY : Rstd ≠ Y`

**Closed-form spec defs (transitive):** `layer_norm_fwd_rms_bias_one_block`, `rmsInvVarFullSpec`, `rmsInvCarrier`, `rmsVarCarrier`, `rmsInputTile`, `xOffset`

<details><summary><code>layer_norm_fwd_rms_bias_one_block</code></summary>

```
/-! ## Forward RMS bias arithmetic slice

This covers `IS_RMS_NORM = true`, `HAS_RESIDUAL = false`, and `HAS_BIAS = true`
for the Python forward kernel. -/
```
```lean
def layer_norm_fwd_rms_bias_one_block
    (X Y W B Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
  x_hat = x * rstd
  y = x_hat * w + b
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>rmsInvVarFullSpec</code></summary>

```
/-- Full-kernel spec for the `Rstd` scalar store of `layer_norm_fwd_rms_one_block`.
Wraps `rmsInvCarrier` with `WithBot.unbotD 0` to match the kernel's
post-execution scalar readback. -/
```
```lean
noncomputable def rmsInvVarFullSpec
    (s : BlockState) (X : RegionName)
    (stride_x_row N BLOCK_N : Nat) (eps : ℝ) : ℝ :=
  WithBot.unbotD 0
    (rmsInvCarrier s X stride_x_row N BLOCK_N eps)
```
</details>

<details><summary><code>rmsInvCarrier</code></summary>

```lean
noncomputable def rmsInvCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (rmsVarCarrier s X stride_x_row N BLOCK_N)))
```
</details>

<details><summary><code>rmsVarCarrier</code></summary>

```lean
noncomputable def rmsVarCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsInputTile s X stride_x_row N BLOCK_N)
        (rmsInputTile s X stride_x_row N BLOCK_N))).data PUnit.unit)
```
</details>

<details><summary><code>rmsInputTile</code></summary>

```lean
noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        if idx.1.val < N then
          some (s.readMem X (xOffset s stride_x_row idx.1))
        else some (0.0 : ℝ)
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

## Public theorem: `layer_norm_fwd_plain_one_block_y_compute_correct`

**Statement:**
```lean
theorem layer_norm_fwd_plain_one_block_y_compute_correct
    (X Y W Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hYMean : Y ≠ Mean)
    (hYRstd : Y ≠ Rstd)
    (hWMean : W ≠ Mean)
    (hWRstd : W ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_plain_one_block X Y W Mean Rstd
        stride_x_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, yOffset s stride_y_row i)))
      (expected := fun i => plainYSpec s X W stride_x_row N BLOCK_N eps i)
```

**Assumptions / layout contracts:**
- `hYMean : Y ≠ Mean`
- `hYRstd : Y ≠ Rstd`
- `hWMean : W ≠ Mean`
- `hWRstd : W ≠ Rstd`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `yOffset`, `layer_norm_fwd_plain_one_block`, `plainYSpec`, `xOffset`, `plainMeanCarrier`, `plainInvCarrier`, `plainInputTile`, `plainVarCarrier`, `plainXbarTile`

<details><summary><code>yOffset</code></summary>

```lean
def yOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_y_row + i.val
```
</details>

<details><summary><code>layer_norm_fwd_plain_one_block</code></summary>

```
/-- Proof-oriented non-RMS/no-residual/no-bias slice of the forward kernel.

This covers the layer-norm arithmetic branch: mean, centered variance, `Rstd`,
weight multiply, and masked `Y` output. -/
```
```lean
def layer_norm_fwd_plain_one_block
    (X Y W Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  mean = tl.sum(x, axis=0) / $(N)
  tl.store(Mean + row, mean)
  xbar = tl.where(cols < $(N), x - mean, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  x_hat = (x - mean) * rstd
  y = x_hat * w
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>plainYSpec</code></summary>

```lean
noncomputable def plainYSpec
    (s : BlockState) (X W : RegionName)
    (stride_x_row N BLOCK_N : Nat) (eps : ℝ) (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun centered inv => centered * inv)
        (Option.map₂ (fun x mean => x - mean)
          (some (s.readMem X (xOffset s stride_x_row i)))
          (plainMeanCarrier s X stride_x_row N BLOCK_N))
        (plainInvCarrier s X stride_x_row N BLOCK_N eps))
      (some (s.readMem W i.val)))
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

<details><summary><code>plainMeanCarrier</code></summary>

```lean
noncomputable def plainMeanCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (plainInputTile s X stride_x_row N BLOCK_N)).data PUnit.unit)
```
</details>

<details><summary><code>plainInvCarrier</code></summary>

```lean
noncomputable def plainInvCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (plainVarCarrier s X stride_x_row N BLOCK_N)))
```
</details>

<details><summary><code>plainInputTile</code></summary>

```lean
noncomputable def plainInputTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem X (xOffset s stride_x_row idx.1))
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>plainVarCarrier</code></summary>

```lean
noncomputable def plainVarCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (plainXbarTile s X stride_x_row N BLOCK_N)
        (plainXbarTile s X stride_x_row N BLOCK_N))).data PUnit.unit)
```
</details>

<details><summary><code>plainXbarTile</code></summary>

```lean
noncomputable def plainXbarTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        Option.map₂ (fun x mean => x - mean)
          ((plainInputTile s X stride_x_row N BLOCK_N).data idx)
          (plainMeanCarrier s X stride_x_row N BLOCK_N)
      else some (0.0 : ℝ) }
```
</details>

## Public theorem: `layer_norm_fwd_plain_bias_one_block_y_compute_correct`

**Statement:**
```lean
theorem layer_norm_fwd_plain_bias_one_block_y_compute_correct
    (X Y W B Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hYMean : Y ≠ Mean)
    (hYRstd : Y ≠ Rstd)
    (hWMean : W ≠ Mean)
    (hWRstd : W ≠ Rstd)
    (hBMean : B ≠ Mean)
    (hBRstd : B ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_plain_bias_one_block X Y W B Mean Rstd
        stride_x_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, yOffset s stride_y_row i)))
      (expected := fun i =>
        plainBiasYSpec s X W B stride_x_row N BLOCK_N eps i)
```

**Assumptions / layout contracts:**
- `hYMean : Y ≠ Mean`
- `hYRstd : Y ≠ Rstd`
- `hWMean : W ≠ Mean`
- `hWRstd : W ≠ Rstd`
- `hBMean : B ≠ Mean`
- `hBRstd : B ≠ Rstd`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `yOffset`, `layer_norm_fwd_plain_bias_one_block`, `plainBiasYSpec`, `xOffset`, `plainMeanCarrier`, `plainInvCarrier`, `plainInputTile`, `plainVarCarrier`, `plainXbarTile`

<details><summary><code>yOffset</code></summary>

```lean
def yOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_y_row + i.val
```
</details>

<details><summary><code>layer_norm_fwd_plain_bias_one_block</code></summary>

```
/-! ## Forward plain bias arithmetic slice

This covers `IS_RMS_NORM = false`, `HAS_RESIDUAL = false`, and
`HAS_BIAS = true` for the Python forward kernel. -/
```
```lean
def layer_norm_fwd_plain_bias_one_block
    (X Y W B Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  mean = tl.sum(x, axis=0) / $(N)
  tl.store(Mean + row, mean)
  xbar = tl.where(cols < $(N), x - mean, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
  x_hat = (x - mean) * rstd
  y = x_hat * w + b
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>plainBiasYSpec</code></summary>

```lean
noncomputable def plainBiasYSpec
    (s : BlockState) (X W B : RegionName)
    (stride_x_row N BLOCK_N : Nat) (eps : ℝ) (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun weighted bias => weighted + bias)
      (Option.map₂ (fun scaled w => scaled * w)
        (Option.map₂ (fun centered inv => centered * inv)
          (Option.map₂ (fun x mean => x - mean)
            (some (s.readMem X (xOffset s stride_x_row i)))
            (plainMeanCarrier s X stride_x_row N BLOCK_N))
          (plainInvCarrier s X stride_x_row N BLOCK_N eps))
        (some (s.readMem W i.val)))
      (some (s.readMem B i.val)))
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

<details><summary><code>plainMeanCarrier</code></summary>

```lean
noncomputable def plainMeanCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (plainInputTile s X stride_x_row N BLOCK_N)).data PUnit.unit)
```
</details>

<details><summary><code>plainInvCarrier</code></summary>

```lean
noncomputable def plainInvCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (plainVarCarrier s X stride_x_row N BLOCK_N)))
```
</details>

<details><summary><code>plainInputTile</code></summary>

```lean
noncomputable def plainInputTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem X (xOffset s stride_x_row idx.1))
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>plainVarCarrier</code></summary>

```lean
noncomputable def plainVarCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (plainXbarTile s X stride_x_row N BLOCK_N)
        (plainXbarTile s X stride_x_row N BLOCK_N))).data PUnit.unit)
```
</details>

<details><summary><code>plainXbarTile</code></summary>

```lean
noncomputable def plainXbarTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        Option.map₂ (fun x mean => x - mean)
          ((plainInputTile s X stride_x_row N BLOCK_N).data idx)
          (plainMeanCarrier s X stride_x_row N BLOCK_N)
      else some (0.0 : ℝ) }
```
</details>

## Public theorem: `layer_norm_fwd_plain_residual_one_block_y_compute_correct`

**Statement:**
```lean
theorem layer_norm_fwd_plain_residual_one_block_y_compute_correct
    (X RESIDUAL Y W Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hYMean : Y ≠ Mean)
    (hYRstd : Y ≠ Rstd)
    (hWMean : W ≠ Mean)
    (hWRstd : W ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_plain_residual_one_block X RESIDUAL Y W
        Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, yOffset s stride_y_row i)))
      (expected := fun i =>
        plainResidualYSpec s X RESIDUAL W stride_x_row stride_res_row N
          BLOCK_N eps i)
```

**Assumptions / layout contracts:**
- `hYMean : Y ≠ Mean`
- `hYRstd : Y ≠ Rstd`
- `hWMean : W ≠ Mean`
- `hWRstd : W ≠ Rstd`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `yOffset`, `layer_norm_fwd_plain_residual_one_block`, `plainResidualYSpec`, `plainResidualInputTile`, `plainResidualMeanCarrier`, `plainResidualInvCarrier`, `xOffset`, `resOffset`, `plainResidualVarCarrier`, `plainResidualXbarTile`

<details><summary><code>yOffset</code></summary>

```lean
def yOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_y_row + i.val
```
</details>

<details><summary><code>layer_norm_fwd_plain_residual_one_block</code></summary>

```
/-! ## Forward residual arithmetic slice

This covers the non-RMS/no-bias forward branch with `HAS_RESIDUAL = true`.
Unlike the branch-named writeback aliases below, this slice computes the
residual addition before the mean/variance/rstd/Y arithmetic. -/
```
```lean
def layer_norm_fwd_plain_residual_one_block
    (X RESIDUAL Y W Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  RESIDUAL += row * $(stride_res_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  residual = tl.load(RESIDUAL + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x += residual
  mean = tl.sum(x, axis=0) / $(N)
  tl.store(Mean + row, mean)
  xbar = tl.where(cols < $(N), x - mean, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  x_hat = (x - mean) * rstd
  y = x_hat * w
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>plainResidualYSpec</code></summary>

```lean
noncomputable def plainResidualYSpec
    (s : BlockState) (X RESIDUAL W : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun centered inv => centered * inv)
        (Option.map₂ (fun x mean => x - mean)
          ((plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
            BLOCK_N).data (i, PUnit.unit))
          (plainResidualMeanCarrier s X RESIDUAL stride_x_row stride_res_row N
            BLOCK_N))
        (plainResidualInvCarrier s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N eps))
      (some (s.readMem W i.val)))
```
</details>

<details><summary><code>plainResidualInputTile</code></summary>

```lean
noncomputable def plainResidualInputTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      Option.map₂ (fun x residual : ℝ => x + residual)
        (if idx.1.val < N then
          some (s.readMem X (xOffset s stride_x_row idx.1))
        else some (0.0 : ℝ))
        (if idx.1.val < N then
          some (s.readMem RESIDUAL (resOffset s stride_res_row idx.1))
        else some (0.0 : ℝ)) }
```
</details>

<details><summary><code>plainResidualMeanCarrier</code></summary>

```lean
noncomputable def plainResidualMeanCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) : WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
        BLOCK_N)).data PUnit.unit)
```
</details>

<details><summary><code>plainResidualInvCarrier</code></summary>

```lean
noncomputable def plainResidualInvCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (plainResidualVarCarrier s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)))
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

<details><summary><code>resOffset</code></summary>

```lean
def resOffset (s : BlockState) (stride_res_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_res_row + i.val
```
</details>

<details><summary><code>plainResidualVarCarrier</code></summary>

```lean
noncomputable def plainResidualVarCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) : WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (plainResidualXbarTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)
        (plainResidualXbarTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N))).data PUnit.unit)
```
</details>

<details><summary><code>plainResidualXbarTile</code></summary>

```lean
noncomputable def plainResidualXbarTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        Option.map₂ (fun x mean => x - mean)
          ((plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
            BLOCK_N).data idx)
          (plainResidualMeanCarrier s X RESIDUAL stride_x_row stride_res_row N
            BLOCK_N)
      else some (0.0 : ℝ) }
```
</details>

## Public theorem: `layer_norm_fwd_plain_residual_one_block_mean_compute_correct`

**Statement:**
```lean
theorem layer_norm_fwd_plain_residual_one_block_mean_compute_correct
    (X RESIDUAL Y W Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hMeanRstd : Mean ≠ Rstd)
    (hMeanY : Mean ≠ Y) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_plain_residual_one_block X RESIDUAL Y W
        Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, s.pid))
      (expected := fun _ =>
        plainResidualMeanFullSpec s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)
```

**Assumptions / layout contracts:**
- `hMeanRstd : Mean ≠ Rstd`
- `hMeanY : Mean ≠ Y`

**Closed-form spec defs (transitive):** `layer_norm_fwd_plain_residual_one_block`, `plainResidualMeanFullSpec`, `plainResidualMeanCarrier`, `plainResidualInputTile`, `xOffset`, `resOffset`

<details><summary><code>layer_norm_fwd_plain_residual_one_block</code></summary>

```
/-! ## Forward residual arithmetic slice

This covers the non-RMS/no-bias forward branch with `HAS_RESIDUAL = true`.
Unlike the branch-named writeback aliases below, this slice computes the
residual addition before the mean/variance/rstd/Y arithmetic. -/
```
```lean
def layer_norm_fwd_plain_residual_one_block
    (X RESIDUAL Y W Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  RESIDUAL += row * $(stride_res_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  residual = tl.load(RESIDUAL + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x += residual
  mean = tl.sum(x, axis=0) / $(N)
  tl.store(Mean + row, mean)
  xbar = tl.where(cols < $(N), x - mean, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  x_hat = (x - mean) * rstd
  y = x_hat * w
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>plainResidualMeanFullSpec</code></summary>

```lean
noncomputable def plainResidualMeanFullSpec
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) : ℝ :=
  WithBot.unbotD 0
    (plainResidualMeanCarrier s X RESIDUAL stride_x_row stride_res_row N
      BLOCK_N)
```
</details>

<details><summary><code>plainResidualMeanCarrier</code></summary>

```lean
noncomputable def plainResidualMeanCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) : WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
        BLOCK_N)).data PUnit.unit)
```
</details>

<details><summary><code>plainResidualInputTile</code></summary>

```lean
noncomputable def plainResidualInputTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      Option.map₂ (fun x residual : ℝ => x + residual)
        (if idx.1.val < N then
          some (s.readMem X (xOffset s stride_x_row idx.1))
        else some (0.0 : ℝ))
        (if idx.1.val < N then
          some (s.readMem RESIDUAL (resOffset s stride_res_row idx.1))
        else some (0.0 : ℝ)) }
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

<details><summary><code>resOffset</code></summary>

```lean
def resOffset (s : BlockState) (stride_res_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_res_row + i.val
```
</details>

## Public theorem: `layer_norm_fwd_plain_residual_one_block_rstd_compute_correct`

**Statement:**
```lean
theorem layer_norm_fwd_plain_residual_one_block_rstd_compute_correct
    (X RESIDUAL Y W Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hRstdY : Rstd ≠ Y) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_plain_residual_one_block X RESIDUAL Y W
        Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, s.pid))
      (expected := fun _ =>
        plainResidualInvVarFullSpec s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N eps)
```

**Assumptions / layout contracts:**
- `hRstdY : Rstd ≠ Y`

**Closed-form spec defs (transitive):** `layer_norm_fwd_plain_residual_one_block`, `plainResidualInvVarFullSpec`, `plainResidualInvCarrier`, `plainResidualVarCarrier`, `plainResidualXbarTile`, `plainResidualInputTile`, `plainResidualMeanCarrier`, `xOffset`, `resOffset`

<details><summary><code>layer_norm_fwd_plain_residual_one_block</code></summary>

```
/-! ## Forward residual arithmetic slice

This covers the non-RMS/no-bias forward branch with `HAS_RESIDUAL = true`.
Unlike the branch-named writeback aliases below, this slice computes the
residual addition before the mean/variance/rstd/Y arithmetic. -/
```
```lean
def layer_norm_fwd_plain_residual_one_block
    (X RESIDUAL Y W Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  RESIDUAL += row * $(stride_res_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  residual = tl.load(RESIDUAL + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x += residual
  mean = tl.sum(x, axis=0) / $(N)
  tl.store(Mean + row, mean)
  xbar = tl.where(cols < $(N), x - mean, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  x_hat = (x - mean) * rstd
  y = x_hat * w
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>plainResidualInvVarFullSpec</code></summary>

```lean
noncomputable def plainResidualInvVarFullSpec
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ) : ℝ :=
  WithBot.unbotD 0
    (plainResidualInvCarrier s X RESIDUAL stride_x_row stride_res_row N
      BLOCK_N eps)
```
</details>

<details><summary><code>plainResidualInvCarrier</code></summary>

```lean
noncomputable def plainResidualInvCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (plainResidualVarCarrier s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)))
```
</details>

<details><summary><code>plainResidualVarCarrier</code></summary>

```lean
noncomputable def plainResidualVarCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) : WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (plainResidualXbarTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)
        (plainResidualXbarTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N))).data PUnit.unit)
```
</details>

<details><summary><code>plainResidualXbarTile</code></summary>

```lean
noncomputable def plainResidualXbarTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        Option.map₂ (fun x mean => x - mean)
          ((plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
            BLOCK_N).data idx)
          (plainResidualMeanCarrier s X RESIDUAL stride_x_row stride_res_row N
            BLOCK_N)
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>plainResidualInputTile</code></summary>

```lean
noncomputable def plainResidualInputTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      Option.map₂ (fun x residual : ℝ => x + residual)
        (if idx.1.val < N then
          some (s.readMem X (xOffset s stride_x_row idx.1))
        else some (0.0 : ℝ))
        (if idx.1.val < N then
          some (s.readMem RESIDUAL (resOffset s stride_res_row idx.1))
        else some (0.0 : ℝ)) }
```
</details>

<details><summary><code>plainResidualMeanCarrier</code></summary>

```lean
noncomputable def plainResidualMeanCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) : WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
        BLOCK_N)).data PUnit.unit)
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

<details><summary><code>resOffset</code></summary>

```lean
def resOffset (s : BlockState) (stride_res_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_res_row + i.val
```
</details>

## Public theorem: `layer_norm_fwd_rms_residual_one_block_y_compute_correct`

**Statement:**
```lean
theorem layer_norm_fwd_rms_residual_one_block_y_compute_correct
    (X RESIDUAL Y W Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hYRstd : Y ≠ Rstd)
    (hWRstd : W ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_rms_residual_one_block X RESIDUAL Y W Rstd
        stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, yOffset s stride_y_row i)))
      (expected := fun i =>
        rmsResidualYSpec s X RESIDUAL W stride_x_row stride_res_row N
          BLOCK_N eps i)
```

**Assumptions / layout contracts:**
- `hYRstd : Y ≠ Rstd`
- `hWRstd : W ≠ Rstd`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `yOffset`, `layer_norm_fwd_rms_residual_one_block`, `rmsResidualYSpec`, `plainResidualInputTile`, `rmsResidualInvCarrier`, `xOffset`, `resOffset`, `rmsResidualVarCarrier`, `rmsResidualXbarTile`

<details><summary><code>yOffset</code></summary>

```lean
def yOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_y_row + i.val
```
</details>

<details><summary><code>layer_norm_fwd_rms_residual_one_block</code></summary>

```
/-! ## Forward RMS residual arithmetic slice

This covers `IS_RMS_NORM = true`, `HAS_RESIDUAL = true`, and `HAS_BIAS = false`
for the Python forward kernel. -/
```
```lean
def layer_norm_fwd_rms_residual_one_block
    (X RESIDUAL Y W Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  RESIDUAL += row * $(stride_res_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  residual = tl.load(RESIDUAL + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x += residual
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  x_hat = x * rstd
  y = x_hat * w
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>rmsResidualYSpec</code></summary>

```lean
noncomputable def rmsResidualYSpec
    (s : BlockState) (X RESIDUAL W : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun x inv => x * inv)
        ((plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N).data (i, PUnit.unit))
        (rmsResidualInvCarrier s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N eps))
      (some (s.readMem W i.val)))
```
</details>

<details><summary><code>plainResidualInputTile</code></summary>

```lean
noncomputable def plainResidualInputTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      Option.map₂ (fun x residual : ℝ => x + residual)
        (if idx.1.val < N then
          some (s.readMem X (xOffset s stride_x_row idx.1))
        else some (0.0 : ℝ))
        (if idx.1.val < N then
          some (s.readMem RESIDUAL (resOffset s stride_res_row idx.1))
        else some (0.0 : ℝ)) }
```
</details>

<details><summary><code>rmsResidualInvCarrier</code></summary>

```lean
noncomputable def rmsResidualInvCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (rmsResidualVarCarrier s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)))
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

<details><summary><code>resOffset</code></summary>

```lean
def resOffset (s : BlockState) (stride_res_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_res_row + i.val
```
</details>

<details><summary><code>rmsResidualVarCarrier</code></summary>

```lean
noncomputable def rmsResidualVarCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) : WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsResidualXbarTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)
        (rmsResidualXbarTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N))).data PUnit.unit)
```
</details>

<details><summary><code>rmsResidualXbarTile</code></summary>

```lean
noncomputable def rmsResidualXbarTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        (plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N).data idx
      else some (0.0 : ℝ) }
```
</details>

## Public theorem: `layer_norm_fwd_rms_residual_one_block_rstd_compute_correct`

**Statement:**
```lean
theorem layer_norm_fwd_rms_residual_one_block_rstd_compute_correct
    (X RESIDUAL Y W Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hRstdY : Rstd ≠ Y) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_rms_residual_one_block X RESIDUAL Y W Rstd
        stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, s.pid))
      (expected := fun _ =>
        rmsResidualInvVarFullSpec s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N eps)
```

**Assumptions / layout contracts:**
- `hRstdY : Rstd ≠ Y`

**Closed-form spec defs (transitive):** `layer_norm_fwd_rms_residual_one_block`, `rmsResidualInvVarFullSpec`, `rmsResidualInvCarrier`, `rmsResidualVarCarrier`, `rmsResidualXbarTile`, `plainResidualInputTile`, `xOffset`, `resOffset`

<details><summary><code>layer_norm_fwd_rms_residual_one_block</code></summary>

```
/-! ## Forward RMS residual arithmetic slice

This covers `IS_RMS_NORM = true`, `HAS_RESIDUAL = true`, and `HAS_BIAS = false`
for the Python forward kernel. -/
```
```lean
def layer_norm_fwd_rms_residual_one_block
    (X RESIDUAL Y W Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  RESIDUAL += row * $(stride_res_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  residual = tl.load(RESIDUAL + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x += residual
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  x_hat = x * rstd
  y = x_hat * w
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>rmsResidualInvVarFullSpec</code></summary>

```lean
noncomputable def rmsResidualInvVarFullSpec
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ) : ℝ :=
  WithBot.unbotD 0
    (rmsResidualInvCarrier s X RESIDUAL stride_x_row stride_res_row N
      BLOCK_N eps)
```
</details>

<details><summary><code>rmsResidualInvCarrier</code></summary>

```lean
noncomputable def rmsResidualInvCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (rmsResidualVarCarrier s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)))
```
</details>

<details><summary><code>rmsResidualVarCarrier</code></summary>

```lean
noncomputable def rmsResidualVarCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) : WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsResidualXbarTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)
        (rmsResidualXbarTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N))).data PUnit.unit)
```
</details>

<details><summary><code>rmsResidualXbarTile</code></summary>

```lean
noncomputable def rmsResidualXbarTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        (plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N).data idx
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>plainResidualInputTile</code></summary>

```lean
noncomputable def plainResidualInputTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      Option.map₂ (fun x residual : ℝ => x + residual)
        (if idx.1.val < N then
          some (s.readMem X (xOffset s stride_x_row idx.1))
        else some (0.0 : ℝ))
        (if idx.1.val < N then
          some (s.readMem RESIDUAL (resOffset s stride_res_row idx.1))
        else some (0.0 : ℝ)) }
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

<details><summary><code>resOffset</code></summary>

```lean
def resOffset (s : BlockState) (stride_res_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_res_row + i.val
```
</details>

## Public theorem: `layer_norm_fwd_rms_residual_bias_one_block_y_compute_correct`

**Statement:**
```lean
theorem layer_norm_fwd_rms_residual_bias_one_block_y_compute_correct
    (X RESIDUAL Y W B Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hYRstd : Y ≠ Rstd)
    (hWRstd : W ≠ Rstd)
    (hBRstd : B ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_rms_residual_bias_one_block X RESIDUAL Y W
        B Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, yOffset s stride_y_row i)))
      (expected := fun i =>
        rmsResidualBiasYSpec s X RESIDUAL W B stride_x_row stride_res_row N
          BLOCK_N eps i)
```

**Assumptions / layout contracts:**
- `hYRstd : Y ≠ Rstd`
- `hWRstd : W ≠ Rstd`
- `hBRstd : B ≠ Rstd`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `yOffset`, `layer_norm_fwd_rms_residual_bias_one_block`, `rmsResidualBiasYSpec`, `plainResidualInputTile`, `rmsResidualInvCarrier`, `xOffset`, `resOffset`, `rmsResidualVarCarrier`, `rmsResidualXbarTile`

<details><summary><code>yOffset</code></summary>

```lean
def yOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_y_row + i.val
```
</details>

<details><summary><code>layer_norm_fwd_rms_residual_bias_one_block</code></summary>

```
/-! ## Forward RMS residual+bias arithmetic slice

This covers `IS_RMS_NORM = true`, `HAS_RESIDUAL = true`, and `HAS_BIAS = true`
for the Python forward kernel. -/
```
```lean
def layer_norm_fwd_rms_residual_bias_one_block
    (X RESIDUAL Y W B Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  RESIDUAL += row * $(stride_res_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  residual = tl.load(RESIDUAL + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x += residual
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
  x_hat = x * rstd
  y = x_hat * w + b
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>rmsResidualBiasYSpec</code></summary>

```lean
noncomputable def rmsResidualBiasYSpec
    (s : BlockState) (X RESIDUAL W B : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun weighted bias => weighted + bias)
      (Option.map₂ (fun scaled w => scaled * w)
        (Option.map₂ (fun x inv => x * inv)
          ((plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
            BLOCK_N).data (i, PUnit.unit))
          (rmsResidualInvCarrier s X RESIDUAL stride_x_row stride_res_row N
            BLOCK_N eps))
        (some (s.readMem W i.val)))
      (some (s.readMem B i.val)))
```
</details>

<details><summary><code>plainResidualInputTile</code></summary>

```lean
noncomputable def plainResidualInputTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      Option.map₂ (fun x residual : ℝ => x + residual)
        (if idx.1.val < N then
          some (s.readMem X (xOffset s stride_x_row idx.1))
        else some (0.0 : ℝ))
        (if idx.1.val < N then
          some (s.readMem RESIDUAL (resOffset s stride_res_row idx.1))
        else some (0.0 : ℝ)) }
```
</details>

<details><summary><code>rmsResidualInvCarrier</code></summary>

```lean
noncomputable def rmsResidualInvCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (rmsResidualVarCarrier s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)))
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

<details><summary><code>resOffset</code></summary>

```lean
def resOffset (s : BlockState) (stride_res_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_res_row + i.val
```
</details>

<details><summary><code>rmsResidualVarCarrier</code></summary>

```lean
noncomputable def rmsResidualVarCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) : WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsResidualXbarTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)
        (rmsResidualXbarTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N))).data PUnit.unit)
```
</details>

<details><summary><code>rmsResidualXbarTile</code></summary>

```lean
noncomputable def rmsResidualXbarTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        (plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N).data idx
      else some (0.0 : ℝ) }
```
</details>

## Public theorem: `layer_norm_fwd_rms_residual_bias_one_block_rstd_compute_correct`

**Statement:**
```lean
theorem layer_norm_fwd_rms_residual_bias_one_block_rstd_compute_correct
    (X RESIDUAL Y W B Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hRstdY : Rstd ≠ Y) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_rms_residual_bias_one_block X RESIDUAL Y W
        B Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, s.pid))
      (expected := fun _ =>
        rmsResidualInvVarFullSpec s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N eps)
```

**Assumptions / layout contracts:**
- `hRstdY : Rstd ≠ Y`

**Closed-form spec defs (transitive):** `layer_norm_fwd_rms_residual_bias_one_block`, `rmsResidualInvVarFullSpec`, `rmsResidualInvCarrier`, `rmsResidualVarCarrier`, `rmsResidualXbarTile`, `plainResidualInputTile`, `xOffset`, `resOffset`

<details><summary><code>layer_norm_fwd_rms_residual_bias_one_block</code></summary>

```
/-! ## Forward RMS residual+bias arithmetic slice

This covers `IS_RMS_NORM = true`, `HAS_RESIDUAL = true`, and `HAS_BIAS = true`
for the Python forward kernel. -/
```
```lean
def layer_norm_fwd_rms_residual_bias_one_block
    (X RESIDUAL Y W B Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  RESIDUAL += row * $(stride_res_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  residual = tl.load(RESIDUAL + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x += residual
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
  x_hat = x * rstd
  y = x_hat * w + b
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>rmsResidualInvVarFullSpec</code></summary>

```lean
noncomputable def rmsResidualInvVarFullSpec
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ) : ℝ :=
  WithBot.unbotD 0
    (rmsResidualInvCarrier s X RESIDUAL stride_x_row stride_res_row N
      BLOCK_N eps)
```
</details>

<details><summary><code>rmsResidualInvCarrier</code></summary>

```lean
noncomputable def rmsResidualInvCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (rmsResidualVarCarrier s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)))
```
</details>

<details><summary><code>rmsResidualVarCarrier</code></summary>

```lean
noncomputable def rmsResidualVarCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) : WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsResidualXbarTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)
        (rmsResidualXbarTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N))).data PUnit.unit)
```
</details>

<details><summary><code>rmsResidualXbarTile</code></summary>

```lean
noncomputable def rmsResidualXbarTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        (plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N).data idx
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>plainResidualInputTile</code></summary>

```lean
noncomputable def plainResidualInputTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      Option.map₂ (fun x residual : ℝ => x + residual)
        (if idx.1.val < N then
          some (s.readMem X (xOffset s stride_x_row idx.1))
        else some (0.0 : ℝ))
        (if idx.1.val < N then
          some (s.readMem RESIDUAL (resOffset s stride_res_row idx.1))
        else some (0.0 : ℝ)) }
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

<details><summary><code>resOffset</code></summary>

```lean
def resOffset (s : BlockState) (stride_res_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_res_row + i.val
```
</details>

## Public theorem: `layer_norm_fwd_plain_residual_bias_one_block_y_compute_correct`

**Statement:**
```lean
theorem layer_norm_fwd_plain_residual_bias_one_block_y_compute_correct
    (X RESIDUAL Y W B Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hYMean : Y ≠ Mean)
    (hYRstd : Y ≠ Rstd)
    (hWMean : W ≠ Mean)
    (hWRstd : W ≠ Rstd)
    (hBMean : B ≠ Mean)
    (hBRstd : B ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_plain_residual_bias_one_block X RESIDUAL Y
        W B Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, yOffset s stride_y_row i)))
      (expected := fun i =>
        plainResidualBiasYSpec s X RESIDUAL W B stride_x_row stride_res_row N
          BLOCK_N eps i)
```

**Assumptions / layout contracts:**
- `hYMean : Y ≠ Mean`
- `hYRstd : Y ≠ Rstd`
- `hWMean : W ≠ Mean`
- `hWRstd : W ≠ Rstd`
- `hBMean : B ≠ Mean`
- `hBRstd : B ≠ Rstd`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `yOffset`, `layer_norm_fwd_plain_residual_bias_one_block`, `plainResidualBiasYSpec`, `plainResidualInputTile`, `plainResidualMeanCarrier`, `plainResidualInvCarrier`, `xOffset`, `resOffset`, `plainResidualVarCarrier`, `plainResidualXbarTile`

<details><summary><code>yOffset</code></summary>

```lean
def yOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_y_row + i.val
```
</details>

<details><summary><code>layer_norm_fwd_plain_residual_bias_one_block</code></summary>

```
/-! ## Forward plain residual+bias arithmetic slice

This covers `IS_RMS_NORM = false`, `HAS_RESIDUAL = true`, and
`HAS_BIAS = true` for the Python forward kernel. -/
```
```lean
def layer_norm_fwd_plain_residual_bias_one_block
    (X RESIDUAL Y W B Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  RESIDUAL += row * $(stride_res_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  residual = tl.load(RESIDUAL + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x += residual
  mean = tl.sum(x, axis=0) / $(N)
  tl.store(Mean + row, mean)
  xbar = tl.where(cols < $(N), x - mean, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
  x_hat = (x - mean) * rstd
  y = x_hat * w + b
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>plainResidualBiasYSpec</code></summary>

```lean
noncomputable def plainResidualBiasYSpec
    (s : BlockState) (X RESIDUAL W B : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun weighted bias => weighted + bias)
      (Option.map₂ (fun scaled w => scaled * w)
        (Option.map₂ (fun centered inv => centered * inv)
          (Option.map₂ (fun x mean => x - mean)
            ((plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
              BLOCK_N).data (i, PUnit.unit))
            (plainResidualMeanCarrier s X RESIDUAL stride_x_row stride_res_row N
              BLOCK_N))
          (plainResidualInvCarrier s X RESIDUAL stride_x_row stride_res_row N
            BLOCK_N eps))
        (some (s.readMem W i.val)))
      (some (s.readMem B i.val)))
```
</details>

<details><summary><code>plainResidualInputTile</code></summary>

```lean
noncomputable def plainResidualInputTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      Option.map₂ (fun x residual : ℝ => x + residual)
        (if idx.1.val < N then
          some (s.readMem X (xOffset s stride_x_row idx.1))
        else some (0.0 : ℝ))
        (if idx.1.val < N then
          some (s.readMem RESIDUAL (resOffset s stride_res_row idx.1))
        else some (0.0 : ℝ)) }
```
</details>

<details><summary><code>plainResidualMeanCarrier</code></summary>

```lean
noncomputable def plainResidualMeanCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) : WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
        BLOCK_N)).data PUnit.unit)
```
</details>

<details><summary><code>plainResidualInvCarrier</code></summary>

```lean
noncomputable def plainResidualInvCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (plainResidualVarCarrier s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)))
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

<details><summary><code>resOffset</code></summary>

```lean
def resOffset (s : BlockState) (stride_res_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_res_row + i.val
```
</details>

<details><summary><code>plainResidualVarCarrier</code></summary>

```lean
noncomputable def plainResidualVarCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) : WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (plainResidualXbarTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)
        (plainResidualXbarTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N))).data PUnit.unit)
```
</details>

<details><summary><code>plainResidualXbarTile</code></summary>

```lean
noncomputable def plainResidualXbarTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        Option.map₂ (fun x mean => x - mean)
          ((plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
            BLOCK_N).data idx)
          (plainResidualMeanCarrier s X RESIDUAL stride_x_row stride_res_row N
            BLOCK_N)
      else some (0.0 : ℝ) }
```
</details>

## Public theorem: `layer_norm_fwd_plain_residual_bias_one_block_mean_compute_correct`

**Statement:**
```lean
theorem layer_norm_fwd_plain_residual_bias_one_block_mean_compute_correct
    (X RESIDUAL Y W B Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hMeanRstd : Mean ≠ Rstd)
    (hMeanY : Mean ≠ Y) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_plain_residual_bias_one_block X RESIDUAL Y
        W B Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, s.pid))
      (expected := fun _ =>
        plainResidualMeanFullSpec s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)
```

**Assumptions / layout contracts:**
- `hMeanRstd : Mean ≠ Rstd`
- `hMeanY : Mean ≠ Y`

**Closed-form spec defs (transitive):** `layer_norm_fwd_plain_residual_bias_one_block`, `plainResidualMeanFullSpec`, `plainResidualMeanCarrier`, `plainResidualInputTile`, `xOffset`, `resOffset`

<details><summary><code>layer_norm_fwd_plain_residual_bias_one_block</code></summary>

```
/-! ## Forward plain residual+bias arithmetic slice

This covers `IS_RMS_NORM = false`, `HAS_RESIDUAL = true`, and
`HAS_BIAS = true` for the Python forward kernel. -/
```
```lean
def layer_norm_fwd_plain_residual_bias_one_block
    (X RESIDUAL Y W B Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  RESIDUAL += row * $(stride_res_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  residual = tl.load(RESIDUAL + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x += residual
  mean = tl.sum(x, axis=0) / $(N)
  tl.store(Mean + row, mean)
  xbar = tl.where(cols < $(N), x - mean, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
  x_hat = (x - mean) * rstd
  y = x_hat * w + b
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>plainResidualMeanFullSpec</code></summary>

```lean
noncomputable def plainResidualMeanFullSpec
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) : ℝ :=
  WithBot.unbotD 0
    (plainResidualMeanCarrier s X RESIDUAL stride_x_row stride_res_row N
      BLOCK_N)
```
</details>

<details><summary><code>plainResidualMeanCarrier</code></summary>

```lean
noncomputable def plainResidualMeanCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) : WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
        BLOCK_N)).data PUnit.unit)
```
</details>

<details><summary><code>plainResidualInputTile</code></summary>

```lean
noncomputable def plainResidualInputTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      Option.map₂ (fun x residual : ℝ => x + residual)
        (if idx.1.val < N then
          some (s.readMem X (xOffset s stride_x_row idx.1))
        else some (0.0 : ℝ))
        (if idx.1.val < N then
          some (s.readMem RESIDUAL (resOffset s stride_res_row idx.1))
        else some (0.0 : ℝ)) }
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

<details><summary><code>resOffset</code></summary>

```lean
def resOffset (s : BlockState) (stride_res_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_res_row + i.val
```
</details>

## Public theorem: `layer_norm_fwd_plain_residual_bias_one_block_rstd_compute_correct`

**Statement:**
```lean
theorem layer_norm_fwd_plain_residual_bias_one_block_rstd_compute_correct
    (X RESIDUAL Y W B Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hRstdY : Rstd ≠ Y) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_plain_residual_bias_one_block X RESIDUAL Y
        W B Mean Rstd stride_x_row stride_res_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, s.pid))
      (expected := fun _ =>
        plainResidualInvVarFullSpec s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N eps)
```

**Assumptions / layout contracts:**
- `hRstdY : Rstd ≠ Y`

**Closed-form spec defs (transitive):** `layer_norm_fwd_plain_residual_bias_one_block`, `plainResidualInvVarFullSpec`, `plainResidualInvCarrier`, `plainResidualVarCarrier`, `plainResidualXbarTile`, `plainResidualInputTile`, `plainResidualMeanCarrier`, `xOffset`, `resOffset`

<details><summary><code>layer_norm_fwd_plain_residual_bias_one_block</code></summary>

```
/-! ## Forward plain residual+bias arithmetic slice

This covers `IS_RMS_NORM = false`, `HAS_RESIDUAL = true`, and
`HAS_BIAS = true` for the Python forward kernel. -/
```
```lean
def layer_norm_fwd_plain_residual_bias_one_block
    (X RESIDUAL Y W B Mean Rstd : RegionName)
    (stride_x_row stride_res_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  RESIDUAL += row * $(stride_res_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  residual = tl.load(RESIDUAL + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x += residual
  mean = tl.sum(x, axis=0) / $(N)
  tl.store(Mean + row, mean)
  xbar = tl.where(cols < $(N), x - mean, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
  x_hat = (x - mean) * rstd
  y = x_hat * w + b
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>plainResidualInvVarFullSpec</code></summary>

```lean
noncomputable def plainResidualInvVarFullSpec
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ) : ℝ :=
  WithBot.unbotD 0
    (plainResidualInvCarrier s X RESIDUAL stride_x_row stride_res_row N
      BLOCK_N eps)
```
</details>

<details><summary><code>plainResidualInvCarrier</code></summary>

```lean
noncomputable def plainResidualInvCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (plainResidualVarCarrier s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)))
```
</details>

<details><summary><code>plainResidualVarCarrier</code></summary>

```lean
noncomputable def plainResidualVarCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) : WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (plainResidualXbarTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N)
        (plainResidualXbarTile s X RESIDUAL stride_x_row stride_res_row N
          BLOCK_N))).data PUnit.unit)
```
</details>

<details><summary><code>plainResidualXbarTile</code></summary>

```lean
noncomputable def plainResidualXbarTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        Option.map₂ (fun x mean => x - mean)
          ((plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
            BLOCK_N).data idx)
          (plainResidualMeanCarrier s X RESIDUAL stride_x_row stride_res_row N
            BLOCK_N)
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>plainResidualInputTile</code></summary>

```lean
noncomputable def plainResidualInputTile
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      Option.map₂ (fun x residual : ℝ => x + residual)
        (if idx.1.val < N then
          some (s.readMem X (xOffset s stride_x_row idx.1))
        else some (0.0 : ℝ))
        (if idx.1.val < N then
          some (s.readMem RESIDUAL (resOffset s stride_res_row idx.1))
        else some (0.0 : ℝ)) }
```
</details>

<details><summary><code>plainResidualMeanCarrier</code></summary>

```lean
noncomputable def plainResidualMeanCarrier
    (s : BlockState) (X RESIDUAL : RegionName)
    (stride_x_row stride_res_row N BLOCK_N : Nat) : WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (plainResidualInputTile s X RESIDUAL stride_x_row stride_res_row N
        BLOCK_N)).data PUnit.unit)
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

<details><summary><code>resOffset</code></summary>

```lean
def resOffset (s : BlockState) (stride_res_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_res_row + i.val
```
</details>

## Public theorem: `layer_norm_fwd_plain_one_block_mean_compute_correct`

**Statement:**
```lean
theorem layer_norm_fwd_plain_one_block_mean_compute_correct
    (X Y W Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hMeanRstd : Mean ≠ Rstd)
    (hMeanY : Mean ≠ Y) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_plain_one_block X Y W Mean Rstd
        stride_x_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, s.pid))
      (expected := fun _ =>
        plainMeanFullSpec s X stride_x_row N BLOCK_N)
```

**Assumptions / layout contracts:**
- `hMeanRstd : Mean ≠ Rstd`
- `hMeanY : Mean ≠ Y`

**Closed-form spec defs (transitive):** `layer_norm_fwd_plain_one_block`, `plainMeanFullSpec`, `plainMeanCarrier`, `plainInputTile`, `xOffset`

<details><summary><code>layer_norm_fwd_plain_one_block</code></summary>

```
/-- Proof-oriented non-RMS/no-residual/no-bias slice of the forward kernel.

This covers the layer-norm arithmetic branch: mean, centered variance, `Rstd`,
weight multiply, and masked `Y` output. -/
```
```lean
def layer_norm_fwd_plain_one_block
    (X Y W Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  mean = tl.sum(x, axis=0) / $(N)
  tl.store(Mean + row, mean)
  xbar = tl.where(cols < $(N), x - mean, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  x_hat = (x - mean) * rstd
  y = x_hat * w
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>plainMeanFullSpec</code></summary>

```lean
noncomputable def plainMeanFullSpec
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) : ℝ :=
  WithBot.unbotD 0 (plainMeanCarrier s X stride_x_row N BLOCK_N)
```
</details>

<details><summary><code>plainMeanCarrier</code></summary>

```lean
noncomputable def plainMeanCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (plainInputTile s X stride_x_row N BLOCK_N)).data PUnit.unit)
```
</details>

<details><summary><code>plainInputTile</code></summary>

```lean
noncomputable def plainInputTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem X (xOffset s stride_x_row idx.1))
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

## Public theorem: `layer_norm_fwd_plain_one_block_rstd_compute_correct`

**Statement:**
```lean
theorem layer_norm_fwd_plain_one_block_rstd_compute_correct
    (X Y W Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hRstdY : Rstd ≠ Y) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_plain_one_block X Y W Mean Rstd
        stride_x_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, s.pid))
      (expected := fun _ =>
        plainInvVarFullSpec s X stride_x_row N BLOCK_N eps)
```

**Assumptions / layout contracts:**
- `hRstdY : Rstd ≠ Y`

**Closed-form spec defs (transitive):** `layer_norm_fwd_plain_one_block`, `plainInvVarFullSpec`, `plainInvCarrier`, `plainVarCarrier`, `plainXbarTile`, `plainInputTile`, `plainMeanCarrier`, `xOffset`

<details><summary><code>layer_norm_fwd_plain_one_block</code></summary>

```
/-- Proof-oriented non-RMS/no-residual/no-bias slice of the forward kernel.

This covers the layer-norm arithmetic branch: mean, centered variance, `Rstd`,
weight multiply, and masked `Y` output. -/
```
```lean
def layer_norm_fwd_plain_one_block
    (X Y W Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  mean = tl.sum(x, axis=0) / $(N)
  tl.store(Mean + row, mean)
  xbar = tl.where(cols < $(N), x - mean, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  x_hat = (x - mean) * rstd
  y = x_hat * w
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>plainInvVarFullSpec</code></summary>

```lean
noncomputable def plainInvVarFullSpec
    (s : BlockState) (X : RegionName)
    (stride_x_row N BLOCK_N : Nat) (eps : ℝ) : ℝ :=
  WithBot.unbotD 0
    (plainInvCarrier s X stride_x_row N BLOCK_N eps)
```
</details>

<details><summary><code>plainInvCarrier</code></summary>

```lean
noncomputable def plainInvCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (plainVarCarrier s X stride_x_row N BLOCK_N)))
```
</details>

<details><summary><code>plainVarCarrier</code></summary>

```lean
noncomputable def plainVarCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (plainXbarTile s X stride_x_row N BLOCK_N)
        (plainXbarTile s X stride_x_row N BLOCK_N))).data PUnit.unit)
```
</details>

<details><summary><code>plainXbarTile</code></summary>

```lean
noncomputable def plainXbarTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        Option.map₂ (fun x mean => x - mean)
          ((plainInputTile s X stride_x_row N BLOCK_N).data idx)
          (plainMeanCarrier s X stride_x_row N BLOCK_N)
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>plainInputTile</code></summary>

```lean
noncomputable def plainInputTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem X (xOffset s stride_x_row idx.1))
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>plainMeanCarrier</code></summary>

```lean
noncomputable def plainMeanCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (plainInputTile s X stride_x_row N BLOCK_N)).data PUnit.unit)
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

## Public theorem: `layer_norm_fwd_plain_bias_one_block_mean_compute_correct`

**Statement:**
```lean
theorem layer_norm_fwd_plain_bias_one_block_mean_compute_correct
    (X Y W B Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hMeanRstd : Mean ≠ Rstd)
    (hMeanY : Mean ≠ Y) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_plain_bias_one_block X Y W B Mean Rstd
        stride_x_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, s.pid))
      (expected := fun _ =>
        plainMeanFullSpec s X stride_x_row N BLOCK_N)
```

**Assumptions / layout contracts:**
- `hMeanRstd : Mean ≠ Rstd`
- `hMeanY : Mean ≠ Y`

**Closed-form spec defs (transitive):** `layer_norm_fwd_plain_bias_one_block`, `plainMeanFullSpec`, `plainMeanCarrier`, `plainInputTile`, `xOffset`

<details><summary><code>layer_norm_fwd_plain_bias_one_block</code></summary>

```
/-! ## Forward plain bias arithmetic slice

This covers `IS_RMS_NORM = false`, `HAS_RESIDUAL = false`, and
`HAS_BIAS = true` for the Python forward kernel. -/
```
```lean
def layer_norm_fwd_plain_bias_one_block
    (X Y W B Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  mean = tl.sum(x, axis=0) / $(N)
  tl.store(Mean + row, mean)
  xbar = tl.where(cols < $(N), x - mean, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
  x_hat = (x - mean) * rstd
  y = x_hat * w + b
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>plainMeanFullSpec</code></summary>

```lean
noncomputable def plainMeanFullSpec
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) : ℝ :=
  WithBot.unbotD 0 (plainMeanCarrier s X stride_x_row N BLOCK_N)
```
</details>

<details><summary><code>plainMeanCarrier</code></summary>

```lean
noncomputable def plainMeanCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (plainInputTile s X stride_x_row N BLOCK_N)).data PUnit.unit)
```
</details>

<details><summary><code>plainInputTile</code></summary>

```lean
noncomputable def plainInputTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem X (xOffset s stride_x_row idx.1))
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

## Public theorem: `layer_norm_fwd_plain_bias_one_block_rstd_compute_correct`

**Statement:**
```lean
theorem layer_norm_fwd_plain_bias_one_block_rstd_compute_correct
    (X Y W B Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ)
    (s : BlockState)
    (hRstdY : Rstd ≠ Y) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_plain_bias_one_block X Y W B Mean Rstd
        stride_x_row stride_y_row N BLOCK_N eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, s.pid))
      (expected := fun _ =>
        plainInvVarFullSpec s X stride_x_row N BLOCK_N eps)
```

**Assumptions / layout contracts:**
- `hRstdY : Rstd ≠ Y`

**Closed-form spec defs (transitive):** `layer_norm_fwd_plain_bias_one_block`, `plainInvVarFullSpec`, `plainInvCarrier`, `plainVarCarrier`, `plainXbarTile`, `plainInputTile`, `plainMeanCarrier`, `xOffset`

<details><summary><code>layer_norm_fwd_plain_bias_one_block</code></summary>

```
/-! ## Forward plain bias arithmetic slice

This covers `IS_RMS_NORM = false`, `HAS_RESIDUAL = false`, and
`HAS_BIAS = true` for the Python forward kernel. -/
```
```lean
def layer_norm_fwd_plain_bias_one_block
    (X Y W B Mean Rstd : RegionName)
    (stride_x_row stride_y_row N BLOCK_N : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  mean = tl.sum(x, axis=0) / $(N)
  tl.store(Mean + row, mean)
  xbar = tl.where(cols < $(N), x - mean, 0.0)
  var = tl.sum(xbar * xbar, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
  x_hat = (x - mean) * rstd
  y = x_hat * w + b
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>plainInvVarFullSpec</code></summary>

```lean
noncomputable def plainInvVarFullSpec
    (s : BlockState) (X : RegionName)
    (stride_x_row N BLOCK_N : Nat) (eps : ℝ) : ℝ :=
  WithBot.unbotD 0
    (plainInvCarrier s X stride_x_row N BLOCK_N eps)
```
</details>

<details><summary><code>plainInvCarrier</code></summary>

```lean
noncomputable def plainInvCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (plainVarCarrier s X stride_x_row N BLOCK_N)))
```
</details>

<details><summary><code>plainVarCarrier</code></summary>

```lean
noncomputable def plainVarCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (plainXbarTile s X stride_x_row N BLOCK_N)
        (plainXbarTile s X stride_x_row N BLOCK_N))).data PUnit.unit)
```
</details>

<details><summary><code>plainXbarTile</code></summary>

```lean
noncomputable def plainXbarTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        Option.map₂ (fun x mean => x - mean)
          ((plainInputTile s X stride_x_row N BLOCK_N).data idx)
          (plainMeanCarrier s X stride_x_row N BLOCK_N)
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>plainInputTile</code></summary>

```lean
noncomputable def plainInputTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem X (xOffset s stride_x_row idx.1))
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>plainMeanCarrier</code></summary>

```lean
noncomputable def plainMeanCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (plainInputTile s X stride_x_row N BLOCK_N)).data PUnit.unit)
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

## Public theorem: `layer_norm_ops_fwd_mean_store_slice_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_fwd_mean_store_slice_compute_correct
    (MeanPre Mean : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_mean_store_slice MeanPre Mean)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, meanRowOffset s))
      (expected := fun _ => meanStoreSpec s MeanPre)
```

**Closed-form spec defs (transitive):** `layer_norm_ops_fwd_mean_store_slice`, `meanRowOffset`, `meanStoreSpec`

<details><summary><code>layer_norm_ops_fwd_mean_store_slice</code></summary>

```
/-- Proof-oriented Mean store slice of `layer_norm_ops.py`'s
`_layer_norm_fwd_1pass_kernel`. Takes a precomputed `MeanPre` scalar (per row)
and proves the unmasked scalar writeback into `Mean` at row offset. -/
```
```lean
def layer_norm_ops_fwd_mean_store_slice
    (MeanPre Mean : RegionName) : ComputeKernel := triton {
  row = tl.program_id(0)
  mean = tl.load(MeanPre + row)
  tl.store(Mean + row, mean)
}
```
</details>

<details><summary><code>meanRowOffset</code></summary>

```lean
def meanRowOffset (s : BlockState) : Nat := s.pid
```
</details>

<details><summary><code>meanStoreSpec</code></summary>

```lean
noncomputable def meanStoreSpec (s : BlockState) (MeanPre : RegionName) : ℝ :=
  s.readMem MeanPre s.pid
```
</details>

## Public theorem: `layer_norm_ops_fwd_rstd_store_slice_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_fwd_rstd_store_slice_compute_correct
    (RstdPre Rstd : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_rstd_store_slice RstdPre Rstd)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, meanRowOffset s))
      (expected := fun _ => rstdStoreSpec s RstdPre)
```

**Closed-form spec defs (transitive):** `layer_norm_ops_fwd_rstd_store_slice`, `meanRowOffset`, `rstdStoreSpec`

<details><summary><code>layer_norm_ops_fwd_rstd_store_slice</code></summary>

```
/-- Proof-oriented Rstd store slice. Same scalar-copy pattern. -/
```
```lean
def layer_norm_ops_fwd_rstd_store_slice
    (RstdPre Rstd : RegionName) : ComputeKernel := triton {
  row = tl.program_id(0)
  rstd = tl.load(RstdPre + row)
  tl.store(Rstd + row, rstd)
}
```
</details>

<details><summary><code>meanRowOffset</code></summary>

```lean
def meanRowOffset (s : BlockState) : Nat := s.pid
```
</details>

<details><summary><code>rstdStoreSpec</code></summary>

```lean
noncomputable def rstdStoreSpec (s : BlockState) (RstdPre : RegionName) : ℝ :=
  s.readMem RstdPre s.pid
```
</details>

## Public theorem: `layer_norm_ops_bwd_row_vector_store_slice_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_bwd_row_vector_store_slice_compute_correct
    (ValuePre Out : RegionName) (stride_out_row N BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_out_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_row_vector_store_slice ValuePre Out
        stride_out_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Out, bwdRowVectorOffset s stride_out_row i)))
      (expected := fun i =>
        bwdRowVectorStoreSpec s ValuePre stride_out_row i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_out_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `bwdRowVectorOffset`, `layer_norm_ops_bwd_row_vector_store_slice`, `bwdRowVectorStoreSpec`

<details><summary><code>bwdRowVectorOffset</code></summary>

```lean
def bwdRowVectorOffset (s : BlockState) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_out_row + i.val
```
</details>

<details><summary><code>layer_norm_ops_bwd_row_vector_store_slice</code></summary>

```
/-- Backward row-vector writeback slice for `DX`, optional `DRESIDUAL_IN`, and
optional recomputed `Y`. The caller supplies the precomputed row tile in
`ValuePre`; the slice proves the masked store to `Out + row * stride + cols`. -/
```
```lean
def layer_norm_ops_bwd_row_vector_store_slice
    (ValuePre Out : RegionName) (stride_out_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  value = tl.load(ValuePre + row * $(stride_out_row) + cols,
    mask=mask, other=0.0)
  tl.store(Out + row * $(stride_out_row) + cols, value, mask=mask)
}
```
</details>

<details><summary><code>bwdRowVectorStoreSpec</code></summary>

```lean
noncomputable def bwdRowVectorStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem ValuePre (bwdRowVectorOffset s stride_out_row i)
```
</details>

## Public theorem: `layer_norm_ops_bwd_dx_store_slice_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_bwd_dx_store_slice_compute_correct
    (DXPre DX : RegionName) (stride_dx_row N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_dx_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_dx_row i)`

**Closed-form spec defs (transitive):** `bwdRowVectorOffset`

<details><summary><code>bwdRowVectorOffset</code></summary>

```lean
def bwdRowVectorOffset (s : BlockState) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_out_row + i.val
```
</details>

## Public theorem: `layer_norm_ops_bwd_dresidual_in_store_slice_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_bwd_dresidual_in_store_slice_compute_correct
    (DXPre DRESIDUAL_IN : RegionName)
    (stride_dres_in_row N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_dres_in_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_dres_in_row i)`

**Closed-form spec defs (transitive):** `bwdRowVectorOffset`

<details><summary><code>bwdRowVectorOffset</code></summary>

```lean
def bwdRowVectorOffset (s : BlockState) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_out_row + i.val
```
</details>

## Public theorem: `layer_norm_ops_bwd_recompute_y_store_slice_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_bwd_recompute_y_store_slice_compute_correct
    (YPre Y : RegionName) (stride_y_row N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_y_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_y_row i)`

**Closed-form spec defs (transitive):** `bwdRowVectorOffset`

<details><summary><code>bwdRowVectorOffset</code></summary>

```lean
def bwdRowVectorOffset (s : BlockState) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_out_row + i.val
```
</details>

## Public theorem: `layer_norm_ops_fwd_residual_out_store_slice_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_fwd_residual_out_store_slice_compute_correct
    (ValuePre RESIDUAL_OUT : RegionName)
    (stride_res_out_row N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => fwdResidualOutOffset s stride_res_out_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_residual_out_store_slice ValuePre
        RESIDUAL_OUT stride_res_out_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (RESIDUAL_OUT,
          fwdResidualOutOffset s stride_res_out_row i)))
      (expected := fun i =>
        fwdResidualOutStoreSpec s ValuePre stride_res_out_row i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => fwdResidualOutOffset s stride_res_out_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `fwdResidualOutOffset`, `layer_norm_ops_fwd_residual_out_store_slice`, `fwdResidualOutStoreSpec`, `bwdRowVectorOffset`, `layer_norm_ops_bwd_row_vector_store_slice`, `bwdRowVectorStoreSpec`

<details><summary><code>fwdResidualOutOffset</code></summary>

```lean
def fwdResidualOutOffset (s : BlockState) (stride_res_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  bwdRowVectorOffset s stride_res_out_row i
```
</details>

<details><summary><code>layer_norm_ops_fwd_residual_out_store_slice</code></summary>

```
/-- Forward residual-output writeback slice for `_layer_norm_fwd_1pass_kernel`.

When `STORE_RESIDUAL_OUT` is true, Python stores the row tile `x` after the
optional residual add into `RESIDUAL_OUT + cols`. The caller supplies that
precomputed row tile in `ValuePre`; this theorem proves the same masked
row-vector writeback shape as the backward row-vector stores. -/
```
```lean
abbrev layer_norm_ops_fwd_residual_out_store_slice
    (ValuePre RESIDUAL_OUT : RegionName)
    (stride_res_out_row N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_row_vector_store_slice ValuePre RESIDUAL_OUT
    stride_res_out_row N BLOCK_N
```
</details>

<details><summary><code>fwdResidualOutStoreSpec</code></summary>

```lean
noncomputable def fwdResidualOutStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_res_out_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  bwdRowVectorStoreSpec s ValuePre stride_res_out_row i
```
</details>

<details><summary><code>bwdRowVectorOffset</code></summary>

```lean
def bwdRowVectorOffset (s : BlockState) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_out_row + i.val
```
</details>

<details><summary><code>layer_norm_ops_bwd_row_vector_store_slice</code></summary>

```
/-- Backward row-vector writeback slice for `DX`, optional `DRESIDUAL_IN`, and
optional recomputed `Y`. The caller supplies the precomputed row tile in
`ValuePre`; the slice proves the masked store to `Out + row * stride + cols`. -/
```
```lean
def layer_norm_ops_bwd_row_vector_store_slice
    (ValuePre Out : RegionName) (stride_out_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  value = tl.load(ValuePre + row * $(stride_out_row) + cols,
    mask=mask, other=0.0)
  tl.store(Out + row * $(stride_out_row) + cols, value, mask=mask)
}
```
</details>

<details><summary><code>bwdRowVectorStoreSpec</code></summary>

```lean
noncomputable def bwdRowVectorStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem ValuePre (bwdRowVectorOffset s stride_out_row i)
```
</details>

## Public theorem: `layer_norm_ops_fwd_y_store_slice_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_fwd_y_store_slice_compute_correct
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => fwdYOffset s stride_y_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_y_store_slice ValuePre Y
        stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, fwdYOffset s stride_y_row i)))
      (expected := fun i => fwdYStoreSpec s ValuePre stride_y_row i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => fwdYOffset s stride_y_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `fwdYOffset`, `layer_norm_ops_fwd_y_store_slice`, `fwdYStoreSpec`, `bwdRowVectorOffset`, `layer_norm_ops_bwd_row_vector_store_slice`, `bwdRowVectorStoreSpec`

<details><summary><code>fwdYOffset</code></summary>

```lean
def fwdYOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  bwdRowVectorOffset s stride_y_row i
```
</details>

<details><summary><code>layer_norm_ops_fwd_y_store_slice</code></summary>

```
/-- Forward output writeback slice for `_layer_norm_fwd_1pass_kernel`.

After the mean/rstd arithmetic and optional bias/residual branches, Python
stores the computed row tile `y` into `Y + cols`. This aliases the generic
row-vector writeback proof with the forward output stride. -/
```
```lean
abbrev layer_norm_ops_fwd_y_store_slice
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_row_vector_store_slice ValuePre Y stride_y_row N BLOCK_N
```
</details>

<details><summary><code>fwdYStoreSpec</code></summary>

```lean
noncomputable def fwdYStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_y_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  bwdRowVectorStoreSpec s ValuePre stride_y_row i
```
</details>

<details><summary><code>bwdRowVectorOffset</code></summary>

```lean
def bwdRowVectorOffset (s : BlockState) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_out_row + i.val
```
</details>

<details><summary><code>layer_norm_ops_bwd_row_vector_store_slice</code></summary>

```
/-- Backward row-vector writeback slice for `DX`, optional `DRESIDUAL_IN`, and
optional recomputed `Y`. The caller supplies the precomputed row tile in
`ValuePre`; the slice proves the masked store to `Out + row * stride + cols`. -/
```
```lean
def layer_norm_ops_bwd_row_vector_store_slice
    (ValuePre Out : RegionName) (stride_out_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  value = tl.load(ValuePre + row * $(stride_out_row) + cols,
    mask=mask, other=0.0)
  tl.store(Out + row * $(stride_out_row) + cols, value, mask=mask)
}
```
</details>

<details><summary><code>bwdRowVectorStoreSpec</code></summary>

```lean
noncomputable def bwdRowVectorStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem ValuePre (bwdRowVectorOffset s stride_out_row i)
```
</details>

## Public theorem: `layer_norm_ops_fwd_bias_y_store_slice_compute_correct`

<details><summary>docstring</summary>

```
/-- Named forward Y writeback for the bias branch of
`_layer_norm_fwd_1pass_kernel`.

The arithmetic producing `ValuePre` is branch-specific (`x_hat * W + B`);
this theorem exposes the Python-observed Y store for that branch using the
shared row-vector writeback proof. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_ops_fwd_bias_y_store_slice_compute_correct
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => fwdYOffset s stride_y_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_y_store_slice ValuePre Y
        stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, fwdYOffset s stride_y_row i)))
      (expected := fun i => fwdYStoreSpec s ValuePre stride_y_row i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => fwdYOffset s stride_y_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `fwdYOffset`, `layer_norm_ops_fwd_y_store_slice`, `fwdYStoreSpec`, `bwdRowVectorOffset`, `layer_norm_ops_bwd_row_vector_store_slice`, `bwdRowVectorStoreSpec`

<details><summary><code>fwdYOffset</code></summary>

```lean
def fwdYOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  bwdRowVectorOffset s stride_y_row i
```
</details>

<details><summary><code>layer_norm_ops_fwd_y_store_slice</code></summary>

```
/-- Forward output writeback slice for `_layer_norm_fwd_1pass_kernel`.

After the mean/rstd arithmetic and optional bias/residual branches, Python
stores the computed row tile `y` into `Y + cols`. This aliases the generic
row-vector writeback proof with the forward output stride. -/
```
```lean
abbrev layer_norm_ops_fwd_y_store_slice
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_row_vector_store_slice ValuePre Y stride_y_row N BLOCK_N
```
</details>

<details><summary><code>fwdYStoreSpec</code></summary>

```lean
noncomputable def fwdYStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_y_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  bwdRowVectorStoreSpec s ValuePre stride_y_row i
```
</details>

<details><summary><code>bwdRowVectorOffset</code></summary>

```lean
def bwdRowVectorOffset (s : BlockState) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_out_row + i.val
```
</details>

<details><summary><code>layer_norm_ops_bwd_row_vector_store_slice</code></summary>

```
/-- Backward row-vector writeback slice for `DX`, optional `DRESIDUAL_IN`, and
optional recomputed `Y`. The caller supplies the precomputed row tile in
`ValuePre`; the slice proves the masked store to `Out + row * stride + cols`. -/
```
```lean
def layer_norm_ops_bwd_row_vector_store_slice
    (ValuePre Out : RegionName) (stride_out_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  value = tl.load(ValuePre + row * $(stride_out_row) + cols,
    mask=mask, other=0.0)
  tl.store(Out + row * $(stride_out_row) + cols, value, mask=mask)
}
```
</details>

<details><summary><code>bwdRowVectorStoreSpec</code></summary>

```lean
noncomputable def bwdRowVectorStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem ValuePre (bwdRowVectorOffset s stride_out_row i)
```
</details>

## Public theorem: `layer_norm_ops_fwd_residual_y_store_slice_compute_correct`

<details><summary>docstring</summary>

```
/-- Named forward Y writeback for the residual-without-bias branch of
`_layer_norm_fwd_1pass_kernel`. The residual-combined normalization arithmetic
is represented by `ValuePre`; this theorem exposes the final masked Y store. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_ops_fwd_residual_y_store_slice_compute_correct
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => fwdYOffset s stride_y_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_y_store_slice ValuePre Y
        stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, fwdYOffset s stride_y_row i)))
      (expected := fun i => fwdYStoreSpec s ValuePre stride_y_row i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => fwdYOffset s stride_y_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `fwdYOffset`, `layer_norm_ops_fwd_y_store_slice`, `fwdYStoreSpec`, `bwdRowVectorOffset`, `layer_norm_ops_bwd_row_vector_store_slice`, `bwdRowVectorStoreSpec`

<details><summary><code>fwdYOffset</code></summary>

```lean
def fwdYOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  bwdRowVectorOffset s stride_y_row i
```
</details>

<details><summary><code>layer_norm_ops_fwd_y_store_slice</code></summary>

```
/-- Forward output writeback slice for `_layer_norm_fwd_1pass_kernel`.

After the mean/rstd arithmetic and optional bias/residual branches, Python
stores the computed row tile `y` into `Y + cols`. This aliases the generic
row-vector writeback proof with the forward output stride. -/
```
```lean
abbrev layer_norm_ops_fwd_y_store_slice
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_row_vector_store_slice ValuePre Y stride_y_row N BLOCK_N
```
</details>

<details><summary><code>fwdYStoreSpec</code></summary>

```lean
noncomputable def fwdYStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_y_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  bwdRowVectorStoreSpec s ValuePre stride_y_row i
```
</details>

<details><summary><code>bwdRowVectorOffset</code></summary>

```lean
def bwdRowVectorOffset (s : BlockState) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_out_row + i.val
```
</details>

<details><summary><code>layer_norm_ops_bwd_row_vector_store_slice</code></summary>

```
/-- Backward row-vector writeback slice for `DX`, optional `DRESIDUAL_IN`, and
optional recomputed `Y`. The caller supplies the precomputed row tile in
`ValuePre`; the slice proves the masked store to `Out + row * stride + cols`. -/
```
```lean
def layer_norm_ops_bwd_row_vector_store_slice
    (ValuePre Out : RegionName) (stride_out_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  value = tl.load(ValuePre + row * $(stride_out_row) + cols,
    mask=mask, other=0.0)
  tl.store(Out + row * $(stride_out_row) + cols, value, mask=mask)
}
```
</details>

<details><summary><code>bwdRowVectorStoreSpec</code></summary>

```lean
noncomputable def bwdRowVectorStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem ValuePre (bwdRowVectorOffset s stride_out_row i)
```
</details>

## Public theorem: `layer_norm_ops_fwd_rms_bias_y_store_slice_compute_correct`

<details><summary>docstring</summary>

```
/-- Named forward Y writeback for the RMS-norm plus bias branch tested by
`layer_norm_ops.py`. RMS-specific arithmetic is represented by `ValuePre`; the
store address and mask are the Python row-vector writeback. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_ops_fwd_rms_bias_y_store_slice_compute_correct
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => fwdYOffset s stride_y_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_y_store_slice ValuePre Y
        stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, fwdYOffset s stride_y_row i)))
      (expected := fun i => fwdYStoreSpec s ValuePre stride_y_row i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => fwdYOffset s stride_y_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `fwdYOffset`, `layer_norm_ops_fwd_y_store_slice`, `fwdYStoreSpec`, `bwdRowVectorOffset`, `layer_norm_ops_bwd_row_vector_store_slice`, `bwdRowVectorStoreSpec`

<details><summary><code>fwdYOffset</code></summary>

```lean
def fwdYOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  bwdRowVectorOffset s stride_y_row i
```
</details>

<details><summary><code>layer_norm_ops_fwd_y_store_slice</code></summary>

```
/-- Forward output writeback slice for `_layer_norm_fwd_1pass_kernel`.

After the mean/rstd arithmetic and optional bias/residual branches, Python
stores the computed row tile `y` into `Y + cols`. This aliases the generic
row-vector writeback proof with the forward output stride. -/
```
```lean
abbrev layer_norm_ops_fwd_y_store_slice
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_row_vector_store_slice ValuePre Y stride_y_row N BLOCK_N
```
</details>

<details><summary><code>fwdYStoreSpec</code></summary>

```lean
noncomputable def fwdYStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_y_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  bwdRowVectorStoreSpec s ValuePre stride_y_row i
```
</details>

<details><summary><code>bwdRowVectorOffset</code></summary>

```lean
def bwdRowVectorOffset (s : BlockState) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_out_row + i.val
```
</details>

<details><summary><code>layer_norm_ops_bwd_row_vector_store_slice</code></summary>

```
/-- Backward row-vector writeback slice for `DX`, optional `DRESIDUAL_IN`, and
optional recomputed `Y`. The caller supplies the precomputed row tile in
`ValuePre`; the slice proves the masked store to `Out + row * stride + cols`. -/
```
```lean
def layer_norm_ops_bwd_row_vector_store_slice
    (ValuePre Out : RegionName) (stride_out_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  value = tl.load(ValuePre + row * $(stride_out_row) + cols,
    mask=mask, other=0.0)
  tl.store(Out + row * $(stride_out_row) + cols, value, mask=mask)
}
```
</details>

<details><summary><code>bwdRowVectorStoreSpec</code></summary>

```lean
noncomputable def bwdRowVectorStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem ValuePre (bwdRowVectorOffset s stride_out_row i)
```
</details>

## Public theorem: `layer_norm_ops_fwd_residual_bias_y_store_slice_compute_correct`

<details><summary>docstring</summary>

```
/-- Named forward Y writeback for the residual-plus-bias branch of
`_layer_norm_fwd_1pass_kernel`.

The residual add and bias add are represented by the supplied `ValuePre` tile;
the theorem proves the final masked Y writeback under the Python row layout. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_ops_fwd_residual_bias_y_store_slice_compute_correct
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => fwdYOffset s stride_y_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_y_store_slice ValuePre Y
        stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, fwdYOffset s stride_y_row i)))
      (expected := fun i => fwdYStoreSpec s ValuePre stride_y_row i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => fwdYOffset s stride_y_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `fwdYOffset`, `layer_norm_ops_fwd_y_store_slice`, `fwdYStoreSpec`, `bwdRowVectorOffset`, `layer_norm_ops_bwd_row_vector_store_slice`, `bwdRowVectorStoreSpec`

<details><summary><code>fwdYOffset</code></summary>

```lean
def fwdYOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  bwdRowVectorOffset s stride_y_row i
```
</details>

<details><summary><code>layer_norm_ops_fwd_y_store_slice</code></summary>

```
/-- Forward output writeback slice for `_layer_norm_fwd_1pass_kernel`.

After the mean/rstd arithmetic and optional bias/residual branches, Python
stores the computed row tile `y` into `Y + cols`. This aliases the generic
row-vector writeback proof with the forward output stride. -/
```
```lean
abbrev layer_norm_ops_fwd_y_store_slice
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_row_vector_store_slice ValuePre Y stride_y_row N BLOCK_N
```
</details>

<details><summary><code>fwdYStoreSpec</code></summary>

```lean
noncomputable def fwdYStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_y_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  bwdRowVectorStoreSpec s ValuePre stride_y_row i
```
</details>

<details><summary><code>bwdRowVectorOffset</code></summary>

```lean
def bwdRowVectorOffset (s : BlockState) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_out_row + i.val
```
</details>

<details><summary><code>layer_norm_ops_bwd_row_vector_store_slice</code></summary>

```
/-- Backward row-vector writeback slice for `DX`, optional `DRESIDUAL_IN`, and
optional recomputed `Y`. The caller supplies the precomputed row tile in
`ValuePre`; the slice proves the masked store to `Out + row * stride + cols`. -/
```
```lean
def layer_norm_ops_bwd_row_vector_store_slice
    (ValuePre Out : RegionName) (stride_out_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  value = tl.load(ValuePre + row * $(stride_out_row) + cols,
    mask=mask, other=0.0)
  tl.store(Out + row * $(stride_out_row) + cols, value, mask=mask)
}
```
</details>

<details><summary><code>bwdRowVectorStoreSpec</code></summary>

```lean
noncomputable def bwdRowVectorStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem ValuePre (bwdRowVectorOffset s stride_out_row i)
```
</details>

## Public theorem: `layer_norm_ops_bwd_param_grad_store_slice_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_bwd_param_grad_store_slice_compute_correct
    (GradPre Out : RegionName) (N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_param_grad_store_slice GradPre Out N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Out, bwdParamGradOffset s N i)))
      (expected := fun i => bwdParamGradStoreSpec s GradPre N i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `bwdParamGradOffset`, `layer_norm_ops_bwd_param_grad_store_slice`, `bwdParamGradStoreSpec`

<details><summary><code>bwdParamGradOffset</code></summary>

```lean
def bwdParamGradOffset (s : BlockState) (N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * N + i.val
```
</details>

<details><summary><code>layer_norm_ops_bwd_param_grad_store_slice</code></summary>

```
/-- Backward partial-gradient writeback slice for `DW` and `DB`. The full Python
kernel stores each program's partial gradient at
`Out + row_block_id * N + cols`; reduction across programs happens later in
Python. -/
```
```lean
def layer_norm_ops_bwd_param_grad_store_slice
    (GradPre Out : RegionName) (N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row_block_id = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  grad = tl.load(GradPre + row_block_id * $(N) + cols,
    mask=mask, other=0.0)
  tl.store(Out + row_block_id * $(N) + cols, grad, mask=mask)
}
```
</details>

<details><summary><code>bwdParamGradStoreSpec</code></summary>

```lean
noncomputable def bwdParamGradStoreSpec
    (s : BlockState) (GradPre : RegionName) (N : Nat) (i : Fin BLOCK_N) : ℝ :=
  s.readMem GradPre (bwdParamGradOffset s N i)
```
</details>

## Public theorem: `layer_norm_ops_bwd_dw_store_slice_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_bwd_dw_store_slice_compute_correct
    (DWPre DW : RegionName) (N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i)`

**Closed-form spec defs (transitive):** `bwdParamGradOffset`

<details><summary><code>bwdParamGradOffset</code></summary>

```lean
def bwdParamGradOffset (s : BlockState) (N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * N + i.val
```
</details>

## Public theorem: `layer_norm_ops_bwd_db_store_slice_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_bwd_db_store_slice_compute_correct
    (DBPre DB : RegionName) (N BLOCK_N : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i)`

**Closed-form spec defs (transitive):** `bwdParamGradOffset`

<details><summary><code>bwdParamGradOffset</code></summary>

```lean
def bwdParamGradOffset (s : BlockState) (N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * N + i.val
```
</details>

## Public theorem: `layer_norm_ops_bwd_rms_one_row_dw_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_bwd_rms_one_row_dw_compute_correct
    (X W DY DX DW Rstd : RegionName)
    (stride_x_row stride_dy_row stride_dx_row N BLOCK_N : Nat)
    (s : BlockState)
    (hDWDX : DW ≠ DX)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRmsDWOffset s N i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_rms_one_row X W DY DX DW Rstd
        stride_x_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DW, bwdRmsDWOffset s N i)))
      (expected := fun i =>
        bwdRmsDWSpec s X DY Rstd stride_x_row stride_dy_row N BLOCK_N i)
```

**Assumptions / layout contracts:**
- `hDWDX : DW ≠ DX`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRmsDWOffset s N i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `bwdRmsDWOffset`, `layer_norm_ops_bwd_rms_one_row`, `bwdRmsDWSpec`, `bwdRmsDYTile`, `bwdRmsXhatTile`, `bwdRmsDYOffset`, `bwdRmsXTile`, `bwdRmsXOffset`

<details><summary><code>bwdRmsDWOffset</code></summary>

```lean
def bwdRmsDWOffset (s : BlockState) (N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * N + i.val
```
</details>

<details><summary><code>layer_norm_ops_bwd_rms_one_row</code></summary>

```
/-! ## Backward RMS arithmetic slice

This slice removes one of the narrow-writeback gaps above: it computes the
tested RMS backward arithmetic for one row, not merely a precomputed `DX`/`DW`
copy. It specializes the full backward kernel to the no-bias/no-residual/
no-recompute RMS branch and one row per program. -/
```
```lean
def layer_norm_ops_bwd_rms_one_row
    (X W DY DX DW Rstd : RegionName)
    (stride_x_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  x = tl.load(X + row * $(stride_x_row) + cols, mask=mask, other=0.0).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  rstd = tl.load(Rstd + row)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  xhat = x * rstd
  xhat = tl.where(mask, xhat, 0.0)
  wdy = w * dy
  dw = dy * xhat
  c1 = tl.sum(xhat * wdy, axis=0) / $(N)
  dx = (wdy - xhat * c1) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, dx, mask=mask)
  tl.store(DW + row * $(N) + cols, dw, mask=mask)
}
```
</details>

<details><summary><code>bwdRmsDWSpec</code></summary>

```lean
noncomputable def bwdRmsDWSpec
    (s : BlockState) (X DY Rstd : RegionName)
    (stride_x_row stride_dy_row N BLOCK_N : Nat) (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun dy xhat => dy * xhat)
      ((bwdRmsDYTile s DY stride_dy_row N BLOCK_N).data (i, PUnit.unit))
      ((bwdRmsXhatTile s X Rstd stride_x_row N BLOCK_N).data (i, PUnit.unit)))
```
</details>

<details><summary><code>bwdRmsDYTile</code></summary>

```lean
noncomputable def bwdRmsDYTile
    (s : BlockState) (DY : RegionName) (stride_dy_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem DY (bwdRmsDYOffset s stride_dy_row idx.1))
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>bwdRmsXhatTile</code></summary>

```lean
noncomputable def bwdRmsXhatTile
    (s : BlockState) (X Rstd : RegionName)
    (stride_x_row N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        match (bwdRmsXTile s X stride_x_row N BLOCK_N).data idx with
        | some x => some (x * s.readMem Rstd s.pid)
        | none => none
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>bwdRmsDYOffset</code></summary>

```lean
def bwdRmsDYOffset (s : BlockState) (stride_dy_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dy_row + i.val
```
</details>

<details><summary><code>bwdRmsXTile</code></summary>

```lean
noncomputable def bwdRmsXTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem X (bwdRmsXOffset s stride_x_row idx.1))
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>bwdRmsXOffset</code></summary>

```lean
def bwdRmsXOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

## Public theorem: `layer_norm_ops_bwd_bias_db_one_row_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_bwd_bias_db_one_row_compute_correct
    (DY DB : RegionName) (stride_dy_row N BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_bias_db_one_row DY DB
        stride_dy_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DB, bwdParamGradOffset s N i)))
      (expected := fun i => bwdBiasDBSpec s DY stride_dy_row i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `bwdParamGradOffset`, `layer_norm_ops_bwd_bias_db_one_row`, `bwdBiasDBSpec`, `bwdRmsDYOffset`

<details><summary><code>bwdParamGradOffset</code></summary>

```lean
def bwdParamGradOffset (s : BlockState) (N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * N + i.val
```
</details>

<details><summary><code>layer_norm_ops_bwd_bias_db_one_row</code></summary>

```
/-! ## Backward bias-gradient arithmetic slice

For the `HAS_BIAS` branch of the backward kernel, each row contributes `dy` to
the partial `DB` accumulator before the per-program partial-gradient store.
This slice proves that arithmetic path directly for one row. -/
```
```lean
def layer_norm_ops_bwd_bias_db_one_row
    (DY DB : RegionName) (stride_dy_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  db = dy
  tl.store(DB + row * $(N) + cols, db, mask=mask)
}
```
</details>

<details><summary><code>bwdBiasDBSpec</code></summary>

```lean
noncomputable def bwdBiasDBSpec
    (s : BlockState) (DY : RegionName) (stride_dy_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem DY (bwdRmsDYOffset s stride_dy_row i)
```
</details>

<details><summary><code>bwdRmsDYOffset</code></summary>

```lean
def bwdRmsDYOffset (s : BlockState) (stride_dy_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dy_row + i.val
```
</details>

## Public theorem: `layer_norm_ops_bwd_plain_bias_one_row_dw_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_bwd_plain_bias_one_row_dw_compute_correct
    (X W DY DX DW DB Mean Rstd : RegionName)
    (stride_x_row stride_dy_row stride_dx_row N BLOCK_N : Nat)
    (s : BlockState)
    (hDWDX : DW ≠ DX)
    (hDWDB : DW ≠ DB)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_plain_bias_one_row X W DY DX DW DB
        Mean Rstd stride_x_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DW, bwdParamGradOffset s N i)))
      (expected := fun i =>
        bwdPlainBiasDWSpec s X DY Mean Rstd stride_x_row stride_dy_row
          N BLOCK_N i)
```

**Assumptions / layout contracts:**
- `hDWDX : DW ≠ DX`
- `hDWDB : DW ≠ DB`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `bwdParamGradOffset`, `layer_norm_ops_bwd_plain_bias_one_row`, `bwdPlainBiasDWSpec`, `bwdRmsDYTile`, `bwdPlainXhatTile`, `bwdRmsDYOffset`, `bwdRmsXOffset`

<details><summary><code>bwdParamGradOffset</code></summary>

```lean
def bwdParamGradOffset (s : BlockState) (N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * N + i.val
```
</details>

<details><summary><code>layer_norm_ops_bwd_plain_bias_one_row</code></summary>

```
/-! ## Backward non-RMS+bias one-row integrated slice

This is the first non-RMS integrated backward row slice for #133. Unlike the smaller
`DX`/`DW`/`DB` slices above, this kernel keeps the Python row arithmetic together:
`xhat`, `wdy`, partial `dw`, partial `db`, `c1`, `c2`, `dx`, then the three
observable stores `DX`, `DW`, and `DB`. The theorems here prove the integrated
`DW` and `DB` stores; `DX` still needs a reduction-heavy readback theorem. -/
```
```lean
def layer_norm_ops_bwd_plain_bias_one_row
    (X W DY DX DW DB Mean Rstd : RegionName)
    (stride_x_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  x = tl.load(X + row * $(stride_x_row) + cols, mask=mask, other=0.0).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  mean = tl.load(Mean + row)
  rstd = tl.load(Rstd + row)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  xhat = (x - mean) * rstd
  xhat = tl.where(mask, xhat, 0.0)
  wdy = w * dy
  dw = dy * xhat
  db = dy
  c1 = tl.sum(xhat * wdy, axis=0) / $(N)
  c2 = tl.sum(wdy, axis=0) / $(N)
  dx = (wdy - (xhat * c1 + c2)) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, dx, mask=mask)
  tl.store(DW + row * $(N) + cols, dw, mask=mask)
  tl.store(DB + row * $(N) + cols, db, mask=mask)
}
```
</details>

<details><summary><code>bwdPlainBiasDWSpec</code></summary>

```lean
noncomputable def bwdPlainBiasDWSpec
    (s : BlockState) (X DY Mean Rstd : RegionName)
    (stride_x_row stride_dy_row N BLOCK_N : Nat) (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun dy xhat => dy * xhat)
      ((bwdRmsDYTile s DY stride_dy_row N BLOCK_N).data (i, PUnit.unit))
      ((bwdPlainXhatTile s X Mean Rstd stride_x_row N BLOCK_N).data
        (i, PUnit.unit)))
```
</details>

<details><summary><code>bwdRmsDYTile</code></summary>

```lean
noncomputable def bwdRmsDYTile
    (s : BlockState) (DY : RegionName) (stride_dy_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem DY (bwdRmsDYOffset s stride_dy_row idx.1))
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>bwdPlainXhatTile</code></summary>

```lean
noncomputable def bwdPlainXhatTile
    (s : BlockState) (X Mean Rstd : RegionName)
    (stride_x_row N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        match
          match
            if idx.1.val < N then
              some (s.readMem X (bwdRmsXOffset s stride_x_row idx.1))
            else some (0.0 : ℝ)
          with
          | some x => some (x - s.readMem Mean s.pid)
          | none => none
        with
        | some x => some (x * s.readMem Rstd s.pid)
        | none => none
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>bwdRmsDYOffset</code></summary>

```lean
def bwdRmsDYOffset (s : BlockState) (stride_dy_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dy_row + i.val
```
</details>

<details><summary><code>bwdRmsXOffset</code></summary>

```lean
def bwdRmsXOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

## Public theorem: `layer_norm_ops_bwd_plain_bias_one_row_db_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_bwd_plain_bias_one_row_db_compute_correct
    (X W DY DX DW DB Mean Rstd : RegionName)
    (stride_x_row stride_dy_row stride_dx_row N BLOCK_N : Nat)
    (s : BlockState)
    (hDBDX : DB ≠ DX)
    (hDBDW : DB ≠ DW)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_plain_bias_one_row X W DY DX DW DB
        Mean Rstd stride_x_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DB, bwdParamGradOffset s N i)))
      (expected := fun i => bwdBiasDBSpec s DY stride_dy_row i)
```

**Assumptions / layout contracts:**
- `hDBDX : DB ≠ DX`
- `hDBDW : DB ≠ DW`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdParamGradOffset s N i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `bwdParamGradOffset`, `layer_norm_ops_bwd_plain_bias_one_row`, `bwdBiasDBSpec`, `bwdRmsDYOffset`

<details><summary><code>bwdParamGradOffset</code></summary>

```lean
def bwdParamGradOffset (s : BlockState) (N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * N + i.val
```
</details>

<details><summary><code>layer_norm_ops_bwd_plain_bias_one_row</code></summary>

```
/-! ## Backward non-RMS+bias one-row integrated slice

This is the first non-RMS integrated backward row slice for #133. Unlike the smaller
`DX`/`DW`/`DB` slices above, this kernel keeps the Python row arithmetic together:
`xhat`, `wdy`, partial `dw`, partial `db`, `c1`, `c2`, `dx`, then the three
observable stores `DX`, `DW`, and `DB`. The theorems here prove the integrated
`DW` and `DB` stores; `DX` still needs a reduction-heavy readback theorem. -/
```
```lean
def layer_norm_ops_bwd_plain_bias_one_row
    (X W DY DX DW DB Mean Rstd : RegionName)
    (stride_x_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  x = tl.load(X + row * $(stride_x_row) + cols, mask=mask, other=0.0).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  mean = tl.load(Mean + row)
  rstd = tl.load(Rstd + row)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  xhat = (x - mean) * rstd
  xhat = tl.where(mask, xhat, 0.0)
  wdy = w * dy
  dw = dy * xhat
  db = dy
  c1 = tl.sum(xhat * wdy, axis=0) / $(N)
  c2 = tl.sum(wdy, axis=0) / $(N)
  dx = (wdy - (xhat * c1 + c2)) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, dx, mask=mask)
  tl.store(DW + row * $(N) + cols, dw, mask=mask)
  tl.store(DB + row * $(N) + cols, db, mask=mask)
}
```
</details>

<details><summary><code>bwdBiasDBSpec</code></summary>

```lean
noncomputable def bwdBiasDBSpec
    (s : BlockState) (DY : RegionName) (stride_dy_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem DY (bwdRmsDYOffset s stride_dy_row i)
```
</details>

<details><summary><code>bwdRmsDYOffset</code></summary>

```lean
def bwdRmsDYOffset (s : BlockState) (stride_dy_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dy_row + i.val
```
</details>

## Public theorem: `layer_norm_ops_bwd_residual_add_store_slice_dx_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_bwd_residual_add_store_slice_dx_compute_correct
    (DXBase DRESIDUAL DX DRESIDUAL_IN : RegionName)
    (stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N : Nat)
    (s : BlockState)
    (hDXDresIn : DX ≠ DRESIDUAL_IN)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRmsDXOffset s stride_dx_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_residual_add_store_slice DXBase DRESIDUAL
        DX DRESIDUAL_IN stride_dx_row stride_dres_row stride_dres_in_row
        N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DX, bwdRmsDXOffset s stride_dx_row i)))
      (expected := fun i =>
        bwdResidualAddSpec s DXBase DRESIDUAL stride_dx_row stride_dres_row i)
```

**Assumptions / layout contracts:**
- `hDXDresIn : DX ≠ DRESIDUAL_IN`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRmsDXOffset s stride_dx_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `bwdRmsDXOffset`, `layer_norm_ops_bwd_residual_add_store_slice`, `bwdResidualAddSpec`, `bwdDResidualOffset`

<details><summary><code>bwdRmsDXOffset</code></summary>

```lean
def bwdRmsDXOffset (s : BlockState) (stride_dx_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dx_row + i.val
```
</details>

<details><summary><code>layer_norm_ops_bwd_residual_add_store_slice</code></summary>

```
/-! ## Backward residual-add arithmetic slice

When `HAS_DRESIDUAL` is true, the Python backward kernel adds the residual
gradient tile into the already-computed `dx`, optionally stores the combined
value to `DRESIDUAL_IN`, and always stores it to `DX`. This slice proves that
arithmetic/writeback path directly. -/
```
```lean
def layer_norm_ops_bwd_residual_add_store_slice
    (DXBase DRESIDUAL DX DRESIDUAL_IN : RegionName)
    (stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  dx = tl.load(DXBase + row * $(stride_dx_row) + cols, mask=mask, other=0.0)
  dres = tl.load(DRESIDUAL + row * $(stride_dres_row) + cols, mask=mask, other=0.0)
  out = dx + dres
  tl.store(DRESIDUAL_IN + row * $(stride_dres_in_row) + cols, out, mask=mask)
  tl.store(DX + row * $(stride_dx_row) + cols, out, mask=mask)
}
```
</details>

<details><summary><code>bwdResidualAddSpec</code></summary>

```lean
noncomputable def bwdResidualAddSpec
    (s : BlockState) (DXBase DRESIDUAL : RegionName)
    (stride_dx_row stride_dres_row : Nat) (i : Fin BLOCK_N) : ℝ :=
  s.readMem DXBase (bwdRmsDXOffset s stride_dx_row i) +
    s.readMem DRESIDUAL (bwdDResidualOffset s stride_dres_row i)
```
</details>

<details><summary><code>bwdDResidualOffset</code></summary>

```lean
def bwdDResidualOffset (s : BlockState) (stride_dres_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dres_row + i.val
```
</details>

## Public theorem: `layer_norm_ops_bwd_residual_add_store_slice_dresidual_in_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_bwd_residual_add_store_slice_dresidual_in_compute_correct
    (DXBase DRESIDUAL DX DRESIDUAL_IN : RegionName)
    (stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N : Nat)
    (s : BlockState)
    (hDresInDX : DRESIDUAL_IN ≠ DX)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdDResidualInOffset s stride_dres_in_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_residual_add_store_slice DXBase DRESIDUAL
        DX DRESIDUAL_IN stride_dx_row stride_dres_row stride_dres_in_row
        N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DRESIDUAL_IN, bwdDResidualInOffset s stride_dres_in_row i)))
      (expected := fun i =>
        bwdResidualAddSpec s DXBase DRESIDUAL stride_dx_row stride_dres_row i)
```

**Assumptions / layout contracts:**
- `hDresInDX : DRESIDUAL_IN ≠ DX`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdDResidualInOffset s stride_dres_in_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `bwdDResidualInOffset`, `layer_norm_ops_bwd_residual_add_store_slice`, `bwdResidualAddSpec`, `bwdRmsDXOffset`, `bwdDResidualOffset`

<details><summary><code>bwdDResidualInOffset</code></summary>

```lean
def bwdDResidualInOffset (s : BlockState) (stride_dres_in_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dres_in_row + i.val
```
</details>

<details><summary><code>layer_norm_ops_bwd_residual_add_store_slice</code></summary>

```
/-! ## Backward residual-add arithmetic slice

When `HAS_DRESIDUAL` is true, the Python backward kernel adds the residual
gradient tile into the already-computed `dx`, optionally stores the combined
value to `DRESIDUAL_IN`, and always stores it to `DX`. This slice proves that
arithmetic/writeback path directly. -/
```
```lean
def layer_norm_ops_bwd_residual_add_store_slice
    (DXBase DRESIDUAL DX DRESIDUAL_IN : RegionName)
    (stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  dx = tl.load(DXBase + row * $(stride_dx_row) + cols, mask=mask, other=0.0)
  dres = tl.load(DRESIDUAL + row * $(stride_dres_row) + cols, mask=mask, other=0.0)
  out = dx + dres
  tl.store(DRESIDUAL_IN + row * $(stride_dres_in_row) + cols, out, mask=mask)
  tl.store(DX + row * $(stride_dx_row) + cols, out, mask=mask)
}
```
</details>

<details><summary><code>bwdResidualAddSpec</code></summary>

```lean
noncomputable def bwdResidualAddSpec
    (s : BlockState) (DXBase DRESIDUAL : RegionName)
    (stride_dx_row stride_dres_row : Nat) (i : Fin BLOCK_N) : ℝ :=
  s.readMem DXBase (bwdRmsDXOffset s stride_dx_row i) +
    s.readMem DRESIDUAL (bwdDResidualOffset s stride_dres_row i)
```
</details>

<details><summary><code>bwdRmsDXOffset</code></summary>

```lean
def bwdRmsDXOffset (s : BlockState) (stride_dx_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dx_row + i.val
```
</details>

<details><summary><code>bwdDResidualOffset</code></summary>

```lean
def bwdDResidualOffset (s : BlockState) (stride_dres_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dres_row + i.val
```
</details>

## Public theorem: `layer_norm_ops_bwd_recompute_y_bias_slice_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_bwd_recompute_y_bias_slice_compute_correct
    (Xhat W B Y : RegionName)
    (stride_xhat_row stride_y_row N BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_y_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_recompute_y_bias_slice Xhat W B Y
        stride_xhat_row stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, bwdRowVectorOffset s stride_y_row i)))
      (expected := fun i =>
        bwdRecomputeYBiasSpec s Xhat W B stride_xhat_row i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_y_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `bwdRowVectorOffset`, `layer_norm_ops_bwd_recompute_y_bias_slice`, `bwdRecomputeYBiasSpec`, `bwdRecomputeXhatOffset`

<details><summary><code>bwdRowVectorOffset</code></summary>

```lean
def bwdRowVectorOffset (s : BlockState) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_out_row + i.val
```
</details>

<details><summary><code>layer_norm_ops_bwd_recompute_y_bias_slice</code></summary>

```
/-! ## Backward recomputed-output arithmetic slice

When `RECOMPUTE_OUTPUT` and `HAS_BIAS` are true, the backward kernel recomputes
`Y` as `xhat * W + B` before continuing with gradient arithmetic. This slice
starts after `xhat` has been produced and proves that branch's masked `Y`
writeback with the Python row layout. -/
```
```lean
def layer_norm_ops_bwd_recompute_y_bias_slice
    (Xhat W B Y : RegionName)
    (stride_xhat_row stride_y_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
  y = xhat * w + b
  tl.store(Y + row * $(stride_y_row) + cols, y, mask=mask)
}
```
</details>

<details><summary><code>bwdRecomputeYBiasSpec</code></summary>

```lean
noncomputable def bwdRecomputeYBiasSpec
    (s : BlockState) (Xhat W B : RegionName) (stride_xhat_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem Xhat (bwdRecomputeXhatOffset s stride_xhat_row i) *
    s.readMem W i.val + s.readMem B i.val
```
</details>

<details><summary><code>bwdRecomputeXhatOffset</code></summary>

```lean
def bwdRecomputeXhatOffset (s : BlockState) (stride_xhat_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_xhat_row + i.val
```
</details>

## Public theorem: `layer_norm_ops_bwd_recompute_y_no_bias_slice_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_bwd_recompute_y_no_bias_slice_compute_correct
    (Xhat W Y : RegionName)
    (stride_xhat_row stride_y_row N BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_y_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_recompute_y_no_bias_slice Xhat W Y
        stride_xhat_row stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, bwdRowVectorOffset s stride_y_row i)))
      (expected := fun i =>
        bwdRecomputeYNoBiasSpec s Xhat W stride_xhat_row i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRowVectorOffset s stride_y_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `bwdRowVectorOffset`, `layer_norm_ops_bwd_recompute_y_no_bias_slice`, `bwdRecomputeYNoBiasSpec`, `bwdRecomputeXhatOffset`

<details><summary><code>bwdRowVectorOffset</code></summary>

```lean
def bwdRowVectorOffset (s : BlockState) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_out_row + i.val
```
</details>

<details><summary><code>layer_norm_ops_bwd_recompute_y_no_bias_slice</code></summary>

```
/-- No-bias recomputed-output branch: `y = xhat * W`. -/
```
```lean
def layer_norm_ops_bwd_recompute_y_no_bias_slice
    (Xhat W Y : RegionName)
    (stride_xhat_row stride_y_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  y = xhat * w
  tl.store(Y + row * $(stride_y_row) + cols, y, mask=mask)
}
```
</details>

<details><summary><code>bwdRecomputeYNoBiasSpec</code></summary>

```lean
noncomputable def bwdRecomputeYNoBiasSpec
    (s : BlockState) (Xhat W : RegionName) (stride_xhat_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem Xhat (bwdRecomputeXhatOffset s stride_xhat_row i) *
    s.readMem W i.val
```
</details>

<details><summary><code>bwdRecomputeXhatOffset</code></summary>

```lean
def bwdRecomputeXhatOffset (s : BlockState) (stride_xhat_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_xhat_row + i.val
```
</details>

## Public theorem: `layer_norm_ops_bwd_c1_reduction_slice_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_bwd_c1_reduction_slice_compute_correct
    (Xhat W DY C1 : RegionName)
    (stride_xhat_row stride_dy_row N BLOCK_N : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_c1_reduction_slice Xhat W DY C1
        stride_xhat_row stride_dy_row N BLOCK_N)
      (initialState := s)
      (write := fun _ : PUnit => some (C1, s.pid))
      (expected := fun _ =>
        bwdC1ReductionSpec s Xhat W DY stride_xhat_row stride_dy_row
          N BLOCK_N)
```

**Closed-form spec defs (transitive):** `layer_norm_ops_bwd_c1_reduction_slice`, `bwdC1ReductionSpec`, `bwdRecomputeXhatOffset`, `bwdRmsWdyTile`, `bwdRmsWTile`, `bwdRmsDYTile`, `bwdRmsDYOffset`

<details><summary><code>layer_norm_ops_bwd_c1_reduction_slice</code></summary>

```
/-! ## Backward `c1`/`c2` reduction-production slices

The full backward kernel computes `c1 = sum(xhat * (W * DY)) / N` in both RMS
and non-RMS branches, and additionally computes `c2 = sum(W * DY) / N` in the
non-RMS branch. These slices isolate the two row-local reductions and their
scalar stores so the reduction production itself is checked independently of
the surrounding branch/control-flow proof. -/
```
```lean
def layer_norm_ops_bwd_c1_reduction_slice
    (Xhat W DY C1 : RegionName)
    (stride_xhat_row stride_dy_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  wdy = w * dy
  c1 = tl.sum(xhat * wdy, axis=0) / $(N)
  tl.store(C1 + row, c1)
}
```
</details>

<details><summary><code>bwdC1ReductionSpec</code></summary>

```lean
noncomputable def bwdC1ReductionSpec
    (s : BlockState) (Xhat W DY : RegionName)
    (stride_xhat_row stride_dy_row N BLOCK_N : Nat) : ℝ :=
  WithBot.unbotD 0
    (Option.map (fun a => a / (N : ℝ))
      ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
        (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
          { data := fun idx =>
              if idx.1.val < N then
                some (s.readMem Xhat
                  (bwdRecomputeXhatOffset s stride_xhat_row idx.1))
              else some (0.0 : ℝ) }
          (bwdRmsWdyTile s W DY stride_dy_row N BLOCK_N))).data PUnit.unit))
```
</details>

<details><summary><code>bwdRecomputeXhatOffset</code></summary>

```lean
def bwdRecomputeXhatOffset (s : BlockState) (stride_xhat_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_xhat_row + i.val
```
</details>

<details><summary><code>bwdRmsWdyTile</code></summary>

```lean
noncomputable def bwdRmsWdyTile
    (s : BlockState) (W DY : RegionName)
    (stride_dy_row N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
    (bwdRmsWTile s W N BLOCK_N)
    (bwdRmsDYTile s DY stride_dy_row N BLOCK_N)
```
</details>

<details><summary><code>bwdRmsWTile</code></summary>

```lean
noncomputable def bwdRmsWTile
    (s : BlockState) (W : RegionName) (N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem W idx.1.val)
      else some (s.undef W idx.1.val) }
```
</details>

<details><summary><code>bwdRmsDYTile</code></summary>

```lean
noncomputable def bwdRmsDYTile
    (s : BlockState) (DY : RegionName) (stride_dy_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem DY (bwdRmsDYOffset s stride_dy_row idx.1))
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>bwdRmsDYOffset</code></summary>

```lean
def bwdRmsDYOffset (s : BlockState) (stride_dy_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dy_row + i.val
```
</details>

## Public theorem: `layer_norm_ops_bwd_c2_reduction_slice_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_bwd_c2_reduction_slice_compute_correct
    (W DY C2 : RegionName)
    (stride_dy_row N BLOCK_N : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_c2_reduction_slice W DY C2
        stride_dy_row N BLOCK_N)
      (initialState := s)
      (write := fun _ : PUnit => some (C2, s.pid))
      (expected := fun _ =>
        bwdC2ReductionSpec s W DY stride_dy_row N BLOCK_N)
```

**Closed-form spec defs (transitive):** `layer_norm_ops_bwd_c2_reduction_slice`, `bwdC2ReductionSpec`, `bwdRmsWdyTile`, `bwdRmsWTile`, `bwdRmsDYTile`, `bwdRmsDYOffset`

<details><summary><code>layer_norm_ops_bwd_c2_reduction_slice</code></summary>

```lean
def layer_norm_ops_bwd_c2_reduction_slice
    (W DY C2 : RegionName)
    (stride_dy_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  wdy = w * dy
  c2 = tl.sum(wdy, axis=0) / $(N)
  tl.store(C2 + row, c2)
}
```
</details>

<details><summary><code>bwdC2ReductionSpec</code></summary>

```lean
noncomputable def bwdC2ReductionSpec
    (s : BlockState) (W DY : RegionName)
    (stride_dy_row N BLOCK_N : Nat) : ℝ :=
  WithBot.unbotD 0
    (Option.map (fun a => a / (N : ℝ))
      ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
        (bwdRmsWdyTile s W DY stride_dy_row N BLOCK_N)).data PUnit.unit))
```
</details>

<details><summary><code>bwdRmsWdyTile</code></summary>

```lean
noncomputable def bwdRmsWdyTile
    (s : BlockState) (W DY : RegionName)
    (stride_dy_row N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
    (bwdRmsWTile s W N BLOCK_N)
    (bwdRmsDYTile s DY stride_dy_row N BLOCK_N)
```
</details>

<details><summary><code>bwdRmsWTile</code></summary>

```lean
noncomputable def bwdRmsWTile
    (s : BlockState) (W : RegionName) (N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem W idx.1.val)
      else some (s.undef W idx.1.val) }
```
</details>

<details><summary><code>bwdRmsDYTile</code></summary>

```lean
noncomputable def bwdRmsDYTile
    (s : BlockState) (DY : RegionName) (stride_dy_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem DY (bwdRmsDYOffset s stride_dy_row idx.1))
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>bwdRmsDYOffset</code></summary>

```lean
def bwdRmsDYOffset (s : BlockState) (stride_dy_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dy_row + i.val
```
</details>

## Public theorem: `layer_norm_ops_bwd_rms_dx_from_c1_slice_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_bwd_rms_dx_from_c1_slice_compute_correct
    (Xhat W DY Rstd C1 DX : RegionName)
    (stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRmsDXOffset s stride_dx_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_rms_dx_from_c1_slice Xhat W DY Rstd
        C1 DX stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DX, bwdRmsDXOffset s stride_dx_row i)))
      (expected := fun i =>
        bwdRmsDXFromC1Spec s Xhat W DY Rstd C1 stride_xhat_row stride_dy_row i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRmsDXOffset s stride_dx_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `bwdRmsDXOffset`, `layer_norm_ops_bwd_rms_dx_from_c1_slice`, `bwdRmsDXFromC1Spec`, `bwdRmsDYOffset`, `bwdRecomputeXhatOffset`

<details><summary><code>bwdRmsDXOffset</code></summary>

```lean
def bwdRmsDXOffset (s : BlockState) (stride_dx_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dx_row + i.val
```
</details>

<details><summary><code>layer_norm_ops_bwd_rms_dx_from_c1_slice</code></summary>

```
/-! ## Backward base-DX formula slice

After the RMS branch has produced `xhat` and reduced `c1`, Python computes
`dx = (w * dy - xhat * c1) * rstd`. This slice proves that formula and the
masked `DX` store directly, leaving only the production of `c1` itself outside
the theorem. -/
```
```lean
def layer_norm_ops_bwd_rms_dx_from_c1_slice
    (Xhat W DY Rstd C1 DX : RegionName)
    (stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  rstd = tl.load(Rstd + row)
  c1 = tl.load(C1 + row)
  wdy = w * dy
  dx = (wdy - xhat * c1) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, dx, mask=mask)
}
```
</details>

<details><summary><code>bwdRmsDXFromC1Spec</code></summary>

```lean
noncomputable def bwdRmsDXFromC1Spec
    (s : BlockState) (Xhat W DY Rstd C1 : RegionName)
    (stride_xhat_row stride_dy_row : Nat) (i : Fin BLOCK_N) : ℝ :=
  (s.readMem W i.val *
      s.readMem DY (bwdRmsDYOffset s stride_dy_row i) -
    s.readMem Xhat (bwdRecomputeXhatOffset s stride_xhat_row i) *
      s.readMem C1 s.pid) *
    s.readMem Rstd s.pid
```
</details>

<details><summary><code>bwdRmsDYOffset</code></summary>

```lean
def bwdRmsDYOffset (s : BlockState) (stride_dy_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dy_row + i.val
```
</details>

<details><summary><code>bwdRecomputeXhatOffset</code></summary>

```lean
def bwdRecomputeXhatOffset (s : BlockState) (stride_xhat_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_xhat_row + i.val
```
</details>

## Public theorem: `layer_norm_ops_bwd_plain_dx_from_c1_c2_slice_compute_correct`

**Statement:**
```lean
theorem layer_norm_ops_bwd_plain_dx_from_c1_c2_slice_compute_correct
    (Xhat W DY Rstd C1 C2 DX : RegionName)
    (stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRmsDXOffset s stride_dx_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_plain_dx_from_c1_c2_slice Xhat W DY
        Rstd C1 C2 DX stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DX, bwdRmsDXOffset s stride_dx_row i)))
      (expected := fun i =>
        bwdPlainDXFromC1C2Spec s Xhat W DY Rstd C1 C2
          stride_xhat_row stride_dy_row i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => bwdRmsDXOffset s stride_dx_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `bwdRmsDXOffset`, `layer_norm_ops_bwd_plain_dx_from_c1_c2_slice`, `bwdPlainDXFromC1C2Spec`, `bwdRmsDYOffset`, `bwdRecomputeXhatOffset`

<details><summary><code>bwdRmsDXOffset</code></summary>

```lean
def bwdRmsDXOffset (s : BlockState) (stride_dx_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dx_row + i.val
```
</details>

<details><summary><code>layer_norm_ops_bwd_plain_dx_from_c1_c2_slice</code></summary>

```
/-- Non-RMS base-DX formula after `c1` and `c2` have been reduced:
`dx = (wdy - (xhat * c1 + c2)) * rstd`. -/
```
```lean
def layer_norm_ops_bwd_plain_dx_from_c1_c2_slice
    (Xhat W DY Rstd C1 C2 DX : RegionName)
    (stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  rstd = tl.load(Rstd + row)
  c1 = tl.load(C1 + row)
  c2 = tl.load(C2 + row)
  wdy = w * dy
  dx = (wdy - (xhat * c1 + c2)) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, dx, mask=mask)
}
```
</details>

<details><summary><code>bwdPlainDXFromC1C2Spec</code></summary>

```lean
noncomputable def bwdPlainDXFromC1C2Spec
    (s : BlockState) (Xhat W DY Rstd C1 C2 : RegionName)
    (stride_xhat_row stride_dy_row : Nat) (i : Fin BLOCK_N) : ℝ :=
  (s.readMem W i.val *
      s.readMem DY (bwdRmsDYOffset s stride_dy_row i) -
    (s.readMem Xhat (bwdRecomputeXhatOffset s stride_xhat_row i) *
        s.readMem C1 s.pid +
      s.readMem C2 s.pid)) *
    s.readMem Rstd s.pid
```
</details>

<details><summary><code>bwdRmsDYOffset</code></summary>

```lean
def bwdRmsDYOffset (s : BlockState) (stride_dy_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dy_row + i.val
```
</details>

<details><summary><code>bwdRecomputeXhatOffset</code></summary>

```lean
def bwdRecomputeXhatOffset (s : BlockState) (stride_xhat_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_xhat_row + i.val
```
</details>

## Public theorem: `layer_norm_ops_fwd_plain_bias_all_outputs_compute_correct_general`

<details><summary>docstring</summary>

```
/-- Plain layer-norm forward, **dimension-general**: exposes the genuine `Y`,
`Mean`, and `Rstd` outputs together for arbitrary feature dim `N`, tile width
`BLOCK_N`, and `Y` row stride `stride_y_row`. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_ops_fwd_plain_bias_all_outputs_compute_correct_general
    (ValuePre MeanPre RstdPre Y Mean Rstd : RegionName) (s : BlockState)
    (stride_y_row N BLOCK_N : Nat) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_y_store_slice ValuePre Y
        stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, fwdYOffset s stride_y_row i)))
      (expected := fun i : Fin BLOCK_N =>
        fwdYStoreSpec s ValuePre stride_y_row i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_mean_store_slice MeanPre Mean)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, meanRowOffset s))
      (expected := fun _ => meanStoreSpec s MeanPre)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_rstd_store_slice RstdPre Rstd)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, meanRowOffset s))
      (expected := fun _ => rstdStoreSpec s RstdPre))
```

**Assumptions / layout contracts:**
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `layer_norm_ops_fwd_y_store_slice`, `fwdYOffset`, `fwdYStoreSpec`, `layer_norm_ops_fwd_mean_store_slice`, `meanRowOffset`, `meanStoreSpec`, `layer_norm_ops_fwd_rstd_store_slice`, `rstdStoreSpec`, `layer_norm_ops_bwd_row_vector_store_slice`, `bwdRowVectorOffset`, `bwdRowVectorStoreSpec`

<details><summary><code>layer_norm_ops_fwd_y_store_slice</code></summary>

```
/-- Forward output writeback slice for `_layer_norm_fwd_1pass_kernel`.

After the mean/rstd arithmetic and optional bias/residual branches, Python
stores the computed row tile `y` into `Y + cols`. This aliases the generic
row-vector writeback proof with the forward output stride. -/
```
```lean
abbrev layer_norm_ops_fwd_y_store_slice
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_row_vector_store_slice ValuePre Y stride_y_row N BLOCK_N
```
</details>

<details><summary><code>fwdYOffset</code></summary>

```lean
def fwdYOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  bwdRowVectorOffset s stride_y_row i
```
</details>

<details><summary><code>fwdYStoreSpec</code></summary>

```lean
noncomputable def fwdYStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_y_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  bwdRowVectorStoreSpec s ValuePre stride_y_row i
```
</details>

<details><summary><code>layer_norm_ops_fwd_mean_store_slice</code></summary>

```
/-- Proof-oriented Mean store slice of `layer_norm_ops.py`'s
`_layer_norm_fwd_1pass_kernel`. Takes a precomputed `MeanPre` scalar (per row)
and proves the unmasked scalar writeback into `Mean` at row offset. -/
```
```lean
def layer_norm_ops_fwd_mean_store_slice
    (MeanPre Mean : RegionName) : ComputeKernel := triton {
  row = tl.program_id(0)
  mean = tl.load(MeanPre + row)
  tl.store(Mean + row, mean)
}
```
</details>

<details><summary><code>meanRowOffset</code></summary>

```lean
def meanRowOffset (s : BlockState) : Nat := s.pid
```
</details>

<details><summary><code>meanStoreSpec</code></summary>

```lean
noncomputable def meanStoreSpec (s : BlockState) (MeanPre : RegionName) : ℝ :=
  s.readMem MeanPre s.pid
```
</details>

<details><summary><code>layer_norm_ops_fwd_rstd_store_slice</code></summary>

```
/-- Proof-oriented Rstd store slice. Same scalar-copy pattern. -/
```
```lean
def layer_norm_ops_fwd_rstd_store_slice
    (RstdPre Rstd : RegionName) : ComputeKernel := triton {
  row = tl.program_id(0)
  rstd = tl.load(RstdPre + row)
  tl.store(Rstd + row, rstd)
}
```
</details>

<details><summary><code>rstdStoreSpec</code></summary>

```lean
noncomputable def rstdStoreSpec (s : BlockState) (RstdPre : RegionName) : ℝ :=
  s.readMem RstdPre s.pid
```
</details>

<details><summary><code>layer_norm_ops_bwd_row_vector_store_slice</code></summary>

```
/-- Backward row-vector writeback slice for `DX`, optional `DRESIDUAL_IN`, and
optional recomputed `Y`. The caller supplies the precomputed row tile in
`ValuePre`; the slice proves the masked store to `Out + row * stride + cols`. -/
```
```lean
def layer_norm_ops_bwd_row_vector_store_slice
    (ValuePre Out : RegionName) (stride_out_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  value = tl.load(ValuePre + row * $(stride_out_row) + cols,
    mask=mask, other=0.0)
  tl.store(Out + row * $(stride_out_row) + cols, value, mask=mask)
}
```
</details>

<details><summary><code>bwdRowVectorOffset</code></summary>

```lean
def bwdRowVectorOffset (s : BlockState) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_out_row + i.val
```
</details>

<details><summary><code>bwdRowVectorStoreSpec</code></summary>

```lean
noncomputable def bwdRowVectorStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem ValuePre (bwdRowVectorOffset s stride_out_row i)
```
</details>

## Public theorem: `layer_norm_ops_fwd_rms_bias_all_outputs_compute_correct_general`

<details><summary>docstring</summary>

```
/-- RMS layer-norm forward with bias, **dimension-general**: exposes the genuine
`Y` and `Rstd` outputs together. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_ops_fwd_rms_bias_all_outputs_compute_correct_general
    (ValuePre RstdPre Y Rstd : RegionName) (s : BlockState)
    (stride_y_row N BLOCK_N : Nat) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_y_store_slice ValuePre Y
        stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, fwdYOffset s stride_y_row i)))
      (expected := fun i : Fin BLOCK_N =>
        fwdYStoreSpec s ValuePre stride_y_row i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_rstd_store_slice RstdPre Rstd)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, meanRowOffset s))
      (expected := fun _ => rstdStoreSpec s RstdPre))
```

**Assumptions / layout contracts:**
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `layer_norm_ops_fwd_y_store_slice`, `fwdYOffset`, `fwdYStoreSpec`, `layer_norm_ops_fwd_rstd_store_slice`, `meanRowOffset`, `rstdStoreSpec`, `layer_norm_ops_bwd_row_vector_store_slice`, `bwdRowVectorOffset`, `bwdRowVectorStoreSpec`

<details><summary><code>layer_norm_ops_fwd_y_store_slice</code></summary>

```
/-- Forward output writeback slice for `_layer_norm_fwd_1pass_kernel`.

After the mean/rstd arithmetic and optional bias/residual branches, Python
stores the computed row tile `y` into `Y + cols`. This aliases the generic
row-vector writeback proof with the forward output stride. -/
```
```lean
abbrev layer_norm_ops_fwd_y_store_slice
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_row_vector_store_slice ValuePre Y stride_y_row N BLOCK_N
```
</details>

<details><summary><code>fwdYOffset</code></summary>

```lean
def fwdYOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  bwdRowVectorOffset s stride_y_row i
```
</details>

<details><summary><code>fwdYStoreSpec</code></summary>

```lean
noncomputable def fwdYStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_y_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  bwdRowVectorStoreSpec s ValuePre stride_y_row i
```
</details>

<details><summary><code>layer_norm_ops_fwd_rstd_store_slice</code></summary>

```
/-- Proof-oriented Rstd store slice. Same scalar-copy pattern. -/
```
```lean
def layer_norm_ops_fwd_rstd_store_slice
    (RstdPre Rstd : RegionName) : ComputeKernel := triton {
  row = tl.program_id(0)
  rstd = tl.load(RstdPre + row)
  tl.store(Rstd + row, rstd)
}
```
</details>

<details><summary><code>meanRowOffset</code></summary>

```lean
def meanRowOffset (s : BlockState) : Nat := s.pid
```
</details>

<details><summary><code>rstdStoreSpec</code></summary>

```lean
noncomputable def rstdStoreSpec (s : BlockState) (RstdPre : RegionName) : ℝ :=
  s.readMem RstdPre s.pid
```
</details>

<details><summary><code>layer_norm_ops_bwd_row_vector_store_slice</code></summary>

```
/-- Backward row-vector writeback slice for `DX`, optional `DRESIDUAL_IN`, and
optional recomputed `Y`. The caller supplies the precomputed row tile in
`ValuePre`; the slice proves the masked store to `Out + row * stride + cols`. -/
```
```lean
def layer_norm_ops_bwd_row_vector_store_slice
    (ValuePre Out : RegionName) (stride_out_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  value = tl.load(ValuePre + row * $(stride_out_row) + cols,
    mask=mask, other=0.0)
  tl.store(Out + row * $(stride_out_row) + cols, value, mask=mask)
}
```
</details>

<details><summary><code>bwdRowVectorOffset</code></summary>

```lean
def bwdRowVectorOffset (s : BlockState) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_out_row + i.val
```
</details>

<details><summary><code>bwdRowVectorStoreSpec</code></summary>

```lean
noncomputable def bwdRowVectorStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem ValuePre (bwdRowVectorOffset s stride_out_row i)
```
</details>

## Public theorem: `layer_norm_ops_fwd_residual_bias_all_outputs_compute_correct_general`

<details><summary>docstring</summary>

```
/-- Residual plain layer-norm forward with bias, **dimension-general**: exposes
the genuine `RESIDUAL_OUT`, `Y`, `Mean`, and `Rstd` outputs together for
arbitrary `RESIDUAL_OUT`/`Y` row strides, feature dim `N`, and tile width
`BLOCK_N`. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_ops_fwd_residual_bias_all_outputs_compute_correct_general
    (ResidualPre ValuePre MeanPre RstdPre RESIDUAL_OUT Y Mean Rstd : RegionName)
    (s : BlockState)
    (stride_res_out_row stride_y_row N BLOCK_N : Nat) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_residual_out_store_slice ResidualPre
        RESIDUAL_OUT stride_res_out_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (RESIDUAL_OUT, fwdResidualOutOffset s stride_res_out_row i)))
      (expected := fun i : Fin BLOCK_N =>
        fwdResidualOutStoreSpec s ResidualPre stride_res_out_row i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_y_store_slice ValuePre Y
        stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, fwdYOffset s stride_y_row i)))
      (expected := fun i : Fin BLOCK_N =>
        fwdYStoreSpec s ValuePre stride_y_row i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_mean_store_slice MeanPre Mean)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, meanRowOffset s))
      (expected := fun _ => meanStoreSpec s MeanPre)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_rstd_store_slice RstdPre Rstd)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, meanRowOffset s))
      (expected := fun _ => rstdStoreSpec s RstdPre))
```

**Assumptions / layout contracts:**
- `fun i : Fin BLOCK_N => i.val < N`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `layer_norm_ops_fwd_residual_out_store_slice`, `fwdResidualOutOffset`, `fwdResidualOutStoreSpec`, `layer_norm_ops_fwd_y_store_slice`, `fwdYOffset`, `fwdYStoreSpec`, `layer_norm_ops_fwd_mean_store_slice`, `meanRowOffset`, `meanStoreSpec`, `layer_norm_ops_fwd_rstd_store_slice`, `rstdStoreSpec`, `layer_norm_ops_bwd_row_vector_store_slice`, `bwdRowVectorOffset`, `bwdRowVectorStoreSpec`

<details><summary><code>layer_norm_ops_fwd_residual_out_store_slice</code></summary>

```
/-- Forward residual-output writeback slice for `_layer_norm_fwd_1pass_kernel`.

When `STORE_RESIDUAL_OUT` is true, Python stores the row tile `x` after the
optional residual add into `RESIDUAL_OUT + cols`. The caller supplies that
precomputed row tile in `ValuePre`; this theorem proves the same masked
row-vector writeback shape as the backward row-vector stores. -/
```
```lean
abbrev layer_norm_ops_fwd_residual_out_store_slice
    (ValuePre RESIDUAL_OUT : RegionName)
    (stride_res_out_row N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_row_vector_store_slice ValuePre RESIDUAL_OUT
    stride_res_out_row N BLOCK_N
```
</details>

<details><summary><code>fwdResidualOutOffset</code></summary>

```lean
def fwdResidualOutOffset (s : BlockState) (stride_res_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  bwdRowVectorOffset s stride_res_out_row i
```
</details>

<details><summary><code>fwdResidualOutStoreSpec</code></summary>

```lean
noncomputable def fwdResidualOutStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_res_out_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  bwdRowVectorStoreSpec s ValuePre stride_res_out_row i
```
</details>

<details><summary><code>layer_norm_ops_fwd_y_store_slice</code></summary>

```
/-- Forward output writeback slice for `_layer_norm_fwd_1pass_kernel`.

After the mean/rstd arithmetic and optional bias/residual branches, Python
stores the computed row tile `y` into `Y + cols`. This aliases the generic
row-vector writeback proof with the forward output stride. -/
```
```lean
abbrev layer_norm_ops_fwd_y_store_slice
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_row_vector_store_slice ValuePre Y stride_y_row N BLOCK_N
```
</details>

<details><summary><code>fwdYOffset</code></summary>

```lean
def fwdYOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  bwdRowVectorOffset s stride_y_row i
```
</details>

<details><summary><code>fwdYStoreSpec</code></summary>

```lean
noncomputable def fwdYStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_y_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  bwdRowVectorStoreSpec s ValuePre stride_y_row i
```
</details>

<details><summary><code>layer_norm_ops_fwd_mean_store_slice</code></summary>

```
/-- Proof-oriented Mean store slice of `layer_norm_ops.py`'s
`_layer_norm_fwd_1pass_kernel`. Takes a precomputed `MeanPre` scalar (per row)
and proves the unmasked scalar writeback into `Mean` at row offset. -/
```
```lean
def layer_norm_ops_fwd_mean_store_slice
    (MeanPre Mean : RegionName) : ComputeKernel := triton {
  row = tl.program_id(0)
  mean = tl.load(MeanPre + row)
  tl.store(Mean + row, mean)
}
```
</details>

<details><summary><code>meanRowOffset</code></summary>

```lean
def meanRowOffset (s : BlockState) : Nat := s.pid
```
</details>

<details><summary><code>meanStoreSpec</code></summary>

```lean
noncomputable def meanStoreSpec (s : BlockState) (MeanPre : RegionName) : ℝ :=
  s.readMem MeanPre s.pid
```
</details>

<details><summary><code>layer_norm_ops_fwd_rstd_store_slice</code></summary>

```
/-- Proof-oriented Rstd store slice. Same scalar-copy pattern. -/
```
```lean
def layer_norm_ops_fwd_rstd_store_slice
    (RstdPre Rstd : RegionName) : ComputeKernel := triton {
  row = tl.program_id(0)
  rstd = tl.load(RstdPre + row)
  tl.store(Rstd + row, rstd)
}
```
</details>

<details><summary><code>rstdStoreSpec</code></summary>

```lean
noncomputable def rstdStoreSpec (s : BlockState) (RstdPre : RegionName) : ℝ :=
  s.readMem RstdPre s.pid
```
</details>

<details><summary><code>layer_norm_ops_bwd_row_vector_store_slice</code></summary>

```
/-- Backward row-vector writeback slice for `DX`, optional `DRESIDUAL_IN`, and
optional recomputed `Y`. The caller supplies the precomputed row tile in
`ValuePre`; the slice proves the masked store to `Out + row * stride + cols`. -/
```
```lean
def layer_norm_ops_bwd_row_vector_store_slice
    (ValuePre Out : RegionName) (stride_out_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  value = tl.load(ValuePre + row * $(stride_out_row) + cols,
    mask=mask, other=0.0)
  tl.store(Out + row * $(stride_out_row) + cols, value, mask=mask)
}
```
</details>

<details><summary><code>bwdRowVectorOffset</code></summary>

```lean
def bwdRowVectorOffset (s : BlockState) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_out_row + i.val
```
</details>

<details><summary><code>bwdRowVectorStoreSpec</code></summary>

```lean
noncomputable def bwdRowVectorStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem ValuePre (bwdRowVectorOffset s stride_out_row i)
```
</details>

## Public theorem: `layer_norm_ops_bwd_rms_core_outputs_compute_correct_general`

<details><summary>docstring</summary>

```
/-- RMS backward, **dimension-general**: exposes the genuine `c1` reduction,
`DX`, and partial `DW` store slices together for arbitrary row strides, feature
dim `N`, and tile width `BLOCK_N`. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_ops_bwd_rms_core_outputs_compute_correct_general
    (X Xhat W DY Rstd C1 DX DW : RegionName) (s : BlockState)
    (stride_xhat_row stride_dy_row stride_x_row stride_dx_row N BLOCK_N : Nat)
    (hDWDX : DW ≠ DX) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_c1_reduction_slice Xhat W DY C1
        stride_xhat_row stride_dy_row N BLOCK_N)
      (initialState := s)
      (write := fun _ : PUnit => some (C1, s.pid))
      (expected := fun _ : PUnit =>
        bwdC1ReductionSpec s Xhat W DY stride_xhat_row stride_dy_row N BLOCK_N)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_rms_dx_from_c1_slice Xhat W DY Rstd C1 DX
        stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DX, bwdRmsDXOffset s stride_dx_row i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdRmsDXFromC1Spec s Xhat W DY Rstd C1 stride_xhat_row stride_dy_row i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_rms_one_row X W DY DX DW Rstd
        stride_x_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DW, bwdRmsDWOffset s N i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdRmsDWSpec s X DY Rstd stride_x_row stride_dy_row N BLOCK_N i))
```

**Assumptions / layout contracts:**
- `hDWDX : DW ≠ DX`
- `fun i : Fin BLOCK_N => i.val < N`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `layer_norm_ops_bwd_c1_reduction_slice`, `bwdC1ReductionSpec`, `layer_norm_ops_bwd_rms_dx_from_c1_slice`, `bwdRmsDXOffset`, `bwdRmsDXFromC1Spec`, `layer_norm_ops_bwd_rms_one_row`, `bwdRmsDWOffset`, `bwdRmsDWSpec`, `bwdRecomputeXhatOffset`, `bwdRmsWdyTile`, `bwdRmsDYOffset`, `bwdRmsDYTile`, `bwdRmsXhatTile`, `bwdRmsWTile`, `bwdRmsXTile`, `bwdRmsXOffset`

<details><summary><code>layer_norm_ops_bwd_c1_reduction_slice</code></summary>

```
/-! ## Backward `c1`/`c2` reduction-production slices

The full backward kernel computes `c1 = sum(xhat * (W * DY)) / N` in both RMS
and non-RMS branches, and additionally computes `c2 = sum(W * DY) / N` in the
non-RMS branch. These slices isolate the two row-local reductions and their
scalar stores so the reduction production itself is checked independently of
the surrounding branch/control-flow proof. -/
```
```lean
def layer_norm_ops_bwd_c1_reduction_slice
    (Xhat W DY C1 : RegionName)
    (stride_xhat_row stride_dy_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  wdy = w * dy
  c1 = tl.sum(xhat * wdy, axis=0) / $(N)
  tl.store(C1 + row, c1)
}
```
</details>

<details><summary><code>bwdC1ReductionSpec</code></summary>

```lean
noncomputable def bwdC1ReductionSpec
    (s : BlockState) (Xhat W DY : RegionName)
    (stride_xhat_row stride_dy_row N BLOCK_N : Nat) : ℝ :=
  WithBot.unbotD 0
    (Option.map (fun a => a / (N : ℝ))
      ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
        (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
          { data := fun idx =>
              if idx.1.val < N then
                some (s.readMem Xhat
                  (bwdRecomputeXhatOffset s stride_xhat_row idx.1))
              else some (0.0 : ℝ) }
          (bwdRmsWdyTile s W DY stride_dy_row N BLOCK_N))).data PUnit.unit))
```
</details>

<details><summary><code>layer_norm_ops_bwd_rms_dx_from_c1_slice</code></summary>

```
/-! ## Backward base-DX formula slice

After the RMS branch has produced `xhat` and reduced `c1`, Python computes
`dx = (w * dy - xhat * c1) * rstd`. This slice proves that formula and the
masked `DX` store directly, leaving only the production of `c1` itself outside
the theorem. -/
```
```lean
def layer_norm_ops_bwd_rms_dx_from_c1_slice
    (Xhat W DY Rstd C1 DX : RegionName)
    (stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  rstd = tl.load(Rstd + row)
  c1 = tl.load(C1 + row)
  wdy = w * dy
  dx = (wdy - xhat * c1) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, dx, mask=mask)
}
```
</details>

<details><summary><code>bwdRmsDXOffset</code></summary>

```lean
def bwdRmsDXOffset (s : BlockState) (stride_dx_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dx_row + i.val
```
</details>

<details><summary><code>bwdRmsDXFromC1Spec</code></summary>

```lean
noncomputable def bwdRmsDXFromC1Spec
    (s : BlockState) (Xhat W DY Rstd C1 : RegionName)
    (stride_xhat_row stride_dy_row : Nat) (i : Fin BLOCK_N) : ℝ :=
  (s.readMem W i.val *
      s.readMem DY (bwdRmsDYOffset s stride_dy_row i) -
    s.readMem Xhat (bwdRecomputeXhatOffset s stride_xhat_row i) *
      s.readMem C1 s.pid) *
    s.readMem Rstd s.pid
```
</details>

<details><summary><code>layer_norm_ops_bwd_rms_one_row</code></summary>

```
/-! ## Backward RMS arithmetic slice

This slice removes one of the narrow-writeback gaps above: it computes the
tested RMS backward arithmetic for one row, not merely a precomputed `DX`/`DW`
copy. It specializes the full backward kernel to the no-bias/no-residual/
no-recompute RMS branch and one row per program. -/
```
```lean
def layer_norm_ops_bwd_rms_one_row
    (X W DY DX DW Rstd : RegionName)
    (stride_x_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  x = tl.load(X + row * $(stride_x_row) + cols, mask=mask, other=0.0).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  rstd = tl.load(Rstd + row)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  xhat = x * rstd
  xhat = tl.where(mask, xhat, 0.0)
  wdy = w * dy
  dw = dy * xhat
  c1 = tl.sum(xhat * wdy, axis=0) / $(N)
  dx = (wdy - xhat * c1) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, dx, mask=mask)
  tl.store(DW + row * $(N) + cols, dw, mask=mask)
}
```
</details>

<details><summary><code>bwdRmsDWOffset</code></summary>

```lean
def bwdRmsDWOffset (s : BlockState) (N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * N + i.val
```
</details>

<details><summary><code>bwdRmsDWSpec</code></summary>

```lean
noncomputable def bwdRmsDWSpec
    (s : BlockState) (X DY Rstd : RegionName)
    (stride_x_row stride_dy_row N BLOCK_N : Nat) (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun dy xhat => dy * xhat)
      ((bwdRmsDYTile s DY stride_dy_row N BLOCK_N).data (i, PUnit.unit))
      ((bwdRmsXhatTile s X Rstd stride_x_row N BLOCK_N).data (i, PUnit.unit)))
```
</details>

<details><summary><code>bwdRecomputeXhatOffset</code></summary>

```lean
def bwdRecomputeXhatOffset (s : BlockState) (stride_xhat_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_xhat_row + i.val
```
</details>

<details><summary><code>bwdRmsWdyTile</code></summary>

```lean
noncomputable def bwdRmsWdyTile
    (s : BlockState) (W DY : RegionName)
    (stride_dy_row N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
    (bwdRmsWTile s W N BLOCK_N)
    (bwdRmsDYTile s DY stride_dy_row N BLOCK_N)
```
</details>

<details><summary><code>bwdRmsDYOffset</code></summary>

```lean
def bwdRmsDYOffset (s : BlockState) (stride_dy_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dy_row + i.val
```
</details>

<details><summary><code>bwdRmsDYTile</code></summary>

```lean
noncomputable def bwdRmsDYTile
    (s : BlockState) (DY : RegionName) (stride_dy_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem DY (bwdRmsDYOffset s stride_dy_row idx.1))
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>bwdRmsXhatTile</code></summary>

```lean
noncomputable def bwdRmsXhatTile
    (s : BlockState) (X Rstd : RegionName)
    (stride_x_row N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        match (bwdRmsXTile s X stride_x_row N BLOCK_N).data idx with
        | some x => some (x * s.readMem Rstd s.pid)
        | none => none
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>bwdRmsWTile</code></summary>

```lean
noncomputable def bwdRmsWTile
    (s : BlockState) (W : RegionName) (N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem W idx.1.val)
      else some (s.undef W idx.1.val) }
```
</details>

<details><summary><code>bwdRmsXTile</code></summary>

```lean
noncomputable def bwdRmsXTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem X (bwdRmsXOffset s stride_x_row idx.1))
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>bwdRmsXOffset</code></summary>

```lean
def bwdRmsXOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

## Public theorem: `layer_norm_ops_bwd_plain_bias_core_outputs_compute_correct_general`

<details><summary>docstring</summary>

```
/-- Plain+bias backward, **dimension-general**: exposes the genuine `c1`/`c2`
reductions, `DX`, partial `DW`, and partial `DB` store slices together for
arbitrary row strides, feature dim `N`, and tile width `BLOCK_N`. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_ops_bwd_plain_bias_core_outputs_compute_correct_general
    (X Xhat W DY DX DW DB Mean Rstd C1 C2 : RegionName) (s : BlockState)
    (stride_xhat_row stride_dy_row stride_x_row stride_dx_row N BLOCK_N : Nat)
    (hDWDX : DW ≠ DX) (hDWDB : DW ≠ DB)
    (hDBDX : DB ≠ DX) (hDBDW : DB ≠ DW) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_c1_reduction_slice Xhat W DY C1
        stride_xhat_row stride_dy_row N BLOCK_N)
      (initialState := s)
      (write := fun _ : PUnit => some (C1, s.pid))
      (expected := fun _ : PUnit =>
        bwdC1ReductionSpec s Xhat W DY stride_xhat_row stride_dy_row N BLOCK_N)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_c2_reduction_slice W DY C2
        stride_dy_row N BLOCK_N)
      (initialState := s)
      (write := fun _ : PUnit => some (C2, s.pid))
      (expected := fun _ : PUnit =>
        bwdC2ReductionSpec s W DY stride_dy_row N BLOCK_N)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_plain_dx_from_c1_c2_slice Xhat W DY Rstd
        C1 C2 DX stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DX, bwdRmsDXOffset s stride_dx_row i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdPlainDXFromC1C2Spec s Xhat W DY Rstd C1 C2 stride_xhat_row
          stride_dy_row i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_plain_bias_one_row X W DY DX DW DB Mean Rstd
        stride_x_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DW, bwdParamGradOffset s N i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdPlainBiasDWSpec s X DY Mean Rstd stride_x_row stride_dy_row N
          BLOCK_N i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_plain_bias_one_row X W DY DX DW DB Mean Rstd
        stride_x_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DB, bwdParamGradOffset s N i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdBiasDBSpec s DY stride_dy_row i))
```

**Assumptions / layout contracts:**
- `hDWDX : DW ≠ DX`
- `hDWDB : DW ≠ DB`
- `hDBDX : DB ≠ DX`
- `hDBDW : DB ≠ DW`
- `fun i : Fin BLOCK_N => i.val < N`
- `fun i : Fin BLOCK_N => i.val < N`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `layer_norm_ops_bwd_c1_reduction_slice`, `bwdC1ReductionSpec`, `layer_norm_ops_bwd_c2_reduction_slice`, `bwdC2ReductionSpec`, `layer_norm_ops_bwd_plain_dx_from_c1_c2_slice`, `bwdRmsDXOffset`, `bwdPlainDXFromC1C2Spec`, `layer_norm_ops_bwd_plain_bias_one_row`, `bwdParamGradOffset`, `bwdPlainBiasDWSpec`, `bwdBiasDBSpec`, `bwdRecomputeXhatOffset`, `bwdRmsWdyTile`, `bwdRmsDYOffset`, `bwdRmsDYTile`, `bwdPlainXhatTile`, `bwdRmsWTile`, `bwdRmsXOffset`

<details><summary><code>layer_norm_ops_bwd_c1_reduction_slice</code></summary>

```
/-! ## Backward `c1`/`c2` reduction-production slices

The full backward kernel computes `c1 = sum(xhat * (W * DY)) / N` in both RMS
and non-RMS branches, and additionally computes `c2 = sum(W * DY) / N` in the
non-RMS branch. These slices isolate the two row-local reductions and their
scalar stores so the reduction production itself is checked independently of
the surrounding branch/control-flow proof. -/
```
```lean
def layer_norm_ops_bwd_c1_reduction_slice
    (Xhat W DY C1 : RegionName)
    (stride_xhat_row stride_dy_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  wdy = w * dy
  c1 = tl.sum(xhat * wdy, axis=0) / $(N)
  tl.store(C1 + row, c1)
}
```
</details>

<details><summary><code>bwdC1ReductionSpec</code></summary>

```lean
noncomputable def bwdC1ReductionSpec
    (s : BlockState) (Xhat W DY : RegionName)
    (stride_xhat_row stride_dy_row N BLOCK_N : Nat) : ℝ :=
  WithBot.unbotD 0
    (Option.map (fun a => a / (N : ℝ))
      ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
        (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
          { data := fun idx =>
              if idx.1.val < N then
                some (s.readMem Xhat
                  (bwdRecomputeXhatOffset s stride_xhat_row idx.1))
              else some (0.0 : ℝ) }
          (bwdRmsWdyTile s W DY stride_dy_row N BLOCK_N))).data PUnit.unit))
```
</details>

<details><summary><code>layer_norm_ops_bwd_c2_reduction_slice</code></summary>

```lean
def layer_norm_ops_bwd_c2_reduction_slice
    (W DY C2 : RegionName)
    (stride_dy_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  wdy = w * dy
  c2 = tl.sum(wdy, axis=0) / $(N)
  tl.store(C2 + row, c2)
}
```
</details>

<details><summary><code>bwdC2ReductionSpec</code></summary>

```lean
noncomputable def bwdC2ReductionSpec
    (s : BlockState) (W DY : RegionName)
    (stride_dy_row N BLOCK_N : Nat) : ℝ :=
  WithBot.unbotD 0
    (Option.map (fun a => a / (N : ℝ))
      ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
        (bwdRmsWdyTile s W DY stride_dy_row N BLOCK_N)).data PUnit.unit))
```
</details>

<details><summary><code>layer_norm_ops_bwd_plain_dx_from_c1_c2_slice</code></summary>

```
/-- Non-RMS base-DX formula after `c1` and `c2` have been reduced:
`dx = (wdy - (xhat * c1 + c2)) * rstd`. -/
```
```lean
def layer_norm_ops_bwd_plain_dx_from_c1_c2_slice
    (Xhat W DY Rstd C1 C2 DX : RegionName)
    (stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  rstd = tl.load(Rstd + row)
  c1 = tl.load(C1 + row)
  c2 = tl.load(C2 + row)
  wdy = w * dy
  dx = (wdy - (xhat * c1 + c2)) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, dx, mask=mask)
}
```
</details>

<details><summary><code>bwdRmsDXOffset</code></summary>

```lean
def bwdRmsDXOffset (s : BlockState) (stride_dx_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dx_row + i.val
```
</details>

<details><summary><code>bwdPlainDXFromC1C2Spec</code></summary>

```lean
noncomputable def bwdPlainDXFromC1C2Spec
    (s : BlockState) (Xhat W DY Rstd C1 C2 : RegionName)
    (stride_xhat_row stride_dy_row : Nat) (i : Fin BLOCK_N) : ℝ :=
  (s.readMem W i.val *
      s.readMem DY (bwdRmsDYOffset s stride_dy_row i) -
    (s.readMem Xhat (bwdRecomputeXhatOffset s stride_xhat_row i) *
        s.readMem C1 s.pid +
      s.readMem C2 s.pid)) *
    s.readMem Rstd s.pid
```
</details>

<details><summary><code>layer_norm_ops_bwd_plain_bias_one_row</code></summary>

```
/-! ## Backward non-RMS+bias one-row integrated slice

This is the first non-RMS integrated backward row slice for #133. Unlike the smaller
`DX`/`DW`/`DB` slices above, this kernel keeps the Python row arithmetic together:
`xhat`, `wdy`, partial `dw`, partial `db`, `c1`, `c2`, `dx`, then the three
observable stores `DX`, `DW`, and `DB`. The theorems here prove the integrated
`DW` and `DB` stores; `DX` still needs a reduction-heavy readback theorem. -/
```
```lean
def layer_norm_ops_bwd_plain_bias_one_row
    (X W DY DX DW DB Mean Rstd : RegionName)
    (stride_x_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  x = tl.load(X + row * $(stride_x_row) + cols, mask=mask, other=0.0).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  mean = tl.load(Mean + row)
  rstd = tl.load(Rstd + row)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  xhat = (x - mean) * rstd
  xhat = tl.where(mask, xhat, 0.0)
  wdy = w * dy
  dw = dy * xhat
  db = dy
  c1 = tl.sum(xhat * wdy, axis=0) / $(N)
  c2 = tl.sum(wdy, axis=0) / $(N)
  dx = (wdy - (xhat * c1 + c2)) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, dx, mask=mask)
  tl.store(DW + row * $(N) + cols, dw, mask=mask)
  tl.store(DB + row * $(N) + cols, db, mask=mask)
}
```
</details>

<details><summary><code>bwdParamGradOffset</code></summary>

```lean
def bwdParamGradOffset (s : BlockState) (N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * N + i.val
```
</details>

<details><summary><code>bwdPlainBiasDWSpec</code></summary>

```lean
noncomputable def bwdPlainBiasDWSpec
    (s : BlockState) (X DY Mean Rstd : RegionName)
    (stride_x_row stride_dy_row N BLOCK_N : Nat) (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun dy xhat => dy * xhat)
      ((bwdRmsDYTile s DY stride_dy_row N BLOCK_N).data (i, PUnit.unit))
      ((bwdPlainXhatTile s X Mean Rstd stride_x_row N BLOCK_N).data
        (i, PUnit.unit)))
```
</details>

<details><summary><code>bwdBiasDBSpec</code></summary>

```lean
noncomputable def bwdBiasDBSpec
    (s : BlockState) (DY : RegionName) (stride_dy_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem DY (bwdRmsDYOffset s stride_dy_row i)
```
</details>

<details><summary><code>bwdRecomputeXhatOffset</code></summary>

```lean
def bwdRecomputeXhatOffset (s : BlockState) (stride_xhat_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_xhat_row + i.val
```
</details>

<details><summary><code>bwdRmsWdyTile</code></summary>

```lean
noncomputable def bwdRmsWdyTile
    (s : BlockState) (W DY : RegionName)
    (stride_dy_row N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
    (bwdRmsWTile s W N BLOCK_N)
    (bwdRmsDYTile s DY stride_dy_row N BLOCK_N)
```
</details>

<details><summary><code>bwdRmsDYOffset</code></summary>

```lean
def bwdRmsDYOffset (s : BlockState) (stride_dy_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dy_row + i.val
```
</details>

<details><summary><code>bwdRmsDYTile</code></summary>

```lean
noncomputable def bwdRmsDYTile
    (s : BlockState) (DY : RegionName) (stride_dy_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem DY (bwdRmsDYOffset s stride_dy_row idx.1))
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>bwdPlainXhatTile</code></summary>

```lean
noncomputable def bwdPlainXhatTile
    (s : BlockState) (X Mean Rstd : RegionName)
    (stride_x_row N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        match
          match
            if idx.1.val < N then
              some (s.readMem X (bwdRmsXOffset s stride_x_row idx.1))
            else some (0.0 : ℝ)
          with
          | some x => some (x - s.readMem Mean s.pid)
          | none => none
        with
        | some x => some (x * s.readMem Rstd s.pid)
        | none => none
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>bwdRmsWTile</code></summary>

```lean
noncomputable def bwdRmsWTile
    (s : BlockState) (W : RegionName) (N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem W idx.1.val)
      else some (s.undef W idx.1.val) }
```
</details>

<details><summary><code>bwdRmsXOffset</code></summary>

```lean
def bwdRmsXOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

## Public theorem: `layer_norm_ops_bwd_residual_add_all_outputs_compute_correct_general`

<details><summary>docstring</summary>

```
/-- Backward residual-add, **dimension-general**: exposes both genuine
observable row-vector stores (`DX` and `DRESIDUAL_IN`) produced by the
`dx += dres` branch, for arbitrary row strides, feature dim `N`, and tile width
`BLOCK_N`. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_ops_bwd_residual_add_all_outputs_compute_correct_general
    (DXBase DRESIDUAL DX DRESIDUAL_IN : RegionName) (s : BlockState)
    (stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N : Nat)
    (hDXDresIn : DX ≠ DRESIDUAL_IN)
    (hDresInDX : DRESIDUAL_IN ≠ DX) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_residual_add_store_slice DXBase DRESIDUAL
        DX DRESIDUAL_IN stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DX, bwdRmsDXOffset s stride_dx_row i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdResidualAddSpec s DXBase DRESIDUAL stride_dx_row stride_dres_row i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_residual_add_store_slice DXBase DRESIDUAL
        DX DRESIDUAL_IN stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DRESIDUAL_IN, bwdDResidualInOffset s stride_dres_in_row i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdResidualAddSpec s DXBase DRESIDUAL stride_dx_row stride_dres_row i))
```

**Assumptions / layout contracts:**
- `hDXDresIn : DX ≠ DRESIDUAL_IN`
- `hDresInDX : DRESIDUAL_IN ≠ DX`
- `fun i : Fin BLOCK_N => i.val < N`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `layer_norm_ops_bwd_residual_add_store_slice`, `bwdRmsDXOffset`, `bwdResidualAddSpec`, `bwdDResidualInOffset`, `bwdDResidualOffset`

<details><summary><code>layer_norm_ops_bwd_residual_add_store_slice</code></summary>

```
/-! ## Backward residual-add arithmetic slice

When `HAS_DRESIDUAL` is true, the Python backward kernel adds the residual
gradient tile into the already-computed `dx`, optionally stores the combined
value to `DRESIDUAL_IN`, and always stores it to `DX`. This slice proves that
arithmetic/writeback path directly. -/
```
```lean
def layer_norm_ops_bwd_residual_add_store_slice
    (DXBase DRESIDUAL DX DRESIDUAL_IN : RegionName)
    (stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  dx = tl.load(DXBase + row * $(stride_dx_row) + cols, mask=mask, other=0.0)
  dres = tl.load(DRESIDUAL + row * $(stride_dres_row) + cols, mask=mask, other=0.0)
  out = dx + dres
  tl.store(DRESIDUAL_IN + row * $(stride_dres_in_row) + cols, out, mask=mask)
  tl.store(DX + row * $(stride_dx_row) + cols, out, mask=mask)
}
```
</details>

<details><summary><code>bwdRmsDXOffset</code></summary>

```lean
def bwdRmsDXOffset (s : BlockState) (stride_dx_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dx_row + i.val
```
</details>

<details><summary><code>bwdResidualAddSpec</code></summary>

```lean
noncomputable def bwdResidualAddSpec
    (s : BlockState) (DXBase DRESIDUAL : RegionName)
    (stride_dx_row stride_dres_row : Nat) (i : Fin BLOCK_N) : ℝ :=
  s.readMem DXBase (bwdRmsDXOffset s stride_dx_row i) +
    s.readMem DRESIDUAL (bwdDResidualOffset s stride_dres_row i)
```
</details>

<details><summary><code>bwdDResidualInOffset</code></summary>

```lean
def bwdDResidualInOffset (s : BlockState) (stride_dres_in_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dres_in_row + i.val
```
</details>

<details><summary><code>bwdDResidualOffset</code></summary>

```lean
def bwdDResidualOffset (s : BlockState) (stride_dres_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dres_row + i.val
```
</details>

## Public theorem: `layer_norm_fwd_1pass_surface_rms_only_y_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the `Y` output of the surface's
RMS/no-residual/no-bias branch. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_fwd_1pass_surface_rms_only_y_compute_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s : BlockState)
    (hYRstd : Y ≠ Rstd) (hWRstd : W ≠ Rstd)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.true Bool.false Bool.false Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, yOffset s stride_y_row i)))
      (expected := fun i => rmsYSpec s X W stride_x_row N BLOCK_N eps i)
```

**Assumptions / layout contracts:**
- `hYRstd : Y ≠ Rstd`
- `hWRstd : W ≠ Rstd`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => yOffset s stride_y_row i)`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `yOffset`, `layer_norm_fwd_1pass_surface`, `rmsYSpec`, `xOffset`, `rmsInvCarrier`, `rmsVarCarrier`, `rmsInputTile`

<details><summary><code>yOffset</code></summary>

```lean
def yOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_y_row + i.val
```
</details>

<details><summary><code>layer_norm_fwd_1pass_surface</code></summary>

```
/-- Faithful transcription of `layer_norm_ops.py`'s `_layer_norm_fwd_1pass_kernel`.

This keeps the residual input, `RESIDUAL_OUT`, RMS-vs-layer norm, bias,
`Mean`/`Rstd` side stores, Python ternary expressions, and masked `Y`
writeback. -/
```
```lean
def layer_norm_fwd_1pass_surface
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row
      N BLOCK_N : Nat)
    (eps : ℝ)
    (IS_RMS_NORM HAS_RESIDUAL STORE_RESIDUAL_OUT HAS_BIAS : Bool) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  if HAS_RESIDUAL {
    RESIDUAL += row * $(stride_res_row)
  }
  if STORE_RESIDUAL_OUT {
    RESIDUAL_OUT += row * $(stride_res_out_row)
  }
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  if HAS_RESIDUAL {
    residual = tl.load(RESIDUAL + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    x += residual
  }
  if STORE_RESIDUAL_OUT {
    tl.store(RESIDUAL_OUT + cols, x, mask=cols < $(N))
  }
  tl.if not IS_RMS_NORM {
    mean = tl.sum(x, axis=0) / $(N)
    tl.store(Mean + row, mean)
    xbar = tl.where(cols < $(N), x - mean, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / $(N)
  } else {
    xbar = tl.where(cols < $(N), x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / $(N)
  }
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  tl.if HAS_BIAS {
    b = tl.load(B + cols, mask=mask).to(tl.float32)
  }
  x_hat = ((x - mean) * rstd if not IS_RMS_NORM else x * rstd)
  y = (x_hat * w + b if HAS_BIAS else x_hat * w)
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>rmsYSpec</code></summary>

```lean
noncomputable def rmsYSpec
    (s : BlockState) (X W : RegionName)
    (stride_x_row N BLOCK_N : Nat) (eps : ℝ) (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun x inv => x * inv)
        (some (s.readMem X (xOffset s stride_x_row i)))
        (rmsInvCarrier s X stride_x_row N BLOCK_N eps))
      (some (s.readMem W i.val)))
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

<details><summary><code>rmsInvCarrier</code></summary>

```lean
noncomputable def rmsInvCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (rmsVarCarrier s X stride_x_row N BLOCK_N)))
```
</details>

<details><summary><code>rmsVarCarrier</code></summary>

```lean
noncomputable def rmsVarCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsInputTile s X stride_x_row N BLOCK_N)
        (rmsInputTile s X stride_x_row N BLOCK_N))).data PUnit.unit)
```
</details>

<details><summary><code>rmsInputTile</code></summary>

```lean
noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        if idx.1.val < N then
          some (s.readMem X (xOffset s stride_x_row idx.1))
        else some (0.0 : ℝ)
      else some (0.0 : ℝ) }
```
</details>

## Public theorem: `layer_norm_fwd_1pass_surface_rms_only_rstd_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the `Rstd` output of the surface's
RMS/no-residual/no-bias branch. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_fwd_1pass_surface_rms_only_rstd_compute_correct
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row N BLOCK_N : Nat)
    (eps : ℝ) (s : BlockState)
    (hRstdY : Rstd ≠ Y) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_fwd_1pass_surface X Y W B RESIDUAL RESIDUAL_OUT
        Mean Rstd stride_x_row stride_y_row stride_res_row stride_res_out_row
        N BLOCK_N eps Bool.true Bool.false Bool.false Bool.false)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, s.pid))
      (expected := fun _ =>
        rmsInvVarFullSpec s X stride_x_row N BLOCK_N eps)
```

**Assumptions / layout contracts:**
- `hRstdY : Rstd ≠ Y`

**Closed-form spec defs (transitive):** `layer_norm_fwd_1pass_surface`, `rmsInvVarFullSpec`, `rmsInvCarrier`, `rmsVarCarrier`, `rmsInputTile`, `xOffset`

<details><summary><code>layer_norm_fwd_1pass_surface</code></summary>

```
/-- Faithful transcription of `layer_norm_ops.py`'s `_layer_norm_fwd_1pass_kernel`.

This keeps the residual input, `RESIDUAL_OUT`, RMS-vs-layer norm, bias,
`Mean`/`Rstd` side stores, Python ternary expressions, and masked `Y`
writeback. -/
```
```lean
def layer_norm_fwd_1pass_surface
    (X Y W B RESIDUAL RESIDUAL_OUT Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_res_row stride_res_out_row
      N BLOCK_N : Nat)
    (eps : ℝ)
    (IS_RMS_NORM HAS_RESIDUAL STORE_RESIDUAL_OUT HAS_BIAS : Bool) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  if HAS_RESIDUAL {
    RESIDUAL += row * $(stride_res_row)
  }
  if STORE_RESIDUAL_OUT {
    RESIDUAL_OUT += row * $(stride_res_out_row)
  }
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  if HAS_RESIDUAL {
    residual = tl.load(RESIDUAL + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    x += residual
  }
  if STORE_RESIDUAL_OUT {
    tl.store(RESIDUAL_OUT + cols, x, mask=cols < $(N))
  }
  tl.if not IS_RMS_NORM {
    mean = tl.sum(x, axis=0) / $(N)
    tl.store(Mean + row, mean)
    xbar = tl.where(cols < $(N), x - mean, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / $(N)
  } else {
    xbar = tl.where(cols < $(N), x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / $(N)
  }
  rstd = 1 / tl.sqrt(var + $(eps))
  tl.store(Rstd + row, rstd)
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  tl.if HAS_BIAS {
    b = tl.load(B + cols, mask=mask).to(tl.float32)
  }
  x_hat = ((x - mean) * rstd if not IS_RMS_NORM else x * rstd)
  y = (x_hat * w + b if HAS_BIAS else x_hat * w)
  tl.store(Y + cols, y, mask=mask)
}
```
</details>

<details><summary><code>rmsInvVarFullSpec</code></summary>

```
/-- Full-kernel spec for the `Rstd` scalar store of `layer_norm_fwd_rms_one_block`.
Wraps `rmsInvCarrier` with `WithBot.unbotD 0` to match the kernel's
post-execution scalar readback. -/
```
```lean
noncomputable def rmsInvVarFullSpec
    (s : BlockState) (X : RegionName)
    (stride_x_row N BLOCK_N : Nat) (eps : ℝ) : ℝ :=
  WithBot.unbotD 0
    (rmsInvCarrier s X stride_x_row N BLOCK_N eps)
```
</details>

<details><summary><code>rmsInvCarrier</code></summary>

```lean
noncomputable def rmsInvCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (rmsVarCarrier s X stride_x_row N BLOCK_N)))
```
</details>

<details><summary><code>rmsVarCarrier</code></summary>

```lean
noncomputable def rmsVarCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsInputTile s X stride_x_row N BLOCK_N)
        (rmsInputTile s X stride_x_row N BLOCK_N))).data PUnit.unit)
```
</details>

<details><summary><code>rmsInputTile</code></summary>

```lean
noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        if idx.1.val < N then
          some (s.readMem X (xOffset s stride_x_row idx.1))
        else some (0.0 : ℝ)
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

## Public theorem: `layer_norm_bwd_surface_zero_rows_db_zero_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing wrapper for the actual backward surface's degenerate
`DB` writeback. This exposes the surface-level `HAS_BIAS=true` partial-bias
store through the standard `ComputeCorrect.Realizes_without_Rounding` interface. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_bwd_surface_zero_rows_db_zero_compute_correct
    (X W B Y DY DX DW DB DRESIDUAL DRESIDUAL_IN Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_dy_row stride_dx_row stride_dres_row
      stride_dres_in_row M N BLOCK_N : Nat)
    (eps : ℝ) (s : BlockState)
    (hDBDW : DB ≠ DW) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_bwd_surface X W B Y DY DX DW DB DRESIDUAL
        DRESIDUAL_IN Mean Rstd stride_x_row stride_y_row stride_dy_row
        stride_dx_row stride_dres_row stride_dres_in_row M N 0 BLOCK_N eps
        Bool.true Bool.false Bool.false Bool.true Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DB, s.pid * N + i.val)))
      (expected := fun _ : Fin BLOCK_N => (0.0 : ℝ))
```

**Assumptions / layout contracts:**
- `hDBDW : DB ≠ DW`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `layer_norm_bwd_surface`

<details><summary><code>layer_norm_bwd_surface</code></summary>

```
/-- Faithful transcription of `layer_norm_ops.py`'s `_layer_norm_bwd_kernel`.

This keeps the row-block launch, optional `DRESIDUAL` input, optional
`DRESIDUAL_IN` output, RMS-vs-layer norm branch, optional bias/`DB`, optional
recomputed `Y`, row loop, `DX`, `DW`, and `DB` stores. `Mean` is always a
region argument on the Lean surface; as in Python, it is read only in the
non-RMS branch. -/
```
```lean
def layer_norm_bwd_surface
    (X W B Y DY DX DW DB DRESIDUAL DRESIDUAL_IN Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_dy_row stride_dx_row stride_dres_row
      stride_dres_in_row M N rows_per_program BLOCK_N : Nat)
    (eps : ℝ)
    (IS_RMS_NORM HAS_DRESIDUAL STORE_DRESIDUAL HAS_BIAS RECOMPUTE_OUTPUT :
      Bool) :
    ComputeKernel := triton {
  row_block_id = tl.program_id(0)
  row_start = row_block_id * $(rows_per_program)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  X += row_start * $(stride_x_row)
  if HAS_DRESIDUAL {
    DRESIDUAL += row_start * $(stride_dres_row)
  }
  if STORE_DRESIDUAL {
    DRESIDUAL_IN += row_start * $(stride_dres_in_row)
  }
  DY += row_start * $(stride_dy_row)
  DX += row_start * $(stride_dx_row)
  if RECOMPUTE_OUTPUT {
    Y += row_start * $(stride_y_row)
  }
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  if RECOMPUTE_OUTPUT {
    if HAS_BIAS {
      b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    }
  }
  dw = tl.zeros([$(BLOCK_N)], dtype=tl.float32)
  if HAS_BIAS {
    db = tl.zeros([$(BLOCK_N)], dtype=tl.float32)
  }
  row_end = min((row_block_id + $(1)) * $(rows_per_program), $(M))
  for row in range(row_start, row_end, $(1)) {
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    tl.if not IS_RMS_NORM {
      mean = tl.load(Mean + row)
    }
    rstd = tl.load(Rstd + row)
    xhat = ((x - mean) * rstd if not IS_RMS_NORM else x * rstd)
    xhat = tl.where(mask, xhat, 0.0)
    if RECOMPUTE_OUTPUT {
      y = (xhat * w + b if HAS_BIAS else xhat * w)
      tl.store(Y + cols, y, mask=mask)
    }
    wdy = w * dy
    dw += dy * xhat
    if HAS_BIAS {
      db += dy
    }
    tl.if not IS_RMS_NORM {
      c1 = tl.sum(xhat * wdy, axis=0) / $(N)
      c2 = tl.sum(wdy, axis=0) / $(N)
      dx = (wdy - (xhat * c1 + c2)) * rstd
    } else {
      c1 = tl.sum(xhat * wdy, axis=0) / $(N)
      dx = (wdy - xhat * c1) * rstd
    }
    if HAS_DRESIDUAL {
      dres = tl.load(DRESIDUAL + cols, mask=mask, other=0).to(tl.float32)
      dx += dres
    }
    if STORE_DRESIDUAL {
      tl.store(DRESIDUAL_IN + cols, dx, mask=mask)
    }
    tl.store(DX + cols, dx, mask=mask)
    X += $(stride_x_row)
    if HAS_DRESIDUAL {
      DRESIDUAL += $(stride_dres_row)
    }
    if STORE_DRESIDUAL {
      DRESIDUAL_IN += $(stride_dres_in_row)
    }
    if RECOMPUTE_OUTPUT {
      Y += $(stride_y_row)
    }
    DY += $(stride_dy_row)
    DX += $(stride_dx_row)
  }
  tl.store(DW + row_block_id * $(N) + cols, dw, mask=mask)
  if HAS_BIAS {
    tl.store(DB + row_block_id * $(N) + cols, db, mask=mask)
  }
}
```
</details>

## Public theorem: `layer_norm_bwd_surface_zero_rows_dw_zero_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing wrapper for the actual backward surface's degenerate
`DW` writeback. It proves the standard correctness surface for the full
backward DSL surface when the row loop has no iterations. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_bwd_surface_zero_rows_dw_zero_compute_correct
    (X W B Y DY DX DW DB DRESIDUAL DRESIDUAL_IN Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_dy_row stride_dx_row stride_dres_row
      stride_dres_in_row M N BLOCK_N : Nat)
    (eps : ℝ) (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_bwd_surface X W B Y DY DX DW DB DRESIDUAL
        DRESIDUAL_IN Mean Rstd stride_x_row stride_y_row stride_dy_row
        stride_dx_row stride_dres_row stride_dres_in_row M N 0 BLOCK_N eps
        Bool.true Bool.false Bool.false Bool.false Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DW, s.pid * N + i.val)))
      (expected := fun _ : Fin BLOCK_N => (0.0 : ℝ))
```

**Assumptions / layout contracts:**
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `layer_norm_bwd_surface`

<details><summary><code>layer_norm_bwd_surface</code></summary>

```
/-- Faithful transcription of `layer_norm_ops.py`'s `_layer_norm_bwd_kernel`.

This keeps the row-block launch, optional `DRESIDUAL` input, optional
`DRESIDUAL_IN` output, RMS-vs-layer norm branch, optional bias/`DB`, optional
recomputed `Y`, row loop, `DX`, `DW`, and `DB` stores. `Mean` is always a
region argument on the Lean surface; as in Python, it is read only in the
non-RMS branch. -/
```
```lean
def layer_norm_bwd_surface
    (X W B Y DY DX DW DB DRESIDUAL DRESIDUAL_IN Mean Rstd : RegionName)
    (stride_x_row stride_y_row stride_dy_row stride_dx_row stride_dres_row
      stride_dres_in_row M N rows_per_program BLOCK_N : Nat)
    (eps : ℝ)
    (IS_RMS_NORM HAS_DRESIDUAL STORE_DRESIDUAL HAS_BIAS RECOMPUTE_OUTPUT :
      Bool) :
    ComputeKernel := triton {
  row_block_id = tl.program_id(0)
  row_start = row_block_id * $(rows_per_program)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  X += row_start * $(stride_x_row)
  if HAS_DRESIDUAL {
    DRESIDUAL += row_start * $(stride_dres_row)
  }
  if STORE_DRESIDUAL {
    DRESIDUAL_IN += row_start * $(stride_dres_in_row)
  }
  DY += row_start * $(stride_dy_row)
  DX += row_start * $(stride_dx_row)
  if RECOMPUTE_OUTPUT {
    Y += row_start * $(stride_y_row)
  }
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  if RECOMPUTE_OUTPUT {
    if HAS_BIAS {
      b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    }
  }
  dw = tl.zeros([$(BLOCK_N)], dtype=tl.float32)
  if HAS_BIAS {
    db = tl.zeros([$(BLOCK_N)], dtype=tl.float32)
  }
  row_end = min((row_block_id + $(1)) * $(rows_per_program), $(M))
  for row in range(row_start, row_end, $(1)) {
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    tl.if not IS_RMS_NORM {
      mean = tl.load(Mean + row)
    }
    rstd = tl.load(Rstd + row)
    xhat = ((x - mean) * rstd if not IS_RMS_NORM else x * rstd)
    xhat = tl.where(mask, xhat, 0.0)
    if RECOMPUTE_OUTPUT {
      y = (xhat * w + b if HAS_BIAS else xhat * w)
      tl.store(Y + cols, y, mask=mask)
    }
    wdy = w * dy
    dw += dy * xhat
    if HAS_BIAS {
      db += dy
    }
    tl.if not IS_RMS_NORM {
      c1 = tl.sum(xhat * wdy, axis=0) / $(N)
      c2 = tl.sum(wdy, axis=0) / $(N)
      dx = (wdy - (xhat * c1 + c2)) * rstd
    } else {
      c1 = tl.sum(xhat * wdy, axis=0) / $(N)
      dx = (wdy - xhat * c1) * rstd
    }
    if HAS_DRESIDUAL {
      dres = tl.load(DRESIDUAL + cols, mask=mask, other=0).to(tl.float32)
      dx += dres
    }
    if STORE_DRESIDUAL {
      tl.store(DRESIDUAL_IN + cols, dx, mask=mask)
    }
    tl.store(DX + cols, dx, mask=mask)
    X += $(stride_x_row)
    if HAS_DRESIDUAL {
      DRESIDUAL += $(stride_dres_row)
    }
    if STORE_DRESIDUAL {
      DRESIDUAL_IN += $(stride_dres_in_row)
    }
    if RECOMPUTE_OUTPUT {
      Y += $(stride_y_row)
    }
    DY += $(stride_dy_row)
    DX += $(stride_dx_row)
  }
  tl.store(DW + row_block_id * $(N) + cols, dw, mask=mask)
  if HAS_BIAS {
    tl.store(DB + row_block_id * $(N) + cols, db, mask=mask)
  }
}
```
</details>
