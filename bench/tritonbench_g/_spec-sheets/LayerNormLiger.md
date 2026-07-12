# Spec sheet — `bench/tritonbench_g/layer_norm_liger/LayerNormLiger.lean`

**Python source:** `bench/tritonbench_g/layer_norm_liger/layer_norm_liger.py`

## Public theorem: `layer_norm_liger_forward_output_summary`

<details><summary>docstring</summary>

```
/-- Public forward summary for Liger layer norm: the full Python forward surface
lowers, and the checked forward kernel characterizes all Python-observable
forward outputs `Y`, `Mean`, and `RSTD`. -/
```
</details>

**Statement:**
```lean
specification layer_norm_liger_forward_output_summary
    (Y X W B Mean RSTD : RegionName)
    (Y_row_stride X_row_stride W_row_stride B_row_stride
      Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hYMean : Y ≠ Mean) (hYRSTD : Y ≠ RSTD)
    (hMeanY : Mean ≠ Y) (hMeanRSTD : Mean ≠ RSTD)
    (hRSTDY : RSTD ≠ Y)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s Y_row_stride i)) :
    (∃ alg, (layer_norm_liger_forward_surface Y Y_row_stride X
      X_row_stride W W_row_stride B B_row_stride Mean Mean_row_stride RSTD
      RSTD_row_stride n_cols eps BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ((ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_liger_forward Y X W B Mean RSTD Y_row_stride
        X_row_stride Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (Y, yOffset s Y_row_stride i)))
      (expected := fun i =>
        layernormYSpec s X W B X_row_stride n_cols BLOCK_SIZE eps i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_liger_forward Y X W B Mean RSTD Y_row_stride
        X_row_stride Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE eps)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, s.pid * Mean_row_stride))
      (expected := fun _ =>
        WithBot.unbotD 0
          (layernormMeanCarrier s X X_row_stride n_cols BLOCK_SIZE))) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_liger_forward Y X W B Mean RSTD Y_row_stride
        X_row_stride Mean_row_stride RSTD_row_stride n_cols BLOCK_SIZE eps)
      (initialState := s)
      (write := fun _ : PUnit => some (RSTD, s.pid * RSTD_row_stride))
      (expected := fun _ =>
        WithBot.unbotD 0
          (layernormInvVarCarrier s X X_row_stride n_cols BLOCK_SIZE eps))))
```

**Assumptions / layout contracts:**
- `hYMean : Y ≠ Mean`
- `hYRSTD : Y ≠ RSTD`
- `hMeanY : Mean ≠ Y`
- `hMeanRSTD : Mean ≠ RSTD`
- `hRSTDY : RSTD ≠ Y`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s Y_row_stride i)`
- `fun i : Fin BLOCK_SIZE => i.val < n_cols`

**Closed-form spec defs (transitive):** `yOffset`, `layer_norm_liger_forward_surface`, `layer_norm_liger_forward`, `layernormYSpec`, `layernormMeanCarrier`, `layernormInvVarCarrier`, `xOffset`, `layernormInputTile`, `layernormVarCarrier`, `layernormCenteredTile`

<details><summary><code>yOffset</code></summary>

```lean
def yOffset (s : BlockState) (Y_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * Y_row_stride + i.val
```
</details>

<details><summary><code>layer_norm_liger_forward_surface</code></summary>

```
/-- Faithful transcription of `layer_norm_liger.py`'s
`_layer_norm_forward_kernel`.

The source accepts `W_row_stride` and `B_row_stride` but does not use them in
pointer arithmetic; they are preserved in the surface signature. -/
```
```lean
def layer_norm_liger_forward_surface
    (Y_ptr : RegionName) (Y_row_stride : Nat)
    (X_ptr : RegionName) (X_row_stride : Nat)
    (W_ptr : RegionName) (W_row_stride : Nat)
    (B_ptr : RegionName) (B_row_stride : Nat)
    (Mean_ptr : RegionName) (Mean_row_stride : Nat)
    (RSTD_ptr : RegionName) (RSTD_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  Y_ptr += row_idx * $(Y_row_stride)
  X_ptr += row_idx * $(X_row_stride)
  Mean_ptr += row_idx * $(Mean_row_stride)
  RSTD_ptr += row_idx * $(RSTD_row_stride)
  X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
  W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
  B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)
  mean = tl.sum(X_row, axis=0) / $(n_cols)
  var = tl.sum((X_row - mean) * (X_row - mean), axis=0) / $(n_cols)
  rstd = tl.rsqrt(var + $(eps))
  tl.store(Mean_ptr, mean)
  tl.store(RSTD_ptr, rstd)
  Y_row = (X_row - mean) * rstd * W_row + B_row
  tl.store(Y_ptr + col_offsets, Y_row, mask=mask)
}
```
</details>

<details><summary><code>layer_norm_liger_forward</code></summary>

```
/-- Proof-oriented forward slice of `layer_norm_liger.py`'s
`_layer_norm_forward_kernel`.

This covers the forward `Y` output for one row block, including masked loads,
mean/variance/rstd computation, `Mean`/`RSTD` side stores, affine transform, and
masked `Y` store. The source forward kernel's `W_row_stride` and `B_row_stride`
parameters are not used in pointer arithmetic, so this slice omits them. -/
```
```lean
def layer_norm_liger_forward
    (Y X W B Mean RSTD : RegionName)
    (Y_row_stride X_row_stride Mean_row_stride RSTD_row_stride n_cols
      BLOCK_SIZE : Nat)
    (eps : ℝ) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)

  Y += row_idx * $(Y_row_stride)
  X += row_idx * $(X_row_stride)
  Mean += row_idx * $(Mean_row_stride)
  RSTD += row_idx * $(RSTD_row_stride)

  X_row = tl.load(X + col_offsets, mask=mask, other=0)
  W_row = tl.load(W + col_offsets, mask=mask, other=0)
  B_row = tl.load(B + col_offsets, mask=mask, other=0)

  mean = tl.sum(X_row, axis=0) / $(n_cols)
  var = tl.sum((X_row - mean) * (X_row - mean), axis=0) / $(n_cols)
  rstd = tl.rsqrt(var + $(eps))
  tl.store(Mean, mean)
  tl.store(RSTD, rstd)
  Y_row = (X_row - mean) * rstd * W_row + B_row
  tl.store(Y + col_offsets, Y_row, mask=mask)
}
```
</details>

<details><summary><code>layernormYSpec</code></summary>

```lean
noncomputable def layernormYSpec
    (s : BlockState) (X W B : RegionName)
    (X_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (idx : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun affine bias => affine + bias)
      (Option.map₂ (fun scaled w => scaled * w)
        (Option.map₂ (fun centered inv => centered * inv)
          (Option.map₂ (fun x mean => x - mean)
            (some (s.readMem X (xOffset s X_row_stride idx)))
            (layernormMeanCarrier s X X_row_stride n_cols BLOCK_SIZE))
          (layernormInvVarCarrier s X X_row_stride n_cols BLOCK_SIZE eps))
        (some (s.readMem W idx.val)))
      (some (s.readMem B idx.val)))
```
</details>

<details><summary><code>layernormMeanCarrier</code></summary>

```lean
noncomputable def layernormMeanCarrier
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (n_cols : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (layernormInputTile s X X_row_stride n_cols BLOCK_SIZE)).data PUnit.unit)
```
</details>

<details><summary><code>layernormInvVarCarrier</code></summary>

```lean
noncomputable def layernormInvVarCarrier
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) : WithBot ℝ :=
  WithBot.realRsqrt
    (Option.map (fun a => a + eps)
      (layernormVarCarrier s X X_row_stride n_cols BLOCK_SIZE))
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (X_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * X_row_stride + i.val
```
</details>

<details><summary><code>layernormInputTile</code></summary>

```lean
noncomputable def layernormInputTile
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < n_cols then
        some (s.readMem X (xOffset s X_row_stride idx.1))
      else some (0 : ℝ) }
```
</details>

<details><summary><code>layernormVarCarrier</code></summary>

```lean
noncomputable def layernormVarCarrier
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (n_cols : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (layernormCenteredTile s X X_row_stride n_cols BLOCK_SIZE)
        (layernormCenteredTile s X X_row_stride n_cols BLOCK_SIZE))).data
        PUnit.unit)
```
</details>

<details><summary><code>layernormCenteredTile</code></summary>

```lean
noncomputable def layernormCenteredTile
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      Option.map₂ (fun x mean => x - mean)
        ((layernormInputTile s X X_row_stride n_cols BLOCK_SIZE).data idx)
        (layernormMeanCarrier s X X_row_stride n_cols BLOCK_SIZE) }
```
</details>

## Also present (pinned special-case summaries)
- `layer_norm_liger_forward_y_compute_correct`
- `layer_norm_liger_forward_mean_store_slice_compute_correct`
- `layer_norm_liger_forward_rstd_store_slice_compute_correct`
- `layer_norm_liger_forward_inv_var_compute_correct`
- `layer_norm_liger_forward_mean_compute_correct`
- `layer_norm_liger_forward_all_outputs_compute_correct`
