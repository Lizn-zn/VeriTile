# Spec sheet — `bench/tritonbench_g/fast_rms_layernorm/FastRmsLayernorm.lean`

**Python source:** `bench/tritonbench_g/fast_rms_layernorm/fast_rms_layernorm.py`

## Public theorem: `rms_layernorm_forward_output_summary`

<details><summary>docstring</summary>

```
/-- Public forward summary for regular RMS layernorm: the full Python forward
surface lowers, and the checked forward kernel characterizes both
Python-observable outputs `Y` and `r`. -/
```
</details>

**Statement:**
```lean
specification rms_layernorm_forward_output_summary
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hYr : Y ≠ r) (hrY : r ≠ Y)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOutOffset s Y_row_stride i)) :
    (∃ alg, (rms_layernorm_forward Y X W r Y_row_stride X_row_stride
      W_row_stride r_row_stride n_cols eps BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
    ((ComputeCorrect.Realizes_without_Rounding
      (kernel := rms_layernorm_forward Y X W r Y_row_stride X_row_stride
        W_row_stride r_row_stride n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (Y, yOutOffset s Y_row_stride i)))
      (expected := fun i =>
        rmsLayernormYSpec s X W X_row_stride W_row_stride n_cols
          BLOCK_SIZE eps i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := rms_layernorm_forward Y X W r Y_row_stride X_row_stride
        W_row_stride r_row_stride n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := fun _ : PUnit => some (r, rOutOffset s r_row_stride))
      (expected := fun _ =>
        rmsInvVarSpec s X X_row_stride n_cols BLOCK_SIZE eps)))
```

**Assumptions / layout contracts:**
- `hYr : Y ≠ r`
- `hrY : r ≠ Y`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOutOffset s Y_row_stride i)`
- `fun i : Fin BLOCK_SIZE => i.val < n_cols`

**Closed-form spec defs (transitive):** `yOutOffset`, `rms_layernorm_forward`, `rmsLayernormYSpec`, `rOutOffset`, `rmsInvVarSpec`, `rowElem`, `rmsInvVarCarrier`, `rmsSumCarrier`, `rmsInputTile`

<details><summary><code>yOutOffset</code></summary>

```lean
def yOutOffset (s : BlockState) (Y_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * Y_row_stride + i.val
```
</details>

<details><summary><code>rms_layernorm_forward</code></summary>

```
/-- Faithful transcription of `fast_rms_layernorm.py`'s
`_rms_layernorm_forward`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter. -/
```
```lean
def rms_layernorm_forward
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)

  Y += row_idx * $(Y_row_stride)
  X += row_idx * $(X_row_stride)
  r += row_idx * $(r_row_stride)

  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets * $(W_row_stride), mask=mask, other=0)

  row_var = tl.sum(X_row * X_row, axis=0) / $(n_cols)
  inv_var = tl.math.rsqrt(row_var + $(eps))
  tl.store(r, inv_var)
  normed = X_row * inv_var
  normed = (normed).to(W_row.dtype)
  output = normed * W_row
  tl.store(Y + col_offsets, output, mask=mask)
}
```
</details>

<details><summary><code>rmsLayernormYSpec</code></summary>

```lean
noncomputable def rmsLayernormYSpec
    (s : BlockState) (X W : RegionName)
    (X_row_stride W_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (idx : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun x w => x * w)
      (Option.map₂ (fun x inv => x * inv)
        (some (rowElem s X X_row_stride idx.val))
        (rmsInvVarCarrier s X X_row_stride n_cols BLOCK_SIZE eps))
      (some (s.readMem W (idx.val * W_row_stride))))
```
</details>

<details><summary><code>rOutOffset</code></summary>

```lean
def rOutOffset (s : BlockState) (r_row_stride : Nat) : Nat :=
  s.pid * r_row_stride
```
</details>

<details><summary><code>rmsInvVarSpec</code></summary>

```lean
noncomputable def rmsInvVarSpec
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) : ℝ :=
  WithBot.unbotD 0 (rmsInvVarCarrier s X X_row_stride n_cols BLOCK_SIZE eps)
```
</details>

<details><summary><code>rowElem</code></summary>

```
/-- Element `j` of **this program's row** of a row-major matrix region `R`
(row = `pid`, row stride `row_stride`): `R[pid·row_stride + j]`. The `X` and
`dY` row loads all use this layout. -/
```
```lean
noncomputable def rowElem (s : BlockState) (R : RegionName)
    (row_stride j : Nat) : ℝ :=
  s.readMem R (s.pid * row_stride + j)
```
</details>

<details><summary><code>rmsInvVarCarrier</code></summary>

```lean
noncomputable def rmsInvVarCarrier
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) : WithBot ℝ :=
  WithBot.realRsqrt
    (Option.map ((fun a => a + eps) ∘ fun a => a / (n_cols : ℝ))
      (rmsSumCarrier s X X_row_stride n_cols BLOCK_SIZE))
```
</details>

<details><summary><code>rmsSumCarrier</code></summary>

```lean
noncomputable def rmsSumCarrier
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (rmsInputTile s X X_row_stride n_cols BLOCK_SIZE)
      (rmsInputTile s X X_row_stride n_cols BLOCK_SIZE))).data PUnit.unit
```
</details>

<details><summary><code>rmsInputTile</code></summary>

```lean
noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < n_cols then some (rowElem s X X_row_stride idx.1.val)
      else some (0 : ℝ) }
```
</details>

## Public theorem: `gemma_rms_layernorm_forward_output_summary`

<details><summary>docstring</summary>

```
/-- Public forward summary for Gemma RMS layernorm: the full Python forward
surface lowers, and the checked forward kernel characterizes both
Python-observable outputs `Y` and `r`. -/
```
</details>

**Statement:**
```lean
specification gemma_rms_layernorm_forward_output_summary
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hYr : Y ≠ r) (hrY : r ≠ Y)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOutOffset s Y_row_stride i)) :
    (∃ alg, (gemma_rms_layernorm_forward Y X W r Y_row_stride
      X_row_stride r_row_stride n_cols eps BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
    ((ComputeCorrect.Realizes_without_Rounding
      (kernel := gemma_rms_layernorm_forward Y X W r Y_row_stride
        X_row_stride r_row_stride n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (Y, yOutOffset s Y_row_stride i)))
      (expected := fun i =>
        gemmaRmsLayernormYSpec s X W X_row_stride n_cols BLOCK_SIZE eps i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := gemma_rms_layernorm_forward Y X W r Y_row_stride
        X_row_stride r_row_stride n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := fun _ : PUnit => some (r, rOutOffset s r_row_stride))
      (expected := fun _ =>
        rmsInvVarSpec s X X_row_stride n_cols BLOCK_SIZE eps)))
```

**Assumptions / layout contracts:**
- `hYr : Y ≠ r`
- `hrY : r ≠ Y`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOutOffset s Y_row_stride i)`
- `fun i : Fin BLOCK_SIZE => i.val < n_cols`

**Closed-form spec defs (transitive):** `yOutOffset`, `gemma_rms_layernorm_forward`, `gemmaRmsLayernormYSpec`, `rOutOffset`, `rmsInvVarSpec`, `rowElem`, `rmsInvVarCarrier`, `rmsSumCarrier`, `rmsInputTile`

<details><summary><code>yOutOffset</code></summary>

```lean
def yOutOffset (s : BlockState) (Y_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * Y_row_stride + i.val
```
</details>

<details><summary><code>gemma_rms_layernorm_forward</code></summary>

```
/-- Faithful transcription of `fast_rms_layernorm.py`'s
`_gemma_rms_layernorm_forward`.

The Python kernel accepts `W_row_stride` but loads `W + col_offsets`, so this
surface preserves that stride-free weight access. -/
```
```lean
def gemma_rms_layernorm_forward
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride r_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)

  Y += row_idx * $(Y_row_stride)
  X += row_idx * $(X_row_stride)
  r += row_idx * $(r_row_stride)

  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)

  row_var = tl.sum(X_row * X_row, axis=0) / $(n_cols)
  inv_var = tl.math.rsqrt(row_var + $(eps))
  tl.store(r, inv_var)
  normed = X_row * inv_var
  output = normed * (W_row + 1.0)

  tl.store(Y + col_offsets, output, mask=mask)
}
```
</details>

<details><summary><code>gemmaRmsLayernormYSpec</code></summary>

```lean
noncomputable def gemmaRmsLayernormYSpec
    (s : BlockState) (X W : RegionName)
    (X_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (idx : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * (w + 1.0))
      (Option.map₂ (fun x inv => x * inv)
        (some (rowElem s X X_row_stride idx.val))
        (rmsInvVarCarrier s X X_row_stride n_cols BLOCK_SIZE eps))
      (some (s.readMem W idx.val)))
```
</details>

<details><summary><code>rOutOffset</code></summary>

```lean
def rOutOffset (s : BlockState) (r_row_stride : Nat) : Nat :=
  s.pid * r_row_stride
```
</details>

<details><summary><code>rmsInvVarSpec</code></summary>

```lean
noncomputable def rmsInvVarSpec
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) : ℝ :=
  WithBot.unbotD 0 (rmsInvVarCarrier s X X_row_stride n_cols BLOCK_SIZE eps)
```
</details>

<details><summary><code>rowElem</code></summary>

```
/-- Element `j` of **this program's row** of a row-major matrix region `R`
(row = `pid`, row stride `row_stride`): `R[pid·row_stride + j]`. The `X` and
`dY` row loads all use this layout. -/
```
```lean
noncomputable def rowElem (s : BlockState) (R : RegionName)
    (row_stride j : Nat) : ℝ :=
  s.readMem R (s.pid * row_stride + j)
```
</details>

<details><summary><code>rmsInvVarCarrier</code></summary>

```lean
noncomputable def rmsInvVarCarrier
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) : WithBot ℝ :=
  WithBot.realRsqrt
    (Option.map ((fun a => a + eps) ∘ fun a => a / (n_cols : ℝ))
      (rmsSumCarrier s X X_row_stride n_cols BLOCK_SIZE))
```
</details>

<details><summary><code>rmsSumCarrier</code></summary>

```lean
noncomputable def rmsSumCarrier
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (rmsInputTile s X X_row_stride n_cols BLOCK_SIZE)
      (rmsInputTile s X X_row_stride n_cols BLOCK_SIZE))).data PUnit.unit
```
</details>

<details><summary><code>rmsInputTile</code></summary>

```lean
noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < n_cols then some (rowElem s X X_row_stride idx.1.val)
      else some (0 : ℝ) }
```
</details>

## Also present (pinned special-case summaries)
- `rms_layernorm_forward_y_compute_correct`
- `gemma_rms_layernorm_forward_y_compute_correct`
- `rms_layernorm_backward_dy_compute_correct`
- `gemma_rms_layernorm_backward_dy_compute_correct`
- `rms_layernorm_forward_inv_var_store_slice_compute_correct`
- `rms_layernorm_forward_inv_var_compute_correct`
- `gemma_rms_layernorm_forward_inv_var_compute_correct`
- `rms_layernorm_forward_all_outputs_compute_correct`
- `gemma_rms_layernorm_forward_all_outputs_compute_correct`
