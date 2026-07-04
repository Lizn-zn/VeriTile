# Spec sheet — `bench/tritonbench_g/fast_layernorm/FastLayernorm.lean`

**Python source:** `bench/tritonbench_g/fast_layernorm/fast_layernorm.py`

## Public theorem: `layernorm_forward_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `layernorm_forward`: the DSL surface lowers to
the algorithm layer, and the kernel is compute-correct on all three
Python-observable outputs — the masked `Y` store (every active lane holds the
affine LayerNorm spec `layernormYSpec`), the scalar reciprocal-std store into `r`
(`invVarFullSpec`), and the scalar mean store into `mu` (`meanFullSpec`). -/
```
</details>

**Statement:**
```lean
theorem layernorm_forward_output_summary
    (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hYr : Y ≠ r) (hYmu : Y ≠ mu)
    (hRY : r ≠ Y) (hRmu : r ≠ mu)
    (hMuY : mu ≠ Y)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOutOffset s Y_row_stride i)) :
    (∃ alg, (layernorm_forward Y X W bias r mu Y_row_stride X_row_stride
        n_cols eps BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := layernorm_forward Y X W bias r mu Y_row_stride X_row_stride
        n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (Y, yOutOffset s Y_row_stride i)))
      (expected := fun i =>
        layernormYSpec s X W bias X_row_stride n_cols BLOCK_SIZE eps i)) ∧
    (ComputeCorrect.Realizes
      (kernel := layernorm_forward Y X W bias r mu Y_row_stride X_row_stride
        n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := fun _ : PUnit => some (r, s.pid))
      (expected := fun _ =>
        invVarFullSpec s X X_row_stride n_cols BLOCK_SIZE eps)) ∧
    (ComputeCorrect.Realizes
      (kernel := layernorm_forward Y X W bias r mu Y_row_stride X_row_stride
        n_cols eps BLOCK_SIZE)
      (initialState := s)
      (write := fun _ : PUnit => some (mu, s.pid))
      (expected := fun _ =>
        meanFullSpec s X X_row_stride n_cols BLOCK_SIZE)))
```

**Assumptions / layout contracts:**
- `hYr : Y ≠ r`
- `hYmu : Y ≠ mu`
- `hRY : r ≠ Y`
- `hRmu : r ≠ mu`
- `hMuY : mu ≠ Y`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOutOffset s Y_row_stride i)`
- `fun i : Fin BLOCK_SIZE => i.val < n_cols`

**Closed-form spec defs (transitive):** `yOutOffset`, `layernorm_forward`, `layernormYSpec`, `invVarFullSpec`, `meanFullSpec`, `rowElem`, `layernormMeanCarrier`, `layernormInvVarCarrier`, `layernormInputTile`, `layernormVarCarrier`, `layernormCenteredTile`

<details><summary><code>yOutOffset</code></summary>

```lean
def yOutOffset (s : BlockState) (Y_row_stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * Y_row_stride + i.val
```
</details>

<details><summary><code>layernorm_forward</code></summary>

```
/-- Faithful transcription of `fast_layernorm.py`'s `layernorm_forward`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter. -/
```
```lean
def layernorm_forward
    (Y X W b r mu : RegionName)
    (Y_row_stride X_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)

  Y += row_idx * $(Y_row_stride)
  X += row_idx * $(X_row_stride)
  r += row_idx
  mu += row_idx

  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0).to(tl.float32)

  mean_X = tl.sum(X_row, axis=0) / $(n_cols)
  XX = X_row - mean_X
  row_var = tl.sum(XX * XX, axis=0) / $(n_cols)
  inv_var = tl.math.rsqrt(row_var + $(eps))
  tl.store(r, inv_var)
  tl.store(mu, mean_X)
  output = (XX * inv_var) * W_row + b_row
  tl.store(Y + col_offsets, output, mask=mask)
}
```
</details>

<details><summary><code>layernormYSpec</code></summary>

```lean
noncomputable def layernormYSpec
    (s : BlockState) (X W bias : RegionName)
    (X_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (idx : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun affine bias => affine + bias)
      (Option.map₂ (fun scaled w => scaled * w)
        (Option.map₂ (fun centered inv => centered * inv)
          (Option.map₂ (fun x mean => x - mean)
            (some (rowElem s X X_row_stride idx.val))
            (layernormMeanCarrier s X X_row_stride n_cols BLOCK_SIZE))
          (layernormInvVarCarrier s X X_row_stride n_cols BLOCK_SIZE eps))
        (some (s.readMem W idx.val)))
      (some (s.readMem bias idx.val)))
```
</details>

<details><summary><code>invVarFullSpec</code></summary>

```
/-- Full-kernel spec for the `inv_var` (rstd) store of `layernorm_forward`.

Wraps `layernormInvVarCarrier` with `WithBot.unbotD 0` so that the readback
of the kernel's scalar write into `r` matches the carrier's value as a
plain `ℝ`. -/
```
```lean
noncomputable def invVarFullSpec
    (s : BlockState) (X : RegionName)
    (X_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) : ℝ :=
  WithBot.unbotD 0
    (layernormInvVarCarrier s X X_row_stride n_cols BLOCK_SIZE eps)
```
</details>

<details><summary><code>meanFullSpec</code></summary>

```
/-- Full-kernel spec for the `mean` (mu) store of `layernorm_forward`.

Wraps `layernormMeanCarrier` with `WithBot.unbotD 0` so that the readback
of the kernel's scalar write into `mu` matches the carrier's value as a
plain `ℝ`. -/
```
```lean
noncomputable def meanFullSpec
    (s : BlockState) (X : RegionName)
    (X_row_stride n_cols BLOCK_SIZE : Nat) : ℝ :=
  WithBot.unbotD 0
    (layernormMeanCarrier s X X_row_stride n_cols BLOCK_SIZE)
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

<details><summary><code>layernormInputTile</code></summary>

```lean
noncomputable def layernormInputTile
    (s : BlockState) (X : RegionName) (X_row_stride n_cols BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < n_cols then some (rowElem s X X_row_stride idx.1.val)
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
        (layernormCenteredTile s X X_row_stride n_cols BLOCK_SIZE))).data PUnit.unit)
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
- `layernorm_backward_dx_compute_correct`
- `layernorm_forward_y_compute_correct`
- `layernorm_forward_inv_var_compute_correct`
- `layernorm_forward_mean_compute_correct`
- `layernorm_forward_inv_var_store_slice_compute_correct`
- `layernorm_forward_mean_store_slice_compute_correct`
