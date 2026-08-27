# Spec sheet — `bench/tritonbench_g/fast_layernorm/FastLayernorm.lean`

**Python source:** `bench/tritonbench_g/fast_layernorm/fast_layernorm.py`

## Public theorem: `layernorm_forward_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `layernorm_forward` implements the exact affine LayerNorm
triple over the active row prefix on its masked three-input / three-output IO
signature — for every disjoint flat placement of the six buffers, every
program id whose active lanes and scalar store cells are in bounds, and every
launch state whose active input lanes hold `xs` (the `X` row), `ws` (the
weights) and `bs` (the bias), the translated pointer kernel terminates, every
active `Y` lane `j` holds
`layernormYSpec n_cols BLOCK_SIZE eps xs ws bs j = ((x - mean)·inv_var)·w + b`,
the scalar cells `r[pid]` / `mu[pid]` hold
`invVarFullSpec n_cols BLOCK_SIZE eps xs = rsqrt(var + eps)` and
`meanFullSpec n_cols BLOCK_SIZE xs = sum(x)/n_cols`, and every other memory
cell is unchanged. `0 < BLOCK_SIZE` is required: the two scalar stores are
unconditional, and the interface carries their in-bounds/frame obligations on
the write-active lane `0`, which must exist. The output buffers must be
pairwise distinct (`Y ≠ r`, `Y ≠ mu`, `r ≠ mu`) so each readback sees through
the other stores. Proof: `Masked2DKernelIO₃ₓ₃.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification layernorm_forward_correctness
    (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (hB : 0 < BLOCK_SIZE)
    (hYr : Y ≠ r) (hYmu : Y ≠ mu) (hRmu : r ≠ mu) :
    layernormForwardIO Y X W bias r mu Y_row_stride X_row_stride n_cols eps
        BLOCK_SIZE ⊨
      fun _ _ xs ws bs =>
        (fun i => layernormYSpec n_cols BLOCK_SIZE eps xs ws bs i,
         fun _ => invVarFullSpec n_cols BLOCK_SIZE eps xs,
         fun _ => meanFullSpec n_cols BLOCK_SIZE xs)
```

**Assumptions / layout contracts:**
- `hB : 0 < BLOCK_SIZE`
- `hYr : Y ≠ r`
- `hYmu : Y ≠ mu`
- `hRmu : r ≠ mu`

**Closed-form spec defs (transitive):** `layernormForwardIO`, `layernormYSpec`, `invVarFullSpec`, `meanFullSpec`, `layernorm_forward`, `layernormMeanCarrier`, `layernormInvVarCarrier`, `layernormInputTile`, `layernormVarCarrier`, `layernormCenteredTile`

<details><summary><code>layernormForwardIO</code></summary>

```
/-- `layernorm_forward`'s masked three-input / three-output **IO signature** —
the whole kernel-specific audit surface of the `⊨` headline:

* `in1`/`in2`/`in3` — the input matrix `X`, the per-column weights `W`, the
  per-column bias `b`;
* `out1`/`out2`/`out3` — the output matrix `Y`, the per-row reciprocal-std
  vector `r`, the per-row mean vector `mu`;
* `B = BLOCK_SIZE` — the row window each program owns;
* `read1`/`write1` — **per-lane row windows**: program `pid` reads its `X` row
  at `pid * X_row_stride + j` and writes its `Y` row at
  `pid * Y_row_stride + j` (the host-side one-program-per-row launch
  convention);
* `read2`/`read3` — the weight and bias windows are **absolute and
  pid-independent**: every program reads `W[j]` / `b[j]`;
* `write2`/`write3` — the **scalar** store cells `r[pid]` / `mu[pid]`, the
  same for every lane;
* `mask` — the active lanes `j < n_cols`, the same for every program: the row
  prefix that actually exists in the matrix. All three loads and the `Y` store
  share it (`read2Mask`/`read3Mask`/`writeMask1` keep their defaults);
* `writeMask2`/`writeMask3` — lane `0` carries each scalar; the other lanes
  are write-inactive and carry no obligations on either side.

The grid is 1-D, so the second program-id axis is an unused parameter: windows
and masks are constant in `pid₁` (the headline's `∀ pid₁` quantification is
vacuous but honest). The windows and masks are declared, not parsed from the
kernel; the headline **proves** the kernel's actual addressing and masking
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the active lanes. -/
```
```lean
def layernormForwardIO (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) : Masked2DKernelIO₃ₓ₃ where
  kernel := layernorm_forward Y X W bias r mu Y_row_stride X_row_stride
    n_cols eps BLOCK_SIZE
  in1 := X
  in2 := W
  in3 := bias
  out1 := Y
  out2 := r
  out3 := mu
  B := BLOCK_SIZE
  read1 := fun pid _ j => pid * X_row_stride + j.val
  read2 := fun _ _ j => j.val
  read3 := fun _ _ j => j.val
  write1 := fun pid _ j => pid * Y_row_stride + j.val
  write2 := fun pid _ _ => pid
  write3 := fun pid _ _ => pid
  mask := fun _ _ j => j.val < n_cols
  writeMask2 := fun _ _ j => j.val = 0
  writeMask3 := fun _ _ j => j.val = 0
```
</details>

<details><summary><code>layernormYSpec</code></summary>

```
/-- Exact affine LayerNorm value computed by the kernel at lane `idx`, as a
pure function of the row `xs`, weights `ws` and bias `bs`:
`((x - mean) * inv_var) * w + b` with the row statistics threaded through
`layernormMeanCarrier` / `layernormInvVarCarrier`. -/
```
```lean
noncomputable def layernormYSpec
    (n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (xs ws bs : Fin BLOCK_SIZE → ℝ) (idx : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun affine bias => affine + bias)
      (Option.map₂ (fun scaled w => scaled * w)
        (Option.map₂ (fun centered inv => centered * inv)
          (Option.map₂ (fun x mean => x - mean)
            (some (xs idx))
            (layernormMeanCarrier n_cols BLOCK_SIZE xs))
          (layernormInvVarCarrier n_cols BLOCK_SIZE eps xs))
        (some (ws idx)))
      (some (bs idx)))
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
    (n_cols BLOCK_SIZE : Nat) (eps : ℝ) (xs : Fin BLOCK_SIZE → ℝ) : ℝ :=
  WithBot.unbotD 0
    (layernormInvVarCarrier n_cols BLOCK_SIZE eps xs)
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
    (n_cols BLOCK_SIZE : Nat) (xs : Fin BLOCK_SIZE → ℝ) : ℝ :=
  WithBot.unbotD 0
    (layernormMeanCarrier n_cols BLOCK_SIZE xs)
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

<details><summary><code>layernormMeanCarrier</code></summary>

```
/-- `mean_X = tl.sum(X_row) / n_cols` over the masked row: masked lanes enter
as `0`, neutral for the sum, so the padded-block sum equals the sum over the
active prefix. -/
```
```lean
noncomputable def layernormMeanCarrier
    (n_cols BLOCK_SIZE : Nat) (xs : Fin BLOCK_SIZE → ℝ) :
    WithBot ℝ :=
  Option.map (fun a => a / (n_cols : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (layernormInputTile n_cols BLOCK_SIZE xs)).data PUnit.unit)
```
</details>

<details><summary><code>layernormInvVarCarrier</code></summary>

```
/-- `inv_var = tl.math.rsqrt(row_var + eps)`. -/
```
```lean
noncomputable def layernormInvVarCarrier
    (n_cols BLOCK_SIZE : Nat) (eps : ℝ) (xs : Fin BLOCK_SIZE → ℝ) :
    WithBot ℝ :=
  WithBot.realRsqrt
    (Option.map (fun a => a + eps)
      (layernormVarCarrier n_cols BLOCK_SIZE xs))
```
</details>

<details><summary><code>layernormInputTile</code></summary>

```
/-- Masked input row tile: lane `j < n_cols` holds `xs j`, masked lanes are
`0`, matching `mask=…, other=0`. Pure in the row values `xs`. -/
```
```lean
noncomputable def layernormInputTile
    (n_cols BLOCK_SIZE : Nat) (xs : Fin BLOCK_SIZE → ℝ) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < n_cols then some (xs idx.1)
      else some (0 : ℝ) }
```
</details>

<details><summary><code>layernormVarCarrier</code></summary>

```
/-- `row_var = tl.sum(XX * XX) / n_cols`. -/
```
```lean
noncomputable def layernormVarCarrier
    (n_cols BLOCK_SIZE : Nat) (xs : Fin BLOCK_SIZE → ℝ) :
    WithBot ℝ :=
  Option.map (fun a => a / (n_cols : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (layernormCenteredTile n_cols BLOCK_SIZE xs)
        (layernormCenteredTile n_cols BLOCK_SIZE xs))).data PUnit.unit)
```
</details>

<details><summary><code>layernormCenteredTile</code></summary>

```
/-- `XX = X_row - mean_X`, lane-wise over the masked row tile. -/
```
```lean
noncomputable def layernormCenteredTile
    (n_cols BLOCK_SIZE : Nat) (xs : Fin BLOCK_SIZE → ℝ) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      Option.map₂ (fun x mean => x - mean)
        ((layernormInputTile n_cols BLOCK_SIZE xs).data idx)
        (layernormMeanCarrier n_cols BLOCK_SIZE xs) }
```
</details>

## Also present (pinned special-case summaries)
- `layernorm_backward_dx_compute_correct`
- `layernorm_forward_inv_var_store_slice_compute_correct`
- `layernorm_forward_mean_store_slice_compute_correct`
