# Spec sheet — `bench/tritonbench_g/fast_rms_layernorm/FastRmsLayernorm.lean`

**Python source:** `bench/tritonbench_g/fast_rms_layernorm/fast_rms_layernorm.py`

## Public theorem: `rms_layernorm_forward_correctness`

<details><summary>docstring</summary>

```
/-- **The headline (plain forward)**: `_rms_layernorm_forward` implements the
exact RMS normalization pair on its masked two-output IO signature — for every
disjoint flat placement of the four buffers, every program id whose active
lanes and scalar rstd cell are in bounds, and every launch state whose active
input lanes hold `xs` (the `X` row) and `ws` (the weights), the translated
pointer kernel terminates, every active `Y` lane `j` holds
`rmsFwdYSpec … = (xs j · inv_var) · ws j`, the rstd cell holds
`rmsFwdInvVarSpec … = rsqrt(sum(x²)/n_cols + eps)`, and every other memory cell
is unchanged. Side conditions, both genuinely forced: `Y ≠ r` (the
unconditional scalar rstd store must not alias the masked row store) and
`0 < BLOCK_SIZE` (the `r` store is unmasked in the kernel, so its safety bound
and frame exclusion are carried by the lane-0 gate `writeMask2`, which needs a
lane). Proof: `Masked2DKernelIO₂ₓ₂.Implements.intro` assembles the region-model
masked triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification rms_layernorm_forward_correctness
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (hYr : Y ≠ r) (hB : 0 < BLOCK_SIZE) :
    rmsLayernormFwdIO Y X W r Y_row_stride X_row_stride W_row_stride
        r_row_stride n_cols eps BLOCK_SIZE ⊨
      fun _ _ xs ws =>
        (fun i => rmsFwdYSpec n_cols BLOCK_SIZE eps xs ws i,
         fun _ => rmsFwdInvVarSpec n_cols BLOCK_SIZE eps xs)
```

**Assumptions / layout contracts:**
- `hYr : Y ≠ r`
- `hB : 0 < BLOCK_SIZE`

**Closed-form spec defs (transitive):** `rmsLayernormFwdIO`, `rmsFwdYSpec`, `rmsFwdInvVarSpec`, `rms_layernorm_forward`, `rmsFwdInvVarCarrier`, `rmsFwdSumCarrier`, `rmsFwdInputTile`

<details><summary><code>rmsLayernormFwdIO</code></summary>

```
/-- `_rms_layernorm_forward`'s masked two-output **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `in1`/`in2`/`out1`/`out2` — which buffer is which argument (the wiring): the
  input matrix `X`, the per-column weights `W`, the output matrix `Y`, the
  per-row rstd vector `r`;
* `B = BLOCK_SIZE` — the row window each program owns;
* `read1`/`write1` — **strided row windows**: program `row_idx` reads its `X`
  row at `row_idx · X_row_stride + j` and writes its `Y` row at
  `row_idx · Y_row_stride + j` (the host-side one-program-per-row launch);
* `read2` — the weight window is **pid-independent and column-strided**: every
  program reads `W[j · W_row_stride]` (the Python kernel scales the offsets by
  `W_row_stride`);
* `write2` — the **scalar** rstd cell `r[row_idx · r_row_stride]`, the same
  for every lane;
* `mask` — the active lanes `j < n_cols`, the same for every program; the
  load masks and the `Y` store mask coincide, so `read2Mask`/`writeMask1` keep
  their `mask` default;
* `writeMask2` — lane `0` carries the scalar rstd; the other lanes are
  write-inactive and carry no obligations on either side.

The grid is 1-D, so the second program-id axis is an unused parameter: windows
and masks are constant in `pid₁` (the headline's `∀ pid₁` quantification is
vacuous but honest). The windows and masks are declared, not parsed from the
kernel; the headline **proves** the kernel's actual addressing and masking
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the active lanes. -/
```
```lean
def rmsLayernormFwdIO (Y X W r : RegionName)
    (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) : Masked2DKernelIO₂ₓ₂ where
  kernel := rms_layernorm_forward Y X W r Y_row_stride X_row_stride
    W_row_stride r_row_stride n_cols eps BLOCK_SIZE
  in1 := X
  in2 := W
  out1 := Y
  out2 := r
  B := BLOCK_SIZE
  read1 := fun row_idx _ j => row_idx * X_row_stride + j.val
  read2 := fun _ _ j => j.val * W_row_stride
  write1 := fun row_idx _ j => row_idx * Y_row_stride + j.val
  write2 := fun row_idx _ _ => row_idx * r_row_stride
  mask := fun _ _ j => j.val < n_cols
  writeMask2 := fun _ _ j => j.val = 0
```
</details>

<details><summary><code>rmsFwdYSpec</code></summary>

```
/-- Pure per-lane `Y` value of the plain forward: `(x · inv_var) · w`. -/
```
```lean
noncomputable def rmsFwdYSpec (n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (xs ws : Fin BLOCK_SIZE → ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun x w => x * w)
      (Option.map₂ (fun x inv => x * inv)
        (some (xs i))
        (rmsFwdInvVarCarrier n_cols BLOCK_SIZE eps xs))
      (some (ws i)))
```
</details>

<details><summary><code>rmsFwdInvVarSpec</code></summary>

```
/-- Pure per-row rstd value stored to `r`: `WithBot.unbotD 0` of the
`rsqrt` carrier. -/
```
```lean
noncomputable def rmsFwdInvVarSpec (n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (xs : Fin BLOCK_SIZE → ℝ) : ℝ :=
  WithBot.unbotD 0 (rmsFwdInvVarCarrier n_cols BLOCK_SIZE eps xs)
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

<details><summary><code>rmsFwdInvVarCarrier</code></summary>

```
/-- Pure `inv_var = rsqrt(sum(x*x)/n_cols + eps)` carrier. -/
```
```lean
noncomputable def rmsFwdInvVarCarrier (n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (xs : Fin BLOCK_SIZE → ℝ) : WithBot ℝ :=
  WithBot.realRsqrt
    (Option.map ((fun a => a + eps) ∘ fun a => a / (n_cols : ℝ))
      (rmsFwdSumCarrier n_cols BLOCK_SIZE xs))
```
</details>

<details><summary><code>rmsFwdSumCarrier</code></summary>

```
/-- Pure `sum(X_row * X_row)` over the masked row (masked lanes enter as `0`,
neutral for the sum). -/
```
```lean
noncomputable def rmsFwdSumCarrier (n_cols BLOCK_SIZE : Nat)
    (xs : Fin BLOCK_SIZE → ℝ) : WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (rmsFwdInputTile n_cols BLOCK_SIZE xs)
      (rmsFwdInputTile n_cols BLOCK_SIZE xs))).data PUnit.unit
```
</details>

<details><summary><code>rmsFwdInputTile</code></summary>

```
/-- Pure masked input row tile: lane `j < n_cols` holds `xs j`, masked lanes
are `0` (matching `mask=…, other=0`). The `xs`-reparametrized form of
`rmsInputTile`. -/
```
```lean
noncomputable def rmsFwdInputTile (n_cols BLOCK_SIZE : Nat)
    (xs : Fin BLOCK_SIZE → ℝ) : Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < n_cols then some (xs idx.1) else some (0 : ℝ) }
```
</details>

## Public theorem: `gemma_rms_layernorm_forward_correctness`

<details><summary>docstring</summary>

```
/-- **The headline (Gemma forward)**: `_gemma_rms_layernorm_forward` implements
the Gemma-scaled RMS normalization pair on its masked two-output IO signature —
as the plain headline, with every active `Y` lane `j` holding
`gemmaRmsFwdYSpec … = (xs j · inv_var) · (ws j + 1)` over the contiguous weight
window. Same genuinely-forced side conditions (`Y ≠ r`, `0 < BLOCK_SIZE`). -/
```
</details>

**Statement:**
```lean
specification gemma_rms_layernorm_forward_correctness
    (Y X W r : RegionName)
    (Y_row_stride X_row_stride r_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (hYr : Y ≠ r) (hB : 0 < BLOCK_SIZE) :
    gemmaRmsLayernormFwdIO Y X W r Y_row_stride X_row_stride r_row_stride
        n_cols eps BLOCK_SIZE ⊨
      fun _ _ xs ws =>
        (fun i => gemmaRmsFwdYSpec n_cols BLOCK_SIZE eps xs ws i,
         fun _ => rmsFwdInvVarSpec n_cols BLOCK_SIZE eps xs)
```

**Assumptions / layout contracts:**
- `hYr : Y ≠ r`
- `hB : 0 < BLOCK_SIZE`

**Closed-form spec defs (transitive):** `gemmaRmsLayernormFwdIO`, `gemmaRmsFwdYSpec`, `rmsFwdInvVarSpec`, `gemma_rms_layernorm_forward`, `rmsFwdInvVarCarrier`, `rmsFwdSumCarrier`, `rmsFwdInputTile`

<details><summary><code>gemmaRmsLayernormFwdIO</code></summary>

```
/-- `_gemma_rms_layernorm_forward`'s masked two-output **IO signature** — as
`rmsLayernormFwdIO`, except the weight window is **contiguous**: the Python
kernel accepts `W_row_stride` but loads `W + col_offsets`, and this signature
preserves that stride-free weight access (`read2 = j`). -/
```
```lean
def gemmaRmsLayernormFwdIO (Y X W r : RegionName)
    (Y_row_stride X_row_stride r_row_stride n_cols : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) : Masked2DKernelIO₂ₓ₂ where
  kernel := gemma_rms_layernorm_forward Y X W r Y_row_stride X_row_stride
    r_row_stride n_cols eps BLOCK_SIZE
  in1 := X
  in2 := W
  out1 := Y
  out2 := r
  B := BLOCK_SIZE
  read1 := fun row_idx _ j => row_idx * X_row_stride + j.val
  read2 := fun _ _ j => j.val
  write1 := fun row_idx _ j => row_idx * Y_row_stride + j.val
  write2 := fun row_idx _ _ => row_idx * r_row_stride
  mask := fun _ _ j => j.val < n_cols
  writeMask2 := fun _ _ j => j.val = 0
```
</details>

<details><summary><code>gemmaRmsFwdYSpec</code></summary>

```
/-- Pure per-lane `Y` value of the Gemma forward: `(x · inv_var) · (w + 1)`. -/
```
```lean
noncomputable def gemmaRmsFwdYSpec (n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (xs ws : Fin BLOCK_SIZE → ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * (w + 1.0))
      (Option.map₂ (fun x inv => x * inv)
        (some (xs i))
        (rmsFwdInvVarCarrier n_cols BLOCK_SIZE eps xs))
      (some (ws i)))
```
</details>

<details><summary><code>rmsFwdInvVarSpec</code></summary>

```
/-- Pure per-row rstd value stored to `r`: `WithBot.unbotD 0` of the
`rsqrt` carrier. -/
```
```lean
noncomputable def rmsFwdInvVarSpec (n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (xs : Fin BLOCK_SIZE → ℝ) : ℝ :=
  WithBot.unbotD 0 (rmsFwdInvVarCarrier n_cols BLOCK_SIZE eps xs)
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

<details><summary><code>rmsFwdInvVarCarrier</code></summary>

```
/-- Pure `inv_var = rsqrt(sum(x*x)/n_cols + eps)` carrier. -/
```
```lean
noncomputable def rmsFwdInvVarCarrier (n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (xs : Fin BLOCK_SIZE → ℝ) : WithBot ℝ :=
  WithBot.realRsqrt
    (Option.map ((fun a => a + eps) ∘ fun a => a / (n_cols : ℝ))
      (rmsFwdSumCarrier n_cols BLOCK_SIZE xs))
```
</details>

<details><summary><code>rmsFwdSumCarrier</code></summary>

```
/-- Pure `sum(X_row * X_row)` over the masked row (masked lanes enter as `0`,
neutral for the sum). -/
```
```lean
noncomputable def rmsFwdSumCarrier (n_cols BLOCK_SIZE : Nat)
    (xs : Fin BLOCK_SIZE → ℝ) : WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (rmsFwdInputTile n_cols BLOCK_SIZE xs)
      (rmsFwdInputTile n_cols BLOCK_SIZE xs))).data PUnit.unit
```
</details>

<details><summary><code>rmsFwdInputTile</code></summary>

```
/-- Pure masked input row tile: lane `j < n_cols` holds `xs j`, masked lanes
are `0` (matching `mask=…, other=0`). The `xs`-reparametrized form of
`rmsInputTile`. -/
```
```lean
noncomputable def rmsFwdInputTile (n_cols BLOCK_SIZE : Nat)
    (xs : Fin BLOCK_SIZE → ℝ) : Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < n_cols then some (xs idx.1) else some (0 : ℝ) }
```
</details>

## Also present (pinned special-case summaries)
- `rms_layernorm_backward_dy_compute_correct`
- `gemma_rms_layernorm_backward_dy_compute_correct`
- `rms_layernorm_forward_inv_var_store_slice_compute_correct`
