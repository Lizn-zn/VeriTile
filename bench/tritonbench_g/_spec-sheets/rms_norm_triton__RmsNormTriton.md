# Spec sheet — `bench/tritonbench_g/rms_norm_triton/RmsNormTriton.lean`

**Python source:** `bench/tritonbench_g/rms_norm_triton/rms_norm_triton.py`

## Public theorem: `rms_norm_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `rms_norm_kernel` implements the exact RMS normalization
over the active row prefix on its masked strided IO signature — for every
disjoint flat placement of the three buffers, every program id whose active
lanes are in bounds, and every launch state whose active input lanes hold `xs`
(the `X` row) and `ws` (the weights), the translated pointer kernel
terminates, every active output lane `j` holds
`rmsNormSpec N BLOCK_SIZE eps xs ws j = (xs j · rrms) · ws j`, and every other
memory cell is unchanged. `0 < y_stride_c` is required: distinct lanes must
scatter to distinct output cells (a zero column stride would self-clobber the
row). No `0 < BLOCK_SIZE` side condition: the kernel's only reduction is a
`sum`, total on empty tiles. Proof: `Masked2DKernelIO₂.Implements.intro`
assembles the region-model masked triple with the flat-memory bridge side
conditions. -/
```
</details>

**Statement:**
```lean
specification rms_norm_kernel_correctness
    (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N BLOCK_SIZE : Nat)
    (eps : ℝ) (hyc : 0 < y_stride_c) :
    rmsNormIO Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps
        BLOCK_SIZE ⊨
      fun _ _ xs ws i => rmsNormSpec N BLOCK_SIZE eps xs ws i
```

**Assumptions / layout contracts:**
- `hyc : 0 < y_stride_c`

**Closed-form spec defs (transitive):** `rmsNormIO`, `rmsNormSpec`, `rms_norm_kernel`, `rmsRrmsCarrier`, `rmsSumCarrier`, `rmsInputTile`

<details><summary><code>rmsNormIO</code></summary>

```
/-- `rms_norm_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `in1`/`in2`/`out` — which buffer is which argument (the wiring): the input
  matrix `X`, the per-column weights `W`, the output matrix `Y`;
* `B = BLOCK_SIZE` — the row window each program owns;
* `read1`/`write` — **per-lane strided windows**: program `pid` reads its `X`
  row at `pid * x_stride_r + j * x_stride_c` and writes its `Y` row at
  `pid * y_stride_r + j * y_stride_c` (the host-side one-program-per-row
  launch convention with general row/column strides);
* `read2` — the weight window is **absolute and pid-independent**: every
  program reads `W[j]` (the host passes the same weight vector to all rows);
* `mask` — the active lanes `j < N`, **the same for every program**: the row
  prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_SIZE = next_power_of_2(N)`) carry no obligations on either side. The
  store mask equals the load mask, so `writeMask` keeps its default.

The grid is 1-D, so the second program-id axis is an unused parameter: windows
and mask are constant in `pid₁` (the headline's `∀ pid₁` quantification is
vacuous but honest). The windows and mask are declared, not parsed from the
kernel; the headline **proves** the kernel's actual addressing and masking
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the active lanes. -/
```
```lean
def rmsNormIO (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) : Masked2DKernelIO₂ where
  kernel := rms_norm_kernel Y X W y_stride_r y_stride_c x_stride_r x_stride_c
    N eps BLOCK_SIZE
  in1 := X
  in2 := W
  out := Y
  B := BLOCK_SIZE
  read1 := fun pid _ j => pid * x_stride_r + j.val * x_stride_c
  read2 := fun _ _ j => j.val
  write := fun pid _ j => pid * y_stride_r + j.val * y_stride_c
  mask := fun _ _ j => j.val < N
```
</details>

<details><summary><code>rmsNormSpec</code></summary>

```
/-- Exact RMS-normalization value computed by the kernel at lane `idx`, as a
pure function of the active row prefix `xs j` and weight prefix `ws j`,
`j < N`: `(x · rrms) · w` with `rrms` threaded through `rmsRrmsCarrier`. -/
```
```lean
noncomputable def rmsNormSpec (N BLOCK_SIZE : Nat) (eps : ℝ)
    (xs ws : Fin BLOCK_SIZE → ℝ) (idx : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun x w => x * w)
      (Option.map₂ (fun x rrms => x * rrms)
        (some (xs idx))
        (rmsRrmsCarrier N BLOCK_SIZE eps xs))
      (some (ws idx)))
```
</details>

<details><summary><code>rms_norm_kernel</code></summary>

```
/-- Faithful transcription of `rms_norm_triton.py`'s `rms_norm_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter. -/
```
```lean
def rms_norm_kernel
    (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  Y += pid * $(y_stride_r)
  X += pid * $(x_stride_r)
  mask = tl.arange(0, $(BLOCK_SIZE)) < $(N)
  cols = tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(X + cols * $(x_stride_c), mask, other=0.0).to(tl.float32)
  var = tl.sum(x * x, axis=0) / $(N)
  rrms = 1 / tl.sqrt(var + $(eps))
  w = tl.load(W + tl.arange(0, $(BLOCK_SIZE)), mask=mask, other=0.0)
  y = (x * rrms).to(Y.dtype.element_ty) * w
  tl.store(Y + cols * $(y_stride_c), y, mask=mask)
}
```
</details>

<details><summary><code>rmsRrmsCarrier</code></summary>

```
/-- `rrms = 1 / sqrt(sum(x*x)/N + eps)`. -/
```
```lean
noncomputable def rmsRrmsCarrier (N BLOCK_SIZE : Nat) (eps : ℝ)
    (xs : Fin BLOCK_SIZE → ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt (Option.map ((fun a => a + eps) ∘ fun a => a / (N : ℝ))
      (rmsSumCarrier N BLOCK_SIZE xs)))
```
</details>

<details><summary><code>rmsSumCarrier</code></summary>

```
/-- `sum(x*x)` over the masked row: masked lanes enter as `0`, neutral for the
sum, so the reduction-over-padded-block equals the sum over the active
prefix. -/
```
```lean
noncomputable def rmsSumCarrier (N BLOCK_SIZE : Nat) (xs : Fin BLOCK_SIZE → ℝ) :
    WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (rmsInputTile N BLOCK_SIZE xs)
      (rmsInputTile N BLOCK_SIZE xs))).data PUnit.unit
```
</details>

<details><summary><code>rmsInputTile</code></summary>

```
/-- Masked input row tile: lane `j < N` holds `xs j`, masked lanes are `0`,
matching `mask=…, other=0.0`. -/
```
```lean
noncomputable def rmsInputTile (N BLOCK_SIZE : Nat) (xs : Fin BLOCK_SIZE → ℝ) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < N then some (xs idx.1)
      else some (0.0 : ℝ) }
```
</details>
