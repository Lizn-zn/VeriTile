# Spec sheet — `bench/tritonbench_g/swiglu_backward/SwigluBackward.lean`

**Python source:** `bench/tritonbench_g/swiglu_backward/swiglu_backward.py`

## Public theorem: `swiglu_bwd_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `_swiglu_bwd_kernel` implements the SwiGLU backward
oracles on its masked three-input / three-output IO signature — for every
disjoint flat placement of the six buffers, every program id whose active
lanes are in bounds, and every launch state whose active input lanes hold
`xs` (the gate), `ys` (the value) and `douts` (the upstream gradient), the
translated pointer kernel terminates, every active `DX` lane `j` holds
`TiledActivation.swigluBwdA (douts j) (xs j) (ys j)`, every active `DY` lane
holds `TiledActivation.swigluBwdB (douts j) (xs j)`, every `OUT` lane that is
active **and** `RECOMPUTE_OUTPUT`-gated holds the recomputed forward
`TiledActivation.swiglu (xs j) (ys j)`, and every other memory cell is
unchanged — one statement covering both heuristic outcomes of the constexpr
flag. The output buffers must be pairwise distinct (`DX ≠ DY`, `OUT ≠ DX`,
`OUT ≠ DY`) so each readback sees through the later stores. Proof:
`Masked2DKernelIO₃ₓ₃.Implements.intro` assembles the region-model masked
triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification swiglu_bwd_kernel_correctness
    (X Y DOUT OUT DX DY : RegionName)
    (stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N : Nat)
    (RECOMPUTE_OUTPUT : Bool)
    (hDXDY : DX ≠ DY) (hOUTDX : OUT ≠ DX) (hOUTDY : OUT ≠ DY) :
    swigluBackwardIO X Y DOUT OUT DX DY stride_x_row stride_y_row
        stride_dout_row stride_out_row stride_dx_row stride_dy_row ncols
        BLOCK_N RECOMPUTE_OUTPUT ⊨
      fun _ _ xs ys douts =>
        (fun j => TiledActivation.swigluBwdA (douts j) (xs j) (ys j),
         fun j => TiledActivation.swigluBwdB (douts j) (xs j),
         fun j => TiledActivation.swiglu (xs j) (ys j))
```

**Assumptions / layout contracts:**
- `hDXDY : DX ≠ DY`
- `hOUTDX : OUT ≠ DX`
- `hOUTDY : OUT ≠ DY`

**Closed-form spec defs (transitive):** `swigluBackwardIO`, `swiglu_bwd_kernel`

<details><summary><code>swigluBackwardIO</code></summary>

```
/-- `_swiglu_bwd_kernel`'s masked three-input / three-output **IO signature** —
the whole kernel-specific audit surface of the `⊨` headline:

* `in1`/`in2`/`in3` — the gate `X`, value `Y`, upstream gradient `DOUT`;
* `out1`/`out2`/`out3` — the input gradients `DX`, `DY`, and the optionally
  recomputed forward output `OUT`;
* `B = BLOCK_N` — the column window each program owns;
* `read1..3`/`write1..3` — **per-lane 2-D windows**: program `(pid₀, pid₁)`
  touches buffer `P` at `pid₀ · stride_p_row + pid₁ · BLOCK_N + j` (the
  host-side `(M, cdiv(N, BLOCK_N))` launch convention: `pid₀` picks the row,
  `pid₁` the column block, each buffer with its own row stride);
* `mask` — the active lanes `pid₁ · BLOCK_N + j < ncols`, shared by all three
  loads and the `DX`/`DY` stores (`read2Mask`/`read3Mask`/`writeMask1`/
  `writeMask2` keep their defaults);
* `writeMask3` — the **constexpr-gated** `OUT` store:
  `RECOMPUTE_OUTPUT = Bool.true ∧ pid₁ · BLOCK_N + j < ncols`. When the heuristic
  flag is off, the `OUT` channel has no write-active lanes, so it carries no
  value, bounds, or frame obligations — exactly the kernel's dead
  `if RECOMPUTE_OUTPUT` branch.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing, masking, and constexpr gating match
them. Buffer sizes are not signature content: the headline quantifies over
every allocation whose extents cover the write-active lanes. -/
```
```lean
def swigluBackwardIO (X Y DOUT OUT DX DY : RegionName)
    (stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N : Nat)
    (RECOMPUTE_OUTPUT : Bool) : Masked2DKernelIO₃ₓ₃ where
  kernel := swiglu_bwd_kernel X Y DOUT OUT DX DY
    stride_x_row stride_y_row stride_dout_row stride_out_row
    stride_dx_row stride_dy_row ncols BLOCK_N RECOMPUTE_OUTPUT
  in1 := X
  in2 := Y
  in3 := DOUT
  out1 := DX
  out2 := DY
  out3 := OUT
  B := BLOCK_N
  read1 := fun pid₀ pid₁ j => pid₀ * stride_x_row + (pid₁ * BLOCK_N + j.val)
  read2 := fun pid₀ pid₁ j => pid₀ * stride_y_row + (pid₁ * BLOCK_N + j.val)
  read3 := fun pid₀ pid₁ j => pid₀ * stride_dout_row + (pid₁ * BLOCK_N + j.val)
  write1 := fun pid₀ pid₁ j => pid₀ * stride_dx_row + (pid₁ * BLOCK_N + j.val)
  write2 := fun pid₀ pid₁ j => pid₀ * stride_dy_row + (pid₁ * BLOCK_N + j.val)
  write3 := fun pid₀ pid₁ j => pid₀ * stride_out_row + (pid₁ * BLOCK_N + j.val)
  mask := fun _ pid₁ j => pid₁ * BLOCK_N + j.val < ncols
  writeMask3 := fun _ pid₁ j =>
    RECOMPUTE_OUTPUT = Bool.true ∧ pid₁ * BLOCK_N + j.val < ncols
```
</details>

<details><summary><code>swiglu_bwd_kernel</code></summary>

```
/-- Faithful transcription of `swiglu_backward.py`'s `_swiglu_bwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N: tl.constexpr` / `RECOMPUTE_OUTPUT: tl.constexpr` -> Lean
  parameters. -/
```
```lean
def swiglu_bwd_kernel
    (X Y DOUT OUT DX DY : RegionName)
    (stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N : Nat)
    (RECOMPUTE_OUTPUT : Bool) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  start_col = tl.program_id(1) * $(BLOCK_N)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  DOUT += row * $(stride_dout_row)
  if RECOMPUTE_OUTPUT {
    OUT += row * $(stride_out_row)
  }
  DX += row * $(stride_dx_row)
  DY += row * $(stride_dy_row)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  y = tl.load(Y + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  dout = tl.load(DOUT + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  x_sigmoid = tl.sigmoid(x)
  dx = x_sigmoid * (1 + x * (1 - x_sigmoid)) * y * dout
  dy = x * x_sigmoid * dout
  tl.store(DX + cols, dx, mask=cols < $(ncols))
  tl.store(DY + cols, dy, mask=cols < $(ncols))
  if RECOMPUTE_OUTPUT {
    out = x * x_sigmoid * y
    tl.store(OUT + cols, out, mask=cols < $(ncols))
  }
}
```
</details>
