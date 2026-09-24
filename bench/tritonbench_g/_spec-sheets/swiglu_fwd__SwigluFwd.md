# Spec sheet — `bench/tritonbench_g/swiglu_fwd/SwigluFwd.lean`

**Python source:** `bench/tritonbench_g/swiglu_fwd/swiglu_fwd.py`

## Public theorem: `swiglu_fwd_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `_swiglu_fwd_kernel` implements the oracle SwiGLU
`TiledActivation.swiglu` (i.e. `x · σ(x) · y`) lane-wise on its masked 2-D IO
signature — for every disjoint flat placement of the three buffers, every
program-id pair `(row, col_block)` whose active lanes are in bounds, and every
launch state whose active input-window lanes hold `xs`/`ys`, the translated
pointer kernel terminates, every active output lane `j` holds
`TiledActivation.swiglu (xs j) (ys j)`, and every other memory cell is
unchanged. Proof: `Masked2DKernelIO₂.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification swiglu_fwd_kernel_correctness
    (X Y OUT : RegionName)
    (stride_x_row stride_y_row stride_out_row ncols BLOCK_N : Nat) :
    swigluIO X Y OUT stride_x_row stride_y_row stride_out_row ncols BLOCK_N
      ⊨ fun _ _ xs ys i => TiledActivation.swiglu (xs i) (ys i)
```

**Closed-form spec defs (transitive):** `swigluIO`, `swiglu_fwd_kernel`

<details><summary><code>swigluIO</code></summary>

```
/-- `_swiglu_fwd_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `in1`/`in2`/`out` — which buffer is which argument (the wiring: gate `X`,
  value `Y`, result `OUT`);
* `B = BLOCK_N` — the column tile each program owns;
* `read1`/`read2`/`write` — program `(row, col_block) = (pid₀, pid₁)` touches
  lane `j` at `pid₀ * stride + pid₁ * BLOCK_N + j` in all three buffers (the
  host-side 2-D grid `(M, cdiv(N, BLOCK_N))` launch convention, per-buffer row
  strides);
* `mask` — the active lanes `pid₁ * BLOCK_N + j < ncols`: the column prefix
  that actually exists in the row. Inactive lanes (the overhang of the last
  column block) carry no obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
```
```lean
def swigluIO (X Y OUT : RegionName)
    (stride_x_row stride_y_row stride_out_row ncols BLOCK_N : Nat) :
    Masked2DKernelIO₂ where
  kernel := swiglu_fwd_kernel X Y OUT stride_x_row stride_y_row stride_out_row
    ncols BLOCK_N
  in1 := X
  in2 := Y
  out := OUT
  B := BLOCK_N
  read1 := fun pid₀ pid₁ j => pid₀ * stride_x_row + pid₁ * BLOCK_N + j.val
  read2 := fun pid₀ pid₁ j => pid₀ * stride_y_row + pid₁ * BLOCK_N + j.val
  write := fun pid₀ pid₁ j => pid₀ * stride_out_row + pid₁ * BLOCK_N + j.val
  mask := fun _ pid₁ j => pid₁ * BLOCK_N + j.val < ncols
```
</details>

<details><summary><code>swiglu_fwd_kernel</code></summary>

```
/-- Faithful transcription of `swiglu_fwd.py`'s `_swiglu_fwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
def swiglu_fwd_kernel
    (X Y OUT : RegionName)
    (stride_x_row stride_y_row stride_out_row ncols BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  start_col = tl.program_id(1) * $(BLOCK_N)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  OUT += row * $(stride_out_row)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  y = tl.load(Y + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  out = x * tl.sigmoid(x) * y
  tl.store(OUT + cols, out, mask=cols < $(ncols))
}
```
</details>
