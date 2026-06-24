# Spec sheet — `bench/tritonbench_g/swiglu_backward/SwigluBackward.lean`

**Python source:** `bench/tritonbench_g/swiglu_backward/swiglu_backward.py`

## Public theorem: `swiglu_bwd_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `_swiglu_bwd_kernel`: the DSL surface lowers to
the algorithm layer, and the (up to three) masked stores are compute-correct —
every active lane writes `swigluBwdA` to `DX`, `swigluBwdB` to `DY`, and, when
`RECOMPUTE_OUTPUT`, the forward `swiglu` to `OUT`; channels are indexed by `Sum`
and out-of-bounds / disabled lanes are preserved. Assumes the three output
regions are pairwise distinct. -/
```
</details>

**Statement:**
```lean
theorem swiglu_bwd_kernel_output_summary
    (X Y DOUT OUT DX DY : RegionName)
    (stride_x_row stride_y_row stride_dout_row stride_out_row
      stride_dx_row stride_dy_row ncols BLOCK_N : Nat)
    (RECOMPUTE_OUTPUT : Bool)
    (s : BlockState)
    (xs ys douts : Fin BLOCK_N → ℝ)
    (hDXDY : DX ≠ DY) (hOUTDX : OUT ≠ DX) (hOUTDY : OUT ≠ DY)
    (h_x : ∀ i : Fin BLOCK_N, s.readMem X (swigluOffset s stride_x_row BLOCK_N i) = xs i)
    (h_y : ∀ i : Fin BLOCK_N, s.readMem Y (swigluOffset s stride_y_row BLOCK_N i) = ys i)
    (h_dout : ∀ i : Fin BLOCK_N, s.readMem DOUT (swigluOffset s stride_dout_row BLOCK_N i) = douts i) :
    (∃ alg, (swiglu_bwd_kernel X Y DOUT OUT DX DY
        stride_x_row stride_y_row stride_dout_row stride_out_row
        stride_dx_row stride_dy_row ncols BLOCK_N RECOMPUTE_OUTPUT).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := swiglu_bwd_kernel X Y DOUT OUT DX DY
        stride_x_row stride_y_row stride_dout_row stride_out_row
        stride_dx_row stride_dy_row ncols BLOCK_N RECOMPUTE_OUTPUT)
      (initialState := s)
      (write := fun i : Sum (Sum (Fin BLOCK_N) (Fin BLOCK_N)) (Fin BLOCK_N) =>
        match i with
        | .inl (.inl lane) =>
            if s.pids 1 * BLOCK_N + lane.val < ncols then
              some (DX, swigluOffset s stride_dx_row BLOCK_N lane)
            else none
        | .inl (.inr lane) =>
            if s.pids 1 * BLOCK_N + lane.val < ncols then
              some (DY, swigluOffset s stride_dy_row BLOCK_N lane)
            else none
        | .inr lane =>
            if RECOMPUTE_OUTPUT then
              if s.pids 1 * BLOCK_N + lane.val < ncols then
                some (OUT, swigluOffset s stride_out_row BLOCK_N lane)
              else none
            else none)
      (expected := fun i =>
        match i with
        | .inl (.inl lane) => TiledActivation.swigluBwdA (douts lane) (xs lane) (ys lane)
        | .inl (.inr lane) => TiledActivation.swigluBwdB (douts lane) (xs lane)
        | .inr lane => TiledActivation.swiglu (xs lane) (ys lane))
```

**Assumptions / layout contracts:**
- `xs ys douts : Fin BLOCK_N → ℝ`
- `hDXDY : DX ≠ DY`
- `hOUTDX : OUT ≠ DX`
- `hOUTDY : OUT ≠ DY`
- `h_x : ∀ i : Fin BLOCK_N, s.readMem X (swigluOffset s stride_x_row BLOCK_N i) = xs i`
- `h_y : ∀ i : Fin BLOCK_N, s.readMem Y (swigluOffset s stride_y_row BLOCK_N i) = ys i`
- `h_dout : ∀ i : Fin BLOCK_N, s.readMem DOUT (swigluOffset s stride_dout_row BLOCK_N i) = douts i`
- `kernel : = swiglu_bwd_kernel X Y DOUT OUT DX DY
        stride_x_row stride_y_row stride_dout_row stride_out_row
        stride_dx_row stride_dy_row ncols BLOCK_N RECOMPUTE_OUTPUT`
- `initialState : = s`
- `expected : = fun i =>
        match i with
        | .inl (.inl lane) => TiledActivation.swigluBwdA (douts lane) (xs lane) (ys lane)
        | .inl (.inr lane) => TiledActivation.swigluBwdB (douts lane) (xs lane)
        | .inr lane => TiledActivation.swiglu (xs lane) (ys lane)`

**Closed-form spec defs (transitive):** `swigluOffset`, `swiglu_bwd_kernel`

<details><summary><code>swigluOffset</code></summary>

```lean
def swigluOffset (s : BlockState) (stride : Nat) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * stride + s.pids 1 * BLOCK_N + i.val
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

## Also present (pinned special-case summaries)
- `swiglu_bwd_kernel_compute_correct`
