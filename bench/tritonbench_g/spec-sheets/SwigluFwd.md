# Spec sheet — `bench/tritonbench_g/swiglu_fwd/SwigluFwd.lean`

**Python source:** `bench/tritonbench_g/swiglu_fwd/swiglu_fwd.py`

## Public theorem: `swiglu_fwd_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `_swiglu_fwd_kernel`: the DSL surface lowers to
the algorithm layer, and the masked store to `OUT` is compute-correct — every
active lane holds `TiledActivation.swiglu (xs i) (ys i)`, out-of-bounds lanes are
preserved. -/
```
</details>

**Statement:**
```lean
theorem swiglu_fwd_kernel_output_summary
    (X Y OUT : RegionName)
    (stride_x_row stride_y_row stride_out_row ncols BLOCK_N : Nat)
    (s : BlockState)
    (xs ys : Fin BLOCK_N → ℝ)
    (h_x : ∀ i : Fin BLOCK_N, s.readMem X (swigluOffset s stride_x_row BLOCK_N i) = xs i)
    (h_y : ∀ i : Fin BLOCK_N, s.readMem Y (swigluOffset s stride_y_row BLOCK_N i) = ys i) :
    (∃ alg, (swiglu_fwd_kernel X Y OUT stride_x_row stride_y_row stride_out_row
        ncols BLOCK_N).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := swiglu_fwd_kernel X Y OUT stride_x_row stride_y_row stride_out_row
        ncols BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => s.pids 1 * BLOCK_N + i.val < ncols)
        (fun i => (OUT, swigluOffset s stride_out_row BLOCK_N i)))
      (expected := fun i => TiledActivation.swiglu (xs i) (ys i))
```

**Assumptions / layout contracts:**
- `xs ys : Fin BLOCK_N → ℝ`
- `h_x : ∀ i : Fin BLOCK_N, s.readMem X (swigluOffset s stride_x_row BLOCK_N i) = xs i`
- `h_y : ∀ i : Fin BLOCK_N, s.readMem Y (swigluOffset s stride_y_row BLOCK_N i) = ys i`
- `kernel : = swiglu_fwd_kernel X Y OUT stride_x_row stride_y_row stride_out_row
        ncols BLOCK_N`
- `initialState : = s`
- `fun i : Fin BLOCK_N => s.pids 1 * BLOCK_N + i.val < ncols`
- `expected : = fun i => TiledActivation.swiglu (xs i) (ys i)`

**Closed-form spec defs (transitive):** `swigluOffset`, `swiglu_fwd_kernel`

<details><summary><code>swigluOffset</code></summary>

```lean
def swigluOffset (s : BlockState) (stride : Nat) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * stride + s.pids 1 * BLOCK_N + i.val
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

## Also present (pinned special-case summaries)
- `swiglu_fwd_kernel_compute_correct`
