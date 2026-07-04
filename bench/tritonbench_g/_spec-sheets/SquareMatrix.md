# Spec sheet — `bench/tritonbench_g/square_matrix/SquareMatrix.lean`

**Python source:** `bench/tritonbench_g/square_matrix/square_matrix.py`

## Public theorem: `square_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `square_kernel`: the DSL surface lowers to the
algorithm layer, and the masked store to `output_ptr` is compute-correct — every
active column holds `xs i * xs i`, out-of-bounds columns are preserved. -/
```
</details>

**Statement:**
```lean
theorem square_kernel_output_summary
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputRowLoadedAt s input_ptr input_row_stride BLOCK_SIZE xs) :
    (∃ alg, (square_kernel output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := square_kernel output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => i.val < n_cols)
          (fun i => (output_ptr, s.pid * output_row_stride + i.val)))
      (expected := fun i => xs i * xs i)
```

**Assumptions / layout contracts:**
- `hBlockSize : 0 < BLOCK_SIZE`
- `xs : Fin BLOCK_SIZE → ℝ`
- `h_x : InputRowLoadedAt s input_ptr input_row_stride BLOCK_SIZE xs`
- `fun i : Fin BLOCK_SIZE => i.val < n_cols`

**Closed-form spec defs (transitive):** `square_kernel`

<details><summary><code>square_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `square_matrix.py`'s `square_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
def square_kernel
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(input_row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))
  square_output = row * row
  output_row_start_ptr = output_ptr + row_idx * $(output_row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, square_output, mask=col_offsets < $(n_cols))
}
```
</details>

## Also present (pinned special-case summaries)
- `square_kernel_compute_correct`
