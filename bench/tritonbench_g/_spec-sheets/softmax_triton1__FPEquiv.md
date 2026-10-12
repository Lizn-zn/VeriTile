# Spec sheet — `bench/tritonbench_g/softmax_triton1/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/softmax_triton1/softmax_triton1.py`

## Public theorem: `softmax_reciprocal_equiv`

**Statement:**
```lean
specification softmax_reciprocal_equiv (R : Rules)
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    original output_ptr input_ptr input_row_stride output_row_stride n_cols BLOCK_SIZE ≡[R]
      reciprocal output_ptr input_ptr input_row_stride output_row_stride n_cols BLOCK_SIZE
```

**Closed-form spec defs (transitive):** `Rules`, `original`, `reciprocal`, `softmax_kernel`, `divisionSite`, `softmax_reciprocal_kernel`

<details><summary><code>Rules</code></summary>

```lean
abbrev Rules := FP.ScalarArithmetic.Rules
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) : Program where
  kernel := softmax_kernel output_ptr input_ptr input_row_stride output_row_stride
    n_cols BLOCK_SIZE
  domain := [divisionSite output_ptr input_ptr input_row_stride output_row_stride
    n_cols BLOCK_SIZE]
```
</details>

<details><summary><code>reciprocal</code></summary>

```lean
def reciprocal (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) : Program :=
  { original output_ptr input_ptr input_row_stride output_row_stride n_cols BLOCK_SIZE with
    kernel := softmax_reciprocal_kernel output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE }
```
</details>

<details><summary><code>softmax_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `softmax_triton1.py`'s `softmax_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
def softmax_kernel
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(input_row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))
  row_minus_max = row - tl.max(row, axis=0)
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  output_row_start_ptr = output_ptr + row_idx * $(output_row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, softmax_output, mask=col_offsets < $(n_cols))
}
```
</details>

<details><summary><code>divisionSite</code></summary>

```
/-- All computed numerator lanes must be finite; the scalar denominator must
be finite and nonzero. Masked lanes still participate in the arithmetic.
No positivity, experimental shape or input distribution is added. -/
```
```lean
def divisionSite (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) : Site :=
  .divScalar (Source.prefixBefore
    (softmax_kernel output_ptr input_ptr input_row_stride output_row_stride
      n_cols BLOCK_SIZE).surfaceBody "softmax_output") BLOCK_SIZE []
    (.ref .real [BLOCK_SIZE] "numerator") (.ref .real [] "denominator")
```
</details>

<details><summary><code>softmax_reciprocal_kernel</code></summary>

```
/-- Reciprocal candidate; input, output, masking and reduction order are unchanged. -/
```
```lean
def softmax_reciprocal_kernel
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(input_row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))
  row_minus_max = row - tl.max(row, axis=0)
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator * (1 / denominator)
  output_row_start_ptr = output_ptr + row_idx * $(output_row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, softmax_output, mask=col_offsets < $(n_cols))
}
```
</details>
