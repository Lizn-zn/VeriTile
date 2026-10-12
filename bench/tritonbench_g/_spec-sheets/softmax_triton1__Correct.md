# Spec sheet — `bench/tritonbench_g/softmax_triton1/Correct.lean`

**Python source:** `bench/tritonbench_g/softmax_triton1/softmax_triton1.py`

## Public theorem: `softmax_kernel_correctness`

<details><summary>docstring</summary>

```
/-- The kernel computes mathematical softmax over the active prefix, with
lane-wise bounds, termination and preservation of every other memory cell.
`0 < BLOCK_SIZE` makes the reduction tile nonempty; the active mask may be empty.
For `n_cols ≤ BLOCK_SIZE`, the formula covers the complete row. -/
```
</details>

**Statement:**
```lean
specification softmax_kernel_correctness
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) :
    Spec.Real (softmaxIO output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE ⊨
      fun xs i => Real.exp (xs i) /
        ∑ j : Fin BLOCK_SIZE, if j.val < n_cols then Real.exp (xs j) else 0)
```

**Assumptions / layout contracts:**
- `hB : 0 < BLOCK_SIZE`

**Closed-form spec defs (transitive):** `softmaxIO`, `softmax_kernel`

<details><summary><code>softmaxIO</code></summary>

```
/-- `softmax_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B = BLOCK_SIZE` — the row window each program owns;
* `read`/`write` — program `pid` reads its row at `pid * input_row_stride` and
  writes it at `pid * output_row_stride` (the host-side one-program-per-row
  launch convention);
* `mask` — the active lanes `j < n_cols`, **the same for every program**: the
  row prefix that actually exists in the matrix. Inactive lanes (the padding
  of `BLOCK_SIZE = next_power_of_2(n_cols)`) carry no obligations on either
  side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
```
```lean
def softmaxIO (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    MaskedKernelIO₁ where
  kernel := softmax_kernel output_ptr input_ptr input_row_stride
    output_row_stride n_cols BLOCK_SIZE
  inp := input_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * input_row_stride
  write := fun pid => pid * output_row_stride
  mask := fun _ j => j.val < n_cols
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

## Public theorem: `softmax_reciprocal_correctness`

**Statement:**
```lean
specification softmax_reciprocal_correctness
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) :
    Spec.Real (reciprocalIO output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE ⊨
      fun xs i => Real.exp (xs i) /
        ∑ j : Fin BLOCK_SIZE, if j.val < n_cols then Real.exp (xs j) else 0)
```

**Assumptions / layout contracts:**
- `hB : 0 < BLOCK_SIZE`

**Closed-form spec defs (transitive):** `reciprocalIO`, `softmaxIO`, `softmax_reciprocal_kernel`, `softmax_kernel`

<details><summary><code>reciprocalIO</code></summary>

```
/-- The candidate retains the same masked windows and memory contract. -/
```
```lean
def reciprocalIO (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₁ :=
  { softmaxIO output_ptr input_ptr input_row_stride output_row_stride n_cols BLOCK_SIZE with
    kernel := softmax_reciprocal_kernel output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE
    projection := by rfl }
```
</details>

<details><summary><code>softmaxIO</code></summary>

```
/-- `softmax_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B = BLOCK_SIZE` — the row window each program owns;
* `read`/`write` — program `pid` reads its row at `pid * input_row_stride` and
  writes it at `pid * output_row_stride` (the host-side one-program-per-row
  launch convention);
* `mask` — the active lanes `j < n_cols`, **the same for every program**: the
  row prefix that actually exists in the matrix. Inactive lanes (the padding
  of `BLOCK_SIZE = next_power_of_2(n_cols)`) carry no obligations on either
  side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
```
```lean
def softmaxIO (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    MaskedKernelIO₁ where
  kernel := softmax_kernel output_ptr input_ptr input_row_stride
    output_row_stride n_cols BLOCK_SIZE
  inp := input_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * input_row_stride
  write := fun pid => pid * output_row_stride
  mask := fun _ j => j.val < n_cols
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
