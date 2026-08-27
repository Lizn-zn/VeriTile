# Spec sheet — `bench/tritonbench_g/square_matrix/SquareMatrix.lean`

**Python source:** `bench/tritonbench_g/square_matrix/square_matrix.py`

## Public theorem: `square_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `square_kernel` implements the lane-wise square
`xs i * xs i` on its masked IO signature — for every disjoint flat placement
of the two buffers, every program id whose active lanes are in bounds, and
every launch state whose active input-row lanes hold `xs`, the translated
pointer kernel terminates, every active output-row lane holds `xs i * xs i`,
and every other memory cell is unchanged. Proof:
`MaskedKernelIO₁.Implements.intro` assembles the region-model masked triple
with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification square_kernel_correctness
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    squareIO output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE ⊨
      fun xs i => xs i * xs i
```

**Closed-form spec defs (transitive):** `squareIO`, `square_kernel`

<details><summary><code>squareIO</code></summary>

```
/-- `square_kernel`'s masked **IO signature** — the whole kernel-specific
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
def squareIO (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    MaskedKernelIO₁ where
  kernel := square_kernel output_ptr input_ptr input_row_stride
    output_row_stride n_cols BLOCK_SIZE
  inp := input_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * input_row_stride
  write := fun pid => pid * output_row_stride
  mask := fun _ j => j.val < n_cols
```
</details>

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
