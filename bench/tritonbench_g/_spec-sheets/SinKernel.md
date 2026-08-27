# Spec sheet — `bench/tritonbench_g/sin_kernel/SinKernel.lean`

**Python source:** `bench/tritonbench_g/sin_kernel/sin_kernel.py`

## Public theorem: `kernel_function_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `kernel_function` implements lane-wise `Real.sin` on its
masked IO signature — for every disjoint flat placement of the two buffers,
every program id whose active lanes are in bounds, and every launch state
whose input window holds `xs` at the active lanes, the translated pointer
kernel terminates, every active output lane holds `Real.sin (xs i)`, and
every other memory cell is unchanged. Proof:
`MaskedKernelIO₁.Implements.intro` assembles the region-model masked triple
with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification kernel_function_correctness
    (x_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    sinIO x_ptr output_ptr n_elements BLOCK_SIZE
      ⊨ fun xs i => Real.sin (xs i)
```

**Closed-form spec defs (transitive):** `sinIO`, `kernel_function`

<details><summary><code>sinIO</code></summary>

```
/-- `kernel_function`'s masked **IO signature** — the whole kernel-specific
audit surface of the headline: which buffer is which argument (the wiring),
where program `pid` reads its input tile / writes its output tile, and the
active-lane predicate `pid * BLOCK_SIZE + j < n_elements` (the shared
load/store mask; the write side defaults to it). The windows and mask are
declared, not parsed from the kernel: they formalize the host-side launch
convention (`offsets = pid * BLOCK_SIZE + arange;
mask = offsets < n_elements`), and the headline **proves** the kernel's actual
addressing and masking match them. Buffer sizes are not signature content: the
headline quantifies over every allocation whose extents cover the active
lanes. -/
```
```lean
def sinIO (x_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₁ where
  kernel := kernel_function x_ptr output_ptr n_elements BLOCK_SIZE
  inp := x_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements
```
</details>

<details><summary><code>kernel_function</code></summary>

```
/-- Faithful 1:1 transcription of `sin_kernel.py`'s `kernel_function`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
def kernel_function
    (x_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  output = tl.math.sin(x)
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>
