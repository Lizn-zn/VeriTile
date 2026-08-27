# Spec sheet — `bench/tritonbench_g/dropout_triton/DropoutTriton.lean`

**Python source:** `bench/tritonbench_g/dropout_triton/dropout_triton.py`

## Public theorem: `dropout_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `_dropout` implements lane-wise keep-gated inverted
dropout on its masked bool-input IO signature — for every disjoint flat
placement of the three buffers, every program id whose active lanes are in
bounds, and every launch state whose input windows hold the data tile `xs`
and the keep tile `bs` at the active lanes, the translated pointer kernel
terminates, every active output lane holds `if bs i then xs i / (1 - p)
else 0`, and every other memory cell is unchanged. Proof:
`BoolMasked2DKernelIO₁.Implements.intro` assembles the region-model masked
triple with the flat-memory bridge side conditions; the store is gated by the
static mask, so `hsub` is the identity. -/
```
</details>

**Statement:**
```lean
specification dropout_kernel_correctness
    (x_ptr x_keep_ptr output_ptr : RegionName)
    (n_elements : Nat) (p : ℝ) (BLOCK_SIZE : Nat) :
    dropoutIO x_ptr x_keep_ptr output_ptr n_elements p BLOCK_SIZE ⊨
      fun _ _ bs xs i => if bs i then xs i / (1 - p) else 0
```

**Closed-form spec defs (transitive):** `dropoutIO`, `dropout_kernel`

<details><summary><code>dropoutIO</code></summary>

```
/-- `_dropout`'s masked bool-input **IO signature** — the whole
kernel-specific audit surface of the headline: which buffer is which argument
(the wiring: `inp` the ℝ data, `mbuf` the `.bool` keep-mask, `out` the
output), where program `pid` reads/writes its `BLOCK_SIZE`-lane window (all
three at `pid * BLOCK_SIZE + j`), and the active-lane predicate
`pid * BLOCK_SIZE + j < n_elements`. The grid is 1D, so the second program-id
axis is unused. The store is masked by the same *static* load mask — the keep
bit gates the stored **value**, not the write set — so `writeMask` keeps the
struct's default (= `mask`). The windows and mask are declared, not parsed
from the kernel: they formalize the host-side launch convention
(`offsets = pid * BLOCK_SIZE + arange; mask = offsets < n_elements`), and the
headline **proves** the kernel's actual addressing and masking match them.
Buffer sizes are not signature content: the headline quantifies over every
allocation whose extents cover the active lanes. -/
```
```lean
def dropoutIO (x_ptr x_keep_ptr output_ptr : RegionName)
    (n_elements : Nat) (p : ℝ) (BLOCK_SIZE : Nat) : BoolMasked2DKernelIO₁ where
  kernel := dropout_kernel x_ptr x_keep_ptr output_ptr n_elements p BLOCK_SIZE
  inp := x_ptr
  mbuf := x_keep_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid _ j => pid * BLOCK_SIZE + j.val
  readm := fun pid _ j => pid * BLOCK_SIZE + j.val
  write := fun pid _ j => pid * BLOCK_SIZE + j.val
  mask := fun pid _ j => pid * BLOCK_SIZE + j.val < n_elements
```
</details>

<details><summary><code>dropout_kernel</code></summary>

```
/-- Faithful transcription of `dropout_triton.py`'s `_dropout`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter.
- `x_keep_ptr` is a typed Lean boolean region so its `tl.load` call does not
  need an extra `dtype=` kwarg. -/
```
```lean
def dropout_kernel
    (x_ptr : RegionName) (x_keep_ptr : Region .bool) (output_ptr : RegionName)
    (n_elements : Nat) (p : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
  output = tl.where(x_keep, x / (1 - $(p)), 0.0)
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>
