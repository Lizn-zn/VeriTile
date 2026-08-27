# Spec sheet — `bench/tritonbench_g/dequantize_rowwise/DequantizeRowwise.lean`

**Python source:** `bench/tritonbench_g/dequantize_rowwise/dequantize_rowwise.py`

## Public theorem: `dequantize_rowwise_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `_dequantize_rowwise` implements per-row dequantization
on its masked IO signature — for every disjoint flat placement of the three
buffers, every program id whose active lanes are in bounds, and every launch
state whose row window holds `xs` and whose scale cell holds `ys` at the
active lanes, the translated pointer kernel terminates, every active output
lane holds `ys i * xs i * inv_127` (the scale times the quantized value times
`1/127`; `ys` is constant across the row's active lanes since every lane
reads the same cell), and every other memory cell is unchanged.

`0 < BLOCK_SIZE` / `0 < P2` are genuinely forced: the scale load is unmasked,
so a launch with no active lanes still reads `state_x[pid]`, whose bound the
lane-wise contract only supplies via an active lane. Both hold for every real
launch. Proof: `Masked2DKernelIO₂.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification dequantize_rowwise_correctness
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (n_elements BLOCK_SIZE P2 : Nat)
    (hBS : 0 < BLOCK_SIZE) (hP2 : 0 < P2) :
    dequantizeRowwiseIO x_ptr state_x output_ptr inv_127 n_elements
      BLOCK_SIZE P2 ⊨ fun _ _ xs ys i => ys i * xs i * inv_127
```

**Assumptions / layout contracts:**
- `hBS : 0 < BLOCK_SIZE`
- `hP2 : 0 < P2`

**Closed-form spec defs (transitive):** `dequantizeRowwiseIO`, `dequantize_rowwise_kernel`

<details><summary><code>dequantizeRowwiseIO</code></summary>

```
/-- `_dequantize_rowwise`'s masked **IO signature** — the whole
kernel-specific audit surface of the headline: which buffer is which
argument (the wiring), where program `pid` reads its `P2`-lane quantized row
(`x_ptr[pid·BLOCK_SIZE + j]`), where it reads its per-row scale — the
**single scalar cell** `state_x[pid]`, the same address at every active lane
— where it writes its output row, and the active-lane predicate
`j < BLOCK_SIZE`. The kernel's grid is 1D, so no field mentions the second
program-id axis. The windows and mask are declared, not parsed from the
kernel: they formalize the host-side launch convention (`offsets =
pid * BLOCK_SIZE + arange; row_mask = arange < BLOCK_SIZE`, one scale per
row), and the headline **proves** the kernel's actual addressing and masking
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the active lanes. -/
```
```lean
def dequantizeRowwiseIO (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (n_elements BLOCK_SIZE P2 : Nat) : Masked2DKernelIO₂ where
  kernel := dequantize_rowwise_kernel x_ptr state_x output_ptr inv_127
    n_elements BLOCK_SIZE P2
  in1 := x_ptr
  in2 := state_x
  out := output_ptr
  B := P2
  read1 := fun pid _ j => pid * BLOCK_SIZE + j.val
  read2 := fun pid _ _ => pid
  write := fun pid _ j => pid * BLOCK_SIZE + j.val
  mask := fun _ _ j => j.val < BLOCK_SIZE
```
</details>

<details><summary><code>dequantize_rowwise_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `dequantize_rowwise.py`'s `_dequantize_rowwise`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` / `P2: tl.constexpr` → Lean `Nat`
  parameters.
- `n_elements` is kept as `_n_elements`: the upstream Triton kernel accepts it
  but does not use it in the body. -/
```
```lean
def dequantize_rowwise_kernel
    (x_ptr state_x output_ptr : RegionName)
    (inv_127 : ℝ) (_n_elements BLOCK_SIZE P2 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  arange = tl.arange(0, $(P2))
  offsets = block_start + arange
  row_mask = arange < $(BLOCK_SIZE)
  x = tl.load(x_ptr + offsets, mask=row_mask)
  max_val = tl.load(state_x + pid)
  output = max_val * x * $(inv_127)
  tl.store(output_ptr + offsets, output, mask=row_mask)
}
```
</details>
