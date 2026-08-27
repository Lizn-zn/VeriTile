# Spec sheet — `bench/tritonbench_g/quantize_global/QuantizeGlobal.lean`

**Python source:** `bench/tritonbench_g/quantize_global/quantize_global.py`

## Public theorem: `quantize_global_correctness`

<details><summary>docstring</summary>

```
/-- **The headline** (blocked tail): the faithful full surface is recorded as
**blocked** at algorithm projection (it stores CUDA `llrint`/int8 results — the
honest, unmodeled blocker), while the checked store slice implements the
pre-rounding scaled quantization on its masked IO signature — for every
disjoint flat placement of the three buffers, every program id whose declared
reads are in bounds, and every launch state whose data window holds `xs` at
the active lanes and whose scale cell holds `ys`, the translated pointer
kernel terminates, every active output lane holds `scale127 * (xs i * ys i)`
(`ys` is constant across the lanes since every lane reads the same cell;
concrete launches use `scale127 = 127.0`), and every other memory cell is
unchanged.

`0 < BLOCK_SIZE` is genuinely forced: the scale load is unmasked and the
active mask is pid-dependent, so a tail program with no active lanes still
reads `absmax_inv_ptr[0]`, whose bound only the unconditional `read2Mask`
clause supplies — via a lane witness, hence `0 < BLOCK_SIZE`. It holds for
every real launch. Proof: `Masked2DKernelIO₂.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification quantize_global_correctness
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ)
    (hBS : 0 < BLOCK_SIZE) :
    (∃ err, (quantize_global_surface x_ptr absmax_inv_ptr output_ptr
        n_elements BLOCK_SIZE).toAlgorithm? = Except.error err) ∧
    (quantizeGlobalIO x_ptr absmax_inv_ptr output_ptr n_elements BLOCK_SIZE
        scale127 ⊨ fun _ _ xs ys i => scale127 * (xs i * ys i))
```

**Assumptions / layout contracts:**
- `hBS : 0 < BLOCK_SIZE`

**Closed-form spec defs (transitive):** `quantize_global_surface`, `quantizeGlobalIO`, `quantize_global_scaled_store_slice`

<details><summary><code>quantize_global_surface</code></summary>

```
/-- Faithful transcription of `quantize_global.py`'s `_quantize_global`.

The CUDA `llrint` operation is preserved as a surface operation; the algorithm
carrier records the pre-cast real value. -/
```
```lean
def quantize_global_surface
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = tl.extra.cuda.libdevice.llrint(127.0 * (x * absmax_inv))
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

<details><summary><code>quantizeGlobalIO</code></summary>

```
/-- The checked store slice's masked **IO signature** — the whole
kernel-specific audit surface of the headline's verified half: which buffer is
which argument (the wiring), where program `pid` reads its `BLOCK_SIZE`-lane
data window (`x_ptr[pid·BLOCK_SIZE + j]`), where it reads its scale — the
**single scalar cell** `absmax_inv_ptr[0]`, the same address at every lane,
read **unmasked** (`read2Mask := True`: the load executes even for a tail
program with no active data lanes) — where it writes its output window, and
the active-lane predicate `pid·BLOCK_SIZE + j < n_elements`. The kernel's grid
is 1D, so no field mentions the second program-id axis. The windows and masks
are declared, not parsed from the kernel: they formalize the host-side launch
convention (`offsets = pid * BLOCK_SIZE + arange; mask = offsets <
n_elements`, one global scale), and the headline **proves** the kernel's
actual addressing and masking match them. Buffer sizes are not signature
content: the headline quantifies over every allocation whose extents cover the
declared reads and writes. -/
```
```lean
def quantizeGlobalIO (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ) : Masked2DKernelIO₂ where
  kernel := quantize_global_scaled_store_slice x_ptr absmax_inv_ptr output_ptr
    n_elements BLOCK_SIZE scale127
  in1 := x_ptr
  in2 := absmax_inv_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read1 := fun pid _ j => pid * BLOCK_SIZE + j.val
  read2 := fun _ _ _ => 0
  write := fun pid _ j => pid * BLOCK_SIZE + j.val
  mask := fun pid _ j => pid * BLOCK_SIZE + j.val < n_elements
  read2Mask := fun _ _ _ => True
```
</details>

<details><summary><code>quantize_global_scaled_store_slice</code></summary>

```
/-- Proof-oriented arithmetic store slice of `quantize_global.py`'s
`_quantize_global`.

This slice proves the masked vector addressing and scaled value before the
backend-specific rounding/cast step. -/
```
```lean
def quantize_global_scaled_store_slice
    (x_ptr absmax_inv_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) (scale127 : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  absmax_inv = tl.load(absmax_inv_ptr)
  output = $(scale127) * (x * absmax_inv)
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>
