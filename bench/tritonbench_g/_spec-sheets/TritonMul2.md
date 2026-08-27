# Spec sheet — `bench/tritonbench_g/triton_mul2/TritonMul2.lean`

**Python source:** `bench/tritonbench_g/triton_mul2/triton_mul2.py`

## Public theorem: `mul2_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `mul2_kernel` implements lane-wise doubling on its masked
IO signature — for every flat placement of the two buffers, every program id
whose active lanes are in bounds, and every launch state whose input window
holds `xs` at the active lanes, the translated pointer kernel terminates,
every active output lane holds `2 * xs i`, and every other memory cell is
unchanged. Proof: `MaskedKernelIO₁.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification mul2_kernel_correctness
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    mul2IO in_ptr0 out_ptr n_elements BLOCK_SIZE
      ⊨ fun xs i => 2 * xs i
```

**Closed-form spec defs (transitive):** `mul2IO`, `mul2_kernel`

<details><summary><code>mul2IO</code></summary>

```
/-- `mul2_kernel`'s masked **IO signature** — the whole kernel-specific audit
surface of the headline: which buffer is which argument (the wiring), where
program `pid` reads its input tile / writes its output tile, and the
active-lane predicate `pid * BLOCK_SIZE + j < n_elements`. The windows and
mask are declared, not parsed from the kernel: they formalize the host-side
launch convention (`offsets = pid * BLOCK_SIZE + arange;
mask = offsets < n_elements`), and the headline **proves** the kernel's actual
addressing and masking match them. Buffer sizes are not signature content: the
headline quantifies over every allocation whose extents cover the active
lanes. -/
```
```lean
def mul2IO (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₁ where
  kernel := mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE
  inp := in_ptr0
  out := out_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements
```
</details>

<details><summary><code>mul2_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `triton_mul2.py`'s `mul2_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
def mul2_kernel
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  output = 2 * x
  tl.store(out_ptr + offsets, output, mask=mask)
}
```
</details>

## Public theorem: `mul2_inplace_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `mul2_inplace_kernel` implements lane-wise doubling on
its masked in-place IO signature — for every flat placement of the buffer,
every program id whose active lanes are in bounds, and every launch state
whose window holds `xs` at the active lanes, the translated pointer kernel
terminates, every active lane of the *same* buffer ends up holding
`2 * xs i` (the doubled originally-loaded value), and every other memory cell
is unchanged. Proof: `MaskedKernelIO₁.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification mul2_inplace_kernel_correctness
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    mul2InplaceIO ptr n_elements BLOCK_SIZE
      ⊨ fun xs i => 2 * xs i
```

**Closed-form spec defs (transitive):** `mul2InplaceIO`, `mul2_inplace_kernel`

<details><summary><code>mul2InplaceIO</code></summary>

```
/-- `mul2_inplace_kernel`'s masked **in-place IO signature**: both argument
roles are wired to the same buffer (`inp = out = ptr`) — the kernel rewrites
the window it read. Windows and mask are the standard launch convention
(`offsets = pid * BLOCK_SIZE + arange; mask = offsets < n_elements`); the
headline **proves** the kernel's actual addressing and masking match them. -/
```
```lean
def mul2InplaceIO (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₁ where
  kernel := mul2_inplace_kernel ptr n_elements BLOCK_SIZE
  inp := ptr
  out := ptr
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements
```
</details>

<details><summary><code>mul2_inplace_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `triton_mul2.py`'s `mul2_inplace_kernel`.

Same allowed mechanical Lean-syntax-only changes as `mul2_kernel`. -/
```
```lean
def mul2_inplace_kernel
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(ptr + offsets, mask=mask)
  output = 2 * x
  tl.store(ptr + offsets, output, mask=mask)
}
```
</details>
