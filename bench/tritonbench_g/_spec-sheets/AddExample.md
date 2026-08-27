# Spec sheet — `bench/tritonbench_g/add_example/AddExample.lean`

**Python source:** `bench/tritonbench_g/add_example/add_example.py`

## Public theorem: `add_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `add_kernel` implements pointwise addition on its masked
IO signature — for every disjoint flat placement of the three buffers, every
program id whose active lanes are in bounds, and every launch state whose
input windows hold `xs`/`ys` at the active lanes, the translated pointer
kernel terminates, every active output lane holds `xs i + ys i`, and every
other memory cell is unchanged. Proof: `MaskedKernelIO₂.Implements.intro`
assembles the region-model masked triple with the flat-memory bridge side
conditions. -/
```
</details>

**Statement:**
```lean
specification add_kernel_correctness
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    addIO in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
      ⊨ fun xs ys i => xs i + ys i
```

**Closed-form spec defs (transitive):** `addIO`, `add_kernel`

<details><summary><code>addIO</code></summary>

```
/-- `add_kernel`'s masked **IO signature** — the whole kernel-specific audit
surface of the headline: which buffer is which argument (the wiring), where
program `pid` reads its input tiles / writes its output tile, and the
active-lane predicate `pid * BLOCK_SIZE + j < n_elements`. The windows and
mask are declared, not parsed from the kernel: they formalize the host-side
launch convention (`offsets = pid * BLOCK_SIZE + arange;
mask = offsets < n_elements`), and the headline **proves** the kernel's actual
addressing and masking match them. Buffer sizes are not signature content: the
headline quantifies over every allocation whose extents cover the active
lanes. -/
```
```lean
def addIO (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₂ where
  kernel := add_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
  in1 := in_ptr0
  in2 := in_ptr1
  out := out_ptr
  B := BLOCK_SIZE
  read1 := fun pid => pid * BLOCK_SIZE
  read2 := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements
```
</details>

<details><summary><code>add_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `add_example.py`'s `add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` annotation → Lean `Nat` parameter
  (the `tl.constexpr` is implicit in Lean params).

Everything else is verbatim from the upstream kernel. -/
```
```lean
def add_kernel
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  y = tl.load(in_ptr1 + offsets, mask=mask)
  output = x + y
  tl.store(out_ptr + offsets, output, mask=mask)
}
```
</details>
