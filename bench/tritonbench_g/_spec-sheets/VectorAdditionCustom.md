# Spec sheet — `bench/tritonbench_g/vector_addition_custom/VectorAdditionCustom.lean`

**Python source:** `bench/tritonbench_g/vector_addition_custom/vector_addition_custom.py`

## Public theorem: `add_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `_add_kernel` implements pointwise addition on its masked
IO signature — for every disjoint flat placement of the three buffers, every
program id whose active lanes are in bounds, and every launch state whose
input windows hold `as`/`bs` at the active lanes, the translated pointer
kernel terminates, every active output lane holds `as i + bs i`, and every
other memory cell is unchanged. Proof: `MaskedKernelIO₂.Implements.intro`
assembles the region-model masked triple with the flat-memory bridge side
conditions. -/
```
</details>

**Statement:**
```lean
specification add_kernel_correctness
    (A B C : RegionName)
    (size BLOCK : Nat) :
    addCustomIO A B C size BLOCK
      ⊨ fun as bs i => as i + bs i
```

**Closed-form spec defs (transitive):** `addCustomIO`, `_add_kernel`

<details><summary><code>addCustomIO</code></summary>

```
/-- `_add_kernel`'s masked **IO signature** — the whole kernel-specific audit
surface of the headline: which buffer is which argument (the wiring), where
program `prog_id` reads its input tiles / writes its output tile, and the
active-lane predicate `prog_id * BLOCK + j < size`. The windows and mask are
declared, not parsed from the kernel: they formalize the host-side launch
convention (`offs = prog_id * BLOCK + arange; mask = offs < size`), and the
headline **proves** the kernel's actual addressing and masking match them.
Buffer sizes are not signature content: the headline quantifies over every
allocation whose extents cover the active lanes. -/
```
```lean
def addCustomIO (A B C : RegionName)
    (size BLOCK : Nat) : MaskedKernelIO₂ where
  kernel := _add_kernel A B C size BLOCK
  in1 := A
  in2 := B
  out := C
  B := BLOCK
  read1 := fun pid => pid * BLOCK
  read2 := fun pid => pid * BLOCK
  write := fun pid => pid * BLOCK
  mask := fun pid j => pid * BLOCK + j.val < size
```
</details>

<details><summary><code>_add_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `vector_addition_custom.py`'s `_add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
def _add_kernel
    (A B C : RegionName)
    (size BLOCK : Nat) :
    ComputeKernel := triton {
  prog_id = tl.program_id(0)
  offs = prog_id * $(BLOCK) + tl.arange(0, $(BLOCK))
  a = tl.load(A + offs, mask=offs < $(size))
  b = tl.load(B + offs, mask=offs < $(size))
  tl.store(C + offs, a + b, mask=offs < $(size))
}
```
</details>
