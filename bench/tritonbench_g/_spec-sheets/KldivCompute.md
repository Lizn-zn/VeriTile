# Spec sheet — `bench/tritonbench_g/kldiv_compute/KldivCompute.lean`

**Python source:** `bench/tritonbench_g/kldiv_compute/kldiv_compute.py`

## Public theorem: `kldivergence_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `kldivergence_kernel` implements the pointwise KL term
`klDivSpec x y = x * log (x / y)` on its masked IO signature — for every
disjoint flat placement of the three buffers, every program id whose active
lanes are in bounds, and every launch state whose input windows hold `xs`/`ys`
at the active lanes, the translated pointer kernel terminates, every active
output lane holds `klDivSpec (xs i) (ys i)`, and every other memory cell is
unchanged. Proof: `MaskedKernelIO₂.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification kldivergence_kernel_correctness
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    kldivIO x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
      ⊨ fun xs ys i => klDivSpec (xs i) (ys i)
```

**Closed-form spec defs (transitive):** `kldivIO`, `kldivergence_kernel`

<details><summary><code>kldivIO</code></summary>

```
/-- `kldivergence_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the headline: which buffer is which argument (the wiring),
where program `pid` reads its input tiles / writes its output tile, and the
active-lane predicate `pid * BLOCK_SIZE + j < n_elements`. The windows and
mask are declared, not parsed from the kernel: they formalize the host-side
launch convention (`offsets = pid * BLOCK_SIZE + arange;
mask = offsets < n_elements`), and the headline **proves** the kernel's actual
addressing and masking match them. Buffer sizes are not signature content: the
headline quantifies over every allocation whose extents cover the active
lanes. -/
```
```lean
def kldivIO (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₂ where
  kernel := kldivergence_kernel x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
  in1 := x_ptr
  in2 := y_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read1 := fun pid => pid * BLOCK_SIZE
  read2 := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements
```
</details>

<details><summary><code>kldivergence_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `kldiv_compute.py`'s
`kldivergence_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
def kldivergence_kernel
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x * tl.log(x / y)
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>
