# Spec sheet — `bench/tritonbench_g/cosine_compute/CosineCompute.lean`

**Python source:** `bench/tritonbench_g/cosine_compute/cosine_compute.py`

## Public theorem: `cos_func_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `cos_func` implements lane-wise `Real.cos` on its masked
IO signature — for every disjoint flat placement of the two buffers, every
program id whose active lanes are in bounds, and every launch state whose
input window holds `xs` at the active lanes, the translated pointer kernel
terminates, every active output lane holds `Real.cos (xs i)`, and every other
memory cell is unchanged. Proof: `MaskedKernelIO₁.Implements.intro` assembles
the region-model masked triple with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification cos_func_correctness
    (a b : RegionName) (n_elements BLOCK_SIZE : Nat) :
    cosIO a b n_elements BLOCK_SIZE ⊨ fun xs i => Real.cos (xs i)
```

**Closed-form spec defs (transitive):** `cosIO`, `cos_func`

<details><summary><code>cosIO</code></summary>

```
/-- `cos_func`'s masked **IO signature** — the whole kernel-specific audit
surface of the headline: which buffer is which argument (the wiring), where
program `pid` reads its input tile / writes its output tile, and the
active-lane predicate `pid * BLOCK_SIZE + j < n_elements`. The windows and
mask are declared, not parsed from the kernel: they formalize the host-side
launch convention (`offset = pid * BLOCK_SIZE + arange;
mask = offset < n_elements`), and the headline **proves** the kernel's actual
addressing and masking match them. Buffer sizes are not signature content: the
headline quantifies over every allocation whose extents cover the active
lanes. -/
```
```lean
def cosIO (a b : RegionName) (n_elements BLOCK_SIZE : Nat) :
    MaskedKernelIO₁ where
  kernel := cos_func a b n_elements BLOCK_SIZE
  inp := a
  out := b
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements
```
</details>

<details><summary><code>cos_func</code></summary>

```
/-- Faithful 1:1 transcription of `cosine_compute.py`'s `cos_func`.

Allowed mechanical Lean-syntax-only changes:
-/
```
```lean
def cos_func
    (a b : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  offset = tl.program_id(0) * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offset < $(n_elements)
  a_value = tl.load(a + offset, mask=mask)
  b_value = tl.cos((a_value).to(tl.float32))
  tl.store(b + offset, b_value, mask=mask)
}
```
</details>
