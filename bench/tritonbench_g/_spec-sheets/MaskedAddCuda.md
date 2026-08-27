# Spec sheet — `bench/tritonbench_g/masked_add_cuda/MaskedAddCuda.lean`

**Python source:** `bench/tritonbench_g/masked_add_cuda/masked_add_cuda.py`

## Public theorem: `masked_add_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `masked_add_kernel` implements the masked in-place fused
add on its bool-input IO signature — for every disjoint flat placement of the
buffers, every program id whose static-mask lanes are in bounds, and every
launch state whose `p`/`grad`/`p_mask` windows hold `xs`/`ys`/`bs` at the
static-mask lanes, the translated pointer kernel terminates, every
write-active lane `j` (in-bounds and `bs j = false`, the kernel's
`mask & ~p_mask`) of the *same* buffer `grad_ptr` ends up holding
`ys j + xs j * alpha` (the fused update of the originally-loaded window), and
every other memory cell — including the in-bounds lanes vetoed by `p_mask` —
is unchanged. Proof: `BoolMasked2DKernelIO₂.Implements.intro` assembles the
region-model masked in-place triple with the flat-memory bridge side
conditions. -/
```
</details>

**Statement:**
```lean
specification masked_add_kernel_correctness
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) :
    maskedAddIO grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
      ⊨ fun _ _ _ xs ys j => ys j + xs j * alpha
```

**Closed-form spec defs (transitive):** `maskedAddIO`, `masked_add_kernel`

<details><summary><code>maskedAddIO</code></summary>

```
/-- `masked_add_kernel`'s masked **in-place IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `in1`/`in2`/`mbuf` — which buffer is which argument (data `p_ptr`, running
  buffer `grad_ptr`, boolean veto `p_mask_ptr`);
* `out = in2 = grad_ptr` — the update is **in-place** (duplicate-region
  wiring, the `triton_mul2` precedent): the triple reads the *old* window
  contents into `ys` and asserts the *new* ones;
* `B = BLOCK_SIZE`, all four windows at lane `j` = `pid₀ * BLOCK_SIZE + j`
  (the host-side 1-D `cdiv(n_elements, BLOCK_SIZE)` launch convention; the
  family's second program id is ignored);
* `mask` — the **static** bounds lanes `pid₀ * BLOCK_SIZE + j < n_elements`
  (the trace-safety/input superset);
* `writeMask` — the **data-dependent** store gate
  `pid₀ * BLOCK_SIZE + j < n_elements ∧ bs j = false`: exactly the kernel's
  `mask = mask & ~p_mask` narrowing.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing, masking, and `p_mask` narrowing
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the static-mask lanes. -/
```
```lean
def maskedAddIO (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) :
    BoolMasked2DKernelIO₂ where
  kernel := masked_add_kernel grad_ptr p_ptr p_mask_ptr n_elements alpha
    BLOCK_SIZE
  in1 := p_ptr
  in2 := grad_ptr
  mbuf := p_mask_ptr
  out := grad_ptr
  B := BLOCK_SIZE
  read1 := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  read2 := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  readm := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  write := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  mask := fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val < n_elements
  writeMask := fun pid₀ _ bs j =>
    pid₀ * BLOCK_SIZE + j.val < n_elements ∧ bs j = Bool.false
```
</details>

<details><summary><code>masked_add_kernel</code></summary>

```
/-- Faithful transcription of `masked_add_cuda.py`'s `masked_add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter. -/
```
```lean
def masked_add_kernel
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
  mask = mask & ~p_mask
  p = tl.load(p_ptr + offsets, mask=mask)
  grad = tl.load(grad_ptr + offsets, mask=mask)
  grad += p * $(alpha)
  tl.store(grad_ptr + offsets, grad, mask=mask)
}
```
</details>
