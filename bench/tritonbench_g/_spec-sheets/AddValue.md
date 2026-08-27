# Spec sheet — `bench/tritonbench_g/add_value/AddValue.lean`

**Python source:** `bench/tritonbench_g/add_value/add_value.py`

## Public theorem: `puzzle1_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `puzzle1_kernel` implements lane-wise add-a-scalar
`xs i + value` on its masked IO signature — for every disjoint flat placement
of the two buffers, every program id whose active lanes are in bounds, and
every launch state whose input window holds `xs` at the active lanes, the
translated pointer kernel terminates, every active output lane holds
`xs i + value`, and every other memory cell is unchanged. Proof:
`MaskedKernelIO₁.Implements.intro` assembles the region-model masked triple
with the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification puzzle1_kernel_correctness
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) :
    addValueIO x_ptr output_ptr N BLOCK_SIZE value
      ⊨ fun xs i => xs i + value
```

**Closed-form spec defs (transitive):** `addValueIO`, `puzzle1_kernel`

<details><summary><code>addValueIO</code></summary>

```
/-- `puzzle1_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the headline: which buffer is which argument (the wiring),
where program `pid` reads its input tile / writes its output tile, and the
active-lane predicate `pid * BLOCK_SIZE + j < N`. The windows and mask are
declared, not parsed from the kernel: they formalize the host-side launch
convention (`offsets = pid * BLOCK_SIZE + arange; mask = offsets < N`), and
the headline **proves** the kernel's actual addressing and masking match
them. Buffer sizes are not signature content: the headline quantifies over
every allocation whose extents cover the active lanes. -/
```
```lean
def addValueIO (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) : MaskedKernelIO₁ where
  kernel := puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value
  inp := x_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < N
```
</details>

<details><summary><code>puzzle1_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `add_value.py`'s `puzzle1_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter.
- `value` (Lean `ℝ` parameter) injected via `$(...)`. -/
```
```lean
def puzzle1_kernel
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(N)
  x = tl.load(x_ptr + offsets, mask=mask)
  output = x + $(value)
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

## Public theorem: `puzzle1_kernel_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` headline**: for every rounding model `R`, `puzzle1_kernel`
implements lane-wise add-a-scalar `xs i + value` on its masked IO signature
at the `.real` output grid. Same full masked Hoare triple as
`puzzle1_kernel_correctness` — ∀ disjoint flat placement, ∀ program id whose
active lanes are in bounds, ∀ launch state whose input window holds `xs` at
the active lanes — but the run is `execR R` and every active output lane is
read back as an `.real`-typed cell holding `R.round .real (xs i + value)`.

The store is untyped (`tl.store(output_ptr + offsets, output, mask=mask)` —
no `.to(...)`), so the honest grid is `.real` and the boundary round
degenerates (`R.round .real = id`): this kernel's rounding face carries the
exact value contract for every `R`, with the *modeling* claim that the
kernel introduces no rounding event of its own. No hypotheses: the block
geometry is universally quantified, `BLOCK_SIZE = 0` is the vacuous
zero-lane launch, and the window is injective outright. -/
```
</details>

**Statement:**
```lean
specification puzzle1_kernel_io_correctness (R : RoundingModel)
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) :
    addValueIO x_ptr output_ptr N BLOCK_SIZE value
      ⊨[R, .real] fun xs i => xs i + value
```

**Closed-form spec defs (transitive):** `addValueIO`, `puzzle1_kernel`

<details><summary><code>addValueIO</code></summary>

```
/-- `puzzle1_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the headline: which buffer is which argument (the wiring),
where program `pid` reads its input tile / writes its output tile, and the
active-lane predicate `pid * BLOCK_SIZE + j < N`. The windows and mask are
declared, not parsed from the kernel: they formalize the host-side launch
convention (`offsets = pid * BLOCK_SIZE + arange; mask = offsets < N`), and
the headline **proves** the kernel's actual addressing and masking match
them. Buffer sizes are not signature content: the headline quantifies over
every allocation whose extents cover the active lanes. -/
```
```lean
def addValueIO (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) : MaskedKernelIO₁ where
  kernel := puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value
  inp := x_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < N
```
</details>

<details><summary><code>puzzle1_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `add_value.py`'s `puzzle1_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter.
- `value` (Lean `ℝ` parameter) injected via `$(...)`. -/
```
```lean
def puzzle1_kernel
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(N)
  x = tl.load(x_ptr + offsets, mask=mask)
  output = x + $(value)
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>
