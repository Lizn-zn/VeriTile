# Spec sheet — `bench/tritonbench_g/relu_triton_kernel/ReluTritonKernel.lean`

**Python source:** `bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py`

## Public theorem: `relu_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `relu_kernel` implements lane-wise
`TiledActivation.relu` (`max 0 ·`) on its masked IO signature — the full
masked Hoare triple over flat pointer memory (see the module docstring),
**general over `N`**: the kernel's `pid == 0` store gate is part of the
audited IO signature (`reluIO.writeMask`), so programs with `pid ≠ 0` carry
a pure frame obligation and no value claim. Proof:
`MaskedKernelIO₁.Implements.intro` assembles the region-model masked triple
with the bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification relu_kernel_correctness
    (x_ptr out_ptr : RegionName) (N block_size : Nat)
    (hB : 0 < block_size) :
    reluIO x_ptr out_ptr N block_size ⊨
      fun xs i => TiledActivation.relu (xs i)
```

**Assumptions / layout contracts:**
- `hB : 0 < block_size`

**Closed-form spec defs (transitive):** `reluIO`, `relu_kernel`

<details><summary><code>reluIO</code></summary>

```
/-- `relu_kernel`'s masked **IO signature** — the whole kernel-specific audit
surface of the headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B` — the window length each program owns;
* `read`/`write` — where program `pid` reads its input tile / writes its
  output tile (`pid * block_size`, the launch convention
  `offsets = pid * block_size + arange`);
* `mask` — program `pid`'s **read-active lanes**, `pid * block_size + j < N`:
  the kernel's load mask (every program loads);
* `writeMask` — program `pid`'s **write-active lanes**,
  `pid = 0 ∧ pid * block_size + j < N`: the kernel's `pid == 0`-gated store
  mask. Programs with `pid ≠ 0` have no write-active lanes; the frame asserts
  they write nothing.

Inactive lanes carry no obligations on either side. The windows and masks are
declared, not parsed from the kernel; the headline **proves** the kernel's
actual addressing, masking, and store gate match them. -/
```
```lean
def reluIO (x_ptr out_ptr : RegionName) (N block_size : Nat) :
    MaskedKernelIO₁ where
  kernel := relu_kernel x_ptr out_ptr N block_size
  inp := x_ptr
  out := out_ptr
  B := block_size
  read := fun pid => pid * block_size
  write := fun pid => pid * block_size
  mask := fun pid j => pid * block_size + j.val < N
  writeMask := fun pid j => pid = 0 ∧ pid * block_size + j.val < N
```
</details>

<details><summary><code>relu_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `relu_triton_kernel.py`'s `relu_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `N: tl.constexpr` / `block_size: tl.constexpr` → Lean `Nat`.
- Python `if cond: body` → `if cond { body }`, the DSL-side gate (block syntax
  uses braces because Lean is whitespace-insensitive; the `if` keyword itself
  is shared). -/
```
```lean
def relu_kernel
    (x_ptr out_ptr : RegionName)
    (N block_size : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  block_start = pid * $(block_size)
  offsets = block_start + tl.arange(0, $(block_size))
  mask = offsets < $(N)
  x = tl.load(x_ptr + offsets, mask=mask)
  result = tl.where(x >= 0, x, 0.0)
  if pid == 0 {
    tl.store(out_ptr + offsets, result, mask=mask)
  }
}
```
</details>
