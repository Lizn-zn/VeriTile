# Spec sheet — `bench/tritonbench_g/mul_exponent_compensator/MulExponentCompensator.lean`

**Python source:** `bench/tritonbench_g/mul_exponent_compensator/mul_exponent_compensator.py`

## Public theorem: `mul_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `mul_kernel` implements the elementwise constant
scaling `xs i * exponentCompensator` on its IO signature. Spelled out (see
`KernelIO₁.Implements`): for every disjoint flat placement of the two
buffers, every program id whose windows are in bounds, and every launch
state whose input window holds `xs` — everything else arbitrary — the
translated pointer kernel terminates, its output window holds
`xs i * exponentCompensator`, and every other memory cell is unchanged.
Proof: `Implements.intro` assembles the region-model triple with the bridge
side conditions. -/
```
</details>

**Statement:**
```lean
specification mul_kernel_correctness (src dst : RegionName) (B : Nat)
    (hB : 0 < B) :
    mulIO src dst B ⊨ fun xs i => xs i * exponentCompensator
```

**Assumptions / layout contracts:**
- `hB : 0 < B`

**Closed-form spec defs (transitive):** `mulIO`, `exponentCompensator`, `mul_kernel`

<details><summary><code>mulIO</code></summary>

```
/-- `mul_kernel`'s **IO signature** — the whole kernel-specific audit
surface of the headline: `src` in, `dst` out; tile lengths
`Bin = Bout = B`; program `pid` reads its tile at `pid * B` of `src` and
writes the same window of `dst`. The windows are declared, not parsed from
the kernel: they formalize the host-side launch convention
(`idxs = pid * BLOCK_SIZE + arange`), and the headline **proves** the
kernel's actual addressing matches them. Buffer sizes are not signature
content: the headline quantifies over every allocation large enough for the
windows. -/
```
```lean
noncomputable def mulIO (src dst : RegionName) (B : Nat) : KernelIO₁ where
  kernel := mul_kernel src dst B
  inp := src
  out := dst
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B
```
</details>

<details><summary><code>exponentCompensator</code></summary>

```
/-- The constexpr multiplier from `mul_exponent_compensator.py`. -/
```
```lean
noncomputable def exponentCompensator : ℝ :=
  (2 : ℝ) ^ (127 - 15)
```
</details>

<details><summary><code>mul_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `mul_exponent_compensator.py`'s
`mul_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python local constexpr literal `2.0 ** (127 - 15)` is represented by the
  Lean constant `exponentCompensator` and injected as a real scalar antiquote.
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
noncomputable def mul_kernel
    (src dst : RegionName)
    (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  exponent_compensator = $((exponentCompensator : ℝ))
  idxs = tl.program_id(0) * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(src + idxs)
  y = x * exponent_compensator
  tl.store(dst + idxs, y)
}
```
</details>
