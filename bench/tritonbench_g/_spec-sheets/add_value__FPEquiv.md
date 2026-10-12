# Spec sheet — `bench/tritonbench_g/add_value/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/add_value/add_value.py`

## Public theorem: `add_value_commuted_equiv`

<details><summary>docstring</summary>

```
/-- Finite operands and admitted add_commute preserve all registers, memory and
execution failure for arbitrary symbolic sizes and scalar value. -/
```
</details>

**Statement:**
```lean
specification add_value_commuted_equiv (x_ptr output_ptr : RegionName)
    (N B : Nat) (value : ℝ) (R : Rules) :
    original x_ptr output_ptr N B value ≡[R] commuted x_ptr output_ptr N B value
```

**Closed-form spec defs (transitive):** `Rules`, `original`, `commuted`, `puzzle1_kernel`, `additionSite`, `puzzle1_commuted_kernel`

<details><summary><code>Rules</code></summary>

```lean
abbrev Rules := FP.ScalarArithmetic.Rules
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (x_ptr output_ptr : RegionName) (N B : Nat) (value : ℝ) : Program where
  kernel := puzzle1_kernel x_ptr output_ptr N B value
  domain := [additionSite x_ptr output_ptr N B value]
```
</details>

<details><summary><code>commuted</code></summary>

```lean
def commuted (x_ptr output_ptr : RegionName) (N B : Nat) (value : ℝ) : Program :=
  { original x_ptr output_ptr N B value with
    kernel := puzzle1_commuted_kernel x_ptr output_ptr N B value }
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

<details><summary><code>additionSite</code></summary>

```
/-- The loaded tile and the scalar interpreted at fp32 must be finite.
Masked lanes still execute arithmetic and are included. Neither positivity nor
nonzero is required; negative and zero values remain in the domain. -/
```
```lean
def additionSite (x_ptr output_ptr : RegionName) (N B : Nat) (value : ℝ) : Site :=
  .addScalar (Source.prefixBefore (puzzle1_kernel x_ptr output_ptr N B value).surfaceBody
    "output") B [] (.ref .real [B] "x") (.const value)
```
</details>

<details><summary><code>puzzle1_commuted_kernel</code></summary>

```
/-- Commuted scalar addition with the original memory operations.
The Lean type annotation selects the scalar data channel; it inserts no cast. -/
```
```lean
def puzzle1_commuted_kernel
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(N)
  x = tl.load(x_ptr + offsets, mask=mask)
  output = $((value : Real)) + x
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>
