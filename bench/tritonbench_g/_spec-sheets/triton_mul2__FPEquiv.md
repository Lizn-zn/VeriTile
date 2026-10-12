# Spec sheet — `bench/tritonbench_g/triton_mul2/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/triton_mul2/triton_mul2.py`

## Public theorem: `mul2_commuted_equiv`

**Statement:**
```lean
specification mul2_commuted_equiv (in_ptr0 out_ptr : RegionName) (n B : Nat) (R : Rules) :
    original in_ptr0 out_ptr n B ≡[R] commuted in_ptr0 out_ptr n B
```

**Closed-form spec defs (transitive):** `Rules`, `original`, `commuted`, `mul2_kernel`, `multiplicationSite`, `mul2_commuted_kernel`

<details><summary><code>Rules</code></summary>

```lean
abbrev Rules := FP.ScalarArithmetic.Rules
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (in_ptr0 out_ptr : RegionName) (n B : Nat) : Program where
  kernel := mul2_kernel in_ptr0 out_ptr n B
  domain := [multiplicationSite in_ptr0 out_ptr n B]
```
</details>

<details><summary><code>commuted</code></summary>

```lean
def commuted (in_ptr0 out_ptr : RegionName) (n B : Nat) : Program :=
  { original in_ptr0 out_ptr n B with kernel := mul2_commuted_kernel in_ptr0 out_ptr n B }
```
</details>

<details><summary><code>mul2_kernel</code></summary>

```
/-- Faithful transcription of `triton_mul2.py`'s `mul2_kernel`.
The Python constexpr tile size is a Lean Nat parameter. -/
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

<details><summary><code>multiplicationSite</code></summary>

```
/-- Check all actually loaded tile lanes and the fp32 literal 2 at the
multiplication site. Masked lanes still execute arithmetic and are included.
No positivity or nonzero condition is required of the input. -/
```
```lean
def multiplicationSite (in_ptr0 out_ptr : RegionName) (n B : Nat) : Site :=
  .mulScalar (Source.prefixBefore (mul2_kernel in_ptr0 out_ptr n B).surfaceBody
    "output") B [] (.ref .real [B] "x") (.const 2)
```
</details>

<details><summary><code>mul2_commuted_kernel</code></summary>

```
/-- Commuted multiplication with a separate output argument. -/
```
```lean
def mul2_commuted_kernel
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  output = x * 2
  tl.store(out_ptr + offsets, output, mask=mask)
}
```
</details>

## Public theorem: `mul2_inplace_commuted_equiv`

<details><summary>docstring</summary>

```
/-- Includes input/output aliasing: the loaded register is rewritten, then
stored back through the original mask. All states and execution failures agree. -/
```
</details>

**Statement:**
```lean
specification mul2_inplace_commuted_equiv (ptr : RegionName) (n B : Nat) (R : Rules) :
    inplaceOriginal ptr n B ≡[R] inplaceCommuted ptr n B
```

**Closed-form spec defs (transitive):** `Rules`, `inplaceOriginal`, `inplaceCommuted`, `mul2_inplace_kernel`, `inplaceMultiplicationSite`, `mul2_inplace_commuted_kernel`

<details><summary><code>Rules</code></summary>

```lean
abbrev Rules := FP.ScalarArithmetic.Rules
```
</details>

<details><summary><code>inplaceOriginal</code></summary>

```lean
def inplaceOriginal (ptr : RegionName) (n B : Nat) : Program where
  kernel := mul2_inplace_kernel ptr n B
  domain := [inplaceMultiplicationSite ptr n B]
```
</details>

<details><summary><code>inplaceCommuted</code></summary>

```lean
def inplaceCommuted (ptr : RegionName) (n B : Nat) : Program :=
  { inplaceOriginal ptr n B with kernel := mul2_inplace_commuted_kernel ptr n B }
```
</details>

<details><summary><code>mul2_inplace_kernel</code></summary>

```
/-- Faithful transcription of `triton_mul2.py`'s `mul2_inplace_kernel`. -/
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

<details><summary><code>inplaceMultiplicationSite</code></summary>

```
/-- The in-place source checks its own load prefix and writes back to ptr. -/
```
```lean
def inplaceMultiplicationSite (ptr : RegionName) (n B : Nat) : Site :=
  .mulScalar (Source.prefixBefore (mul2_inplace_kernel ptr n B).surfaceBody
    "output") B [] (.ref .real [B] "x") (.const 2)
```
</details>

<details><summary><code>mul2_inplace_commuted_kernel</code></summary>

```
/-- Commuted multiplication writing back to the input buffer. -/
```
```lean
def mul2_inplace_commuted_kernel
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(ptr + offsets, mask=mask)
  output = x * 2
  tl.store(ptr + offsets, output, mask=mask)
}
```
</details>
