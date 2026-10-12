# Spec sheet — `bench/tritonbench_g/vector_addition/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/vector_addition/vector_addition.py`

## Public theorem: `add_commuted_equiv`

<details><summary>docstring</summary>

```
/-- The admitted add_commute atom preserves complete state and execution failure. -/
```
</details>

**Statement:**
```lean
specification add_commuted_equiv (x_ptr y_ptr output_ptr : RegionName) (n B : Nat) (R : Rules) :
    original x_ptr y_ptr output_ptr n B ≡[R] commuted x_ptr y_ptr output_ptr n B
```

**Closed-form spec defs (transitive):** `Rules`, `original`, `commuted`, `add_kernel`, `additionSite`, `add_commuted_kernel`

<details><summary><code>Rules</code></summary>

```lean
abbrev Rules := FP.ScalarArithmetic.Rules
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (x_ptr y_ptr output_ptr : RegionName) (n B : Nat) : Program where
  kernel := add_kernel x_ptr y_ptr output_ptr n B
  domain := [additionSite x_ptr y_ptr output_ptr n B]
```
</details>

<details><summary><code>commuted</code></summary>

```lean
def commuted (x_ptr y_ptr output_ptr : RegionName) (n B : Nat) : Program :=
  { original x_ptr y_ptr output_ptr n B with kernel := add_commuted_kernel x_ptr y_ptr output_ptr n B }
```
</details>

<details><summary><code>add_kernel</code></summary>

```
/-- Faithful transcription of `vector_addition.py`'s `add_kernel`. -/
```
```lean
def add_kernel
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>

<details><summary><code>additionSite</code></summary>

```
/-- Both loaded operands must be finite at the actual addition site, including
masked lanes whose arithmetic is evaluated. No experimental dimensions or
input distribution are restrictions on this symbolic derivation. -/
```
```lean
def additionSite (x_ptr y_ptr output_ptr : RegionName) (n B : Nat) : Site :=
  .add (Source.prefixBefore (add_kernel x_ptr y_ptr output_ptr n B).surfaceBody "output") [B]
    (.ref .real [B] "x") (.ref .real [B] "y")
```
</details>

<details><summary><code>add_commuted_kernel</code></summary>

```
/-- Candidate with the same loads, mask and store, adding y to x. -/
```
```lean
def add_commuted_kernel
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = y + x
  tl.store(output_ptr + offsets, output, mask=mask)
}
```
</details>
