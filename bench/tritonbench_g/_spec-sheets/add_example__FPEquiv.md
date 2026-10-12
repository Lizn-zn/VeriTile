# Spec sheet — `bench/tritonbench_g/add_example/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/add_example/add_example.py`

## Public theorem: `add_commuted_equiv`

<details><summary>docstring</summary>

```
/-- The admitted add_commute atom preserves complete state and execution failure. -/
```
</details>

**Statement:**
```lean
specification add_commuted_equiv (in_ptr0 in_ptr1 out_ptr : RegionName) (n B : Nat) (R : Rules) :
    original in_ptr0 in_ptr1 out_ptr n B ≡[R] commuted in_ptr0 in_ptr1 out_ptr n B
```

**Closed-form spec defs (transitive):** `Rules`, `original`, `commuted`, `add_kernel`, `additionSite`, `add_commuted_kernel`

<details><summary><code>Rules</code></summary>

```lean
abbrev Rules := FP.ScalarArithmetic.Rules
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (in_ptr0 in_ptr1 out_ptr : RegionName) (n B : Nat) : Program where
  kernel := add_kernel in_ptr0 in_ptr1 out_ptr n B
  domain := [additionSite in_ptr0 in_ptr1 out_ptr n B]
```
</details>

<details><summary><code>commuted</code></summary>

```lean
def commuted (in_ptr0 in_ptr1 out_ptr : RegionName) (n B : Nat) : Program :=
  { original in_ptr0 in_ptr1 out_ptr n B with kernel := add_commuted_kernel in_ptr0 in_ptr1 out_ptr n B }
```
</details>

<details><summary><code>add_kernel</code></summary>

```
/-- Faithful transcription of `add_example.py`'s `add_kernel`. -/
```
```lean
def add_kernel
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  y = tl.load(in_ptr1 + offsets, mask=mask)
  output = x + y
  tl.store(out_ptr + offsets, output, mask=mask)
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
def additionSite (in_ptr0 in_ptr1 out_ptr : RegionName) (n B : Nat) : Site :=
  .add (Source.prefixBefore (add_kernel in_ptr0 in_ptr1 out_ptr n B).surfaceBody "output") [B]
    (.ref .real [B] "x") (.ref .real [B] "y")
```
</details>

<details><summary><code>add_commuted_kernel</code></summary>

```
/-- Candidate with the same loads, mask and store, adding y to x. -/
```
```lean
def add_commuted_kernel
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  y = tl.load(in_ptr1 + offsets, mask=mask)
  output = y + x
  tl.store(out_ptr + offsets, output, mask=mask)
}
```
</details>
