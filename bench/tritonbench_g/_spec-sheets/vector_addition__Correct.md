# Spec sheet — `bench/tritonbench_g/vector_addition/Correct.lean`

**Python source:** `bench/tritonbench_g/vector_addition/vector_addition.py`

## Public theorem: `add_kernel_correctness`

<details><summary>docstring</summary>

```
/-- Original source implements mathematical pointwise addition. -/
```
</details>

**Statement:**
```lean
specification add_kernel_correctness
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    Spec.Real (addIO x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
      ⊨ fun xs ys i => xs i + ys i)
```

**Closed-form spec defs (transitive):** `addIO`, `add_kernel`

<details><summary><code>addIO</code></summary>

```
/-- Symbolic windows and mask, verified against the actual source accesses. -/
```
```lean
def addIO (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₂ where
  kernel := add_kernel x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
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

## Public theorem: `add_commuted_correctness`

<details><summary>docstring</summary>

```
/-- The candidate implements the same independent formula. -/
```
</details>

**Statement:**
```lean
specification add_commuted_correctness (x_ptr y_ptr output_ptr : RegionName) (n B : Nat) :
    Spec.Real (commutedIO x_ptr y_ptr output_ptr n B ⊨ fun xs ys i => xs i + ys i)
```

**Closed-form spec defs (transitive):** `commutedIO`, `addIO`, `add_commuted_kernel`, `add_kernel`

<details><summary><code>commutedIO</code></summary>

```
/-- Same masked IO for the commuted candidate. -/
```
```lean
def commutedIO (x_ptr y_ptr output_ptr : RegionName) (n B : Nat) : MaskedKernelIO₂ :=
  { addIO x_ptr y_ptr output_ptr n B with
    kernel := add_commuted_kernel x_ptr y_ptr output_ptr n B
    projection := by rfl }
```
</details>

<details><summary><code>addIO</code></summary>

```
/-- Symbolic windows and mask, verified against the actual source accesses. -/
```
```lean
def addIO (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₂ where
  kernel := add_kernel x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
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
