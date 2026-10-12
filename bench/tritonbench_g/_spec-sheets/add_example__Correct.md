# Spec sheet — `bench/tritonbench_g/add_example/Correct.lean`

**Python source:** `bench/tritonbench_g/add_example/add_example.py`

## Public theorem: `add_kernel_correctness`

<details><summary>docstring</summary>

```
/-- Original source implements mathematical pointwise addition. -/
```
</details>

**Statement:**
```lean
specification add_kernel_correctness
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    Spec.Real (addIO in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
      ⊨ fun xs ys i => xs i + ys i)
```

**Closed-form spec defs (transitive):** `addIO`, `add_kernel`

<details><summary><code>addIO</code></summary>

```
/-- Symbolic windows and mask, verified against the actual source accesses. -/
```
```lean
def addIO (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₂ where
  kernel := add_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
  in1 := in_ptr0
  in2 := in_ptr1
  out := out_ptr
  B := BLOCK_SIZE
  read1 := fun pid => pid * BLOCK_SIZE
  read2 := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements
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

## Public theorem: `add_commuted_correctness`

<details><summary>docstring</summary>

```
/-- The candidate implements the same independent formula. -/
```
</details>

**Statement:**
```lean
specification add_commuted_correctness (in_ptr0 in_ptr1 out_ptr : RegionName) (n B : Nat) :
    Spec.Real (commutedIO in_ptr0 in_ptr1 out_ptr n B ⊨ fun xs ys i => xs i + ys i)
```

**Closed-form spec defs (transitive):** `commutedIO`, `addIO`, `add_commuted_kernel`, `add_kernel`

<details><summary><code>commutedIO</code></summary>

```
/-- Same masked IO for the commuted candidate. -/
```
```lean
def commutedIO (in_ptr0 in_ptr1 out_ptr : RegionName) (n B : Nat) : MaskedKernelIO₂ :=
  { addIO in_ptr0 in_ptr1 out_ptr n B with
    kernel := add_commuted_kernel in_ptr0 in_ptr1 out_ptr n B
    projection := by rfl }
```
</details>

<details><summary><code>addIO</code></summary>

```
/-- Symbolic windows and mask, verified against the actual source accesses. -/
```
```lean
def addIO (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₂ where
  kernel := add_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
  in1 := in_ptr0
  in2 := in_ptr1
  out := out_ptr
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
