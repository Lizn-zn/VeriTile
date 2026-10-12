# Spec sheet — `bench/tritonbench_g/triton_mul2/Correct.lean`

**Python source:** `bench/tritonbench_g/triton_mul2/triton_mul2.py`

## Public theorem: `mul2_kernel_correctness`

**Statement:**
```lean
specification mul2_kernel_correctness
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    Spec.Real (mul2IO in_ptr0 out_ptr n_elements BLOCK_SIZE
      ⊨ fun xs i => 2 * xs i)
```

**Closed-form spec defs (transitive):** `mul2IO`, `mul2_kernel`

<details><summary><code>mul2IO</code></summary>

```
/-- Symbolic windows and tail mask for ordinary output. -/
```
```lean
def mul2IO (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₁ where
  kernel := mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE
  inp := in_ptr0
  out := out_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements
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

## Public theorem: `mul2_inplace_kernel_correctness`

**Statement:**
```lean
specification mul2_inplace_kernel_correctness
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    Spec.Real (mul2InplaceIO ptr n_elements BLOCK_SIZE
      ⊨ fun xs i => 2 * xs i)
```

**Closed-form spec defs (transitive):** `mul2InplaceIO`, `mul2_inplace_kernel`

<details><summary><code>mul2InplaceIO</code></summary>

```
/-- Both roles share the same region for in-place output. -/
```
```lean
def mul2InplaceIO (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₁ where
  kernel := mul2_inplace_kernel ptr n_elements BLOCK_SIZE
  inp := ptr
  out := ptr
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements
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

## Public theorem: `mul2_commuted_correctness`

**Statement:**
```lean
specification mul2_commuted_correctness (in_ptr0 out_ptr : RegionName) (n B : Nat) :
    Spec.Real (commutedIO in_ptr0 out_ptr n B ⊨ fun xs i => 2 * xs i)
```

**Closed-form spec defs (transitive):** `commutedIO`, `mul2IO`, `mul2_commuted_kernel`, `mul2_kernel`

<details><summary><code>commutedIO</code></summary>

```
/-- The ordinary candidate retains the original masked contract. -/
```
```lean
def commutedIO (in_ptr0 out_ptr : RegionName) (n B : Nat) : MaskedKernelIO₁ :=
  { mul2IO in_ptr0 out_ptr n B with
    kernel := mul2_commuted_kernel in_ptr0 out_ptr n B
    projection := by rfl }
```
</details>

<details><summary><code>mul2IO</code></summary>

```
/-- Symbolic windows and tail mask for ordinary output. -/
```
```lean
def mul2IO (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₁ where
  kernel := mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE
  inp := in_ptr0
  out := out_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements
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

## Public theorem: `mul2_inplace_commuted_correctness`

**Statement:**
```lean
specification mul2_inplace_commuted_correctness (ptr : RegionName) (n B : Nat) :
    Spec.Real (inplaceCommutedIO ptr n B ⊨ fun xs i => 2 * xs i)
```

**Closed-form spec defs (transitive):** `inplaceCommutedIO`, `mul2InplaceIO`, `mul2_inplace_commuted_kernel`, `mul2_inplace_kernel`

<details><summary><code>inplaceCommutedIO</code></summary>

```
/-- The in-place candidate keeps inp = out = ptr, with the same tail mask. -/
```
```lean
def inplaceCommutedIO (ptr : RegionName) (n B : Nat) : MaskedKernelIO₁ :=
  { mul2InplaceIO ptr n B with
    kernel := mul2_inplace_commuted_kernel ptr n B
    projection := by rfl }
```
</details>

<details><summary><code>mul2InplaceIO</code></summary>

```
/-- Both roles share the same region for in-place output. -/
```
```lean
def mul2InplaceIO (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₁ where
  kernel := mul2_inplace_kernel ptr n_elements BLOCK_SIZE
  inp := ptr
  out := ptr
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements
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
