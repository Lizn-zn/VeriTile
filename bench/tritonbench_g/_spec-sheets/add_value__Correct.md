# Spec sheet — `bench/tritonbench_g/add_value/Correct.lean`

**Python source:** `bench/tritonbench_g/add_value/add_value.py`

## Public theorem: `puzzle1_kernel_correctness`

<details><summary>docstring</summary>

```
/-- Mathematical correctness of the original source. -/
```
</details>

**Statement:**
```lean
specification puzzle1_kernel_correctness
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) :
    Spec.Real (addValueIO x_ptr output_ptr N BLOCK_SIZE value
      ⊨ fun xs i => xs i + value)
```

**Closed-form spec defs (transitive):** `addValueIO`, `puzzle1_kernel`

<details><summary><code>addValueIO</code></summary>

```
/-- Symbolic input/output windows and the actual tail mask. -/
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

## Public theorem: `puzzle1_commuted_correctness`

**Statement:**
```lean
specification puzzle1_commuted_correctness (x_ptr output_ptr : RegionName)
    (N B : Nat) (value : ℝ) :
    Spec.Real (commutedIO x_ptr output_ptr N B value ⊨ fun xs i => xs i + value)
```

**Closed-form spec defs (transitive):** `commutedIO`, `addValueIO`, `puzzle1_commuted_kernel`, `puzzle1_kernel`

<details><summary><code>commutedIO</code></summary>

```
/-- The candidate retains the same windows and mask. -/
```
```lean
def commutedIO (x_ptr output_ptr : RegionName) (N B : Nat) (value : ℝ) : MaskedKernelIO₁ :=
  { addValueIO x_ptr output_ptr N B value with
    kernel := puzzle1_commuted_kernel x_ptr output_ptr N B value
    projection := by rfl }
```
</details>

<details><summary><code>addValueIO</code></summary>

```
/-- Symbolic input/output windows and the actual tail mask. -/
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
