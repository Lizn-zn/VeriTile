# Spec sheet — `bench/tritonbench_g/vector_addition_custom/Correct.lean`

**Python source:** `bench/tritonbench_g/vector_addition_custom/vector_addition_custom.py`

## Public theorem: `add_kernel_correctness`

**Statement:**
```lean
specification add_kernel_correctness
    (A B C : RegionName)
    (size BLOCK : Nat) :
    Spec.Real (addCustomIO A B C size BLOCK
      ⊨ fun as bs i => as i + bs i)
```

**Closed-form spec defs (transitive):** `addCustomIO`, `_add_kernel`

<details><summary><code>addCustomIO</code></summary>

```
/-- Symbolic windows and the actual active-lane mask. -/
```
```lean
def addCustomIO (A B C : RegionName)
    (size BLOCK : Nat) : MaskedKernelIO₂ where
  kernel := _add_kernel A B C size BLOCK
  in1 := A
  in2 := B
  out := C
  B := BLOCK
  read1 := fun pid => pid * BLOCK
  read2 := fun pid => pid * BLOCK
  write := fun pid => pid * BLOCK
  mask := fun pid j => pid * BLOCK + j.val < size
```
</details>

<details><summary><code>_add_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `vector_addition_custom.py`'s `_add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
def _add_kernel
    (A B C : RegionName)
    (size BLOCK : Nat) :
    ComputeKernel := triton {
  prog_id = tl.program_id(0)
  offs = prog_id * $(BLOCK) + tl.arange(0, $(BLOCK))
  a = tl.load(A + offs, mask=offs < $(size))
  b = tl.load(B + offs, mask=offs < $(size))
  tl.store(C + offs, a + b, mask=offs < $(size))
}
```
</details>

## Public theorem: `add_commuted_correctness`

**Statement:**
```lean
specification add_commuted_correctness (A B C : RegionName) (size BLOCK : Nat) :
    Spec.Real (commutedIO A B C size BLOCK ⊨ fun as bs i => as i + bs i)
```

**Closed-form spec defs (transitive):** `commutedIO`, `addCustomIO`, `_add_commuted_kernel`, `_add_kernel`

<details><summary><code>commutedIO</code></summary>

```
/-- The candidate keeps the same windows and mask. -/
```
```lean
def commutedIO (A B C : RegionName) (size BLOCK : Nat) : MaskedKernelIO₂ :=
  { addCustomIO A B C size BLOCK with
    kernel := _add_commuted_kernel A B C size BLOCK
    projection := by rfl }
```
</details>

<details><summary><code>addCustomIO</code></summary>

```
/-- Symbolic windows and the actual active-lane mask. -/
```
```lean
def addCustomIO (A B C : RegionName)
    (size BLOCK : Nat) : MaskedKernelIO₂ where
  kernel := _add_kernel A B C size BLOCK
  in1 := A
  in2 := B
  out := C
  B := BLOCK
  read1 := fun pid => pid * BLOCK
  read2 := fun pid => pid * BLOCK
  write := fun pid => pid * BLOCK
  mask := fun pid j => pid * BLOCK + j.val < size
```
</details>

<details><summary><code>_add_commuted_kernel</code></summary>

```
/-- Commute only the value expression of the original masked store. -/
```
```lean
def _add_commuted_kernel
    (A B C : RegionName)
    (size BLOCK : Nat) :
    ComputeKernel := triton {
  prog_id = tl.program_id(0)
  offs = prog_id * $(BLOCK) + tl.arange(0, $(BLOCK))
  a = tl.load(A + offs, mask=offs < $(size))
  b = tl.load(B + offs, mask=offs < $(size))
  tl.store(C + offs, b + a, mask=offs < $(size))
}
```
</details>

<details><summary><code>_add_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `vector_addition_custom.py`'s `_add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
def _add_kernel
    (A B C : RegionName)
    (size BLOCK : Nat) :
    ComputeKernel := triton {
  prog_id = tl.program_id(0)
  offs = prog_id * $(BLOCK) + tl.arange(0, $(BLOCK))
  a = tl.load(A + offs, mask=offs < $(size))
  b = tl.load(B + offs, mask=offs < $(size))
  tl.store(C + offs, a + b, mask=offs < $(size))
}
```
</details>
