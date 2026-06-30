# Spec sheet — `bench/tritonbench_g/nested_loops_processing/NestedLoopsProcessing.lean`

**Python source:** `bench/tritonbench_g/nested_loops_processing/nested_loops_processing.py`

## Public theorem: `nested3_output_summary_general`

<details><summary>docstring</summary>

```
/-! ## Dimension-general output summary

The full kernel body is stride-parametric: the lowering and every per-store
copy-correctness fact above hold over symbolic `(stride_m, stride_n)`. The
summary below bundles, over arbitrary strides, (a) that the full nested-loop
surface lowers and (b) the complete genuine copy-correctness sweep — all 24
`2×2` block stores of the three nested `range(0, 2)` loops, each carrying its own
`2×2`-offset injectivity hypothesis and asserting a verbatim copy from the
shifted INPUT region to the shifted OUTPUT region (`expected` reads input
memory, never `exec(...).readMem`). The Python wrapper's contiguous square
layouts (`stride_m ∈ {8, 4, 16, 2}`, `stride_n = 1`) are instances. -/
```
</details>

**Statement:**
```lean
theorem nested3_output_summary_general
    (in_ptr out_ptr : RegionName) (stride_m stride_n : Nat)
    (s : BlockState)
    (hInj0 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffset stride_m stride_n idx))
    (hInj2 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 2 idx))
    (hInj4 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 4 idx))
    (hInj6 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 6 idx))
    (hInj8 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 8 idx))
    (hInj10 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 10 idx))
    (hInj12 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 12 idx))
    (hInj14 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 14 idx))
    (hInj16 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 16 idx))
    (hInj18 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 18 idx))
    (hInj20 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 20 idx))
    (hInj22 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 22 idx))
    (hInj24 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 24 idx))
    (hInj26 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 26 idx))
    (hInj28 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 28 idx))
    (hInj30 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 30 idx))
    (hInj32 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 32 idx))
    (hInj34 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 34 idx))
    (hInj36 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 36 idx))
    (hInj38 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 38 idx))
    (hInj40 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 40 idx))
    (hInj42 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 42 idx))
    (hInj44 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 44 idx))
    (hInj46 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 46 idx)) :
    -- (a) The full nested-loop surface lowers to the algorithm layer.
    (∃ alg, (nested3 in_ptr out_ptr stride_m stride_n).toAlgorithm? =
      Except.ok alg) ∧
    -- (b1) Initial a1 store: verbatim copy at shift 0.
    ComputeCorrect.Realizes
      (kernel := nested3_first_a1_store in_ptr out_ptr stride_m stride_n)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffset stride_m stride_n idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffset stride_m stride_n idx)) ∧
    -- (b2) First a2 store (i,j,k)=(0,0,0).
    ComputeCorrect.Realizes
      (kernel := nested3_first_a2_store in_ptr out_ptr stride_m stride_n)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 2 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 2 idx)) ∧
    -- (b3) First a3 store (i,j,k)=(0,0,0).
    ComputeCorrect.Realizes
      (kernel := nested3_first_a3_store in_ptr out_ptr stride_m stride_n)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 4 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 4 idx)) ∧
    -- (b4) Second k: a1 reused (in 0 → out 6).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 0 6)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 6 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 0 idx)) ∧
    -- (b5) Second k: a2 (in 2 → out 8).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 2 8)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 8 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 2 idx)) ∧
    -- (b6) Second k: a3 (in 6 → out 10).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 6 10)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 10 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 6 idx)) ∧
    -- (b7) Second j, first k: a1 reused (in 0 → out 12).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 0 12)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 12 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 0 idx)) ∧
    -- (b8) Second j, first k: a2 (in 8 → out 14).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 8 14)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 14 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 8 idx)) ∧
    -- (b9) Second j, first k: a3 (in 10 → out 16).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 10 16)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 16 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 10 idx)) ∧
    -- (b10) Second j, second k: a1 reused (in 0 → out 18).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 0 18)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 18 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 0 idx)) ∧
    -- (b11) Second j, second k: a2 reused (in 8 → out 20).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 8 20)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 20 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 8 idx)) ∧
    -- (b12) Second j, second k: a3 (in 12 → out 22).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 12 22)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 22 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 12 idx)) ∧
    -- (b13) Second i, first j, first k: a1 (in 14 → out 24).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 14 24)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 24 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 14 idx)) ∧
    -- (b14) Second i, first j, first k: a2 (in 16 → out 26).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 16 26)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 26 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 16 idx)) ∧
    -- (b15) Second i, first j, first k: a3 (in 18 → out 28).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 18 28)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 28 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 18 idx)) ∧
    -- (b16) Second i, first j, second k: a1 reused (in 14 → out 30).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 14 30)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 30 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 14 idx)) ∧
    -- (b17) Second i, first j, second k: a2 reused (in 16 → out 32).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 16 32)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 32 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 16 idx)) ∧
    -- (b18) Second i, first j, second k: a3 (in 20 → out 34).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 20 34)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 34 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 20 idx)) ∧
    -- (b19) Second i, second j, first k: a1 reused (in 14 → out 36).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 14 36)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 36 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 14 idx)) ∧
    -- (b20) Second i, second j, first k: a2 (in 22 → out 38).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 22 38)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 38 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 22 idx)) ∧
    -- (b21) Second i, second j, first k: a3 (in 24 → out 40).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 24 40)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 40 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 24 idx)) ∧
    -- (b22) Second i, second j, second k: a1 reused (in 14 → out 42).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 14 42)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 42 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 14 idx)) ∧
    -- (b23) Second i, second j, second k: a2 reused (in 22 → out 44).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 22 44)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 44 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 22 idx)) ∧
    -- (b24) Second i, second j, second k: final a3 (in 26 → out 46).
    ComputeCorrect.Realizes
      (kernel := nested3_shifted_copy_store in_ptr out_ptr stride_m stride_n 26 46)
      (initialState := s)
      (write := fun idx : TileIndex [2, 2] =>
        some (out_ptr, matrixOffsetShift stride_m stride_n 46 idx))
      (expected := fun idx =>
        s.readMem in_ptr (matrixOffsetShift stride_m stride_n 26 idx))
```

**Assumptions / layout contracts:**
- `hInj0 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffset stride_m stride_n idx)`
- `hInj2 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 2 idx)`
- `hInj4 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 4 idx)`
- `hInj6 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 6 idx)`
- `hInj8 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 8 idx)`
- `hInj10 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 10 idx)`
- `hInj12 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 12 idx)`
- `hInj14 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 14 idx)`
- `hInj16 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 16 idx)`
- `hInj18 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 18 idx)`
- `hInj20 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 20 idx)`
- `hInj22 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 22 idx)`
- `hInj24 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 24 idx)`
- `hInj26 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 26 idx)`
- `hInj28 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 28 idx)`
- `hInj30 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 30 idx)`
- `hInj32 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 32 idx)`
- `hInj34 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 34 idx)`
- `hInj36 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 36 idx)`
- `hInj38 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 38 idx)`
- `hInj40 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 40 idx)`
- `hInj42 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 42 idx)`
- `hInj44 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 44 idx)`
- `hInj46 : Function.Injective
      (fun idx : TileIndex [2, 2] => matrixOffsetShift stride_m stride_n 46 idx)`

**Closed-form spec defs (transitive):** `matrixOffset`, `matrixOffsetShift`, `nested3`, `nested3_first_a1_store`, `nested3_first_a2_store`, `nested3_first_a3_store`, `nested3_shifted_copy_store`, `rowIndex`, `colIndex`, `nested3_shifted_store`

<details><summary><code>matrixOffset</code></summary>

```lean
def matrixOffset (stride_m stride_n : Nat) (idx : TileIndex [2, 2]) : Nat :=
  rowIndex idx * stride_m + colIndex idx * stride_n
```
</details>

<details><summary><code>matrixOffsetShift</code></summary>

```lean
def matrixOffsetShift (stride_m stride_n SHIFT : Nat)
    (idx : TileIndex [2, 2]) : Nat :=
  matrixOffset stride_m stride_n idx + SHIFT * stride_n
```
</details>

<details><summary><code>nested3</code></summary>

```
/-- Faithful transcription of `nested_loops_processing.py`'s `nested3`.

Allowed mechanical Lean-syntax-only changes:
- Python literal loop bounds `range(0, 2)` are written as
  `range(0, 2, 1)`.
- Scalar constants in pointer increments are antiquoted so they are inferred
  in the Nat offset channel. -/
```
```lean
def nested3 (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat) : ComputeKernel := triton {
  offs_am = tl.arange(0, 2)
  offs_an = tl.arange(0, 2)
  a_ptrs = in_ptr + (offs_am[:, None] * $(stride_m) +
    offs_an[None, :] * $(stride_n))
  offs_cm = tl.arange(0, 2)
  offs_cn = tl.arange(0, 2)
  c_ptrs = out_ptr + $(stride_m) * offs_cm[:, None] +
    $(stride_n) * offs_cn[None, :]
  for i in range(0, $(2), $(1)) {
    a1 = tl.load(a_ptrs)
    for j in range(0, $(2), $(1)) {
      a_ptrs += $(2) * $(stride_n)
      a2 = tl.load(a_ptrs)
      for k in range(0, $(2), $(1)) {
        a_ptrs += $(2) * $(stride_n)
        a3 = tl.load(a_ptrs)
        tl.store(c_ptrs, a1)
        c_ptrs += $(2) * $(stride_n)
        tl.store(c_ptrs, a2)
        c_ptrs += $(2) * $(stride_n)
        tl.store(c_ptrs, a3)
        c_ptrs += $(2) * $(stride_n)
      }
    }
    a_ptrs += $(2) * $(stride_n)
  }
}
```
</details>

<details><summary><code>nested3_first_a1_store</code></summary>

```
/-- Proof-oriented first 2x2 transfer slice of `nested_loops_processing.py`'s
`nested3`.

The full kernel repeatedly advances `a_ptrs` and `c_ptrs` through nested static
loops. This slice captures the initial `a1 = tl.load(a_ptrs)` and first
`tl.store(c_ptrs, a1)` block. -/
```
```lean
def nested3_first_a1_store (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat) : ComputeKernel := triton {
  offs_am = tl.arange(0, 2)
  offs_an = tl.arange(0, 2)
  a = tl.load(in_ptr + offs_am[:, None] * $(stride_m) +
    offs_an[None, :] * $(stride_n))
  tl.store(out_ptr + offs_am[:, None] * $(stride_m) +
    offs_an[None, :] * $(stride_n), a)
}
```
</details>

<details><summary><code>nested3_first_a2_store</code></summary>

```
/-- First `a2` store of the initial `(i,j,k) = (0,0,0)` path. -/
```
```lean
abbrev nested3_first_a2_store (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat) : ComputeKernel :=
  nested3_shifted_store in_ptr out_ptr stride_m stride_n 2
```
</details>

<details><summary><code>nested3_first_a3_store</code></summary>

```
/-- First `a3` store of the initial `(i,j,k) = (0,0,0)` path. -/
```
```lean
abbrev nested3_first_a3_store (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat) : ComputeKernel :=
  nested3_shifted_store in_ptr out_ptr stride_m stride_n 4
```
</details>

<details><summary><code>nested3_shifted_copy_store</code></summary>

```
/-- Generic shifted copy slice for later nested-loop stores where the source
`a_ptrs` shift and destination `c_ptrs` shift are no longer the same. -/
```
```lean
def nested3_shifted_copy_store (in_ptr out_ptr : RegionName)
    (stride_m stride_n IN_SHIFT OUT_SHIFT : Nat) : ComputeKernel := triton {
  offs_am = tl.arange(0, 2)
  offs_an = tl.arange(0, 2)
  a = tl.load(in_ptr + offs_am[:, None] * $(stride_m) +
    offs_an[None, :] * $(stride_n) + $(IN_SHIFT) * $(stride_n))
  tl.store(out_ptr + offs_am[:, None] * $(stride_m) +
    offs_an[None, :] * $(stride_n) + $(OUT_SHIFT) * $(stride_n), a)
}
```
</details>

<details><summary><code>rowIndex</code></summary>

```lean
def rowIndex (idx : TileIndex [2, 2]) : Nat :=
  idx.1.val
```
</details>

<details><summary><code>colIndex</code></summary>

```lean
def colIndex (idx : TileIndex [2, 2]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>nested3_shifted_store</code></summary>

```
/-- Proof-oriented shifted 2x2 transfer slice for the inner nested-loop stores.

`SHIFT = 2` models the first `a_ptrs += 2 * stride_n` / `c_ptrs += 2 * stride_n`
step before the `a2` store in the first inner iteration; `SHIFT = 4` models the
next pointer advance and the corresponding `a3` store. -/
```
```lean
def nested3_shifted_store (in_ptr out_ptr : RegionName)
    (stride_m stride_n SHIFT : Nat) : ComputeKernel := triton {
  offs_am = tl.arange(0, 2)
  offs_an = tl.arange(0, 2)
  a = tl.load(in_ptr + offs_am[:, None] * $(stride_m) +
    offs_an[None, :] * $(stride_n) + $(SHIFT) * $(stride_n))
  tl.store(out_ptr + offs_am[:, None] * $(stride_m) +
    offs_an[None, :] * $(stride_n) + $(SHIFT) * $(stride_n), a)
}
```
</details>

## Also present (pinned special-case summaries)
- `nested3_first_a1_store_compute_correct`
- `nested3_shifted_store_compute_correct`
- `nested3_first_a2_store_compute_correct`
- `nested3_first_a3_store_compute_correct`
- `nested3_shifted_copy_store_compute_correct`
- `nested3_second_k_a1_store_compute_correct`
- `nested3_second_k_a2_store_compute_correct`
- `nested3_second_k_a3_store_compute_correct`
- `nested3_second_j_first_k_a1_store_compute_correct`
- `nested3_second_j_first_k_a2_store_compute_correct`
- `nested3_second_j_first_k_a3_store_compute_correct`
- `nested3_second_j_second_k_a1_store_compute_correct`
- `nested3_second_j_second_k_a2_store_compute_correct`
- `nested3_second_j_second_k_a3_store_compute_correct`
- `nested3_second_i_first_j_first_k_a1_store_compute_correct`
- `nested3_second_i_first_j_first_k_a2_store_compute_correct`
- `nested3_second_i_first_j_first_k_a3_store_compute_correct`
- `nested3_second_i_first_j_second_k_a1_store_compute_correct`
- `nested3_second_i_first_j_second_k_a2_store_compute_correct`
- `nested3_second_i_first_j_second_k_a3_store_compute_correct`
- `nested3_second_i_second_j_first_k_a1_store_compute_correct`
- `nested3_second_i_second_j_first_k_a2_store_compute_correct`
- `nested3_second_i_second_j_first_k_a3_store_compute_correct`
- `nested3_second_i_second_j_second_k_a1_store_compute_correct`
- `nested3_second_i_second_j_second_k_a2_store_compute_correct`
- `nested3_second_i_second_j_second_k_a3_store_compute_correct`
