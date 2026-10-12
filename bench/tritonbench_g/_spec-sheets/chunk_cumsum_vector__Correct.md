# Spec sheet — `bench/tritonbench_g/chunk_cumsum_vector/Correct.lean`

**Python source:** `bench/tritonbench_g/chunk_cumsum_vector/chunk_cumsum_vector.py`

## Public theorem: `chunk_cumsum_vector_correctness`

**Statement:**
```lean
specification chunk_cumsum_vector_correctness (p : Layout) (s o : RegionName)
    (dtype out_dtype : FloatDType) (hSO : s ≠ o) (hBT : 0 < p.BT)
    (hLayout : noOutputOverlap p) :
    Spec.Real (originalIO p s o dtype out_dtype ⊨ expected p)
```

**Assumptions / layout contracts:**
- `hSO : s ≠ o`
- `hBT : 0 < p.BT`
- `hLayout : noOutputOverlap p`

**Closed-form spec defs (transitive):** `Layout`, `noOutputOverlap`, `originalIO`, `expected`, `original`

<details><summary><code>Layout</code></summary>

```
/-- Symbolic head/time/channel strides, logical T/S sizes and BT/BS block sizes. -/
```
```lean
abbrev Layout := RealProgram.Layout
```
</details>

<details><summary><code>noOutputOverlap</code></summary>

```
/-- Distinct active output elements have distinct addresses over the whole time
span, including different chunks. Masked feature lanes need not be injective. -/
```
```lean
def noOutputOverlap (p : Layout) : Prop :=
  ∀ (feature head t u : Nat) (i j : Fin p.BS),
    (t < p.T ∧ feature * p.BS + i.val < p.S) →
    (u < p.T ∧ feature * p.BS + j.val < p.S) →
    head * p.headStride + t * p.timeStride + (feature * p.BS + i.val) * p.channelStride =
      head * p.headStride + u * p.timeStride + (feature * p.BS + j.val) * p.channelStride →
    t = u ∧ i = j
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- The complete logical input/output window of one feature-block/head program.
The same strides and masks are used by the original source's block pointers. -/
```
```lean
def originalIO (p : Layout) (s o : RegionName) (dtype out_dtype : FloatDType) :
    Masked3DTileKernelIO₁ where
  kernel := (TypedSource.original dtype out_dtype s o p.headStride p.timeStride p.channelStride
    p.T p.S p.BT p.BS).eraseDType
  projection := (RealIO.io Bool.false p s o dtype out_dtype).projection
  inp := s
  out := o
  shape := [p.T, p.BS]
  read := fun feature head _ i =>
    head * p.headStride + i.1.val * p.timeStride + (feature * p.BS + i.2.1.val) * p.channelStride
  write := fun feature head _ i =>
    head * p.headStride + i.1.val * p.timeStride + (feature * p.BS + i.2.1.val) * p.channelStride
  mask := fun feature _ _ i => feature * p.BS + i.2.1.val < p.S
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Initial input values up to the output time index, independently per column. -/
```
```lean
def expected (p : Layout) (_feature _head : Nat) (xs : TileIndex [p.T,p.BS] → ℝ)
    (i : TileIndex [p.T,p.BS]) : ℝ :=
  ∑ k : Fin p.T, if k.val ≤ i.1.val then xs (k,i.2.1,PUnit.unit) else 0
```
</details>

<details><summary><code>original</code></summary>

```lean
def original := vectorCumsumSource% original
```
</details>

## Public theorem: `chunk_sum_left_correctness`

**Statement:**
```lean
specification chunk_sum_left_correctness (p : Layout) (s o : RegionName)
    (dtype out_dtype : FloatDType) (hSO : s ≠ o) (hBT : 0 < p.BT)
    (hLayout : noOutputOverlap p) :
    Spec.Real (sumLeftIO p s o dtype out_dtype ⊨ expected p)
```

**Assumptions / layout contracts:**
- `hSO : s ≠ o`
- `hBT : 0 < p.BT`
- `hLayout : noOutputOverlap p`

**Closed-form spec defs (transitive):** `Layout`, `noOutputOverlap`, `sumLeftIO`, `expected`, `originalIO`, `sumLeft`, `original`

<details><summary><code>Layout</code></summary>

```
/-- Symbolic head/time/channel strides, logical T/S sizes and BT/BS block sizes. -/
```
```lean
abbrev Layout := RealProgram.Layout
```
</details>

<details><summary><code>noOutputOverlap</code></summary>

```
/-- Distinct active output elements have distinct addresses over the whole time
span, including different chunks. Masked feature lanes need not be injective. -/
```
```lean
def noOutputOverlap (p : Layout) : Prop :=
  ∀ (feature head t u : Nat) (i j : Fin p.BS),
    (t < p.T ∧ feature * p.BS + i.val < p.S) →
    (u < p.T ∧ feature * p.BS + j.val < p.S) →
    head * p.headStride + t * p.timeStride + (feature * p.BS + i.val) * p.channelStride =
      head * p.headStride + u * p.timeStride + (feature * p.BS + j.val) * p.channelStride →
    t = u ∧ i = j
```
</details>

<details><summary><code>sumLeftIO</code></summary>

```lean
def sumLeftIO (p : Layout) (s o : RegionName) (dtype out_dtype : FloatDType) :
    Masked3DTileKernelIO₁ :=
  { originalIO p s o dtype out_dtype with
    kernel := (TypedSource.sumLeft dtype out_dtype s o p.headStride p.timeStride p.channelStride
      p.T p.S p.BT p.BS).eraseDType
    projection := (RealIO.io Bool.true p s o dtype out_dtype).projection }
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Initial input values up to the output time index, independently per column. -/
```
```lean
def expected (p : Layout) (_feature _head : Nat) (xs : TileIndex [p.T,p.BS] → ℝ)
    (i : TileIndex [p.T,p.BS]) : ℝ :=
  ∑ k : Fin p.T, if k.val ≤ i.1.val then xs (k,i.2.1,PUnit.unit) else 0
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- The complete logical input/output window of one feature-block/head program.
The same strides and masks are used by the original source's block pointers. -/
```
```lean
def originalIO (p : Layout) (s o : RegionName) (dtype out_dtype : FloatDType) :
    Masked3DTileKernelIO₁ where
  kernel := (TypedSource.original dtype out_dtype s o p.headStride p.timeStride p.channelStride
    p.T p.S p.BT p.BS).eraseDType
  projection := (RealIO.io Bool.false p s o dtype out_dtype).projection
  inp := s
  out := o
  shape := [p.T, p.BS]
  read := fun feature head _ i =>
    head * p.headStride + i.1.val * p.timeStride + (feature * p.BS + i.2.1.val) * p.channelStride
  write := fun feature head _ i =>
    head * p.headStride + i.1.val * p.timeStride + (feature * p.BS + i.2.1.val) * p.channelStride
  mask := fun feature _ _ i => feature * p.BS + i.2.1.val < p.S
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft := vectorCumsumSource% sumLeft
```
</details>

<details><summary><code>original</code></summary>

```lean
def original := vectorCumsumSource% original
```
</details>
