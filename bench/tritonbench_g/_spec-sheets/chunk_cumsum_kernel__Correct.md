# Spec sheet — `bench/tritonbench_g/chunk_cumsum_kernel/Correct.lean`

**Python source:** `bench/tritonbench_g/chunk_cumsum_kernel/chunk_cumsum_kernel.py`

## Public theorem: `chunk_cumsum_correctness`

**Statement:**
```lean
specification chunk_cumsum_correctness (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) (hSO : o ≠ s) (hBT : 0 < BT) :
    Spec.Real (originalIO dtype out_dtype s o T BT ⊨ expected T BT)
```

**Assumptions / layout contracts:**
- `hSO : o ≠ s`
- `hBT : 0 < BT`

**Closed-form spec defs (transitive):** `originalIO`, `expected`, `original`

<details><summary><code>originalIO</code></summary>

```
/-- Each program scans one row, loading and emitting BT-lane windows. Both
read and write masks retain the source boundary check. The real IO contract
quantifies over disjoint allocations with sufficient space for active accesses. -/
```
```lean
def originalIO (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) :
    StreamEmitMasked2DKernelIO₁ where
  kernel := (TypedSource.original dtype out_dtype s o T BT).eraseDType
  projection := (RealIO.io Bool.false dtype out_dtype s o T BT).projection
  inp1 := s
  out := o
  T := ccNumChunks T BT
  B1 := BT
  C := BT
  read1 := fun pid _ t j => pid * T + (t.val * BT + j.val)
  write := fun pid _ t j => pid * T + (t.val * BT + j.val)
  mask1 := fun _ _ t j => t.val * BT + j.val < T
  writeMask := fun _ _ t j => t.val * BT + j.val < T
```
</details>

<details><summary><code>expected</code></summary>

```
/-- The sum of the independent initial input cells up to the output's flat
row index. Masked tail cells never enter the formula. -/
```
```lean
def expected (T BT : Nat) (_pid₀ _pid₁ : Nat)
    (xs : Fin (ccNumChunks T BT) → Fin BT → ℝ)
    (t : Fin (ccNumChunks T BT)) (j : Fin BT) : ℝ :=
  ∑ u : Fin (ccNumChunks T BT), ∑ e : Fin BT,
    if u.val * BT + e.val ≤ t.val * BT + j.val ∧ u.val * BT + e.val < T
    then xs u e else 0
```
</details>

<details><summary><code>original</code></summary>

```lean
def original := chunkCumsumSource% original
```
</details>

## Public theorem: `chunk_sum_left_correctness`

**Statement:**
```lean
specification chunk_sum_left_correctness (dtype out_dtype : FloatDType)
    (s o : RegionName) (T BT : Nat) (hSO : o ≠ s) (hBT : 0 < BT) :
    Spec.Real (sumLeftIO dtype out_dtype s o T BT ⊨ expected T BT)
```

**Assumptions / layout contracts:**
- `hSO : o ≠ s`
- `hBT : 0 < BT`

**Closed-form spec defs (transitive):** `sumLeftIO`, `expected`, `originalIO`, `sumLeft`, `original`

<details><summary><code>sumLeftIO</code></summary>

```
/-- The candidate changes only the order of each carry addition. -/
```
```lean
def sumLeftIO (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) :
    StreamEmitMasked2DKernelIO₁ :=
  { originalIO dtype out_dtype s o T BT with
    kernel := (TypedSource.sumLeft dtype out_dtype s o T BT).eraseDType
    projection := (RealIO.io Bool.true dtype out_dtype s o T BT).projection }
```
</details>

<details><summary><code>expected</code></summary>

```
/-- The sum of the independent initial input cells up to the output's flat
row index. Masked tail cells never enter the formula. -/
```
```lean
def expected (T BT : Nat) (_pid₀ _pid₁ : Nat)
    (xs : Fin (ccNumChunks T BT) → Fin BT → ℝ)
    (t : Fin (ccNumChunks T BT)) (j : Fin BT) : ℝ :=
  ∑ u : Fin (ccNumChunks T BT), ∑ e : Fin BT,
    if u.val * BT + e.val ≤ t.val * BT + j.val ∧ u.val * BT + e.val < T
    then xs u e else 0
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- Each program scans one row, loading and emitting BT-lane windows. Both
read and write masks retain the source boundary check. The real IO contract
quantifies over disjoint allocations with sufficient space for active accesses. -/
```
```lean
def originalIO (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) :
    StreamEmitMasked2DKernelIO₁ where
  kernel := (TypedSource.original dtype out_dtype s o T BT).eraseDType
  projection := (RealIO.io Bool.false dtype out_dtype s o T BT).projection
  inp1 := s
  out := o
  T := ccNumChunks T BT
  B1 := BT
  C := BT
  read1 := fun pid _ t j => pid * T + (t.val * BT + j.val)
  write := fun pid _ t j => pid * T + (t.val * BT + j.val)
  mask1 := fun _ _ t j => t.val * BT + j.val < T
  writeMask := fun _ _ t j => t.val * BT + j.val < T
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft := chunkCumsumSource% sumLeft
```
</details>

<details><summary><code>original</code></summary>

```lean
def original := chunkCumsumSource% original
```
</details>
