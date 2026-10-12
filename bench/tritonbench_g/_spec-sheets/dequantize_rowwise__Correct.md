# Spec sheet — `bench/tritonbench_g/dequantize_rowwise/Correct.lean`

**Python source:** `bench/tritonbench_g/dequantize_rowwise/dequantize_rowwise.py`

## Public theorem: `original_correct`

**Statement:**
```lean
specification original_correct (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat)
    (positiveBlock : 0<block) (positiveLanes : 0<lanes) :
    Spec.Real (originalIO x scale out inv n block lanes ⊨ᵢ fun _ _ xs ys j => ys j*(xs j : ℝ)*inv)
```

**Assumptions / layout contracts:**
- `positiveBlock : 0<block`
- `positiveLanes : 0<lanes`

**Closed-form spec defs (transitive):** `originalIO`, `original`

<details><summary><code>originalIO</code></summary>

```
/-- The row uses its lane mask; all active lanes refer to the same unmasked
scale cell. The positive-size conditions below guarantee such a lane exists. -/
```
```lean
def originalIO (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) : Masked2DKernelIO₂ where
  kernel := (original x scale out inv n block lanes).eraseDType
  projection := by
    have h : (original x scale out inv n block lanes).toAlgorithm? =
        .ok (original x scale out inv n block lanes).toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType,h,ComputeKernel.toAlgorithm?_fromKernelBody,
      ComputeKernel.toAlgKernel_fromKernelBody]
  in1 := x
  in2 := scale
  out := out
  B := lanes
  read1 := fun pid _ j => pid*block+j.val
  read2 := fun pid _ _ => pid
  write := fun pid _ j => pid*block+j.val
  mask := fun _ _ j => j.val<block
```
</details>

<details><summary><code>original</code></summary>

```lean
def original := dequantizeRowwiseSource% original
```
</details>

## Public theorem: `index_commuted_correct`

**Statement:**
```lean
specification index_commuted_correct (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat)
    (positiveBlock : 0<block) (positiveLanes : 0<lanes) :
    Spec.Real (indexCommutedIO x scale out inv n block lanes ⊨ᵢ fun _ _ xs ys j => ys j*(xs j : ℝ)*inv)
```

**Assumptions / layout contracts:**
- `positiveBlock : 0<block`
- `positiveLanes : 0<lanes`

**Closed-form spec defs (transitive):** `indexCommutedIO`, `originalIO`, `indexCommuted`, `original`

<details><summary><code>indexCommutedIO</code></summary>

```lean
def indexCommutedIO (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) : Masked2DKernelIO₂ :=
  { originalIO x scale out inv n block lanes with
    kernel := (indexCommuted x scale out inv n block lanes).eraseDType
    projection := by
      have h : (indexCommuted x scale out inv n block lanes).toAlgorithm? =
          .ok (indexCommuted x scale out inv n block lanes).toAlgKernel := rfl
      simp only [ComputeKernel.eraseDType,h,ComputeKernel.toAlgorithm?_fromKernelBody,
        ComputeKernel.toAlgKernel_fromKernelBody] }
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- The row uses its lane mask; all active lanes refer to the same unmasked
scale cell. The positive-size conditions below guarantee such a lane exists. -/
```
```lean
def originalIO (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) : Masked2DKernelIO₂ where
  kernel := (original x scale out inv n block lanes).eraseDType
  projection := by
    have h : (original x scale out inv n block lanes).toAlgorithm? =
        .ok (original x scale out inv n block lanes).toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType,h,ComputeKernel.toAlgorithm?_fromKernelBody,
      ComputeKernel.toAlgKernel_fromKernelBody]
  in1 := x
  in2 := scale
  out := out
  B := lanes
  read1 := fun pid _ j => pid*block+j.val
  read2 := fun pid _ _ => pid
  write := fun pid _ j => pid*block+j.val
  mask := fun _ _ j => j.val<block
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted := dequantizeRowwiseSource% indexCommuted
```
</details>

<details><summary><code>original</code></summary>

```lean
def original := dequantizeRowwiseSource% original
```
</details>
