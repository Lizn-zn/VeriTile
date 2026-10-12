# Spec sheet — `bench/tritonbench_g/bgmv_shrink_kernel/Correct.lean`

**Python source:** `bench/tritonbench_g/bgmv_shrink_kernel/bgmv_shrink_kernel.py`

## Public theorem: `bgmv_shrink_correctness`

**Statement:**
```lean
specification bgmv_shrink_correctness (p : Parameters) (storage : TypedSource.Storage) (out_dtype : FloatDType)
    (hS : 0 < p.BK * p.splits) (hstride : 0 < p.outputStride) (hN : p.N ≤ p.BN) :
    Spec.Real (originalIO p storage out_dtype ⊨ expected p)
```

**Assumptions / layout contracts:**
- `hS : 0 < p.BK * p.splits`
- `hstride : 0 < p.outputStride`
- `hN : p.N ≤ p.BN`

**Closed-form spec defs (transitive):** `Parameters`, `originalIO`, `expected`, `original`, `specialize`, `scalarLeft`

<details><summary><code>Parameters</code></summary>

```lean
abbrev Parameters := RealSetupSource.Parameters
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- Initial X/W values are pinned only on this split's rank indices below K.
All N output lanes are allocated; initial output is pinned only for atomic
addition. Metadata is read at batch. Allocation roles are vector, weights,
output and signed metadata, with the read/write output allocated once. -/
```
```lean
def originalIO : RealIO.IO p where
  kernel := (TypedSource.original storage out_dtype p.input p.weights p.output p.N p.K p.indices p.scaling
    p.inputBatchStride p.inputStride p.slotStride p.rowStride p.colStride p.outputBatchStride
    p.outputStride p.BN p.BK p.splits).eraseDType
  projection := RealIO.projection p storage out_dtype Bool.false
  readIndex := fun batch => batch
  readInput := fun batch k => batch * p.inputBatchStride + k.val
  readWeight := fun slot n k => p.slotStride * slot.toNat + n.val * p.rowStride + k.val * p.colStride
  write := fun batch n => batch * p.outputBatchStride + n.val * p.outputStride
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Independent rank-slice formula. Every field of xs is an INITIAL input,
including initialOutput; no computed output is assumed. Each block contributes
BK ranks separated from the next by BK*SPLIT_K, with zero beyond K. -/
```
```lean
def expected (p : Parameters) (split : Nat) (xs : Inputs p) (n : Fin p.N) : ℝ :=
  let v := p.scaling * ∑ c ∈ Finset.range ((p.K + p.BK * p.splits - 1) / (p.BK * p.splits)),
    ∑ e : Fin p.BK,
      let k := c * (p.BK * p.splits) + (e.val + split * p.BK)
      if hk : k < p.K then xs.vector ⟨k, hk⟩ * xs.weights n ⟨k, hk⟩ else 0
  if p.splits = 1 then v else xs.initialOutput n + v
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (storage : Storage) : KernelSource := specialize storage Bool.false
```
</details>

<details><summary><code>specialize</code></summary>

```lean
def specialize (storage : Storage) (swapped : Bool) : KernelSource :=
  match storage, swapped with
  | .fp16, Bool.false => bgmvShrinkSource%[tl.float16, original]
  | .fp16, Bool.true => bgmvShrinkSource%[tl.float16, scalarLeft]
  | .bf16, Bool.false => bgmvShrinkSource%[tl.bfloat16, original]
  | .bf16, Bool.true => bgmvShrinkSource%[tl.bfloat16, scalarLeft]
```
</details>

<details><summary><code>scalarLeft</code></summary>

```lean
def scalarLeft (storage : Storage) : KernelSource := specialize storage Bool.true

#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>

## Public theorem: `bgmv_scalar_left_correctness`

**Statement:**
```lean
specification bgmv_scalar_left_correctness (p : Parameters) (storage : TypedSource.Storage) (out_dtype : FloatDType)
    (hS : 0 < p.BK * p.splits) (hstride : 0 < p.outputStride) (hN : p.N ≤ p.BN) :
    Spec.Real (scalarLeftIO p storage out_dtype ⊨ expected p)
```

**Assumptions / layout contracts:**
- `hS : 0 < p.BK * p.splits`
- `hstride : 0 < p.outputStride`
- `hN : p.N ≤ p.BN`

**Closed-form spec defs (transitive):** `Parameters`, `scalarLeftIO`, `expected`, `originalIO`, `scalarLeft`, `original`, `specialize`

<details><summary><code>Parameters</code></summary>

```lean
abbrev Parameters := RealSetupSource.Parameters
```
</details>

<details><summary><code>scalarLeftIO</code></summary>

```lean
def scalarLeftIO : RealIO.IO p :=
  { originalIO p storage out_dtype with
    kernel := (TypedSource.scalarLeft storage out_dtype p.input p.weights p.output p.N p.K p.indices p.scaling
      p.inputBatchStride p.inputStride p.slotStride p.rowStride p.colStride p.outputBatchStride
      p.outputStride p.BN p.BK p.splits).eraseDType
    projection := RealIO.projection p storage out_dtype Bool.true }
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Independent rank-slice formula. Every field of xs is an INITIAL input,
including initialOutput; no computed output is assumed. Each block contributes
BK ranks separated from the next by BK*SPLIT_K, with zero beyond K. -/
```
```lean
def expected (p : Parameters) (split : Nat) (xs : Inputs p) (n : Fin p.N) : ℝ :=
  let v := p.scaling * ∑ c ∈ Finset.range ((p.K + p.BK * p.splits - 1) / (p.BK * p.splits)),
    ∑ e : Fin p.BK,
      let k := c * (p.BK * p.splits) + (e.val + split * p.BK)
      if hk : k < p.K then xs.vector ⟨k, hk⟩ * xs.weights n ⟨k, hk⟩ else 0
  if p.splits = 1 then v else xs.initialOutput n + v
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- Initial X/W values are pinned only on this split's rank indices below K.
All N output lanes are allocated; initial output is pinned only for atomic
addition. Metadata is read at batch. Allocation roles are vector, weights,
output and signed metadata, with the read/write output allocated once. -/
```
```lean
def originalIO : RealIO.IO p where
  kernel := (TypedSource.original storage out_dtype p.input p.weights p.output p.N p.K p.indices p.scaling
    p.inputBatchStride p.inputStride p.slotStride p.rowStride p.colStride p.outputBatchStride
    p.outputStride p.BN p.BK p.splits).eraseDType
  projection := RealIO.projection p storage out_dtype Bool.false
  readIndex := fun batch => batch
  readInput := fun batch k => batch * p.inputBatchStride + k.val
  readWeight := fun slot n k => p.slotStride * slot.toNat + n.val * p.rowStride + k.val * p.colStride
  write := fun batch n => batch * p.outputBatchStride + n.val * p.outputStride
```
</details>

<details><summary><code>scalarLeft</code></summary>

```lean
def scalarLeft (storage : Storage) : KernelSource := specialize storage Bool.true

#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (storage : Storage) : KernelSource := specialize storage Bool.false
```
</details>

<details><summary><code>specialize</code></summary>

```lean
def specialize (storage : Storage) (swapped : Bool) : KernelSource :=
  match storage, swapped with
  | .fp16, Bool.false => bgmvShrinkSource%[tl.float16, original]
  | .fp16, Bool.true => bgmvShrinkSource%[tl.float16, scalarLeft]
  | .bf16, Bool.false => bgmvShrinkSource%[tl.bfloat16, original]
  | .bf16, Bool.true => bgmvShrinkSource%[tl.bfloat16, scalarLeft]
```
</details>
