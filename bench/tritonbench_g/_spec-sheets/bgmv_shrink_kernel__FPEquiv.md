# Spec sheet — `bench/tritonbench_g/bgmv_shrink_kernel/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/bgmv_shrink_kernel/bgmv_shrink_kernel.py`

## Public theorem: `bgmv_scaling_equiv`

**Statement:**
```lean
specification bgmv_scaling_equiv (R : FP.ScalarArithmetic.Rules) :
    original storage out_dtype x w out N K indices scaling sx sk sw0 swn swk sy syn BN BK splits ≡[R]
    scalarLeft storage out_dtype x w out N K indices scaling sx sk sw0 swn swk sy syn BN BK splits
```

**Closed-form spec defs (transitive):** `original`, `scalarLeft`

<details><summary><code>original</code></summary>

```lean
def original : Program where
  kernel := TypedSource.original storage out_dtype x w out N K indices scaling
    sx sk sw0 swn swk sy syn BN BK splits
  domain := FPEquivProofs.requirementsFor (TypedSource.original storage out_dtype x w out N K indices scaling
    sx sk sw0 swn swk sy syn BN BK splits) BN scaling
```
</details>

<details><summary><code>scalarLeft</code></summary>

```lean
def scalarLeft : Program :=
  { original storage out_dtype x w out N K indices scaling sx sk sw0 swn swk sy syn BN BK splits with
    kernel := TypedSource.scalarLeft storage out_dtype x w out N K indices scaling
      sx sk sw0 swn swk sy syn BN BK splits }
```
</details>
