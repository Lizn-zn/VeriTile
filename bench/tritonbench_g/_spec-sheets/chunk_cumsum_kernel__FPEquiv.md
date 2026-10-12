# Spec sheet — `bench/tritonbench_g/chunk_cumsum_kernel/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/chunk_cumsum_kernel/chunk_cumsum_kernel.py`

## Public theorem: `chunk_sum_left_equiv`

**Statement:**
```lean
specification chunk_sum_left_equiv (R : FP.ScalarArithmetic.Rules) :
    original dtype out_dtype s o T BT ≡[R] sumLeft dtype out_dtype s o T BT
```

**Closed-form spec defs (transitive):** `original`, `sumLeft`

<details><summary><code>original</code></summary>

```lean
def original : Program where
  kernel := TypedSource.original dtype out_dtype s o T BT
  domain := FPEquivProofs.requirementsFor (TypedSource.original dtype out_dtype s o T BT)
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft : Program :=
  { original dtype out_dtype s o T BT with
    kernel := TypedSource.sumLeft dtype out_dtype s o T BT }
```
</details>
