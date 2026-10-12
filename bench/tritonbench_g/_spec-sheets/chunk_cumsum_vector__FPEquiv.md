# Spec sheet — `bench/tritonbench_g/chunk_cumsum_vector/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/chunk_cumsum_vector/chunk_cumsum_vector.py`

## Public theorem: `chunk_sum_left_equiv`

**Statement:**
```lean
specification chunk_sum_left_equiv (R : FP.ScalarArithmetic.Rules) :
    original dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS ≡[R]
      sumLeft dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS
```

**Closed-form spec defs (transitive):** `original`, `sumLeft`

<details><summary><code>original</code></summary>

```lean
def original : Program where
  kernel := TypedSource.original dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS
  domain := FPEquivProofs.requirementsFor
    (TypedSource.original dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS) BT BS
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft : Program :=
  { original dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS with
    kernel := TypedSource.sumLeft dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS }
```
</details>
