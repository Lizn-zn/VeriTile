# Spec sheet — `bench/tritonbench_g/bmm_optimized/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/bmm_optimized/bmm_optimized.py`

## Public theorem: `bmm_dot_first_equiv`

**Statement:**
```lean
specification bmm_dot_first_equiv (R : FP.ScalarArithmetic.Rules) :
    original dtype divM divN divK A B O M N K BM BN BK GROUP_M ≡[R]
    dotFirst dtype divM divN divK A B O M N K BM BN BK GROUP_M
```

**Closed-form spec defs (transitive):** `original`, `dotFirst`

<details><summary><code>original</code></summary>

```lean
def original : Program where
  kernel := TypedSource.original dtype divM divN divK A B O M N K BM BN BK GROUP_M
  domain := FPEquivProofs.requirementsFor
    (TypedSource.original dtype divM divN divK A B O M N K BM BN BK GROUP_M) dtype BM BN BK
```
</details>

<details><summary><code>dotFirst</code></summary>

```lean
def dotFirst : Program :=
  { original dtype divM divN divK A B O M N K BM BN BK GROUP_M with
    kernel := TypedSource.dotFirst dtype divM divN divK A B O M N K BM BN BK GROUP_M }
```
</details>
