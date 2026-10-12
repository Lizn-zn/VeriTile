# Spec sheet — `bench/tritonbench_g/bgmv_expand_slice/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/bgmv_expand_slice/bgmv_expand_slice.py`

## Public theorem: `bgmv_output_first_equiv`

**Statement:**
```lean
specification bgmv_output_first_equiv (R : FP.ScalarArithmetic.Rules) :
    original input weight cast out_dtype x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add ≡[R]
    outputFirst input weight cast out_dtype x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add
```

**Closed-form spec defs (transitive):** `original`, `outputFirst`

<details><summary><code>original</code></summary>

```lean
def original : Program where
  kernel := TypedSource.original input weight cast out_dtype x w out N K indices
    sx sk sw0 swn swk sy syn slice BN BK splits even add
  domain := FPEquivProofs.requirementsFor (TypedSource.original input weight cast out_dtype x w out N K indices
    sx sk sw0 swn swk sy syn slice BN BK splits even add)
```
</details>

<details><summary><code>outputFirst</code></summary>

```lean
def outputFirst : Program :=
  { original input weight cast out_dtype x w out N K indices sx sk sw0 swn swk sy syn slice BN BK splits even add with
    kernel := TypedSource.outputFirst input weight cast out_dtype x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add }
```
</details>
