# Spec sheet — `bench/tritonbench_g/dequantize_matmul/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/dequantize_matmul/dequantize_matmul.py`

## Public theorem: `index_commuted_equiv`

**Statement:**
```lean
specification index_commuted_equiv (output : Storage) (weights scales out : RegionName)
    (K N weightRow weightCol outputRow outputCol BN BK : Nat) :
    original output weights scales out K N weightRow weightCol outputRow outputCol BN BK ≡[assumptions]
      indexCommuted output weights scales out K N weightRow weightCol outputRow outputCol BN BK
```

**Closed-form spec defs (transitive):** `original`, `assumptions`, `indexCommuted`, `kernel`

<details><summary><code>original</code></summary>

```lean
def original (output : Storage) (weights scales out : RegionName)
    (K N weightRow weightCol outputRow outputCol BN BK : Nat) : Program where
  kernel := TypedSource.kernel output Bool.false weights scales out K N weightRow weightCol outputRow outputCol BN BK
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (output : Storage) (weights scales out : RegionName)
    (K N weightRow weightCol outputRow outputCol BN BK : Nat) : Program where
  kernel := TypedSource.kernel output Bool.true weights scales out K N weightRow weightCol outputRow outputCol BN BK
  domain := []
```
</details>

<details><summary><code>kernel</code></summary>

```lean
def kernel (output : Storage) (swapped : Bool) := match output with
  | .fp16 => if swapped then dequantizeMatmulSource%[tl.float16,indexCommuted] else dequantizeMatmulSource%[tl.float16,original]
  | .bf16 => if swapped then dequantizeMatmulSource%[tl.bfloat16,indexCommuted] else dequantizeMatmulSource%[tl.bfloat16,original]
  | .fp32 => if swapped then dequantizeMatmulSource%[tl.float32,indexCommuted] else dequantizeMatmulSource%[tl.float32,original]
```
</details>
