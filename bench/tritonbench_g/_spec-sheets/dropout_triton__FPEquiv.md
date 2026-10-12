# Spec sheet — `bench/tritonbench_g/dropout_triton/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/dropout_triton/dropout_triton.py`

## Public theorem: `dropout_equiv`

**Statement:**
```lean
specification dropout_equiv (storage : Storage) (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ) :
    original storage keep x k y n B p ≡[assumptions] indexCommuted storage keep x k y n B p
```

**Closed-form spec defs (transitive):** `original`, `assumptions`, `indexCommuted`, `source`

<details><summary><code>original</code></summary>

```lean
def original (storage : Storage) (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ) : Program where
  kernel := source storage keep Bool.false x k y n B p
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
def indexCommuted (storage : Storage) (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ) : Program where
  kernel := source storage keep Bool.true x k y n B p
  domain := []
```
</details>

<details><summary><code>source</code></summary>

```lean
def source (storage : Storage) (keep : KeepStorage) (swapped : Bool) : sourceType :=
  match storage, keep, swapped with
  | .fp32, .int32, Bool.false => dropoutSource%[tl.float32, tl.int32, original]
  | .fp32, .int32, Bool.true => dropoutSource%[tl.float32, tl.int32, indexCommuted]
  | .fp16, .int32, Bool.false => dropoutSource%[tl.float16, tl.int32, original]
  | .fp16, .int32, Bool.true => dropoutSource%[tl.float16, tl.int32, indexCommuted]
  | .bf16, .int32, Bool.false => dropoutSource%[tl.bfloat16, tl.int32, original]
  | .bf16, .int32, Bool.true => dropoutSource%[tl.bfloat16, tl.int32, indexCommuted]
  | .fp32, .bool, Bool.false => dropoutSource%[tl.float32, tl.int1, original]
  | .fp32, .bool, Bool.true => dropoutSource%[tl.float32, tl.int1, indexCommuted]
  | .fp16, .bool, Bool.false => dropoutSource%[tl.float16, tl.int1, original]
  | .fp16, .bool, Bool.true => dropoutSource%[tl.float16, tl.int1, indexCommuted]
  | .bf16, .bool, Bool.false => dropoutSource%[tl.bfloat16, tl.int1, original]
  | .bf16, .bool, Bool.true => dropoutSource%[tl.bfloat16, tl.int1, indexCommuted]
```
</details>
