# Spec sheet — `bench/tritonbench_g/embedding_triton_kernel/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/embedding_triton_kernel/embedding_triton_kernel.py`

## Public theorem: `embedding_equiv`

**Statement:**
```lean
specification embedding_equiv (storage : Storage) (weight ids out : RegionName)
    (first last : Int) (sw so n h D N NN : Nat) :
    original storage weight ids out first last sw so n h D N NN ≡[assumptions]
      indexCommuted storage weight ids out first last sw so n h D N NN
```

**Closed-form spec defs (transitive):** `original`, `assumptions`, `indexCommuted`, `source`

<details><summary><code>original</code></summary>

```lean
def original (storage : Storage) (weight ids out : RegionName)
    (first last : Int) (sw so n h D N NN : Nat) : Program where
  kernel := source storage Bool.false weight ids out first last sw so n h D N NN
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
def indexCommuted (storage : Storage) (weight ids out : RegionName)
    (first last : Int) (sw so n h D N NN : Nat) : Program where
  kernel := source storage Bool.true weight ids out first last sw so n h D N NN
  domain := []
```
</details>

<details><summary><code>source</code></summary>

```lean
def source (storage : Storage) (swapped : Bool) : sourceType :=
  match storage, swapped with
  | .fp32, Bool.false => embeddingSource%[tl.float32, original]
  | .fp32, Bool.true => embeddingSource%[tl.float32, indexCommuted]
  | .fp16, Bool.false => embeddingSource%[tl.float16, original]
  | .fp16, Bool.true => embeddingSource%[tl.float16, indexCommuted]
  | .bf16, Bool.false => embeddingSource%[tl.bfloat16, original]
  | .bf16, Bool.true => embeddingSource%[tl.bfloat16, indexCommuted]
```
</details>
