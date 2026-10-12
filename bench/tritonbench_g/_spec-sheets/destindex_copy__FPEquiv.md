# Spec sheet — `bench/tritonbench_g/destindex_copy/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/destindex_copy/destindex_copy.py`

## Public theorem: `index_commuted_equiv`

**Statement:**
```lean
specification index_commuted_equiv (p : Layout) (nopeType ropeType : TypedSource.Storage) :
    original p nopeType ropeType ≡[assumptions] indexCommuted p nopeType ropeType
```

**Closed-form spec defs (transitive):** `original`, `assumptions`, `indexCommuted`, `kernel`

<details><summary><code>original</code></summary>

```lean
def original (p : Layout) (nopeType ropeType : TypedSource.Storage) : Program where
  kernel := p.source nopeType ropeType Bool.false
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
def indexCommuted (p : Layout) (nopeType ropeType : TypedSource.Storage) : Program where
  kernel := p.source nopeType ropeType Bool.true
  domain := []
```
</details>

<details><summary><code>kernel</code></summary>

```lean
def kernel (nopeType ropeType : Storage) (swapped : Bool) : Source :=
  match nopeType, ropeType, swapped with
  | .fp32, .fp32, Bool.false => destindexCopySource%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => destindexCopySource%[tl.float32, tl.float32, indexCommuted]
  | .fp32, .fp16, Bool.false => destindexCopySource%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => destindexCopySource%[tl.float32, tl.float16, indexCommuted]
  | .fp32, .bf16, Bool.false => destindexCopySource%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => destindexCopySource%[tl.float32, tl.bfloat16, indexCommuted]
  | .fp16, .fp32, Bool.false => destindexCopySource%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => destindexCopySource%[tl.float16, tl.float32, indexCommuted]
  | .fp16, .fp16, Bool.false => destindexCopySource%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => destindexCopySource%[tl.float16, tl.float16, indexCommuted]
  | .fp16, .bf16, Bool.false => destindexCopySource%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => destindexCopySource%[tl.float16, tl.bfloat16, indexCommuted]
  | .bf16, .fp32, Bool.false => destindexCopySource%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => destindexCopySource%[tl.bfloat16, tl.float32, indexCommuted]
  | .bf16, .fp16, Bool.false => destindexCopySource%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => destindexCopySource%[tl.bfloat16, tl.float16, indexCommuted]
  | .bf16, .bf16, Bool.false => destindexCopySource%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => destindexCopySource%[tl.bfloat16, tl.bfloat16, indexCommuted]
```
</details>
