# Spec sheet — `bench/tritonbench_g/cache_transform_triton/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/cache_transform_triton/cache_transform_triton.py`

## Public theorem: `prefill_index_commuted_equiv`

**Statement:**
```lean
specification prefill_index_commuted_equiv :
    prefillOriginal p cosType sinType total N ≡[assumptions]
    prefillIndexCommuted p cosType sinType total N
```

**Closed-form spec defs (transitive):** `prefillOriginal`, `assumptions`, `prefillIndexCommuted`, `prefill`

<details><summary><code>prefillOriginal</code></summary>

```lean
def prefillOriginal : Program where
  kernel := p.prefill cosType sinType Bool.false total N
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>prefillIndexCommuted</code></summary>

```lean
def prefillIndexCommuted : Program :=
  { prefillOriginal p cosType sinType total N with
    kernel := p.prefill cosType sinType Bool.true total N }
```
</details>

<details><summary><code>prefill</code></summary>

```lean
def prefill (cosType sinType : Storage) (commuted : Bool) : PrefillSource :=
  match cosType, sinType, commuted with
  | .fp32, .fp32, Bool.false => cachePrefill%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => cachePrefill%[tl.float32, tl.float32, commuted]
  | .fp32, .fp16, Bool.false => cachePrefill%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => cachePrefill%[tl.float32, tl.float16, commuted]
  | .fp32, .bf16, Bool.false => cachePrefill%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => cachePrefill%[tl.float32, tl.bfloat16, commuted]
  | .fp16, .fp32, Bool.false => cachePrefill%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => cachePrefill%[tl.float16, tl.float32, commuted]
  | .fp16, .fp16, Bool.false => cachePrefill%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => cachePrefill%[tl.float16, tl.float16, commuted]
  | .fp16, .bf16, Bool.false => cachePrefill%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => cachePrefill%[tl.float16, tl.bfloat16, commuted]
  | .bf16, .fp32, Bool.false => cachePrefill%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => cachePrefill%[tl.bfloat16, tl.float32, commuted]
  | .bf16, .fp16, Bool.false => cachePrefill%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => cachePrefill%[tl.bfloat16, tl.float16, commuted]
  | .bf16, .bf16, Bool.false => cachePrefill%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => cachePrefill%[tl.bfloat16, tl.bfloat16, commuted]
```
</details>

## Public theorem: `decode_index_commuted_equiv`

**Statement:**
```lean
specification decode_index_commuted_equiv :
    decodeOriginal p cosType sinType N ≡[assumptions] decodeIndexCommuted p cosType sinType N
```

**Closed-form spec defs (transitive):** `decodeOriginal`, `assumptions`, `decodeIndexCommuted`, `decode`

<details><summary><code>decodeOriginal</code></summary>

```lean
def decodeOriginal : Program where
  kernel := p.decode cosType sinType Bool.false N
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>decodeIndexCommuted</code></summary>

```lean
def decodeIndexCommuted : Program :=
  { decodeOriginal p cosType sinType N with kernel := p.decode cosType sinType Bool.true N }
```
</details>

<details><summary><code>decode</code></summary>

```lean
def decode (cosType sinType : Storage) (commuted : Bool) : DecodeSource :=
  match cosType, sinType, commuted with
  | .fp32, .fp32, Bool.false => cacheDecode%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => cacheDecode%[tl.float32, tl.float32, commuted]
  | .fp32, .fp16, Bool.false => cacheDecode%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => cacheDecode%[tl.float32, tl.float16, commuted]
  | .fp32, .bf16, Bool.false => cacheDecode%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => cacheDecode%[tl.float32, tl.bfloat16, commuted]
  | .fp16, .fp32, Bool.false => cacheDecode%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => cacheDecode%[tl.float16, tl.float32, commuted]
  | .fp16, .fp16, Bool.false => cacheDecode%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => cacheDecode%[tl.float16, tl.float16, commuted]
  | .fp16, .bf16, Bool.false => cacheDecode%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => cacheDecode%[tl.float16, tl.bfloat16, commuted]
  | .bf16, .fp32, Bool.false => cacheDecode%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => cacheDecode%[tl.bfloat16, tl.float32, commuted]
  | .bf16, .fp16, Bool.false => cacheDecode%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => cacheDecode%[tl.bfloat16, tl.float16, commuted]
  | .bf16, .bf16, Bool.false => cacheDecode%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => cacheDecode%[tl.bfloat16, tl.bfloat16, commuted]

#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>
