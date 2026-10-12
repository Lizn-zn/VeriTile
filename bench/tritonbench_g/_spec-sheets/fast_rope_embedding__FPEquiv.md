# Spec sheet — `bench/tritonbench_g/fast_rope_embedding/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/fast_rope_embedding/fast_rope_embedding.py`

## Public theorem: `rope_equiv`

**Statement:**
```lean
specification rope_equiv (input cosine sine : Storage) (Q C S : RegionName)
    (QR CR SR seq H heads B : Nat) (backwards : Bool) :
    program input cosine sine Bool.false Q C S QR CR SR seq H heads B backwards ≡[assumptions]
    program input cosine sine Bool.true Q C S QR CR SR seq H heads B backwards
```

**Closed-form spec defs (transitive):** `program`, `assumptions`, `rope`

<details><summary><code>program</code></summary>

```lean
def program (input cosine sine : Storage) (swapped : Bool) (Q C S : RegionName)
    (QR CR SR seq H heads B : Nat) (backwards : Bool) : Program where
  kernel := rope input cosine sine swapped Q C S QR CR SR seq H heads B backwards
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>rope</code></summary>

```lean
def rope (input cosine sine : Storage) (swapped : Bool) := match input,cosine,sine with
  | .fp16,.fp16,.fp16 => if swapped then fastRope%[tl.float16,tl.float16,tl.float16,indexCommuted] else fastRope%[tl.float16,tl.float16,tl.float16,original]
  | .fp16,.fp16,.bf16 => if swapped then fastRope%[tl.float16,tl.float16,tl.bfloat16,indexCommuted] else fastRope%[tl.float16,tl.float16,tl.bfloat16,original]
  | .fp16,.fp16,.fp32 => if swapped then fastRope%[tl.float16,tl.float16,tl.float32,indexCommuted] else fastRope%[tl.float16,tl.float16,tl.float32,original]
  | .fp16,.bf16,.fp16 => if swapped then fastRope%[tl.float16,tl.bfloat16,tl.float16,indexCommuted] else fastRope%[tl.float16,tl.bfloat16,tl.float16,original]
  | .fp16,.bf16,.bf16 => if swapped then fastRope%[tl.float16,tl.bfloat16,tl.bfloat16,indexCommuted] else fastRope%[tl.float16,tl.bfloat16,tl.bfloat16,original]
  | .fp16,.bf16,.fp32 => if swapped then fastRope%[tl.float16,tl.bfloat16,tl.float32,indexCommuted] else fastRope%[tl.float16,tl.bfloat16,tl.float32,original]
  | .fp16,.fp32,.fp16 => if swapped then fastRope%[tl.float16,tl.float32,tl.float16,indexCommuted] else fastRope%[tl.float16,tl.float32,tl.float16,original]
  | .fp16,.fp32,.bf16 => if swapped then fastRope%[tl.float16,tl.float32,tl.bfloat16,indexCommuted] else fastRope%[tl.float16,tl.float32,tl.bfloat16,original]
  | .fp16,.fp32,.fp32 => if swapped then fastRope%[tl.float16,tl.float32,tl.float32,indexCommuted] else fastRope%[tl.float16,tl.float32,tl.float32,original]
  | .bf16,.fp16,.fp16 => if swapped then fastRope%[tl.bfloat16,tl.float16,tl.float16,indexCommuted] else fastRope%[tl.bfloat16,tl.float16,tl.float16,original]
  | .bf16,.fp16,.bf16 => if swapped then fastRope%[tl.bfloat16,tl.float16,tl.bfloat16,indexCommuted] else fastRope%[tl.bfloat16,tl.float16,tl.bfloat16,original]
  | .bf16,.fp16,.fp32 => if swapped then fastRope%[tl.bfloat16,tl.float16,tl.float32,indexCommuted] else fastRope%[tl.bfloat16,tl.float16,tl.float32,original]
  | .bf16,.bf16,.fp16 => if swapped then fastRope%[tl.bfloat16,tl.bfloat16,tl.float16,indexCommuted] else fastRope%[tl.bfloat16,tl.bfloat16,tl.float16,original]
  | .bf16,.bf16,.bf16 => if swapped then fastRope%[tl.bfloat16,tl.bfloat16,tl.bfloat16,indexCommuted] else fastRope%[tl.bfloat16,tl.bfloat16,tl.bfloat16,original]
  | .bf16,.bf16,.fp32 => if swapped then fastRope%[tl.bfloat16,tl.bfloat16,tl.float32,indexCommuted] else fastRope%[tl.bfloat16,tl.bfloat16,tl.float32,original]
  | .bf16,.fp32,.fp16 => if swapped then fastRope%[tl.bfloat16,tl.float32,tl.float16,indexCommuted] else fastRope%[tl.bfloat16,tl.float32,tl.float16,original]
  | .bf16,.fp32,.bf16 => if swapped then fastRope%[tl.bfloat16,tl.float32,tl.bfloat16,indexCommuted] else fastRope%[tl.bfloat16,tl.float32,tl.bfloat16,original]
  | .bf16,.fp32,.fp32 => if swapped then fastRope%[tl.bfloat16,tl.float32,tl.float32,indexCommuted] else fastRope%[tl.bfloat16,tl.float32,tl.float32,original]
  | .fp32,.fp16,.fp16 => if swapped then fastRope%[tl.float32,tl.float16,tl.float16,indexCommuted] else fastRope%[tl.float32,tl.float16,tl.float16,original]
  | .fp32,.fp16,.bf16 => if swapped then fastRope%[tl.float32,tl.float16,tl.bfloat16,indexCommuted] else fastRope%[tl.float32,tl.float16,tl.bfloat16,original]
  | .fp32,.fp16,.fp32 => if swapped then fastRope%[tl.float32,tl.float16,tl.float32,indexCommuted] else fastRope%[tl.float32,tl.float16,tl.float32,original]
  | .fp32,.bf16,.fp16 => if swapped then fastRope%[tl.float32,tl.bfloat16,tl.float16,indexCommuted] else fastRope%[tl.float32,tl.bfloat16,tl.float16,original]
  | .fp32,.bf16,.bf16 => if swapped then fastRope%[tl.float32,tl.bfloat16,tl.bfloat16,indexCommuted] else fastRope%[tl.float32,tl.bfloat16,tl.bfloat16,original]
  | .fp32,.bf16,.fp32 => if swapped then fastRope%[tl.float32,tl.bfloat16,tl.float32,indexCommuted] else fastRope%[tl.float32,tl.bfloat16,tl.float32,original]
  | .fp32,.fp32,.fp16 => if swapped then fastRope%[tl.float32,tl.float32,tl.float16,indexCommuted] else fastRope%[tl.float32,tl.float32,tl.float16,original]
  | .fp32,.fp32,.bf16 => if swapped then fastRope%[tl.float32,tl.float32,tl.bfloat16,indexCommuted] else fastRope%[tl.float32,tl.float32,tl.bfloat16,original]
  | .fp32,.fp32,.fp32 => if swapped then fastRope%[tl.float32,tl.float32,tl.float32,indexCommuted] else fastRope%[tl.float32,tl.float32,tl.float32,original]
```
</details>
