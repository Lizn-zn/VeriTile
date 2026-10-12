# Spec sheet — `bench/tritonbench_g/destindex_copy/Correct.lean`

**Python source:** `bench/tritonbench_g/destindex_copy/destindex_copy.py`

## Public theorem: `destindex_copy_correctness`

**Statement:**
```lean
specification destindex_copy_correctness :
    Spec.Real (originalIO p nopeType ropeType ⊨ expected)
```

**Closed-form spec defs (transitive):** `originalIO`, `expected`, `kernel`

<details><summary><code>originalIO</code></summary>

```lean
def originalIO (p : Layout) (nopeType ropeType : TypedSource.Storage) : RealIO.IO p where
  kernel := (p.source nopeType ropeType Bool.false).eraseDType
  projection := (SourceIO.sourceIO p nopeType ropeType Bool.false).projection
  readMetadata := id
  readNope := fun pid j => pid*p.stride_kv_nope_bs + p.stride_kv_nope_d*j.val
  readRope := fun pid j => pid*p.stride_kv_rope_bs + p.stride_kv_rope_d*j.val
  writeNope := fun row j => row.toNat*p.stride_o_nope_bs + p.stride_o_nope_d*j.val
  writeRope := fun row j => row.toNat*p.stride_o_rope_bs + p.stride_o_rope_d*j.val
```
</details>

<details><summary><code>expected</code></summary>

```lean
def expected {N R : Nat} (xs : RealIO.Inputs N R) : (Fin N → ℝ) × (Fin R → ℝ) :=
  (xs.nope,xs.rope)

variable (p : Layout) (nopeType ropeType : TypedSource.Storage)
    (hOut : p.O_nope ≠ p.O_rope)
    (hNope : Function.Injective (fun j : Fin p.BLOCK_DMODEL_NOPE => p.stride_o_nope_d*j.val))
    (hRope : Function.Injective (fun j : Fin p.BLOCK_DMODEL_ROPE => p.stride_o_rope_d*j.val))
include hOut hNope hRope
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

## Public theorem: `index_commuted_correctness`

**Statement:**
```lean
specification index_commuted_correctness :
    Spec.Real (indexCommutedIO p nopeType ropeType ⊨ expected)
```

**Closed-form spec defs (transitive):** `indexCommutedIO`, `expected`, `originalIO`, `kernel`

<details><summary><code>indexCommutedIO</code></summary>

```lean
def indexCommutedIO (p : Layout) (nopeType ropeType : TypedSource.Storage) : RealIO.IO p :=
  { originalIO p nopeType ropeType with
    kernel := (p.source nopeType ropeType Bool.true).eraseDType
    projection := (SourceIO.sourceIO p nopeType ropeType Bool.true).projection }
```
</details>

<details><summary><code>expected</code></summary>

```lean
def expected {N R : Nat} (xs : RealIO.Inputs N R) : (Fin N → ℝ) × (Fin R → ℝ) :=
  (xs.nope,xs.rope)

variable (p : Layout) (nopeType ropeType : TypedSource.Storage)
    (hOut : p.O_nope ≠ p.O_rope)
    (hNope : Function.Injective (fun j : Fin p.BLOCK_DMODEL_NOPE => p.stride_o_nope_d*j.val))
    (hRope : Function.Injective (fun j : Fin p.BLOCK_DMODEL_ROPE => p.stride_o_rope_d*j.val))
include hOut hNope hRope
```
</details>

<details><summary><code>originalIO</code></summary>

```lean
def originalIO (p : Layout) (nopeType ropeType : TypedSource.Storage) : RealIO.IO p where
  kernel := (p.source nopeType ropeType Bool.false).eraseDType
  projection := (SourceIO.sourceIO p nopeType ropeType Bool.false).projection
  readMetadata := id
  readNope := fun pid j => pid*p.stride_kv_nope_bs + p.stride_kv_nope_d*j.val
  readRope := fun pid j => pid*p.stride_kv_rope_bs + p.stride_kv_rope_d*j.val
  writeNope := fun row j => row.toNat*p.stride_o_nope_bs + p.stride_o_nope_d*j.val
  writeRope := fun row j => row.toNat*p.stride_o_rope_bs + p.stride_o_rope_d*j.val
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
