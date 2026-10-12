# Spec sheet — `bench/tritonbench_g/destindex_copy_kv1/Correct.lean`

**Python source:** `bench/tritonbench_g/destindex_copy_kv1/destindex_copy_kv1.py`

## Public theorem: `destindex_copy_kv_correctness`

**Statement:**
```lean
specification destindex_copy_kv_correctness :
    Spec.Real (originalIO p input output ⊨ expected)
```

**Closed-form spec defs (transitive):** `originalIO`, `expected`, `kernel`

<details><summary><code>originalIO</code></summary>

```lean
def originalIO (p : Layout) (input output : TypedSource.Storage) : RealIO.IO p where
  kernel := (p.source input output Bool.false).eraseDType
  projection := (SourceIO.sourceIO p input output Bool.false).projection
  readMetadata := id
  read := fun pid j => pid*p.stride_k_bs + p.stride_k_h*j.1.val + p.stride_k_d*j.2.1.val
  write := fun row j => row.toNat*p.stride_o_bs + p.stride_o_h*j.1.val + p.stride_o_d*j.2.1.val
  active := fun j => j.1.val<p.head_num ∧ j.2.1.val<p.head_dim
```
</details>

<details><summary><code>expected</code></summary>

```lean
def expected {H D : Nat} (xs : RealIO.Inputs H D) : TileIndex [H,D] → ℝ := xs.values

variable (p : Layout) (input output : TypedSource.Storage)
    (hLayout : Function.Injective (fun j : TileIndex [p.BLOCK_HEAD,p.BLOCK_DMODEL] =>
      p.stride_o_h*j.1.val + p.stride_o_d*j.2.1.val))
include hLayout
```
</details>

<details><summary><code>kernel</code></summary>

```lean
def kernel (input output : Storage) (swapped : Bool) : Source :=
  match input, output, swapped with
  | .fp32, .fp32, Bool.false => destindexCopyKv1Source%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => destindexCopyKv1Source%[tl.float32, tl.float32, indexCommuted]
  | .fp32, .fp16, Bool.false => destindexCopyKv1Source%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => destindexCopyKv1Source%[tl.float32, tl.float16, indexCommuted]
  | .fp32, .bf16, Bool.false => destindexCopyKv1Source%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => destindexCopyKv1Source%[tl.float32, tl.bfloat16, indexCommuted]
  | .fp16, .fp32, Bool.false => destindexCopyKv1Source%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => destindexCopyKv1Source%[tl.float16, tl.float32, indexCommuted]
  | .fp16, .fp16, Bool.false => destindexCopyKv1Source%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => destindexCopyKv1Source%[tl.float16, tl.float16, indexCommuted]
  | .fp16, .bf16, Bool.false => destindexCopyKv1Source%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => destindexCopyKv1Source%[tl.float16, tl.bfloat16, indexCommuted]
  | .bf16, .fp32, Bool.false => destindexCopyKv1Source%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => destindexCopyKv1Source%[tl.bfloat16, tl.float32, indexCommuted]
  | .bf16, .fp16, Bool.false => destindexCopyKv1Source%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => destindexCopyKv1Source%[tl.bfloat16, tl.float16, indexCommuted]
  | .bf16, .bf16, Bool.false => destindexCopyKv1Source%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => destindexCopyKv1Source%[tl.bfloat16, tl.bfloat16, indexCommuted]
```
</details>

## Public theorem: `index_commuted_correctness`

**Statement:**
```lean
specification index_commuted_correctness :
    Spec.Real (indexCommutedIO p input output ⊨ expected)
```

**Closed-form spec defs (transitive):** `indexCommutedIO`, `expected`, `originalIO`, `kernel`

<details><summary><code>indexCommutedIO</code></summary>

```lean
def indexCommutedIO (p : Layout) (input output : TypedSource.Storage) : RealIO.IO p :=
  { originalIO p input output with
    kernel := (p.source input output Bool.true).eraseDType
    projection := (SourceIO.sourceIO p input output Bool.true).projection }
```
</details>

<details><summary><code>expected</code></summary>

```lean
def expected {H D : Nat} (xs : RealIO.Inputs H D) : TileIndex [H,D] → ℝ := xs.values

variable (p : Layout) (input output : TypedSource.Storage)
    (hLayout : Function.Injective (fun j : TileIndex [p.BLOCK_HEAD,p.BLOCK_DMODEL] =>
      p.stride_o_h*j.1.val + p.stride_o_d*j.2.1.val))
include hLayout
```
</details>

<details><summary><code>originalIO</code></summary>

```lean
def originalIO (p : Layout) (input output : TypedSource.Storage) : RealIO.IO p where
  kernel := (p.source input output Bool.false).eraseDType
  projection := (SourceIO.sourceIO p input output Bool.false).projection
  readMetadata := id
  read := fun pid j => pid*p.stride_k_bs + p.stride_k_h*j.1.val + p.stride_k_d*j.2.1.val
  write := fun row j => row.toNat*p.stride_o_bs + p.stride_o_h*j.1.val + p.stride_o_d*j.2.1.val
  active := fun j => j.1.val<p.head_num ∧ j.2.1.val<p.head_dim
```
</details>

<details><summary><code>kernel</code></summary>

```lean
def kernel (input output : Storage) (swapped : Bool) : Source :=
  match input, output, swapped with
  | .fp32, .fp32, Bool.false => destindexCopyKv1Source%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => destindexCopyKv1Source%[tl.float32, tl.float32, indexCommuted]
  | .fp32, .fp16, Bool.false => destindexCopyKv1Source%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => destindexCopyKv1Source%[tl.float32, tl.float16, indexCommuted]
  | .fp32, .bf16, Bool.false => destindexCopyKv1Source%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => destindexCopyKv1Source%[tl.float32, tl.bfloat16, indexCommuted]
  | .fp16, .fp32, Bool.false => destindexCopyKv1Source%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => destindexCopyKv1Source%[tl.float16, tl.float32, indexCommuted]
  | .fp16, .fp16, Bool.false => destindexCopyKv1Source%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => destindexCopyKv1Source%[tl.float16, tl.float16, indexCommuted]
  | .fp16, .bf16, Bool.false => destindexCopyKv1Source%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => destindexCopyKv1Source%[tl.float16, tl.bfloat16, indexCommuted]
  | .bf16, .fp32, Bool.false => destindexCopyKv1Source%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => destindexCopyKv1Source%[tl.bfloat16, tl.float32, indexCommuted]
  | .bf16, .fp16, Bool.false => destindexCopyKv1Source%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => destindexCopyKv1Source%[tl.bfloat16, tl.float16, indexCommuted]
  | .bf16, .bf16, Bool.false => destindexCopyKv1Source%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => destindexCopyKv1Source%[tl.bfloat16, tl.bfloat16, indexCommuted]
```
</details>
