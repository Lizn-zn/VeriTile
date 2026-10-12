# Spec sheet — `bench/tritonbench_g/cache_transform_triton/Correct.lean`

**Python source:** `bench/tritonbench_g/cache_transform_triton/cache_transform_triton.py`

## Public theorem: `prefill_cache_correctness`

**Statement:**
```lean
specification prefill_cache_correctness :
    Spec.Real (prefillOriginalIO p cosType sinType total N ⊨[total,hN] prefillExpected hN)
```

**Closed-form spec defs (transitive):** `prefillOriginalIO`, `prefillExpected`, `prefill`

<details><summary><code>prefillOriginalIO</code></summary>

```lean
def prefillOriginalIO (p : Parameters) (cosType sinType : TypedSource.Storage)
    (total N : Nat) : PrefillIO.IO p N where
  kernel := (p.prefill cosType sinType Bool.false total N).eraseDType
  projection := (PrefillIO.sourceIO p cosType sinType Bool.false total N).projection
  readMetadata := fun i => i.val
  read := fun row i => row * p.rowStride + i.val * p.hiddenStride
  write := fun token i => token * p.rowStride + i.val * p.hiddenStride
```
</details>

<details><summary><code>prefillExpected</code></summary>

```
/-- Select the greatest cumulative boundary at or before the token, exactly as
the source's signed maximum does, then gather both cache rows. -/
```
```lean
def prefillExpected {H N : Nat} (hN : 0 < N) (token : Nat)
    (xs : PrefillInputs H N) (i : Fin H) : ℝ × ℝ :=
  let boundary := (Finset.univ : Finset (Fin N)).sup' ⟨⟨0,hN⟩, Finset.mem_univ _⟩
    (fun j => if xs.ends j ≤ (token : Int) then xs.ends j else 0)
  let row := ((token : Int) - boundary).toNat
  (xs.cos row i, xs.sin row i)
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

## Public theorem: `prefill_index_commuted_correctness`

**Statement:**
```lean
specification prefill_index_commuted_correctness :
    Spec.Real (prefillIndexCommutedIO p cosType sinType total N ⊨[total,hN] prefillExpected hN)
```

**Closed-form spec defs (transitive):** `prefillIndexCommutedIO`, `prefillExpected`, `prefillOriginalIO`, `prefill`

<details><summary><code>prefillIndexCommutedIO</code></summary>

```lean
def prefillIndexCommutedIO (p : Parameters) (cosType sinType : TypedSource.Storage)
    (total N : Nat) : PrefillIO.IO p N :=
  { prefillOriginalIO p cosType sinType total N with
    kernel := (p.prefill cosType sinType Bool.true total N).eraseDType
    projection := (PrefillIO.sourceIO p cosType sinType Bool.true total N).projection }
```
</details>

<details><summary><code>prefillExpected</code></summary>

```
/-- Select the greatest cumulative boundary at or before the token, exactly as
the source's signed maximum does, then gather both cache rows. -/
```
```lean
def prefillExpected {H N : Nat} (hN : 0 < N) (token : Nat)
    (xs : PrefillInputs H N) (i : Fin H) : ℝ × ℝ :=
  let boundary := (Finset.univ : Finset (Fin N)).sup' ⟨⟨0,hN⟩, Finset.mem_univ _⟩
    (fun j => if xs.ends j ≤ (token : Int) then xs.ends j else 0)
  let row := ((token : Int) - boundary).toNat
  (xs.cos row i, xs.sin row i)
```
</details>

<details><summary><code>prefillOriginalIO</code></summary>

```lean
def prefillOriginalIO (p : Parameters) (cosType sinType : TypedSource.Storage)
    (total N : Nat) : PrefillIO.IO p N where
  kernel := (p.prefill cosType sinType Bool.false total N).eraseDType
  projection := (PrefillIO.sourceIO p cosType sinType Bool.false total N).projection
  readMetadata := fun i => i.val
  read := fun row i => row * p.rowStride + i.val * p.hiddenStride
  write := fun token i => token * p.rowStride + i.val * p.hiddenStride
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

## Public theorem: `decoding_cache_correctness`

**Statement:**
```lean
specification decoding_cache_correctness :
    Spec.Real (decodeOriginalIO p cosType sinType N ⊨[N] decodeExpected)
```

**Closed-form spec defs (transitive):** `decodeOriginalIO`, `decodeExpected`, `decode`, `Parameters`

<details><summary><code>decodeOriginalIO</code></summary>

```lean
def decodeOriginalIO (p : Parameters) (cosType sinType : TypedSource.Storage)
    (N : Nat) : DecodeIO.IO p where
  kernel := (p.decode cosType sinType Bool.false N).eraseDType
  projection := (DecodeIO.sourceIO p cosType sinType Bool.false N).projection
  readMetadata := id
  read := fun row i => row * p.rowStride + i.val * p.hiddenStride
  write := fun seq i => seq * p.rowStride + i.val * p.hiddenStride
```
</details>

<details><summary><code>decodeExpected</code></summary>

```
/-- Gather the row selected by each active sequence's signed metadata. -/
```
```lean
def decodeExpected {H N : Nat} (xs : DecodeInputs H N) (seq : Fin N) (i : Fin H) : ℝ × ℝ :=
  (xs.cos (xs.rows seq).toNat i, xs.sin (xs.rows seq).toNat i)

section Prefill
open scoped PrefillIO
variable (p : Parameters) (cosType sinType : TypedSource.Storage) (total N : Nat)
    (hN : 0 < N) (hRegion : p.cosOut ≠ p.sinOut) (hStride : 0 < p.hiddenStride)
include hRegion hStride
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

<details><summary><code>Parameters</code></summary>

```lean
abbrev Parameters := Layout
```
</details>

## Public theorem: `decode_index_commuted_correctness`

**Statement:**
```lean
specification decode_index_commuted_correctness :
    Spec.Real (decodeIndexCommutedIO p cosType sinType N ⊨[N] decodeExpected)
```

**Closed-form spec defs (transitive):** `decodeIndexCommutedIO`, `decodeExpected`, `decodeOriginalIO`, `decode`, `Parameters`

<details><summary><code>decodeIndexCommutedIO</code></summary>

```lean
def decodeIndexCommutedIO (p : Parameters) (cosType sinType : TypedSource.Storage)
    (N : Nat) : DecodeIO.IO p :=
  { decodeOriginalIO p cosType sinType N with
    kernel := (p.decode cosType sinType Bool.true N).eraseDType
    projection := (DecodeIO.sourceIO p cosType sinType Bool.true N).projection }
```
</details>

<details><summary><code>decodeExpected</code></summary>

```
/-- Gather the row selected by each active sequence's signed metadata. -/
```
```lean
def decodeExpected {H N : Nat} (xs : DecodeInputs H N) (seq : Fin N) (i : Fin H) : ℝ × ℝ :=
  (xs.cos (xs.rows seq).toNat i, xs.sin (xs.rows seq).toNat i)

section Prefill
open scoped PrefillIO
variable (p : Parameters) (cosType sinType : TypedSource.Storage) (total N : Nat)
    (hN : 0 < N) (hRegion : p.cosOut ≠ p.sinOut) (hStride : 0 < p.hiddenStride)
include hRegion hStride
```
</details>

<details><summary><code>decodeOriginalIO</code></summary>

```lean
def decodeOriginalIO (p : Parameters) (cosType sinType : TypedSource.Storage)
    (N : Nat) : DecodeIO.IO p where
  kernel := (p.decode cosType sinType Bool.false N).eraseDType
  projection := (DecodeIO.sourceIO p cosType sinType Bool.false N).projection
  readMetadata := id
  read := fun row i => row * p.rowStride + i.val * p.hiddenStride
  write := fun seq i => seq * p.rowStride + i.val * p.hiddenStride
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

<details><summary><code>Parameters</code></summary>

```lean
abbrev Parameters := Layout
```
</details>
