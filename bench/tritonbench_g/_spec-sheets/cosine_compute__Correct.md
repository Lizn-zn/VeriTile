# Spec sheet — `bench/tritonbench_g/cosine_compute/Correct.lean`

**Python source:** `bench/tritonbench_g/cosine_compute/cosine_compute.py`

## Public theorem: `original_correct`

**Statement:**
```lean
specification original_correct (storage : Storage) (a b : RegionName) (n B : Nat) :
    Spec.Real (originalIO storage a b n B ⊨ fun xs i => Real.cos (xs i))
```

**Closed-form spec defs (transitive):** `originalIO`, `kernel`, `indexCommuted`, `original`

<details><summary><code>originalIO</code></summary>

```lean
def originalIO (storage : Storage) (a b : RegionName) (n B : Nat) : MaskedKernelIO₁ where
  kernel := (kernel storage Bool.false a b n B).eraseDType
  projection := RealProof.projection storage Bool.false a b n B
  inp := a
  out := b
  B := B
  read := fun pid => pid*B
  write := fun pid => pid*B
  mask := fun pid j => pid*B+j.val < n
```
</details>

<details><summary><code>kernel</code></summary>

```lean
def kernel (storage : Storage) (swapped : Bool) (a b : RegionName) (n B : Nat) :=
  (if swapped then indexCommuted storage else original storage) a b n B
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (storage : Storage) := match storage with
  | .fp16 => cosineSource%[tl.float16,indexCommuted]
  | .bf16 => cosineSource%[tl.bfloat16,indexCommuted]
  | .fp32 => cosineSource%[tl.float32,indexCommuted]
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (storage : Storage) := match storage with
  | .fp16 => cosineSource%[tl.float16,original]
  | .bf16 => cosineSource%[tl.bfloat16,original]
  | .fp32 => cosineSource%[tl.float32,original]
```
</details>

## Public theorem: `index_commuted_correct`

**Statement:**
```lean
specification index_commuted_correct (storage : Storage) (a b : RegionName) (n B : Nat) :
    Spec.Real (indexCommutedIO storage a b n B ⊨ fun xs i => Real.cos (xs i))
```

**Closed-form spec defs (transitive):** `indexCommutedIO`, `originalIO`, `kernel`, `indexCommuted`, `original`

<details><summary><code>indexCommutedIO</code></summary>

```lean
def indexCommutedIO (storage : Storage) (a b : RegionName) (n B : Nat) : MaskedKernelIO₁ :=
  { originalIO storage a b n B with
    kernel := (kernel storage Bool.true a b n B).eraseDType
    projection := RealProof.projection storage Bool.true a b n B }
```
</details>

<details><summary><code>originalIO</code></summary>

```lean
def originalIO (storage : Storage) (a b : RegionName) (n B : Nat) : MaskedKernelIO₁ where
  kernel := (kernel storage Bool.false a b n B).eraseDType
  projection := RealProof.projection storage Bool.false a b n B
  inp := a
  out := b
  B := B
  read := fun pid => pid*B
  write := fun pid => pid*B
  mask := fun pid j => pid*B+j.val < n
```
</details>

<details><summary><code>kernel</code></summary>

```lean
def kernel (storage : Storage) (swapped : Bool) (a b : RegionName) (n B : Nat) :=
  (if swapped then indexCommuted storage else original storage) a b n B
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (storage : Storage) := match storage with
  | .fp16 => cosineSource%[tl.float16,indexCommuted]
  | .bf16 => cosineSource%[tl.bfloat16,indexCommuted]
  | .fp32 => cosineSource%[tl.float32,indexCommuted]
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (storage : Storage) := match storage with
  | .fp16 => cosineSource%[tl.float16,original]
  | .bf16 => cosineSource%[tl.bfloat16,original]
  | .fp32 => cosineSource%[tl.float32,original]
```
</details>
