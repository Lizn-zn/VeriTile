# Spec sheet — `bench/tritonbench_g/dropout_triton/Correct.lean`

**Python source:** `bench/tritonbench_g/dropout_triton/dropout_triton.py`

## Public theorem: `dropout_correctness`

**Statement:**
```lean
specification dropout_correctness (storage : Storage) (keep : KeepStorage)
    (x k y : RegionName) (n B : Nat) (p : ℝ) :
    Spec.Real (correct_target (original storage keep x k y n B p) keep x k y n B p)
```

**Closed-form spec defs (transitive):** `correct_target`, `original`, `offset`, `expected`, `Outside`, `source`, `keepBit`, `indexCommuted`

<details><summary><code>correct_target</code></summary>

```lean
def correct_target (kernel : ComputeKernel) (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds x k y n B u A.extent →
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B, offset u B j<n →
        v.readMem A.flat (A.addr y (offset u B j))=expected keep u x k p (offset u B j)) ∧
      (∀ r o, Outside A u y n B r o → v.mem r o=(A.flattenState u).mem r o)
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (storage : Storage) (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ) : ComputeKernel :=
  (source storage keep Bool.false x k y n B p).eraseDType
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (u : BlockState) (B : Nat) (j : Fin B) : Nat := u.pid*B+j.val
```
</details>

<details><summary><code>expected</code></summary>

```lean
def expected (keep : KeepStorage) (u : BlockState) (x k : RegionName) (p : ℝ) (o : Nat) : ℝ :=
  if keepBit keep u k o then u.readMem x o/(1-p) else 0

structure Bounds (x k y : RegionName) (n B : Nat) (u : BlockState) (bounds : RegionBounds) : Prop where
  input : ∀ j : Fin B, offset u B j<n → offset u B j<bounds x
  keep : ∀ j : Fin B, offset u B j<n → offset u B j<bounds k
  output : ∀ j : Fin B, offset u B j<n → offset u B j<bounds y
```
</details>

<details><summary><code>Outside</code></summary>

```lean
def Outside (A : FlatAlloc) (u : BlockState) (y : RegionName) (n B : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ j : Fin B, offset u B j<n → o≠A.addr y (offset u B j)
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

<details><summary><code>keepBit</code></summary>

```lean
def keepBit (keep : KeepStorage) (u : BlockState) (r : RegionName) (o : Nat) : Bool :=
  match keep with
  | .int32 => decide (u.readMemValue .int r o≠0)
  | .bool => u.readMemValue .bool r o
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (storage : Storage) (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ) : ComputeKernel :=
  (source storage keep Bool.true x k y n B p).eraseDType
```
</details>

## Public theorem: `index_commuted_correctness`

**Statement:**
```lean
specification index_commuted_correctness (storage : Storage) (keep : KeepStorage)
    (x k y : RegionName) (n B : Nat) (p : ℝ) :
    Spec.Real (correct_target (indexCommuted storage keep x k y n B p) keep x k y n B p)
```

**Closed-form spec defs (transitive):** `correct_target`, `indexCommuted`, `offset`, `expected`, `Outside`, `source`, `keepBit`, `original`

<details><summary><code>correct_target</code></summary>

```lean
def correct_target (kernel : ComputeKernel) (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds x k y n B u A.extent →
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B, offset u B j<n →
        v.readMem A.flat (A.addr y (offset u B j))=expected keep u x k p (offset u B j)) ∧
      (∀ r o, Outside A u y n B r o → v.mem r o=(A.flattenState u).mem r o)
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (storage : Storage) (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ) : ComputeKernel :=
  (source storage keep Bool.true x k y n B p).eraseDType
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (u : BlockState) (B : Nat) (j : Fin B) : Nat := u.pid*B+j.val
```
</details>

<details><summary><code>expected</code></summary>

```lean
def expected (keep : KeepStorage) (u : BlockState) (x k : RegionName) (p : ℝ) (o : Nat) : ℝ :=
  if keepBit keep u k o then u.readMem x o/(1-p) else 0

structure Bounds (x k y : RegionName) (n B : Nat) (u : BlockState) (bounds : RegionBounds) : Prop where
  input : ∀ j : Fin B, offset u B j<n → offset u B j<bounds x
  keep : ∀ j : Fin B, offset u B j<n → offset u B j<bounds k
  output : ∀ j : Fin B, offset u B j<n → offset u B j<bounds y
```
</details>

<details><summary><code>Outside</code></summary>

```lean
def Outside (A : FlatAlloc) (u : BlockState) (y : RegionName) (n B : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ j : Fin B, offset u B j<n → o≠A.addr y (offset u B j)
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

<details><summary><code>keepBit</code></summary>

```lean
def keepBit (keep : KeepStorage) (u : BlockState) (r : RegionName) (o : Nat) : Bool :=
  match keep with
  | .int32 => decide (u.readMemValue .int r o≠0)
  | .bool => u.readMemValue .bool r o
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (storage : Storage) (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ) : ComputeKernel :=
  (source storage keep Bool.false x k y n B p).eraseDType
```
</details>
