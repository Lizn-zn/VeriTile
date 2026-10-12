# Spec sheet — `bench/tritonbench_g/embedding_triton_kernel/Correct.lean`

**Python source:** `bench/tritonbench_g/embedding_triton_kernel/embedding_triton_kernel.py`

## Public theorem: `embedding_correctness`

**Statement:**
```lean
specification embedding_correctness (storage : Storage) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) :
    Spec.Real (correct_target (original storage weight ids out first last sw so n h D N NN)
      weight ids out first last sw so n h D N NN)
```

**Closed-form spec defs (transitive):** `correct_target`, `original`, `row`, `expected`, `Outside`, `source`, `token`, `Written`, `indexCommuted`

<details><summary><code>correct_target</code></summary>

```
/-- Actual finite allocation and successful flat execution are part of the
contract, together with the independent lookup and full tagged-cell frame. -/
```
```lean
def correct_target (kernel : ComputeKernel) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) : Prop :=
  weight≠out → ids≠out → min D h≤so →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds u weight ids out first last sw so n h D N NN A.extent →
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t*NN<N → ∀ jr : Fin NN, ∀ jd : Fin D,
        row u.pid N NN t jr.val<n → jd.val<h →
        v.readMem A.flat (A.addr out (row u.pid N NN t jr.val*so+jd.val))=
          expected u weight ids first last sw (row u.pid N NN t jr.val) jd.val) ∧
      (∀ r o, Outside A u out so n h D N NN r o → v.mem r o=(A.flattenState u).mem r o)
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (storage : Storage) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) : ComputeKernel :=
  (source storage Bool.false weight ids out first last sw so n h D N NN).eraseDType
```
</details>

<details><summary><code>row</code></summary>

```lean
def row (pid N NN t lane : Nat) : Nat := pid*N+t*NN+lane
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Independent mathematical lookup, before any output stores. Negative IDs
are valid exactly when they lie in the declared signed vocabulary interval. -/
```
```lean
def expected (u : BlockState) (weight ids : RegionName) (first last : Int)
    (weightStride r d : Nat) : ℝ :=
  let id := token u ids r
  if first ≤ id ∧ id < last then u.readMem weight ((id-first).toNat*weightStride+d) else 0
```
</details>

<details><summary><code>Outside</code></summary>

```lean
def Outside (A : FlatAlloc) (u : BlockState) (out : RegionName)
    (so n h D N NN : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ z, Written u.pid N NN D n h so z → o≠A.addr out z
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

<details><summary><code>token</code></summary>

```lean
def token (u : BlockState) (ids : RegionName) (r : Nat) : Int := u.readMemValue .int ids r
```
</details>

<details><summary><code>Written</code></summary>

```
/-- All source store locations, including a possible final overshooting chunk. -/
```
```lean
def Written (pid N NN D n hidden outStride : Nat) (o : Nat) : Prop :=
  ∃ t, t*NN<N ∧ ∃ r, r<NN ∧ ∃ d, d<D ∧
    row pid N NN t r<n ∧ d<hidden ∧ o=row pid N NN t r*outStride+d

/-- Only source-active token, weight and output addresses require storage.
Bounds are on initial inputs, independently of any execution result. -/
structure Bounds (u : BlockState) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) (bounds : RegionBounds) : Prop where
  idAccess : ∀ t, t*NN<N → ∀ jr : Fin NN,
    row u.pid N NN t jr.val<n → row u.pid N NN t jr.val<bounds ids
  weightAccess : ∀ t, t*NN<N → ∀ jr : Fin NN, ∀ jd : Fin D,
    row u.pid N NN t jr.val<n → jd.val<h →
    first≤token u ids (row u.pid N NN t jr.val) →
    token u ids (row u.pid N NN t jr.val)<last →
    (token u ids (row u.pid N NN t jr.val)-first).toNat*sw+jd.val<bounds weight
  outputAccess : ∀ t, t*NN<N → ∀ jr : Fin NN, ∀ jd : Fin D,
    row u.pid N NN t jr.val<n → jd.val<h →
    row u.pid N NN t jr.val*so+jd.val<bounds out
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (storage : Storage) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) : ComputeKernel :=
  (source storage Bool.true weight ids out first last sw so n h D N NN).eraseDType
```
</details>

## Public theorem: `index_commuted_correctness`

**Statement:**
```lean
specification index_commuted_correctness (storage : Storage) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) :
    Spec.Real (correct_target (indexCommuted storage weight ids out first last sw so n h D N NN)
      weight ids out first last sw so n h D N NN)
```

**Closed-form spec defs (transitive):** `correct_target`, `indexCommuted`, `row`, `expected`, `Outside`, `source`, `token`, `Written`, `original`

<details><summary><code>correct_target</code></summary>

```
/-- Actual finite allocation and successful flat execution are part of the
contract, together with the independent lookup and full tagged-cell frame. -/
```
```lean
def correct_target (kernel : ComputeKernel) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) : Prop :=
  weight≠out → ids≠out → min D h≤so →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds u weight ids out first last sw so n h D N NN A.extent →
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t*NN<N → ∀ jr : Fin NN, ∀ jd : Fin D,
        row u.pid N NN t jr.val<n → jd.val<h →
        v.readMem A.flat (A.addr out (row u.pid N NN t jr.val*so+jd.val))=
          expected u weight ids first last sw (row u.pid N NN t jr.val) jd.val) ∧
      (∀ r o, Outside A u out so n h D N NN r o → v.mem r o=(A.flattenState u).mem r o)
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (storage : Storage) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) : ComputeKernel :=
  (source storage Bool.true weight ids out first last sw so n h D N NN).eraseDType
```
</details>

<details><summary><code>row</code></summary>

```lean
def row (pid N NN t lane : Nat) : Nat := pid*N+t*NN+lane
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Independent mathematical lookup, before any output stores. Negative IDs
are valid exactly when they lie in the declared signed vocabulary interval. -/
```
```lean
def expected (u : BlockState) (weight ids : RegionName) (first last : Int)
    (weightStride r d : Nat) : ℝ :=
  let id := token u ids r
  if first ≤ id ∧ id < last then u.readMem weight ((id-first).toNat*weightStride+d) else 0
```
</details>

<details><summary><code>Outside</code></summary>

```lean
def Outside (A : FlatAlloc) (u : BlockState) (out : RegionName)
    (so n h D N NN : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ z, Written u.pid N NN D n h so z → o≠A.addr out z
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

<details><summary><code>token</code></summary>

```lean
def token (u : BlockState) (ids : RegionName) (r : Nat) : Int := u.readMemValue .int ids r
```
</details>

<details><summary><code>Written</code></summary>

```
/-- All source store locations, including a possible final overshooting chunk. -/
```
```lean
def Written (pid N NN D n hidden outStride : Nat) (o : Nat) : Prop :=
  ∃ t, t*NN<N ∧ ∃ r, r<NN ∧ ∃ d, d<D ∧
    row pid N NN t r<n ∧ d<hidden ∧ o=row pid N NN t r*outStride+d

/-- Only source-active token, weight and output addresses require storage.
Bounds are on initial inputs, independently of any execution result. -/
structure Bounds (u : BlockState) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) (bounds : RegionBounds) : Prop where
  idAccess : ∀ t, t*NN<N → ∀ jr : Fin NN,
    row u.pid N NN t jr.val<n → row u.pid N NN t jr.val<bounds ids
  weightAccess : ∀ t, t*NN<N → ∀ jr : Fin NN, ∀ jd : Fin D,
    row u.pid N NN t jr.val<n → jd.val<h →
    first≤token u ids (row u.pid N NN t jr.val) →
    token u ids (row u.pid N NN t jr.val)<last →
    (token u ids (row u.pid N NN t jr.val)-first).toNat*sw+jd.val<bounds weight
  outputAccess : ∀ t, t*NN<N → ∀ jr : Fin NN, ∀ jd : Fin D,
    row u.pid N NN t jr.val<n → jd.val<h →
    row u.pid N NN t jr.val*so+jd.val<bounds out
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (storage : Storage) (weight ids out : RegionName) (first last : Int)
    (sw so n h D N NN : Nat) : ComputeKernel :=
  (source storage Bool.false weight ids out first last sw so n h D N NN).eraseDType
```
</details>
