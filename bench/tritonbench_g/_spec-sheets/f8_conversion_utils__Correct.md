# Spec sheet — `bench/tritonbench_g/f8_conversion_utils/Correct.lean`

**Python source:** `bench/tritonbench_g/f8_conversion_utils/f8_conversion_utils.py`

## Public theorem: `f8_to_f16_correctness`

**Statement:**
```lean
specification f8_to_f16_correctness (format : Format) (Y X : RegionName) (N B : Nat) :
    Spec.Real (correct_target (originalToHalf format Y X N B) Y X N B)
```

**Closed-form spec defs (transitive):** `correct_target`, `originalToHalf`, `offset`, `Outside`, `toHalf`

<details><summary><code>correct_target</code></summary>

```
/-- Erased real copying: actual finite bounds, successful full execution,
initial-input values and the full tagged-cell frame. -/
```
```lean
def correct_target (kernel : ComputeKernel) (Y X : RegionName) (N B : Nat) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds Y X N B u A.extent →
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B, offset u B j<N → v.readMem A.flat (A.addr Y (offset u B j))=u.readMem X (offset u B j)) ∧
      (∀ r o, Outside A u Y N B r o → v.mem r o=(A.flattenState u).mem r o)
```
</details>

<details><summary><code>originalToHalf</code></summary>

```lean
def originalToHalf (format : Format) (Y X : RegionName) (N B : Nat) : ComputeKernel :=
  (toHalf format Bool.false Y X N B).eraseDType
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (u : BlockState) (B : Nat) (j : Fin B) : Nat := u.pid*B+j.val
```
</details>

<details><summary><code>Outside</code></summary>

```lean
def Outside (A : FlatAlloc) (u : BlockState) (Y : RegionName) (N B : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ j : Fin B, offset u B j<N → o≠A.addr Y (offset u B j)
```
</details>

<details><summary><code>toHalf</code></summary>

```lean
def toHalf (format : Format) (swapped : Bool) : sourceType :=
  match format,swapped with
  | .e4,Bool.false => f8Forward%[tl.float8e4nv,tl.float16,original]
  | .e4,Bool.true => f8Forward%[tl.float8e4nv,tl.float16,indexCommuted]
  | .e5,Bool.false => f8Forward%[tl.float8e5,tl.float16,original]
  | .e5,Bool.true => f8Forward%[tl.float8e5,tl.float16,indexCommuted]
```
</details>

## Public theorem: `f8_to_f16_index_commuted_correctness`

**Statement:**
```lean
specification f8_to_f16_index_commuted_correctness (format : Format) (Y X : RegionName) (N B : Nat) :
    Spec.Real (correct_target (indexCommutedToHalf format Y X N B) Y X N B)
```

**Closed-form spec defs (transitive):** `correct_target`, `indexCommutedToHalf`, `offset`, `Outside`, `toHalf`

<details><summary><code>correct_target</code></summary>

```
/-- Erased real copying: actual finite bounds, successful full execution,
initial-input values and the full tagged-cell frame. -/
```
```lean
def correct_target (kernel : ComputeKernel) (Y X : RegionName) (N B : Nat) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds Y X N B u A.extent →
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B, offset u B j<N → v.readMem A.flat (A.addr Y (offset u B j))=u.readMem X (offset u B j)) ∧
      (∀ r o, Outside A u Y N B r o → v.mem r o=(A.flattenState u).mem r o)
```
</details>

<details><summary><code>indexCommutedToHalf</code></summary>

```lean
def indexCommutedToHalf (format : Format) (Y X : RegionName) (N B : Nat) : ComputeKernel :=
  (toHalf format Bool.true Y X N B).eraseDType
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (u : BlockState) (B : Nat) (j : Fin B) : Nat := u.pid*B+j.val
```
</details>

<details><summary><code>Outside</code></summary>

```lean
def Outside (A : FlatAlloc) (u : BlockState) (Y : RegionName) (N B : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ j : Fin B, offset u B j<N → o≠A.addr Y (offset u B j)
```
</details>

<details><summary><code>toHalf</code></summary>

```lean
def toHalf (format : Format) (swapped : Bool) : sourceType :=
  match format,swapped with
  | .e4,Bool.false => f8Forward%[tl.float8e4nv,tl.float16,original]
  | .e4,Bool.true => f8Forward%[tl.float8e4nv,tl.float16,indexCommuted]
  | .e5,Bool.false => f8Forward%[tl.float8e5,tl.float16,original]
  | .e5,Bool.true => f8Forward%[tl.float8e5,tl.float16,indexCommuted]
```
</details>

## Public theorem: `to_fp8_correctness`

**Statement:**
```lean
specification to_fp8_correctness (input : Input) (format : Format) (Y X : RegionName) (N B : Nat) :
    Spec.Real (correct_target (originalToFP8 input format Y X N B) Y X N B)
```

**Closed-form spec defs (transitive):** `correct_target`, `originalToFP8`, `offset`, `Outside`, `toFP8`

<details><summary><code>correct_target</code></summary>

```
/-- Erased real copying: actual finite bounds, successful full execution,
initial-input values and the full tagged-cell frame. -/
```
```lean
def correct_target (kernel : ComputeKernel) (Y X : RegionName) (N B : Nat) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds Y X N B u A.extent →
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B, offset u B j<N → v.readMem A.flat (A.addr Y (offset u B j))=u.readMem X (offset u B j)) ∧
      (∀ r o, Outside A u Y N B r o → v.mem r o=(A.flattenState u).mem r o)
```
</details>

<details><summary><code>originalToFP8</code></summary>

```lean
def originalToFP8 (input : Input) (format : Format) (Y X : RegionName) (N B : Nat) : ComputeKernel :=
  (toFP8 input format Bool.false Y X N B).eraseDType
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (u : BlockState) (B : Nat) (j : Fin B) : Nat := u.pid*B+j.val
```
</details>

<details><summary><code>Outside</code></summary>

```lean
def Outside (A : FlatAlloc) (u : BlockState) (Y : RegionName) (N B : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ j : Fin B, offset u B j<N → o≠A.addr Y (offset u B j)
```
</details>

<details><summary><code>toFP8</code></summary>

```lean
def toFP8 (input : Input) (format : Format) (swapped : Bool) : sourceType :=
  match input,format,swapped with
  | .fp16,.e4,Bool.false => f8Reverse%[tl.float16,tl.float8e4nv,original]
  | .fp16,.e4,Bool.true => f8Reverse%[tl.float16,tl.float8e4nv,indexCommuted]
  | .fp16,.e5,Bool.false => f8Reverse%[tl.float16,tl.float8e5,original]
  | .fp16,.e5,Bool.true => f8Reverse%[tl.float16,tl.float8e5,indexCommuted]
  | .fp32,.e4,Bool.false => f8Reverse%[tl.float32,tl.float8e4nv,original]
  | .fp32,.e4,Bool.true => f8Reverse%[tl.float32,tl.float8e4nv,indexCommuted]
  | .fp32,.e5,Bool.false => f8Reverse%[tl.float32,tl.float8e5,original]
  | .fp32,.e5,Bool.true => f8Reverse%[tl.float32,tl.float8e5,indexCommuted]
```
</details>

## Public theorem: `to_fp8_index_commuted_correctness`

**Statement:**
```lean
specification to_fp8_index_commuted_correctness (input : Input) (format : Format) (Y X : RegionName) (N B : Nat) :
    Spec.Real (correct_target (indexCommutedToFP8 input format Y X N B) Y X N B)
```

**Closed-form spec defs (transitive):** `correct_target`, `indexCommutedToFP8`, `offset`, `Outside`, `toFP8`

<details><summary><code>correct_target</code></summary>

```
/-- Erased real copying: actual finite bounds, successful full execution,
initial-input values and the full tagged-cell frame. -/
```
```lean
def correct_target (kernel : ComputeKernel) (Y X : RegionName) (N B : Nat) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds Y X N B u A.extent →
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ j : Fin B, offset u B j<N → v.readMem A.flat (A.addr Y (offset u B j))=u.readMem X (offset u B j)) ∧
      (∀ r o, Outside A u Y N B r o → v.mem r o=(A.flattenState u).mem r o)
```
</details>

<details><summary><code>indexCommutedToFP8</code></summary>

```lean
def indexCommutedToFP8 (input : Input) (format : Format) (Y X : RegionName) (N B : Nat) : ComputeKernel :=
  (toFP8 input format Bool.true Y X N B).eraseDType

structure Bounds (Y X : RegionName) (N B : Nat) (u : BlockState) (bounds : RegionBounds) : Prop where
  input : ∀ j : Fin B, offset u B j<N → offset u B j<bounds X
  output : ∀ j : Fin B, offset u B j<N → offset u B j<bounds Y
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (u : BlockState) (B : Nat) (j : Fin B) : Nat := u.pid*B+j.val
```
</details>

<details><summary><code>Outside</code></summary>

```lean
def Outside (A : FlatAlloc) (u : BlockState) (Y : RegionName) (N B : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ j : Fin B, offset u B j<N → o≠A.addr Y (offset u B j)
```
</details>

<details><summary><code>toFP8</code></summary>

```lean
def toFP8 (input : Input) (format : Format) (swapped : Bool) : sourceType :=
  match input,format,swapped with
  | .fp16,.e4,Bool.false => f8Reverse%[tl.float16,tl.float8e4nv,original]
  | .fp16,.e4,Bool.true => f8Reverse%[tl.float16,tl.float8e4nv,indexCommuted]
  | .fp16,.e5,Bool.false => f8Reverse%[tl.float16,tl.float8e5,original]
  | .fp16,.e5,Bool.true => f8Reverse%[tl.float16,tl.float8e5,indexCommuted]
  | .fp32,.e4,Bool.false => f8Reverse%[tl.float32,tl.float8e4nv,original]
  | .fp32,.e4,Bool.true => f8Reverse%[tl.float32,tl.float8e4nv,indexCommuted]
  | .fp32,.e5,Bool.false => f8Reverse%[tl.float32,tl.float8e5,original]
  | .fp32,.e5,Bool.true => f8Reverse%[tl.float32,tl.float8e5,indexCommuted]
```
</details>
