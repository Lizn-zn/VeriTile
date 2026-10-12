# Spec sheet — `bench/tritonbench_g/fast_rope_embedding/Correct.lean`

**Python source:** `bench/tritonbench_g/fast_rope_embedding/fast_rope_embedding.py`

## Public theorem: `original_correctness`

**Statement:**
```lean
specification original_correctness (input cosine sine : Storage) (Q C S : RegionName)
    (QR CR SR seq H heads B : Nat) (backwards : Bool) :
    Spec.Real (correct_target (rope input cosine sine Bool.false Q C S QR CR SR seq H heads B backwards).eraseDType.toAlgKernel
      Q C S QR CR SR seq H heads B backwards)
```

**Closed-form spec defs (transitive):** `correct_target`, `rope`, `start`, `stop`, `first`, `firstValue`, `second`, `coefficientAddress`, `secondValue`, `Outside`, `signedSine`

<details><summary><code>correct_target</code></summary>

```lean
def correct_target (kernel : Kernel) (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (backwards : Bool) : Prop :=
  ∀ A : FlatAlloc,A.Disjoint → (∀ r,r∉A.regions → A.extent r=0) →
  ∀ u : BlockState,u.undef=(fun _ _ => 0) → Bounds u Q C S QR CR SR seq H heads B A.extent →
  0<seq → H/2≤B →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ k,start (u.pids 1)≤k → k<stop (u.pids 1) heads →
        ∀ j : Fin B,j.val<H/2 →
        v.readMem A.flat (A.addr Q (first u.pid QR H k j.val))=
          firstValue backwards
            (u.readMem Q (first u.pid QR H k j.val))
            (u.readMem Q (second u.pid QR H k j.val))
            (u.readMem C (coefficientAddress u.pid CR seq j.val))
            (u.readMem S (coefficientAddress u.pid SR seq j.val)) ∧
        v.readMem A.flat (A.addr Q (second u.pid QR H k j.val))=
          secondValue backwards
            (u.readMem Q (first u.pid QR H k j.val))
            (u.readMem Q (second u.pid QR H k j.val))
            (u.readMem C (coefficientAddress u.pid CR seq j.val))
            (u.readMem S (coefficientAddress u.pid SR seq j.val))) ∧
      (∀ r o,Outside A u Q QR H heads B r o → v.mem r o=(A.flattenState u).mem r o)
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

<details><summary><code>start</code></summary>

```lean
def start (group : Nat) := group*4
```
</details>

<details><summary><code>stop</code></summary>

```lean
def stop (group heads : Nat) := min (start group+4) heads

structure Bounds (u : BlockState) (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (bounds : RegionBounds) : Prop where
  cosine : ∀ j : Fin B,j.val<H/2 → coefficientAddress u.pid CR seq j.val<bounds C
  sine : ∀ j : Fin B,j.val<H/2 → coefficientAddress u.pid SR seq j.val<bounds S
  first : ∀ k,start (u.pids 1)≤k → k<stop (u.pids 1) heads →
    ∀ j : Fin B,j.val<H/2 → first u.pid QR H k j.val<bounds Q
  second : ∀ k,start (u.pids 1)≤k → k<stop (u.pids 1) heads →
    ∀ j : Fin B,j.val<H/2 → second u.pid QR H k j.val<bounds Q
```
</details>

<details><summary><code>first</code></summary>

```lean
def first (row QR H head lane : Nat) := row*QR+head*H+lane
```
</details>

<details><summary><code>firstValue</code></summary>

```lean
def firstValue (backwards : Bool) (a b c s : ℝ) := a*c-b*signedSine backwards s
```
</details>

<details><summary><code>second</code></summary>

```lean
def second (row QR H head lane : Nat) := first row QR H head lane+H/2
```
</details>

<details><summary><code>coefficientAddress</code></summary>

```lean
def coefficientAddress (row stride seq lane : Nat) := (row%seq)*stride+lane
```
</details>

<details><summary><code>secondValue</code></summary>

```lean
def secondValue (backwards : Bool) (a b c s : ℝ) := b*c+a*signedSine backwards s
```
</details>

<details><summary><code>Outside</code></summary>

```lean
def Outside (A : FlatAlloc) (u : BlockState) (Q : RegionName) (QR H heads B : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ k,start (u.pids 1)≤k → k<stop (u.pids 1) heads →
    ∀ j : Fin B,j.val<H/2 →
      o≠A.addr Q (first u.pid QR H k j.val) ∧ o≠A.addr Q (second u.pid QR H k j.val)
```
</details>

<details><summary><code>signedSine</code></summary>

```lean
def signedSine (backwards : Bool) (x : ℝ) := if backwards then -x else x
```
</details>

## Public theorem: `index_commuted_correctness`

**Statement:**
```lean
specification index_commuted_correctness (input cosine sine : Storage) (Q C S : RegionName)
    (QR CR SR seq H heads B : Nat) (backwards : Bool) :
    Spec.Real (correct_target (rope input cosine sine Bool.true Q C S QR CR SR seq H heads B backwards).eraseDType.toAlgKernel
      Q C S QR CR SR seq H heads B backwards)
```

**Closed-form spec defs (transitive):** `correct_target`, `rope`, `start`, `stop`, `first`, `firstValue`, `second`, `coefficientAddress`, `secondValue`, `Outside`, `signedSine`

<details><summary><code>correct_target</code></summary>

```lean
def correct_target (kernel : Kernel) (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (backwards : Bool) : Prop :=
  ∀ A : FlatAlloc,A.Disjoint → (∀ r,r∉A.regions → A.extent r=0) →
  ∀ u : BlockState,u.undef=(fun _ _ => 0) → Bounds u Q C S QR CR SR seq H heads B A.extent →
  0<seq → H/2≤B →
    ∃ v,exec (A.flattenKernel kernel) (A.flattenState u)=some v ∧
      (∀ k,start (u.pids 1)≤k → k<stop (u.pids 1) heads →
        ∀ j : Fin B,j.val<H/2 →
        v.readMem A.flat (A.addr Q (first u.pid QR H k j.val))=
          firstValue backwards
            (u.readMem Q (first u.pid QR H k j.val))
            (u.readMem Q (second u.pid QR H k j.val))
            (u.readMem C (coefficientAddress u.pid CR seq j.val))
            (u.readMem S (coefficientAddress u.pid SR seq j.val)) ∧
        v.readMem A.flat (A.addr Q (second u.pid QR H k j.val))=
          secondValue backwards
            (u.readMem Q (first u.pid QR H k j.val))
            (u.readMem Q (second u.pid QR H k j.val))
            (u.readMem C (coefficientAddress u.pid CR seq j.val))
            (u.readMem S (coefficientAddress u.pid SR seq j.val))) ∧
      (∀ r o,Outside A u Q QR H heads B r o → v.mem r o=(A.flattenState u).mem r o)
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

<details><summary><code>start</code></summary>

```lean
def start (group : Nat) := group*4
```
</details>

<details><summary><code>stop</code></summary>

```lean
def stop (group heads : Nat) := min (start group+4) heads

structure Bounds (u : BlockState) (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (bounds : RegionBounds) : Prop where
  cosine : ∀ j : Fin B,j.val<H/2 → coefficientAddress u.pid CR seq j.val<bounds C
  sine : ∀ j : Fin B,j.val<H/2 → coefficientAddress u.pid SR seq j.val<bounds S
  first : ∀ k,start (u.pids 1)≤k → k<stop (u.pids 1) heads →
    ∀ j : Fin B,j.val<H/2 → first u.pid QR H k j.val<bounds Q
  second : ∀ k,start (u.pids 1)≤k → k<stop (u.pids 1) heads →
    ∀ j : Fin B,j.val<H/2 → second u.pid QR H k j.val<bounds Q
```
</details>

<details><summary><code>first</code></summary>

```lean
def first (row QR H head lane : Nat) := row*QR+head*H+lane
```
</details>

<details><summary><code>firstValue</code></summary>

```lean
def firstValue (backwards : Bool) (a b c s : ℝ) := a*c-b*signedSine backwards s
```
</details>

<details><summary><code>second</code></summary>

```lean
def second (row QR H head lane : Nat) := first row QR H head lane+H/2
```
</details>

<details><summary><code>coefficientAddress</code></summary>

```lean
def coefficientAddress (row stride seq lane : Nat) := (row%seq)*stride+lane
```
</details>

<details><summary><code>secondValue</code></summary>

```lean
def secondValue (backwards : Bool) (a b c s : ℝ) := b*c+a*signedSine backwards s
```
</details>

<details><summary><code>Outside</code></summary>

```lean
def Outside (A : FlatAlloc) (u : BlockState) (Q : RegionName) (QR H heads B : Nat) (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨ ∀ k,start (u.pids 1)≤k → k<stop (u.pids 1) heads →
    ∀ j : Fin B,j.val<H/2 →
      o≠A.addr Q (first u.pid QR H k j.val) ∧ o≠A.addr Q (second u.pid QR H k j.val)
```
</details>

<details><summary><code>signedSine</code></summary>

```lean
def signedSine (backwards : Bool) (x : ℝ) := if backwards then -x else x
```
</details>
