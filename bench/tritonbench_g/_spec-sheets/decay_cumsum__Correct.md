# Spec sheet — `bench/tritonbench_g/decay_cumsum/Correct.lean`

**Python source:** `bench/tritonbench_g/decay_cumsum/decay_cumsum.py`

## Public theorem: `forward_original_correct`

**Statement:**
```lean
specification forward_original_correct (storage : Storage) (g go : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ)
    (hne : g≠go) (hBK : BK≤DK) :
    Spec.Real (originalIO storage g go sH sT sD B H T BT BK DK scale ⊨
      fun _ _ _ xs t j => scan BT BK xs t j)
```

**Assumptions / layout contracts:**
- `hne : g≠go`
- `hBK : BK≤DK`

**Closed-form spec defs (transitive):** `originalIO`, `scan`, `backward`

<details><summary><code>originalIO</code></summary>

```
/-- Seven initial input streams and three output streams. DQInter/DKInter are
updated in place; the complete contract includes the frame of every other cell. -/
```
```lean
def originalIO (source gate : Storage)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ) : StreamGroupedEmitMasked3DKernelIO where
  kernel := (backward source gate Bool.false DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T BT BK DK scale).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType,BackwardCorrect.source_projects,
      ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody]
  nIn := 7
  nOut := 3
  bufs := [DQInner, DQInter, DKInner, DKInter, Q, K, G, DG]
  inp := fun i => match i with
    | ⟨0, _⟩ => G
    | ⟨1, _⟩ => DQInner
    | ⟨2, _⟩ => DQInter
    | ⟨3, _⟩ => DKInner
    | ⟨4, _⟩ => DKInter
    | ⟨5, _⟩ => Q
    | ⟨6, _⟩ => K
    | ⟨n + 7, h⟩ => absurd h (by omega)
  out := fun o => match o with
    | ⟨0, _⟩ => DQInter
    | ⟨1, _⟩ => DKInter
    | ⟨2, _⟩ => DG
    | ⟨n + 3, h⟩ => absurd h (by omega)
  T := BT
  B := BK
  read := fun _ p₀ p₁ p₂ t j =>
    p₂ * sH + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  readMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK
  write := fun _ p₀ p₁ p₂ t j =>
    p₂ * sH + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  writeMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK
```
</details>

<details><summary><code>scan</code></summary>

```
/-- Independent prefix sum of the initially loaded rows. -/
```
```lean
def scan (BT BK : Nat) (xs : Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  1.44269504 * ∑ k : Fin (t.val+1), xs (Fin.castLE t.isLt k) j
```
</details>

<details><summary><code>backward</code></summary>

```lean
def backward (source gate : Storage) (swapped : Bool) := match source, gate with
  | .fp16, .fp16 => if swapped then decayBackwardSource%[tl.float16,tl.float16,indexCommuted] else decayBackwardSource%[tl.float16,tl.float16,original]
  | .fp16, .bf16 => if swapped then decayBackwardSource%[tl.float16,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.float16,tl.bfloat16,original]
  | .fp16, .fp32 => if swapped then decayBackwardSource%[tl.float16,tl.float32,indexCommuted] else decayBackwardSource%[tl.float16,tl.float32,original]
  | .bf16, .fp16 => if swapped then decayBackwardSource%[tl.bfloat16,tl.float16,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.float16,original]
  | .bf16, .bf16 => if swapped then decayBackwardSource%[tl.bfloat16,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.bfloat16,original]
  | .bf16, .fp32 => if swapped then decayBackwardSource%[tl.bfloat16,tl.float32,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.float32,original]
  | .fp32, .fp16 => if swapped then decayBackwardSource%[tl.float32,tl.float16,indexCommuted] else decayBackwardSource%[tl.float32,tl.float16,original]
  | .fp32, .bf16 => if swapped then decayBackwardSource%[tl.float32,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.float32,tl.bfloat16,original]
  | .fp32, .fp32 => if swapped then decayBackwardSource%[tl.float32,tl.float32,indexCommuted] else decayBackwardSource%[tl.float32,tl.float32,original]
```
</details>

## Public theorem: `forward_index_commuted_correct`

**Statement:**
```lean
specification forward_index_commuted_correct (storage : Storage) (g go : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ)
    (hne : g≠go) (hBK : BK≤DK) :
    Spec.Real (indexCommutedIO storage g go sH sT sD B H T BT BK DK scale ⊨
      fun _ _ _ xs t j => scan BT BK xs t j)
```

**Assumptions / layout contracts:**
- `hne : g≠go`
- `hBK : BK≤DK`

**Closed-form spec defs (transitive):** `indexCommutedIO`, `scan`, `originalIO`, `backward`

<details><summary><code>indexCommutedIO</code></summary>

```lean
def indexCommutedIO (source gate : Storage) (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ) : StreamGroupedEmitMasked3DKernelIO :=
  { originalIO source gate DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T DK BT BK scale with
    kernel := (backward source gate Bool.true DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T BT BK DK scale).eraseDType
    projection := by
      simp only [ComputeKernel.eraseDType,BackwardCorrect.source_projects,
        ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody] }
```
</details>

<details><summary><code>scan</code></summary>

```
/-- Independent prefix sum of the initially loaded rows. -/
```
```lean
def scan (BT BK : Nat) (xs : Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  1.44269504 * ∑ k : Fin (t.val+1), xs (Fin.castLE t.isLt k) j
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- Seven initial input streams and three output streams. DQInter/DKInter are
updated in place; the complete contract includes the frame of every other cell. -/
```
```lean
def originalIO (source gate : Storage)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ) : StreamGroupedEmitMasked3DKernelIO where
  kernel := (backward source gate Bool.false DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T BT BK DK scale).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType,BackwardCorrect.source_projects,
      ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody]
  nIn := 7
  nOut := 3
  bufs := [DQInner, DQInter, DKInner, DKInter, Q, K, G, DG]
  inp := fun i => match i with
    | ⟨0, _⟩ => G
    | ⟨1, _⟩ => DQInner
    | ⟨2, _⟩ => DQInter
    | ⟨3, _⟩ => DKInner
    | ⟨4, _⟩ => DKInter
    | ⟨5, _⟩ => Q
    | ⟨6, _⟩ => K
    | ⟨n + 7, h⟩ => absurd h (by omega)
  out := fun o => match o with
    | ⟨0, _⟩ => DQInter
    | ⟨1, _⟩ => DKInter
    | ⟨2, _⟩ => DG
    | ⟨n + 3, h⟩ => absurd h (by omega)
  T := BT
  B := BK
  read := fun _ p₀ p₁ p₂ t j =>
    p₂ * sH + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  readMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK
  write := fun _ p₀ p₁ p₂ t j =>
    p₂ * sH + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  writeMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK
```
</details>

<details><summary><code>backward</code></summary>

```lean
def backward (source gate : Storage) (swapped : Bool) := match source, gate with
  | .fp16, .fp16 => if swapped then decayBackwardSource%[tl.float16,tl.float16,indexCommuted] else decayBackwardSource%[tl.float16,tl.float16,original]
  | .fp16, .bf16 => if swapped then decayBackwardSource%[tl.float16,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.float16,tl.bfloat16,original]
  | .fp16, .fp32 => if swapped then decayBackwardSource%[tl.float16,tl.float32,indexCommuted] else decayBackwardSource%[tl.float16,tl.float32,original]
  | .bf16, .fp16 => if swapped then decayBackwardSource%[tl.bfloat16,tl.float16,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.float16,original]
  | .bf16, .bf16 => if swapped then decayBackwardSource%[tl.bfloat16,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.bfloat16,original]
  | .bf16, .fp32 => if swapped then decayBackwardSource%[tl.bfloat16,tl.float32,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.float32,original]
  | .fp32, .fp16 => if swapped then decayBackwardSource%[tl.float32,tl.float16,indexCommuted] else decayBackwardSource%[tl.float32,tl.float16,original]
  | .fp32, .bf16 => if swapped then decayBackwardSource%[tl.float32,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.float32,tl.bfloat16,original]
  | .fp32, .fp32 => if swapped then decayBackwardSource%[tl.float32,tl.float32,indexCommuted] else decayBackwardSource%[tl.float32,tl.float32,original]
```
</details>

## Public theorem: `prepare_original_correct`

**Statement:**
```lean
specification prepare_original_correct (source gate : Storage) (Q K G QG KG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ)
    (hQ_QG : Q≠QG) (hQ_KG : Q≠KG) (hK_QG : K≠QG) (hK_KG : K≠KG)
    (hG_QG : G≠QG) (hG_KG : G≠KG) (hQG_KG : QG≠KG)
    (hBK : BK≤DK) (hBT : 0<BT) :
    Spec.Real (originalIO source gate Q K G QG KG sH sT sD B H T DK BT BK scale ⊨
      fun _ _ _ xs o t j => values BT BK scale xs o t j)
```

**Assumptions / layout contracts:**
- `hQ_QG : Q≠QG`
- `hQ_KG : Q≠KG`
- `hK_QG : K≠QG`
- `hK_KG : K≠KG`
- `hG_QG : G≠QG`
- `hG_KG : G≠KG`
- `hQG_KG : QG≠KG`
- `hBK : BK≤DK`
- `hBT : 0<BT`

**Closed-form spec defs (transitive):** `originalIO`, `values`, `backward`, `dq`, `dk`, `dg`

<details><summary><code>originalIO</code></summary>

```
/-- Seven initial input streams and three output streams. DQInter/DKInter are
updated in place; the complete contract includes the frame of every other cell. -/
```
```lean
def originalIO (source gate : Storage)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ) : StreamGroupedEmitMasked3DKernelIO where
  kernel := (backward source gate Bool.false DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T BT BK DK scale).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType,BackwardCorrect.source_projects,
      ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody]
  nIn := 7
  nOut := 3
  bufs := [DQInner, DQInter, DKInner, DKInter, Q, K, G, DG]
  inp := fun i => match i with
    | ⟨0, _⟩ => G
    | ⟨1, _⟩ => DQInner
    | ⟨2, _⟩ => DQInter
    | ⟨3, _⟩ => DKInner
    | ⟨4, _⟩ => DKInter
    | ⟨5, _⟩ => Q
    | ⟨6, _⟩ => K
    | ⟨n + 7, h⟩ => absurd h (by omega)
  out := fun o => match o with
    | ⟨0, _⟩ => DQInter
    | ⟨1, _⟩ => DKInter
    | ⟨2, _⟩ => DG
    | ⟨n + 3, h⟩ => absurd h (by omega)
  T := BT
  B := BK
  read := fun _ p₀ p₁ p₂ t j =>
    p₂ * sH + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  readMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK
  write := fun _ p₀ p₁ p₂ t j =>
    p₂ * sH + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  writeMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK
```
</details>

<details><summary><code>values</code></summary>

```lean
noncomputable def values (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (o : Fin 3) (t : Fin BT) (j : Fin BK) : ℝ :=
  match o with
  | ⟨0, _⟩ => dq BT BK xs t j
  | ⟨1, _⟩ => dk BT BK xs t j
  | ⟨2, _⟩ => dg BT BK xs t j
  | ⟨n + 3, h3⟩ => absurd h3 (by omega)
```
</details>

<details><summary><code>backward</code></summary>

```lean
def backward (source gate : Storage) (swapped : Bool) := match source, gate with
  | .fp16, .fp16 => if swapped then decayBackwardSource%[tl.float16,tl.float16,indexCommuted] else decayBackwardSource%[tl.float16,tl.float16,original]
  | .fp16, .bf16 => if swapped then decayBackwardSource%[tl.float16,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.float16,tl.bfloat16,original]
  | .fp16, .fp32 => if swapped then decayBackwardSource%[tl.float16,tl.float32,indexCommuted] else decayBackwardSource%[tl.float16,tl.float32,original]
  | .bf16, .fp16 => if swapped then decayBackwardSource%[tl.bfloat16,tl.float16,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.float16,original]
  | .bf16, .bf16 => if swapped then decayBackwardSource%[tl.bfloat16,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.bfloat16,original]
  | .bf16, .fp32 => if swapped then decayBackwardSource%[tl.bfloat16,tl.float32,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.float32,original]
  | .fp32, .fp16 => if swapped then decayBackwardSource%[tl.float32,tl.float16,indexCommuted] else decayBackwardSource%[tl.float32,tl.float16,original]
  | .fp32, .bf16 => if swapped then decayBackwardSource%[tl.float32,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.float32,tl.bfloat16,original]
  | .fp32, .fp32 => if swapped then decayBackwardSource%[tl.float32,tl.float32,indexCommuted] else decayBackwardSource%[tl.float32,tl.float32,original]
```
</details>

<details><summary><code>dq</code></summary>

```lean
noncomputable def dq (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  xs (⟨1, by omega⟩ : Fin 7) t j
    + xs (⟨2, by omega⟩ : Fin 7) t j
      * Real.exp (xs (⟨0, by omega⟩ : Fin 7) t j * Real.log 2)
```
</details>

<details><summary><code>dk</code></summary>

```lean
noncomputable def dk (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  if h : 0 < BT then
    xs (⟨3, by omega⟩ : Fin 7) t j
      + xs (⟨4, by omega⟩ : Fin 7) t j
        * Real.exp
            ((xs (⟨0, by omega⟩ : Fin 7) ⟨BT - 1, Nat.sub_lt h Nat.one_pos⟩ j
                - xs (⟨0, by omega⟩ : Fin 7) t j) * Real.log 2)
  else 0
```
</details>

<details><summary><code>dg</code></summary>

```lean
noncomputable def dg (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  ∑ d : Fin (BT - t.val),
    (dq BT BK xs ⟨t.val + d.val, by omega⟩ j
        * xs (⟨5, by omega⟩ : Fin 7) ⟨t.val + d.val, by omega⟩ j
      - dk BT BK xs ⟨t.val + d.val, by omega⟩ j
        * xs (⟨6, by omega⟩ : Fin 7) ⟨t.val + d.val, by omega⟩ j)
```
</details>

## Public theorem: `prepare_index_commuted_correct`

**Statement:**
```lean
specification prepare_index_commuted_correct (source gate : Storage) (Q K G QG KG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ)
    (hQ_QG : Q≠QG) (hQ_KG : Q≠KG) (hK_QG : K≠QG) (hK_KG : K≠KG)
    (hG_QG : G≠QG) (hG_KG : G≠KG) (hQG_KG : QG≠KG)
    (hBK : BK≤DK) (hBT : 0<BT) :
    Spec.Real (indexCommutedIO source gate Q K G QG KG sH sT sD B H T DK BT BK scale ⊨
      fun _ _ _ xs o t j => values BT BK scale xs o t j)
```

**Assumptions / layout contracts:**
- `hQ_QG : Q≠QG`
- `hQ_KG : Q≠KG`
- `hK_QG : K≠QG`
- `hK_KG : K≠KG`
- `hG_QG : G≠QG`
- `hG_KG : G≠KG`
- `hQG_KG : QG≠KG`
- `hBK : BK≤DK`
- `hBT : 0<BT`

**Closed-form spec defs (transitive):** `indexCommutedIO`, `values`, `originalIO`, `backward`, `dq`, `dk`, `dg`

<details><summary><code>indexCommutedIO</code></summary>

```lean
def indexCommutedIO (source gate : Storage) (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ) : StreamGroupedEmitMasked3DKernelIO :=
  { originalIO source gate DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T DK BT BK scale with
    kernel := (backward source gate Bool.true DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T BT BK DK scale).eraseDType
    projection := by
      simp only [ComputeKernel.eraseDType,BackwardCorrect.source_projects,
        ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody] }
```
</details>

<details><summary><code>values</code></summary>

```lean
noncomputable def values (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (o : Fin 3) (t : Fin BT) (j : Fin BK) : ℝ :=
  match o with
  | ⟨0, _⟩ => dq BT BK xs t j
  | ⟨1, _⟩ => dk BT BK xs t j
  | ⟨2, _⟩ => dg BT BK xs t j
  | ⟨n + 3, h3⟩ => absurd h3 (by omega)
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- Seven initial input streams and three output streams. DQInter/DKInter are
updated in place; the complete contract includes the frame of every other cell. -/
```
```lean
def originalIO (source gate : Storage)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ) : StreamGroupedEmitMasked3DKernelIO where
  kernel := (backward source gate Bool.false DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T BT BK DK scale).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType,BackwardCorrect.source_projects,
      ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody]
  nIn := 7
  nOut := 3
  bufs := [DQInner, DQInter, DKInner, DKInter, Q, K, G, DG]
  inp := fun i => match i with
    | ⟨0, _⟩ => G
    | ⟨1, _⟩ => DQInner
    | ⟨2, _⟩ => DQInter
    | ⟨3, _⟩ => DKInner
    | ⟨4, _⟩ => DKInter
    | ⟨5, _⟩ => Q
    | ⟨6, _⟩ => K
    | ⟨n + 7, h⟩ => absurd h (by omega)
  out := fun o => match o with
    | ⟨0, _⟩ => DQInter
    | ⟨1, _⟩ => DKInter
    | ⟨2, _⟩ => DG
    | ⟨n + 3, h⟩ => absurd h (by omega)
  T := BT
  B := BK
  read := fun _ p₀ p₁ p₂ t j =>
    p₂ * sH + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  readMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK
  write := fun _ p₀ p₁ p₂ t j =>
    p₂ * sH + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  writeMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK
```
</details>

<details><summary><code>backward</code></summary>

```lean
def backward (source gate : Storage) (swapped : Bool) := match source, gate with
  | .fp16, .fp16 => if swapped then decayBackwardSource%[tl.float16,tl.float16,indexCommuted] else decayBackwardSource%[tl.float16,tl.float16,original]
  | .fp16, .bf16 => if swapped then decayBackwardSource%[tl.float16,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.float16,tl.bfloat16,original]
  | .fp16, .fp32 => if swapped then decayBackwardSource%[tl.float16,tl.float32,indexCommuted] else decayBackwardSource%[tl.float16,tl.float32,original]
  | .bf16, .fp16 => if swapped then decayBackwardSource%[tl.bfloat16,tl.float16,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.float16,original]
  | .bf16, .bf16 => if swapped then decayBackwardSource%[tl.bfloat16,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.bfloat16,original]
  | .bf16, .fp32 => if swapped then decayBackwardSource%[tl.bfloat16,tl.float32,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.float32,original]
  | .fp32, .fp16 => if swapped then decayBackwardSource%[tl.float32,tl.float16,indexCommuted] else decayBackwardSource%[tl.float32,tl.float16,original]
  | .fp32, .bf16 => if swapped then decayBackwardSource%[tl.float32,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.float32,tl.bfloat16,original]
  | .fp32, .fp32 => if swapped then decayBackwardSource%[tl.float32,tl.float32,indexCommuted] else decayBackwardSource%[tl.float32,tl.float32,original]
```
</details>

<details><summary><code>dq</code></summary>

```lean
noncomputable def dq (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  xs (⟨1, by omega⟩ : Fin 7) t j
    + xs (⟨2, by omega⟩ : Fin 7) t j
      * Real.exp (xs (⟨0, by omega⟩ : Fin 7) t j * Real.log 2)
```
</details>

<details><summary><code>dk</code></summary>

```lean
noncomputable def dk (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  if h : 0 < BT then
    xs (⟨3, by omega⟩ : Fin 7) t j
      + xs (⟨4, by omega⟩ : Fin 7) t j
        * Real.exp
            ((xs (⟨0, by omega⟩ : Fin 7) ⟨BT - 1, Nat.sub_lt h Nat.one_pos⟩ j
                - xs (⟨0, by omega⟩ : Fin 7) t j) * Real.log 2)
  else 0
```
</details>

<details><summary><code>dg</code></summary>

```lean
noncomputable def dg (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  ∑ d : Fin (BT - t.val),
    (dq BT BK xs ⟨t.val + d.val, by omega⟩ j
        * xs (⟨5, by omega⟩ : Fin 7) ⟨t.val + d.val, by omega⟩ j
      - dk BT BK xs ⟨t.val + d.val, by omega⟩ j
        * xs (⟨6, by omega⟩ : Fin 7) ⟨t.val + d.val, by omega⟩ j)
```
</details>

## Public theorem: `backward_original_correct`

**Statement:**
```lean
specification backward_original_correct (source gate : Storage) (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ)
    (hDKInner_DQInter : DKInner ≠ DQInter) (hDKInter_DQInter : DKInter ≠ DQInter)
    (hQ_DQInter : Q ≠ DQInter) (hQ_DKInter : Q ≠ DKInter)
    (hK_DQInter : K ≠ DQInter) (hK_DKInter : K ≠ DKInter)
    (hDQInter_DKInter : DQInter ≠ DKInter)
    (hDQInter_DG : DQInter ≠ DG) (hDKInter_DG : DKInter ≠ DG)
    (hG_DQInter : G ≠ DQInter) (hG_DKInter : G ≠ DKInter) (hG_DG : G ≠ DG)
    (hQ_DG : Q ≠ DG) (hK_DG : K ≠ DG)
    (hDQInner_DQInter : DQInner ≠ DQInter) (hDQInner_DKInter : DQInner ≠ DKInter)
    (hDQInner_DG : DQInner ≠ DG)
    (hDKInner_DKInter : DKInner ≠ DKInter) (hDKInner_DG : DKInner ≠ DG)
    (hBK : BK ≤ DK) (hBT : 0 < BT) :
    Spec.Real (originalIO source gate DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T DK BT BK scale ⊨
      fun _ _ _ xs o t j => values BT BK xs o t j)
```

**Assumptions / layout contracts:**
- `hDKInner_DQInter : DKInner ≠ DQInter`
- `hDKInter_DQInter : DKInter ≠ DQInter`
- `hQ_DQInter : Q ≠ DQInter`
- `hQ_DKInter : Q ≠ DKInter`
- `hK_DQInter : K ≠ DQInter`
- `hK_DKInter : K ≠ DKInter`
- `hDQInter_DKInter : DQInter ≠ DKInter`
- `hDQInter_DG : DQInter ≠ DG`
- `hDKInter_DG : DKInter ≠ DG`
- `hG_DQInter : G ≠ DQInter`
- `hG_DKInter : G ≠ DKInter`
- `hG_DG : G ≠ DG`
- `hQ_DG : Q ≠ DG`
- `hK_DG : K ≠ DG`
- `hDQInner_DQInter : DQInner ≠ DQInter`
- `hDQInner_DKInter : DQInner ≠ DKInter`
- `hDQInner_DG : DQInner ≠ DG`
- `hDKInner_DKInter : DKInner ≠ DKInter`
- `hDKInner_DG : DKInner ≠ DG`
- `hBK : BK ≤ DK`
- `hBT : 0 < BT`

**Closed-form spec defs (transitive):** `originalIO`, `values`, `backward`, `dq`, `dk`, `dg`

<details><summary><code>originalIO</code></summary>

```
/-- Seven initial input streams and three output streams. DQInter/DKInter are
updated in place; the complete contract includes the frame of every other cell. -/
```
```lean
def originalIO (source gate : Storage)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ) : StreamGroupedEmitMasked3DKernelIO where
  kernel := (backward source gate Bool.false DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T BT BK DK scale).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType,BackwardCorrect.source_projects,
      ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody]
  nIn := 7
  nOut := 3
  bufs := [DQInner, DQInter, DKInner, DKInter, Q, K, G, DG]
  inp := fun i => match i with
    | ⟨0, _⟩ => G
    | ⟨1, _⟩ => DQInner
    | ⟨2, _⟩ => DQInter
    | ⟨3, _⟩ => DKInner
    | ⟨4, _⟩ => DKInter
    | ⟨5, _⟩ => Q
    | ⟨6, _⟩ => K
    | ⟨n + 7, h⟩ => absurd h (by omega)
  out := fun o => match o with
    | ⟨0, _⟩ => DQInter
    | ⟨1, _⟩ => DKInter
    | ⟨2, _⟩ => DG
    | ⟨n + 3, h⟩ => absurd h (by omega)
  T := BT
  B := BK
  read := fun _ p₀ p₁ p₂ t j =>
    p₂ * sH + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  readMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK
  write := fun _ p₀ p₁ p₂ t j =>
    p₂ * sH + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  writeMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK
```
</details>

<details><summary><code>values</code></summary>

```lean
noncomputable def values (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (o : Fin 3) (t : Fin BT) (j : Fin BK) : ℝ :=
  match o with
  | ⟨0, _⟩ => dq BT BK xs t j
  | ⟨1, _⟩ => dk BT BK xs t j
  | ⟨2, _⟩ => dg BT BK xs t j
  | ⟨n + 3, h3⟩ => absurd h3 (by omega)
```
</details>

<details><summary><code>backward</code></summary>

```lean
def backward (source gate : Storage) (swapped : Bool) := match source, gate with
  | .fp16, .fp16 => if swapped then decayBackwardSource%[tl.float16,tl.float16,indexCommuted] else decayBackwardSource%[tl.float16,tl.float16,original]
  | .fp16, .bf16 => if swapped then decayBackwardSource%[tl.float16,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.float16,tl.bfloat16,original]
  | .fp16, .fp32 => if swapped then decayBackwardSource%[tl.float16,tl.float32,indexCommuted] else decayBackwardSource%[tl.float16,tl.float32,original]
  | .bf16, .fp16 => if swapped then decayBackwardSource%[tl.bfloat16,tl.float16,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.float16,original]
  | .bf16, .bf16 => if swapped then decayBackwardSource%[tl.bfloat16,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.bfloat16,original]
  | .bf16, .fp32 => if swapped then decayBackwardSource%[tl.bfloat16,tl.float32,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.float32,original]
  | .fp32, .fp16 => if swapped then decayBackwardSource%[tl.float32,tl.float16,indexCommuted] else decayBackwardSource%[tl.float32,tl.float16,original]
  | .fp32, .bf16 => if swapped then decayBackwardSource%[tl.float32,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.float32,tl.bfloat16,original]
  | .fp32, .fp32 => if swapped then decayBackwardSource%[tl.float32,tl.float32,indexCommuted] else decayBackwardSource%[tl.float32,tl.float32,original]
```
</details>

<details><summary><code>dq</code></summary>

```lean
noncomputable def dq (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  xs (⟨1, by omega⟩ : Fin 7) t j
    + xs (⟨2, by omega⟩ : Fin 7) t j
      * Real.exp (xs (⟨0, by omega⟩ : Fin 7) t j * Real.log 2)
```
</details>

<details><summary><code>dk</code></summary>

```lean
noncomputable def dk (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  if h : 0 < BT then
    xs (⟨3, by omega⟩ : Fin 7) t j
      + xs (⟨4, by omega⟩ : Fin 7) t j
        * Real.exp
            ((xs (⟨0, by omega⟩ : Fin 7) ⟨BT - 1, Nat.sub_lt h Nat.one_pos⟩ j
                - xs (⟨0, by omega⟩ : Fin 7) t j) * Real.log 2)
  else 0
```
</details>

<details><summary><code>dg</code></summary>

```lean
noncomputable def dg (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  ∑ d : Fin (BT - t.val),
    (dq BT BK xs ⟨t.val + d.val, by omega⟩ j
        * xs (⟨5, by omega⟩ : Fin 7) ⟨t.val + d.val, by omega⟩ j
      - dk BT BK xs ⟨t.val + d.val, by omega⟩ j
        * xs (⟨6, by omega⟩ : Fin 7) ⟨t.val + d.val, by omega⟩ j)
```
</details>

## Public theorem: `backward_index_commuted_correct`

**Statement:**
```lean
specification backward_index_commuted_correct (source gate : Storage) (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ)
    (hDKInner_DQInter : DKInner ≠ DQInter) (hDKInter_DQInter : DKInter ≠ DQInter)
    (hQ_DQInter : Q ≠ DQInter) (hQ_DKInter : Q ≠ DKInter)
    (hK_DQInter : K ≠ DQInter) (hK_DKInter : K ≠ DKInter)
    (hDQInter_DKInter : DQInter ≠ DKInter)
    (hDQInter_DG : DQInter ≠ DG) (hDKInter_DG : DKInter ≠ DG)
    (hG_DQInter : G ≠ DQInter) (hG_DKInter : G ≠ DKInter) (hG_DG : G ≠ DG)
    (hQ_DG : Q ≠ DG) (hK_DG : K ≠ DG)
    (hDQInner_DQInter : DQInner ≠ DQInter) (hDQInner_DKInter : DQInner ≠ DKInter)
    (hDQInner_DG : DQInner ≠ DG)
    (hDKInner_DKInter : DKInner ≠ DKInter) (hDKInner_DG : DKInner ≠ DG)
    (hBK : BK ≤ DK) (hBT : 0 < BT) :
    Spec.Real (indexCommutedIO source gate DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T DK BT BK scale ⊨
      fun _ _ _ xs o t j => values BT BK xs o t j)
```

**Assumptions / layout contracts:**
- `hDKInner_DQInter : DKInner ≠ DQInter`
- `hDKInter_DQInter : DKInter ≠ DQInter`
- `hQ_DQInter : Q ≠ DQInter`
- `hQ_DKInter : Q ≠ DKInter`
- `hK_DQInter : K ≠ DQInter`
- `hK_DKInter : K ≠ DKInter`
- `hDQInter_DKInter : DQInter ≠ DKInter`
- `hDQInter_DG : DQInter ≠ DG`
- `hDKInter_DG : DKInter ≠ DG`
- `hG_DQInter : G ≠ DQInter`
- `hG_DKInter : G ≠ DKInter`
- `hG_DG : G ≠ DG`
- `hQ_DG : Q ≠ DG`
- `hK_DG : K ≠ DG`
- `hDQInner_DQInter : DQInner ≠ DQInter`
- `hDQInner_DKInter : DQInner ≠ DKInter`
- `hDQInner_DG : DQInner ≠ DG`
- `hDKInner_DKInter : DKInner ≠ DKInter`
- `hDKInner_DG : DKInner ≠ DG`
- `hBK : BK ≤ DK`
- `hBT : 0 < BT`

**Closed-form spec defs (transitive):** `indexCommutedIO`, `values`, `originalIO`, `backward`, `dq`, `dk`, `dg`

<details><summary><code>indexCommutedIO</code></summary>

```lean
def indexCommutedIO (source gate : Storage) (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ) : StreamGroupedEmitMasked3DKernelIO :=
  { originalIO source gate DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T DK BT BK scale with
    kernel := (backward source gate Bool.true DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T BT BK DK scale).eraseDType
    projection := by
      simp only [ComputeKernel.eraseDType,BackwardCorrect.source_projects,
        ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody] }
```
</details>

<details><summary><code>values</code></summary>

```lean
noncomputable def values (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (o : Fin 3) (t : Fin BT) (j : Fin BK) : ℝ :=
  match o with
  | ⟨0, _⟩ => dq BT BK xs t j
  | ⟨1, _⟩ => dk BT BK xs t j
  | ⟨2, _⟩ => dg BT BK xs t j
  | ⟨n + 3, h3⟩ => absurd h3 (by omega)
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- Seven initial input streams and three output streams. DQInter/DKInter are
updated in place; the complete contract includes the frame of every other cell. -/
```
```lean
def originalIO (source gate : Storage)
    (DQInner DQInter DKInner DKInter Q K G DG : RegionName)
    (sH sT sD B H T DK BT BK : Nat) (scale : ℝ) : StreamGroupedEmitMasked3DKernelIO where
  kernel := (backward source gate Bool.false DQInner DQInter DKInner DKInter Q K G DG sH sT sD B H T BT BK DK scale).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType,BackwardCorrect.source_projects,
      ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody]
  nIn := 7
  nOut := 3
  bufs := [DQInner, DQInter, DKInner, DKInter, Q, K, G, DG]
  inp := fun i => match i with
    | ⟨0, _⟩ => G
    | ⟨1, _⟩ => DQInner
    | ⟨2, _⟩ => DQInter
    | ⟨3, _⟩ => DKInner
    | ⟨4, _⟩ => DKInter
    | ⟨5, _⟩ => Q
    | ⟨6, _⟩ => K
    | ⟨n + 7, h⟩ => absurd h (by omega)
  out := fun o => match o with
    | ⟨0, _⟩ => DQInter
    | ⟨1, _⟩ => DKInter
    | ⟨2, _⟩ => DG
    | ⟨n + 3, h⟩ => absurd h (by omega)
  T := BT
  B := BK
  read := fun _ p₀ p₁ p₂ t j =>
    p₂ * sH + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  readMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK
  write := fun _ p₀ p₁ p₂ t j =>
    p₂ * sH + p₁ * BT * DK + p₀ * BK + j.val + t.val * DK
  writeMask := fun _ p₀ _ _ _ j => p₀ * BK + j.val < DK
```
</details>

<details><summary><code>backward</code></summary>

```lean
def backward (source gate : Storage) (swapped : Bool) := match source, gate with
  | .fp16, .fp16 => if swapped then decayBackwardSource%[tl.float16,tl.float16,indexCommuted] else decayBackwardSource%[tl.float16,tl.float16,original]
  | .fp16, .bf16 => if swapped then decayBackwardSource%[tl.float16,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.float16,tl.bfloat16,original]
  | .fp16, .fp32 => if swapped then decayBackwardSource%[tl.float16,tl.float32,indexCommuted] else decayBackwardSource%[tl.float16,tl.float32,original]
  | .bf16, .fp16 => if swapped then decayBackwardSource%[tl.bfloat16,tl.float16,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.float16,original]
  | .bf16, .bf16 => if swapped then decayBackwardSource%[tl.bfloat16,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.bfloat16,original]
  | .bf16, .fp32 => if swapped then decayBackwardSource%[tl.bfloat16,tl.float32,indexCommuted] else decayBackwardSource%[tl.bfloat16,tl.float32,original]
  | .fp32, .fp16 => if swapped then decayBackwardSource%[tl.float32,tl.float16,indexCommuted] else decayBackwardSource%[tl.float32,tl.float16,original]
  | .fp32, .bf16 => if swapped then decayBackwardSource%[tl.float32,tl.bfloat16,indexCommuted] else decayBackwardSource%[tl.float32,tl.bfloat16,original]
  | .fp32, .fp32 => if swapped then decayBackwardSource%[tl.float32,tl.float32,indexCommuted] else decayBackwardSource%[tl.float32,tl.float32,original]
```
</details>

<details><summary><code>dq</code></summary>

```lean
noncomputable def dq (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  xs (⟨1, by omega⟩ : Fin 7) t j
    + xs (⟨2, by omega⟩ : Fin 7) t j
      * Real.exp (xs (⟨0, by omega⟩ : Fin 7) t j * Real.log 2)
```
</details>

<details><summary><code>dk</code></summary>

```lean
noncomputable def dk (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  if h : 0 < BT then
    xs (⟨3, by omega⟩ : Fin 7) t j
      + xs (⟨4, by omega⟩ : Fin 7) t j
        * Real.exp
            ((xs (⟨0, by omega⟩ : Fin 7) ⟨BT - 1, Nat.sub_lt h Nat.one_pos⟩ j
                - xs (⟨0, by omega⟩ : Fin 7) t j) * Real.log 2)
  else 0
```
</details>

<details><summary><code>dg</code></summary>

```lean
noncomputable def dg (BT BK : Nat)
    (xs : Fin 7 → Fin BT → Fin BK → ℝ) (t : Fin BT) (j : Fin BK) : ℝ :=
  ∑ d : Fin (BT - t.val),
    (dq BT BK xs ⟨t.val + d.val, by omega⟩ j
        * xs (⟨5, by omega⟩ : Fin 7) ⟨t.val + d.val, by omega⟩ j
      - dk BT BK xs ⟨t.val + d.val, by omega⟩ j
        * xs (⟨6, by omega⟩ : Fin 7) ⟨t.val + d.val, by omega⟩ j)
```
</details>
