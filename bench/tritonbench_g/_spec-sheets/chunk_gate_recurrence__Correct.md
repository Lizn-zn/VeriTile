# Spec sheet — `bench/tritonbench_g/chunk_gate_recurrence/Correct.lean`

**Python source:** `bench/tritonbench_g/chunk_gate_recurrence/chunk_gate_recurrence.py`

## Public theorem: `forward_correctness`

<details><summary>docstring</summary>

```
/-- S and D remain unchanged through the loop. Seed/output aliasing is allowed
because initialization consumes the seed before writing O. -/
```
</details>

**Statement:**
```lean
specification forward_correctness (p : Parameters) (S D O seed : RegionName)
    (storage gate seedStorage : TypedSource.Storage) (H : Nat) (hasSeed : Bool)
    (hN : 0 < p.N) (hBK : p.BK ≤ p.K) (hBV : p.BV ≤ p.V) (hSO : S ≠ O) (hDO : D ≠ O) :
    Spec.Real (originalForwardIO p S D O seed storage gate seedStorage H hasSeed ⊨ expectedForward p hasSeed)
```

**Assumptions / layout contracts:**
- `hasSeed : Bool`
- `hN : 0 < p.N`
- `hBK : p.BK ≤ p.K`
- `hBV : p.BV ≤ p.V`
- `hSO : S ≠ O`
- `hDO : D ≠ O`

**Closed-form spec defs (transitive):** `Parameters`, `originalForwardIO`, `expectedForward`, `forward`, `specializeForward`

<details><summary><code>Parameters</code></summary>

```
/-- N chunks, matrix dimensions K/V and unmasked tile dimensions BK/BV. -/
```
```lean
abbrev Parameters := Layout
```
</details>

<details><summary><code>originalForwardIO</code></summary>

```
/-- Forward reads S/D for t<N-1 and an optional seed before the first store.
It writes all N output tiles. The bounds constructor checks precisely those
addresses; N=1 needs no S/D allocation, and the seed may alias O. -/
```
```lean
def originalForwardIO (p : Parameters) (S D O seed : RegionName)
    (storage gate seedStorage : TypedSource.Storage) (H : Nat) (hasSeed : Bool) : RealIO.ForwardIO p where
  kernel := (TypedSource.forward storage gate seedStorage S D O seed H p.N p.K p.V p.BK p.BV hasSeed).eraseDType
  projection := RealIO.forward_projection p S D O seed storage gate seedStorage H Bool.false hasSeed
  read := fun s =>
    { source := fun t e j => s.readMem S (address p (launch s) t e.val j.val)
      gate := fun t => s.readMem D (gateAddress p (launch s) t)
      seed := fun e j => s.readMem seed (seedAddress p (launch s) e.val j.val) }
  bounds := fun s => RealForwardSafety.Bounds p (launch s) S D O seed hasSeed
  output := O
  write := fun l t e j => address p l t e.val j.val
```
</details>

<details><summary><code>expectedForward</code></summary>

```
/-- Exclusive recurrence: O[0]=seed (or zero), O[t+1]=O[t]*D[t]+S[t].
This closed form depends only on the initial arrays, never on output memory. -/
```
```lean
def expectedForward (p : Parameters) (hasSeed : Bool) (a : RealIO.ForwardInputs p)
    (c : Nat) (e : Fin p.BK) (j : Fin p.BV) : ℝ :=
  (if hasSeed then a.seed e j else 0) * (∏ t ∈ Finset.range c, a.gate t) +
    ∑ t ∈ Finset.range c, a.source t e j * ∏ u ∈ Finset.Ico (t+1) c, a.gate u
```
</details>

<details><summary><code>forward</code></summary>

```lean
def forward (source gate seed : Storage) :=
  specializeForward source gate Bool.false seed.dtype
```
</details>

<details><summary><code>specializeForward</code></summary>

```lean
def specializeForward (source gate : Storage) (swapped : Bool) :=
  match source, gate, swapped with
  | .fp32, .fp32, Bool.false => gateForwardSource%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => gateForwardSource%[tl.float32, tl.float32, sumLeft]
  | .fp32, .fp16, Bool.false => gateForwardSource%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => gateForwardSource%[tl.float32, tl.float16, sumLeft]
  | .fp32, .bf16, Bool.false => gateForwardSource%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => gateForwardSource%[tl.float32, tl.bfloat16, sumLeft]
  | .fp16, .fp32, Bool.false => gateForwardSource%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => gateForwardSource%[tl.float16, tl.float32, sumLeft]
  | .fp16, .fp16, Bool.false => gateForwardSource%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => gateForwardSource%[tl.float16, tl.float16, sumLeft]
  | .fp16, .bf16, Bool.false => gateForwardSource%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => gateForwardSource%[tl.float16, tl.bfloat16, sumLeft]
  | .bf16, .fp32, Bool.false => gateForwardSource%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => gateForwardSource%[tl.bfloat16, tl.float32, sumLeft]
  | .bf16, .fp16, Bool.false => gateForwardSource%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => gateForwardSource%[tl.bfloat16, tl.float16, sumLeft]
  | .bf16, .bf16, Bool.false => gateForwardSource%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => gateForwardSource%[tl.bfloat16, tl.bfloat16, sumLeft]
```
</details>

## Public theorem: `forward_sum_left_correctness`

**Statement:**
```lean
specification forward_sum_left_correctness (p : Parameters) (S D O seed : RegionName)
    (storage gate seedStorage : TypedSource.Storage) (H : Nat) (hasSeed : Bool)
    (hN : 0 < p.N) (hBK : p.BK ≤ p.K) (hBV : p.BV ≤ p.V) (hSO : S ≠ O) (hDO : D ≠ O) :
    Spec.Real (sumLeftForwardIO p S D O seed storage gate seedStorage H hasSeed ⊨ expectedForward p hasSeed)
```

**Assumptions / layout contracts:**
- `hasSeed : Bool`
- `hN : 0 < p.N`
- `hBK : p.BK ≤ p.K`
- `hBV : p.BV ≤ p.V`
- `hSO : S ≠ O`
- `hDO : D ≠ O`

**Closed-form spec defs (transitive):** `Parameters`, `sumLeftForwardIO`, `expectedForward`, `originalForwardIO`, `forwardSumLeft`, `forward`, `specializeForward`

<details><summary><code>Parameters</code></summary>

```
/-- N chunks, matrix dimensions K/V and unmasked tile dimensions BK/BV. -/
```
```lean
abbrev Parameters := Layout
```
</details>

<details><summary><code>sumLeftForwardIO</code></summary>

```lean
def sumLeftForwardIO (p : Parameters) (S D O seed : RegionName)
    (storage gate seedStorage : TypedSource.Storage) (H : Nat) (hasSeed : Bool) : RealIO.ForwardIO p :=
  { originalForwardIO p S D O seed storage gate seedStorage H hasSeed with
    kernel := (TypedSource.forwardSumLeft storage gate seedStorage S D O seed H p.N p.K p.V p.BK p.BV hasSeed).eraseDType
    projection := RealIO.forward_projection p S D O seed storage gate seedStorage H Bool.true hasSeed }
```
</details>

<details><summary><code>expectedForward</code></summary>

```
/-- Exclusive recurrence: O[0]=seed (or zero), O[t+1]=O[t]*D[t]+S[t].
This closed form depends only on the initial arrays, never on output memory. -/
```
```lean
def expectedForward (p : Parameters) (hasSeed : Bool) (a : RealIO.ForwardInputs p)
    (c : Nat) (e : Fin p.BK) (j : Fin p.BV) : ℝ :=
  (if hasSeed then a.seed e j else 0) * (∏ t ∈ Finset.range c, a.gate t) +
    ∑ t ∈ Finset.range c, a.source t e j * ∏ u ∈ Finset.Ico (t+1) c, a.gate u
```
</details>

<details><summary><code>originalForwardIO</code></summary>

```
/-- Forward reads S/D for t<N-1 and an optional seed before the first store.
It writes all N output tiles. The bounds constructor checks precisely those
addresses; N=1 needs no S/D allocation, and the seed may alias O. -/
```
```lean
def originalForwardIO (p : Parameters) (S D O seed : RegionName)
    (storage gate seedStorage : TypedSource.Storage) (H : Nat) (hasSeed : Bool) : RealIO.ForwardIO p where
  kernel := (TypedSource.forward storage gate seedStorage S D O seed H p.N p.K p.V p.BK p.BV hasSeed).eraseDType
  projection := RealIO.forward_projection p S D O seed storage gate seedStorage H Bool.false hasSeed
  read := fun s =>
    { source := fun t e j => s.readMem S (address p (launch s) t e.val j.val)
      gate := fun t => s.readMem D (gateAddress p (launch s) t)
      seed := fun e j => s.readMem seed (seedAddress p (launch s) e.val j.val) }
  bounds := fun s => RealForwardSafety.Bounds p (launch s) S D O seed hasSeed
  output := O
  write := fun l t e j => address p l t e.val j.val
```
</details>

<details><summary><code>forwardSumLeft</code></summary>

```lean
def forwardSumLeft (source gate seed : Storage) :=
  specializeForward source gate Bool.true seed.dtype
```
</details>

<details><summary><code>forward</code></summary>

```lean
def forward (source gate seed : Storage) :=
  specializeForward source gate Bool.false seed.dtype
```
</details>

<details><summary><code>specializeForward</code></summary>

```lean
def specializeForward (source gate : Storage) (swapped : Bool) :=
  match source, gate, swapped with
  | .fp32, .fp32, Bool.false => gateForwardSource%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => gateForwardSource%[tl.float32, tl.float32, sumLeft]
  | .fp32, .fp16, Bool.false => gateForwardSource%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => gateForwardSource%[tl.float32, tl.float16, sumLeft]
  | .fp32, .bf16, Bool.false => gateForwardSource%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => gateForwardSource%[tl.float32, tl.bfloat16, sumLeft]
  | .fp16, .fp32, Bool.false => gateForwardSource%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => gateForwardSource%[tl.float16, tl.float32, sumLeft]
  | .fp16, .fp16, Bool.false => gateForwardSource%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => gateForwardSource%[tl.float16, tl.float16, sumLeft]
  | .fp16, .bf16, Bool.false => gateForwardSource%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => gateForwardSource%[tl.float16, tl.bfloat16, sumLeft]
  | .bf16, .fp32, Bool.false => gateForwardSource%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => gateForwardSource%[tl.bfloat16, tl.float32, sumLeft]
  | .bf16, .fp16, Bool.false => gateForwardSource%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => gateForwardSource%[tl.bfloat16, tl.float16, sumLeft]
  | .bf16, .bf16, Bool.false => gateForwardSource%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => gateForwardSource%[tl.bfloat16, tl.bfloat16, sumLeft]
```
</details>

## Public theorem: `backward_correctness`

<details><summary>docstring</summary>

```
/-- DI/DG are distinct from one another and from S/D/DS (Separation).
DL is distinct from DI/DG but may alias any input: its store follows all loads.
The positive block dimensions fit K/V; flat allocation and active-address
bounds are required by the IO contract. -/
```
</details>

**Statement:**
```lean
specification backward_correctness (p : Parameters) (S D DI DG DL DS : RegionName)
    (storage gate incoming : TypedSource.Storage) (H : Nat)
    (hN : 0 < p.N) (hBKpos : 0 < p.BK) (hBVpos : 0 < p.BV) (hBK : p.BK ≤ p.K) (hBV : p.BV ≤ p.V)
    (sep : RealBackwardLoop.Separation S D DI DG DS) (hIDL : DI ≠ DL) (hGDL : DG ≠ DL) :
    Spec.Real (originalBackwardIO p S D DI DG DL DS storage gate incoming H ⊨ expectedBackward p)
```

**Assumptions / layout contracts:**
- `hN : 0 < p.N`
- `hBKpos : 0 < p.BK`
- `hBVpos : 0 < p.BV`
- `hBK : p.BK ≤ p.K`
- `hBV : p.BV ≤ p.V`
- `hIDL : DI ≠ DL`
- `hGDL : DG ≠ DL`

**Closed-form spec defs (transitive):** `Parameters`, `originalBackwardIO`, `expectedBackward`, `backward`, `specializeBackward`

<details><summary><code>Parameters</code></summary>

```
/-- N chunks, matrix dimensions K/V and unmasked tile dimensions BK/BV. -/
```
```lean
abbrev Parameters := Layout
```
</details>

<details><summary><code>originalBackwardIO</code></summary>

```
/-- Backward reads saved S for t<N-1 and D/DS for all N chunks. DI/DG write
only t<N-1; DL writes the final seed-gradient tile. The bounds constructor
checks those actual addresses, with no requirement on final dead pointers.
For N=1 there is no S read or DI/DG store; only D/DS and DL are accessed. -/
```
```lean
def originalBackwardIO (p : Parameters) (S D DI DG DL DS : RegionName)
    (storage gate incoming : TypedSource.Storage) (H : Nat) : RealIO.BackwardIO p where
  kernel := (TypedSource.backward storage gate incoming S D DI DG DL DS H p.N p.K p.V p.BK p.BV).eraseDType
  projection := RealIO.backward_projection p S D DI DG DL DS storage gate incoming H Bool.false
  read := fun s =>
    { source := fun t e j => s.readMem S (address p (launch s) t e.val j.val)
      gate := fun t => s.readMem D (gateAddress p (launch s) t)
      incoming := fun t e j => s.readMem DS (address p (launch s) t e.val j.val) }
  bounds := fun s => RealBackwardSafety.Bounds p (launch s) S D DI DG DL DS
  inputGradient := DI
  gateGradient := DG
  seedGradient := DL
  writeInput := fun l t e j => address p l t e.val j.val
  writeGate := fun l t => l.head*p.N*(p.K/p.BK)*(p.V/p.BV) +
    l.key*(p.V/p.BV) + l.value + t*(p.K/p.BK)*(p.V/p.BV)
  writeSeed := fun l e j => seedAddress p l e.val j.val
```
</details>

<details><summary><code>expectedBackward</code></summary>

```
/-- Reverse suffix C[N]=0, C[t]=C[t+1]*D[t]+DS[t]. The kernel emits
DI[t]=C[t+1], DG[t]=sum over the complete BK×BV tile of C[t+1]*S[t], and
DL=C[0]. Host zeroing of the last DI/DG slices is not a kernel output. -/
```
```lean
def expectedBackward (p : Parameters) (a : RealIO.BackwardInputs p) : RealIO.BackwardOutputs p :=
  let suffix := fun (c : Nat) (e : Fin p.BK) (j : Fin p.BV) =>
    ∑ u ∈ Finset.Ico c p.N, a.incoming u e j * ∏ w ∈ Finset.Ico c u, a.gate w
  { inputGradient := fun t => suffix (t+1)
    gateGradient := fun t => ∑ e : Fin p.BK, ∑ j : Fin p.BV, suffix (t+1) e j * a.source t e j
    seedGradient := suffix 0 }
```
</details>

<details><summary><code>backward</code></summary>

```lean
def backward (source gate incoming : Storage) :=
  specializeBackward gate incoming Bool.false source.dtype
```
</details>

<details><summary><code>specializeBackward</code></summary>

```lean
def specializeBackward (gate incoming : Storage) (swapped : Bool) :=
  match gate, incoming, swapped with
  | .fp32, .fp32, Bool.false => gateBackwardSource%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => gateBackwardSource%[tl.float32, tl.float32, sumLeft]
  | .fp32, .fp16, Bool.false => gateBackwardSource%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => gateBackwardSource%[tl.float32, tl.float16, sumLeft]
  | .fp32, .bf16, Bool.false => gateBackwardSource%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => gateBackwardSource%[tl.float32, tl.bfloat16, sumLeft]
  | .fp16, .fp32, Bool.false => gateBackwardSource%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => gateBackwardSource%[tl.float16, tl.float32, sumLeft]
  | .fp16, .fp16, Bool.false => gateBackwardSource%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => gateBackwardSource%[tl.float16, tl.float16, sumLeft]
  | .fp16, .bf16, Bool.false => gateBackwardSource%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => gateBackwardSource%[tl.float16, tl.bfloat16, sumLeft]
  | .bf16, .fp32, Bool.false => gateBackwardSource%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => gateBackwardSource%[tl.bfloat16, tl.float32, sumLeft]
  | .bf16, .fp16, Bool.false => gateBackwardSource%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => gateBackwardSource%[tl.bfloat16, tl.float16, sumLeft]
  | .bf16, .bf16, Bool.false => gateBackwardSource%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => gateBackwardSource%[tl.bfloat16, tl.bfloat16, sumLeft]
```
</details>

## Public theorem: `backward_sum_left_correctness`

**Statement:**
```lean
specification backward_sum_left_correctness (p : Parameters) (S D DI DG DL DS : RegionName)
    (storage gate incoming : TypedSource.Storage) (H : Nat)
    (hN : 0 < p.N) (hBKpos : 0 < p.BK) (hBVpos : 0 < p.BV) (hBK : p.BK ≤ p.K) (hBV : p.BV ≤ p.V)
    (sep : RealBackwardLoop.Separation S D DI DG DS) (hIDL : DI ≠ DL) (hGDL : DG ≠ DL) :
    Spec.Real (sumLeftBackwardIO p S D DI DG DL DS storage gate incoming H ⊨ expectedBackward p)
```

**Assumptions / layout contracts:**
- `hN : 0 < p.N`
- `hBKpos : 0 < p.BK`
- `hBVpos : 0 < p.BV`
- `hBK : p.BK ≤ p.K`
- `hBV : p.BV ≤ p.V`
- `hIDL : DI ≠ DL`
- `hGDL : DG ≠ DL`

**Closed-form spec defs (transitive):** `Parameters`, `sumLeftBackwardIO`, `expectedBackward`, `originalBackwardIO`, `backwardSumLeft`, `backward`, `specializeBackward`

<details><summary><code>Parameters</code></summary>

```
/-- N chunks, matrix dimensions K/V and unmasked tile dimensions BK/BV. -/
```
```lean
abbrev Parameters := Layout
```
</details>

<details><summary><code>sumLeftBackwardIO</code></summary>

```lean
def sumLeftBackwardIO (p : Parameters) (S D DI DG DL DS : RegionName)
    (storage gate incoming : TypedSource.Storage) (H : Nat) : RealIO.BackwardIO p :=
  { originalBackwardIO p S D DI DG DL DS storage gate incoming H with
    kernel := (TypedSource.backwardSumLeft storage gate incoming S D DI DG DL DS H p.N p.K p.V p.BK p.BV).eraseDType
    projection := RealIO.backward_projection p S D DI DG DL DS storage gate incoming H Bool.true }
```
</details>

<details><summary><code>expectedBackward</code></summary>

```
/-- Reverse suffix C[N]=0, C[t]=C[t+1]*D[t]+DS[t]. The kernel emits
DI[t]=C[t+1], DG[t]=sum over the complete BK×BV tile of C[t+1]*S[t], and
DL=C[0]. Host zeroing of the last DI/DG slices is not a kernel output. -/
```
```lean
def expectedBackward (p : Parameters) (a : RealIO.BackwardInputs p) : RealIO.BackwardOutputs p :=
  let suffix := fun (c : Nat) (e : Fin p.BK) (j : Fin p.BV) =>
    ∑ u ∈ Finset.Ico c p.N, a.incoming u e j * ∏ w ∈ Finset.Ico c u, a.gate w
  { inputGradient := fun t => suffix (t+1)
    gateGradient := fun t => ∑ e : Fin p.BK, ∑ j : Fin p.BV, suffix (t+1) e j * a.source t e j
    seedGradient := suffix 0 }
```
</details>

<details><summary><code>originalBackwardIO</code></summary>

```
/-- Backward reads saved S for t<N-1 and D/DS for all N chunks. DI/DG write
only t<N-1; DL writes the final seed-gradient tile. The bounds constructor
checks those actual addresses, with no requirement on final dead pointers.
For N=1 there is no S read or DI/DG store; only D/DS and DL are accessed. -/
```
```lean
def originalBackwardIO (p : Parameters) (S D DI DG DL DS : RegionName)
    (storage gate incoming : TypedSource.Storage) (H : Nat) : RealIO.BackwardIO p where
  kernel := (TypedSource.backward storage gate incoming S D DI DG DL DS H p.N p.K p.V p.BK p.BV).eraseDType
  projection := RealIO.backward_projection p S D DI DG DL DS storage gate incoming H Bool.false
  read := fun s =>
    { source := fun t e j => s.readMem S (address p (launch s) t e.val j.val)
      gate := fun t => s.readMem D (gateAddress p (launch s) t)
      incoming := fun t e j => s.readMem DS (address p (launch s) t e.val j.val) }
  bounds := fun s => RealBackwardSafety.Bounds p (launch s) S D DI DG DL DS
  inputGradient := DI
  gateGradient := DG
  seedGradient := DL
  writeInput := fun l t e j => address p l t e.val j.val
  writeGate := fun l t => l.head*p.N*(p.K/p.BK)*(p.V/p.BV) +
    l.key*(p.V/p.BV) + l.value + t*(p.K/p.BK)*(p.V/p.BV)
  writeSeed := fun l e j => seedAddress p l e.val j.val
```
</details>

<details><summary><code>backwardSumLeft</code></summary>

```lean
def backwardSumLeft (source gate incoming : Storage) :=
  specializeBackward gate incoming Bool.true source.dtype
```
</details>

<details><summary><code>backward</code></summary>

```lean
def backward (source gate incoming : Storage) :=
  specializeBackward gate incoming Bool.false source.dtype
```
</details>

<details><summary><code>specializeBackward</code></summary>

```lean
def specializeBackward (gate incoming : Storage) (swapped : Bool) :=
  match gate, incoming, swapped with
  | .fp32, .fp32, Bool.false => gateBackwardSource%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => gateBackwardSource%[tl.float32, tl.float32, sumLeft]
  | .fp32, .fp16, Bool.false => gateBackwardSource%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => gateBackwardSource%[tl.float32, tl.float16, sumLeft]
  | .fp32, .bf16, Bool.false => gateBackwardSource%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => gateBackwardSource%[tl.float32, tl.bfloat16, sumLeft]
  | .fp16, .fp32, Bool.false => gateBackwardSource%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => gateBackwardSource%[tl.float16, tl.float32, sumLeft]
  | .fp16, .fp16, Bool.false => gateBackwardSource%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => gateBackwardSource%[tl.float16, tl.float16, sumLeft]
  | .fp16, .bf16, Bool.false => gateBackwardSource%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => gateBackwardSource%[tl.float16, tl.bfloat16, sumLeft]
  | .bf16, .fp32, Bool.false => gateBackwardSource%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => gateBackwardSource%[tl.bfloat16, tl.float32, sumLeft]
  | .bf16, .fp16, Bool.false => gateBackwardSource%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => gateBackwardSource%[tl.bfloat16, tl.float16, sumLeft]
  | .bf16, .bf16, Bool.false => gateBackwardSource%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => gateBackwardSource%[tl.bfloat16, tl.bfloat16, sumLeft]
```
</details>
