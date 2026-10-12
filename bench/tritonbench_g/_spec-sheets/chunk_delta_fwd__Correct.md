# Spec sheet — `bench/tritonbench_g/chunk_delta_fwd/Correct.lean`

**Python source:** `bench/tritonbench_g/chunk_delta_fwd/chunk_delta_fwd.py`

## Public theorem: `chunk_delta_correctness`

<details><summary>docstring</summary>

```
/-- Inputs K/D/V remain unchanged through every loop iteration. History and
corrected V are distinct; when final storage is enabled it is distinct from
both. Seed may alias outputs because initialization reads it before all stores. -/
```
</details>

**Statement:**
```lean
specification chunk_delta_correctness (p : Parameters) (r : Regions)
    (kd value initial final : TypedSource.Storage) (useSeed storeFinal : Bool)
    (hK : p.K ≤ p.BK) (hBT : 0 < p.BT) (hBC : 0 < p.BC) (hcover : p.T ≤ p.NT*p.BT)
    (hs : RealOuterStep.Separate r)
    (hfsep : storeFinal = Bool.true → r.history ≠ r.final ∧ r.corrected ≠ r.final) :
    Spec.Real (originalIO p r kd value initial final useSeed storeFinal ⊨ expected p useSeed)
```

**Assumptions / layout contracts:**
- `hK : p.K ≤ p.BK`
- `hBT : 0 < p.BT`
- `hBC : 0 < p.BC`
- `hcover : p.T ≤ p.NT*p.BT`
- `hs : RealOuterStep.Separate r`
- `hfsep : storeFinal = Bool.true → r.history ≠ r.final ∧ r.corrected ≠ r.final`

**Closed-form spec defs (transitive):** `Parameters`, `originalIO`, `expected`, `original`, `specialize`, `sumLeft`

<details><summary><code>Parameters</code></summary>

```
/-- Dimensions, tiles, iteration count and all eight source strides. -/
```
```lean
abbrev Parameters := Layout
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- Bounds check only active cells. The two input matrices use the original
K×T / T×K views. History uses historyTime as its row stride, while seed and
final state are contiguous K×V. The valid launch owns the whole K dimension;
active corrected/history/final addresses must not collide within each output.
All input arrays are read from initial memory, before any output stores. -/
```
```lean
def originalIO (p : Layout) (r : Regions) (kd value initial final : TypedSource.Storage)
    (useSeed storeFinal : Bool) : RealIO.IO p where
  kernel := (TypedSource.original kd value initial final r.key r.value r.delta r.corrected r.history r.initial r.final
    p.keyHead p.keyTime p.keyDim p.valueHead p.valueTime p.valueDim p.historyHead p.historyTime
    p.H p.T p.K p.V p.BT p.BC p.BK p.BV p.NT useSeed storeFinal).eraseDType
  projection := RealIO.projection p r kd value initial final Bool.false useSeed storeFinal
  read := fun s => ⟨s.readMem r.key,s.readMem r.delta,s.readMem r.value⟩
  seed := fun s => s.readMem r.initial
  valid := fun s => s.pids 0 = 0 ∧ RealInnerLoop.NoCollision p (RealInit.launch s) ∧
    RealOuterLoop.HistoryNoCollision p (RealInit.launch s) ∧
    (storeFinal = Bool.true → RealStateStore.NoCollision p (RealInit.launch s) (RealInit.base p (RealInit.launch s)) p.V)
  bounds := fun s => RealSafety.Bounds p (RealInit.launch s) r useSeed storeFinal
  history := r.history
  corrected := r.corrected
  final := r.final
  storeFinal := storeFinal
  historyWindow := fun l c e j => RealOuterStep.historyAddress p l c e.val j.val
  valueWindow := fun l t j => valueAddress p l t j.val
  finalWindow := fun l e j => RealStateStore.address p l (RealInit.base p l) p.V e.val j.val
  stateMask := fun l e j => RealStateStore.active p l e.val j.val
  valueMask := fun l t j => activeValue p l t j.val
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Independent matrix recurrence (RealWindow.state):
H₀ = seed when enabled, otherwise zero;
C_c(t,j) = V(t,j) - Σ_{e<K} D(t,e) H_c(e,j);
H_{c+1}(e,j) = H_c(e,j) + Σ_{t in [c*BT,c*BT+width), t<T} K(t,e) C_c(t,j).

width = ceil(BT/BC)*BC, so adjacent windows may overlap. The final corrected
value at row t uses H_(t/BT), because the last visiting chunk overwrites it.
There is no assumption that BC divides BT. -/
```
```lean
def expected (p : Parameters) (useSeed : Bool) (l : Launch) (a : Inputs) (a₀ : Nat → ℝ) : RealIO.Outputs p :=
  let xs : RealMath.Inputs :=
    { key := fun t e => a.key (l.head*p.keyHead + e*p.keyDim + t*p.keyTime)
      delta := fun t e => a.delta (l.head*p.keyHead + t*p.keyTime + e*p.keyDim)
      value := fun t j => a.value (l.head*p.valueHead + t*p.valueTime + j*p.valueDim)
      seed := fun e j => a₀ (l.head*p.K*p.V + e*p.V + j) }
  let width := ((p.BT+p.BC-1)/p.BC)*p.BC
  let state := RealWindow.state xs p.T p.K p.BT width useSeed
  { history := fun c e j => state c e.val (l.value*p.BV+j.val)
    corrected := fun t j => xs.value t (l.value*p.BV+j.val) -
      ∑ e : Fin p.K, xs.delta t e.val * state (t/p.BT) e.val (l.value*p.BV+j.val)
    final := fun e j => state p.NT e.val (l.value*p.BV+j.val) }
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (kd value initial final : Storage) := specialize kd value initial final Bool.false
```
</details>

<details><summary><code>specialize</code></summary>

```lean
def specialize (kd value initial final : Storage) (swapped : Bool) :=
  let dtype := kd.dtype
  let out_dtype := initial.dtype
  match value, final, swapped with
  | .fp32, .fp32, Bool.false => chunkDeltaSource%[tl.float32, tl.float32, original] dtype out_dtype
  | .fp32, .fp32, Bool.true => chunkDeltaSource%[tl.float32, tl.float32, sumLeft] dtype out_dtype
  | .fp32, .fp16, Bool.false => chunkDeltaSource%[tl.float32, tl.float16, original] dtype out_dtype
  | .fp32, .fp16, Bool.true => chunkDeltaSource%[tl.float32, tl.float16, sumLeft] dtype out_dtype
  | .fp32, .bf16, Bool.false => chunkDeltaSource%[tl.float32, tl.bfloat16, original] dtype out_dtype
  | .fp32, .bf16, Bool.true => chunkDeltaSource%[tl.float32, tl.bfloat16, sumLeft] dtype out_dtype
  | .fp16, .fp32, Bool.false => chunkDeltaSource%[tl.float16, tl.float32, original] dtype out_dtype
  | .fp16, .fp32, Bool.true => chunkDeltaSource%[tl.float16, tl.float32, sumLeft] dtype out_dtype
  | .fp16, .fp16, Bool.false => chunkDeltaSource%[tl.float16, tl.float16, original] dtype out_dtype
  | .fp16, .fp16, Bool.true => chunkDeltaSource%[tl.float16, tl.float16, sumLeft] dtype out_dtype
  | .fp16, .bf16, Bool.false => chunkDeltaSource%[tl.float16, tl.bfloat16, original] dtype out_dtype
  | .fp16, .bf16, Bool.true => chunkDeltaSource%[tl.float16, tl.bfloat16, sumLeft] dtype out_dtype
  | .bf16, .fp32, Bool.false => chunkDeltaSource%[tl.bfloat16, tl.float32, original] dtype out_dtype
  | .bf16, .fp32, Bool.true => chunkDeltaSource%[tl.bfloat16, tl.float32, sumLeft] dtype out_dtype
  | .bf16, .fp16, Bool.false => chunkDeltaSource%[tl.bfloat16, tl.float16, original] dtype out_dtype
  | .bf16, .fp16, Bool.true => chunkDeltaSource%[tl.bfloat16, tl.float16, sumLeft] dtype out_dtype
  | .bf16, .bf16, Bool.false => chunkDeltaSource%[tl.bfloat16, tl.bfloat16, original] dtype out_dtype
  | .bf16, .bf16, Bool.true => chunkDeltaSource%[tl.bfloat16, tl.bfloat16, sumLeft] dtype out_dtype
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft (kd value initial final : Storage) := specialize kd value initial final Bool.true
```
</details>

## Public theorem: `chunk_sum_left_correctness`

**Statement:**
```lean
specification chunk_sum_left_correctness (p : Parameters) (r : Regions)
    (kd value initial final : TypedSource.Storage) (useSeed storeFinal : Bool)
    (hK : p.K ≤ p.BK) (hBT : 0 < p.BT) (hBC : 0 < p.BC) (hcover : p.T ≤ p.NT*p.BT)
    (hs : RealOuterStep.Separate r)
    (hfsep : storeFinal = Bool.true → r.history ≠ r.final ∧ r.corrected ≠ r.final) :
    Spec.Real (sumLeftIO p r kd value initial final useSeed storeFinal ⊨ expected p useSeed)
```

**Assumptions / layout contracts:**
- `hK : p.K ≤ p.BK`
- `hBT : 0 < p.BT`
- `hBC : 0 < p.BC`
- `hcover : p.T ≤ p.NT*p.BT`
- `hs : RealOuterStep.Separate r`
- `hfsep : storeFinal = Bool.true → r.history ≠ r.final ∧ r.corrected ≠ r.final`

**Closed-form spec defs (transitive):** `Parameters`, `sumLeftIO`, `expected`, `originalIO`, `sumLeft`, `original`, `specialize`

<details><summary><code>Parameters</code></summary>

```
/-- Dimensions, tiles, iteration count and all eight source strides. -/
```
```lean
abbrev Parameters := Layout
```
</details>

<details><summary><code>sumLeftIO</code></summary>

```lean
def sumLeftIO (p : Parameters) (r : Regions) (kd value initial final : TypedSource.Storage)
    (useSeed storeFinal : Bool) : RealIO.IO p :=
  { originalIO p r kd value initial final useSeed storeFinal with
    kernel := (TypedSource.sumLeft kd value initial final r.key r.value r.delta r.corrected r.history r.initial r.final
      p.keyHead p.keyTime p.keyDim p.valueHead p.valueTime p.valueDim p.historyHead p.historyTime
      p.H p.T p.K p.V p.BT p.BC p.BK p.BV p.NT useSeed storeFinal).eraseDType
    projection := RealIO.projection p r kd value initial final Bool.true useSeed storeFinal }
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Independent matrix recurrence (RealWindow.state):
H₀ = seed when enabled, otherwise zero;
C_c(t,j) = V(t,j) - Σ_{e<K} D(t,e) H_c(e,j);
H_{c+1}(e,j) = H_c(e,j) + Σ_{t in [c*BT,c*BT+width), t<T} K(t,e) C_c(t,j).

width = ceil(BT/BC)*BC, so adjacent windows may overlap. The final corrected
value at row t uses H_(t/BT), because the last visiting chunk overwrites it.
There is no assumption that BC divides BT. -/
```
```lean
def expected (p : Parameters) (useSeed : Bool) (l : Launch) (a : Inputs) (a₀ : Nat → ℝ) : RealIO.Outputs p :=
  let xs : RealMath.Inputs :=
    { key := fun t e => a.key (l.head*p.keyHead + e*p.keyDim + t*p.keyTime)
      delta := fun t e => a.delta (l.head*p.keyHead + t*p.keyTime + e*p.keyDim)
      value := fun t j => a.value (l.head*p.valueHead + t*p.valueTime + j*p.valueDim)
      seed := fun e j => a₀ (l.head*p.K*p.V + e*p.V + j) }
  let width := ((p.BT+p.BC-1)/p.BC)*p.BC
  let state := RealWindow.state xs p.T p.K p.BT width useSeed
  { history := fun c e j => state c e.val (l.value*p.BV+j.val)
    corrected := fun t j => xs.value t (l.value*p.BV+j.val) -
      ∑ e : Fin p.K, xs.delta t e.val * state (t/p.BT) e.val (l.value*p.BV+j.val)
    final := fun e j => state p.NT e.val (l.value*p.BV+j.val) }
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- Bounds check only active cells. The two input matrices use the original
K×T / T×K views. History uses historyTime as its row stride, while seed and
final state are contiguous K×V. The valid launch owns the whole K dimension;
active corrected/history/final addresses must not collide within each output.
All input arrays are read from initial memory, before any output stores. -/
```
```lean
def originalIO (p : Layout) (r : Regions) (kd value initial final : TypedSource.Storage)
    (useSeed storeFinal : Bool) : RealIO.IO p where
  kernel := (TypedSource.original kd value initial final r.key r.value r.delta r.corrected r.history r.initial r.final
    p.keyHead p.keyTime p.keyDim p.valueHead p.valueTime p.valueDim p.historyHead p.historyTime
    p.H p.T p.K p.V p.BT p.BC p.BK p.BV p.NT useSeed storeFinal).eraseDType
  projection := RealIO.projection p r kd value initial final Bool.false useSeed storeFinal
  read := fun s => ⟨s.readMem r.key,s.readMem r.delta,s.readMem r.value⟩
  seed := fun s => s.readMem r.initial
  valid := fun s => s.pids 0 = 0 ∧ RealInnerLoop.NoCollision p (RealInit.launch s) ∧
    RealOuterLoop.HistoryNoCollision p (RealInit.launch s) ∧
    (storeFinal = Bool.true → RealStateStore.NoCollision p (RealInit.launch s) (RealInit.base p (RealInit.launch s)) p.V)
  bounds := fun s => RealSafety.Bounds p (RealInit.launch s) r useSeed storeFinal
  history := r.history
  corrected := r.corrected
  final := r.final
  storeFinal := storeFinal
  historyWindow := fun l c e j => RealOuterStep.historyAddress p l c e.val j.val
  valueWindow := fun l t j => valueAddress p l t j.val
  finalWindow := fun l e j => RealStateStore.address p l (RealInit.base p l) p.V e.val j.val
  stateMask := fun l e j => RealStateStore.active p l e.val j.val
  valueMask := fun l t j => activeValue p l t j.val
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft (kd value initial final : Storage) := specialize kd value initial final Bool.true
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (kd value initial final : Storage) := specialize kd value initial final Bool.false
```
</details>

<details><summary><code>specialize</code></summary>

```lean
def specialize (kd value initial final : Storage) (swapped : Bool) :=
  let dtype := kd.dtype
  let out_dtype := initial.dtype
  match value, final, swapped with
  | .fp32, .fp32, Bool.false => chunkDeltaSource%[tl.float32, tl.float32, original] dtype out_dtype
  | .fp32, .fp32, Bool.true => chunkDeltaSource%[tl.float32, tl.float32, sumLeft] dtype out_dtype
  | .fp32, .fp16, Bool.false => chunkDeltaSource%[tl.float32, tl.float16, original] dtype out_dtype
  | .fp32, .fp16, Bool.true => chunkDeltaSource%[tl.float32, tl.float16, sumLeft] dtype out_dtype
  | .fp32, .bf16, Bool.false => chunkDeltaSource%[tl.float32, tl.bfloat16, original] dtype out_dtype
  | .fp32, .bf16, Bool.true => chunkDeltaSource%[tl.float32, tl.bfloat16, sumLeft] dtype out_dtype
  | .fp16, .fp32, Bool.false => chunkDeltaSource%[tl.float16, tl.float32, original] dtype out_dtype
  | .fp16, .fp32, Bool.true => chunkDeltaSource%[tl.float16, tl.float32, sumLeft] dtype out_dtype
  | .fp16, .fp16, Bool.false => chunkDeltaSource%[tl.float16, tl.float16, original] dtype out_dtype
  | .fp16, .fp16, Bool.true => chunkDeltaSource%[tl.float16, tl.float16, sumLeft] dtype out_dtype
  | .fp16, .bf16, Bool.false => chunkDeltaSource%[tl.float16, tl.bfloat16, original] dtype out_dtype
  | .fp16, .bf16, Bool.true => chunkDeltaSource%[tl.float16, tl.bfloat16, sumLeft] dtype out_dtype
  | .bf16, .fp32, Bool.false => chunkDeltaSource%[tl.bfloat16, tl.float32, original] dtype out_dtype
  | .bf16, .fp32, Bool.true => chunkDeltaSource%[tl.bfloat16, tl.float32, sumLeft] dtype out_dtype
  | .bf16, .fp16, Bool.false => chunkDeltaSource%[tl.bfloat16, tl.float16, original] dtype out_dtype
  | .bf16, .fp16, Bool.true => chunkDeltaSource%[tl.bfloat16, tl.float16, sumLeft] dtype out_dtype
  | .bf16, .bf16, Bool.false => chunkDeltaSource%[tl.bfloat16, tl.bfloat16, original] dtype out_dtype
  | .bf16, .bf16, Bool.true => chunkDeltaSource%[tl.bfloat16, tl.bfloat16, sumLeft] dtype out_dtype
```
</details>
