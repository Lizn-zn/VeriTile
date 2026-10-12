# Spec sheet — `bench/tritonbench_g/block_sparse_attn/Correct.lean`

**Python source:** `bench/tritonbench_g/block_sparse_attn/block_sparse_attn.py`

## Public theorem: `block_sparse_attention_correctness`

<details><summary>docstring</summary>

```
/-- Positive key-block width supports the row maximum; the row stride keeps
all stored channels separate, including when the second branch is enabled. -/
```
</details>

**Statement:**
```lean
specification block_sparse_attention_correctness (storage : TypedSource.Storage) (BM BN BD : Nat) (p : Parameters)
    (hN : 0 < BN) (hstride : BD ≤ p.oRow) (htwo : 2 ≤ p.blocks → 2 * BD ≤ p.oRow) :
    Spec.Real (originalIO storage BM BN BD p ⊨[BN] expected BM BN BD p)
```

**Assumptions / layout contracts:**
- `hN : 0 < BN`
- `hstride : BD ≤ p.oRow`
- `htwo : 2 ≤ p.blocks → 2 * BD ≤ p.oRow`

**Closed-form spec defs (transitive):** `Parameters`, `originalIO`, `expected`, `original`, `requirements`, `specialize`, `csrSchedule`, `scalarLeft`

<details><summary><code>Parameters</code></summary>

```
/-- Q/K/V/Out and signed rows/cols; independent CSR row/column strides;
independent tensor batch/head/row strides; head counts, context, scale,
one/two-channel selection and query/key mask flags. -/
```
```lean
abbrev Parameters := RealProgram.Parameters
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- Q/K/V pins cover their mathematical context; metadata pins cover the two
row endpoints and visited column slots. Allocation bounds use active physical
lanes, including padding for evenM/evenN. Width is BD or exactly 2*BD.
Output channels share one row-masked window, with no scratch allocation. -/
```
```lean
def originalIO (storage : TypedSource.Storage) (BM BN BD : Nat) (p : Parameters) : RealIO.IO BM where
  kernel := (TypedSource.original storage p.Out p.Q p.K p.V p.rows p.cols
    p.rowStride p.columnStride p.layouts p.scale
    p.qBatch p.qHead p.qRow p.kBatch p.kHead p.kRow p.vBatch p.vHead p.vRow
    p.oBatch p.oHead p.oRow p.heads p.kvHeads p.context BM BN BD p.blocks p.evenM p.evenN).eraseDType
  projection := RealContract.projection storage Bool.false BM BN BD p
  Q := p.Q
  K := p.K
  V := p.V
  rows := p.rows
  columns := p.cols
  output := p.Out
  context := p.context
  width := if 2 ≤ p.blocks then 2 * BD else BD
  evenM := p.evenM
  evenN := p.evenN
  rowSlot := fun pid₀ pid₁ next =>
    (pid₁ % p.heads % p.layouts) * p.rowStride + pid₀ + if next then 1 else 0
  columnSlot := fun pid₀ pid₁ xs visit =>
    (((pid₁ % p.heads % p.layouts) * p.columnStride : Nat) : Int) +
      xs.rowIndex (((pid₁ % p.heads % p.layouts) * p.rowStride + pid₀ : Nat) : Int) + (visit : Int)
  queryWindow := fun pid₁ r d => pid₁ / p.heads * p.qBatch + pid₁ % p.heads * p.qHead + r * p.qRow + d
  keyWindow := fun pid₁ r d => pid₁ / p.heads * p.kBatch + (pid₁ % p.heads / (p.heads / p.kvHeads)) * p.kHead + r.toNat * p.kRow + d
  valueWindow := fun pid₁ r d => pid₁ / p.heads * p.vBatch + (pid₁ % p.heads / (p.heads / p.kvHeads)) * p.vHead + r.toNat * p.vRow + d
  outputWindow := fun pid₀ pid₁ r d => pid₁ / p.heads * p.oBatch + pid₁ % p.heads * p.oHead + (pid₀ * BM + r.val) * p.oRow + d
  domain := requirements p
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Independent softmax-weighted value over CSR visits and their key lanes.
The score uses one or two channel blocks, then scales the complete dot.
Causally future keys have zero weight. Duplicate keys retain multiplicity. -/
```
```lean
def expected (BM BN BD : Nat) (p : Parameters) (pid₀ pid₁ : Nat)
    (xs : Inputs) (r : Fin BM) (d : Nat) : ℝ :=
  let layout := pid₁ % p.heads % p.layouts
  let first := xs.rowIndex ((layout * p.rowStride + pid₀ : Nat) : Int)
  let last := xs.rowIndex ((layout * p.rowStride + pid₀ + 1 : Nat) : Int)
  let count := (last - first).toNat
  let row := pid₀ * BM + r.val
  let key := fun visit (lane : Fin BN) =>
    xs.columnIndex (((layout * p.columnStride : Nat) : Int) + first + (visit : Int)) * (BN : Int) + (lane.val : Int)
  let score := fun k =>
    ((∑ c : Fin BD, xs.query row c.val * xs.key k c.val) +
      if 2 ≤ p.blocks then ∑ c : Fin BD, xs.query row (BD + c.val) * xs.key k (BD + c.val) else 0) * p.scale
  let weight := fun visit lane => if key visit lane ≤ (row : Int) then Real.exp (score (key visit lane)) else 0
  (∑ visit ∈ Finset.range count, ∑ lane : Fin BN, weight visit lane * xs.value (key visit lane) d) /
  (∑ visit ∈ Finset.range count, ∑ lane : Fin BN, weight visit lane)
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (storage : Storage) : KernelSource := specialize storage Bool.false
```
</details>

<details><summary><code>requirements</code></summary>

```
/-- Signed visits themselves may be negative. Their full metadata slot and
the selected key-block index must be nonnegative. Allocation upper bounds
are supplied separately by the windows; an empty CSR row is allowed. -/
```
```lean
def requirements (p : Parameters) (pid₀ pid₁ : Nat) (xs : Inputs) : Prop :=
  let sched := csrSchedule p pid₀ pid₁
  let slot := fun visit : Nat => ((sched.layout * p.columnStride : Nat) : Int) + CSRMath.first sched xs + (visit : Int)
  (∀ visit, visit < CSRMath.count sched xs → 0 ≤ slot visit) ∧
  (∀ visit, visit < CSRMath.count sched xs → 0 ≤ xs.columnIndex (slot visit))
```
</details>

<details><summary><code>specialize</code></summary>

```lean
def specialize (storage : Storage) (swapped : Bool) : KernelSource :=
  match storage, swapped with
  | .fp32, Bool.false => blockSparseSource%[tl.float32, original]
  | .fp32, Bool.true => blockSparseSource%[tl.float32, scalarLeft]
  | .fp16, Bool.false => blockSparseSource%[tl.float16, original]
  | .fp16, Bool.true => blockSparseSource%[tl.float16, scalarLeft]
  | .bf16, Bool.false => blockSparseSource%[tl.bfloat16, original]
  | .bf16, Bool.true => blockSparseSource%[tl.bfloat16, scalarLeft]
```
</details>

<details><summary><code>csrSchedule</code></summary>

```lean
def csrSchedule (p : Parameters) (pid₀ pid₁ : Nat) : CSRMath.Schedule where
  layout := pid₁ % p.heads % p.layouts
  rowStride := p.rowStride
  columnStride := p.columnStride
  queryBlock := pid₀
```
</details>

<details><summary><code>scalarLeft</code></summary>

```lean
def scalarLeft (storage : Storage) : KernelSource := specialize storage Bool.true

#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>

## Public theorem: `block_sparse_scalar_left_correctness`

**Statement:**
```lean
specification block_sparse_scalar_left_correctness (storage : TypedSource.Storage) (BM BN BD : Nat) (p : Parameters)
    (hN : 0 < BN) (hstride : BD ≤ p.oRow) (htwo : 2 ≤ p.blocks → 2 * BD ≤ p.oRow) :
    Spec.Real (scalarLeftIO storage BM BN BD p ⊨[BN] expected BM BN BD p)
```

**Assumptions / layout contracts:**
- `hN : 0 < BN`
- `hstride : BD ≤ p.oRow`
- `htwo : 2 ≤ p.blocks → 2 * BD ≤ p.oRow`

**Closed-form spec defs (transitive):** `Parameters`, `scalarLeftIO`, `expected`, `originalIO`, `scalarLeft`, `original`, `requirements`, `specialize`, `csrSchedule`

<details><summary><code>Parameters</code></summary>

```
/-- Q/K/V/Out and signed rows/cols; independent CSR row/column strides;
independent tensor batch/head/row strides; head counts, context, scale,
one/two-channel selection and query/key mask flags. -/
```
```lean
abbrev Parameters := RealProgram.Parameters
```
</details>

<details><summary><code>scalarLeftIO</code></summary>

```lean
def scalarLeftIO (storage : TypedSource.Storage) (BM BN BD : Nat) (p : Parameters) : RealIO.IO BM :=
  { originalIO storage BM BN BD p with
    kernel := (TypedSource.scalarLeft storage p.Out p.Q p.K p.V p.rows p.cols
      p.rowStride p.columnStride p.layouts p.scale
      p.qBatch p.qHead p.qRow p.kBatch p.kHead p.kRow p.vBatch p.vHead p.vRow
      p.oBatch p.oHead p.oRow p.heads p.kvHeads p.context BM BN BD p.blocks p.evenM p.evenN).eraseDType
    projection := RealContract.projection storage Bool.true BM BN BD p }
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Independent softmax-weighted value over CSR visits and their key lanes.
The score uses one or two channel blocks, then scales the complete dot.
Causally future keys have zero weight. Duplicate keys retain multiplicity. -/
```
```lean
def expected (BM BN BD : Nat) (p : Parameters) (pid₀ pid₁ : Nat)
    (xs : Inputs) (r : Fin BM) (d : Nat) : ℝ :=
  let layout := pid₁ % p.heads % p.layouts
  let first := xs.rowIndex ((layout * p.rowStride + pid₀ : Nat) : Int)
  let last := xs.rowIndex ((layout * p.rowStride + pid₀ + 1 : Nat) : Int)
  let count := (last - first).toNat
  let row := pid₀ * BM + r.val
  let key := fun visit (lane : Fin BN) =>
    xs.columnIndex (((layout * p.columnStride : Nat) : Int) + first + (visit : Int)) * (BN : Int) + (lane.val : Int)
  let score := fun k =>
    ((∑ c : Fin BD, xs.query row c.val * xs.key k c.val) +
      if 2 ≤ p.blocks then ∑ c : Fin BD, xs.query row (BD + c.val) * xs.key k (BD + c.val) else 0) * p.scale
  let weight := fun visit lane => if key visit lane ≤ (row : Int) then Real.exp (score (key visit lane)) else 0
  (∑ visit ∈ Finset.range count, ∑ lane : Fin BN, weight visit lane * xs.value (key visit lane) d) /
  (∑ visit ∈ Finset.range count, ∑ lane : Fin BN, weight visit lane)
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- Q/K/V pins cover their mathematical context; metadata pins cover the two
row endpoints and visited column slots. Allocation bounds use active physical
lanes, including padding for evenM/evenN. Width is BD or exactly 2*BD.
Output channels share one row-masked window, with no scratch allocation. -/
```
```lean
def originalIO (storage : TypedSource.Storage) (BM BN BD : Nat) (p : Parameters) : RealIO.IO BM where
  kernel := (TypedSource.original storage p.Out p.Q p.K p.V p.rows p.cols
    p.rowStride p.columnStride p.layouts p.scale
    p.qBatch p.qHead p.qRow p.kBatch p.kHead p.kRow p.vBatch p.vHead p.vRow
    p.oBatch p.oHead p.oRow p.heads p.kvHeads p.context BM BN BD p.blocks p.evenM p.evenN).eraseDType
  projection := RealContract.projection storage Bool.false BM BN BD p
  Q := p.Q
  K := p.K
  V := p.V
  rows := p.rows
  columns := p.cols
  output := p.Out
  context := p.context
  width := if 2 ≤ p.blocks then 2 * BD else BD
  evenM := p.evenM
  evenN := p.evenN
  rowSlot := fun pid₀ pid₁ next =>
    (pid₁ % p.heads % p.layouts) * p.rowStride + pid₀ + if next then 1 else 0
  columnSlot := fun pid₀ pid₁ xs visit =>
    (((pid₁ % p.heads % p.layouts) * p.columnStride : Nat) : Int) +
      xs.rowIndex (((pid₁ % p.heads % p.layouts) * p.rowStride + pid₀ : Nat) : Int) + (visit : Int)
  queryWindow := fun pid₁ r d => pid₁ / p.heads * p.qBatch + pid₁ % p.heads * p.qHead + r * p.qRow + d
  keyWindow := fun pid₁ r d => pid₁ / p.heads * p.kBatch + (pid₁ % p.heads / (p.heads / p.kvHeads)) * p.kHead + r.toNat * p.kRow + d
  valueWindow := fun pid₁ r d => pid₁ / p.heads * p.vBatch + (pid₁ % p.heads / (p.heads / p.kvHeads)) * p.vHead + r.toNat * p.vRow + d
  outputWindow := fun pid₀ pid₁ r d => pid₁ / p.heads * p.oBatch + pid₁ % p.heads * p.oHead + (pid₀ * BM + r.val) * p.oRow + d
  domain := requirements p
```
</details>

<details><summary><code>scalarLeft</code></summary>

```lean
def scalarLeft (storage : Storage) : KernelSource := specialize storage Bool.true

#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (storage : Storage) : KernelSource := specialize storage Bool.false
```
</details>

<details><summary><code>requirements</code></summary>

```
/-- Signed visits themselves may be negative. Their full metadata slot and
the selected key-block index must be nonnegative. Allocation upper bounds
are supplied separately by the windows; an empty CSR row is allowed. -/
```
```lean
def requirements (p : Parameters) (pid₀ pid₁ : Nat) (xs : Inputs) : Prop :=
  let sched := csrSchedule p pid₀ pid₁
  let slot := fun visit : Nat => ((sched.layout * p.columnStride : Nat) : Int) + CSRMath.first sched xs + (visit : Int)
  (∀ visit, visit < CSRMath.count sched xs → 0 ≤ slot visit) ∧
  (∀ visit, visit < CSRMath.count sched xs → 0 ≤ xs.columnIndex (slot visit))
```
</details>

<details><summary><code>specialize</code></summary>

```lean
def specialize (storage : Storage) (swapped : Bool) : KernelSource :=
  match storage, swapped with
  | .fp32, Bool.false => blockSparseSource%[tl.float32, original]
  | .fp32, Bool.true => blockSparseSource%[tl.float32, scalarLeft]
  | .fp16, Bool.false => blockSparseSource%[tl.float16, original]
  | .fp16, Bool.true => blockSparseSource%[tl.float16, scalarLeft]
  | .bf16, Bool.false => blockSparseSource%[tl.bfloat16, original]
  | .bf16, Bool.true => blockSparseSource%[tl.bfloat16, scalarLeft]
```
</details>

<details><summary><code>csrSchedule</code></summary>

```lean
def csrSchedule (p : Parameters) (pid₀ pid₁ : Nat) : CSRMath.Schedule where
  layout := pid₁ % p.heads % p.layouts
  rowStride := p.rowStride
  columnStride := p.columnStride
  queryBlock := pid₀
```
</details>
