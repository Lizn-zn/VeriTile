# Spec sheet — `bench/tritonbench_g/chunk_gla_simple/Correct.lean`

**Python source:** `bench/tritonbench_g/chunk_gla_simple/chunk_gla_simple.py`

## Public theorem: `original_correct`

**Statement:**
```lean
specification original_correct (p : Layout) (r : Regions) (dot value gate : Storage) (out : FloatDType) (scale : ℝ) :
    original p r dot value gate out scale ⊨ formula p scale
```

**Closed-form spec defs (transitive):** `original`, `formula`

<details><summary><code>original</code></summary>

```lean
def original (p : Layout) (r : Regions) (dot value gate : Storage) (out : FloatDType) (scale : ℝ) :
    IO Launch Inputs (TileIndex [p.BT,p.BV]) where
  kernel := (ChunkGlaSimple.original dot value gate out r.query r.key r.value r.history r.gate r.output
    p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey scale p.T p.K p.V p.BT p.BK p.BV).eraseDType
  projection := RealIO.projection p r dot value gate out scale Bool.false
  grid := fun s => ⟨s.pids 0,s.pids 1,s.pids 2⟩
  read := fun s => ⟨s.readMem r.query,s.readMem r.key,s.readMem r.value,s.readMem r.history,s.readMem r.gate⟩
  valid := fun s => NoCollision p (launch s)
  bounds := fun s => Safe.Bounds p (launch s) r
  out := r.output
  write := fun l i => l.head*p.valueHead+(l.chunk*p.BT+i.1.val)*p.valueTime+(l.value*p.BV+i.2.1.val)
  mask := fun l i => l.chunk*p.BT+i.1.val < p.T ∧ l.value*p.BV+i.2.1.val < p.V
```
</details>

<details><summary><code>formula</code></summary>

```
/-- `query`, `key`, `history`, `gate`, `value` read the initial arrays using
the source strides and zero extension. These are mathematical accessors,
independent of kernel execution. -/
```
```lean
def formula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : TileIndex [p.BT,p.BV]) : ℝ :=
  ((∑ kb ∈ Finset.range (numBlocks p), ∑ e : Fin p.BK,
      query p l a i.1.val (kb*p.BK+e.val)*history p l a (kb*p.BK+e.val) i.2.1.val)*Real.exp (gate p l a i.1.val) +
    ∑ j : Fin p.BT,
      (if j.val ≤ i.1.val then
        (∑ kb ∈ Finset.range (numBlocks p), ∑ e : Fin p.BK,
          query p l a i.1.val (kb*p.BK+e.val)*key p l a (kb*p.BK+e.val) j.val)*
          Real.exp (gate p l a i.1.val-gate p l a j.val)
      else 0)*value p l a j.val i.2.1.val)*scale
```
</details>

## Public theorem: `scaleFirst_correct`

**Statement:**
```lean
specification scaleFirst_correct (p : Layout) (r : Regions) (dot value gate : Storage) (out : FloatDType) (scale : ℝ) :
    scaleFirst p r dot value gate out scale ⊨ formula p scale
```

**Closed-form spec defs (transitive):** `scaleFirst`, `formula`, `original`

<details><summary><code>scaleFirst</code></summary>

```lean
def scaleFirst (p : Layout) (r : Regions) (dot value gate : Storage) (out : FloatDType) (scale : ℝ) :
    IO Launch Inputs (TileIndex [p.BT,p.BV]) :=
  { original p r dot value gate out scale with
    kernel := (ChunkGlaSimple.scaleFirst dot value gate out r.query r.key r.value r.history r.gate r.output
      p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey scale p.T p.K p.V p.BT p.BK p.BV).eraseDType
    projection := RealIO.projection p r dot value gate out scale Bool.true }
```
</details>

<details><summary><code>formula</code></summary>

```
/-- `query`, `key`, `history`, `gate`, `value` read the initial arrays using
the source strides and zero extension. These are mathematical accessors,
independent of kernel execution. -/
```
```lean
def formula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : TileIndex [p.BT,p.BV]) : ℝ :=
  ((∑ kb ∈ Finset.range (numBlocks p), ∑ e : Fin p.BK,
      query p l a i.1.val (kb*p.BK+e.val)*history p l a (kb*p.BK+e.val) i.2.1.val)*Real.exp (gate p l a i.1.val) +
    ∑ j : Fin p.BT,
      (if j.val ≤ i.1.val then
        (∑ kb ∈ Finset.range (numBlocks p), ∑ e : Fin p.BK,
          query p l a i.1.val (kb*p.BK+e.val)*key p l a (kb*p.BK+e.val) j.val)*
          Real.exp (gate p l a i.1.val-gate p l a j.val)
      else 0)*value p l a j.val i.2.1.val)*scale
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (p : Layout) (r : Regions) (dot value gate : Storage) (out : FloatDType) (scale : ℝ) :
    IO Launch Inputs (TileIndex [p.BT,p.BV]) where
  kernel := (ChunkGlaSimple.original dot value gate out r.query r.key r.value r.history r.gate r.output
    p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey scale p.T p.K p.V p.BT p.BK p.BV).eraseDType
  projection := RealIO.projection p r dot value gate out scale Bool.false
  grid := fun s => ⟨s.pids 0,s.pids 1,s.pids 2⟩
  read := fun s => ⟨s.readMem r.query,s.readMem r.key,s.readMem r.value,s.readMem r.history,s.readMem r.gate⟩
  valid := fun s => NoCollision p (launch s)
  bounds := fun s => Safe.Bounds p (launch s) r
  out := r.output
  write := fun l i => l.head*p.valueHead+(l.chunk*p.BT+i.1.val)*p.valueTime+(l.value*p.BV+i.2.1.val)
  mask := fun l i => l.chunk*p.BT+i.1.val < p.T ∧ l.value*p.BV+i.2.1.val < p.V
```
</details>
