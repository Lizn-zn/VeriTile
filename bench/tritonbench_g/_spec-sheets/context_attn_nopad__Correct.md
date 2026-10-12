# Spec sheet — `bench/tritonbench_g/context_attn_nopad/Correct.lean`

**Python source:** `bench/tritonbench_g/context_attn_nopad/context_attn_nopad.py`

## Public theorem: `original_correct`

**Statement:**
```lean
specification original_correct (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) (hBN : 0 < p.BN) :
    Spec.Real (originalIO p r storage scale ⊨ expected p scale)
```

**Assumptions / layout contracts:**
- `hBN : 0 < p.BN`

**Closed-form spec defs (transitive):** `originalIO`, `expected`, `weight`, `score`

<details><summary><code>originalIO</code></summary>

```lean
def originalIO (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : RealIO.IO p where
  kernel := (kernel p r storage Bool.false scale).eraseDType
  projection := RealIO.projection p r storage Bool.false scale
  read := fun u => {
    q := u.readMem r.query, k := u.readMem r.key, v := u.readMem r.value,
    start := u.readMemValue .int r.start, sequence := u.readMemValue .int r.sequence }
  valid := fun l a => ∀ i j, RealIO.active p l a i → RealIO.active p l a j →
    RealIO.write p l a i = RealIO.write p l a j → i = j
  -- Two metadata cells, masked Q, reached K/V loads and active stores.
  -- The start is nonnegative so converting the K/V pointer advance preserves
  -- its signed value; every active address is also checked before conversion.
  bounds := AddressBounds.Valid p r
  out := r.out
  write := fun l a i => ((a.start l.batch+queryRow p l i.1.val)*p.outRow+
    l.head*p.outHead+i.2.1.val*p.outDim).toNat
  mask := fun l a i => (queryRow p l i.1.val : Int) < Formula.sequence p l a
```
</details>

<details><summary><code>expected</code></summary>

```lean
def expected (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : RealIO.Lane p) : ℝ :=
  (∑ j : Fin (count p l a*p.BN), weight p l a scale i.1.val j.val * value p l a j.val i.2.1.val) /
    ∑ j : Fin (count p l a*p.BN), weight p l a scale i.1.val j.val
```
</details>

<details><summary><code>weight</code></summary>

```
/-- The causal negative-infinity mask has zero real weight; unmasked lanes use
natural exponential, matching tl.exp after precision erasure. -/
```
```lean
def weight (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  if j ≤ queryRow p l i then Real.exp (score p l a scale i j) else 0
```
</details>

<details><summary><code>score</code></summary>

```
/-- Query/key/value lookups use only the initial arrays and source time masks. -/
```
```lean
def score (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  (∑ e : Fin p.D, query p l a i e.val * key p l a j e.val) * scale
```
</details>

## Public theorem: `scaleFirst_correct`

**Statement:**
```lean
specification scaleFirst_correct (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) (hBN : 0 < p.BN) :
    Spec.Real (scaleFirstIO p r storage scale ⊨ expected p scale)
```

**Assumptions / layout contracts:**
- `hBN : 0 < p.BN`

**Closed-form spec defs (transitive):** `scaleFirstIO`, `expected`, `originalIO`, `weight`, `score`

<details><summary><code>scaleFirstIO</code></summary>

```lean
def scaleFirstIO (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : RealIO.IO p :=
  { originalIO p r storage scale with
    kernel := (kernel p r storage Bool.true scale).eraseDType
    projection := RealIO.projection p r storage Bool.true scale }
```
</details>

<details><summary><code>expected</code></summary>

```lean
def expected (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : RealIO.Lane p) : ℝ :=
  (∑ j : Fin (count p l a*p.BN), weight p l a scale i.1.val j.val * value p l a j.val i.2.1.val) /
    ∑ j : Fin (count p l a*p.BN), weight p l a scale i.1.val j.val
```
</details>

<details><summary><code>originalIO</code></summary>

```lean
def originalIO (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : RealIO.IO p where
  kernel := (kernel p r storage Bool.false scale).eraseDType
  projection := RealIO.projection p r storage Bool.false scale
  read := fun u => {
    q := u.readMem r.query, k := u.readMem r.key, v := u.readMem r.value,
    start := u.readMemValue .int r.start, sequence := u.readMemValue .int r.sequence }
  valid := fun l a => ∀ i j, RealIO.active p l a i → RealIO.active p l a j →
    RealIO.write p l a i = RealIO.write p l a j → i = j
  -- Two metadata cells, masked Q, reached K/V loads and active stores.
  -- The start is nonnegative so converting the K/V pointer advance preserves
  -- its signed value; every active address is also checked before conversion.
  bounds := AddressBounds.Valid p r
  out := r.out
  write := fun l a i => ((a.start l.batch+queryRow p l i.1.val)*p.outRow+
    l.head*p.outHead+i.2.1.val*p.outDim).toNat
  mask := fun l a i => (queryRow p l i.1.val : Int) < Formula.sequence p l a
```
</details>

<details><summary><code>weight</code></summary>

```
/-- The causal negative-infinity mask has zero real weight; unmasked lanes use
natural exponential, matching tl.exp after precision erasure. -/
```
```lean
def weight (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  if j ≤ queryRow p l i then Real.exp (score p l a scale i j) else 0
```
</details>

<details><summary><code>score</code></summary>

```
/-- Query/key/value lookups use only the initial arrays and source time masks. -/
```
```lean
def score (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  (∑ e : Fin p.D, query p l a i e.val * key p l a j e.val) * scale
```
</details>
