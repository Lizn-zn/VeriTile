# Spec sheet — `bench/tritonbench_g/context_attn_bloom/Correct.lean`

**Python source:** `bench/tritonbench_g/context_attn_bloom/context_attn_bloom.py`

## Public theorem: `original_correct`

**Statement:**
```lean
specification original_correct (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) (hBN : 0 < p.BN) :
    Spec.Real (originalIO p r storage scale ⊨ expected p scale)
```

**Assumptions / layout contracts:**
- `hBN : 0 < p.BN`

**Closed-form spec defs (transitive):** `originalIO`, `expected`, `normalizedBlocks`, `denominator`, `numerator`, `score`

<details><summary><code>originalIO</code></summary>

```lean
def originalIO (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : RealIO.IO p where
  kernel := (kernel p r storage Bool.false scale).eraseDType
  projection := RealIO.projection p r storage Bool.false scale
  read := fun u => {
    q := u.readMem r.query, k := u.readMem r.key, v := u.readMem r.value,
    start := u.readMemValue .int r.start, sequence := u.readMemValue .int r.sequence,
    prompt := u.readMemValue .int r.prompt, request := u.readMemValue .int r.request,
    tokens := u.readMemValue .int r.tokens }
  valid := fun l a => 0 ≤ a.prompt l.batch ∧
    ∀ i j, RealIO.active p l a i → RealIO.active p l a j →
      RealIO.write p l a i = RealIO.write p l a j → i = j
  -- Four metadata cells, masked Q, reached page-table/K/V lanes and output.
  -- Each address is checked as signed and nonnegative before conversion.
  bounds := AddressBounds.Valid p r
  out := r.out
  write := fun l a i => ((a.start l.batch+queryRow p l i.1.val)*p.outRow+
    l.head*p.outHead+i.2.1.val*p.outDim).toNat
  mask := fun l a i => (queryRow p l i.1.val : Int) < sequence a l ∧ i.2.1.val < p.headDim
```
</details>

<details><summary><code>expected</code></summary>

```lean
def expected (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : RealIO.Lane p) : ℝ :=
  normalizedBlocks p l a scale i.1.val i.2.1.val (count p l a)
```
</details>

<details><summary><code>normalizedBlocks</code></summary>

```
/-- Independent prefix-sum recurrence. The separate acc_scale guard leaves
the earlier accumulator unscaled for wholly future blocks. Replacing this
formula with ideal causal softmax would change the source's real behavior. -/
```
```lean
def normalizedBlocks (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d : Nat) : Nat → ℝ
  | 0 => 0
  | c+1 => normalizedBlocks p l a scale i d c *
      (if ((c*p.BN : Nat) : Int) ≤ (queryRow p l i : Int)+a.prompt l.batch then
        denominator p l a scale i c / denominator p l a scale i (c+1) else 1) +
      (numerator p l a scale i d (c+1)-numerator p l a scale i d c) / denominator p l a scale i (c+1)
```
</details>

<details><summary><code>denominator</code></summary>

```lean
def denominator (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i c : Nat) : ℝ :=
  ∑ j : Fin (count p l a*p.BN), if j.val < c*p.BN then Real.exp (score p l a scale i j.val) else 0
```
</details>

<details><summary><code>numerator</code></summary>

```lean
def numerator (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d c : Nat) : ℝ :=
  ∑ j : Fin (count p l a*p.BN),
    if j.val < c*p.BN then Real.exp (score p l a scale i j.val)*value p l a j.val d else 0
```
</details>

<details><summary><code>score</code></summary>

```
/-- Masked query/key/value lookups read the initial arrays and signed page
indices. The causal mask uses the actual finite sentinel, so future lanes
still have positive mathematical exponential weights. -/
```
```lean
def score (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  if (j : Int) ≤ queryRow p l i+a.prompt l.batch then
    scale * ∑ e : Fin p.D, query p l a i e.val * key p l a j e.val
  else -100000000
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

**Closed-form spec defs (transitive):** `scaleFirstIO`, `expected`, `originalIO`, `normalizedBlocks`, `denominator`, `numerator`, `score`

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
  normalizedBlocks p l a scale i.1.val i.2.1.val (count p l a)
```
</details>

<details><summary><code>originalIO</code></summary>

```lean
def originalIO (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : RealIO.IO p where
  kernel := (kernel p r storage Bool.false scale).eraseDType
  projection := RealIO.projection p r storage Bool.false scale
  read := fun u => {
    q := u.readMem r.query, k := u.readMem r.key, v := u.readMem r.value,
    start := u.readMemValue .int r.start, sequence := u.readMemValue .int r.sequence,
    prompt := u.readMemValue .int r.prompt, request := u.readMemValue .int r.request,
    tokens := u.readMemValue .int r.tokens }
  valid := fun l a => 0 ≤ a.prompt l.batch ∧
    ∀ i j, RealIO.active p l a i → RealIO.active p l a j →
      RealIO.write p l a i = RealIO.write p l a j → i = j
  -- Four metadata cells, masked Q, reached page-table/K/V lanes and output.
  -- Each address is checked as signed and nonnegative before conversion.
  bounds := AddressBounds.Valid p r
  out := r.out
  write := fun l a i => ((a.start l.batch+queryRow p l i.1.val)*p.outRow+
    l.head*p.outHead+i.2.1.val*p.outDim).toNat
  mask := fun l a i => (queryRow p l i.1.val : Int) < sequence a l ∧ i.2.1.val < p.headDim
```
</details>

<details><summary><code>normalizedBlocks</code></summary>

```
/-- Independent prefix-sum recurrence. The separate acc_scale guard leaves
the earlier accumulator unscaled for wholly future blocks. Replacing this
formula with ideal causal softmax would change the source's real behavior. -/
```
```lean
def normalizedBlocks (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d : Nat) : Nat → ℝ
  | 0 => 0
  | c+1 => normalizedBlocks p l a scale i d c *
      (if ((c*p.BN : Nat) : Int) ≤ (queryRow p l i : Int)+a.prompt l.batch then
        denominator p l a scale i c / denominator p l a scale i (c+1) else 1) +
      (numerator p l a scale i d (c+1)-numerator p l a scale i d c) / denominator p l a scale i (c+1)
```
</details>

<details><summary><code>denominator</code></summary>

```lean
def denominator (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i c : Nat) : ℝ :=
  ∑ j : Fin (count p l a*p.BN), if j.val < c*p.BN then Real.exp (score p l a scale i j.val) else 0
```
</details>

<details><summary><code>numerator</code></summary>

```lean
def numerator (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d c : Nat) : ℝ :=
  ∑ j : Fin (count p l a*p.BN),
    if j.val < c*p.BN then Real.exp (score p l a scale i j.val)*value p l a j.val d else 0
```
</details>

<details><summary><code>score</code></summary>

```
/-- Masked query/key/value lookups read the initial arrays and signed page
indices. The causal mask uses the actual finite sentinel, so future lanes
still have positive mathematical exponential weights. -/
```
```lean
def score (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  if (j : Int) ≤ queryRow p l i+a.prompt l.batch then
    scale * ∑ e : Fin p.D, query p l a i e.val * key p l a j e.val
  else -100000000
```
</details>
