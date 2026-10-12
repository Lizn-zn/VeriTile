# Spec sheet — `bench/tritonbench_g/bmm_optimized/Correct.lean`

**Python source:** `bench/tritonbench_g/bmm_optimized/bmm_optimized.py`

## Public theorem: `bmm_optimized_correctness`

**Statement:**
```lean
specification bmm_optimized_correctness :
    Spec.Real (originalIO dtype BM BN BK p ⊨ expected)
```

**Closed-form spec defs (transitive):** `originalIO`, `expected`, `Parameters`

<details><summary><code>originalIO</code></summary>

```
/-- Each input window covers exactly K entries for the selected output rows
or columns. Tail rows/columns are masked. The contract requires disjoint A/B/O
allocations and zero unspecified load values; both actual num_programs values
must match the displayed grid. -/
```
```lean
def originalIO (dtype : TypedSource.Storage) (BM BN BK : Nat) (p : Parameters) : RealIO.IO BM BN p.K where
  kernel := (RealProgram.source dtype Bool.false BM BN BK p).eraseDType
  projection := by
    rw [RealProgram.storage_erasure]
    exact (RealViews.io Bool.false BM BN BK p).projection
  A := p.A
  B := p.B
  O := p.O
  gridX := BmmChunkBwd.RealMath.blocks p.M BM
  gridY := BmmChunkBwd.RealMath.blocks p.N BN
  aWindow := fun px py pb i => pb * p.M * p.K + RealViews.row BM BN p px py i.1 * p.K + i.2.1.val
  bWindow := fun px py pb i => pb * p.K * p.N + i.1.val * p.N + RealViews.column BM BN p px py i.2.1
  outputWindow := fun px py pb i => pb * p.M * p.N + RealViews.row BM BN p px py i.1 * p.N + RealViews.column BM BN p px py i.2.1
  aMask := fun px py _ i => RealViews.row BM BN p px py i.1 < p.M
  bMask := fun px py _ i => RealViews.column BM BN p px py i.2.1 < p.N
  outputMask := fun px py _ i => RealViews.row BM BN p px py i.1 < p.M ∧ RealViews.column BM BN p px py i.2.1 < p.N
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Ordinary matrix multiplication on initial input arrays. -/
```
```lean
def expected {BM BN K : Nat} (xs : Inputs BM BN K) (i : TileIndex [BM,BN]) : ℝ :=
  ∑ k : Fin K, xs.a i.1 k * xs.b k i.2.1

section
variable (dtype : TypedSource.Storage) (BM BN BK : Nat) (p : Parameters)
    (hBM : 0 < BM) (hBN : 0 < BN) (hBK : 0 < BK) (hGroup : 0 < p.groupM)
    (hDivM : p.divM = Bool.true → BM ∣ p.M)
    (hDivN : p.divN = Bool.true → BN ∣ p.N)
    (hDivK : p.divK = Bool.true → BK ∣ p.K)

include hBM hBN hBK hGroup hDivM hDivN hDivK
```
</details>

<details><summary><code>Parameters</code></summary>

```
/-- Buffer names, logical M/N/K, CTA grouping and all three mask-elision flags. -/
```
```lean
abbrev Parameters := RealProgram.Parameters
```
</details>

## Public theorem: `bmm_dot_first_correctness`

**Statement:**
```lean
specification bmm_dot_first_correctness :
    Spec.Real (dotFirstIO dtype BM BN BK p ⊨ expected)
```

**Closed-form spec defs (transitive):** `dotFirstIO`, `expected`, `originalIO`, `Parameters`

<details><summary><code>dotFirstIO</code></summary>

```
/-- Commute the fp32 carry and dot result in each actual K-loop iteration. -/
```
```lean
def dotFirstIO (dtype : TypedSource.Storage) (BM BN BK : Nat) (p : Parameters) : RealIO.IO BM BN p.K :=
  { originalIO dtype BM BN BK p with
    kernel := (RealProgram.source dtype Bool.true BM BN BK p).eraseDType
    projection := by
      rw [RealProgram.storage_erasure]
      exact (RealViews.io Bool.true BM BN BK p).projection }
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Ordinary matrix multiplication on initial input arrays. -/
```
```lean
def expected {BM BN K : Nat} (xs : Inputs BM BN K) (i : TileIndex [BM,BN]) : ℝ :=
  ∑ k : Fin K, xs.a i.1 k * xs.b k i.2.1

section
variable (dtype : TypedSource.Storage) (BM BN BK : Nat) (p : Parameters)
    (hBM : 0 < BM) (hBN : 0 < BN) (hBK : 0 < BK) (hGroup : 0 < p.groupM)
    (hDivM : p.divM = Bool.true → BM ∣ p.M)
    (hDivN : p.divN = Bool.true → BN ∣ p.N)
    (hDivK : p.divK = Bool.true → BK ∣ p.K)

include hBM hBN hBK hGroup hDivM hDivN hDivK
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- Each input window covers exactly K entries for the selected output rows
or columns. Tail rows/columns are masked. The contract requires disjoint A/B/O
allocations and zero unspecified load values; both actual num_programs values
must match the displayed grid. -/
```
```lean
def originalIO (dtype : TypedSource.Storage) (BM BN BK : Nat) (p : Parameters) : RealIO.IO BM BN p.K where
  kernel := (RealProgram.source dtype Bool.false BM BN BK p).eraseDType
  projection := by
    rw [RealProgram.storage_erasure]
    exact (RealViews.io Bool.false BM BN BK p).projection
  A := p.A
  B := p.B
  O := p.O
  gridX := BmmChunkBwd.RealMath.blocks p.M BM
  gridY := BmmChunkBwd.RealMath.blocks p.N BN
  aWindow := fun px py pb i => pb * p.M * p.K + RealViews.row BM BN p px py i.1 * p.K + i.2.1.val
  bWindow := fun px py pb i => pb * p.K * p.N + i.1.val * p.N + RealViews.column BM BN p px py i.2.1
  outputWindow := fun px py pb i => pb * p.M * p.N + RealViews.row BM BN p px py i.1 * p.N + RealViews.column BM BN p px py i.2.1
  aMask := fun px py _ i => RealViews.row BM BN p px py i.1 < p.M
  bMask := fun px py _ i => RealViews.column BM BN p px py i.2.1 < p.N
  outputMask := fun px py _ i => RealViews.row BM BN p px py i.1 < p.M ∧ RealViews.column BM BN p px py i.2.1 < p.N
```
</details>

<details><summary><code>Parameters</code></summary>

```
/-- Buffer names, logical M/N/K, CTA grouping and all three mask-elision flags. -/
```
```lean
abbrev Parameters := RealProgram.Parameters
```
</details>
