# Spec sheet — `bench/tritonbench_g/attention_kernel/Correct.lean`

**Python source:** `bench/tritonbench_g/attention_kernel/attention_kernel.py`

## Public theorem: `attention_correctness`

**Statement:**
```lean
specification attention_correctness
    (hBM : 0 < BM) (hBN : 0 < BN) (hHD : 0 < HD) (hnB : 1 ≤ nB) :
    Spec.Real (originalIO storage Q K V B0 Out sm_scale sqh skh sbh BM BN HD BIAS sbm nB ⊨
      fun _ _ _ xs ys zs ws => expected nB BM BN HD sm_scale xs ys zs ws)
```

**Assumptions / layout contracts:**
- `hBM : 0 < BM`
- `hBN : 0 < BN`
- `hHD : 0 < HD`
- `hnB : 1 ≤ nB`

**Closed-form spec defs (transitive):** `originalIO`, `expected`, `original`, `score`, `value`, `query`, `key`, `blockBias`, `laneBias`

<details><summary><code>originalIO</code></summary>

```
/-- Static Q, streamed K/V and a packed two-window bias stream. In read4,
columns below BLOCK_N are static b1; the last column is the step-dependent b0.
V uses the source's K-head offset, and output uses its Q-head offset. -/
```
```lean
def originalIO (storage : TypedSource.Storage) (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_kh stride_b0h BLOCK_M BLOCK_N HEAD BIAS_LAST_SIZE stride_b0m nB : Nat) :
    StreamMasked3DKernelIO₄ where
  kernel := (TypedSource.original storage Q K V B0 Out sm_scale
    stride_qh HEAD 1 stride_kh HEAD 1 stride_kh HEAD 1 stride_qh HEAD 1
    stride_b0h stride_b0m 2 4 (BLOCK_N * nB) 0 BIAS_LAST_SIZE 128 HEAD BLOCK_M BLOCK_N).eraseDType
  projection := RealSource.original_projection_supported ..
  inp1 := Q
  inp2 := K
  inp3 := V
  inp4 := B0
  out := Out
  T := nB
  B1 := BLOCK_M * HEAD
  B2 := HEAD * BLOCK_N
  B3 := BLOCK_N * HEAD
  B4 := BLOCK_M * (BLOCK_N + 1)
  C := BLOCK_M * HEAD
  outDType := .real
  read1 := fun p₀ p₁ _ _ j =>
    p₁ * stride_qh + (p₀ * BLOCK_M + j.val / HEAD) * HEAD + j.val % HEAD
  read2 := fun _ p₁ _ t j =>
    p₁ * stride_kh + j.val / BLOCK_N + (t.val * BLOCK_N + j.val % BLOCK_N) * HEAD
  read3 := fun _ p₁ _ t j =>
    p₁ * stride_kh + (t.val * BLOCK_N + j.val / HEAD) * HEAD + j.val % HEAD
  read4 := fun p₀ p₁ _ t j =>
    p₁ * stride_b0h + (p₀ * BLOCK_M + j.val / (BLOCK_N + 1)) * stride_b0m +
      (if j.val % (BLOCK_N + 1) < BLOCK_N then
        j.val % (BLOCK_N + 1) % BIAS_LAST_SIZE + BIAS_LAST_SIZE
      else t.val)
  write := fun p₀ p₁ _ j =>
    p₁ * stride_qh + (p₀ * BLOCK_M + j.val / HEAD) * HEAD + (j.val % HEAD) * 1
  mask1 := fun _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ => True
  mask3 := fun _ _ _ _ _ => True
  mask4 := fun _ _ _ _ _ => True
  writeMask := fun _ _ _ _ => True
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Base-2 weighted sum over all keys, independent of the online recurrence. -/
```
```lean
noncomputable def expected (T BM BN HD : Nat) (scale : ℝ)
    (xs : Fin T → Fin (BM * HD) → ℝ) (ys : Fin T → Fin (HD * BN) → ℝ)
    (zs : Fin T → Fin (BN * HD) → ℝ) (ws : Fin T → Fin (BM * (BN + 1)) → ℝ)
    (j : Fin (BM * HD)) : ℝ :=
  let i := Lane2D.decode j
  (Finset.univ.sum fun k : Fin (BN * T) =>
    pow2 (score T BM BN HD scale xs ys ws i.1 k) * value T BN HD zs (k, i.2.1, PUnit.unit)) /
  (Finset.univ.sum fun k : Fin (BN * T) => pow2 (score T BM BN HD scale xs ys ws i.1 k))

section
variable (storage : TypedSource.Storage) (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (sqh skh sbh BM BN HD BIAS sbm nB : Nat)
```
</details>

<details><summary><code>original</code></summary>

```
/-- Select the host-supported storage type before constructing the complete source. -/
```
```lean
def original (storage : Storage) :=
  match storage with
  | .fp16 => FP16.original
  | .bf16 => BF16.original
```
</details>

<details><summary><code>score</code></summary>

```lean
noncomputable def score (T BM BN HD : Nat) (scale : ℝ)
    (xs : Fin T → Fin (BM * HD) → ℝ) (ys : Fin T → Fin (HD * BN) → ℝ)
    (ws : Fin T → Fin (BM * (BN + 1)) → ℝ) (r : Fin BM) (k : Fin (BN * T)) : ℝ :=
  (scale * 1.44269504) *
    Finset.univ.sum (fun d : Fin HD => query T BM HD xs (r, d, PUnit.unit) * key T HD BN ys d k)
    + (blockBias T BM BN ws r (k.val / BN) + laneBias T BM BN ws r (k.val % BN)) * 1.44269504
```
</details>

<details><summary><code>value</code></summary>

```lean
noncomputable abbrev value := RealKernel.akIOvT
```
</details>

<details><summary><code>query</code></summary>

```
/-- These helpers only index the mathematical input arrays: Q and static b1
come from the first slice; K/V and b0 use the global key's block index. -/
```
```lean
noncomputable abbrev query := RealKernel.akIOqT
```
</details>

<details><summary><code>key</code></summary>

```lean
noncomputable abbrev key := RealKernel.akIOkT
```
</details>

<details><summary><code>blockBias</code></summary>

```lean
noncomputable abbrev blockBias := RealKernel.akIOb0T
```
</details>

<details><summary><code>laneBias</code></summary>

```lean
noncomputable abbrev laneBias := RealKernel.akIOb1T
```
</details>

## Public theorem: `attention_reciprocal_correctness`

**Statement:**
```lean
specification attention_reciprocal_correctness
    (hBM : 0 < BM) (hBN : 0 < BN) (hHD : 0 < HD) (hnB : 1 ≤ nB) :
    Spec.Real (reciprocalIO storage Q K V B0 Out sm_scale sqh skh sbh BM BN HD BIAS sbm nB ⊨
      fun _ _ _ xs ys zs ws => expected nB BM BN HD sm_scale xs ys zs ws)
```

**Assumptions / layout contracts:**
- `hBM : 0 < BM`
- `hBN : 0 < BN`
- `hHD : 0 < HD`
- `hnB : 1 ≤ nB`

**Closed-form spec defs (transitive):** `reciprocalIO`, `expected`, `originalIO`, `reciprocal`, `score`, `value`, `original`, `query`, `key`, `blockBias`, `laneBias`

<details><summary><code>reciprocalIO</code></summary>

```
/-- The candidate changes only final normalization, preserving all IO windows. -/
```
```lean
def reciprocalIO (storage : TypedSource.Storage) (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (sqh skh sbh BM BN HD BIAS sbm nB : Nat) : StreamMasked3DKernelIO₄ :=
  { originalIO storage Q K V B0 Out sm_scale sqh skh sbh BM BN HD BIAS sbm nB with
    kernel := (TypedSource.reciprocal storage Q K V B0 Out sm_scale
      sqh HD 1 skh HD 1 skh HD 1 sqh HD 1 sbh sbm 2 4 (BN * nB) 0 BIAS 128 HD BM BN).eraseDType
    projection := RealSource.reciprocal_projection_supported .. }
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Base-2 weighted sum over all keys, independent of the online recurrence. -/
```
```lean
noncomputable def expected (T BM BN HD : Nat) (scale : ℝ)
    (xs : Fin T → Fin (BM * HD) → ℝ) (ys : Fin T → Fin (HD * BN) → ℝ)
    (zs : Fin T → Fin (BN * HD) → ℝ) (ws : Fin T → Fin (BM * (BN + 1)) → ℝ)
    (j : Fin (BM * HD)) : ℝ :=
  let i := Lane2D.decode j
  (Finset.univ.sum fun k : Fin (BN * T) =>
    pow2 (score T BM BN HD scale xs ys ws i.1 k) * value T BN HD zs (k, i.2.1, PUnit.unit)) /
  (Finset.univ.sum fun k : Fin (BN * T) => pow2 (score T BM BN HD scale xs ys ws i.1 k))

section
variable (storage : TypedSource.Storage) (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (sqh skh sbh BM BN HD BIAS sbm nB : Nat)
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- Static Q, streamed K/V and a packed two-window bias stream. In read4,
columns below BLOCK_N are static b1; the last column is the step-dependent b0.
V uses the source's K-head offset, and output uses its Q-head offset. -/
```
```lean
def originalIO (storage : TypedSource.Storage) (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_kh stride_b0h BLOCK_M BLOCK_N HEAD BIAS_LAST_SIZE stride_b0m nB : Nat) :
    StreamMasked3DKernelIO₄ where
  kernel := (TypedSource.original storage Q K V B0 Out sm_scale
    stride_qh HEAD 1 stride_kh HEAD 1 stride_kh HEAD 1 stride_qh HEAD 1
    stride_b0h stride_b0m 2 4 (BLOCK_N * nB) 0 BIAS_LAST_SIZE 128 HEAD BLOCK_M BLOCK_N).eraseDType
  projection := RealSource.original_projection_supported ..
  inp1 := Q
  inp2 := K
  inp3 := V
  inp4 := B0
  out := Out
  T := nB
  B1 := BLOCK_M * HEAD
  B2 := HEAD * BLOCK_N
  B3 := BLOCK_N * HEAD
  B4 := BLOCK_M * (BLOCK_N + 1)
  C := BLOCK_M * HEAD
  outDType := .real
  read1 := fun p₀ p₁ _ _ j =>
    p₁ * stride_qh + (p₀ * BLOCK_M + j.val / HEAD) * HEAD + j.val % HEAD
  read2 := fun _ p₁ _ t j =>
    p₁ * stride_kh + j.val / BLOCK_N + (t.val * BLOCK_N + j.val % BLOCK_N) * HEAD
  read3 := fun _ p₁ _ t j =>
    p₁ * stride_kh + (t.val * BLOCK_N + j.val / HEAD) * HEAD + j.val % HEAD
  read4 := fun p₀ p₁ _ t j =>
    p₁ * stride_b0h + (p₀ * BLOCK_M + j.val / (BLOCK_N + 1)) * stride_b0m +
      (if j.val % (BLOCK_N + 1) < BLOCK_N then
        j.val % (BLOCK_N + 1) % BIAS_LAST_SIZE + BIAS_LAST_SIZE
      else t.val)
  write := fun p₀ p₁ _ j =>
    p₁ * stride_qh + (p₀ * BLOCK_M + j.val / HEAD) * HEAD + (j.val % HEAD) * 1
  mask1 := fun _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ => True
  mask3 := fun _ _ _ _ _ => True
  mask4 := fun _ _ _ _ _ => True
  writeMask := fun _ _ _ _ => True
```
</details>

<details><summary><code>reciprocal</code></summary>

```lean
def reciprocal (storage : Storage) :=
  match storage with
  | .fp16 => FP16.reciprocal
  | .bf16 => BF16.reciprocal
```
</details>

<details><summary><code>score</code></summary>

```lean
noncomputable def score (T BM BN HD : Nat) (scale : ℝ)
    (xs : Fin T → Fin (BM * HD) → ℝ) (ys : Fin T → Fin (HD * BN) → ℝ)
    (ws : Fin T → Fin (BM * (BN + 1)) → ℝ) (r : Fin BM) (k : Fin (BN * T)) : ℝ :=
  (scale * 1.44269504) *
    Finset.univ.sum (fun d : Fin HD => query T BM HD xs (r, d, PUnit.unit) * key T HD BN ys d k)
    + (blockBias T BM BN ws r (k.val / BN) + laneBias T BM BN ws r (k.val % BN)) * 1.44269504
```
</details>

<details><summary><code>value</code></summary>

```lean
noncomputable abbrev value := RealKernel.akIOvT
```
</details>

<details><summary><code>original</code></summary>

```
/-- Select the host-supported storage type before constructing the complete source. -/
```
```lean
def original (storage : Storage) :=
  match storage with
  | .fp16 => FP16.original
  | .bf16 => BF16.original
```
</details>

<details><summary><code>query</code></summary>

```
/-- These helpers only index the mathematical input arrays: Q and static b1
come from the first slice; K/V and b0 use the global key's block index. -/
```
```lean
noncomputable abbrev query := RealKernel.akIOqT
```
</details>

<details><summary><code>key</code></summary>

```lean
noncomputable abbrev key := RealKernel.akIOkT
```
</details>

<details><summary><code>blockBias</code></summary>

```lean
noncomputable abbrev blockBias := RealKernel.akIOb0T
```
</details>

<details><summary><code>laneBias</code></summary>

```lean
noncomputable abbrev laneBias := RealKernel.akIOb1T
```
</details>
