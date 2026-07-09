# Spec sheet — `bench/tritonbench_g/batched_vecmat_mult/BatchedVecmatMult.lean`

**Python source:** `bench/tritonbench_g/batched_vecmat_mult/batched_vecmat_mult.py`

## Public theorem: `batched_vecmat_one_row_block_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the one-row batched vecmat slice. -/
```
</details>

**Statement:**
```lean
theorem batched_vecmat_one_row_block_compute_correct
    (A B output : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := batched_vecmat_one_row_block A B output dim_n dim_k N K
        BLOCK_N BLOCK_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => nIndex s BLOCK_N i < N)
        (fun i => (output, outOffset s dim_n BLOCK_N i)))
      (expected := fun i =>
        vecmatSpec s A B dim_n dim_k N K BLOCK_N BLOCK_K i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i)`
- `fun i : Fin BLOCK_N => nIndex s BLOCK_N i < N`

**Closed-form spec defs (transitive):** `outOffset`, `batched_vecmat_one_row_block`, `nIndex`, `vecmatSpec`, `vecmatProdTile`, `aOffset`, `bOffset`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset (s : BlockState) (dim_n BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * dim_n + (s.pids 1 * BLOCK_N + i.val)
```
</details>

<details><summary><code>batched_vecmat_one_row_block</code></summary>

```
/-- Proof-oriented one-`m`, one-`n`-block slice of
`batched_vecmat_mult.py`'s `batched_vecmat_kernel`.

This captures a single row of A and one N block of the corresponding B batch:
load `A[m, k]`, load `B[m, n, k]`, reduce over K, and store `output[m, n]`. -/
```
```lean
def batched_vecmat_one_row_block
    (A B output : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  m_index = tl.program_id(0)
  n_index = tl.program_id(1)
  offset_n = n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  offset_k = tl.arange(0, $(BLOCK_K))
  a = tl.load(A + m_index * $(dim_k) + offset_k,
    mask=offset_k < $(K), other=0.0)
  b = tl.load(B + m_index * $(dim_n) * $(dim_k) +
      offset_n[:, None] * $(dim_k) + offset_k[None, :],
    mask=(offset_n[:, None] < $(N)) and (offset_k[None, :] < $(K)),
    other=0.0)
  acc = tl.sum(a[None, :] * b, axis=1)
  tl.store(output + m_index * $(dim_n) + offset_n, acc,
    mask=offset_n < $(N))
}
```
</details>

<details><summary><code>nIndex</code></summary>

```lean
def nIndex (s : BlockState) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + i.val
```
</details>

<details><summary><code>vecmatSpec</code></summary>

```lean
noncomputable def vecmatSpec
    (s : BlockState) (A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    ((Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
      (vecmatProdTile s A B dim_n dim_k N K BLOCK_N BLOCK_K)).data
        (i, PUnit.unit))
```
</details>

<details><summary><code>vecmatProdTile</code></summary>

```lean
noncomputable def vecmatProdTile
    (s : BlockState) (A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat) :
    Tile .real [BLOCK_N, BLOCK_K] :=
  { data := fun idx =>
      let ni := (TileShape.dropInsertedIndex [BLOCK_N] 1 1 (idx.1, 0, PUnit.unit)).1
      let kj := (TileShape.dropInsertedIndex [BLOCK_K] 0 1 (0, idx.2.1, PUnit.unit)).1
      Option.map₂ (fun a b => a * b)
        (if kj.val < K then
          some (s.readMem A (aOffset s dim_k kj))
        else some (0.0 : ℝ))
        (if nIndex s BLOCK_N ni < N ∧ kj.val < K then
          some (s.readMem B (bOffset s dim_n dim_k BLOCK_N ni kj))
        else some (0.0 : ℝ)) }
```
</details>

<details><summary><code>aOffset</code></summary>

```lean
def aOffset (s : BlockState) (dim_k : Nat) (j : Fin BLOCK_K) : Nat :=
  s.pids 0 * dim_k + j.val
```
</details>

<details><summary><code>bOffset</code></summary>

```lean
def bOffset (s : BlockState) (dim_n dim_k BLOCK_N : Nat)
    (i : Fin BLOCK_N) (j : Fin BLOCK_K) : Nat :=
  s.pids 0 * dim_n * dim_k + nIndex s BLOCK_N i * dim_k + j.val
```
</details>

## Public theorem: `batched_vecmat_one_row_k_block_compute_correct`

**Statement:**
```lean
theorem batched_vecmat_one_row_k_block_compute_correct
    (A B output : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := batched_vecmat_one_row_k_block A B output dim_n dim_k N K
        BLOCK_N BLOCK_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => nIndex s BLOCK_N i < N)
        (fun i => (output, outOffset s dim_n BLOCK_N i)))
      (expected := fun i =>
        vecmatSpecK s A B dim_n dim_k N K BLOCK_N BLOCK_K i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i)`
- `fun i : Fin BLOCK_N => nIndex s BLOCK_N i < N`

**Closed-form spec defs (transitive):** `outOffset`, `batched_vecmat_one_row_k_block`, `nIndex`, `vecmatSpecK`, `vecmatProdTileK`, `kIndex`, `aOffsetK`, `bOffsetK`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset (s : BlockState) (dim_n BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * dim_n + (s.pids 1 * BLOCK_N + i.val)
```
</details>

<details><summary><code>batched_vecmat_one_row_k_block</code></summary>

```
/-- One-row, one-`k_index` loop slice of `batched_vecmat_kernel`.

Unlike `batched_vecmat_one_row_block`, this slice includes the Python loop's
`k_index * block_k` offset in both A and B loads, so it represents an arbitrary
iteration of the K-block loop. -/
```
```lean
def batched_vecmat_one_row_k_block
    (A B output : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  m_index = tl.program_id(0)
  n_index = tl.program_id(1)
  k_index = tl.program_id(2)
  offset_n = n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  offset_k = k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K))
  a = tl.load(A + m_index * $(dim_k) + offset_k,
    mask=offset_k < $(K), other=0.0)
  b = tl.load(B + m_index * $(dim_n) * $(dim_k) +
      offset_n[:, None] * $(dim_k) + offset_k[None, :],
    mask=(offset_n[:, None] < $(N)) and (offset_k[None, :] < $(K)),
    other=0.0)
  acc = tl.sum(a[None, :] * b, axis=1)
  tl.store(output + m_index * $(dim_n) + offset_n, acc,
    mask=offset_n < $(N))
}
```
</details>

<details><summary><code>nIndex</code></summary>

```lean
def nIndex (s : BlockState) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + i.val
```
</details>

<details><summary><code>vecmatSpecK</code></summary>

```lean
noncomputable def vecmatSpecK
    (s : BlockState) (A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    ((Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
      (vecmatProdTileK s A B dim_n dim_k N K BLOCK_N BLOCK_K)).data
        (i, PUnit.unit))
```
</details>

<details><summary><code>vecmatProdTileK</code></summary>

```lean
noncomputable def vecmatProdTileK
    (s : BlockState) (A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat) :
    Tile .real [BLOCK_N, BLOCK_K] :=
  { data := fun idx =>
      let ni := (TileShape.dropInsertedIndex [BLOCK_N] 1 1 (idx.1, 0, PUnit.unit)).1
      let kj := (TileShape.dropInsertedIndex [BLOCK_K] 0 1 (0, idx.2.1, PUnit.unit)).1
      Option.map₂ (fun a b => a * b)
        (if kIndex s BLOCK_K kj < K then
          some (s.readMem A (aOffsetK s dim_k BLOCK_K kj))
        else some (0.0 : ℝ))
        (if nIndex s BLOCK_N ni < N ∧ kIndex s BLOCK_K kj < K then
          some (s.readMem B (bOffsetK s dim_n dim_k BLOCK_N BLOCK_K ni kj))
        else some (0.0 : ℝ)) }
```
</details>

<details><summary><code>kIndex</code></summary>

```lean
def kIndex (s : BlockState) (BLOCK_K : Nat) (j : Fin BLOCK_K) : Nat :=
  s.pids 2 * BLOCK_K + j.val
```
</details>

<details><summary><code>aOffsetK</code></summary>

```lean
def aOffsetK (s : BlockState) (dim_k BLOCK_K : Nat) (j : Fin BLOCK_K) : Nat :=
  s.pids 0 * dim_k + kIndex s BLOCK_K j
```
</details>

<details><summary><code>bOffsetK</code></summary>

```lean
def bOffsetK (s : BlockState) (dim_n dim_k BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_N) (j : Fin BLOCK_K) : Nat :=
  s.pids 0 * dim_n * dim_k + nIndex s BLOCK_N i * dim_k +
    kIndex s BLOCK_K j
```
</details>

## Public theorem: `batched_vecmat_one_row_k_accum_slice_compute_correct`

**Statement:**
```lean
theorem batched_vecmat_one_row_k_accum_slice_compute_correct
    (AccPre A B Out : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := batched_vecmat_one_row_k_accum_slice AccPre A B Out
        dim_n dim_k N K BLOCK_N BLOCK_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => nIndex s BLOCK_N i < N)
        (fun i => (Out, outOffset s dim_n BLOCK_N i)))
      (expected := fun i =>
        vecmatAccumSpecK s AccPre A B dim_n dim_k N K BLOCK_N BLOCK_K i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i)`
- `fun i : Fin BLOCK_N => nIndex s BLOCK_N i < N`

**Closed-form spec defs (transitive):** `outOffset`, `batched_vecmat_one_row_k_accum_slice`, `nIndex`, `vecmatAccumSpecK`, `vecmatProdTileK`, `kIndex`, `aOffsetK`, `bOffsetK`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset (s : BlockState) (dim_n BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * dim_n + (s.pids 1 * BLOCK_N + i.val)
```
</details>

<details><summary><code>batched_vecmat_one_row_k_accum_slice</code></summary>

```
/-- Materialized accumulator update for one K-block loop iteration.

The Python kernel keeps `vecmat` in registers and performs `vecmat += ...`.
This proof slice materializes the incoming accumulator in `AccPre`, adds the
current `k_index` contribution, and stores the updated accumulator to `Out`. -/
```
```lean
def batched_vecmat_one_row_k_accum_slice
    (AccPre A B Out : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  m_index = tl.program_id(0)
  n_index = tl.program_id(1)
  k_index = tl.program_id(2)
  offset_n = n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  offset_k = k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K))
  mask_n = offset_n < $(N)
  old = tl.load(AccPre + m_index * $(dim_n) + offset_n,
    mask=mask_n, other=0.0)
  a = tl.load(A + m_index * $(dim_k) + offset_k,
    mask=offset_k < $(K), other=0.0)
  b = tl.load(B + m_index * $(dim_n) * $(dim_k) +
      offset_n[:, None] * $(dim_k) + offset_k[None, :],
    mask=(offset_n[:, None] < $(N)) and (offset_k[None, :] < $(K)),
    other=0.0)
  delta = tl.sum(a[None, :] * b, axis=1)
  acc = old + delta
  tl.store(Out + m_index * $(dim_n) + offset_n, acc, mask=mask_n)
}
```
</details>

<details><summary><code>nIndex</code></summary>

```lean
def nIndex (s : BlockState) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + i.val
```
</details>

<details><summary><code>vecmatAccumSpecK</code></summary>

```lean
noncomputable def vecmatAccumSpecK
    (s : BlockState) (AccPre A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun old delta => old + delta)
      (some (s.readMem AccPre (outOffset s dim_n BLOCK_N i)))
      ((Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
        (vecmatProdTileK s A B dim_n dim_k N K BLOCK_N BLOCK_K)).data
          (i, PUnit.unit)))
```
</details>

<details><summary><code>vecmatProdTileK</code></summary>

```lean
noncomputable def vecmatProdTileK
    (s : BlockState) (A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat) :
    Tile .real [BLOCK_N, BLOCK_K] :=
  { data := fun idx =>
      let ni := (TileShape.dropInsertedIndex [BLOCK_N] 1 1 (idx.1, 0, PUnit.unit)).1
      let kj := (TileShape.dropInsertedIndex [BLOCK_K] 0 1 (0, idx.2.1, PUnit.unit)).1
      Option.map₂ (fun a b => a * b)
        (if kIndex s BLOCK_K kj < K then
          some (s.readMem A (aOffsetK s dim_k BLOCK_K kj))
        else some (0.0 : ℝ))
        (if nIndex s BLOCK_N ni < N ∧ kIndex s BLOCK_K kj < K then
          some (s.readMem B (bOffsetK s dim_n dim_k BLOCK_N BLOCK_K ni kj))
        else some (0.0 : ℝ)) }
```
</details>

<details><summary><code>kIndex</code></summary>

```lean
def kIndex (s : BlockState) (BLOCK_K : Nat) (j : Fin BLOCK_K) : Nat :=
  s.pids 2 * BLOCK_K + j.val
```
</details>

<details><summary><code>aOffsetK</code></summary>

```lean
def aOffsetK (s : BlockState) (dim_k BLOCK_K : Nat) (j : Fin BLOCK_K) : Nat :=
  s.pids 0 * dim_k + kIndex s BLOCK_K j
```
</details>

<details><summary><code>bOffsetK</code></summary>

```lean
def bOffsetK (s : BlockState) (dim_n dim_k BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_N) (j : Fin BLOCK_K) : Nat :=
  s.pids 0 * dim_n * dim_k + nIndex s BLOCK_N i * dim_k +
    kIndex s BLOCK_K j
```
</details>

## Public theorem: `batched_vecmat_one_row_const_k_accum_slice_compute_correct`

**Statement:**
```lean
theorem batched_vecmat_one_row_const_k_accum_slice_compute_correct
    (AccPre A B Out : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K kIdx : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := batched_vecmat_one_row_const_k_accum_slice AccPre A B Out
        dim_n dim_k N K BLOCK_N BLOCK_K kIdx)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => nIndex s BLOCK_N i < N)
        (fun i => (Out, outOffset s dim_n BLOCK_N i)))
      (expected := fun i =>
        vecmatConstKAccumSpec s AccPre A B dim_n dim_k N K BLOCK_N BLOCK_K
          kIdx i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i)`
- `fun i : Fin BLOCK_N => nIndex s BLOCK_N i < N`

**Closed-form spec defs (transitive):** `outOffset`, `batched_vecmat_one_row_const_k_accum_slice`, `nIndex`, `vecmatConstKAccumSpec`, `vecmatAccumSpecK`, `withKIndex`, `vecmatProdTileK`, `kIndex`, `aOffsetK`, `bOffsetK`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset (s : BlockState) (dim_n BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * dim_n + (s.pids 1 * BLOCK_N + i.val)
```
</details>

<details><summary><code>batched_vecmat_one_row_const_k_accum_slice</code></summary>

```
/-- A proof-oriented loop-iteration slice with `k_index` fixed by the caller.

The Python kernel's `for k_index in range(k_blocks)` counter is not a program
ID.  This slice keeps the same accumulator update as
`batched_vecmat_one_row_k_accum_slice`, but binds `k_index` to a literal loop
iteration so wrappers can name the exact `k=0`, `k=1`, ... Python iterations. -/
```
```lean
def batched_vecmat_one_row_const_k_accum_slice
    (AccPre A B Out : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K kIdx : Nat) :
    ComputeKernel := triton {
  m_index = tl.program_id(0)
  n_index = tl.program_id(1)
  k_index = $(kIdx)
  offset_n = n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  offset_k = k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K))
  mask_n = offset_n < $(N)
  old = tl.load(AccPre + m_index * $(dim_n) + offset_n,
    mask=mask_n, other=0.0)
  a = tl.load(A + m_index * $(dim_k) + offset_k,
    mask=offset_k < $(K), other=0.0)
  b = tl.load(B + m_index * $(dim_n) * $(dim_k) +
      offset_n[:, None] * $(dim_k) + offset_k[None, :],
    mask=(offset_n[:, None] < $(N)) and (offset_k[None, :] < $(K)),
    other=0.0)
  delta = tl.sum(a[None, :] * b, axis=1)
  acc = old + delta
  tl.store(Out + m_index * $(dim_n) + offset_n, acc, mask=mask_n)
}
```
</details>

<details><summary><code>nIndex</code></summary>

```lean
def nIndex (s : BlockState) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + i.val
```
</details>

<details><summary><code>vecmatConstKAccumSpec</code></summary>

```lean
noncomputable def vecmatConstKAccumSpec
    (s : BlockState) (AccPre A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K kIdx : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  vecmatAccumSpecK (withKIndex s kIdx) AccPre A B dim_n dim_k N K
    BLOCK_N BLOCK_K i
```
</details>

<details><summary><code>vecmatAccumSpecK</code></summary>

```lean
noncomputable def vecmatAccumSpecK
    (s : BlockState) (AccPre A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun old delta => old + delta)
      (some (s.readMem AccPre (outOffset s dim_n BLOCK_N i)))
      ((Tile.reduceSum (shape := [BLOCK_N, BLOCK_K]) ⟨1, by simp⟩ Bool.false
        (vecmatProdTileK s A B dim_n dim_k N K BLOCK_N BLOCK_K)).data
          (i, PUnit.unit)))
```
</details>

<details><summary><code>withKIndex</code></summary>

```lean
def withKIndex (s : BlockState) (kIdx : Nat) : BlockState :=
  { s with pids := fun ax => if ax = 2 then kIdx else s.pids ax }
```
</details>

<details><summary><code>vecmatProdTileK</code></summary>

```lean
noncomputable def vecmatProdTileK
    (s : BlockState) (A B : RegionName)
    (dim_n dim_k N K BLOCK_N BLOCK_K : Nat) :
    Tile .real [BLOCK_N, BLOCK_K] :=
  { data := fun idx =>
      let ni := (TileShape.dropInsertedIndex [BLOCK_N] 1 1 (idx.1, 0, PUnit.unit)).1
      let kj := (TileShape.dropInsertedIndex [BLOCK_K] 0 1 (0, idx.2.1, PUnit.unit)).1
      Option.map₂ (fun a b => a * b)
        (if kIndex s BLOCK_K kj < K then
          some (s.readMem A (aOffsetK s dim_k BLOCK_K kj))
        else some (0.0 : ℝ))
        (if nIndex s BLOCK_N ni < N ∧ kIndex s BLOCK_K kj < K then
          some (s.readMem B (bOffsetK s dim_n dim_k BLOCK_N BLOCK_K ni kj))
        else some (0.0 : ℝ)) }
```
</details>

<details><summary><code>kIndex</code></summary>

```lean
def kIndex (s : BlockState) (BLOCK_K : Nat) (j : Fin BLOCK_K) : Nat :=
  s.pids 2 * BLOCK_K + j.val
```
</details>

<details><summary><code>aOffsetK</code></summary>

```lean
def aOffsetK (s : BlockState) (dim_k BLOCK_K : Nat) (j : Fin BLOCK_K) : Nat :=
  s.pids 0 * dim_k + kIndex s BLOCK_K j
```
</details>

<details><summary><code>bOffsetK</code></summary>

```lean
def bOffsetK (s : BlockState) (dim_n dim_k BLOCK_N BLOCK_K : Nat)
    (i : Fin BLOCK_N) (j : Fin BLOCK_K) : Nat :=
  s.pids 0 * dim_n * dim_k + nIndex s BLOCK_N i * dim_k +
    kIndex s BLOCK_K j
```
</details>

## Public theorem: `batched_vecmat_test_first_k_accum_slice_compute_correct`

<details><summary>docstring</summary>

```
/-- Python `test_vecmat` uses `K = 128` and `block_k = 64`, hence exactly two
loop iterations.  This names the first accumulator update (`k_index = 0`). -/
```
</details>

**Statement:**
```lean
theorem batched_vecmat_test_first_k_accum_slice_compute_correct
    (AccPre A B Out : RegionName)
    (dim_n dim_k N BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i)`

**Closed-form spec defs (transitive):** `outOffset`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset (s : BlockState) (dim_n BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * dim_n + (s.pids 1 * BLOCK_N + i.val)
```
</details>

## Public theorem: `batched_vecmat_test_second_k_accum_slice_compute_correct`

<details><summary>docstring</summary>

```
/-- Python `test_vecmat`'s second and final accumulator update
(`k_index = 1`) for `K = 128`, `block_k = 64`. -/
```
</details>

**Statement:**
```lean
theorem batched_vecmat_test_second_k_accum_slice_compute_correct
    (AccPre A B Out : RegionName)
    (dim_n dim_k N BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s dim_n BLOCK_N i)`

**Closed-form spec defs (transitive):** `outOffset`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset (s : BlockState) (dim_n BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * dim_n + (s.pids 1 * BLOCK_N + i.val)
```
</details>

## Public theorem: `batched_vecmat_block_output_store_slice_compute_correct`

**Statement:**
```lean
theorem batched_vecmat_block_output_store_slice_compute_correct
    (VecmatPre output : RegionName) (dim_n BLOCK_M BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        blockOutOffset s dim_n BLOCK_M BLOCK_N idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := batched_vecmat_block_output_store_slice VecmatPre output
        dim_n BLOCK_M BLOCK_N)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        some (output, blockOutOffset s dim_n BLOCK_M BLOCK_N idx))
      (expected := fun idx =>
        blockOutputStoreSpec s VecmatPre dim_n BLOCK_M BLOCK_N idx)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        blockOutOffset s dim_n BLOCK_M BLOCK_N idx)`

**Closed-form spec defs (transitive):** `blockOutOffset`, `batched_vecmat_block_output_store_slice`, `blockOutputStoreSpec`, `mIndex`, `nIndex`

<details><summary><code>blockOutOffset</code></summary>

```lean
def blockOutOffset (s : BlockState) (dim_n BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  mIndex s BLOCK_M idx.1 * dim_n + nIndex s BLOCK_N idx.2.1
```
</details>

<details><summary><code>batched_vecmat_block_output_store_slice</code></summary>

```
/-- Final vectorized `BLOCK_M × BLOCK_N` output store from the Python surface.

The Python wrapper asserts divisibility, so the final store is unmasked:
`tl.store(output + output_tile, vecmat)`. This slice materializes `vecmat` in
`VecmatPre` and proves the full two-dimensional writeback shape. -/
```
```lean
def batched_vecmat_block_output_store_slice
    (VecmatPre output : RegionName) (dim_n BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  m_index = tl.program_id(0)
  n_index = tl.program_id(1)
  offset_m = m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offset_n = n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  out_tile = offset_m[:, None] * $(dim_n) + offset_n[None, :]
  vecmat = tl.load(VecmatPre + out_tile)
  tl.store(output + out_tile, vecmat)
}
```
</details>

<details><summary><code>blockOutputStoreSpec</code></summary>

```lean
noncomputable def blockOutputStoreSpec
    (s : BlockState) (VecmatPre : RegionName) (dim_n BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  s.readMem VecmatPre (blockOutOffset s dim_n BLOCK_M BLOCK_N idx)
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>nIndex</code></summary>

```lean
def nIndex (s : BlockState) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + i.val
```
</details>

## Also present (pinned special-case summaries)
- `batched_vecmat_closed_form_correct`
