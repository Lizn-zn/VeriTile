# Spec sheet — `bench/tritonbench_g/batched_vecmat_mult/BatchedVecmatMult.lean`

**Python source:** `bench/tritonbench_g/batched_vecmat_mult/batched_vecmat_mult.py`

## Public theorem: `batched_vecmat_closed_form_correct`

<details><summary>docstring</summary>

```
/-- **Closed-form GEMV correctness for `batched_vecmat_mult` (general statement).**

For arbitrary `dim_n`, tile dims `BLOCK_M`/`BLOCK_N`, K-block size `BLOCK_K`, and
K-block count `numKBlocks` (so the contracted dim is `dim_k = BLOCK_K · numKBlocks`),
every output cell of the computed `BLOCK_M × BLOCK_N` tile equals the genuine
batched vector-matrix value `Σ_{kk < dim_k} A[m,kk] · B[m,n,kk]` (over ℝ) — NOT
the kernel's own executed value.

Layout: `A[m,kk]` at `A + m·dim_k + kk`, `B[m,n,kk]` at
`B + m·dim_n·dim_k + n·dim_k + kk`, `out[m,n]` at `output + m·dim_n + n` (the
kernel's row-major pointer arithmetic). Preconditions: `0 < BLOCK_M`,
`0 < BLOCK_N`, `0 < BLOCK_K`, output-offset injectivity, clean initial `undef`. -/
```
</details>

**Statement:**
```lean
specification batched_vecmat_closed_form_correct
    (A B output : RegionName) (s : BlockState)
    (_dim_m dim_n BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (hBM : 0 < BLOCK_M) (hBN : 0 < BLOCK_N) (hBK : 0 < BLOCK_K)
    (hInj : Function.Injective (vecmatOutOffset s dim_n BLOCK_M BLOCK_N))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := batched_vecmat_surface A B output _dim_m dim_n (BLOCK_K * numKBlocks)
        BLOCK_M BLOCK_N BLOCK_K)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        some (output, vecmatOutOffset s dim_n BLOCK_M BLOCK_N idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        gemvSpec s A B dim_n (BLOCK_K * numKBlocks) BLOCK_M BLOCK_N BLOCK_K numKBlocks
          idx.1 idx.2.1)
```

**Assumptions / layout contracts:**
- `hBM : 0 < BLOCK_M`
- `hBN : 0 < BLOCK_N`
- `hBK : 0 < BLOCK_K`
- `hInj : Function.Injective (vecmatOutOffset s dim_n BLOCK_M BLOCK_N)`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `vecmatOutOffset`, `batched_vecmat_surface`, `gemvSpec`, `gRow`, `gCol`, `aElem`, `bElem`

<details><summary><code>vecmatOutOffset</code></summary>

```
/-- The output store address for tile lane `(i,j)`: `gRow i · dim_n + gCol j`. -/
```
```lean
def vecmatOutOffset (s : BlockState) (dim_n BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  gRow s BLOCK_M idx.1 * dim_n + gCol s BLOCK_N idx.2.1
```
</details>

<details><summary><code>batched_vecmat_surface</code></summary>

```
/-- Faithful transcription of `batched_vecmat_mult.py`'s `batched_vecmat_kernel`.

The Python wrapper asserts that `M`, `N`, and `K` are divisible by their block
sizes, so this surface keeps the same unmasked block loads and stores. The
Python body vectorizes the `block_m` rows and writes the reduction as
`tl.broadcast(a, b)` followed by `tl.trans(tl.sum(..., axis=2))`.

Allowed mechanical Lean-syntax-only changes apply. -/
```
```lean
def batched_vecmat_surface
    (A B output : RegionName)
    (_dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  m_index = tl.program_id(0)
  n_index = tl.program_id(1)
  output_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[:, None] * $(dim_n) +
    (n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N)))[None, :]
  vecmat = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=A.dtype.element_ty)
  k_blocks = $(dim_k) // $(BLOCK_K)
  for k_index in range(k_blocks) {
    a_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[:, None] * $(dim_k) +
      (k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K)))[None, :]
    a = tl.load(A + a_tile)
    b_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[None, :, None] *
      $(dim_n) * $(dim_k) +
      (n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N)))[:, None, None] * $(dim_k) +
      (k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K)))[None, None, :]
    b = tl.load(B + b_tile)
    expanded_a, _ = tl.broadcast(a, b)
    vecmat += tl.trans(tl.sum(expanded_a * b, axis=2))
  }
  tl.store(output + output_tile, vecmat)
}
```
</details>

<details><summary><code>gemvSpec</code></summary>

```
/-- **Genuine GEMV spec**: `out[i,j] = Σ_{kk < BLOCK_K·numKBlocks} A[i,kk] · B[i,j,kk]`. -/
```
```lean
noncomputable def gemvSpec (s : BlockState) (A B : RegionName)
    (dim_n dim_k BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (i : Fin BLOCK_M) (j : Fin BLOCK_N) : ℝ :=
  (Finset.range (BLOCK_K * numKBlocks)).sum
    (fun kk => aElem s A dim_k BLOCK_M i kk
      * bElem s B dim_n dim_k BLOCK_M BLOCK_N i j kk)
```
</details>

<details><summary><code>gRow</code></summary>

```
/-- Global output row of tile lane `i`: `m_index · BLOCK_M + i`. -/
```
```lean
def gRow (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>gCol</code></summary>

```
/-- Global output column of tile lane `j`: `n_index · BLOCK_N + j`. -/
```
```lean
def gCol (s : BlockState) (BLOCK_N : Nat) (j : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + j.val
```
</details>

<details><summary><code>aElem</code></summary>

```
/-- `A[m, kk] = readMem A (gRow i · dim_k + kk)` (the kernel's row-major A layout). -/
```
```lean
noncomputable def aElem (s : BlockState) (A : RegionName) (dim_k BLOCK_M : Nat)
    (i : Fin BLOCK_M) (kk : Nat) : ℝ :=
  s.readMem A (gRow s BLOCK_M i * dim_k + kk)
```
</details>

<details><summary><code>bElem</code></summary>

```
/-- `B[m, n, kk] = readMem B (gRow i · dim_n · dim_k + gCol j · dim_k + kk)`
(the kernel's row-major B layout). -/
```
```lean
noncomputable def bElem (s : BlockState) (B : RegionName)
    (dim_n dim_k BLOCK_M BLOCK_N : Nat)
    (i : Fin BLOCK_M) (j : Fin BLOCK_N) (kk : Nat) : ℝ :=
  s.readMem B (gRow s BLOCK_M i * dim_n * dim_k + gCol s BLOCK_N j * dim_k + kk)
```
</details>

## Public theorem: `batched_vecmat_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S1 fold genre).** For every
rounding model `R`, the faithful `batched_vecmat_mult` surface implements, on
its `StreamMasked2DKernelIO₂` signature, the **ideal ℝ batched GEMV fold**
over the streamed tiles: output lane `l = (i, j)` holds
`∑ t, ∑ e, A-tile[t](i,e) · B-tile[t](j,i,e)` — the spec `f` is exact real
arithmetic. The kernel has **no rounding events** (both region loads, the
`expanded_a * b` product, the axis-2 `tl.sum`, the `tl.trans` and the store
are all at `.real`), so the skin's boundary quantization degenerates: the
readback contract's `R.round .real` is the identity by the model's defining
`round_real`, and the `.real` terminal store is exact under `execR R`
(`RoundingModel.storeValue_real`).

Layer map: the 5-statement prologue and the whole dynamic K-loop are
cast-free, so under `execR R` they collapse verbatim onto the exact stepper
and the proven `vecmat_preLoop` / `vecmat_step` / `forRangeDyn_inv` stack
above is reused unchanged; only the single store statement is re-proved on
the `R` side (`vecmat_postLoopR`).

Both hypotheses are truth-forced:

* `hBK : 0 < BLOCK_K` — the loop trip count is the runtime register
  `k_blocks = dim_k // block_k`, which is `numKBlocks` at
  `dim_k = BLOCK_K · numKBlocks` only for a nonzero block size; at
  `BLOCK_K = 0` the division is `0` and the loop never runs, so the fold
  would be empty while the spec sums nothing either — but the block index
  arithmetic `l / BLOCK_K` is meaningless. It holds for every real launch
  (`tl.arange(0, 0)` is not a tile, and the Python wrapper asserts
  `K % block_k == 0`).
* `hBN : BLOCK_N ≤ dim_n` — output-offset injectivity (`rowMajor2D_inj`):
  the output tile's column width must fit the row stride `dim_n`, otherwise
  distinct lanes `(i, j)` collide at `i·dim_n + j` and the per-lane readback
  would be last-writer-wins, making the statement false. It holds for every
  valid tiling (the Python wrapper asserts `N % block_n == 0` with the grid
  covering `N`).

Relation to the exact surface: the exact headline
`batched_vecmat_closed_form_correct` (`Realizes_without_Rounding`) above is
retained unchanged; this `⊨[R]` face restates the same GEMV content on the
streaming skin, for every `R` at once. Both faces are kept per the
rounding-as-default doctrine. -/
```
</details>

**Statement:**
```lean
specification batched_vecmat_io_correctness (R : RoundingModel)
    (A B output : RegionName)
    (_dim_m dim_n BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (hBK : 0 < BLOCK_K) (hBN : BLOCK_N ≤ dim_n) :
    batchedVecmatIO A B output _dim_m dim_n BLOCK_M BLOCK_N BLOCK_K numKBlocks
      ⊨[R] fun _ _ xs ys l =>
        ∑ t : Fin numKBlocks, ∑ e : Fin BLOCK_K,
          xs t (aLane BLOCK_M BLOCK_N BLOCK_K l e) * ys t (bLane BLOCK_M BLOCK_N BLOCK_K l e)
```

**Assumptions / layout contracts:**
- `hBK : 0 < BLOCK_K`
- `hBN : BLOCK_N ≤ dim_n`

**Closed-form spec defs (transitive):** `batchedVecmatIO`, `aLane`, `bLane`, `batched_vecmat_surface`

<details><summary><code>batchedVecmatIO</code></summary>

```
/-- **Streaming IO signature** of `batched_vecmat_mult` on the two-stream
fold skin (S1: fold + terminal store). Step `t` of the K-loop reads the
`[BLOCK_M, BLOCK_K]` `A`-tile and the `[BLOCK_N, BLOCK_M, BLOCK_K]` `B`-tile;
after the loop one `[BLOCK_M, BLOCK_N]` output tile is stored at the
**`.real`** grid (`outDType` default — the kernel's
`tl.store(output + output_tile, vecmat)` is untyped, so the terminal store
has no quantization event). The kernel uses a genuine **2-D** pid grid, so
`pid₀ = m_index` and `pid₁ = n_index` both feed the windows:

* `read1` lane `l = (i, e)` (row-major over `[BLOCK_M, BLOCK_K]`), step `t`:
  `(pid₀·BLOCK_M + i)·dim_k + (t·BLOCK_K + e)` — the kernel's `a_tile`.
* `read2` lane `l = (j, i, e)` (row-major over
  `[BLOCK_N, BLOCK_M, BLOCK_K]`, i.e. `j = l/BLOCK_K/BLOCK_M`,
  `i = l/BLOCK_K % BLOCK_M`, `e = l % BLOCK_K`), step `t`:
  `(pid₀·BLOCK_M + i)·dim_n·dim_k + (pid₁·BLOCK_N + j)·dim_k + (t·BLOCK_K + e)`
  — the kernel's `b_tile`.
* `write` lane `l = (i, j)`:
  `(pid₀·BLOCK_M + i)·dim_n + (pid₁·BLOCK_N + j)` — the kernel's
  `output_tile` (= `vecmatOutOffset` in pid form).

The kernel has no masks (the Python wrapper asserts exact divisibility): all
windows are `True`. -/
```
```lean
def batchedVecmatIO (A B output : RegionName)
    (_dim_m dim_n BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat) :
    StreamMasked2DKernelIO₂ where
  kernel := batched_vecmat_surface A B output _dim_m dim_n (BLOCK_K * numKBlocks)
    BLOCK_M BLOCK_N BLOCK_K
  inp1 := A
  inp2 := B
  out := output
  T := numKBlocks
  B1 := BLOCK_M * BLOCK_K
  B2 := BLOCK_N * BLOCK_M * BLOCK_K
  C := BLOCK_M * BLOCK_N
  read1 := fun p₀ _ t l =>
    (p₀ * BLOCK_M + l.val / BLOCK_K) * (BLOCK_K * numKBlocks)
      + (t.val * BLOCK_K + l.val % BLOCK_K)
  read2 := fun p₀ p₁ t l =>
    (p₀ * BLOCK_M + l.val / BLOCK_K % BLOCK_M) * dim_n * (BLOCK_K * numKBlocks)
      + (p₁ * BLOCK_N + l.val / BLOCK_K / BLOCK_M) * (BLOCK_K * numKBlocks)
      + (t.val * BLOCK_K + l.val % BLOCK_K)
  write := fun p₀ p₁ l =>
    (p₀ * BLOCK_M + l.val / BLOCK_N) * dim_n + (p₁ * BLOCK_N + l.val % BLOCK_N)
  mask1 := fun _ _ _ _ => True
  mask2 := fun _ _ _ _ => True
  writeMask := fun _ _ _ => True
```
</details>

<details><summary><code>aLane</code></summary>

```
/-- The `A`-stream lane feeding output lane `l` at inner key `e`: the row of
`l` (row-major over the `[BLOCK_M, BLOCK_N]` output tile) paired with `e`
over the `[BLOCK_M, BLOCK_K]` per-step `A`-tile. -/
```
```lean
def aLane (BLOCK_M BLOCK_N BLOCK_K : Nat) (l : Fin (BLOCK_M * BLOCK_N)) (e : Fin BLOCK_K) :
    Fin (BLOCK_M * BLOCK_K) :=
  Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit)
```
</details>

<details><summary><code>bLane</code></summary>

```
/-- The `B`-stream lane feeding output lane `l` at inner key `e`: the
`[BLOCK_N, BLOCK_M, BLOCK_K]` cell `(j, i, e)` flattened row-major through
two nested `Lane2D` bridges — the `(j, i)` pair first (into
`Fin (BLOCK_N * BLOCK_M)`), then the key. -/
```
```lean
def bLane (BLOCK_M BLOCK_N BLOCK_K : Nat) (l : Fin (BLOCK_M * BLOCK_N)) (e : Fin BLOCK_K) :
    Fin (BLOCK_N * BLOCK_M * BLOCK_K) :=
  Lane2D.encode
    (Lane2D.encode ((Lane2D.decode l).2.1, (Lane2D.decode l).1, PUnit.unit), e, PUnit.unit)
```
</details>

<details><summary><code>batched_vecmat_surface</code></summary>

```
/-- Faithful transcription of `batched_vecmat_mult.py`'s `batched_vecmat_kernel`.

The Python wrapper asserts that `M`, `N`, and `K` are divisible by their block
sizes, so this surface keeps the same unmasked block loads and stores. The
Python body vectorizes the `block_m` rows and writes the reduction as
`tl.broadcast(a, b)` followed by `tl.trans(tl.sum(..., axis=2))`.

Allowed mechanical Lean-syntax-only changes apply. -/
```
```lean
def batched_vecmat_surface
    (A B output : RegionName)
    (_dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  m_index = tl.program_id(0)
  n_index = tl.program_id(1)
  output_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[:, None] * $(dim_n) +
    (n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N)))[None, :]
  vecmat = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=A.dtype.element_ty)
  k_blocks = $(dim_k) // $(BLOCK_K)
  for k_index in range(k_blocks) {
    a_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[:, None] * $(dim_k) +
      (k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K)))[None, :]
    a = tl.load(A + a_tile)
    b_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[None, :, None] *
      $(dim_n) * $(dim_k) +
      (n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N)))[:, None, None] * $(dim_k) +
      (k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K)))[None, None, :]
    b = tl.load(B + b_tile)
    expanded_a, _ = tl.broadcast(a, b)
    vecmat += tl.trans(tl.sum(expanded_a * b, axis=2))
  }
  tl.store(output + output_tile, vecmat)
}
```
</details>

## Also present (pinned special-case summaries)
- `batched_vecmat_one_row_block_compute_correct`
- `batched_vecmat_one_row_k_block_compute_correct`
- `batched_vecmat_one_row_k_accum_slice_compute_correct`
- `batched_vecmat_one_row_const_k_accum_slice_compute_correct`
- `batched_vecmat_test_first_k_accum_slice_compute_correct`
- `batched_vecmat_test_second_k_accum_slice_compute_correct`
- `batched_vecmat_block_output_store_slice_compute_correct`
