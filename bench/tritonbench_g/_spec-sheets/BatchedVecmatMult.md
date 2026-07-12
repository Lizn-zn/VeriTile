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

## Also present (pinned special-case summaries)
- `batched_vecmat_one_row_block_compute_correct`
- `batched_vecmat_one_row_k_block_compute_correct`
- `batched_vecmat_one_row_k_accum_slice_compute_correct`
- `batched_vecmat_one_row_const_k_accum_slice_compute_correct`
- `batched_vecmat_test_first_k_accum_slice_compute_correct`
- `batched_vecmat_test_second_k_accum_slice_compute_correct`
- `batched_vecmat_block_output_store_slice_compute_correct`
