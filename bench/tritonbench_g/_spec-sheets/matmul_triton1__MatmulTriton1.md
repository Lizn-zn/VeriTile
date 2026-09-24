# Spec sheet — `bench/tritonbench_g/matmul_triton1/MatmulTriton1.lean`

**Python source:** `bench/tritonbench_g/matmul_triton1/matmul_triton1.py`

## Public theorem: `matmul_triton1_closed_form_correct`

<details><summary>docstring</summary>

```
/-- **Closed-form correctness for `matmul_triton1` (general statement).**

For arbitrary `n_size`, tile dims `M`/`N`, K-block size `BLOCK_K`, and K-block
count `numKBlocks` (so `k_size = BLOCK_K · numKBlocks`), every output cell of the
computed `M × N` tile equals the genuine matrix product
`Σ_{k < BLOCK_K·numKBlocks} X[i,k] · Y[k,j]` (over ℝ) of the loaded `X`/`Y`
tiles — NOT the kernel's own executed value.

Layout: `X[i,k]` at `X + rowIndex i · k_size + k`, `Y[k,j]` at
`Y + k · n_size + colIndex j`, `Z[i,j]` at `Z + rowIndex i · n_size + colIndex j`
(the kernel's row-major pointer arithmetic). Preconditions: `0 < BLOCK_K` and
`N ≤ NS` (tile width `BLOCK_N` ≤ row stride `n_size`, always true for a valid
tiling — this discharges output-offset injectivity via `zOffset_injective_of_le`),
plus a clean initial `undef`. -/
```
</details>

**Statement:**
```lean
specification matmul_triton1_closed_form_correct
    (X Y Z : RegionName) (s : BlockState)
    (NS M BLOCK_K N numKBlocks : Nat) (hBK : 0 < BLOCK_K)
    (hN : N ≤ NS)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := matmul_triton1_surface X Y Z 0 (BLOCK_K * numKBlocks) NS M BLOCK_K N)
      (initialState := s)
      (write := fun idx : TileIndex [M, N] => some (Z, zOffset s NS N M idx))
      (expected := fun idx : TileIndex [M, N] =>
        matmulSpec s X Y (BLOCK_K * numKBlocks) NS N M N BLOCK_K numKBlocks idx.1 idx.2.1)
```

**Assumptions / layout contracts:**
- `hBK : 0 < BLOCK_K`
- `hN : N ≤ NS`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `matmul_triton1_surface`, `zOffset`, `matmulSpec`, `rowIndex`, `colIndex`, `xElem`, `yElem`, `numNBlocks`

<details><summary><code>matmul_triton1_surface</code></summary>

```
/-- Faithful transcription of `matmul_triton1.py`'s `matmul_kernel`.

Python passes `m_size` but the kernel body does not use it; this surface keeps
the signature position as `_m_size`. -/
```
```lean
def matmul_triton1_surface
    (X Y Z : RegionName)
    (_m_size k_size n_size m_block_size k_block_size n_block_size : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  num_n_blocks = tl.cdiv($(n_size), $(n_block_size))
  m_block = pid // num_n_blocks
  n_block = pid % num_n_blocks
  m_offsets = tl.arange(0, $(m_block_size)) + m_block * $(m_block_size)
  n_offsets = tl.arange(0, $(n_block_size)) + n_block * $(n_block_size)
  k_offsets = tl.arange(0, $(k_block_size))
  x_ptrs = X + m_offsets[:, None] * $(k_size) + k_offsets[None, :]
  y_ptrs = Y + k_offsets[:, None] * $(n_size) + n_offsets[None, :]
  z_ptrs = Z + m_offsets[:, None] * $(n_size) + n_offsets[None, :]
  z = tl.zeros([$(m_block_size), $(n_block_size)], dtype=tl.float32)
  for kk in range($(0), $(k_size), $(k_block_size)) {
    x_sub = tl.load(x_ptrs)
    y_sub = tl.load(y_ptrs)
    z += tl.dot(x_sub, y_sub, allow_tf32=false)
    x_ptrs += $(k_block_size)
    y_ptrs += $(k_block_size) * $(n_size)
  }
  tl.store(z_ptrs, z)
}
```
</details>

<details><summary><code>zOffset</code></summary>

```
/-- The output store address for tile lane `(i,j)`: `rowIndex i · NS + colIndex j`. -/
```
```lean
def zOffset (s0 : BlockState) (NS N M : Nat) (idx : TileIndex [M, N]) : Nat :=
  rowIndex s0 NS N M idx.1 * NS + colIndex s0 NS N N idx.2.1
```
</details>

<details><summary><code>matmulSpec</code></summary>

```
/-- **Genuine GEMM spec**: `C[i,j] = Σ_{k < BLOCK_K·numKBlocks} X[i,k] · Y[k,j]`,
an instance of the shared `gemmSum` (`Kernel.Matmul`) with this kernel's `X`/`Y`
layout accessors. -/
```
```lean
noncomputable def matmulSpec (s : BlockState) (X Y : RegionName)
    (KS NS NB M N BLOCK_K numKBlocks : Nat) (i : Fin M) (j : Fin N) : ℝ :=
  gemmSum (xElem s X KS NS NB M i) (yElem s Y NS NB N j) (BLOCK_K * numKBlocks)
```
</details>

<details><summary><code>rowIndex</code></summary>

```
/-- Global output row of tile lane `i`: `(pid / num_n_blocks) · M + i`. -/
```
```lean
def rowIndex (s : BlockState) (NS NB M : Nat) (i : Fin M) : Nat :=
  s.pids 0 / numNBlocks NS NB * M + i.val
```
</details>

<details><summary><code>colIndex</code></summary>

```
/-- Global output column of tile lane `j`: `(pid % num_n_blocks) · N + j`. -/
```
```lean
def colIndex (s : BlockState) (NS NB N : Nat) (j : Fin N) : Nat :=
  s.pids 0 % numNBlocks NS NB * N + j.val
```
</details>

<details><summary><code>xElem</code></summary>

```
/-- `X[i, k] = readMem X (rowIndex i · KS + k)` (kernel's row-major X layout). -/
```
```lean
noncomputable def xElem (s : BlockState) (X : RegionName) (KS NS NB M : Nat)
    (i : Fin M) (k : Nat) : ℝ :=
  s.readMem X (rowIndex s NS NB M i * KS + k)
```
</details>

<details><summary><code>yElem</code></summary>

```
/-- `Y[k, j] = readMem Y (k · NS + colIndex j)` (kernel's row-major Y layout). -/
```
```lean
noncomputable def yElem (s : BlockState) (Y : RegionName) (NS NB N : Nat)
    (j : Fin N) (k : Nat) : ℝ :=
  s.readMem Y (k * NS + colIndex s NS NB N j)
```
</details>

<details><summary><code>numNBlocks</code></summary>

```
/-- `num_n_blocks = cdiv n_size n_block_size` (the kernel's tile column count). -/
```
```lean
def numNBlocks (n_size n_block_size : Nat) : Nat := cdiv n_size n_block_size
```
</details>

## Public theorem: `matmul_triton1_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S1 fold genre).** For every
rounding model `R`, the faithful `matmul_triton1` surface implements, on its
`StreamMasked2DKernelIO₂` signature, the **ideal ℝ GEMM fold** over the
streamed tiles: output lane `l = (i, j)` holds
`∑ t, ∑ e, X-tile[t](i,e) · Y-tile[t](e,j)` — the spec `f` is exact real
arithmetic. The kernel has **no rounding events** (loads, dot and store are
all at `.real`), so the skin's boundary quantization degenerates: the
readback contract's `R.round .real` is the identity by the model's defining
`round_real`, and the `.real` terminal store is exact under `execR R`
(`stepStmtR` delegates `.real` writes to the exact `writeMemTyped`;
`RoundingModel.storeValue_real`).

Layer map: the prologue and the whole K-loop are cast-free, so under
`execR R` they collapse verbatim onto the exact stepper and the proven
`preLoop` / `matmul_step` / `forRange_inv` stack above is reused unchanged;
only the terminal store is re-proved on the `R` side (`matmul_postLoopR`).

Both hypotheses are truth-forced:

* `hBK : 0 < BLOCK_K` — the surface's K-loop steps by `BLOCK_K`
  (`range(0, k_size, k_block_size)`); at `BLOCK_K = 0` the loop never
  advances, `execR` diverges from the invariant stack (`forRange_inv`
  requires a nonzero step), and the block-index arithmetic `i / BLOCK_K` is
  meaningless. It holds for every real launch (`tl.arange(0, 0)` is not a
  tile).
* `hN : N ≤ NS` — the output window `row·n_size + col` is injective only
  when the tile width `n_block_size` fits the row stride `n_size`
  (`zOffset_injective_of_le` / `rowMajor2D_inj`, exactly the exact
  headline's precondition); with colliding output lanes the per-lane
  readback would be last-writer-wins and the statement false. It holds for
  every valid tiling.

Relation to the exact surface: the exact headline
`matmul_triton1_closed_form_correct` (`Realizes_without_Rounding`) above is
retained unchanged; this `⊨[R]` face restates the same GEMM content on the
streaming skin, for every `R` at once (at the `.real` grid the two faces
carry the same exact cell). Both faces are kept per the rounding-as-default
doctrine. -/
```
</details>

**Statement:**
```lean
specification matmul_triton1_io_correctness (R : RoundingModel)
    (X Y Z : RegionName) (NS M BLOCK_K N numKBlocks : Nat)
    (hBK : 0 < BLOCK_K) (hN : N ≤ NS) :
    matmulTriton1IO X Y Z NS M BLOCK_K N numKBlocks ⊨[R] fun _ _ xs ys l =>
      ∑ t : Fin numKBlocks, ∑ e : Fin BLOCK_K,
        xs t (aLane M N BLOCK_K l e) * ys t (bLane M N BLOCK_K l e)
```

**Assumptions / layout contracts:**
- `hBK : 0 < BLOCK_K`
- `hN : N ≤ NS`

**Closed-form spec defs (transitive):** `matmulTriton1IO`, `aLane`, `bLane`, `matmul_triton1_surface`, `numNBlocks`

<details><summary><code>matmulTriton1IO</code></summary>

```
/-- **Streaming IO signature** of `matmul_triton1` on the two-stream fold
skin (S1: fold + terminal store). Step `c` of the K-loop reads the
`[M, BLOCK_K]` `X`-tile and the `[BLOCK_K, N]` `Y`-tile; after the loop one
`[M, N]` output tile is stored at the **`.real`** grid (`outDType` default —
the kernel's store is untyped `tl.store(z_ptrs, z)` at `.real`, so the
terminal store has no quantization event). The kernel uses only
`program_id(0)`: the `(m, n)` tile coordinate is the linear split
`pid₀ / num_n_blocks` × `pid₀ % num_n_blocks` (with
`num_n_blocks = cdiv n_size n_block_size = numNBlocks NS N`), so all three
windows read `pid₀` only. They transcribe the kernel's pointer arithmetic
exactly:

* `read1` lane `l = (i, e)` (row-major over `[M, BLOCK_K]`), step `t`:
  `(pid₀/num_n_blocks·M + i) · k_size + (t·BLOCK_K + e)` with
  `k_size = BLOCK_K · numKBlocks` — the invariant's `x_ptrs` cell
  `rowIndex(i)·k_size + e + t·BLOCK_K` after `t` advances.
* `read2` lane `l = (e, j)` (row-major over `[BLOCK_K, N]`), step `t`:
  `(t·BLOCK_K + e) · n_size + (pid₀%num_n_blocks·N + j)` — the `y_ptrs`
  cell after `t` advances of `BLOCK_K·n_size`.
* `write` lane `l = (i, j)`:
  `(pid₀/num_n_blocks·M + i) · n_size + (pid₀%num_n_blocks·N + j)` — the
  kernel's `z_ptrs` (= `zOffset` in pid form).

The kernel has no masks: all windows are `True`. -/
```
```lean
def matmulTriton1IO (X Y Z : RegionName) (NS M BLOCK_K N numKBlocks : Nat) :
    StreamMasked2DKernelIO₂ where
  kernel := matmul_triton1_surface X Y Z 0 (BLOCK_K * numKBlocks) NS M BLOCK_K N
  inp1 := X
  inp2 := Y
  out := Z
  T := numKBlocks
  B1 := M * BLOCK_K
  B2 := BLOCK_K * N
  C := M * N
  read1 := fun p₀ _ t l =>
    (p₀ / numNBlocks NS N * M + l.val / BLOCK_K) * (BLOCK_K * numKBlocks)
      + (t.val * BLOCK_K + l.val % BLOCK_K)
  read2 := fun p₀ _ t l =>
    (t.val * BLOCK_K + l.val / N) * NS + (p₀ % numNBlocks NS N * N + l.val % N)
  write := fun p₀ _ l =>
    (p₀ / numNBlocks NS N * M + l.val / N) * NS + (p₀ % numNBlocks NS N * N + l.val % N)
  mask1 := fun _ _ _ _ => True
  mask2 := fun _ _ _ _ => True
  writeMask := fun _ _ _ => True
```
</details>

<details><summary><code>aLane</code></summary>

```
/-- The `X`-stream lane feeding output lane `l` at inner key `e`: the row of
`l` (row-major over the `[M, N]` output tile) paired with `e` over the
`[M, BLOCK_K]` per-step `X`-tile, both via the shared `Lane2D` bridge. -/
```
```lean
def aLane (M N BK : Nat) (l : Fin (M * N)) (e : Fin BK) : Fin (M * BK) :=
  Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit)
```
</details>

<details><summary><code>bLane</code></summary>

```
/-- The `Y`-stream lane feeding output lane `l` at inner key `e`: `e` paired
with the column of `l` over the `[BLOCK_K, N]` per-step `Y`-tile. -/
```
```lean
def bLane (M N BK : Nat) (l : Fin (M * N)) (e : Fin BK) : Fin (BK * N) :=
  Lane2D.encode (e, (Lane2D.decode l).2.1, PUnit.unit)
```
</details>

<details><summary><code>matmul_triton1_surface</code></summary>

```
/-- Faithful transcription of `matmul_triton1.py`'s `matmul_kernel`.

Python passes `m_size` but the kernel body does not use it; this surface keeps
the signature position as `_m_size`. -/
```
```lean
def matmul_triton1_surface
    (X Y Z : RegionName)
    (_m_size k_size n_size m_block_size k_block_size n_block_size : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  num_n_blocks = tl.cdiv($(n_size), $(n_block_size))
  m_block = pid // num_n_blocks
  n_block = pid % num_n_blocks
  m_offsets = tl.arange(0, $(m_block_size)) + m_block * $(m_block_size)
  n_offsets = tl.arange(0, $(n_block_size)) + n_block * $(n_block_size)
  k_offsets = tl.arange(0, $(k_block_size))
  x_ptrs = X + m_offsets[:, None] * $(k_size) + k_offsets[None, :]
  y_ptrs = Y + k_offsets[:, None] * $(n_size) + n_offsets[None, :]
  z_ptrs = Z + m_offsets[:, None] * $(n_size) + n_offsets[None, :]
  z = tl.zeros([$(m_block_size), $(n_block_size)], dtype=tl.float32)
  for kk in range($(0), $(k_size), $(k_block_size)) {
    x_sub = tl.load(x_ptrs)
    y_sub = tl.load(y_ptrs)
    z += tl.dot(x_sub, y_sub, allow_tf32=false)
    x_ptrs += $(k_block_size)
    y_ptrs += $(k_block_size) * $(n_size)
  }
  tl.store(z_ptrs, z)
}
```
</details>

<details><summary><code>numNBlocks</code></summary>

```
/-- `num_n_blocks = cdiv n_size n_block_size` (the kernel's tile column count). -/
```
```lean
def numNBlocks (n_size n_block_size : Nat) : Nat := cdiv n_size n_block_size
```
</details>
