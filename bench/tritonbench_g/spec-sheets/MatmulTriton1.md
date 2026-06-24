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
theorem matmul_triton1_closed_form_correct
    (X Y Z : RegionName) (s : BlockState)
    (NS M BLOCK_K N numKBlocks : Nat) (hBK : 0 < BLOCK_K)
    (hN : N ≤ NS)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
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
