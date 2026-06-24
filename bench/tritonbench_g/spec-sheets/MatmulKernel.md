# Spec sheet — `bench/tritonbench_g/matmul_kernel/MatmulKernel.lean`

**Python source:** `bench/tritonbench_g/matmul_kernel/matmul_kernel.py`

## Public theorem: `matmul_kernel_closed_form_correct`

<details><summary>docstring</summary>

```
/-- **Closed-form correctness for `matmul_kernel` (general statement).**

For arbitrary 2-D program coordinates `(pid_m, pid_n)`, tile dims `BM`/`BN`,
K-block size `BLOCK_K`, and K-block count `numKBlocks` (so the contracted
dimension is `K = BLOCK_K · numKBlocks`), every output cell of the computed
`BM × BN` tile equals `fp16( Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j] )` — the
genuine matrix product (over ℝ) of the loaded `A`/`B` tiles, cast to float16 —
**not** the kernel's own executed value.

Layout: `A[i,k]` at `A + offs_am(i) · 4096 + k`, `B[k,j]` at
`B + k · 4096 + offs_bn(j)`, `C[i,j]` at `C + 4096 · offs_cm(i) + offs_cn(j)`,
with `offs_am(i) = (pid_m·BM + i) % 4096`, `offs_bn(j) = (pid_n·BN + j) % 4096`
(the kernel's row-major pointer arithmetic with `% 4096` wrap). Preconditions:
`BN ≤ 4096` (column-block width ≤ the row stride 4096, always true for a valid
tiling — this discharges output-offset injectivity via
`matmul_kernel_output_offset_injective`) and clean initial `undef`. -/
```
</details>

**Statement:**
```lean
theorem matmul_kernel_closed_form_correct
    (C A B : RegionName) (s : BlockState)
    (BM BN BLOCK_K numKBlocks : Nat)
    (hBN : BN ≤ 4096)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := matmul_kernel_surface C A B BM BN BLOCK_K numKBlocks)
      (initialState := s)
      (write := fun idx : TileIndex [BM, BN] => some (C, cOffset s BM BN idx))
      (expected := fun idx : TileIndex [BM, BN] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (matmulSpec s A B BM BN BLOCK_K numKBlocks idx.1 idx.2.1))))
```

**Assumptions / layout contracts:**
- `hBN : BN ≤ 4096`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `matmul_kernel_surface`, `cOffset`, `matmulSpec`, `rowGlobal`, `colGlobal`, `aElem`, `bElem`, `rowIndex`, `colIndex`

<details><summary><code>matmul_kernel_surface</code></summary>

```
/-- Faithful transcription of `matmul_kernel.py`'s `matmul_kernel`.

The Python kernel hard-codes `M = N = K = 4096` and contiguous strides; here `K`
is presented as `BLOCK_SIZE_K · numKBlocks` so the loop trip count
`cdiv(K, BLOCK_SIZE_K) = numKBlocks` is exact, and the loop bound is supplied as
the antiquoted `numKBlocks` (`= tl.cdiv(4096, BLOCK_SIZE_K)` for the modeled
block shapes). All other surface structure — the `% 4096` index wrap, the fused
`tl.dot(a, b, accumulator)`, the `float16` cast, and the row-major store — is
transcribed verbatim. -/
```
```lean
def matmul_kernel_surface
    (C A B : RegionName) (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K numKBlocks : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(axis=0)
  pid_n = tl.program_id(axis=1)
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(4096)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(4096)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + (offs_am[:, None] * $(4096) + offs_k[None, :] * $(1))
  b_ptrs = B + (offs_k[:, None] * $(4096) + offs_bn[None, :] * $(1))
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += $(BLOCK_SIZE_K) * $(1)
    b_ptrs += $(BLOCK_SIZE_K) * $(4096)
  }
  c = tl.cast(accumulator, tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(4096) * offs_cm[:, None] + $(1) * offs_cn[None, :]
  tl.store(c_ptrs, c)
}
```
</details>

<details><summary><code>cOffset</code></summary>

```
/-- The output store address for tile lane `(i,j)`: `4096 · rowGlobal i + colGlobal j`
(the kernel's `c_ptrs`, which uses the **un-wrapped** `offs_cm` / `offs_cn`). -/
```
```lean
def cOffset (s0 : BlockState) (BM BN : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  4096 * rowGlobal s0 BM idx.1 + 1 * colGlobal s0 BN idx.2.1
```
</details>

<details><summary><code>matmulSpec</code></summary>

```
/-- **Genuine GEMM spec** (over ℝ): `C[i,j] = Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j]`,
an instance of the shared `gemmSum` (`Math.Matmul`) with this kernel's `A`/`B`
layout accessors. -/
```
```lean
noncomputable def matmulSpec (s : BlockState) (A B : RegionName)
    (BM BN BLOCK_K numKBlocks : Nat) (i : Fin BM) (j : Fin BN) : ℝ :=
  gemmSum (aElem s A BM i) (bElem s B BN j) (BLOCK_K * numKBlocks)
```
</details>

<details><summary><code>rowGlobal</code></summary>

```
/-- Global output row of tile lane `i` for program `(pid_m, pid_n)`:
`pid_m · BLOCK_M + i`, **before** the `% 4096` wrap. -/
```
```lean
def rowGlobal (s : BlockState) (BM : Nat) (i : Fin BM) : Nat :=
  s.pids 0 * BM + i.val
```
</details>

<details><summary><code>colGlobal</code></summary>

```
/-- Global output column of tile lane `j`: `pid_n · BLOCK_N + j`, before wrap. -/
```
```lean
def colGlobal (s : BlockState) (BN : Nat) (j : Fin BN) : Nat :=
  s.pids 1 * BN + j.val
```
</details>

<details><summary><code>aElem</code></summary>

```
/-- `A[i, k] = readMem A (offs_am i · 4096 + k)` (kernel's row-major A layout). -/
```
```lean
noncomputable def aElem (s : BlockState) (A : RegionName) (BM : Nat)
    (i : Fin BM) (k : Nat) : ℝ :=
  s.readMem A (rowIndex s BM i * 4096 + k)
```
</details>

<details><summary><code>bElem</code></summary>

```
/-- `B[k, j] = readMem B (k · 4096 + offs_bn j)` (kernel's row-major B layout). -/
```
```lean
noncomputable def bElem (s : BlockState) (B : RegionName) (BN : Nat)
    (j : Fin BN) (k : Nat) : ℝ :=
  s.readMem B (k * 4096 + colIndex s BN j)
```
</details>

<details><summary><code>rowIndex</code></summary>

```
/-- The `% 4096`-wrapped A-row index of tile lane `i` (the kernel's `offs_am`). -/
```
```lean
def rowIndex (s : BlockState) (BM : Nat) (i : Fin BM) : Nat :=
  rowGlobal s BM i % 4096
```
</details>

<details><summary><code>colIndex</code></summary>

```
/-- The `% 4096`-wrapped B-column index of tile lane `j` (the kernel's `offs_bn`). -/
```
```lean
def colIndex (s : BlockState) (BN : Nat) (j : Fin BN) : Nat :=
  colGlobal s BN j % 4096
```
</details>
