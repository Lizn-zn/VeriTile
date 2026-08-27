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
specification matmul_kernel_closed_form_correct
    (C A B : RegionName) (s : BlockState)
    (BM BN BLOCK_K numKBlocks : Nat)
    (hBN : BN ≤ 4096)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
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

## Public theorem: `matmul_kernel_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S1 fold genre; first streaming
`⊨[R]` in the bench).** For every rounding model `R`, the faithful
`matmul_kernel` surface implements, on its `StreamMasked2DKernelIO₂`
signature, the **ideal ℝ GEMM fold** over the streamed tiles: output lane
`l = (i, j)` holds `∑ t, ∑ e, A-tile[t](i,e) · B-tile[t](e,j)` — the spec `f`
is exact real arithmetic; the single boundary quantization is carried by the
skin's readback contract (`readMemAs .fp16` holds
`fp16.ofReal (R.round .fp16 (f …))`), where the kernel's two rounding events
(the `tl.cast(…, tl.float16)` and the fp16-typed store) collapse to one
`R.round .fp16` by the defining `round_idem`.

Layer map: the prologue and the whole K-loop are cast-free, so under
`execR R` they collapse verbatim onto the exact stepper and the proven
`preLoop` / `matmul_step` / `forRange_inv` stack above is reused unchanged;
only the 5-statement store tail is re-proved on the `R` side
(`matmul_postLoopR`).

The single hypothesis `hBN : BN ≤ 4096` is truth-forced: the output window
`4096·row + col` is injective only when the column-block width fits the
hard-coded row stride `4096` (`matmul_kernel_output_offset_injective` /
`rowMajor2D_inj`); with colliding output lanes the per-lane readback would be
last-writer-wins and the statement false. It holds for every valid tiling.

Relation to the exact surface: the exact headline
`matmul_kernel_closed_form_correct` (`Realizes_without_Rounding`) above is
retained unchanged; this `⊨[R]` face strictly generalizes its content — at
`R := .triv` the readback contract degenerates to the exact fp16-cast cell of
the same GEMM value. Both faces are kept per the rounding-as-default
doctrine. -/
```
</details>

**Statement:**
```lean
specification matmul_kernel_io_correctness (R : RoundingModel)
    (C A B : RegionName) (BM BN BK T : Nat) (hBN : BN ≤ 4096) :
    matmulKernelIO C A B BM BN BK T ⊨[R] fun _ _ xs ys l =>
      ∑ t : Fin T, ∑ e : Fin BK, xs t (aLane BM BN BK l e) * ys t (bLane BM BN BK l e)
```

**Assumptions / layout contracts:**
- `hBN : BN ≤ 4096`

**Closed-form spec defs (transitive):** `matmulKernelIO`, `aLane`, `bLane`, `matmul_kernel_surface`

<details><summary><code>matmulKernelIO</code></summary>

```
/-- **Streaming IO signature** of `matmul_kernel` on the two-stream fold skin
(S1: fold + terminal store). Step `t` of the K-loop reads the `[BM, BLOCK_K]`
`A`-tile and the `[BLOCK_K, BN]` `B`-tile; after the loop one `[BM, BN]`
output tile is stored at the **fp16** grid (`outDType := .fp16` — the
kernel's `tl.cast(…, tl.float16)` + fp16 store). The windows transcribe the
kernel's pointer arithmetic exactly:

* `read1` lane `l = (i, e)` (row-major over `[BM, BLOCK_K]`), step `t`:
  `((pid₀·BM + i) % 4096) · 4096 + (t·BLOCK_K + e)` — the invariant's
  `a_ptrs` cell `offs_am(i)·4096 + e·1 + t·BLOCK_K` after `t` advances.
* `read2` lane `l = (e, j)` (row-major over `[BLOCK_K, BN]`), step `t`:
  `(t·BLOCK_K + e) · 4096 + (pid₁·BN + j) % 4096` — the `b_ptrs` cell.
* `write` lane `l = (i, j)`: `4096·(pid₀·BM + i) + 1·(pid₁·BN + j)` — the
  kernel's un-wrapped `c_ptrs` (= `cOffset` in pid form).

The kernel has no masks: all windows are `True`. -/
```
```lean
def matmulKernelIO (C A B : RegionName) (BM BN BK T : Nat) : StreamMasked2DKernelIO₂ where
  kernel := matmul_kernel_surface C A B BM BN BK T
  inp1 := A
  inp2 := B
  out := C
  T := T
  B1 := BM * BK
  B2 := BK * BN
  C := BM * BN
  outDType := .fp16
  read1 := fun p₀ _ t l => ((p₀ * BM + l.val / BK) % 4096) * 4096 + (t.val * BK + l.val % BK)
  read2 := fun _ p₁ t l => (t.val * BK + l.val / BN) * 4096 + (p₁ * BN + l.val % BN) % 4096
  write := fun p₀ p₁ l => 4096 * (p₀ * BM + l.val / BN) + 1 * (p₁ * BN + l.val % BN)
  mask1 := fun _ _ _ _ => True
  mask2 := fun _ _ _ _ => True
  writeMask := fun _ _ _ => True
```
</details>

<details><summary><code>aLane</code></summary>

```
/-- The `A`-stream lane feeding output lane `l` at inner key `e`: the row of
`l` (row-major over the `[BM, BN]` output tile) paired with `e` over the
`[BM, BLOCK_K]` per-step `A`-tile, both via the shared `Lane2D` bridge. -/
```
```lean
def aLane (BM BN BK : Nat) (l : Fin (BM * BN)) (e : Fin BK) : Fin (BM * BK) :=
  Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit)
```
</details>

<details><summary><code>bLane</code></summary>

```
/-- The `B`-stream lane feeding output lane `l` at inner key `e`: `e` paired
with the column of `l` over the `[BLOCK_K, BN]` per-step `B`-tile. -/
```
```lean
def bLane (BM BN BK : Nat) (l : Fin (BM * BN)) (e : Fin BK) : Fin (BK * BN) :=
  Lane2D.encode (e, (Lane2D.decode l).2.1, PUnit.unit)
```
</details>

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
