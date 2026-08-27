# Spec sheet — `bench/tritonbench_g/matmul_tma/MatmulTma.lean`

**Python source:** `bench/tritonbench_g/matmul_tma/matmul_tma.py`

## Public theorem: `matmul_tma_f32_closed_form_correct`

<details><summary>docstring</summary>

```
/-- **Closed-form correctness for the f32 `matmul_tma` (general statement).**

For any matrix/tile dimensions and strides, every output cell of the computed
`BLOCK_M × BLOCK_N` tile equals the genuine matrix product
`Σ_{e < BLOCK_K} A[i,e] · B[e,j]` (over ℝ) of the loaded `A`/`B` tiles — *not*
the kernel's own executed value. Layout: `A[i,e]` at
`A + i·stride_am + e·stride_ak`, `B[e,j]` at `B + e·stride_bk + j·stride_bn`,
`C[i,j]` at `C + i·stride_cm + j·stride_cn` (the block pointers' offset-`(0,0)`
addresses). Precondition: output-offset injectivity. -/
```
</details>

**Statement:**
```lean
specification matmul_tma_f32_closed_form_correct
    (A B C : RegionName) (s : BlockState)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat)
    (hcn : stride_cn = 1) (hcm : BLOCK_N ≤ stride_cm) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := matmul_tma_f32_surface A B C M N K stride_am stride_ak
        stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        some (C, cOffset stride_cm stride_cn idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        matmulSpec s A B stride_am stride_ak stride_bk stride_bn BLOCK_K
          idx.1.val idx.2.1.val)
```

**Assumptions / layout contracts:**
- `hcn : stride_cn = 1`
- `hcm : BLOCK_N ≤ stride_cm`

**Closed-form spec defs (transitive):** `matmul_tma_f32_surface`, `cOffset`, `matmulSpec`, `aElem`, `bElem`

<details><summary><code>matmul_tma_f32_surface</code></summary>

```
/-- Faithful transcription of `matmul_tma.py`'s `matmul_tma_load_store` for the
`OUTPUT_F16 = false` branch (float32 output, no downcast). The TMA `order` tuple
is scheduling metadata the DSL erases into the same block-pointer AST. -/
```
```lean
def matmul_tma_f32_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  a_block_ptr = tl.make_block_ptr(base=A, shape=($(M), $(K)),
    strides=($(stride_am), $(stride_ak)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_M), $(BLOCK_K)), order=(1, 0))
  b_block_ptr = tl.make_block_ptr(base=B, shape=($(K), $(N)),
    strides=($(stride_bk), $(stride_bn)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_K), $(BLOCK_N)), order=(0, 1))
  c_block_ptr = tl.make_block_ptr(base=C, shape=($(M), $(N)),
    strides=($(stride_cm), $(stride_cn)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_M), $(BLOCK_N)), order=(1, 0))
  a = tl.load(a_block_ptr)
  b = tl.load(b_block_ptr)
  c = tl.dot(a, b)
  tl.store(c_block_ptr, c)
}
```
</details>

<details><summary><code>cOffset</code></summary>

```
/-- The output store address for tile lane `(i,j)`: `i · stride_cm + j · stride_cn`
(the `c_block_ptr` address with offsets `(0, 0)`). -/
```
```lean
def cOffset (stride_cm stride_cn : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  idx.1.val * stride_cm + idx.2.1.val * stride_cn
```
</details>

<details><summary><code>matmulSpec</code></summary>

```
/-- **Genuine GEMM spec** (over ℝ): `C[i,j] = Σ_{e < BLOCK_K} A[i,e] · B[e,j]`. -/
```
```lean
noncomputable def matmulSpec (s : BlockState) (A B : RegionName)
    (stride_am stride_ak stride_bk stride_bn BLOCK_K : Nat)
    (i j : Nat) : ℝ :=
  (Finset.univ : Finset (Fin BLOCK_K)).sum
    (fun e => aElem s A stride_am stride_ak i e.val
              * bElem s B stride_bk stride_bn e.val j)
```
</details>

<details><summary><code>aElem</code></summary>

```
/-- `A[i, e] = readMem A (i · stride_am + e · stride_ak)` — the address of tile
lane `(i, e)` of the `a_block_ptr` view (offsets `(0, 0)`). -/
```
```lean
noncomputable def aElem (s : BlockState) (A : RegionName) (stride_am stride_ak : Nat)
    (i e : Nat) : ℝ :=
  s.readMem A (i * stride_am + e * stride_ak)
```
</details>

<details><summary><code>bElem</code></summary>

```
/-- `B[e, j] = readMem B (e · stride_bk + j · stride_bn)` — the address of tile
lane `(e, j)` of the `b_block_ptr` view (offsets `(0, 0)`). -/
```
```lean
noncomputable def bElem (s : BlockState) (B : RegionName) (stride_bk stride_bn : Nat)
    (e j : Nat) : ℝ :=
  s.readMem B (e * stride_bk + j * stride_bn)
```
</details>

## Public theorem: `matmul_tma_f16_closed_form_correct`

<details><summary>docstring</summary>

```
/-- **Closed-form correctness for the fp16 `matmul_tma`.** Every output cell of
the computed tile equals `fp16(Σ_{e<BLOCK_K} A[i,e]·B[e,j])` — the genuine
matrix product over ℝ cast to float16. -/
```
</details>

**Statement:**
```lean
specification matmul_tma_f16_closed_form_correct
    (A B C : RegionName) (s : BlockState)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat)
    (hcn : stride_cn = 1) (hcm : BLOCK_N ≤ stride_cm) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := matmul_tma_f16_surface A B C M N K stride_am stride_ak
        stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        some (C, cOffset stride_cm stride_cn idx))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (matmulSpec s A B stride_am stride_ak stride_bk stride_bn BLOCK_K
              idx.1.val idx.2.1.val))))
```

**Assumptions / layout contracts:**
- `hcn : stride_cn = 1`
- `hcm : BLOCK_N ≤ stride_cm`

**Closed-form spec defs (transitive):** `matmul_tma_f16_surface`, `cOffset`, `matmulSpec`, `aElem`, `bElem`

<details><summary><code>matmul_tma_f16_surface</code></summary>

```
/-- Faithful transcription of `matmul_tma_load_store` for the `OUTPUT_F16 = true`
branch (the dot result is downcast to `float16` before the store). -/
```
```lean
def matmul_tma_f16_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  a_block_ptr = tl.make_block_ptr(base=A, shape=($(M), $(K)),
    strides=($(stride_am), $(stride_ak)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_M), $(BLOCK_K)), order=(1, 0))
  b_block_ptr = tl.make_block_ptr(base=B, shape=($(K), $(N)),
    strides=($(stride_bk), $(stride_bn)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_K), $(BLOCK_N)), order=(0, 1))
  c_block_ptr = tl.make_block_ptr(base=C, shape=($(M), $(N)),
    strides=($(stride_cm), $(stride_cn)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_M), $(BLOCK_N)), order=(1, 0))
  a = tl.load(a_block_ptr)
  b = tl.load(b_block_ptr)
  c = (tl.dot(a, b)).to(tl.float16)
  tl.store(c_block_ptr, c)
}
```
</details>

<details><summary><code>cOffset</code></summary>

```
/-- The output store address for tile lane `(i,j)`: `i · stride_cm + j · stride_cn`
(the `c_block_ptr` address with offsets `(0, 0)`). -/
```
```lean
def cOffset (stride_cm stride_cn : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  idx.1.val * stride_cm + idx.2.1.val * stride_cn
```
</details>

<details><summary><code>matmulSpec</code></summary>

```
/-- **Genuine GEMM spec** (over ℝ): `C[i,j] = Σ_{e < BLOCK_K} A[i,e] · B[e,j]`. -/
```
```lean
noncomputable def matmulSpec (s : BlockState) (A B : RegionName)
    (stride_am stride_ak stride_bk stride_bn BLOCK_K : Nat)
    (i j : Nat) : ℝ :=
  (Finset.univ : Finset (Fin BLOCK_K)).sum
    (fun e => aElem s A stride_am stride_ak i e.val
              * bElem s B stride_bk stride_bn e.val j)
```
</details>

<details><summary><code>aElem</code></summary>

```
/-- `A[i, e] = readMem A (i · stride_am + e · stride_ak)` — the address of tile
lane `(i, e)` of the `a_block_ptr` view (offsets `(0, 0)`). -/
```
```lean
noncomputable def aElem (s : BlockState) (A : RegionName) (stride_am stride_ak : Nat)
    (i e : Nat) : ℝ :=
  s.readMem A (i * stride_am + e * stride_ak)
```
</details>

<details><summary><code>bElem</code></summary>

```
/-- `B[e, j] = readMem B (e · stride_bk + j · stride_bn)` — the address of tile
lane `(e, j)` of the `b_block_ptr` view (offsets `(0, 0)`). -/
```
```lean
noncomputable def bElem (s : BlockState) (B : RegionName) (stride_bk stride_bn : Nat)
    (e j : Nat) : ℝ :=
  s.readMem B (e * stride_bk + j * stride_bn)
```
</details>

## Public theorem: `matmul_tma_f32_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for `matmul_tma.py`'s
`matmul_tma_load_store`, `OUTPUT_F16 = false` branch: for every disjoint flat
placement of the three buffers, and every launch state whose `A` and `B` tiles are
pinned, the translated pointer kernel terminates, every cell of the output tile
holds the genuine GEMM value `Σ_{e < BLOCK_K} xs[i, e] · ys[e, j]`, and every
other memory cell is unchanged.

This is the first **contraction** on an `io ⊨ f` face, and it is what the
per-channel-shape skin exists for: the three channels live on three different lane
sets (`[BLOCK_M, BLOCK_K]`, `[BLOCK_K, BLOCK_N]`, `[BLOCK_M, BLOCK_N]`), related
only through `f`, which reads both inputs at lanes the output index does not name.

Dimension-general in `M`, `N`, `K`, all six strides and all three block sizes.
Honest side-condition: output-address injectivity (`hInj`) — the same hypothesis
the closed-form theorems take, dischargeable for row-major `C` by
`cOffset_injective_of_rowMajor`. -/
```
</details>

**Statement:**
```lean
specification matmul_tma_f32_io_correctness (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat)
    (hInj : Function.Injective
      (cOffset (BLOCK_M := BLOCK_M) (BLOCK_N := BLOCK_N) stride_cm stride_cn)) :
    matmulTmaF32IO A B C M N K stride_am stride_ak stride_bk stride_bn stride_cm
        stride_cn BLOCK_M BLOCK_N BLOCK_K
      ⊨ fun _p₀ _p₁ xs ys idx =>
          matmulSpecOf BLOCK_M BLOCK_N BLOCK_K xs ys idx
```

**Closed-form spec defs (transitive):** `cOffset`, `matmulTmaF32IO`, `matmulSpecOf`, `matmul_tma_f32_surface`

<details><summary><code>cOffset</code></summary>

```
/-- The output store address for tile lane `(i,j)`: `i · stride_cm + j · stride_cn`
(the `c_block_ptr` address with offsets `(0, 0)`). -/
```
```lean
def cOffset (stride_cm stride_cn : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  idx.1.val * stride_cm + idx.2.1.val * stride_cn
```
</details>

<details><summary><code>matmulTmaF32IO</code></summary>

```
/-- IO signature of the f32 branch on the **per-channel-shape** surface: `A` on
`[BLOCK_M, BLOCK_K]`, `B` on `[BLOCK_K, BLOCK_N]`, `C` on `[BLOCK_M, BLOCK_N]`,
every lane of all three active (the block pointers carry no `boundary_check`),
and the addresses are the block pointers' offset-`(0,0)` maps. -/
```
```lean
def matmulTmaF32IO (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) : MaskedTileShapedKernelIO₂ where
  kernel := matmul_tma_f32_surface A B C M N K stride_am stride_ak stride_bk
    stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K
  in1 := A
  in2 := B
  out := C
  shape1 := [BLOCK_M, BLOCK_K]
  shape2 := [BLOCK_K, BLOCK_N]
  shapeOut := [BLOCK_M, BLOCK_N]
  read1 := fun _p₀ _p₁ k => k.1.val * stride_am + k.2.1.val * stride_ak
  read2 := fun _p₀ _p₁ k => k.1.val * stride_bk + k.2.1.val * stride_bn
  write := fun _p₀ _p₁ o => cOffset stride_cm stride_cn o
  mask1 := fun _p₀ _p₁ _ => True
  mask2 := fun _p₀ _p₁ _ => True
  writeMask := fun _p₀ _p₁ _ => True
```
</details>

<details><summary><code>matmulSpecOf</code></summary>

```
/-- Value-level GEMM spec: `Σ_{e < BLOCK_K} xs[i, e] · ys[e, j]`, over the
*loaded values* rather than over memory — which is what the IO surface
quantifies. Reading both inputs at lanes the output index does not name is the
whole reason the channels need separate shapes. -/
```
```lean
noncomputable def matmulSpecOf (BLOCK_M BLOCK_N BLOCK_K : Nat)
    (xs : TileIndex [BLOCK_M, BLOCK_K] → ℝ)
    (ys : TileIndex [BLOCK_K, BLOCK_N] → ℝ)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  (Finset.univ : Finset (Fin BLOCK_K)).sum
    (fun e => xs (idx.1, e, PUnit.unit) * ys (e, idx.2.1, PUnit.unit))
```
</details>

<details><summary><code>matmul_tma_f32_surface</code></summary>

```
/-- Faithful transcription of `matmul_tma.py`'s `matmul_tma_load_store` for the
`OUTPUT_F16 = false` branch (float32 output, no downcast). The TMA `order` tuple
is scheduling metadata the DSL erases into the same block-pointer AST. -/
```
```lean
def matmul_tma_f32_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  a_block_ptr = tl.make_block_ptr(base=A, shape=($(M), $(K)),
    strides=($(stride_am), $(stride_ak)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_M), $(BLOCK_K)), order=(1, 0))
  b_block_ptr = tl.make_block_ptr(base=B, shape=($(K), $(N)),
    strides=($(stride_bk), $(stride_bn)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_K), $(BLOCK_N)), order=(0, 1))
  c_block_ptr = tl.make_block_ptr(base=C, shape=($(M), $(N)),
    strides=($(stride_cm), $(stride_cn)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_M), $(BLOCK_N)), order=(1, 0))
  a = tl.load(a_block_ptr)
  b = tl.load(b_block_ptr)
  c = tl.dot(a, b)
  tl.store(c_block_ptr, c)
}
```
</details>

## Public theorem: `matmul_tma_f16_io_correctnessR`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` headline** for `matmul_tma.py`'s `matmul_tma_load_store`,
`OUTPUT_F16 = true` branch: for **every** rounding model `R`, every disjoint flat
placement of the three buffers, and every launch state whose `A` and `B` tiles are
pinned, the kernel run under `execR R` terminates, every cell of the output tile
reads back at `.fp16` holding

  `R.round .fp16 (Σ_{e < BLOCK_K} xs[i, e] · ys[e, j])`

and every other memory cell is unchanged.

Unlike the **cast-free** `⊨[R]` port faces (which transport an exact result
verbatim because `execR R = exec`), this one is not a transport: `.to(tl.float16)`
is a genuine rounding site, so the `R.round .fp16` on the right is the
quantization the kernel actually performs. Eight other ports already state an
fp16 boundary this way — see the section docstring for the list; this face's
contribution is that `matmul_tma`'s fp16 branch previously had no IO face at
all, and that it is the first fp16 boundary on the per-channel-shape skin. `f` remains the exact ℝ contraction — the face's content is precisely
"the fp16 output is the real GEMM value, rounded once".

The kernel rounds the value **twice** (the cast, then the fp16 typed store);
`RoundingModel.round_idem` collapses that to the single round stated here.

Dimension-general in `M`, `N`, `K`, all six strides and all three block sizes.
Honest side-condition: output-address injectivity (`hInj`) — the same hypothesis
the exact closed forms take, dischargeable for row-major `C` by
`cOffset_injective_of_rowMajor`. -/
```
</details>

**Statement:**
```lean
specification matmul_tma_f16_io_correctnessR (Rm : RoundingModel)
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat)
    (hInj : Function.Injective
      (cOffset (BLOCK_M := BLOCK_M) (BLOCK_N := BLOCK_N) stride_cm stride_cn)) :
    matmulTmaF16IO A B C M N K stride_am stride_ak stride_bk stride_bn stride_cm
        stride_cn BLOCK_M BLOCK_N BLOCK_K
      ⊨[Rm, FloatDType.fp16] fun _p₀ _p₁ xs ys idx =>
          matmulSpecOf BLOCK_M BLOCK_N BLOCK_K xs ys idx
```

**Closed-form spec defs (transitive):** `cOffset`, `matmulTmaF16IO`, `matmulSpecOf`, `matmul_tma_f16_surface`

<details><summary><code>cOffset</code></summary>

```
/-- The output store address for tile lane `(i,j)`: `i · stride_cm + j · stride_cn`
(the `c_block_ptr` address with offsets `(0, 0)`). -/
```
```lean
def cOffset (stride_cm stride_cn : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  idx.1.val * stride_cm + idx.2.1.val * stride_cn
```
</details>

<details><summary><code>matmulTmaF16IO</code></summary>

```
/-- IO signature of the fp16 branch on the per-channel-shape surface — the same
windows as `matmulTmaF32IO`, on the fp16 kernel. -/
```
```lean
def matmulTmaF16IO (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) : MaskedTileShapedKernelIO₂ where
  kernel := matmul_tma_f16_surface A B C M N K stride_am stride_ak stride_bk
    stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K
  in1 := A
  in2 := B
  out := C
  shape1 := [BLOCK_M, BLOCK_K]
  shape2 := [BLOCK_K, BLOCK_N]
  shapeOut := [BLOCK_M, BLOCK_N]
  read1 := fun _p₀ _p₁ k => k.1.val * stride_am + k.2.1.val * stride_ak
  read2 := fun _p₀ _p₁ k => k.1.val * stride_bk + k.2.1.val * stride_bn
  write := fun _p₀ _p₁ o => cOffset stride_cm stride_cn o
  mask1 := fun _p₀ _p₁ _ => True
  mask2 := fun _p₀ _p₁ _ => True
  writeMask := fun _p₀ _p₁ _ => True
```
</details>

<details><summary><code>matmulSpecOf</code></summary>

```
/-- Value-level GEMM spec: `Σ_{e < BLOCK_K} xs[i, e] · ys[e, j]`, over the
*loaded values* rather than over memory — which is what the IO surface
quantifies. Reading both inputs at lanes the output index does not name is the
whole reason the channels need separate shapes. -/
```
```lean
noncomputable def matmulSpecOf (BLOCK_M BLOCK_N BLOCK_K : Nat)
    (xs : TileIndex [BLOCK_M, BLOCK_K] → ℝ)
    (ys : TileIndex [BLOCK_K, BLOCK_N] → ℝ)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  (Finset.univ : Finset (Fin BLOCK_K)).sum
    (fun e => xs (idx.1, e, PUnit.unit) * ys (e, idx.2.1, PUnit.unit))
```
</details>

<details><summary><code>matmul_tma_f16_surface</code></summary>

```
/-- Faithful transcription of `matmul_tma_load_store` for the `OUTPUT_F16 = true`
branch (the dot result is downcast to `float16` before the store). -/
```
```lean
def matmul_tma_f16_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  a_block_ptr = tl.make_block_ptr(base=A, shape=($(M), $(K)),
    strides=($(stride_am), $(stride_ak)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_M), $(BLOCK_K)), order=(1, 0))
  b_block_ptr = tl.make_block_ptr(base=B, shape=($(K), $(N)),
    strides=($(stride_bk), $(stride_bn)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_K), $(BLOCK_N)), order=(0, 1))
  c_block_ptr = tl.make_block_ptr(base=C, shape=($(M), $(N)),
    strides=($(stride_cm), $(stride_cn)), offsets=($(0), $(0)),
    block_shape=($(BLOCK_M), $(BLOCK_N)), order=(1, 0))
  a = tl.load(a_block_ptr)
  b = tl.load(b_block_ptr)
  c = (tl.dot(a, b)).to(tl.float16)
  tl.store(c_block_ptr, c)
}
```
</details>
