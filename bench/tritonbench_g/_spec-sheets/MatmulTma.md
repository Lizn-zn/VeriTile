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
