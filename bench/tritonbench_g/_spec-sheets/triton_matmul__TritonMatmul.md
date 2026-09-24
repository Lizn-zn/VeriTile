# Spec sheet — `bench/tritonbench_g/triton_matmul/TritonMatmul.lean`

**Python source:** `bench/tritonbench_g/triton_matmul/triton_matmul.py`

## Public theorem: `triton_matmul_f16_closed_form_correct`

<details><summary>docstring</summary>

```
/-- **Closed-form correctness for `triton_matmul` (fp16 epilogue arm, general
statement).**

For arbitrary linear program id `pid`, tile dims `BM`/`BN`, K-block size
`BLOCK_K`, and K-block count `numKBlocks` (so the contracted dimension is
`K = BLOCK_K · numKBlocks`), every **active** output cell of the computed
`BM × BN` tile equals `fp16( Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j] )` —
the genuine matrix product (over ℝ) of the loaded `A`/`B` tiles, cast to
float16 — **not** the kernel's own executed value; inactive lanes are left
untouched.

Layout: `A[i,k]` at `A + offs_am(i)·stride_am + k·stride_ak`, `B[k,j]` at
`B + k·stride_bk + offs_bn(j)·stride_bn`, `C[i,j]` at
`C + stride_cm·offs_cm(i) + stride_cn·offs_cn(j)`, with `pid_m`/`pid_n` derived
by the kernel's L2-grouping schedule,
`offs_am(i) = if pid_m·BM + i < M then pid_m·BM + i else 0` and
`offs_bn(j) = if pid_n·BN + j < N then pid_n·BN + j else 0` (the row-major
pointer arithmetic with the kernel's `tl.where` index clamp). Preconditions:
output-offset injectivity and clean initial `undef`. -/
```
</details>

**Statement:**
```lean
specification triton_matmul_f16_closed_form_correct
    (A B C : RegionName) (s : BlockState)
    (M N BM BN GM sam sak sbk sbn scm scn BLOCK_K numKBlocks : Nat) (K : Nat)
    (hK : K = BLOCK_K * numKBlocks)
    (hcn : scn = 1) (hbnle : BN ≤ scm)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_matmul_f16_surface A B C M N K sam sak sbk sbn scm scn
        BM BN BLOCK_K GM numKBlocks)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BM BN GM)
        (fun idx => (C, cOffset s M N BM BN GM scm scn idx)))
      (expected := fun idx : TileIndex [BM, BN] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (matmulSpec s A B M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks idx.1 idx.2.1))))
```

**Assumptions / layout contracts:**
- `hK : K = BLOCK_K * numKBlocks`
- `hcn : scn = 1`
- `hbnle : BN ≤ scm`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `triton_matmul_f16_surface`, `active`, `cOffset`, `matmulSpec`, `rowGlobal`, `colGlobal`, `aElem`, `bElem`, `pidM`, `pidN`, `rowIndex`, `colIndex`, `kernelMin`, `clampIdx`

<details><summary><code>triton_matmul_f16_surface</code></summary>

```
/-- Faithful transcription of `triton_matmul.py`'s `matmul_kernel`, **fp16
epilogue arm** (`c_ptr.dtype.element_ty ≠ tl.float8e4nv`).

The contracted dimension is presented as `K = BLOCK_SIZE_K · numKBlocks` so the
loop bound `tl.cdiv(K, BLOCK_SIZE_K) = numKBlocks` is exact; it is supplied as
the antiquoted `numKBlocks`. All other surface structure — the L2-grouping
schedule, the `tl.where` index clamps, the `tl.max_contiguous`/`tl.multiple_of`
hints (DSL-erased to their value argument), the per-block
`offs_k < K - kk·BLOCK_K` load masks, the fused `tl.dot(a, b, accumulator)`,
the `float16` cast, and the `(row<M)&(col<N)`-masked store — is transcribed
verbatim. -/
```
```lean
def triton_matmul_f16_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  start_m = pid_m * $(BLOCK_SIZE_M)
  start_n = pid_n * $(BLOCK_SIZE_N)
  offs_am = start_m + tl.arange(0, $(BLOCK_SIZE_M))
  offs_bn = start_n + tl.arange(0, $(BLOCK_SIZE_N))
  offs_am = tl.where(offs_am < $(M), offs_am, $(0))
  offs_bn = tl.where(offs_bn < $(N), offs_bn, $(0))
  offs_am = tl.max_contiguous(tl.multiple_of(offs_am, $(BLOCK_SIZE_M)), $(BLOCK_SIZE_M))
  offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, $(BLOCK_SIZE_N)), $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
  b_ptrs = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for kk in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}
```
</details>

<details><summary><code>active</code></summary>

```
/-- The boundary predicate `(row < M) & (col < N)` for tile lane `(i,j)`. -/
```
```lean
def active (s0 : BlockState) (M N BM BN GM : Nat) (idx : TileIndex [BM, BN]) : Prop :=
  rowGlobal s0 M N BM BN GM idx.1 < M ∧ colGlobal s0 M N BM BN GM idx.2.1 < N
```
</details>

<details><summary><code>cOffset</code></summary>

```
/-- The output store address for tile lane `(i,j)`: `scm · offs_cm i + scn · offs_cn j`
(the kernel's `c_ptrs`, using the **un-clamped** global `offs_cm`/`offs_cn`,
recomputed fresh from `pid_m`/`pid_n`). -/
```
```lean
def cOffset (s0 : BlockState) (M N BM BN GM scm scn : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  scm * rowGlobal s0 M N BM BN GM idx.1 + scn * colGlobal s0 M N BM BN GM idx.2.1
```
</details>

<details><summary><code>matmulSpec</code></summary>

```
/-- **Genuine matmul spec** (over ℝ):
`C[i,j] = Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j]`. -/
```
```lean
noncomputable def matmulSpec (s : BlockState) (A B : RegionName)
    (M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks : Nat)
    (i : Fin BM) (j : Fin BN) : ℝ :=
  gemmSum (aElem s A M N BM BN GM sam sak i) (bElem s B M N BM BN GM sbk sbn j)
    (BLOCK_K * numKBlocks)
```
</details>

<details><summary><code>rowGlobal</code></summary>

```
/-- Global output row of tile lane `i`: `pid_m · BLOCK_M + i`, **before** the
`tl.where` clamp (the kernel's `start_m + arange`, and its `offs_cm`). -/
```
```lean
def rowGlobal (s : BlockState) (M N BM BN GM : Nat) (i : Fin BM) : Nat :=
  pidM (s.pids 0) M N BM BN GM * BM + i.val
```
</details>

<details><summary><code>colGlobal</code></summary>

```
/-- Global output column of tile lane `j`: `pid_n · BLOCK_N + j`, before clamp. -/
```
```lean
def colGlobal (s : BlockState) (M N BM BN GM : Nat) (j : Fin BN) : Nat :=
  pidN (s.pids 0) M N BM BN GM * BN + j.val
```
</details>

<details><summary><code>aElem</code></summary>

```
/-- `A[i, k] = readMem A (offs_am i · stride_am + k · stride_ak)`. -/
```
```lean
noncomputable def aElem (s : BlockState) (A : RegionName) (M N BM BN GM sam sak : Nat)
    (i : Fin BM) (k : Nat) : ℝ :=
  s.readMem A (rowIndex s M N BM BN GM i * sam + k * sak)
```
</details>

<details><summary><code>bElem</code></summary>

```
/-- `B[k, j] = readMem B (k · stride_bk + offs_bn j · stride_bn)`. -/
```
```lean
noncomputable def bElem (s : BlockState) (B : RegionName) (M N BM BN GM sbk sbn : Nat)
    (j : Fin BN) (k : Nat) : ℝ :=
  s.readMem B (k * sbk + colIndex s M N BM BN GM j * sbn)
```
</details>

<details><summary><code>pidM</code></summary>

```
/-- The kernel's L2-grouping derivation of `pid_m` from the linear `pid` —
**this** kernel's spelling `first_pid_m + (pid % group_size_m)` (the twin
`matmul_triton_autotune` reduces `pid % num_pid_in_group` first). -/
```
```lean
def pidM (pid M N BM BN GM : Nat) : Nat :=
  let num_pid_m := cdiv M BM
  let num_pid_n := cdiv N BN
  let num_pid_in_group := GM * num_pid_n
  let group_id := pid / num_pid_in_group
  let first_pid_m := group_id * GM
  let group_size_m := kernelMin (num_pid_m - first_pid_m) GM
  first_pid_m + (pid % group_size_m)
```
</details>

<details><summary><code>pidN</code></summary>

```
/-- The kernel's L2-grouping derivation of `pid_n` from the linear `pid`. -/
```
```lean
def pidN (pid M N BM BN GM : Nat) : Nat :=
  let num_pid_m := cdiv M BM
  let num_pid_n := cdiv N BN
  let num_pid_in_group := GM * num_pid_n
  let group_id := pid / num_pid_in_group
  let first_pid_m := group_id * GM
  let group_size_m := kernelMin (num_pid_m - first_pid_m) GM
  (pid % num_pid_in_group) / group_size_m
```
</details>

<details><summary><code>rowIndex</code></summary>

```
/-- The clamped A-row index of tile lane `i` (the kernel's `offs_am` after the
`tl.where(offs_am < M, offs_am, 0)` reassignment). -/
```
```lean
def rowIndex (s : BlockState) (M N BM BN GM : Nat) (i : Fin BM) : Nat :=
  clampIdx (rowGlobal s M N BM BN GM i) M
```
</details>

<details><summary><code>colIndex</code></summary>

```
/-- The clamped B-column index of tile lane `j` (the kernel's `offs_bn` after
the `tl.where(offs_bn < N, offs_bn, 0)` reassignment). -/
```
```lean
def colIndex (s : BlockState) (M N BM BN GM : Nat) (j : Fin BN) : Nat :=
  clampIdx (colGlobal s M N BM BN GM j) N
```
</details>

<details><summary><code>kernelMin</code></summary>

```
/-- `min` as the kernel's `tl.where(a < b, a, b)` spells it. -/
```
```lean
def kernelMin (a b : Nat) : Nat := if a < b then a else b
```
</details>

<details><summary><code>clampIdx</code></summary>

```
/-- The kernel's `tl.where(v < bound, v, 0)` index clamp (this kernel's
out-of-range guard, where the `matmul_triton_autotune` twin wraps with `%`). -/
```
```lean
def clampIdx (v bound : Nat) : Nat := if v < bound then v else 0
```
</details>

## Public theorem: `triton_matmul_f8_closed_form_correct`

<details><summary>docstring</summary>

```
/-- **Closed-form correctness for `triton_matmul` (fp8 epilogue arm, general
statement)**: identical to `triton_matmul_f16_closed_form_correct` except the
epilogue grid — every active output cell holds
`f8e4( Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j] )`, the genuine matrix
product cast to `float8e4nv` (the `c_ptr.dtype.element_ty == tl.float8e4nv`
constexpr arm). -/
```
</details>

**Statement:**
```lean
specification triton_matmul_f8_closed_form_correct
    (A B C : RegionName) (s : BlockState)
    (M N BM BN GM sam sak sbk sbn scm scn BLOCK_K numKBlocks : Nat) (K : Nat)
    (hK : K = BLOCK_K * numKBlocks)
    (hcn : scn = 1) (hbnle : BN ≤ scm)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := triton_matmul_f8_surface A B C M N K sam sak sbk sbn scm scn
        BM BN BLOCK_K GM numKBlocks)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BM BN GM)
        (fun idx => (C, cOffset s M N BM BN GM scm scn idx)))
      (expected := fun idx : TileIndex [BM, BN] =>
        MemCell.of .f8e4
          (FloatDType.real.cast FloatDType.f8e4
            (some (matmulSpec s A B M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks idx.1 idx.2.1))))
```

**Assumptions / layout contracts:**
- `hK : K = BLOCK_K * numKBlocks`
- `hcn : scn = 1`
- `hbnle : BN ≤ scm`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `triton_matmul_f8_surface`, `active`, `cOffset`, `matmulSpec`, `rowGlobal`, `colGlobal`, `aElem`, `bElem`, `pidM`, `pidN`, `rowIndex`, `colIndex`, `kernelMin`, `clampIdx`

<details><summary><code>triton_matmul_f8_surface</code></summary>

```
/-- Faithful transcription of `triton_matmul.py`'s `matmul_kernel`, **fp8
epilogue arm** (`c_ptr.dtype.element_ty == tl.float8e4nv`): identical to
`triton_matmul_f16_surface` except the epilogue cast
`c = (accumulator).to(tl.float8e4nv)`. -/
```
```lean
def triton_matmul_f8_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  start_m = pid_m * $(BLOCK_SIZE_M)
  start_n = pid_n * $(BLOCK_SIZE_N)
  offs_am = start_m + tl.arange(0, $(BLOCK_SIZE_M))
  offs_bn = start_n + tl.arange(0, $(BLOCK_SIZE_N))
  offs_am = tl.where(offs_am < $(M), offs_am, $(0))
  offs_bn = tl.where(offs_bn < $(N), offs_bn, $(0))
  offs_am = tl.max_contiguous(tl.multiple_of(offs_am, $(BLOCK_SIZE_M)), $(BLOCK_SIZE_M))
  offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, $(BLOCK_SIZE_N)), $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
  b_ptrs = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for kk in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  c = (accumulator).to(tl.float8e4nv)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}
```
</details>

<details><summary><code>active</code></summary>

```
/-- The boundary predicate `(row < M) & (col < N)` for tile lane `(i,j)`. -/
```
```lean
def active (s0 : BlockState) (M N BM BN GM : Nat) (idx : TileIndex [BM, BN]) : Prop :=
  rowGlobal s0 M N BM BN GM idx.1 < M ∧ colGlobal s0 M N BM BN GM idx.2.1 < N
```
</details>

<details><summary><code>cOffset</code></summary>

```
/-- The output store address for tile lane `(i,j)`: `scm · offs_cm i + scn · offs_cn j`
(the kernel's `c_ptrs`, using the **un-clamped** global `offs_cm`/`offs_cn`,
recomputed fresh from `pid_m`/`pid_n`). -/
```
```lean
def cOffset (s0 : BlockState) (M N BM BN GM scm scn : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  scm * rowGlobal s0 M N BM BN GM idx.1 + scn * colGlobal s0 M N BM BN GM idx.2.1
```
</details>

<details><summary><code>matmulSpec</code></summary>

```
/-- **Genuine matmul spec** (over ℝ):
`C[i,j] = Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j]`. -/
```
```lean
noncomputable def matmulSpec (s : BlockState) (A B : RegionName)
    (M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks : Nat)
    (i : Fin BM) (j : Fin BN) : ℝ :=
  gemmSum (aElem s A M N BM BN GM sam sak i) (bElem s B M N BM BN GM sbk sbn j)
    (BLOCK_K * numKBlocks)
```
</details>

<details><summary><code>rowGlobal</code></summary>

```
/-- Global output row of tile lane `i`: `pid_m · BLOCK_M + i`, **before** the
`tl.where` clamp (the kernel's `start_m + arange`, and its `offs_cm`). -/
```
```lean
def rowGlobal (s : BlockState) (M N BM BN GM : Nat) (i : Fin BM) : Nat :=
  pidM (s.pids 0) M N BM BN GM * BM + i.val
```
</details>

<details><summary><code>colGlobal</code></summary>

```
/-- Global output column of tile lane `j`: `pid_n · BLOCK_N + j`, before clamp. -/
```
```lean
def colGlobal (s : BlockState) (M N BM BN GM : Nat) (j : Fin BN) : Nat :=
  pidN (s.pids 0) M N BM BN GM * BN + j.val
```
</details>

<details><summary><code>aElem</code></summary>

```
/-- `A[i, k] = readMem A (offs_am i · stride_am + k · stride_ak)`. -/
```
```lean
noncomputable def aElem (s : BlockState) (A : RegionName) (M N BM BN GM sam sak : Nat)
    (i : Fin BM) (k : Nat) : ℝ :=
  s.readMem A (rowIndex s M N BM BN GM i * sam + k * sak)
```
</details>

<details><summary><code>bElem</code></summary>

```
/-- `B[k, j] = readMem B (k · stride_bk + offs_bn j · stride_bn)`. -/
```
```lean
noncomputable def bElem (s : BlockState) (B : RegionName) (M N BM BN GM sbk sbn : Nat)
    (j : Fin BN) (k : Nat) : ℝ :=
  s.readMem B (k * sbk + colIndex s M N BM BN GM j * sbn)
```
</details>

<details><summary><code>pidM</code></summary>

```
/-- The kernel's L2-grouping derivation of `pid_m` from the linear `pid` —
**this** kernel's spelling `first_pid_m + (pid % group_size_m)` (the twin
`matmul_triton_autotune` reduces `pid % num_pid_in_group` first). -/
```
```lean
def pidM (pid M N BM BN GM : Nat) : Nat :=
  let num_pid_m := cdiv M BM
  let num_pid_n := cdiv N BN
  let num_pid_in_group := GM * num_pid_n
  let group_id := pid / num_pid_in_group
  let first_pid_m := group_id * GM
  let group_size_m := kernelMin (num_pid_m - first_pid_m) GM
  first_pid_m + (pid % group_size_m)
```
</details>

<details><summary><code>pidN</code></summary>

```
/-- The kernel's L2-grouping derivation of `pid_n` from the linear `pid`. -/
```
```lean
def pidN (pid M N BM BN GM : Nat) : Nat :=
  let num_pid_m := cdiv M BM
  let num_pid_n := cdiv N BN
  let num_pid_in_group := GM * num_pid_n
  let group_id := pid / num_pid_in_group
  let first_pid_m := group_id * GM
  let group_size_m := kernelMin (num_pid_m - first_pid_m) GM
  (pid % num_pid_in_group) / group_size_m
```
</details>

<details><summary><code>rowIndex</code></summary>

```
/-- The clamped A-row index of tile lane `i` (the kernel's `offs_am` after the
`tl.where(offs_am < M, offs_am, 0)` reassignment). -/
```
```lean
def rowIndex (s : BlockState) (M N BM BN GM : Nat) (i : Fin BM) : Nat :=
  clampIdx (rowGlobal s M N BM BN GM i) M
```
</details>

<details><summary><code>colIndex</code></summary>

```
/-- The clamped B-column index of tile lane `j` (the kernel's `offs_bn` after
the `tl.where(offs_bn < N, offs_bn, 0)` reassignment). -/
```
```lean
def colIndex (s : BlockState) (M N BM BN GM : Nat) (j : Fin BN) : Nat :=
  clampIdx (colGlobal s M N BM BN GM j) N
```
</details>

<details><summary><code>kernelMin</code></summary>

```
/-- `min` as the kernel's `tl.where(a < b, a, b)` spells it. -/
```
```lean
def kernelMin (a b : Nat) : Nat := if a < b then a else b
```
</details>

<details><summary><code>clampIdx</code></summary>

```
/-- The kernel's `tl.where(v < bound, v, 0)` index clamp (this kernel's
out-of-range guard, where the `matmul_triton_autotune` twin wraps with `%`). -/
```
```lean
def clampIdx (v bound : Nat) : Nat := if v < bound then v else 0
```
</details>

## Public theorem: `triton_matmul_f16_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline, fp16 epilogue arm (wave-5 S1 fold
genre).** For every rounding model `R`, the faithful fp16-arm surface
implements, on its `StreamMasked2DKernelIO₂` signature, the **ideal ℝ GEMM
fold** over the streamed tiles: output lane `l = (i, j)` holds
`∑ t, ∑ e, A-tile[t](i,e) · B-tile[t](e,j)` — the spec `f` is exact real
arithmetic, and the single boundary quantization is carried by the skin's
readback contract (`readMemAs .fp16` holds `fp16.ofReal (R.round .fp16 (f …))`),
where the kernel's two rounding events (the `.to(tl.float16)` cast and the
fp16-typed masked store) collapse to one `R.round .fp16` by the defining
`round_idem`.

Layer map: the prologue (with the `tl.where` index clamps) and the whole
K-loop (masked loads, `other=0.0`) are cast-free, so under `execR R` they
collapse verbatim onto the exact stepper and the proven `preLoop` /
`matmul_step` / `forRange_inv` stack above is reused unchanged; only the
6-statement store tail is re-proved on the `R` side (`matmul_postLoopR`).

Every hypothesis is truth-forced:

* `hK : K = BK · numKBlocks` — the exact-multiple contraction presentation
  shared with the exact surface: it makes the trip count
  `cdiv(K, BLOCK_K) = numKBlocks` exact and the per-step
  `offs_k < K - kk·BK` load masks all-true, which is what lets the streamed
  pins cover every lane the `tl.dot` consumes (at non-multiple `K` the spec
  sum would over-count the tail block and the statement would be false).
* `hcn : scn = 1` and `hBN : BN ≤ scm` — output-offset injectivity
  (`rowMajor2D_inj`): the column stride is the unit stride and the
  column-block width fits the row stride, so distinct output lanes hit
  distinct addresses; with colliding lanes the per-lane readback would be
  last-writer-wins and the statement false. Both hold for every valid
  row-major tiling.

Relation to the exact surface: the exact headline
`triton_matmul_f16_closed_form_correct` (`Realizes_without_Rounding`) above
is retained unchanged; this `⊨[R]` face strictly generalizes its content — at
`R := .triv` the readback contract degenerates to the exact fp16-cast cell of
the same GEMM value. Both faces are kept per the rounding-as-default
doctrine. -/
```
</details>

**Statement:**
```lean
specification triton_matmul_f16_io_correctness (R : RoundingModel)
    (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat)
    (hK : K = BK * numKBlocks) (hcn : scn = 1) (hBN : BN ≤ scm) :
    tritonMatmulF16IO A B C M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks
      ⊨[R] fun _ _ xs ys l =>
        ∑ t : Fin numKBlocks, ∑ e : Fin BK,
          xs t (aLane BM BN BK l e) * ys t (bLane BM BN BK l e)
```

**Assumptions / layout contracts:**
- `hK : K = BK * numKBlocks`
- `hcn : scn = 1`
- `hBN : BN ≤ scm`

**Closed-form spec defs (transitive):** `tritonMatmulF16IO`, `aLane`, `bLane`, `triton_matmul_f16_surface`, `clampIdx`, `pidM`, `pidN`, `kernelMin`

<details><summary><code>tritonMatmulF16IO</code></summary>

```
/-- **Streaming IO signature** of `triton_matmul`'s **fp16 epilogue arm** on
the two-stream fold skin (S1: fold + terminal store). Step `t` of the K-loop
reads the `[BM, BLOCK_K]` `A`-tile and the `[BLOCK_K, BN]` `B`-tile; after
the loop one `[BM, BN]` output tile is stored at the **fp16** grid
(`outDType := .fp16` — this arm's `.to(tl.float16)` + fp16 store). The kernel
schedules on a **single** linear `pid` (`program_id(0)`; the skin's `pid₁`
slot is unused), so every window derives `(pid_m, pid_n)` through the
transcribed L2-grouping arithmetic `pidM`/`pidN`:

* `read1` lane `l = (i, e)` (row-major over `[BM, BLOCK_K]`), step `t`:
  `clampIdx(pid_m·BM + i, M)·sam + (t·BK + e)·sak` — the invariant's `a_ptrs`
  cell after `t` advances, through the kernel's `tl.where` row clamp.
* `read2` lane `l = (e, j)` (row-major over `[BLOCK_K, BN]`), step `t`:
  `(t·BK + e)·sbk + clampIdx(pid_n·BN + j, N)·sbn` — the `b_ptrs` cell.
* `write` lane `l = (i, j)`: `scm·(pid_m·BM + i) + scn·(pid_n·BN + j)` — the
  kernel's un-clamped `c_ptrs` (= `cOffset` in pid form).
* `mask1`/`mask2` transcribe the loads' `offs_k < K - kk·BK` windows in the
  per-lane spelling `t·BK + e < K`; `writeMask` transcribes the store's
  `(row<M) & (col<N)` boundary mask verbatim. -/
```
```lean
def tritonMatmulF16IO (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat) :
    StreamMasked2DKernelIO₂ where
  kernel := triton_matmul_f16_surface A B C M N K sam sak sbk sbn scm scn BM BN BK GM
    numKBlocks
  inp1 := A
  inp2 := B
  out := C
  T := numKBlocks
  B1 := BM * BK
  B2 := BK * BN
  C := BM * BN
  outDType := .fp16
  read1 := fun p₀ _ t l =>
    clampIdx (pidM p₀ M N BM BN GM * BM + l.val / BK) M * sam + (t.val * BK + l.val % BK) * sak
  read2 := fun p₀ _ t l =>
    (t.val * BK + l.val / BN) * sbk + clampIdx (pidN p₀ M N BM BN GM * BN + l.val % BN) N * sbn
  write := fun p₀ _ l =>
    scm * (pidM p₀ M N BM BN GM * BM + l.val / BN) + scn * (pidN p₀ M N BM BN GM * BN + l.val % BN)
  mask1 := fun _ _ t l => t.val * BK + l.val % BK < K
  mask2 := fun _ _ t l => t.val * BK + l.val / BN < K
  writeMask := fun p₀ _ l =>
    pidM p₀ M N BM BN GM * BM + l.val / BN < M ∧ pidN p₀ M N BM BN GM * BN + l.val % BN < N
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

<details><summary><code>triton_matmul_f16_surface</code></summary>

```
/-- Faithful transcription of `triton_matmul.py`'s `matmul_kernel`, **fp16
epilogue arm** (`c_ptr.dtype.element_ty ≠ tl.float8e4nv`).

The contracted dimension is presented as `K = BLOCK_SIZE_K · numKBlocks` so the
loop bound `tl.cdiv(K, BLOCK_SIZE_K) = numKBlocks` is exact; it is supplied as
the antiquoted `numKBlocks`. All other surface structure — the L2-grouping
schedule, the `tl.where` index clamps, the `tl.max_contiguous`/`tl.multiple_of`
hints (DSL-erased to their value argument), the per-block
`offs_k < K - kk·BLOCK_K` load masks, the fused `tl.dot(a, b, accumulator)`,
the `float16` cast, and the `(row<M)&(col<N)`-masked store — is transcribed
verbatim. -/
```
```lean
def triton_matmul_f16_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  start_m = pid_m * $(BLOCK_SIZE_M)
  start_n = pid_n * $(BLOCK_SIZE_N)
  offs_am = start_m + tl.arange(0, $(BLOCK_SIZE_M))
  offs_bn = start_n + tl.arange(0, $(BLOCK_SIZE_N))
  offs_am = tl.where(offs_am < $(M), offs_am, $(0))
  offs_bn = tl.where(offs_bn < $(N), offs_bn, $(0))
  offs_am = tl.max_contiguous(tl.multiple_of(offs_am, $(BLOCK_SIZE_M)), $(BLOCK_SIZE_M))
  offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, $(BLOCK_SIZE_N)), $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
  b_ptrs = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for kk in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}
```
</details>

<details><summary><code>clampIdx</code></summary>

```
/-- The kernel's `tl.where(v < bound, v, 0)` index clamp (this kernel's
out-of-range guard, where the `matmul_triton_autotune` twin wraps with `%`). -/
```
```lean
def clampIdx (v bound : Nat) : Nat := if v < bound then v else 0
```
</details>

<details><summary><code>pidM</code></summary>

```
/-- The kernel's L2-grouping derivation of `pid_m` from the linear `pid` —
**this** kernel's spelling `first_pid_m + (pid % group_size_m)` (the twin
`matmul_triton_autotune` reduces `pid % num_pid_in_group` first). -/
```
```lean
def pidM (pid M N BM BN GM : Nat) : Nat :=
  let num_pid_m := cdiv M BM
  let num_pid_n := cdiv N BN
  let num_pid_in_group := GM * num_pid_n
  let group_id := pid / num_pid_in_group
  let first_pid_m := group_id * GM
  let group_size_m := kernelMin (num_pid_m - first_pid_m) GM
  first_pid_m + (pid % group_size_m)
```
</details>

<details><summary><code>pidN</code></summary>

```
/-- The kernel's L2-grouping derivation of `pid_n` from the linear `pid`. -/
```
```lean
def pidN (pid M N BM BN GM : Nat) : Nat :=
  let num_pid_m := cdiv M BM
  let num_pid_n := cdiv N BN
  let num_pid_in_group := GM * num_pid_n
  let group_id := pid / num_pid_in_group
  let first_pid_m := group_id * GM
  let group_size_m := kernelMin (num_pid_m - first_pid_m) GM
  (pid % num_pid_in_group) / group_size_m
```
</details>

<details><summary><code>kernelMin</code></summary>

```
/-- `min` as the kernel's `tl.where(a < b, a, b)` spells it. -/
```
```lean
def kernelMin (a b : Nat) : Nat := if a < b then a else b
```
</details>

## Public theorem: `triton_matmul_f8_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline, fp8 epilogue arm** — the corpus's first
fp8 matmul face. Identical ideal-ℝ GEMM fold spec to
`triton_matmul_f16_io_correctness`; the boundary grid is `.f8e4`: the output
window reads back `.f8e4`-typed cells holding
`f8e4.ofReal (R.round .f8e4 (∑ t, ∑ e, A·B))`, the kernel's
`.to(tl.float8e4nv)` cast + fp8-typed masked store collapsed to one
`R.round .f8e4` by the defining `round_idem`. Hypotheses truth-forced exactly
as in the fp16 arm. -/
```
</details>

**Statement:**
```lean
specification triton_matmul_f8_io_correctness (R : RoundingModel)
    (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat)
    (hK : K = BK * numKBlocks) (hcn : scn = 1) (hBN : BN ≤ scm) :
    tritonMatmulF8IO A B C M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks
      ⊨[R] fun _ _ xs ys l =>
        ∑ t : Fin numKBlocks, ∑ e : Fin BK,
          xs t (aLane BM BN BK l e) * ys t (bLane BM BN BK l e)
```

**Assumptions / layout contracts:**
- `hK : K = BK * numKBlocks`
- `hcn : scn = 1`
- `hBN : BN ≤ scm`

**Closed-form spec defs (transitive):** `tritonMatmulF8IO`, `aLane`, `bLane`, `triton_matmul_f8_surface`, `clampIdx`, `pidM`, `pidN`, `kernelMin`

<details><summary><code>tritonMatmulF8IO</code></summary>

```
/-- **Streaming IO signature** of `triton_matmul`'s **fp8 epilogue arm**:
identical windows to `tritonMatmulF16IO`, with the fp8 surface and the
`outDType := .f8e4` boundary grid (this arm's `.to(tl.float8e4nv)` + fp8
store). -/
```
```lean
def tritonMatmulF8IO (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat) :
    StreamMasked2DKernelIO₂ where
  kernel := triton_matmul_f8_surface A B C M N K sam sak sbk sbn scm scn BM BN BK GM
    numKBlocks
  inp1 := A
  inp2 := B
  out := C
  T := numKBlocks
  B1 := BM * BK
  B2 := BK * BN
  C := BM * BN
  outDType := .f8e4
  read1 := fun p₀ _ t l =>
    clampIdx (pidM p₀ M N BM BN GM * BM + l.val / BK) M * sam + (t.val * BK + l.val % BK) * sak
  read2 := fun p₀ _ t l =>
    (t.val * BK + l.val / BN) * sbk + clampIdx (pidN p₀ M N BM BN GM * BN + l.val % BN) N * sbn
  write := fun p₀ _ l =>
    scm * (pidM p₀ M N BM BN GM * BM + l.val / BN) + scn * (pidN p₀ M N BM BN GM * BN + l.val % BN)
  mask1 := fun _ _ t l => t.val * BK + l.val % BK < K
  mask2 := fun _ _ t l => t.val * BK + l.val / BN < K
  writeMask := fun p₀ _ l =>
    pidM p₀ M N BM BN GM * BM + l.val / BN < M ∧ pidN p₀ M N BM BN GM * BN + l.val % BN < N
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

<details><summary><code>triton_matmul_f8_surface</code></summary>

```
/-- Faithful transcription of `triton_matmul.py`'s `matmul_kernel`, **fp8
epilogue arm** (`c_ptr.dtype.element_ty == tl.float8e4nv`): identical to
`triton_matmul_f16_surface` except the epilogue cast
`c = (accumulator).to(tl.float8e4nv)`. -/
```
```lean
def triton_matmul_f8_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  start_m = pid_m * $(BLOCK_SIZE_M)
  start_n = pid_n * $(BLOCK_SIZE_N)
  offs_am = start_m + tl.arange(0, $(BLOCK_SIZE_M))
  offs_bn = start_n + tl.arange(0, $(BLOCK_SIZE_N))
  offs_am = tl.where(offs_am < $(M), offs_am, $(0))
  offs_bn = tl.where(offs_bn < $(N), offs_bn, $(0))
  offs_am = tl.max_contiguous(tl.multiple_of(offs_am, $(BLOCK_SIZE_M)), $(BLOCK_SIZE_M))
  offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, $(BLOCK_SIZE_N)), $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
  b_ptrs = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for kk in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  c = (accumulator).to(tl.float8e4nv)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}
```
</details>

<details><summary><code>clampIdx</code></summary>

```
/-- The kernel's `tl.where(v < bound, v, 0)` index clamp (this kernel's
out-of-range guard, where the `matmul_triton_autotune` twin wraps with `%`). -/
```
```lean
def clampIdx (v bound : Nat) : Nat := if v < bound then v else 0
```
</details>

<details><summary><code>pidM</code></summary>

```
/-- The kernel's L2-grouping derivation of `pid_m` from the linear `pid` —
**this** kernel's spelling `first_pid_m + (pid % group_size_m)` (the twin
`matmul_triton_autotune` reduces `pid % num_pid_in_group` first). -/
```
```lean
def pidM (pid M N BM BN GM : Nat) : Nat :=
  let num_pid_m := cdiv M BM
  let num_pid_n := cdiv N BN
  let num_pid_in_group := GM * num_pid_n
  let group_id := pid / num_pid_in_group
  let first_pid_m := group_id * GM
  let group_size_m := kernelMin (num_pid_m - first_pid_m) GM
  first_pid_m + (pid % group_size_m)
```
</details>

<details><summary><code>pidN</code></summary>

```
/-- The kernel's L2-grouping derivation of `pid_n` from the linear `pid`. -/
```
```lean
def pidN (pid M N BM BN GM : Nat) : Nat :=
  let num_pid_m := cdiv M BM
  let num_pid_n := cdiv N BN
  let num_pid_in_group := GM * num_pid_n
  let group_id := pid / num_pid_in_group
  let first_pid_m := group_id * GM
  let group_size_m := kernelMin (num_pid_m - first_pid_m) GM
  (pid % num_pid_in_group) / group_size_m
```
</details>

<details><summary><code>kernelMin</code></summary>

```
/-- `min` as the kernel's `tl.where(a < b, a, b)` spells it. -/
```
```lean
def kernelMin (a b : Nat) : Nat := if a < b then a else b
```
</details>
