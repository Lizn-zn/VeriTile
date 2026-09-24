# Spec sheet — `bench/tritonbench_g/matmul_leakyrelu_fp8/MatmulLeakyreluFp8.lean`

**Python source:** `bench/tritonbench_g/matmul_leakyrelu_fp8/matmul_leakyrelu_fp8.py`

## Public theorem: `matmul_leakyrelu_fp8_closed_form_correct`

<details><summary>docstring</summary>

```
/-- **Closed-form correctness for `matmul_leakyrelu_fp8` (general statement).**

For arbitrary `M`/`N`, tile dims `BM`/`BN`, strides, group size, K-block size
`BK`, and K-block count `numKBlocks` (so `K = BK · numKBlocks`), every in-bounds
output cell of the computed `BM × BN` tile equals the genuine fused value
`fp16(leakyrelu(Σ_{k < BK·numKBlocks} A[i,k] · B[k,j]))` (over ℝ, with a final
fp16 output cast) of the loaded `A`/`B` tiles — NOT the kernel's own executed
value.

`PM`/`PN` are the kernel's own grouped `pid_m`/`pid_n`. Preconditions: `0 < BK`;
all tile rows/cols in-bounds (`PM·BM + i < M`, `PN·BN + j < N`), making the
modular addressing the identity and the store mask all-true; output-address
injectivity; clean initial `undef`. -/
```
</details>

**Statement:**
```lean
specification matmul_leakyrelu_fp8_closed_form_correct
    (A B C : RegionName) (s : BlockState)
    (M N SAM SAK SBK SBN SCM SCN BM BN BK GROUP numKBlocks : Nat) (hBK : 0 < BK)
    (hcn : SCN = 1) (hbnle : BN ≤ SCM)
    (hmlt : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) M N BM BN GROUP) BM i < M)
    (hnlt : ∀ j : Fin BN, colIndex (pidN (s.pids 0) M N BM BN GROUP) BN j < N)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := matmul_leaky_relu_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN BM BN BK GROUP)
      (initialState := s)
      (write := fun idx : TileIndex [BM, BN] =>
        some (C, cOffset s (pidM (s.pids 0) M N BM BN GROUP) (pidN (s.pids 0) M N BM BN GROUP) BM BN SCM SCN idx))
      (expected := fun idx : TileIndex [BM, BN] =>
        outputCell s A B (pidM (s.pids 0) M N BM BN GROUP) (pidN (s.pids 0) M N BM BN GROUP)
          BM BN M N SAM SAK SBK SBN BK numKBlocks idx)
```

**Assumptions / layout contracts:**
- `hBK : 0 < BK`
- `hcn : SCN = 1`
- `hbnle : BN ≤ SCM`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `rowIndex`, `pidM`, `colIndex`, `pidN`, `matmul_leaky_relu_surface`, `cOffset`, `outputCell`, `numPidInGroup`, `groupSizeM`, `leakyrelu`, `matmulSpec`, `aElem`, `bElem`

<details><summary><code>rowIndex</code></summary>

```
/-- Global output row of tile lane `i`: `PM · BM + i`. -/
```
```lean
def rowIndex (PM BM : Nat) (i : Fin BM) : Nat := PM * BM + i.val
```
</details>

<details><summary><code>pidM</code></summary>

```
/-- The kernel's derived `pid_m` for program `pid`. -/
```
```lean
def pidM (pid M N BM BN GROUP : Nat) : Nat :=
  pid / numPidInGroup N BN GROUP * GROUP
    + pid % numPidInGroup N BN GROUP % groupSizeM pid M N BM BN GROUP
```
</details>

<details><summary><code>colIndex</code></summary>

```
/-- Global output column of tile lane `j`: `PN · BN + j`. -/
```
```lean
def colIndex (PN BN : Nat) (j : Fin BN) : Nat := PN * BN + j.val
```
</details>

<details><summary><code>pidN</code></summary>

```
/-- The kernel's derived `pid_n` for program `pid`. -/
```
```lean
def pidN (pid M N BM BN GROUP : Nat) : Nat :=
  pid % numPidInGroup N BN GROUP / groupSizeM pid M N BM BN GROUP
```
</details>

<details><summary><code>matmul_leaky_relu_surface</code></summary>

```
/-- Surface transcription of `matmul_leakyrelu_fp8.py`'s `matmul_kernel` for
`ACTIVATION == "leaky_relu"`. -/
```
```lean
def matmul_leaky_relu_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
  b_ptrs = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  accumulator = tl.where(accumulator >= 0.0, accumulator, 0.01 * accumulator)
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}
```
</details>

<details><summary><code>cOffset</code></summary>

```
/-! ## Masked fp16 output-store machinery (reused for the activated store) -/
```
```lean
def cOffset (_s : BlockState) (PM PN BM BN stride_cm stride_cn : Nat)
    (idx : TileIndex [BM, BN]) : Nat :=
  stride_cm * rowIndex PM BM idx.1 + stride_cn * colIndex PN BN idx.2.1
```
</details>

<details><summary><code>outputCell</code></summary>

```
/-- The genuine output cell: `fp16(leakyrelu(Σ_k A·B))` as a `MemCell`. -/
```
```lean
noncomputable def outputCell (s0 : BlockState) (A B : RegionName)
    (PM PN BM BN M N SAM SAK SBK SBN BK numKBlocks : Nat) (idx : TileIndex [BM, BN]) : MemCell :=
  MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue
    (FloatDType.real.cast FloatDType.fp16
      (some (leakyrelu (matmulSpec s0 A B PM PN BM BN M N SAM SAK SBK SBN BK numKBlocks idx.1 idx.2.1))))))
```
</details>

<details><summary><code>numPidInGroup</code></summary>

```
/-- The kernel's grouped `num_pid_in_group = GROUP · cdiv N BN`. -/
```
```lean
def numPidInGroup (N BN GROUP : Nat) : Nat := GROUP * cdiv N BN
```
</details>

<details><summary><code>groupSizeM</code></summary>

```
/-- The kernel's `group_size_m` for program `pid`
(`min(num_pid_m − first_pid_m, GROUP)`, written with `<` as the kernel does). -/
```
```lean
def groupSizeM (pid M N BM BN GROUP : Nat) : Nat :=
  let num_pid_m := cdiv M BM
  let first_pid_m := pid / numPidInGroup N BN GROUP * GROUP
  if num_pid_m - first_pid_m < GROUP then num_pid_m - first_pid_m else GROUP
```
</details>

<details><summary><code>leakyrelu</code></summary>

```
/-- Leaky-ReLU activation `if v ≥ 0 then v else 0.01·v`. -/
```
```lean
noncomputable def leakyrelu (v : ℝ) : ℝ := if v ≥ 0 then v else (1e-2 : ℝ) * v
```
</details>

<details><summary><code>matmulSpec</code></summary>

```
/-- **Genuine GEMM spec**: `C[i,j] = Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j]`. -/
```
```lean
noncomputable def matmulSpec (s : BlockState) (A B : RegionName)
    (PM PN BM BN M N SAM SAK SBK SBN BLOCK_K numKBlocks : Nat) (i : Fin BM) (j : Fin BN) : ℝ :=
  gemmSum (aElem s A PM BM M SAM SAK i) (bElem s B PN BN N SBK SBN j) (BLOCK_K * numKBlocks)
```
</details>

<details><summary><code>aElem</code></summary>

```
/-- `A[i, k] = readMem A (offs_am i · stride_am + k · stride_ak)` (kernel's strided
A layout, with `offs_am i = (PM·BM + i) % M`). -/
```
```lean
noncomputable def aElem (s : BlockState) (A : RegionName) (PM BM M SAM SAK : Nat)
    (i : Fin BM) (k : Nat) : ℝ :=
  s.readMem A (rowIndex PM BM i % M * SAM + k * SAK)
```
</details>

<details><summary><code>bElem</code></summary>

```
/-- `B[k, j] = readMem B (k · stride_bk + offs_bn j · stride_bn)` (kernel's strided
B layout, with `offs_bn j = (PN·BN + j) % N`). -/
```
```lean
noncomputable def bElem (s : BlockState) (B : RegionName) (PN BN N SBK SBN : Nat)
    (j : Fin BN) (k : Nat) : ℝ :=
  s.readMem B (k * SBK + colIndex PN BN j % N * SBN)
```
</details>

## Public theorem: `matmul_leakyrelu_fp8_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S1 fold genre).** For every
rounding model `R`, the faithful `matmul_leakyrelu_fp8` surface implements, on
its `StreamMasked2DKernelIO₂` signature, the **ideal ℝ leaky-ReLU GEMM fold**
over the streamed tiles: output lane `l = (i, j)` holds
`leakyrelu (∑ t, ∑ e, A-tile[t](i,e) · B-tile[t](e,j))` — the spec `f` is
exact real arithmetic with the activation expressed through the named
`leakyrelu` brancher, and the single boundary quantization is carried by the
skin's readback contract (`readMemAs .fp16` holds
`fp16.ofReal (R.round .fp16 (f …))`), where the kernel's two rounding events
(the `.to(tl.float16)` cast and the fp16-typed masked store) collapse to one
`R.round .fp16` by the defining `round_idem`.

Layer map: the 15-statement prologue and the whole dynamic K-loop (masked
loads, `other=0.0`) are cast-free, so under `execR R` they collapse verbatim
onto the exact stepper and the proven `mlr_preLoop` / `mlr_step` /
`forRangeAux_inv` stack above is reused unchanged; only the 7-statement
activation+store tail is re-proved on the `R` side (`mlr_postLoopR`).

Every hypothesis is truth-forced:

* `hBK : 0 < BLOCK_K` — the loop's trip count is `tl.cdiv(K, BLOCK_K)`,
  which equals `numKBlocks` at `K = BLOCK_K · numKBlocks` only for a nonzero
  block size; at `BLOCK_K = 0` the block-index arithmetic `l / BLOCK_K` is
  meaningless. It holds for every real launch (`tl.arange(0, 0)` is not a
  tile).
* `hscn : stride_cn = 1` and `hbnle : BLOCK_N ≤ stride_cm` —
  output-offset injectivity (`rowMajor2D_inj`, exactly the exact headline's
  precondition): the column stride is the unit stride and the column-block
  width fits the row stride, so distinct output lanes hit distinct
  addresses; with colliding lanes the per-lane readback would be
  last-writer-wins and the statement false. Both hold for every valid
  row-major tiling.

Note this `⊨[R]` face is *stronger* than the exact headline
`matmul_leakyrelu_fp8_closed_form_correct` in one respect: that one assumes the
store mask is all-true (`hmlt`/`hnlt`), while the skin carries the genuine
`(offs_cm < M) & (offs_cn < N)` boundary mask in `writeMask`, so no
in-bounds-tile hypothesis is needed here. Both faces are kept per the
rounding-as-default doctrine. -/
```
</details>

**Statement:**
```lean
specification matmul_leakyrelu_fp8_io_correctness (R : RoundingModel)
    (A B C : RegionName)
    (M N SAM SAK SBK SBN SCM SCN BM BN BK GROUP numKBlocks : Nat)
    (hBK : 0 < BK) (hscn : SCN = 1) (hbnle : BN ≤ SCM) :
    matmulLeakyreluFp8IO A B C M N SAM SAK SBK SBN SCM SCN BM BN BK GROUP numKBlocks
      ⊨[R] fun _ _ xs ys l =>
        leakyrelu (∑ t : Fin numKBlocks, ∑ e : Fin BK,
          xs t (aLane BM BN BK l e) * ys t (bLane BM BN BK l e))
```

**Assumptions / layout contracts:**
- `hBK : 0 < BK`
- `hscn : SCN = 1`
- `hbnle : BN ≤ SCM`

**Closed-form spec defs (transitive):** `matmulLeakyreluFp8IO`, `leakyrelu`, `aLane`, `bLane`, `matmul_leaky_relu_surface`, `pidM`, `pidN`, `numPidInGroup`, `groupSizeM`

<details><summary><code>matmulLeakyreluFp8IO</code></summary>

```
/-- **Streaming IO signature** of `matmul_leakyrelu_fp8` on the two-stream fold
skin (S1: fold + terminal store). Step `t` of the K-loop reads the
`[BM, BLOCK_K]` `A`-tile and the `[BLOCK_K, BN]` `B`-tile; after the loop one
`[BM, BN]` output tile is stored at the **fp16** grid (`outDType := .fp16` —
the kernel's `(accumulator).to(tl.float16)` survives lowering as an
`Op.castFloat` feeding a `Stmt.store .fp16`). The kernel schedules on a
**single** linear `pid` (`program_id(0)`; the skin's `pid₁` slot is unused),
so every window derives `(pid_m, pid_n)` through the transcribed L2-grouping
arithmetic `pidM`/`pidN`:

* `read1` lane `l = (i, e)` (row-major over `[BM, BLOCK_K]`), step `t`:
  `((pid_m·BM + i) % M)·stride_am + (t·BK + e)·stride_ak` — the invariant's
  `a_ptrs` cell after `t` advances.
* `read2` lane `l = (e, j)` (row-major over `[BLOCK_K, BN]`), step `t`:
  `(t·BK + e)·stride_bk + ((pid_n·BN + j) % N)·stride_bn` — the `b_ptrs`
  cell.
* `write` lane `l = (i, j)`:
  `stride_cm·(pid_m·BM + i) + stride_cn·(pid_n·BN + j)` — the kernel's
  un-wrapped `c_ptrs` (= `cOffset` in pid form).
* `mask1`/`mask2` transcribe the loads' `offs_k < K − k·BK` windows in the
  per-lane spelling `t·BK + e < K`; `writeMask` transcribes the store's
  `(offs_cm < M) & (offs_cn < N)` boundary mask verbatim. -/
```
```lean
def matmulLeakyreluFp8IO (A B C : RegionName)
    (M N SAM SAK SBK SBN SCM SCN BM BN BK GROUP numKBlocks : Nat) :
    StreamMasked2DKernelIO₂ where
  kernel := matmul_leaky_relu_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN
    BM BN BK GROUP
  inp1 := A
  inp2 := B
  out := C
  T := numKBlocks
  B1 := BM * BK
  B2 := BK * BN
  C := BM * BN
  outDType := .fp16
  read1 := fun p₀ _ t l =>
    (pidM p₀ M N BM BN GROUP * BM + l.val / BK) % M * SAM + (t.val * BK + l.val % BK) * SAK
  read2 := fun p₀ _ t l =>
    (t.val * BK + l.val / BN) * SBK + (pidN p₀ M N BM BN GROUP * BN + l.val % BN) % N * SBN
  write := fun p₀ _ l =>
    SCM * (pidM p₀ M N BM BN GROUP * BM + l.val / BN)
      + SCN * (pidN p₀ M N BM BN GROUP * BN + l.val % BN)
  mask1 := fun _ _ t l => t.val * BK + l.val % BK < BK * numKBlocks
  mask2 := fun _ _ t l => t.val * BK + l.val / BN < BK * numKBlocks
  writeMask := fun p₀ _ l =>
    pidM p₀ M N BM BN GROUP * BM + l.val / BN < M ∧
      pidN p₀ M N BM BN GROUP * BN + l.val % BN < N
```
</details>

<details><summary><code>leakyrelu</code></summary>

```
/-- Leaky-ReLU activation `if v ≥ 0 then v else 0.01·v`. -/
```
```lean
noncomputable def leakyrelu (v : ℝ) : ℝ := if v ≥ 0 then v else (1e-2 : ℝ) * v
```
</details>

<details><summary><code>aLane</code></summary>

```
/-- The `A`-stream lane feeding output lane `l` at inner key `e`: the row of
`l` (row-major over the `[BM, BN]` output tile) paired with `e` over the
`[BM, BLOCK_K]` per-step `A`-tile. -/
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

<details><summary><code>matmul_leaky_relu_surface</code></summary>

```
/-- Surface transcription of `matmul_leakyrelu_fp8.py`'s `matmul_kernel` for
`ACTIVATION == "leaky_relu"`. -/
```
```lean
def matmul_leaky_relu_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
  b_ptrs = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  accumulator = tl.where(accumulator >= 0.0, accumulator, 0.01 * accumulator)
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}
```
</details>

<details><summary><code>pidM</code></summary>

```
/-- The kernel's derived `pid_m` for program `pid`. -/
```
```lean
def pidM (pid M N BM BN GROUP : Nat) : Nat :=
  pid / numPidInGroup N BN GROUP * GROUP
    + pid % numPidInGroup N BN GROUP % groupSizeM pid M N BM BN GROUP
```
</details>

<details><summary><code>pidN</code></summary>

```
/-- The kernel's derived `pid_n` for program `pid`. -/
```
```lean
def pidN (pid M N BM BN GROUP : Nat) : Nat :=
  pid % numPidInGroup N BN GROUP / groupSizeM pid M N BM BN GROUP
```
</details>

<details><summary><code>numPidInGroup</code></summary>

```
/-- The kernel's grouped `num_pid_in_group = GROUP · cdiv N BN`. -/
```
```lean
def numPidInGroup (N BN GROUP : Nat) : Nat := GROUP * cdiv N BN
```
</details>

<details><summary><code>groupSizeM</code></summary>

```
/-- The kernel's `group_size_m` for program `pid`
(`min(num_pid_m − first_pid_m, GROUP)`, written with `<` as the kernel does). -/
```
```lean
def groupSizeM (pid M N BM BN GROUP : Nat) : Nat :=
  let num_pid_m := cdiv M BM
  let first_pid_m := pid / numPidInGroup N BN GROUP * GROUP
  if num_pid_m - first_pid_m < GROUP then num_pid_m - first_pid_m else GROUP
```
</details>
