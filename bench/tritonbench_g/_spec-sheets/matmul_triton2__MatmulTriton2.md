# Spec sheet — `bench/tritonbench_g/matmul_triton2/MatmulTriton2.lean`

**Python source:** `bench/tritonbench_g/matmul_triton2/matmul_triton2.py`

## Public theorem: `matmul_triton2_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Public dimension-general summary**: for arbitrary matrix dims `M`/`N`, tile
dims `BM`/`BN`, group size `GM`, strides, K-block size `BK` (`0 < BK`), and
K-block count `numKBlocks` (so `K = BK · numKBlocks`), the full `matmul_triton2`
surface (1) lowers to the algorithm layer and (2) realizes the genuine matrix
product `Σ_{k < BK·numKBlocks} A[i,k]·B[k,j]` (over ℝ, reading INPUT memory) on
every active output lane. The `expected` is the closed-form `matmulSpec`, NOT the
kernel's own executed value. -/
```
</details>

**Statement:**
```lean
specification matmul_triton2_output_summary_general
    (A B C : RegionName) (s : BlockState)
    (M N BM BN GM SAM SAK SBK SBN SCM SCN BK numKBlocks : Nat) (hBK : 0 < BK)
    (hInj : Function.Injective (cOffset s M N BM BN GM SCM SCN))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (matmul_triton2_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN
        BM BN BK GM).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := matmul_triton2_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN
        BM BN BK GM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BM BN GM)
        (fun idx => (C, cOffset s M N BM BN GM SCM SCN idx)))
      (expected := fun idx : TileIndex [BM, BN] =>
        matmulSpec s A B M N BM BN GM SAM SAK SBK SBN BK numKBlocks idx.1 idx.2.1)
```

**Assumptions / layout contracts:**
- `hBK : 0 < BK`
- `hInj : Function.Injective (cOffset s M N BM BN GM SCM SCN)`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `cOffset`, `matmul_triton2_surface`, `active`, `matmulSpec`, `rowIndex`, `colIndex`, `cdiv`, `aElem`, `bElem`, `pidM`, `pidN`, `numPidM`, `numPidN`

<details><summary><code>cOffset</code></summary>

```
/-- The output store address for tile lane `(i,j)`: `rowIndex i · SCM + colIndex j · SCN`. -/
```
```lean
def cOffset (s0 : BlockState) (M N BM BN GM SCM SCN : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  rowIndex s0 M N BM BN GM idx.1 * SCM + colIndex s0 M N BM BN GM idx.2.1 * SCN
```
</details>

<details><summary><code>matmul_triton2_surface</code></summary>

```
/-- Faithful transcription of `matmul_triton2.py`'s `matmul_kernel`. -/
```
```lean
def matmul_triton2_surface
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
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_bn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
  b_ptrs = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    accumulator += tl.dot(a, b)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  c_ptrs = C + offs_am[:, None] * $(stride_cm) + offs_bn[None, :] * $(stride_cn)
  c_mask = (offs_am[:, None] < $(M)) & (offs_bn[None, :] < $(N))
  tl.store(c_ptrs, accumulator, mask=c_mask)
}
```
</details>

<details><summary><code>active</code></summary>

```
/-- Active output lane: `rowIndex i < M ∧ colIndex j < N`. -/
```
```lean
def active (s0 : BlockState) (M N BM BN GM : Nat) (idx : TileIndex [BM, BN]) : Prop :=
  rowIndex s0 M N BM BN GM idx.1 < M ∧ colIndex s0 M N BM BN GM idx.2.1 < N

instance (s0 : BlockState) (M N BM BN GM : Nat) (idx : TileIndex [BM, BN]) :
    Decidable (active s0 M N BM BN GM idx) := by unfold active; infer_instance
```
</details>

<details><summary><code>matmulSpec</code></summary>

```
/-- **Genuine GEMM spec**: `C[i,j] = Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j]`. -/
```
```lean
noncomputable def matmulSpec (s : BlockState) (A B : RegionName)
    (M N BM BN GM SAM SAK SBK SBN BLOCK_K numKBlocks : Nat) (i : Fin BM) (j : Fin BN) : ℝ :=
  (Finset.range (BLOCK_K * numKBlocks)).sum
    (fun k => aElem s A M N BM BN GM SAM SAK i k * bElem s B M N BM BN GM SBK SBN j k)
```
</details>

<details><summary><code>rowIndex</code></summary>

```
/-- Global output row of tile lane `i`: `pid_m · BLOCK_M + i`. -/
```
```lean
def rowIndex (s : BlockState) (M N BM BN GM : Nat) (i : Fin BM) : Nat :=
  pidM s M N BM BN GM * BM + i.val
```
</details>

<details><summary><code>colIndex</code></summary>

```
/-- Global output column of tile lane `j`: `pid_n · BLOCK_N + j`. -/
```
```lean
def colIndex (s : BlockState) (M N BM BN GM : Nat) (j : Fin BN) : Nat :=
  pidN s M N BM BN GM * BN + j.val
```
</details>

<details><summary><code>cdiv</code></summary>

```
/-- Ceiling division `⌈a / b⌉`, matching Triton's `tl.cdiv`. -/
```
```lean
def cdiv (a b : Nat) : Nat := (a + b - 1) / b
```
</details>

<details><summary><code>aElem</code></summary>

```
/-- `A[i, k] = readMem A (rowIndex i · SAM + k · SAK)` (kernel's A layout). -/
```
```lean
noncomputable def aElem (s : BlockState) (A : RegionName) (M N BM BN GM SAM SAK : Nat)
    (i : Fin BM) (k : Nat) : ℝ :=
  s.readMem A (rowIndex s M N BM BN GM i * SAM + k * SAK)
```
</details>

<details><summary><code>bElem</code></summary>

```
/-- `B[k, j] = readMem B (k · SBK + colIndex j · SBN)` (kernel's B layout). -/
```
```lean
noncomputable def bElem (s : BlockState) (B : RegionName) (M N BM BN GM SBK SBN : Nat)
    (j : Fin BN) (k : Nat) : ℝ :=
  s.readMem B (k * SBK + colIndex s M N BM BN GM j * SBN)
```
</details>

<details><summary><code>pidM</code></summary>

```
/-- Group-scheduled tile-row index of program `pid`. Mirrors the kernel's
`pid_m = first_pid_m + (pid % group_size_m)` with `group_size_m =
min(num_pid_m − first_pid_m, GROUP_SIZE_M)` and `first_pid_m = (pid //
num_pid_in_group)·GROUP_SIZE_M`. -/
```
```lean
def pidM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  let nm := numPidM M BM
  let nn := numPidN N BN
  let nig := GM * nn
  let gid := s.pids 0 / nig
  let fpm := gid * GM
  let gsm := if nm - fpm < GM then nm - fpm else GM
  fpm + s.pids 0 % gsm
```
</details>

<details><summary><code>pidN</code></summary>

```
/-- Group-scheduled tile-col index of program `pid`. Mirrors
`pid_n = (pid % num_pid_in_group) // group_size_m`. -/
```
```lean
def pidN (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  let nm := numPidM M BM
  let nn := numPidN N BN
  let nig := GM * nn
  let gid := s.pids 0 / nig
  let fpm := gid * GM
  let gsm := if nm - fpm < GM then nm - fpm else GM
  s.pids 0 % nig / gsm
```
</details>

<details><summary><code>numPidM</code></summary>

```
/-- `num_pid_m = cdiv M BLOCK_M`. -/
```
```lean
def numPidM (M BM : Nat) : Nat := cdiv M BM
```
</details>

<details><summary><code>numPidN</code></summary>

```
/-- `num_pid_n = cdiv N BLOCK_N`. -/
```
```lean
def numPidN (N BN : Nat) : Nat := cdiv N BN
```
</details>

## Public theorem: `matmul_triton2_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S1 fold genre).** For every
rounding model `R`, the faithful `matmul_triton2` surface implements, on its
`StreamMasked2DKernelIO₂` signature, the **ideal ℝ GEMM fold** over the
streamed tiles: output lane `l = (i, j)` holds
`∑ t, ∑ e, A-tile[t](i,e) · B-tile[t](e,j)` — the spec `f` is exact real
arithmetic. The kernel has **no rounding events** (masked loads, `tl.dot` and
the masked store are all at `.real`), so the skin's boundary quantization
degenerates: the readback contract's `R.round .real` is the identity by the
model's defining `round_real`, and the `.real` terminal store is exact under
`execR R` (`stepStmtR` delegates `.real` writes to the exact `writeMemTyped`;
`RoundingModel.storeValue_real`).

Layer map: the 15-statement prologue and the whole dynamic K-loop are
cast-free, so under `execR R` they collapse verbatim onto the exact stepper
and the proven `preLoop` / `matmul_step` / `forRangeDyn_inv` stack above is
reused unchanged; only the three-statement store tail is re-proved on the `R`
side (`matmul_triton2_postLoopR`).

Every hypothesis is truth-forced:

* `hBK : 0 < BLOCK_K` — the loop's trip count is `tl.cdiv(K, BLOCK_K)`, which
  is `numKBlocks` only for a nonzero block size; at `BLOCK_K = 0` the
  contraction `K = BLOCK_K · numKBlocks` is empty and the block-index
  arithmetic is meaningless. It holds for every real launch
  (`tl.arange(0, 0)` is not a tile).
* `hcn : stride_cn = 1` and `hBN : BLOCK_N ≤ stride_cm` — output-offset
  injectivity (`rowMajor2D_inj`): the column stride is the unit stride and
  the column-block width fits the row stride, so distinct output lanes hit
  distinct addresses; with colliding lanes the per-lane readback would be
  last-writer-wins and the statement false. Both hold for every valid
  row-major tiling (the Python launch passes `c.stride() = (N, 1)`).

Relation to the exact surface: the exact headline
`matmul_triton2_closed_form_correct` (`Realizes_without_Rounding`) above is
retained unchanged; this `⊨[R]` face restates the same GEMM content on the
streaming skin, for every `R` at once (at the `.real` grid the two faces
carry the same exact cell). Both faces are kept per the rounding-as-default
doctrine. -/
```
</details>

**Statement:**
```lean
specification matmul_triton2_io_correctness (R : RoundingModel)
    (A B C : RegionName)
    (M N BM BN GM SAM SAK SBK SBN SCM SCN BK numKBlocks : Nat)
    (hBK : 0 < BK) (hcn : SCN = 1) (hBN : BN ≤ SCM) :
    matmulTriton2IO A B C M N BM BN GM SAM SAK SBK SBN SCM SCN BK numKBlocks
      ⊨[R] fun _ _ xs ys l =>
        ∑ t : Fin numKBlocks, ∑ e : Fin BK,
          xs t (aLane BM BN BK l e) * ys t (bLane BM BN BK l e)
```

**Assumptions / layout contracts:**
- `hBK : 0 < BK`
- `hcn : SCN = 1`
- `hBN : BN ≤ SCM`

**Closed-form spec defs (transitive):** `matmulTriton2IO`, `aLane`, `bLane`, `matmul_triton2_surface`, `pidMAt`, `pidNAt`, `cdiv`, `numPidM`, `numPidN`

<details><summary><code>matmulTriton2IO</code></summary>

```
/-- **Streaming IO signature** of `matmul_triton2` on the two-stream fold
skin (S1: fold + terminal store). Step `t` of the K-loop reads the
`[BM, BLOCK_K]` `A`-tile and the `[BLOCK_K, BN]` `B`-tile; after the loop one
`[BM, BN]` output tile is stored at the **`.real`** grid (`outDType` default
— the kernel's `tl.store(c_ptrs, accumulator, mask=c_mask)` is untyped, so
the terminal store has no quantization event). The kernel schedules on a
**single** linear `pid` (`program_id(0)`; the skin's `pid₁` slot is unused),
so every window derives `(pid_m, pid_n)` through the transcribed L2-grouping
arithmetic `pidMAt`/`pidNAt`:

* `read1` lane `l = (i, e)` (row-major over `[BM, BLOCK_K]`), step `t`:
  `(pid_m·BM + i)·stride_am + (t·BK + e)·stride_ak` — the invariant's
  `a_ptrs` cell after `t` advances of `BK·stride_ak`.
* `read2` lane `l = (e, j)` (row-major over `[BLOCK_K, BN]`), step `t`:
  `(t·BK + e)·stride_bk + (pid_n·BN + j)·stride_bn` — the `b_ptrs` cell.
* `write` lane `l = (i, j)`:
  `(pid_m·BM + i)·stride_cm + (pid_n·BN + j)·stride_cn` — the kernel's
  `c_ptrs` (= `cOffset` in pid form).
* `mask1`/`mask2` transcribe the loads' `offs_k < K − k·BK` windows in the
  per-lane spelling `t·BK + e < K`; `writeMask` transcribes the store's
  `(offs_am < M) & (offs_bn < N)` boundary mask verbatim. -/
```
```lean
def matmulTriton2IO (A B C : RegionName)
    (M N BM BN GM SAM SAK SBK SBN SCM SCN BK numKBlocks : Nat) :
    StreamMasked2DKernelIO₂ where
  kernel := matmul_triton2_surface A B C M N (BK * numKBlocks) SAM SAK SBK SBN SCM SCN
    BM BN BK GM
  inp1 := A
  inp2 := B
  out := C
  T := numKBlocks
  B1 := BM * BK
  B2 := BK * BN
  C := BM * BN
  read1 := fun p₀ _ t l =>
    (pidMAt p₀ M N BM BN GM * BM + l.val / BK) * SAM + (t.val * BK + l.val % BK) * SAK
  read2 := fun p₀ _ t l =>
    (t.val * BK + l.val / BN) * SBK + (pidNAt p₀ M N BM BN GM * BN + l.val % BN) * SBN
  write := fun p₀ _ l =>
    (pidMAt p₀ M N BM BN GM * BM + l.val / BN) * SCM
      + (pidNAt p₀ M N BM BN GM * BN + l.val % BN) * SCN
  mask1 := fun _ _ t l => t.val * BK + l.val % BK < BK * numKBlocks
  mask2 := fun _ _ t l => t.val * BK + l.val / BN < BK * numKBlocks
  writeMask := fun p₀ _ l =>
    pidMAt p₀ M N BM BN GM * BM + l.val / BN < M ∧
      pidNAt p₀ M N BM BN GM * BN + l.val % BN < N
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

<details><summary><code>matmul_triton2_surface</code></summary>

```
/-- Faithful transcription of `matmul_triton2.py`'s `matmul_kernel`. -/
```
```lean
def matmul_triton2_surface
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
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_bn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
  b_ptrs = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    accumulator += tl.dot(a, b)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  c_ptrs = C + offs_am[:, None] * $(stride_cm) + offs_bn[None, :] * $(stride_cn)
  c_mask = (offs_am[:, None] < $(M)) & (offs_bn[None, :] < $(N))
  tl.store(c_ptrs, accumulator, mask=c_mask)
}
```
</details>

<details><summary><code>pidMAt</code></summary>

```
/-- `pidM` as a function of the raw `program_id(0)` value: the skin's windows
are functions of `pid₀`, not of a `BlockState`. Definitionally `pidM`. -/
```
```lean
def pidMAt (p₀ M N BM BN GM : Nat) : Nat :=
  let nm := numPidM M BM
  let nn := numPidN N BN
  let nig := GM * nn
  let gid := p₀ / nig
  let fpm := gid * GM
  let gsm := if nm - fpm < GM then nm - fpm else GM
  fpm + p₀ % gsm
```
</details>

<details><summary><code>pidNAt</code></summary>

```
/-- `pidN` as a function of the raw `program_id(0)` value. -/
```
```lean
def pidNAt (p₀ M N BM BN GM : Nat) : Nat :=
  let nm := numPidM M BM
  let nn := numPidN N BN
  let nig := GM * nn
  let gid := p₀ / nig
  let fpm := gid * GM
  let gsm := if nm - fpm < GM then nm - fpm else GM
  p₀ % nig / gsm
```
</details>

<details><summary><code>cdiv</code></summary>

```
/-- Ceiling division `⌈a / b⌉`, matching Triton's `tl.cdiv`. -/
```
```lean
def cdiv (a b : Nat) : Nat := (a + b - 1) / b
```
</details>

<details><summary><code>numPidM</code></summary>

```
/-- `num_pid_m = cdiv M BLOCK_M`. -/
```
```lean
def numPidM (M BM : Nat) : Nat := cdiv M BM
```
</details>

<details><summary><code>numPidN</code></summary>

```
/-- `num_pid_n = cdiv N BLOCK_N`. -/
```
```lean
def numPidN (N BN : Nat) : Nat := cdiv N BN
```
</details>

## Also present (pinned special-case summaries)
- `matmul_triton2_closed_form_correct`
