# Spec sheet — `bench/tritonbench_g/matmul_triton_autotune/MatmulTritonAutotune.lean`

**Python source:** `bench/tritonbench_g/matmul_triton_autotune/matmul_triton_autotune.py`

## Public theorem: `matmul_autotune_closed_form_correct`

<details><summary>docstring</summary>

```
/-- **Closed-form correctness for `matmul_triton_autotune` (general statement).**

For arbitrary linear program id `pid`, tile dims `BM`/`BN`, K-block size `BLOCK_K`,
and K-block count `numKBlocks` (so the contracted dimension is
`K = BLOCK_K · numKBlocks`), every **active** output cell of the computed
`BM × BN` tile equals `fp16( act( Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j] ) )` —
the genuine matrix product (over ℝ) of the loaded `A`/`B` tiles, optionally passed
through `leaky_relu`, cast to float16 — **not** the kernel's own executed value;
inactive lanes are left untouched.

Layout: `A[i,k]` at `A + offs_am(i)·stride_am + k·stride_ak`, `B[k,j]` at
`B + k·stride_bk + offs_bn(j)·stride_bn`, `C[i,j]` at
`C + stride_cm·offs_cm(i) + stride_cn·offs_cn(j)`, with `pid_m`/`pid_n` derived by
the kernel's L2-grouping schedule, `offs_am(i) = (pid_m·BM + i) % M`,
`offs_bn(j) = (pid_n·BN + j) % N` (the row-major pointer arithmetic with index
wrap). Preconditions: output-offset injectivity and clean initial `undef`. -/
```
</details>

**Statement:**
```lean
specification matmul_autotune_closed_form_correct
    (A B C : RegionName) (s : BlockState)
    (M N BM BN GM sam sak sbk sbn scm scn BLOCK_K numKBlocks : Nat) (K : Nat)
    (hK : K = BLOCK_K * numKBlocks) (ACTIVATION : Bool)
    (hcn : scn = 1) (hbnle : BN ≤ scm)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := matmul_autotune_surface A B C M N K sam sak sbk sbn scm scn
        BM BN BLOCK_K GM numKBlocks ACTIVATION)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BM BN GM)
        (fun idx => (C, cOffset s M N BM BN GM scm scn idx)))
      (expected := fun idx : TileIndex [BM, BN] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (matmulSpec s A B M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks ACTIVATION idx.1 idx.2.1))))
```

**Assumptions / layout contracts:**
- `hK : K = BLOCK_K * numKBlocks`
- `hcn : scn = 1`
- `hbnle : BN ≤ scm`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `matmul_autotune_surface`, `active`, `cOffset`, `matmulSpec`, `rowGlobal`, `colGlobal`, `act`, `aElem`, `bElem`, `pidM`, `pidN`, `leakyReLU`, `rowIndex`, `colIndex`, `kernelMin`

<details><summary><code>matmul_autotune_surface</code></summary>

```
/-- Faithful transcription of `matmul_triton_autotune.py`'s `matmul_kernel`.

The contracted dimension is presented as `K = BLOCK_SIZE_K · numKBlocks` so the
loop bound `tl.cdiv(K, BLOCK_SIZE_K) = numKBlocks` is exact; it is supplied as the
antiquoted `numKBlocks`. All other surface structure — the L2-grouping schedule,
the per-block `offs_k < K - k·BLOCK_K` load masks, the fused
`tl.dot(a, b, accumulator)`, the optional `leaky_relu`, the `float16` cast, and
the `(row<M)&(col<N)`-masked store — is transcribed verbatim. The Python string
constexpr `ACTIVATION == "leaky_relu"` is the Lean `Bool` parameter `ACTIVATION`. -/
```
```lean
def matmul_autotune_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat)
    (ACTIVATION : Bool) :
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
  for kk in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  if ACTIVATION {
    accumulator = leaky_relu(accumulator)
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
(the kernel's `c_ptrs`, using the **un-wrapped** global `offs_cm`/`offs_cn`). -/
```
```lean
def cOffset (s0 : BlockState) (M N BM BN GM scm scn : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  scm * rowGlobal s0 M N BM BN GM idx.1 + scn * colGlobal s0 M N BM BN GM idx.2.1
```
</details>

<details><summary><code>matmulSpec</code></summary>

```
/-- **Genuine matmul+activation spec** (over ℝ):
`C[i,j] = act( Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j] )`. -/
```
```lean
noncomputable def matmulSpec (s : BlockState) (A B : RegionName)
    (M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks : Nat) (ACTIVATION : Bool)
    (i : Fin BM) (j : Fin BN) : ℝ :=
  act ACTIVATION
    (gemmSum (aElem s A M N BM BN GM sam sak i) (bElem s B M N BM BN GM sbk sbn j)
      (BLOCK_K * numKBlocks))
```
</details>

<details><summary><code>rowGlobal</code></summary>

```
/-- Global output row of tile lane `i`: `pid_m · BLOCK_M + i`, **before** the
`% M` wrap (the kernel's `offs_cm`). -/
```
```lean
def rowGlobal (s : BlockState) (M N BM BN GM : Nat) (i : Fin BM) : Nat :=
  pidM (s.pids 0) M N BM BN GM * BM + i.val
```
</details>

<details><summary><code>colGlobal</code></summary>

```
/-- Global output column of tile lane `j`: `pid_n · BLOCK_N + j`, before wrap. -/
```
```lean
def colGlobal (s : BlockState) (M N BM BN GM : Nat) (j : Fin BN) : Nat :=
  pidN (s.pids 0) M N BM BN GM * BN + j.val
```
</details>

<details><summary><code>act</code></summary>

```
/-- The applied activation: `leakyReLU` when `ACTIVATION`, else the identity. -/
```
```lean
noncomputable def act (ACTIVATION : Bool) (x : ℝ) : ℝ :=
  if ACTIVATION then leakyReLU x else x
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
/-- The kernel's L2-grouping derivation of `pid_m` from the linear `pid`. -/
```
```lean
def pidM (pid M N BM BN GM : Nat) : Nat :=
  let num_pid_m := cdiv M BM
  let num_pid_n := cdiv N BN
  let num_pid_in_group := GM * num_pid_n
  let group_id := pid / num_pid_in_group
  let first_pid_m := group_id * GM
  let group_size_m := kernelMin (num_pid_m - first_pid_m) GM
  first_pid_m + ((pid % num_pid_in_group) % group_size_m)
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

<details><summary><code>leakyReLU</code></summary>

```
/-- Real-valued leaky-ReLU activation (slope `0.01` below zero), matching the
kernel's `leaky_relu`. -/
```
```lean
noncomputable def leakyReLU (x : ℝ) : ℝ := if x ≥ 0 then x else 0.01 * x
```
</details>

<details><summary><code>rowIndex</code></summary>

```
/-- The `% M`-wrapped A-row index of tile lane `i` (the kernel's `offs_am`). -/
```
```lean
def rowIndex (s : BlockState) (M N BM BN GM : Nat) (i : Fin BM) : Nat :=
  rowGlobal s M N BM BN GM i % M
```
</details>

<details><summary><code>colIndex</code></summary>

```
/-- The `% N`-wrapped B-column index of tile lane `j` (the kernel's `offs_bn`). -/
```
```lean
def colIndex (s : BlockState) (M N BM BN GM : Nat) (j : Fin BN) : Nat :=
  colGlobal s M N BM BN GM j % N
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

## Public theorem: `matmul_autotune_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S1 fold genre).** For every
rounding model `R` and **both** values of the constexpr `ACTIVATION` gate,
the faithful `matmul_triton_autotune` surface implements, on its
`StreamMasked2DKernelIO₂` signature, the **ideal ℝ activated GEMM fold**
over the streamed tiles: output lane `l = (i, j)` holds
`act ACTIVATION (∑ t, ∑ e, A-tile[t](i,e) · B-tile[t](e,j))` — the spec `f`
is exact real arithmetic with the leaky-ReLU branch expressed through the
named `act` brancher (identity when the gate is off), and the single
boundary quantization is carried by the skin's readback contract
(`readMemAs .fp16` holds `fp16.ofReal (R.round .fp16 (f …))`), where the
kernel's two rounding events (the `.to(tl.float16)` cast and the fp16-typed
masked store) collapse to one `R.round .fp16` by the defining `round_idem`.

Layer map: the prologue, the whole K-loop (masked loads, `other=0.0`), and
the activation tail are cast-free, so under `execR R` they collapse verbatim
onto the exact stepper and the proven `preLoop` / `matmul_step` /
`forRange_inv` / `matmulActStmt_eval` stack above is reused unchanged; only
the 6-statement store tail is re-proved on the `R` side
(`matmul_autotune_postLoopR`).

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
`matmul_autotune_closed_form_correct` (`Realizes_without_Rounding`) above is
retained unchanged; this `⊨[R]` face strictly generalizes its content — at
`R := .triv` the readback contract degenerates to the exact fp16-cast cell
of the same activated GEMM value. Both faces are kept per the
rounding-as-default doctrine. -/
```
</details>

**Statement:**
```lean
specification matmul_autotune_io_correctness (R : RoundingModel)
    (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat) (ACTIVATION : Bool)
    (hK : K = BK * numKBlocks) (hcn : scn = 1) (hBN : BN ≤ scm) :
    matmulAutotuneIO A B C M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks ACTIVATION
      ⊨[R] fun _ _ xs ys l =>
        act ACTIVATION (∑ t : Fin numKBlocks, ∑ e : Fin BK,
          xs t (aLane BM BN BK l e) * ys t (bLane BM BN BK l e))
```

**Assumptions / layout contracts:**
- `hK : K = BK * numKBlocks`
- `hcn : scn = 1`
- `hBN : BN ≤ scm`

**Closed-form spec defs (transitive):** `matmulAutotuneIO`, `act`, `aLane`, `bLane`, `matmul_autotune_surface`, `pidM`, `pidN`, `leakyReLU`, `kernelMin`

<details><summary><code>matmulAutotuneIO</code></summary>

```
/-- **Streaming IO signature** of `matmul_triton_autotune` on the two-stream
fold skin (S1: fold + terminal store). Step `t` of the K-loop reads the
`[BM, BLOCK_K]` `A`-tile and the `[BLOCK_K, BN]` `B`-tile; after the loop
one `[BM, BN]` output tile is stored at the **fp16** grid
(`outDType := .fp16` — the kernel's `.to(tl.float16)` + fp16 store). The
kernel schedules on a **single** linear `pid` (`program_id(0)`; the skin's
`pid₁` slot is unused), so every window derives `(pid_m, pid_n)` through the
transcribed L2-grouping arithmetic `pidM`/`pidN`:

* `read1` lane `l = (i, e)` (row-major over `[BM, BLOCK_K]`), step `t`:
  `((pid_m·BM + i) % M)·sam + (t·BK + e)·sak` — the invariant's `a_ptrs`
  cell after `t` advances.
* `read2` lane `l = (e, j)` (row-major over `[BLOCK_K, BN]`), step `t`:
  `(t·BK + e)·sbk + ((pid_n·BN + j) % N)·sbn` — the `b_ptrs` cell.
* `write` lane `l = (i, j)`: `scm·(pid_m·BM + i) + scn·(pid_n·BN + j)` — the
  kernel's un-wrapped `c_ptrs` (= `cOffset` in pid form).
* `mask1`/`mask2` transcribe the loads' `offs_k < K - kk·BK` windows in the
  per-lane spelling `t·BK + e < K`; `writeMask` transcribes the store's
  `(row<M) & (col<N)` boundary mask verbatim.

The constexpr `ACTIVATION` gate is the `Bool` parameter, kept symbolic. -/
```
```lean
def matmulAutotuneIO (A B C : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM numKBlocks : Nat) (ACTIVATION : Bool) :
    StreamMasked2DKernelIO₂ where
  kernel := matmul_autotune_surface A B C M N K sam sak sbk sbn scm scn BM BN BK GM
    numKBlocks ACTIVATION
  inp1 := A
  inp2 := B
  out := C
  T := numKBlocks
  B1 := BM * BK
  B2 := BK * BN
  C := BM * BN
  outDType := .fp16
  read1 := fun p₀ _ t l =>
    ((pidM p₀ M N BM BN GM * BM + l.val / BK) % M) * sam + (t.val * BK + l.val % BK) * sak
  read2 := fun p₀ _ t l =>
    (t.val * BK + l.val / BN) * sbk + ((pidN p₀ M N BM BN GM * BN + l.val % BN) % N) * sbn
  write := fun p₀ _ l =>
    scm * (pidM p₀ M N BM BN GM * BM + l.val / BN) + scn * (pidN p₀ M N BM BN GM * BN + l.val % BN)
  mask1 := fun _ _ t l => t.val * BK + l.val % BK < K
  mask2 := fun _ _ t l => t.val * BK + l.val / BN < K
  writeMask := fun p₀ _ l =>
    pidM p₀ M N BM BN GM * BM + l.val / BN < M ∧ pidN p₀ M N BM BN GM * BN + l.val % BN < N
```
</details>

<details><summary><code>act</code></summary>

```
/-- The applied activation: `leakyReLU` when `ACTIVATION`, else the identity. -/
```
```lean
noncomputable def act (ACTIVATION : Bool) (x : ℝ) : ℝ :=
  if ACTIVATION then leakyReLU x else x
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

<details><summary><code>matmul_autotune_surface</code></summary>

```
/-- Faithful transcription of `matmul_triton_autotune.py`'s `matmul_kernel`.

The contracted dimension is presented as `K = BLOCK_SIZE_K · numKBlocks` so the
loop bound `tl.cdiv(K, BLOCK_SIZE_K) = numKBlocks` is exact; it is supplied as the
antiquoted `numKBlocks`. All other surface structure — the L2-grouping schedule,
the per-block `offs_k < K - k·BLOCK_K` load masks, the fused
`tl.dot(a, b, accumulator)`, the optional `leaky_relu`, the `float16` cast, and
the `(row<M)&(col<N)`-masked store — is transcribed verbatim. The Python string
constexpr `ACTIVATION == "leaky_relu"` is the Lean `Bool` parameter `ACTIVATION`. -/
```
```lean
def matmul_autotune_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat)
    (ACTIVATION : Bool) :
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
  for kk in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  if ACTIVATION {
    accumulator = leaky_relu(accumulator)
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

<details><summary><code>pidM</code></summary>

```
/-- The kernel's L2-grouping derivation of `pid_m` from the linear `pid`. -/
```
```lean
def pidM (pid M N BM BN GM : Nat) : Nat :=
  let num_pid_m := cdiv M BM
  let num_pid_n := cdiv N BN
  let num_pid_in_group := GM * num_pid_n
  let group_id := pid / num_pid_in_group
  let first_pid_m := group_id * GM
  let group_size_m := kernelMin (num_pid_m - first_pid_m) GM
  first_pid_m + ((pid % num_pid_in_group) % group_size_m)
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

<details><summary><code>leakyReLU</code></summary>

```
/-- Real-valued leaky-ReLU activation (slope `0.01` below zero), matching the
kernel's `leaky_relu`. -/
```
```lean
noncomputable def leakyReLU (x : ℝ) : ℝ := if x ≥ 0 then x else 0.01 * x
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
