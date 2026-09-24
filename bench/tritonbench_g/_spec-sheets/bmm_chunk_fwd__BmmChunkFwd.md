# Spec sheet — `bench/tritonbench_g/bmm_chunk_fwd/BmmChunkFwd.lean`

**Python source:** `bench/tritonbench_g/bmm_chunk_fwd/bmm_chunk_fwd.py`

## Public theorem: `bmm_chunk_fwd_output_summary_general`

<details><summary>docstring</summary>

```
/-- **`bmm_chunk_fwd` output summary (dimension-general).**

The genuine batched-matmul surface (1) lowers to the algorithm layer and (2)
realizes the per-program matrix product `Σ_{k < BK·numKBlocks} A[i,k]·B[k,j]`
over ℝ on every in-bounds output lane — the closed-form spec read from INPUT
memory (`outputCell`/`bmmSpec` via `bmm_chunk_fwd_closed_form_correct`), NOT the
kernel's own executed value.

Preconditions: `0 < BK`; all tile rows/cols in-bounds
(`PM·BM+i < chunk_size`, `PN·BN+j < chunk_size`), making the load/store masks
all-true; output-address injectivity; clean initial `undef`. `PB/PC/PH/PM/PN`
are the kernel's own derived program coordinates. -/
```
</details>

**Statement:**
```lean
specification bmm_chunk_fwd_output_summary_general
    (A B Out : RegionName) (s : BlockState)
    (chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK numKBlocks : Nat)
    (hBK : 0 < BK)
    (hInj : Function.Injective (outOffset (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
      chunk_size SOB SOC SOH SOM SON (pidM (s.pids 0) chunk_size BN) (pidN (s.pids 0) chunk_size BN) BM BN))
    (hmlt : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) chunk_size BN) BM i < chunk_size)
    (hnlt : ∀ j : Fin BN, colIndex (pidN (s.pids 0) chunk_size BN) BN j < chunk_size)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (bmm_matmul_surface A B Out chunk_size (BK * numKBlocks) ngroups
        SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := bmm_matmul_surface A B Out chunk_size (BK * numKBlocks) ngroups
        SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK)
      (initialState := s)
      (write := fun idx : TileIndex [BM, BN] =>
        some (Out, outOffset (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
          chunk_size SOB SOC SOH SOM SON (pidM (s.pids 0) chunk_size BN) (pidN (s.pids 0) chunk_size BN) BM BN idx))
      (expected := fun idx : TileIndex [BM, BN] =>
        outputCell s A B (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
          (pidM (s.pids 0) chunk_size BN) (pidN (s.pids 0) chunk_size BN)
          chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks idx)
```

**Assumptions / layout contracts:**
- `hBK : 0 < BK`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `outOffset`, `pidC`, `pidH`, `pidM`, `pidN`, `rowIndex`, `colIndex`, `bmm_matmul_surface`, `outputCell`, `numPidN`, `cdiv`, `bmmSpec`, `aElem`, `bElem`, `batchOff`

<details><summary><code>outOffset</code></summary>

```
/-- The kernel's output offset for tile lane `idx`. -/
```
```lean
def outOffset (PB PC PH chunk_size SOB SOC SOH SOM SON PM PN BM BN : Nat)
    (idx : TileIndex [BM, BN]) : Nat :=
  PB * SOB + PC * SOC + PH * SOH + SOM * rowIndex PM BM idx.1 + SON * colIndex PN BN idx.2.1
```
</details>

<details><summary><code>pidC</code></summary>

```
/-- `pid_c = pid_ch // ngroups` (program axis 2 split by `ngroups`). -/
```
```lean
def pidC (PCH ngroups : Nat) : Nat := PCH / ngroups
```
</details>

<details><summary><code>pidH</code></summary>

```
/-- `pid_h = pid_ch - pid_c · ngroups`. -/
```
```lean
def pidH (PCH ngroups : Nat) : Nat := PCH - PCH / ngroups * ngroups
```
</details>

<details><summary><code>pidM</code></summary>

```
/-- `pid_m = program_id(0) // num_pid_n`. -/
```
```lean
def pidM (P0 chunk_size BN : Nat) : Nat := P0 / numPidN chunk_size BN
```
</details>

<details><summary><code>pidN</code></summary>

```
/-- `pid_n = program_id(0) % num_pid_n`. -/
```
```lean
def pidN (P0 chunk_size BN : Nat) : Nat := P0 % numPidN chunk_size BN
```
</details>

<details><summary><code>rowIndex</code></summary>

```
/-- Global chunk row of tile lane `i`: `PM · BM + i`. -/
```
```lean
def rowIndex (PM BM : Nat) (i : Fin BM) : Nat := PM * BM + i.val
```
</details>

<details><summary><code>colIndex</code></summary>

```
/-- Global chunk col of tile lane `j`: `PN · BN + j`. -/
```
```lean
def colIndex (PN BN : Nat) (j : Fin BN) : Nat := PN * BN + j.val
```
</details>

<details><summary><code>bmm_matmul_surface</code></summary>

```
/-- The genuine batched-matmul surface: chunked `acc += tl.dot(a, b)` with the
kernel's batch/chunk/head pointer offsets, K-block dot loop, per-block K-tail and
`chunk_size` row/col load masks, and the masked output store. -/
```
```lean
def bmm_matmul_surface
    (a_ptr b_ptr out_ptr : RegionName)
    (chunk_size K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_b_batch stride_b_seqlen stride_b_head stride_bk
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=1)
  pid_ch = tl.program_id(axis=2)
  pid_c = pid_ch // $(ngroups)
  pid_h = pid_ch - pid_c * $(ngroups)
  num_pid_n = tl.cdiv($(chunk_size), $(BLOCK_SIZE_N))
  pid_m = tl.program_id(axis=0) // num_pid_n
  pid_n = tl.program_id(axis=0) % num_pid_n
  a_ptr += pid_b * $(stride_a_batch) +
    pid_c * $(chunk_size) * $(stride_a_seqlen) + pid_h * $(stride_a_head)
  b_ptr += pid_b * $(stride_b_batch) +
    pid_c * $(chunk_size) * $(stride_b_seqlen) + pid_h * $(stride_b_head)
  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = a_ptr + offs_m[:, None] * $(stride_a_seqlen) + offs_k[None, :] * $(stride_ak)
  b_ptrs = b_ptr + offs_k[:, None] * $(stride_bk) + offs_n[None, :] * $(stride_b_seqlen)
  acc = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
    a = tl.load(a_ptrs, mask=(offs_m[:, None] < $(chunk_size)) &
      (offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K)), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K)) &
      (offs_n[None, :] < $(chunk_size)), other=0.0)
    acc += tl.dot(a, b)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  out = (acc).to(out_ptr.dtype.element_ty)
  out_ptr += pid_b * $(stride_out_batch) +
    pid_c * $(stride_out_chunk) + pid_h * $(stride_out_head)
  out_ptrs = out_ptr + $(stride_outm) * offs_m[:, None] + offs_n[None, :] * $(stride_outn)
  tl.store(out_ptrs, out, mask=(offs_m[:, None] < $(chunk_size)) &
    (offs_n[None, :] < $(chunk_size)))
}
```
</details>

<details><summary><code>outputCell</code></summary>

```
/-- The genuine output cell `Σ_k A·B` as a real `MemCell`. -/
```
```lean
noncomputable def outputCell (s0 : BlockState) (A B : RegionName)
    (PB PC PH PM PN chunk_size BM BN SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks : Nat)
    (idx : TileIndex [BM, BN]) : MemCell :=
  MemCell.of .real (FloatDType.real.ofReal (FloatDType.real.storeValue
    (some (bmmSpec s0 A B PB PC PH PM PN BM BN chunk_size SAB SAS SAH SAK SBB SBS SBH SBK BK numKBlocks idx.1 idx.2.1))))
```
</details>

<details><summary><code>numPidN</code></summary>

```
/-- The kernel's `num_pid_n = cdiv(chunk_size, BLOCK_SIZE_N)`. -/
```
```lean
def numPidN (chunk_size BN : Nat) : Nat := cdiv chunk_size BN
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

<details><summary><code>bmmSpec</code></summary>

```
/-- **Genuine batched-matmul spec**:
`Out[i,j] = Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j]`. -/
```
```lean
noncomputable def bmmSpec (s : BlockState) (A B : RegionName)
    (PB PC PH PM PN BM BN chunk_size SAB SAS SAH SAK SBB SBS SBH SBK
      BLOCK_K numKBlocks : Nat) (i : Fin BM) (j : Fin BN) : ℝ :=
  (Finset.range (BLOCK_K * numKBlocks)).sum
    (fun k => aElem s A PB PC PH PM BM chunk_size SAB SAS SAH SAK i k
      * bElem s B PB PC PH PN BN chunk_size SBB SBS SBH SBK j k)
```
</details>

<details><summary><code>aElem</code></summary>

```
/-- `A[i, k] = readMem A (batchOff_a + (PM·BM+i)·stride_a_seqlen + k·stride_ak)`. -/
```
```lean
noncomputable def aElem (s : BlockState) (A : RegionName)
    (PB PC PH PM BM chunk_size SAB SAS SAH SAK : Nat) (i : Fin BM) (k : Nat) : ℝ :=
  s.readMem A (batchOff PB PC PH chunk_size SAB SAS SAH + rowIndex PM BM i * SAS + k * SAK)
```
</details>

<details><summary><code>bElem</code></summary>

```
/-- `B[k, j] = readMem B (batchOff_b + k·stride_bk + (PN·BN+j)·stride_b_seqlen)`. -/
```
```lean
noncomputable def bElem (s : BlockState) (B : RegionName)
    (PB PC PH PN BN chunk_size SBB SBS SBH SBK : Nat) (j : Fin BN) (k : Nat) : ℝ :=
  s.readMem B (batchOff PB PC PH chunk_size SBB SBS SBH + k * SBK + colIndex PN BN j * SBS)
```
</details>

<details><summary><code>batchOff</code></summary>

```
/-- The kernel's `a`/`b` batch+chunk+head base offset:
`pid_b·stride_batch + pid_c·chunk_size·stride_seqlen + pid_h·stride_head`. -/
```
```lean
def batchOff (PB PC PH chunk_size stride_batch stride_seqlen stride_head : Nat) : Nat :=
  PB * stride_batch + PC * chunk_size * stride_seqlen + PH * stride_head
```
</details>

## Public theorem: `bmm_chunk_fwd_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S1 fold genre).**

For **every** rounding model `R` — no rounding-boundary hypothesis is needed, the
surface is cast-free — the faithful genuine batched-matmul surface of
`bmm_chunk_fwd` implements, on its `StreamMetaMasked3DKernelIO₂` signature at
`nMeta := 0`, the **ideal ℝ batched matrix product** over the streamed tiles:
every output lane `l = (i, n)` holds

`∑_{t < numKBlocks} ∑_{e < BK} aTile(t)[i, e] · bTile(t)[e, n]`

— exact real arithmetic, the `K = BK · numKBlocks`-long contraction split
row-major into the `numKBlocks` streamed K-blocks. This is `bmmSpec` re-expressed
on the skin's per-step tiles (`bmmSpec_eq_streamSum`).

Layer map: the prologue, the K-loop and the tail are all cast-free (the surface
has no `Op.castFloat`), so under `execR R` they collapse verbatim onto the exact
stepper and the proven `bmm_preLoop` / `bmm_step` / `bmm_loop` invariant tower
above is reused unchanged; only the masked terminal store is re-proved on the `R`
side (`bmm_postLoopR`). The store is `.real`-typed, so the skin's boundary
quantization degenerates: the readback's `R.round .real` is the identity
(`round_real_apply`).

Hypotheses, each inherited from the exact headline
`bmm_chunk_fwd_output_summary_general` (see its docstring), restated in pid form
because the skin quantifies the launch state internally:

* `hBK : 0 < BK` — the K-loop's block size; at `0` the `cdiv K BK` trip count and
  the block index are meaningless. Exact headline's `hBK`.
* `hmlt` / `hnlt` — every tile row / column of every program is `< chunk_size`.
  This is what makes the kernel's own per-block K-tail and `chunk_size` row/col
  load masks and its store mask all-true, i.e. what makes the skin's full-window
  `mask1` / `mask2` / `writeMask` the honest read/write sets. Exact headline's
  `hmlt` / `hnlt`, with `s.pids 0` generalized to `∀ pid₀`.
* `hOutInj` — output-address injectivity of the write window. With colliding
  output lanes the per-lane readback would be last-writer-wins and the statement
  false. As in the exact headline it is carried as an open side condition, not
  discharged.

The exact headline's `hundef` is **not** a hypothesis here: the skin's triple
already pins `s₀.undef = fun _ _ => 0`.

Relation to the exact surface: `bmm_chunk_fwd_output_summary_general`
(`Realizes_without_Rounding`) above is retained unchanged; this `⊨[R]` face
restates the same batched matmul on the streaming skin, for every `R` at once (at
the `.real` grid the two faces carry the same exact cell). Both are kept per the
rounding-as-default doctrine. -/
```
</details>

**Statement:**
```lean
specification bmm_chunk_fwd_io_correctness (R : RoundingModel) (A B Out : RegionName)
    (chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON
      BM BN BK numKBlocks : Nat)
    (hBK : 0 < BK)
    (hmlt : ∀ (pid₀ : Nat) (i : Fin BM), pidM pid₀ chunk_size BN * BM + i.val < chunk_size)
    (hnlt : ∀ (pid₀ : Nat) (j : Fin BN), pidN pid₀ chunk_size BN * BN + j.val < chunk_size)
    (hOutInj : ∀ pid₀ pid₁ pid₂ : Nat,
      Function.Injective (fun idx : TileIndex [BM, BN] =>
        bmmOutAddr pid₀ pid₁ pid₂ ngroups chunk_size BM BN SOB SOC SOH SOM SON
          idx.1.val idx.2.1.val)) :
    bmm_chunk_fwd_IO A B Out chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK
        SOB SOC SOH SOM SON BM BN BK numKBlocks ⊨[R]
      fun _ _ _ _ xs ys l => bmmStreamSum BM BN BK numKBlocks xs ys l
```

**Assumptions / layout contracts:**
- `hBK : 0 < BK`
- `hmlt : ∀ (pid₀ : Nat) (i : Fin BM), pidM pid₀ chunk_size BN * BM + i.val < chunk_size`
- `hnlt : ∀ (pid₀ : Nat) (j : Fin BN), pidN pid₀ chunk_size BN * BN + j.val < chunk_size`
- `hOutInj : ∀ pid₀ pid₁ pid₂ : Nat,
      Function.Injective (fun idx : TileIndex [BM, BN] =>
        bmmOutAddr pid₀ pid₁ pid₂ ngroups chunk_size BM BN SOB SOC SOH SOM SON
          idx.1.val idx.2.1.val)`

**Closed-form spec defs (transitive):** `pidM`, `pidN`, `bmmOutAddr`, `bmm_chunk_fwd_IO`, `bmmStreamSum`, `numPidN`, `pidC`, `pidH`, `bmm_matmul_surface`, `bmmAAddr`, `bmmBAddr`, `cdiv`, `batchOff`

<details><summary><code>pidM</code></summary>

```
/-- `pid_m = program_id(0) // num_pid_n`. -/
```
```lean
def pidM (P0 chunk_size BN : Nat) : Nat := P0 / numPidN chunk_size BN
```
</details>

<details><summary><code>pidN</code></summary>

```
/-- `pid_n = program_id(0) % num_pid_n`. -/
```
```lean
def pidN (P0 chunk_size BN : Nat) : Nat := P0 % numPidN chunk_size BN
```
</details>

<details><summary><code>bmmOutAddr</code></summary>

```
/-- Lane `(i, j)`'s terminal `Out` store address — the pid form of `outOffset`. -/
```
```lean
private def bmmOutAddr (pid₀ pid₁ pid₂ ngroups chunk_size BM BN
    SOB SOC SOH SOM SON i j : Nat) : Nat :=
  pid₁ * SOB + pidC pid₂ ngroups * SOC + pidH pid₂ ngroups * SOH
    + SOM * (pidM pid₀ chunk_size BN * BM + i) + SON * (pidN pid₀ chunk_size BN * BN + j)
```
</details>

<details><summary><code>bmm_chunk_fwd_IO</code></summary>

```
/-- **Streaming IO signature** of `bmm_chunk_fwd`'s genuine batched-matmul
surface on the metadata-parametrized two-stream fold skin (S1: fold + terminal
masked store, 3-D pid grid), at `nMeta := 0` — this kernel loads **no** scalar
metadata, so the slot vector is empty (`sty` / `mbuf` / `mwin` are `Fin.elim0`,
and `List.ofFn io.mbuf ++ [A, B, Out]` collapses to `[A, B, Out]`) and every
window is a function of the three pids alone. The 3-D grid is the kernel's own:
`pid₀` = flattened `(chunk-row, chunk-col)` tile, `pid₁` = batch, `pid₂` =
chunk·group.

Per step the kernel reads a `[BM, BK]` tile of `A` and a `[BK, BN]` tile of `B`
(both masked, with `other = 0.0`); after the `numKBlocks`-step fold one
`[BM, BN]` output tile is masked-stored at the **`.real`** grid (`outDType`
default — the Python `acc.to(out_ptr.dtype.element_ty)` transcribes to the plain
`out = acc` register copy, so the `tl.store` is a `Stmt.store .real` and carries
no quantization event).

* `read1` lane `j = (i, e)` (row-major over `[BM, BK]`): `bmmAAddr`, the `a_ptrs`
  cell at K-block `t`.
* `read2` lane `j = (e, n)` (row-major over `[BK, BN]`): `bmmBAddr`.
* `write` lane `j = (i, n)` (row-major over `[BM, BN]`): `bmmOutAddr`.
* `mask1` / `mask2` / `writeMask`: the full windows — under the headline's
  in-bounds pins the kernel's own row/col and K-tail masks are all-true, so every
  lane really is read / written (see the section docstring). -/
```
```lean
def bmm_chunk_fwd_IO (A B Out : RegionName)
    (chunk_size ngroups SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON
      BM BN BK numKBlocks : Nat) :
    StreamMetaMasked3DKernelIO₂ where
  kernel := bmm_matmul_surface A B Out chunk_size (BK * numKBlocks) ngroups
    SAB SAS SAH SAK SBB SBS SBH SBK SOB SOC SOH SOM SON BM BN BK
  inp1 := A
  inp2 := B
  out := Out
  nMeta := 0
  sty := Fin.elim0
  mbuf := Fin.elim0
  mwin := Fin.elim0
  T := numKBlocks
  B1 := BM * BK
  B2 := BK * BN
  C := BM * BN
  outDType := .real
  read1 := fun pid₀ pid₁ pid₂ _ t j =>
    bmmAAddr pid₀ pid₁ pid₂ ngroups chunk_size BM BN BK SAB SAS SAH SAK
      t.val (j.val / BK) (j.val % BK)
  read2 := fun pid₀ pid₁ pid₂ _ t j =>
    bmmBAddr pid₀ pid₁ pid₂ ngroups chunk_size BN BK SBB SBS SBH SBK
      t.val (j.val / BN) (j.val % BN)
  write := fun pid₀ pid₁ pid₂ _ j =>
    bmmOutAddr pid₀ pid₁ pid₂ ngroups chunk_size BM BN SOB SOC SOH SOM SON
      (j.val / BN) (j.val % BN)
  mask1 := fun _ _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ _ => True
  writeMask := fun _ _ _ _ _ => True
```
</details>

<details><summary><code>bmmStreamSum</code></summary>

```
/-- **The streamed batched-matmul spec**: the ideal ℝ value of output lane
`l = (i, j)` as a single `numKBlocks`-step fold over the streamed tiles,

`∑_{t < numKBlocks} ∑_{e < BK} aTile(t)[i, e] · bTile(t)[e, j]`.

This is the `⊨[R]` face of `bmmSpec`: the same `∑_{k < K} A[i,k]·B[k,j]`
batched-matmul reference, re-expressed on the skin's per-step tiles instead of on
`s0`-indexed `readMem`s (`bmmSpec_eq_streamSum`). -/
```
```lean
noncomputable def bmmStreamSum (BM BN BK numKBlocks : Nat)
    (xs : Fin numKBlocks → Fin (BM * BK) → ℝ)
    (ys : Fin numKBlocks → Fin (BK * BN) → ℝ)
    (l : Fin (BM * BN)) : ℝ :=
  ∑ t : Fin numKBlocks, ∑ e : Fin BK,
    xs t (Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit))
      * ys t (Lane2D.encode (e, (Lane2D.decode l).2.1, PUnit.unit))
```
</details>

<details><summary><code>numPidN</code></summary>

```
/-- The kernel's `num_pid_n = cdiv(chunk_size, BLOCK_SIZE_N)`. -/
```
```lean
def numPidN (chunk_size BN : Nat) : Nat := cdiv chunk_size BN
```
</details>

<details><summary><code>pidC</code></summary>

```
/-- `pid_c = pid_ch // ngroups` (program axis 2 split by `ngroups`). -/
```
```lean
def pidC (PCH ngroups : Nat) : Nat := PCH / ngroups
```
</details>

<details><summary><code>pidH</code></summary>

```
/-- `pid_h = pid_ch - pid_c · ngroups`. -/
```
```lean
def pidH (PCH ngroups : Nat) : Nat := PCH - PCH / ngroups * ngroups
```
</details>

<details><summary><code>bmm_matmul_surface</code></summary>

```
/-- The genuine batched-matmul surface: chunked `acc += tl.dot(a, b)` with the
kernel's batch/chunk/head pointer offsets, K-block dot loop, per-block K-tail and
`chunk_size` row/col load masks, and the masked output store. -/
```
```lean
def bmm_matmul_surface
    (a_ptr b_ptr out_ptr : RegionName)
    (chunk_size K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_b_batch stride_b_seqlen stride_b_head stride_bk
      stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=1)
  pid_ch = tl.program_id(axis=2)
  pid_c = pid_ch // $(ngroups)
  pid_h = pid_ch - pid_c * $(ngroups)
  num_pid_n = tl.cdiv($(chunk_size), $(BLOCK_SIZE_N))
  pid_m = tl.program_id(axis=0) // num_pid_n
  pid_n = tl.program_id(axis=0) % num_pid_n
  a_ptr += pid_b * $(stride_a_batch) +
    pid_c * $(chunk_size) * $(stride_a_seqlen) + pid_h * $(stride_a_head)
  b_ptr += pid_b * $(stride_b_batch) +
    pid_c * $(chunk_size) * $(stride_b_seqlen) + pid_h * $(stride_b_head)
  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = a_ptr + offs_m[:, None] * $(stride_a_seqlen) + offs_k[None, :] * $(stride_ak)
  b_ptrs = b_ptr + offs_k[:, None] * $(stride_bk) + offs_n[None, :] * $(stride_b_seqlen)
  acc = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
    a = tl.load(a_ptrs, mask=(offs_m[:, None] < $(chunk_size)) &
      (offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K)), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K)) &
      (offs_n[None, :] < $(chunk_size)), other=0.0)
    acc += tl.dot(a, b)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  out = (acc).to(out_ptr.dtype.element_ty)
  out_ptr += pid_b * $(stride_out_batch) +
    pid_c * $(stride_out_chunk) + pid_h * $(stride_out_head)
  out_ptrs = out_ptr + $(stride_outm) * offs_m[:, None] + offs_n[None, :] * $(stride_outn)
  tl.store(out_ptrs, out, mask=(offs_m[:, None] < $(chunk_size)) &
    (offs_n[None, :] < $(chunk_size)))
}
```
</details>

<details><summary><code>bmmAAddr</code></summary>

```
/-- Step `t`, lane `(i, e)`'s `A` read address: the kernel's `a_ptrs` cell at
K-block `t`, i.e. `aElem`'s address at contracted index `k = t·BK + e`. -/
```
```lean
private def bmmAAddr (pid₀ pid₁ pid₂ ngroups chunk_size BM BN BK
    SAB SAS SAH SAK t i e : Nat) : Nat :=
  batchOff pid₁ (pidC pid₂ ngroups) (pidH pid₂ ngroups) chunk_size SAB SAS SAH
    + (pidM pid₀ chunk_size BN * BM + i) * SAS + (t * BK + e) * SAK
```
</details>

<details><summary><code>bmmBAddr</code></summary>

```
/-- Step `t`, lane `(e, j)`'s `B` read address: the kernel's `b_ptrs` cell at
K-block `t`, i.e. `bElem`'s address at contracted index `k = t·BK + e`. -/
```
```lean
private def bmmBAddr (pid₀ pid₁ pid₂ ngroups chunk_size BN BK
    SBB SBS SBH SBK t e j : Nat) : Nat :=
  batchOff pid₁ (pidC pid₂ ngroups) (pidH pid₂ ngroups) chunk_size SBB SBS SBH
    + (t * BK + e) * SBK + (pidN pid₀ chunk_size BN * BN + j) * SBS
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

<details><summary><code>batchOff</code></summary>

```
/-- The kernel's `a`/`b` batch+chunk+head base offset:
`pid_b·stride_batch + pid_c·chunk_size·stride_seqlen + pid_h·stride_head`. -/
```
```lean
def batchOff (PB PC PH chunk_size stride_batch stride_seqlen stride_head : Nat) : Nat :=
  PB * stride_batch + PC * chunk_size * stride_seqlen + PH * stride_head
```
</details>

## Also present (pinned special-case summaries)
- `bmm_chunk_fwd_closed_form_correct`
