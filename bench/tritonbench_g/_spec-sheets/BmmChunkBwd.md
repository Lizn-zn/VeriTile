# Spec sheet — `bench/tritonbench_g/bmm_chunk_bwd/BmmChunkBwd.lean`

**Python source:** `bench/tritonbench_g/bmm_chunk_bwd/bmm_chunk_bwd.py`

## Public theorem: `bmm_chunk_bwd_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general output summary** for `bmm_chunk_bwd` (batched
matmul-backward). For arbitrary `chunk_size`, `K`, `ngroups`, tile dims `BM`/`BN`,
strides, CS-block size `BCS` and count `numCSBlocks` (contracted bound
`CSL = BCS·numCSBlocks`): the genuine surface lowers to the algorithm layer and
realizes the per-program matrix-product backward
`Σ_{cs<BCS·numCSBlocks} Dout[i,cs]·A[cs,j]` (read from input memory) on every
in-bounds output lane. Preconditions: `0 < BCS`; tile rows/cols in-bounds
(`PM·BM+i < CSL` and `< chunk_size`, `PN·BN+j < K`); output-address injectivity;
clean initial `undef`. -/
```
</details>

**Statement:**
```lean
specification bmm_chunk_bwd_output_summary_general
    (A Dout Db : RegionName) (s : BlockState)
    (chunk_size ngroups SAB SAS SAH SAK SDB SDC SDH SDM SDN SOB SOS SOH SOK BM BN BCS numCSBlocks K : Nat) (hBCS : 0 < BCS)
    (hInj : Function.Injective (dbOffset (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
      chunk_size SOB SOS SOH SOK (pidM (s.pids 0) K BN) (pidN (s.pids 0) K BN) BM BN))
    (hmlt : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) K BN) BM i < BCS * numCSBlocks)
    (hmlt' : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) K BN) BM i < chunk_size)
    (hnlt : ∀ j : Fin BN, colIndex (pidN (s.pids 0) K BN) BN j < K)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (bbwd_matmul_surface A Dout Db chunk_size (BCS * numCSBlocks) K ngroups
        SAB SAS SAH SAK SDB SDC SDH SDM SDN SOB SOS SOH SOK BM BN BCS).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := bbwd_matmul_surface A Dout Db chunk_size (BCS * numCSBlocks) K ngroups
        SAB SAS SAH SAK SDB SDC SDH SDM SDN SOB SOS SOH SOK BM BN BCS)
      (initialState := s)
      (write := fun idx : TileIndex [BM, BN] =>
        some (Db, dbOffset (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
          chunk_size SOB SOS SOH SOK (pidM (s.pids 0) K BN) (pidN (s.pids 0) K BN) BM BN idx))
      (expected := fun idx : TileIndex [BM, BN] =>
        dbCell s Dout A (s.pids 1) (pidC (s.pids 2) ngroups) (pidH (s.pids 2) ngroups)
          (pidM (s.pids 0) K BN) (pidN (s.pids 0) K BN)
          chunk_size BM BN SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS numCSBlocks idx)
```

**Assumptions / layout contracts:**
- `hBCS : 0 < BCS`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `dbOffset`, `pidC`, `pidH`, `pidM`, `pidN`, `rowIndex`, `colIndex`, `bbwd_matmul_surface`, `dbCell`, `numPidN`, `cdiv`, `bbwdSpec`, `doutElem`, `aElem`, `doutOff`, `batchOff`

<details><summary><code>dbOffset</code></summary>

```
/-- The kernel's output offset for tile lane `idx`. -/
```
```lean
def dbOffset (PB PC PH chunk_size SOB SOS SOH SOK PM PN BM BN : Nat)
    (idx : TileIndex [BM, BN]) : Nat :=
  PB * SOB + PC * chunk_size * SOS + PH * SOH + SOS * rowIndex PM BM idx.1 + colIndex PN BN idx.2.1 * SOK
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
def pidM (P0 K BN : Nat) : Nat := P0 / numPidN K BN
```
</details>

<details><summary><code>pidN</code></summary>

```
/-- `pid_n = program_id(0) % num_pid_n`. -/
```
```lean
def pidN (P0 K BN : Nat) : Nat := P0 % numPidN K BN
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
/-- Global output col of tile lane `j`: `PN · BN + j`. -/
```
```lean
def colIndex (PN BN : Nat) (j : Fin BN) : Nat := PN * BN + j.val
```
</details>

<details><summary><code>bbwd_matmul_surface</code></summary>

```
/-- The genuine batched-matmul-backward surface: chunked `acc += tl.dot(dout, a)`
with the kernel's batch/chunk/head pointer offsets, CS-block dot loop, per-block
CS-tail and `chunk_size`/`K` row/col load masks, and the masked output store. -/
```
```lean
def bbwd_matmul_surface
    (A Dout Db : RegionName)
    (chunk_size CSL K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_dout_batch stride_dout_chunk stride_dout_head
      stride_dout_csize_m stride_dout_csize_n
      stride_db_batch stride_db_seqlen stride_db_head stride_db_k
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=1)
  pid_ch = tl.program_id(axis=2)
  pid_c = pid_ch // $(ngroups)
  pid_h = pid_ch - pid_c * $(ngroups)
  num_pid_n = tl.cdiv($(K), $(BLOCK_SIZE_N))
  pid_m = tl.program_id(axis=0) // num_pid_n
  pid_n = tl.program_id(axis=0) % num_pid_n
  Dout += pid_b * $(stride_dout_batch) +
    pid_c * $(stride_dout_chunk) + pid_h * $(stride_dout_head)
  A += pid_b * $(stride_a_batch) +
    pid_c * $(chunk_size) * $(stride_a_seqlen) + pid_h * $(stride_a_head)
  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  offs_cs = tl.arange(0, $(BLOCK_SIZE_CS))
  dout_ptrs = Dout + offs_m[:, None] * $(stride_dout_csize_n) + offs_cs[None, :] * $(stride_dout_csize_m)
  a_ptrs = A + offs_cs[:, None] * $(stride_a_seqlen) + offs_n[None, :] * $(stride_ak)
  acc = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for cs in range($(0), tl.cdiv($(CSL), $(BLOCK_SIZE_CS)), $(1)) {
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < $(chunk_size)) &
      (offs_cs[None, :] < $(CSL) - cs * $(BLOCK_SIZE_CS)), other=0.0)
    a = tl.load(a_ptrs, mask=(offs_cs[:, None] < $(CSL) - cs * $(BLOCK_SIZE_CS)) &
      (offs_n[None, :] < $(K)), other=0.0)
    acc += tl.dot(dout, a)
    dout_ptrs += $(BLOCK_SIZE_CS) * $(stride_dout_csize_m)
    a_ptrs += $(BLOCK_SIZE_CS) * $(stride_a_seqlen)
  }
  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  db = (acc).to(Db.dtype.element_ty)
  Db += pid_b * $(stride_db_batch) + pid_c * $(chunk_size) * $(stride_db_seqlen) + pid_h * $(stride_db_head)
  db_ptrs = Db + $(stride_db_seqlen) * offs_m[:, None] + offs_n[None, :] * $(stride_db_k)
  tl.store(db_ptrs, db, mask=(offs_m[:, None] < $(CSL)) &
    (offs_n[None, :] < $(K)))
}
```
</details>

<details><summary><code>dbCell</code></summary>

```
/-- The genuine output cell `Σ_cs Dout·A` as a real `MemCell`. -/
```
```lean
noncomputable def dbCell (s0 : BlockState) (Dout A : RegionName)
    (PB PC PH PM PN chunk_size BM BN SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS numCSBlocks : Nat)
    (idx : TileIndex [BM, BN]) : MemCell :=
  MemCell.of .real (FloatDType.real.ofReal (FloatDType.real.storeValue
    (some (bbwdSpec s0 Dout A PB PC PH PM PN BM BN chunk_size SDB SDC SDH SDN SDM SAB SAS SAH SAK BCS numCSBlocks idx.1 idx.2.1))))
```
</details>

<details><summary><code>numPidN</code></summary>

```
/-- The kernel's `num_pid_n = cdiv(K, BLOCK_SIZE_N)`. -/
```
```lean
def numPidN (K BN : Nat) : Nat := cdiv K BN
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

<details><summary><code>bbwdSpec</code></summary>

```
/-- **Genuine batched-matmul-backward spec**:
`Db[i,j] = Σ_{cs < BLOCK_CS·numCSBlocks} Dout[i,cs] · A[cs,j]`. -/
```
```lean
noncomputable def bbwdSpec (s : BlockState) (Dout A : RegionName)
    (PB PC PH PM PN BM BN chunk_size SDB SDC SDH SDN SDM SAB SAS SAH SAK
      BLOCK_CS numCSBlocks : Nat) (i : Fin BM) (j : Fin BN) : ℝ :=
  (Finset.range (BLOCK_CS * numCSBlocks)).sum
    (fun cs => doutElem s Dout PB PC PH PM BM SDB SDC SDH SDN SDM i cs
      * aElem s A PB PC PH PN BN chunk_size SAB SAS SAH SAK j cs)
```
</details>

<details><summary><code>doutElem</code></summary>

```
/-- `Dout[i, cs] = readMem Dout (doutOff + (PM·BM+i)·stride_dout_csize_n + cs·stride_dout_csize_m)`. -/
```
```lean
noncomputable def doutElem (s : BlockState) (Dout : RegionName)
    (PB PC PH PM BM SDB SDC SDH SDN SDM : Nat) (i : Fin BM) (cs : Nat) : ℝ :=
  s.readMem Dout (doutOff PB PC PH SDB SDC SDH + rowIndex PM BM i * SDN + cs * SDM)
```
</details>

<details><summary><code>aElem</code></summary>

```
/-- `A[cs, j] = readMem A (batchOff_a + cs·stride_a_seqlen + (PN·BN+j)·stride_ak)`. -/
```
```lean
noncomputable def aElem (s : BlockState) (A : RegionName)
    (PB PC PH PN BN chunk_size SAB SAS SAH SAK : Nat) (j : Fin BN) (cs : Nat) : ℝ :=
  s.readMem A (batchOff PB PC PH chunk_size SAB SAS SAH + cs * SAS + colIndex PN BN j * SAK)
```
</details>

<details><summary><code>doutOff</code></summary>

```
/-- The kernel's `dout` batch+chunk+head base offset (no `chunk_size` multiply):
`pid_b·stride_dout_batch + pid_c·stride_dout_chunk + pid_h·stride_dout_head`. -/
```
```lean
def doutOff (PB PC PH SDB SDC SDH : Nat) : Nat :=
  PB * SDB + PC * SDC + PH * SDH
```
</details>

<details><summary><code>batchOff</code></summary>

```
/-- The kernel's `a`/`db` batch+chunk+head base offset (with `chunk_size`):
`pid_b·stride_batch + pid_c·chunk_size·stride_seqlen + pid_h·stride_head`. -/
```
```lean
def batchOff (PB PC PH chunk_size stride_batch stride_seqlen stride_head : Nat) : Nat :=
  PB * stride_batch + PC * chunk_size * stride_seqlen + PH * stride_head
```
</details>

## Public theorem: `bmm_chunk_bwd_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S1 fold genre, 3-D grid).**

For every rounding model `R`, the faithful `HAS_RESIDUAL = false` surface of
`bmm_chunk_bwd` implements, on its `StreamMetaMasked3DKernelIO₂` signature at
`nMeta := 0`, the **ideal ℝ batched-matmul-backward fold** over the streamed
tiles: every write-active output lane `l = (i, n)` holds

`∑_{t < numCSBlocks} ∑_{e < BLOCK_CS} maskedDout(t)[i,e] · maskedA(t)[e,n]`

— exact real arithmetic, the two factors zeroed exactly where the kernel's own
`other = 0.0` load masks are false (`row ≥ chunk_size` / `col ≥ K`). On an
interior program, where the whole tile is in range, both guards are true and this
is the plain `Σ_cs Dout[i,cs]·A[cs,n]` of the exact headline.

Layer map: the whole kernel is **cast-free** (`.real` loads, `.real` dot/add,
the Python `.to(db_ptr.dtype.element_ty)` cast erased at translation, `.real`
store), so under `execR R` the prologue and the CS-loop collapse verbatim onto
the exact stepper. The store is `.real`-typed, so the skin's boundary
quantization degenerates: the readback's `R.round .real` is the identity
(`round_real_apply`). Hence **no** `R`-side hypothesis is needed — this face
holds for every rounding model.

Why the *masked* contract: the skin quantifies over **all** `(pid₀, pid₁, pid₂)
: ℕ³`, and `pidM pid₀ K BN = pid₀ / cdiv K BN` is unbounded, so the exact
headline's tile-level in-range premises (`hmlt` / `hmlt'` / `hnlt`, stated for
one fixed launch state) would be *contradictory* if universally quantified over
`pid₀` — the face would be vacuous whenever `BM, BN, K > 0`. The per-lane
masked chain (`bbwd_stepM` / `bbwd_loopM` / `bbwd_postLoopR`) replaces them,
covering every program of the grid instead.

Hypotheses:

* `hBCS : 0 < BCS` — the CS-loop steps by `BLOCK_SIZE_CS`; at `0` the loop never
  advances and `cdiv(CSL, BCS)` is meaningless. Exact headline's `hBCS`.
* `hOutInj` — output-address injectivity of the write window, the pid-form
  spelling of the exact headline's `hInj` (the skin quantifies the launch state
  internally, so the state-indexed spelling is not expressible here). It is
  pid-uniform and therefore satisfiable: shifting the base by the batch/chunk
  offsets does not affect injectivity in `(i, n)`. With colliding output lanes
  the per-lane readback would be last-writer-wins and the statement false. As in
  the exact headline it is carried as an open side condition, not discharged.

The exact headline's `hundef` is **not** a hypothesis here: the skin's triple
already pins `s₀.undef = fun _ _ => 0`, which is exactly the zero default the
two `other = 0.0` masked loads fall back on.

Relation to the exact surface: `bmm_chunk_bwd_output_summary_general`
(`Realizes_without_Rounding`) above is retained unchanged; this `⊨[R]` face
restates the same batched matmul backward on the streaming skin, for every `R`
and every program of the grid at once. Both are kept per the
rounding-as-default doctrine. -/
```
</details>

**Statement:**
```lean
specification bmm_chunk_bwd_io_correctness (R : RoundingModel)
    (A Dout Db : RegionName)
    (chunk_size K ngroups SAB SAS SAH SAK SDB SDC SDH SDM SDN SOB SOS SOH SOK
      BM BN BCS numCSBlocks : Nat)
    (hBCS : 0 < BCS)
    (hOutInj : ∀ pid₀ pid₁ pid₂ : Nat,
      Function.Injective (dbOffset pid₁ (pidC pid₂ ngroups) (pidH pid₂ ngroups)
        chunk_size SOB SOS SOH SOK (pidM pid₀ K BN) (pidN pid₀ K BN) BM BN)) :
    bmm_chunk_bwd_IO A Dout Db chunk_size K ngroups SAB SAS SAH SAK SDB SDC SDH SDM SDN
        SOB SOS SOH SOK BM BN BCS numCSBlocks ⊨[R]
      fun pid₀ _ _ _ xs ys l =>
        bbwdStreamSum (pidM pid₀ K BN) (pidN pid₀ K BN) BM BN BCS numCSBlocks chunk_size K
          xs ys l
```

**Assumptions / layout contracts:**
- `hBCS : 0 < BCS`

**Closed-form spec defs (transitive):** `dbOffset`, `pidC`, `pidH`, `pidM`, `pidN`, `bmm_chunk_bwd_IO`, `bbwdStreamSum`, `rowIndex`, `colIndex`, `numPidN`, `bbwd_matmul_surface`, `bbwdDoutAddr`, `bbwdAAddr`, `bbwdDbAddr`, `bbwdWriteOk`, `bbwdDoutLane`, `bbwdALane`, `cdiv`, `doutOff`, `batchOff`

<details><summary><code>dbOffset</code></summary>

```
/-- The kernel's output offset for tile lane `idx`. -/
```
```lean
def dbOffset (PB PC PH chunk_size SOB SOS SOH SOK PM PN BM BN : Nat)
    (idx : TileIndex [BM, BN]) : Nat :=
  PB * SOB + PC * chunk_size * SOS + PH * SOH + SOS * rowIndex PM BM idx.1 + colIndex PN BN idx.2.1 * SOK
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
def pidM (P0 K BN : Nat) : Nat := P0 / numPidN K BN
```
</details>

<details><summary><code>pidN</code></summary>

```
/-- `pid_n = program_id(0) % num_pid_n`. -/
```
```lean
def pidN (P0 K BN : Nat) : Nat := P0 % numPidN K BN
```
</details>

<details><summary><code>bmm_chunk_bwd_IO</code></summary>

```
/-- **Streaming IO signature** of `bmm_chunk_bwd` on the metadata-parametrized
two-stream fold skin (S1: fold + terminal masked store, 3-D pid grid), at
`nMeta := 0` — this kernel loads **no** scalar metadata, so the slot vector is
empty (`sty`/`mbuf`/`mwin` are `Fin.elim0`, and the region list degenerates to
`[Dout, A, Db]`) and every window is a function of the three pids alone. The
grid is the kernel's own: `pid₀` = flattened `(m, n)` output tile, `pid₁` =
batch, `pid₂` = chunk·group (split into `pid_c` / `pid_h` by `ngroups`).

Step `t` of the CS-loop reads the `[BLOCK_M, BLOCK_CS]` `Dout` tile and the
`[BLOCK_CS, BLOCK_N]` `A` tile; after the loop one `[BLOCK_M, BLOCK_N]` tile is
masked-stored at the **`.real`** grid (`outDType` default — the Python
`.to(db_ptr.dtype.element_ty)` cast erases at translation, so the transcribed
`tl.store` carries no quantization event). The windows transcribe the kernel's
pointer arithmetic exactly:

* `read1` lane `j = (i, e)` (row-major over `[BLOCK_M, BLOCK_CS]`), step `t`:
  `doutOff + (pid_m·BM+i)·SDN + e·SDM + t·BCS·SDM` — the invariant's
  `dout_ptrs` cell after `t` advances.
* `read2` lane `j = (e, n)` (row-major over `[BLOCK_CS, BLOCK_N]`), step `t`:
  `batchOff + e·SAS + (pid_n·BN+n)·SAK + t·BCS·SAS` — the `a_ptrs` cell.
* `write` lane `j = (i, n)`: `dbOffset` in lane form.

`writeMask` is the kernel's **own** terminal store mask `(offs_m <
chunk_size_limit) & (offs_n < K)`, so the frame is exact: the face claims a
value on precisely the lanes the program writes, for every pid of the grid.
`mask1` / `mask2` are `True`: the consumer pins (and bounds) the whole streamed
tile, and the spec `bbwdStreamSum` carries the kernel's own per-lane load
masking itself. -/
```
```lean
def bmm_chunk_bwd_IO (A Dout Db : RegionName)
    (chunk_size K ngroups SAB SAS SAH SAK SDB SDC SDH SDM SDN SOB SOS SOH SOK
      BM BN BCS numCSBlocks : Nat) :
    StreamMetaMasked3DKernelIO₂ where
  kernel := bbwd_matmul_surface A Dout Db chunk_size (BCS * numCSBlocks) K ngroups
    SAB SAS SAH SAK SDB SDC SDH SDM SDN SOB SOS SOH SOK BM BN BCS
  inp1 := Dout
  inp2 := A
  out := Db
  nMeta := 0
  sty := Fin.elim0
  mbuf := Fin.elim0
  mwin := Fin.elim0
  T := numCSBlocks
  B1 := BM * BCS
  B2 := BCS * BN
  C := BM * BN
  outDType := .real
  read1 := fun pid₀ pid₁ pid₂ _ t j =>
    bbwdDoutAddr pid₁ (pidC pid₂ ngroups) (pidH pid₂ ngroups) (pidM pid₀ K BN)
      BM BCS SDB SDC SDH SDN SDM t.val j.val
  read2 := fun pid₀ pid₁ pid₂ _ t j =>
    bbwdAAddr pid₁ (pidC pid₂ ngroups) (pidH pid₂ ngroups) (pidN pid₀ K BN)
      BN BCS chunk_size SAB SAS SAH SAK t.val j.val
  write := fun pid₀ pid₁ pid₂ _ j =>
    bbwdDbAddr pid₁ (pidC pid₂ ngroups) (pidH pid₂ ngroups) (pidM pid₀ K BN) (pidN pid₀ K BN)
      BM BN chunk_size SOB SOS SOH SOK j.val
  mask1 := fun _ _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ _ => True
  writeMask := fun pid₀ _ _ _ j =>
    bbwdWriteOk (pidM pid₀ K BN) (pidN pid₀ K BN) BM BN (BCS * numCSBlocks) K j.val
```
</details>

<details><summary><code>bbwdStreamSum</code></summary>

```
/-- **The streamed batched-matmul-backward spec**: the ideal ℝ value of output
lane `l = (i, n)` as a `T`-step fold over the streamed tiles,

`∑_{t < T} ∑_{e < BLOCK_CS} maskedDout(t)[i, e] · maskedA(t)[e, n]`,

where the two factors are zeroed exactly where the kernel's own `other = 0.0`
load masks are false — the `dout` tile on rows `≥ chunk_size`, the `a` tile on
columns `≥ K`. (The kernel's third conjunct, the per-block CS tail
`offs_cs < chunk_size_limit - cs·BLOCK_CS`, is all-true at
`CSL = BLOCK_CS·numCSBlocks` and so contributes nothing.) This is the `⊨[R]`
face of `bbwdSpec`: the same `Σ_cs Dout·A` reference, split into the kernel's own
`BLOCK_CS`-sized contraction blocks and carrying the kernel's own masking. -/
```
```lean
noncomputable def bbwdStreamSum (PM PN BM BN BCS T chunk_size K : Nat)
    (xs : Fin T → Fin (BM * BCS) → ℝ) (ys : Fin T → Fin (BCS * BN) → ℝ)
    (l : Fin (BM * BN)) : ℝ :=
  ∑ t : Fin T, ∑ e : Fin BCS,
    (if PM * BM + l.val / BN < chunk_size then xs t (bbwdDoutLane BM BN BCS l e) else 0)
      * (if PN * BN + l.val % BN < K then ys t (bbwdALane BM BN BCS l e) else 0)
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
/-- Global output col of tile lane `j`: `PN · BN + j`. -/
```
```lean
def colIndex (PN BN : Nat) (j : Fin BN) : Nat := PN * BN + j.val
```
</details>

<details><summary><code>numPidN</code></summary>

```
/-- The kernel's `num_pid_n = cdiv(K, BLOCK_SIZE_N)`. -/
```
```lean
def numPidN (K BN : Nat) : Nat := cdiv K BN
```
</details>

<details><summary><code>bbwd_matmul_surface</code></summary>

```
/-- The genuine batched-matmul-backward surface: chunked `acc += tl.dot(dout, a)`
with the kernel's batch/chunk/head pointer offsets, CS-block dot loop, per-block
CS-tail and `chunk_size`/`K` row/col load masks, and the masked output store. -/
```
```lean
def bbwd_matmul_surface
    (A Dout Db : RegionName)
    (chunk_size CSL K ngroups
      stride_a_batch stride_a_seqlen stride_a_head stride_ak
      stride_dout_batch stride_dout_chunk stride_dout_head
      stride_dout_csize_m stride_dout_csize_n
      stride_db_batch stride_db_seqlen stride_db_head stride_db_k
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=1)
  pid_ch = tl.program_id(axis=2)
  pid_c = pid_ch // $(ngroups)
  pid_h = pid_ch - pid_c * $(ngroups)
  num_pid_n = tl.cdiv($(K), $(BLOCK_SIZE_N))
  pid_m = tl.program_id(axis=0) // num_pid_n
  pid_n = tl.program_id(axis=0) % num_pid_n
  Dout += pid_b * $(stride_dout_batch) +
    pid_c * $(stride_dout_chunk) + pid_h * $(stride_dout_head)
  A += pid_b * $(stride_a_batch) +
    pid_c * $(chunk_size) * $(stride_a_seqlen) + pid_h * $(stride_a_head)
  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  offs_cs = tl.arange(0, $(BLOCK_SIZE_CS))
  dout_ptrs = Dout + offs_m[:, None] * $(stride_dout_csize_n) + offs_cs[None, :] * $(stride_dout_csize_m)
  a_ptrs = A + offs_cs[:, None] * $(stride_a_seqlen) + offs_n[None, :] * $(stride_ak)
  acc = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for cs in range($(0), tl.cdiv($(CSL), $(BLOCK_SIZE_CS)), $(1)) {
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < $(chunk_size)) &
      (offs_cs[None, :] < $(CSL) - cs * $(BLOCK_SIZE_CS)), other=0.0)
    a = tl.load(a_ptrs, mask=(offs_cs[:, None] < $(CSL) - cs * $(BLOCK_SIZE_CS)) &
      (offs_n[None, :] < $(K)), other=0.0)
    acc += tl.dot(dout, a)
    dout_ptrs += $(BLOCK_SIZE_CS) * $(stride_dout_csize_m)
    a_ptrs += $(BLOCK_SIZE_CS) * $(stride_a_seqlen)
  }
  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  db = (acc).to(Db.dtype.element_ty)
  Db += pid_b * $(stride_db_batch) + pid_c * $(chunk_size) * $(stride_db_seqlen) + pid_h * $(stride_db_head)
  db_ptrs = Db + $(stride_db_seqlen) * offs_m[:, None] + offs_n[None, :] * $(stride_db_k)
  tl.store(db_ptrs, db, mask=(offs_m[:, None] < $(CSL)) &
    (offs_n[None, :] < $(K)))
}
```
</details>

<details><summary><code>bbwdDoutAddr</code></summary>

```
/-- The `Dout` window: `dout_ptrs` cell `(i, e) = (j / BCS, j % BCS)` after `t`
block advances — `doutOff + (PM·BM+i)·SDN + e·SDM + t·BCS·SDM`. -/
```
```lean
private def bbwdDoutAddr (PB PC PH PM BM BCS SDB SDC SDH SDN SDM : Nat) (t j : Nat) : Nat :=
  doutOff PB PC PH SDB SDC SDH + (PM * BM + j / BCS) * SDN + (j % BCS) * SDM + t * BCS * SDM
```
</details>

<details><summary><code>bbwdAAddr</code></summary>

```
/-- The `A` window: `a_ptrs` cell `(e, n) = (j / BN, j % BN)` after `t` block
advances — `batchOff + e·SAS + (PN·BN+n)·SAK + t·BCS·SAS`. -/
```
```lean
private def bbwdAAddr (PB PC PH PN BN BCS chunk_size SAB SAS SAH SAK : Nat) (t j : Nat) : Nat :=
  batchOff PB PC PH chunk_size SAB SAS SAH + (j / BN) * SAS + (PN * BN + j % BN) * SAK
    + t * BCS * SAS
```
</details>

<details><summary><code>bbwdDbAddr</code></summary>

```
/-- The output window: `db_ptrs` cell `(i, n) = (j / BN, j % BN)`, i.e.
`dbOffset` in lane form. -/
```
```lean
private def bbwdDbAddr (PB PC PH PM PN BM BN chunk_size SOB SOS SOH SOK : Nat) (j : Nat) : Nat :=
  PB * SOB + PC * chunk_size * SOS + PH * SOH + SOS * (PM * BM + j / BN)
    + (PN * BN + j % BN) * SOK
```
</details>

<details><summary><code>bbwdWriteOk</code></summary>

```
/-- The terminal store's active set in **lane** form (the shape the skin's
`writeMask` field takes). -/
```
```lean
def bbwdWriteOk (PM PN BM BN CSL K : Nat) (j : Nat) : Prop :=
  PM * BM + j / BN < CSL ∧ PN * BN + j % BN < K

instance (PM PN BM BN CSL K j : Nat) : Decidable (bbwdWriteOk PM PN BM BN CSL K j) := by
  unfold bbwdWriteOk; infer_instance
```
</details>

<details><summary><code>bbwdDoutLane</code></summary>

```
/-- The `Dout`-stream lane feeding output lane `l` at contraction key `e`: the
row of `l` (row-major over the `[BM, BN]` output tile) paired with `e` over the
`[BM, BLOCK_CS]` per-step `Dout` tile, both through the shared `Lane2D`
bridge. -/
```
```lean
def bbwdDoutLane (BM BN BCS : Nat) (l : Fin (BM * BN)) (e : Fin BCS) : Fin (BM * BCS) :=
  Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit)
```
</details>

<details><summary><code>bbwdALane</code></summary>

```
/-- The `A`-stream lane feeding output lane `l` at contraction key `e`: `e`
paired with the column of `l` over the `[BLOCK_CS, BN]` per-step `A` tile. -/
```
```lean
def bbwdALane (BM BN BCS : Nat) (l : Fin (BM * BN)) (e : Fin BCS) : Fin (BCS * BN) :=
  Lane2D.encode (e, (Lane2D.decode l).2.1, PUnit.unit)
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

<details><summary><code>doutOff</code></summary>

```
/-- The kernel's `dout` batch+chunk+head base offset (no `chunk_size` multiply):
`pid_b·stride_dout_batch + pid_c·stride_dout_chunk + pid_h·stride_dout_head`. -/
```
```lean
def doutOff (PB PC PH SDB SDC SDH : Nat) : Nat :=
  PB * SDB + PC * SDC + PH * SDH
```
</details>

<details><summary><code>batchOff</code></summary>

```
/-- The kernel's `a`/`db` batch+chunk+head base offset (with `chunk_size`):
`pid_b·stride_batch + pid_c·chunk_size·stride_seqlen + pid_h·stride_head`. -/
```
```lean
def batchOff (PB PC PH chunk_size stride_batch stride_seqlen stride_head : Nat) : Nat :=
  PB * stride_batch + PC * chunk_size * stride_seqlen + PH * stride_head
```
</details>

## Also present (pinned special-case summaries)
- `bmm_chunk_bwd_closed_form_correct`
