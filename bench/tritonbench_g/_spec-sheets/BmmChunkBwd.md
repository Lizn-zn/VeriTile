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

## Also present (pinned special-case summaries)
- `bmm_chunk_bwd_closed_form_correct`
