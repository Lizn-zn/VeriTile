# Spec sheet — `bench/tritonbench_g/bmm_chunk_fwd/BmmChunkFwd.lean`

**Python source:** `bench/tritonbench_g/bmm_chunk_fwd/bmm_chunk_fwd.py`

## Public theorem: `bmm_chunk_fwd_python_case1_summary`

<details><summary>docstring</summary>

```
/-- **Python case 1 summary** (ungrouped: `batch=2`, `seqlen=128`, `K=64`,
`chunk_size=32`; strides `a=b=(8192,64,0,1)`, `out=(4096,1024,0,32,1)`,
`ngroups=1`; `BLOCK_SIZE_K=32`, `numKBlocks=2`): the genuine batched matmul
surface lowers to the algorithm layer and realizes the per-program matrix
product `Σ_{k<64} A[i,k]·B[k,j]` on every in-bounds output lane. -/
```
</details>

**Statement:**
```lean
theorem bmm_chunk_fwd_python_case1_summary
    (A B Out : RegionName) (s : BlockState) (BM BN : Nat)
    (hInj : Function.Injective (outOffset (s.pids 1) (pidC (s.pids 2) 1) (pidH (s.pids 2) 1)
      32 4096 1024 0 32 1 (pidM (s.pids 0) 32 BN) (pidN (s.pids 0) 32 BN) BM BN))
    (hmlt : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) 32 BN) BM i < 32)
    (hnlt : ∀ j : Fin BN, colIndex (pidN (s.pids 0) 32 BN) BN j < 32)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (bmm_matmul_surface A B Out 32 64 1 8192 64 0 1 8192 64 0 1
        4096 1024 0 32 1 BM BN 32).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := bmm_matmul_surface A B Out 32 64 1 8192 64 0 1 8192 64 0 1
        4096 1024 0 32 1 BM BN 32)
      (initialState := s)
      (write := fun idx : TileIndex [BM, BN] =>
        some (Out, outOffset (s.pids 1) (pidC (s.pids 2) 1) (pidH (s.pids 2) 1)
          32 4096 1024 0 32 1 (pidM (s.pids 0) 32 BN) (pidN (s.pids 0) 32 BN) BM BN idx))
      (expected := fun idx : TileIndex [BM, BN] =>
        outputCell s A B (s.pids 1) (pidC (s.pids 2) 1) (pidH (s.pids 2) 1)
          (pidM (s.pids 0) 32 BN) (pidN (s.pids 0) 32 BN)
          32 BM BN 8192 64 0 1 8192 64 0 1 32 2 idx)
```

**Assumptions / layout contracts:**
- `hundef : ∀ rg o, s.undef rg o = 0`
- `kernel : = bmm_matmul_surface A B Out 32 64 1 8192 64 0 1 8192 64 0 1
        4096 1024 0 32 1 BM BN 32`
- `initialState : = s`

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

## Public theorem: `bmm_chunk_fwd_python_case4_summary`

<details><summary>docstring</summary>

```
/-- **Python case 4 summary** (grouped: `batch=2`, `seqlen=128`, `ngroups=4`,
`K=64`, `chunk_size=32`; strides `a=b=(32768,256,64,1)`,
`out=(16384,4096,1024,32,1)`; `BLOCK_SIZE_K=32`, `numKBlocks=2`): the genuine
batched matmul surface lowers and realizes `Σ_{k<64} A[i,k]·B[k,j]` per program. -/
```
</details>

**Statement:**
```lean
theorem bmm_chunk_fwd_python_case4_summary
    (A B Out : RegionName) (s : BlockState) (BM BN : Nat)
    (hInj : Function.Injective (outOffset (s.pids 1) (pidC (s.pids 2) 4) (pidH (s.pids 2) 4)
      32 16384 4096 1024 32 1 (pidM (s.pids 0) 32 BN) (pidN (s.pids 0) 32 BN) BM BN))
    (hmlt : ∀ i : Fin BM, rowIndex (pidM (s.pids 0) 32 BN) BM i < 32)
    (hnlt : ∀ j : Fin BN, colIndex (pidN (s.pids 0) 32 BN) BN j < 32)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (bmm_matmul_surface A B Out 32 64 4 32768 256 64 1 32768 256 64 1
        16384 4096 1024 32 1 BM BN 32).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := bmm_matmul_surface A B Out 32 64 4 32768 256 64 1 32768 256 64 1
        16384 4096 1024 32 1 BM BN 32)
      (initialState := s)
      (write := fun idx : TileIndex [BM, BN] =>
        some (Out, outOffset (s.pids 1) (pidC (s.pids 2) 4) (pidH (s.pids 2) 4)
          32 16384 4096 1024 32 1 (pidM (s.pids 0) 32 BN) (pidN (s.pids 0) 32 BN) BM BN idx))
      (expected := fun idx : TileIndex [BM, BN] =>
        outputCell s A B (s.pids 1) (pidC (s.pids 2) 4) (pidH (s.pids 2) 4)
          (pidM (s.pids 0) 32 BN) (pidN (s.pids 0) 32 BN)
          32 BM BN 32768 256 64 1 32768 256 64 1 32 2 idx)
```

**Assumptions / layout contracts:**
- `hundef : ∀ rg o, s.undef rg o = 0`
- `kernel : = bmm_matmul_surface A B Out 32 64 4 32768 256 64 1 32768 256 64 1
        16384 4096 1024 32 1 BM BN 32`
- `initialState : = s`

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

## Also present (pinned special-case summaries)
- `bmm_chunk_fwd_closed_form_correct`
