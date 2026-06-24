# Spec sheet — `bench/tritonbench_g/matmul_triton2/MatmulTriton2.lean`

**Python source:** `bench/tritonbench_g/matmul_triton2/matmul_triton2.py`

## Public theorem: `matmul_triton2_python_case1_output_summary`

<details><summary>docstring</summary>

```
/-- **Public Python case 1 summary**: the full `256×256` matmul surface realizes
the genuine matrix product `Σ_{k<256} A[i,k]·B[k,j]` on every active output lane
(`BLOCK_K = 64`, `numKBlocks = 4`, so `K = 256`). -/
```
</details>

**Statement:**
```lean
theorem matmul_triton2_python_case1_output_summary
    (A B C : RegionName) (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (matmul_triton2_surface A B C 256 256 256 256 1 256 1 256 1 128 256 64 8).toAlgorithm?
        = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := matmul_triton2_surface A B C 256 256 256 256 1 256 1 256 1 128 256 64 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 256 256 128 256 8)
        (fun idx => (C, cOffset s 256 256 128 256 8 256 1 idx)))
      (expected := fun idx : TileIndex [128, 256] =>
        matmulSpec s A B 256 256 128 256 8 256 1 256 1 64 4 idx.1 idx.2.1)
```

**Assumptions / layout contracts:**
- `hundef : ∀ rg o, s.undef rg o = 0`
- `kernel : = matmul_triton2_surface A B C 256 256 256 256 1 256 1 256 1 128 256 64 8`
- `initialState : = s`
- `expected : = fun idx : TileIndex [128, 256] =>
        matmulSpec s A B 256 256 128 256 8 256 1 256 1 64 4 idx.1 idx.2.1`

**Closed-form spec defs (transitive):** `matmul_triton2_surface`, `active`, `cOffset`, `matmulSpec`, `cdiv`, `rowIndex`, `colIndex`, `aElem`, `bElem`, `pidM`, `pidN`, `numPidM`, `numPidN`

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

<details><summary><code>cOffset</code></summary>

```
/-- The output store address for tile lane `(i,j)`: `rowIndex i · SCM + colIndex j · SCN`. -/
```
```lean
def cOffset (s0 : BlockState) (M N BM BN GM SCM SCN : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  rowIndex s0 M N BM BN GM idx.1 * SCM + colIndex s0 M N BM BN GM idx.2.1 * SCN
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

<details><summary><code>cdiv</code></summary>

```
/-- Ceiling division `⌈a / b⌉`, matching Triton's `tl.cdiv`. -/
```
```lean
def cdiv (a b : Nat) : Nat := (a + b - 1) / b
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

## Public theorem: `matmul_triton2_python_case2_output_summary`

<details><summary>docstring</summary>

```
/-- **Public Python case 2 summary**: the full `64×64` matmul surface realizes the
genuine matrix product `Σ_{k<64} A[i,k]·B[k,j]` on every active output lane
(`BLOCK_K = 32`, `numKBlocks = 2`, so `K = 64`). -/
```
</details>

**Statement:**
```lean
theorem matmul_triton2_python_case2_output_summary
    (A B C : RegionName) (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (matmul_triton2_surface A B C 64 64 64 64 1 64 1 64 1 32 64 32 8).toAlgorithm?
        = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := matmul_triton2_surface A B C 64 64 64 64 1 64 1 64 1 32 64 32 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 64 64 32 64 8)
        (fun idx => (C, cOffset s 64 64 32 64 8 64 1 idx)))
      (expected := fun idx : TileIndex [32, 64] =>
        matmulSpec s A B 64 64 32 64 8 64 1 64 1 32 2 idx.1 idx.2.1)
```

**Assumptions / layout contracts:**
- `hundef : ∀ rg o, s.undef rg o = 0`
- `kernel : = matmul_triton2_surface A B C 64 64 64 64 1 64 1 64 1 32 64 32 8`
- `initialState : = s`
- `expected : = fun idx : TileIndex [32, 64] =>
        matmulSpec s A B 64 64 32 64 8 64 1 64 1 32 2 idx.1 idx.2.1`

**Closed-form spec defs (transitive):** `matmul_triton2_surface`, `active`, `cOffset`, `matmulSpec`, `cdiv`, `rowIndex`, `colIndex`, `aElem`, `bElem`, `pidM`, `pidN`, `numPidM`, `numPidN`

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

<details><summary><code>cOffset</code></summary>

```
/-- The output store address for tile lane `(i,j)`: `rowIndex i · SCM + colIndex j · SCN`. -/
```
```lean
def cOffset (s0 : BlockState) (M N BM BN GM SCM SCN : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  rowIndex s0 M N BM BN GM idx.1 * SCM + colIndex s0 M N BM BN GM idx.2.1 * SCN
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

<details><summary><code>cdiv</code></summary>

```
/-- Ceiling division `⌈a / b⌉`, matching Triton's `tl.cdiv`. -/
```
```lean
def cdiv (a b : Nat) : Nat := (a + b - 1) / b
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

## Also present (pinned special-case summaries)
- `matmul_triton2_closed_form_correct`
