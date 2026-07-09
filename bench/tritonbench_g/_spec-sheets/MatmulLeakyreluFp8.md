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
theorem matmul_leakyrelu_fp8_closed_form_correct
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
