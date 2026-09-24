# Spec sheet — `bench/tritonbench_g/matmul_dequantize/MatmulDequantize.lean`

**Python source:** `bench/tritonbench_g/matmul_dequantize/matmul_dequantize.py`

## Public theorem: `matmul_dequantize_matmul4_exec_genuine`

<details><summary>docstring</summary>

```
/-- **Genuine, dimension-general correctness.** For every launch state, the kernel
runs to completion and every in-range output lane of `C` holds `accSpec`: the sum
over all `num_pid_k` K steps of `tl.dot` between the row-masked `A` tile and the
dequantized `B` tile, with `scales` / `zeros` read at that step's group row.

`hInj` says distinct output lanes get distinct `C` addresses — without it the
readback could not name a lane. `cAddr_injective` discharges it for a row-major
`C`. The `NO_GROUPS` flag is free: both configurations are covered by the same
statement, since `groupRow` is what differs and `accSpec` takes it into account. -/
```
</details>

**Statement:**
```lean
specification matmul_dequantize_matmul4_exec_genuine
    (a_ptr c_ptr scales_ptr : RegionName) (b_ptr zeros_ptr : Region .nat)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      groupsize : Nat) (NO_GROUPS : Bool) (BM BN BK GM : Nat) (s : BlockState)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN
        (pidM s M N BM BN GM) (pidN s M N BM BN GM) i)) :
    ∃ sF, exec (matmul_dequantize_matmul4_surface a_ptr c_ptr scales_ptr b_ptr zeros_ptr
        M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
        groupsize NO_GROUPS BM BN BK GM).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (pidM s M N BM BN GM * BM + idx.1.val < M
            ∧ pidN s M N BM BN GM * BN + idx.2.1.val < N) →
          sF.readMem c_ptr (cAddr stride_cm stride_cn BM BN (pidM s M N BM BN GM)
              (pidN s M N BM BN GM) idx)
            = accSpec s a_ptr scales_ptr b_ptr zeros_ptr NO_GROUPS M K groupsize
                stride_am stride_ak stride_bk stride_bn stride_scales_g
                stride_scales_n stride_zeros_g stride_zeros_n BM BN BK
                (pidM s M N BM BN GM) (pidN s M N BM BN GM)
                idx.1.val idx.2.1.val
```

**Closed-form spec defs (transitive):** `cAddr`, `pidM`, `pidN`, `matmul_dequantize_matmul4_surface`, `accSpec`, `firstPidM`, `groupSizeM`, `numPidInGroup`, `numPidK`, `accStep`, `groupId`, `numPidM`, `numPidN`, `aElem`, `bDequant`, `bNibble`, `scalesElem`, `groupRow`, `zeroScaled`, `bWord`, `zerosNibble`, `zerosWord`

<details><summary><code>cAddr</code></summary>

```
/-- The `C` store address for output cell `(r, c)`. -/
```
```lean
def cAddr (stride_cm stride_cn BM BN pm pn : Nat)
    (idx : TileIndex [BM, BN]) : Nat :=
  stride_cm * (pm * BM + idx.1.val) + stride_cn * (pn * BN + idx.2.1.val)
```
</details>

<details><summary><code>pidM</code></summary>

```
/-- `pid_m = first_pid_m + (pid % group_size_m)`. -/
```
```lean
def pidM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  firstPidM s N BN GM + s.pids 0 % groupSizeM s M N BM BN GM
```
</details>

<details><summary><code>pidN</code></summary>

```
/-- `pid_n = (pid % num_pid_in_group) // group_size_m`. -/
```
```lean
def pidN (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  s.pids 0 % numPidInGroup N BN GM / groupSizeM s M N BM BN GM
```
</details>

<details><summary><code>matmul_dequantize_matmul4_surface</code></summary>

```
/-! ## Kernel surface (faithful transcription) -/
```
```lean
def matmul_dequantize_matmul4_surface
    (a_ptr c_ptr scales_ptr : RegionName) (b_ptr zeros_ptr : Region .nat)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      groupsize : Nat) (NO_GROUPS : Bool)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) :
    ComputeKernel := triton {
  bits = $(4)
  infearure_per_bits = $(8)
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_k = tl.cdiv($(K), $(BLOCK_SIZE_K))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_bn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = a_ptr + (offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak))
  a_mask = (offs_am[:, None] < $(M))
  b_ptrs = b_ptr + ((offs_k[:, None] // infearure_per_bits) * $(stride_bk) + offs_bn[None, :] * $(stride_bn))
  scales_ptrs = scales_ptr + offs_bn * $(stride_scales_n)
  zeros_ptrs = zeros_ptr + ((offs_bn // infearure_per_bits) * $(stride_zeros_n))
  shifter = (offs_k % infearure_per_bits) * bits
  zeros_shifter = (offs_bn % infearure_per_bits) * bits
  if NO_GROUPS {
    scales = tl.load(scales_ptrs)
    zeros = tl.load(zeros_ptrs)
    zeros = (zeros >> zeros_shifter) & $(15)
    zeros = zeros * scales
  }
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), num_pid_k, $(1)) {
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs)
    if not NO_GROUPS {
      g_id = k // ($(groupsize) // $(BLOCK_SIZE_K))
      ptr = scales_ptrs + g_id * $(stride_scales_g)
      scales = tl.load(ptr)
      ptr = zeros_ptrs + g_id * $(stride_zeros_g)
      zeros = tl.load(ptr)
      zeros = (zeros >> zeros_shifter) & $(15)
      zeros = (zeros) * scales
    }
    b = (b >> shifter[:, None]) & $(15)
    b = b * scales[None, :] - zeros[None, :]
    accumulator += tl.dot(a, b)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += ($(BLOCK_SIZE_K) // infearure_per_bits) * $(stride_bk)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = c_ptr + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, accumulator, mask=c_mask)
}
```
</details>

<details><summary><code>accSpec</code></summary>

```
/-- **The stored value.** `accumulator` after all `num_pid_k` K steps. (The source
also computes `c = accumulator.to(...)` but stores `accumulator`, so this is what
lands in memory.) -/
```
```lean
noncomputable def accSpec (s : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M K groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK pm pn : Nat) (r c : Nat) : ℝ :=
  ∑ k : Fin (numPidK K BK),
    accStep s a scales b zeros NO_GROUPS M groupsize stride_am stride_ak
      stride_bk stride_bn stride_scales_g stride_scales_n
      stride_zeros_g stride_zeros_n BM BN BK pm pn k.val r c
```
</details>

<details><summary><code>firstPidM</code></summary>

```
/-- `first_pid_m = group_id * GROUP_SIZE_M`. -/
```
```lean
def firstPidM (s : BlockState) (N BN GM : Nat) : Nat := groupId s N BN GM * GM
```
</details>

<details><summary><code>groupSizeM</code></summary>

```
/-- `group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)`. -/
```
```lean
def groupSizeM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  min (numPidM M BM - firstPidM s N BN GM) GM
```
</details>

<details><summary><code>numPidInGroup</code></summary>

```
/-- `num_pid_in_group = GROUP_SIZE_M * num_pid_n`. -/
```
```lean
def numPidInGroup (N BN GM : Nat) : Nat := GM * numPidN N BN
```
</details>

<details><summary><code>numPidK</code></summary>

```
/-- `num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)` — the K-loop trip count in
`matmul4_kernel`. In `matmul_kernel` below the same binding is **dead** (that
loop's bound is `numKBlocks`), but it is still transcribed and walked. -/
```
```lean
def numPidK (K BK : Nat) : Nat := (K + BK - 1) / BK
```
</details>

<details><summary><code>accStep</code></summary>

```
/-- One K step's contribution to output cell `(r, c)`. -/
```
```lean
noncomputable def accStep (s : BlockState) (a scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (M groupsize stride_am stride_ak stride_bk stride_bn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      BM BN BK pm pn : Nat) (k r c : Nat) : ℝ :=
  ∑ e : Fin BK,
    aElem s a M stride_am stride_ak BM BK pm r k e.val
      * bDequant s scales b zeros NO_GROUPS groupsize stride_bk stride_bn
          stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
          BN BK pn k e.val c
```
</details>

<details><summary><code>groupId</code></summary>

```
/-- `group_id = pid // num_pid_in_group`. -/
```
```lean
def groupId (s : BlockState) (N BN GM : Nat) : Nat :=
  s.pids 0 / numPidInGroup N BN GM
```
</details>

<details><summary><code>numPidM</code></summary>

```
/-- `num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)`. -/
```
```lean
def numPidM (M BM : Nat) : Nat := (M + BM - 1) / BM
```
</details>

<details><summary><code>numPidN</code></summary>

```
/-- `num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)`. -/
```
```lean
def numPidN (N BN : Nat) : Nat := (N + BN - 1) / BN
```
</details>

<details><summary><code>aElem</code></summary>

```
/-- `a[r, k*BK + e]` at K step `k`. Masked on the row axis only — that is the
kernel's `a_mask = offs_am[:, None] < M` with `other=0.0`. There is deliberately
no K-axis mask: the source documents `K % BLOCK_SIZE_K == 0` as a precondition,
and transcribing a mask it does not have would be unfaithful. -/
```
```lean
noncomputable def aElem (s : BlockState) (a : RegionName)
    (M stride_am stride_ak BM BK pm : Nat) (r k e : Nat) : ℝ :=
  if pm * BM + r < M then
    s.readMem a ((pm * BM + r) * stride_am + (e + k * BK) * stride_ak)
  else 0
```
</details>

<details><summary><code>bDequant</code></summary>

```
/-- `b` after unpack, scale and zero-point shift at K step `k`:
`b * scales - zeros`, where `zeros` is already scaled. -/
```
```lean
noncomputable def bDequant (s : BlockState) (scales : RegionName)
    (b zeros : Region .nat) (NO_GROUPS : Bool)
    (groupsize stride_bk stride_bn stride_scales_g stride_scales_n
      stride_zeros_g stride_zeros_n BN BK pn : Nat) (k e c : Nat) : ℝ :=
  (bNibble s b stride_bk stride_bn BN BK pn k e c : ℝ)
      * scalesElem s scales stride_scales_g stride_scales_n BN pn
          (groupRow NO_GROUPS groupsize BK k) c
    - zeroScaled s scales zeros stride_scales_g stride_scales_n
        stride_zeros_g stride_zeros_n BN pn (groupRow NO_GROUPS groupsize BK k) c
```
</details>

<details><summary><code>bNibble</code></summary>

```
/-- The unpacked 4-bit weight: `(word >> (e % 8) * 4) & 0xF`. -/
```
```lean
def bNibble (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BN BK pn : Nat) (k e c : Nat) : Nat :=
  bWord s b stride_bk stride_bn BN BK pn k e c >>> (e % 8 * 4) &&& 15
```
</details>

<details><summary><code>scalesElem</code></summary>

```
/-- `scales[g, pn*BN + c]`. -/
```
```lean
noncomputable def scalesElem (s : BlockState) (scales : RegionName)
    (stride_scales_g stride_scales_n BN pn : Nat) (g c : Nat) : ℝ :=
  s.readMem scales ((pn * BN + c) * stride_scales_n + g * stride_scales_g)
```
</details>

<details><summary><code>groupRow</code></summary>

```
/-- The group row read at K step `k`. Under `NO_GROUPS` the scales and zeros are
loaded once before the loop **from the base pointers**, i.e. at row `0`; otherwise
each step reloads at `k // (groupsize // BLOCK_SIZE_K)`. -/
```
```lean
def groupRow (NO_GROUPS : Bool) (groupsize BK k : Nat) : Nat :=
  if NO_GROUPS then 0 else k / (groupsize / BK)
```
</details>

<details><summary><code>zeroScaled</code></summary>

```
/-- `zeros` after the kernel's `zeros = zeros * scales`: the **scaled**
zero-point. Both branches apply this before the loop body uses it. -/
```
```lean
noncomputable def zeroScaled (s : BlockState) (scales : RegionName)
    (zeros : Region .nat)
    (stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n BN pn : Nat)
    (g c : Nat) : ℝ :=
  (zerosNibble s zeros stride_zeros_g stride_zeros_n BN pn g c : ℝ)
    * scalesElem s scales stride_scales_g stride_scales_n BN pn g c
```
</details>

<details><summary><code>bWord</code></summary>

```
/-- The packed 32-bit word holding the weight nibble for `(k, e, c)`. Unmasked,
matching the source's bare `tl.load(b_ptrs)`. Eight weights share a word along K,
hence the `e / 8`. -/
```
```lean
def bWord (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BN BK pn : Nat) (k e c : Nat) : Nat :=
  s.readMemValue .nat b
    ((e / 8 + k * (BK / 8)) * stride_bk + (pn * BN + c) * stride_bn)
```
</details>

<details><summary><code>zerosNibble</code></summary>

```
/-- The unpacked 4-bit zero-point. -/
```
```lean
def zerosNibble (s : BlockState) (zeros : Region .nat)
    (stride_zeros_g stride_zeros_n BN pn : Nat) (g c : Nat) : Nat :=
  zerosWord s zeros stride_zeros_g stride_zeros_n BN pn g c
    >>> ((pn * BN + c) % 8 * 4) &&& 15
```
</details>

<details><summary><code>zerosWord</code></summary>

```
/-- The packed word holding the zero-point nibble for column `pn*BN + c`. Eight
zero-points share a word along N, hence the `/ 8`. -/
```
```lean
def zerosWord (s : BlockState) (zeros : Region .nat)
    (stride_zeros_g stride_zeros_n BN pn : Nat) (g c : Nat) : Nat :=
  s.readMemValue .nat zeros
    ((pn * BN + c) / 8 * stride_zeros_n + g * stride_zeros_g)
```
</details>

## Public theorem: `matmul_dequantize_matmul_exec_genuine`

<details><summary>docstring</summary>

```
/-- **Genuine, dimension-general correctness.** For every launch state, the
`SPLIT_K = 1` arm of the kernel runs to completion, and every in-range output
lane of `C` holds `i4AccSpec`: the sum over all `numKBlocks` K steps of `tl.dot`
between the loaded `A` tile and the signed-dequantized weight tile
`(nib(B) − nib(BZP)) · BS`, with per-lane group rows `(e + k·BK) / group_size`.
The store writes `c = accumulator.to(tl.float16)` — a genuine fp16
quantization event, so every active output cell is a **`.fp16`-typed memory
cell** and the conclusion is stated at the `MemCell` level:
`C[row, col] = MemCell.of .fp16 (fp16(accSpec))` (the `matmul_tma` f16-branch
precedent; the placeholder `FloatDType.cast` is the identity, so the carried
value is `i4AccSpec` itself).

The hypotheses are the kernel's own launch facts: `hK` is the source's
`assert K % (BLOCK_SIZE_K * SPLIT_K) == 0` (the loop trip count `numKBlocks`
is exact — the loads are unmasked); `hBK8` is the source's
`assert BLOCK_SIZE_K % 8 == 0` (eight nibbles per packed word); `hpid1` is
grid axis 1 having extent `SPLIT_K = 1`; `hInj` says distinct output lanes get
distinct `C` addresses — `cAddr_injective` discharges it for a row-major `C`.
The `% M` / `% N` offset wraps disappear on exactly the lanes the store mask
lets through (`Nat.mod_eq_of_lt`). -/
```
</details>

**Statement:**
```lean
specification matmul_dequantize_matmul_exec_genuine
    (a_ptr c_ptr bs_ptr : RegionName) (b_ptr bzp_ptr : Region .nat)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_bsk stride_bsn stride_bzpk stride_bzpn group_size : Nat)
    (BM BN BK GM numKBlocks : Nat) (s : BlockState)
    (hK : K = BK * numKBlocks)
    (hBK8 : BK % 8 = 0)
    (hpid1 : s.pids 1 = 0)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN
        (pidM s M N BM BN GM) (pidN s M N BM BN GM) i)) :
    ∃ sF, exec (matmul_dequantize_matmul_surface a_ptr c_ptr bs_ptr b_ptr bzp_ptr
        M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        stride_bsk stride_bsn stride_bzpk stride_bzpn group_size
        BM BN BK GM numKBlocks).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (pidM s M N BM BN GM * BM + idx.1.val < M
            ∧ pidN s M N BM BN GM * BN + idx.2.1.val < N) →
          sF.mem c_ptr (cAddr stride_cm stride_cn BM BN (pidM s M N BM BN GM)
              (pidN s M N BM BN GM) idx)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (i4AccSpec s a_ptr bs_ptr b_ptr bzp_ptr group_size
                  stride_am stride_ak stride_bk stride_bn stride_bsk stride_bsn
                  stride_bzpk stride_bzpn BK numKBlocks
                  (pidM s M N BM BN GM * BM + idx.1.val)
                  (pidN s M N BM BN GM * BN + idx.2.1.val))))
```

**Assumptions / layout contracts:**
- `hK : K = BK * numKBlocks`
- `hBK8 : BK % 8 = 0`
- `hpid1 : s.pids 1 = 0`

**Closed-form spec defs (transitive):** `cAddr`, `pidM`, `pidN`, `matmul_dequantize_matmul_surface`, `i4AccSpec`, `firstPidM`, `groupSizeM`, `numPidInGroup`, `i4AccStep`, `groupId`, `numPidM`, `numPidN`, `i4AElem`, `i4BDequant`, `i4BNibble`, `bzpNibble`, `bsElem`, `i4BWord`, `bzpWord`

<details><summary><code>cAddr</code></summary>

```
/-- The `C` store address for output cell `(r, c)`. -/
```
```lean
def cAddr (stride_cm stride_cn BM BN pm pn : Nat)
    (idx : TileIndex [BM, BN]) : Nat :=
  stride_cm * (pm * BM + idx.1.val) + stride_cn * (pn * BN + idx.2.1.val)
```
</details>

<details><summary><code>pidM</code></summary>

```
/-- `pid_m = first_pid_m + (pid % group_size_m)`. -/
```
```lean
def pidM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  firstPidM s N BN GM + s.pids 0 % groupSizeM s M N BM BN GM
```
</details>

<details><summary><code>pidN</code></summary>

```
/-- `pid_n = (pid % num_pid_in_group) // group_size_m`. -/
```
```lean
def pidN (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  s.pids 0 % numPidInGroup N BN GM / groupSizeM s M N BM BN GM
```
</details>

<details><summary><code>matmul_dequantize_matmul_surface</code></summary>

```
/-! ## Kernel surface (faithful transcription, `SPLIT_K = 1` arm) -/
```
```lean
def matmul_dequantize_matmul_surface
    (a_ptr c_ptr bs_ptr : RegionName) (b_ptr bzp_ptr : Region .nat)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_bsk stride_bsn stride_bzpk stride_bzpn
      group_size : Nat)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  pid_sp_k = tl.program_id(axis=1)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_k = tl.cdiv($(K), $(BLOCK_SIZE_K))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = pid_sp_k * $(BLOCK_SIZE_K) + tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = a_ptr + (offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak))
  b_ptrs = b_ptr + ((offs_k[:, None] // $(8)) * $(stride_bk) + offs_bn[None, :] * $(stride_bn))
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), $(numKBlocks), $(1)) {
    bs_ptrs = bs_ptr + (((offs_k[:, None] + k * $(BLOCK_SIZE_K)) // $(group_size)) * $(stride_bsk) + offs_bn[None, :] * $(stride_bsn))
    bzp_ptrs = bzp_ptr + (((offs_k[:, None] + k * $(BLOCK_SIZE_K)) // $(group_size)) * $(stride_bzpk) + (offs_bn[None, :] // $(8)) * $(stride_bzpn))
    b_shift_bits = (offs_k[:, None] % $(8)) * $(4)
    bzp_shift_bits = (offs_bn[None, :] % $(8)) * $(4)
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    bs = tl.load(bs_ptrs)
    bzp = tl.load(bzp_ptrs)
    int_b = (b >> b_shift_bits) & $(15)
    int_bzp = (bzp >> bzp_shift_bits) & $(15)
    b = ((tl.cast(int_b, tl.int32) - tl.cast(int_bzp, tl.int32)) * bs).to(tl.float16)
    accumulator += tl.dot((a).to(tl.float16), (b).to(tl.float16))
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += ($(BLOCK_SIZE_K) * $(stride_bk) // $(8))
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = c_ptr + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}
```
</details>

<details><summary><code>i4AccSpec</code></summary>

```
/-- **The stored value.** `accumulator` after all `numKBlocks` K steps (the
kernel stores `c = accumulator.to(...)`, which erases to `accumulator`). -/
```
```lean
noncomputable def i4AccSpec (s : BlockState) (a bs : RegionName)
    (b bzp : Region .nat)
    (group_size stride_am stride_ak stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn BK numKBlocks : Nat) (row col : Nat) : ℝ :=
  ∑ k : Fin numKBlocks,
    i4AccStep s a bs b bzp group_size stride_am stride_ak stride_bk stride_bn
      stride_bsk stride_bsn stride_bzpk stride_bzpn BK row col k.val
```
</details>

<details><summary><code>firstPidM</code></summary>

```
/-- `first_pid_m = group_id * GROUP_SIZE_M`. -/
```
```lean
def firstPidM (s : BlockState) (N BN GM : Nat) : Nat := groupId s N BN GM * GM
```
</details>

<details><summary><code>groupSizeM</code></summary>

```
/-- `group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)`. -/
```
```lean
def groupSizeM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  min (numPidM M BM - firstPidM s N BN GM) GM
```
</details>

<details><summary><code>numPidInGroup</code></summary>

```
/-- `num_pid_in_group = GROUP_SIZE_M * num_pid_n`. -/
```
```lean
def numPidInGroup (N BN GM : Nat) : Nat := GM * numPidN N BN
```
</details>

<details><summary><code>i4AccStep</code></summary>

```
/-- One K step's contribution to output cell `(row, col)`. -/
```
```lean
noncomputable def i4AccStep (s : BlockState) (a bs : RegionName)
    (b bzp : Region .nat)
    (group_size stride_am stride_ak stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn BK : Nat) (row col k : Nat) : ℝ :=
  ∑ e : Fin BK,
    i4AElem s a stride_am stride_ak BK row k e.val
      * i4BDequant s bs b bzp group_size stride_bk stride_bn stride_bsk stride_bsn
          stride_bzpk stride_bzpn BK k e.val col
```
</details>

<details><summary><code>groupId</code></summary>

```
/-- `group_id = pid // num_pid_in_group`. -/
```
```lean
def groupId (s : BlockState) (N BN GM : Nat) : Nat :=
  s.pids 0 / numPidInGroup N BN GM
```
</details>

<details><summary><code>numPidM</code></summary>

```
/-- `num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)`. -/
```
```lean
def numPidM (M BM : Nat) : Nat := (M + BM - 1) / BM
```
</details>

<details><summary><code>numPidN</code></summary>

```
/-- `num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)`. -/
```
```lean
def numPidN (N BN : Nat) : Nat := (N + BN - 1) / BN
```
</details>

<details><summary><code>i4AElem</code></summary>

```
/-- `a[row, k*BK + e]` at K step `k`. Unmasked, matching the source's bare
`tl.load(a_ptrs)`. -/
```
```lean
noncomputable def i4AElem (s : BlockState) (a : RegionName)
    (stride_am stride_ak BK : Nat) (row k e : Nat) : ℝ :=
  s.readMem a (row * stride_am + (e + k * BK) * stride_ak)
```
</details>

<details><summary><code>i4BDequant</code></summary>

```
/-- The dequantized weight at `(k, e, col)`: the **signed** nibble difference
(`ℤ`-subtraction — this is what the `.int` hop buys; ℕ subtraction would
truncate), embedded into ℝ and scaled. -/
```
```lean
noncomputable def i4BDequant (s : BlockState) (bs : RegionName)
    (b bzp : Region .nat)
    (group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn BK : Nat) (k e col : Nat) : ℝ :=
  (((i4BNibble s b stride_bk stride_bn BK k e col : ℤ)
      - (bzpNibble s bzp group_size stride_bzpk stride_bzpn BK k e col : ℤ) : ℤ) : ℝ)
    * bsElem s bs group_size stride_bsk stride_bsn BK k e col
```
</details>

<details><summary><code>i4BNibble</code></summary>

```
/-- The unpacked 4-bit weight: `(word >> (e % 8) * 4) & 0xF`. -/
```
```lean
def i4BNibble (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BK : Nat) (k e col : Nat) : Nat :=
  i4BWord s b stride_bk stride_bn BK k e col >>> (e % 8 * 4) &&& 15
```
</details>

<details><summary><code>bzpNibble</code></summary>

```
/-- The unpacked 4-bit zero-point: `(word >> (col % 8) * 4) & 0xF`. -/
```
```lean
def bzpNibble (s : BlockState) (bzp : Region .nat)
    (group_size stride_bzpk stride_bzpn BK : Nat) (k e col : Nat) : Nat :=
  bzpWord s bzp group_size stride_bzpk stride_bzpn BK k e col
    >>> (col % 8 * 4) &&& 15
```
</details>

<details><summary><code>bsElem</code></summary>

```
/-- `bs[(e + k*BK) // group_size, col]` — the scale at lane `(k, e, col)`'s
group row. Per-lane: the group row varies **within** the `[BK, BN]` tile
whenever `group_size < BLOCK_SIZE_K`. -/
```
```lean
noncomputable def bsElem (s : BlockState) (bs : RegionName)
    (group_size stride_bsk stride_bsn BK : Nat) (k e col : Nat) : ℝ :=
  s.readMem bs ((e + k * BK) / group_size * stride_bsk + col * stride_bsn)
```
</details>

<details><summary><code>i4BWord</code></summary>

```
/-- The packed 32-bit word holding the weight nibble for `(k, e, col)`. Eight
weights share a word along K, hence the `e / 8`. -/
```
```lean
def i4BWord (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BK : Nat) (k e col : Nat) : Nat :=
  s.readMemValue .nat b ((e / 8 + k * (BK / 8)) * stride_bk + col * stride_bn)
```
</details>

<details><summary><code>bzpWord</code></summary>

```
/-- The packed word holding the zero-point nibble for `(k, e, col)`. Eight
zero-points share a word along N, hence the `col / 8`. -/
```
```lean
def bzpWord (s : BlockState) (bzp : Region .nat)
    (group_size stride_bzpk stride_bzpn BK : Nat) (k e col : Nat) : Nat :=
  s.readMemValue .nat bzp
    ((e + k * BK) / group_size * stride_bzpk + col / 8 * stride_bzpn)
```
</details>

## Public theorem: `matmul_dequantize_dequantize_exec_genuine`

<details><summary>docstring</summary>

```
/-- **Genuine, dimension-general correctness.** For every launch state, the
kernel runs to completion, and every in-range lane `(i, j)` of the program's
`[BK, BN]` output tile — absolute row `k = pid₀·BK + i < K`, absolute column
`n = pid₁·BN + j < N`, exactly the lanes the store mask lets through — holds
the dequantized weight

`fp_b[k, n] = ((dqBWord >> (k % 8)·4) & 0xF − (dqZpWord >> (n % 8)·4) & 0xF) · sc`

as an ℝ equation whose nibble difference is computed in ℤ, where `dqBWord` is
the packed word at `b[k / 8, n]`, `dqZpWord` the packed word at
`zp[k / group_size, n / 8]`, and `sc` the scale at `bs[k / group_size, n]` —
all read from the **launch** state's memory (`dequantSpec`).

Every dimension (`K`, `N`, `group_size`, all eight strides, `BK`, `BN`) and
both program ids are free; the only other hypothesis is `hInj` — distinct
tile lanes must land on distinct `fp_b` cells — which `fpbAddr_injective`
discharges for the host's row-major `fp_b`. No `hundef` is needed: all three
loads carry `other=`, and the store mask equals the load mask. -/
```
</details>

**Statement:**
```lean
specification matmul_dequantize_dequantize_exec_genuine
    (b_ptr : Region .nat) (b_scale_ptr : RegionName) (b_zp_ptr : Region .nat)
    (fpb_ptr : RegionName)
    (K N group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn stride_fpbk stride_fpbn : Nat)
    (BK BN : Nat) (s : BlockState)
    (hInj : Function.Injective
      (fun idx : TileIndex [BK, BN] =>
        fpbAddr stride_fpbk stride_fpbn BK BN (s.pids 0) (s.pids 1) idx)) :
    ∃ sF, exec ((matmul_dequantize_dequantize_surface b_ptr b_scale_ptr b_zp_ptr fpb_ptr
        K N group_size stride_bk stride_bn stride_bsk stride_bsn stride_bzpk
        stride_bzpn stride_fpbk stride_fpbn BK BN).toAlgKernel) s = some sF
      ∧ ∀ idx : TileIndex [BK, BN],
          (s.pids 0 * BK + idx.1.val < K ∧ s.pids 1 * BN + idx.2.1.val < N) →
          sF.readMem fpb_ptr
              (fpbAddr stride_fpbk stride_fpbn BK BN (s.pids 0) (s.pids 1) idx)
            = dequantSpec s b_scale_ptr b_ptr b_zp_ptr group_size stride_bk
                stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn
                (s.pids 0 * BK + idx.1.val) (s.pids 1 * BN + idx.2.1.val)
```

**Closed-form spec defs (transitive):** `fpbAddr`, `matmul_dequantize_dequantize_surface`, `dequantSpec`, `dqBNibble`, `dqZpNibble`, `dqScElem`, `dqBWord`, `dqZpWord`

<details><summary><code>fpbAddr</code></summary>

```
/-- The `fp_b` store address for tile cell `idx` of program `(p₀, p₁)`:
`(p₀·BK + i) · stride_fpbk + (p₁·BN + j) · stride_fpbn`. -/
```
```lean
def fpbAddr (stride_fpbk stride_fpbn BK BN p₀ p₁ : Nat)
    (idx : TileIndex [BK, BN]) : Nat :=
  (p₀ * BK + idx.1.val) * stride_fpbk + (p₁ * BN + idx.2.1.val) * stride_fpbn
```
</details>

<details><summary><code>matmul_dequantize_dequantize_surface</code></summary>

```
/-! ## Kernel surface (faithful transcription)

Allowed mechanical Lean-syntax-only changes: Python `BLOCK_SIZE_K:
tl.constexpr` / `BLOCK_SIZE_N: tl.constexpr` and the runtime scalar arguments
(`K`, `N`, `group_size`, the eight strides) become Lean `Nat` binders; the
packed-word regions are typed `Region .nat`. Everything else is the two
disclosed respells in the module preamble. -/
```
```lean
def matmul_dequantize_dequantize_surface
    (b_ptr : Region .nat) (b_scale_ptr : RegionName) (b_zp_ptr : Region .nat)
    (fpb_ptr : RegionName)
    (K N group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn stride_fpbk stride_fpbn : Nat)
    (BLOCK_SIZE_K BLOCK_SIZE_N : Nat) : ComputeKernel := triton {
  k_block_idx = tl.program_id(axis=0)
  n_block_idx = tl.program_id(axis=1)
  offs_k = k_block_idx * $(BLOCK_SIZE_K) + tl.arange(0, $(BLOCK_SIZE_K))
  offs_n = n_block_idx * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  fpb_offs = offs_k[:, None] * $(stride_fpbk) + offs_n[None, :] * $(stride_fpbn)
  b_offs = (offs_k[:, None] // $(8)) * $(stride_bk) + offs_n[None, :] * $(stride_bn)
  bzp_offs = (offs_k[:, None] // $(group_size)) * $(stride_bzpk) + (offs_n[None, :] // $(8)) * $(stride_bzpn)
  bs_offs = (offs_k[:, None] // $(group_size)) * $(stride_bsk) + offs_n[None, :] * $(stride_bsn)
  n_mask = offs_n[None, :] < $(N)
  k_mask = offs_k[:, None] < $(K)
  mask = n_mask & k_mask
  int32_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
  zp_b = tl.load(b_zp_ptr + bzp_offs, mask=mask, other=0.0)
  scale_b = tl.load(b_scale_ptr + bs_offs, mask=mask, other=0.0)
  b_shift = (offs_k[:, None] % $(8)) * $(4)
  bzp_shift = (offs_n[None, :] % $(8)) * $(4)
  fp_weight = (tl.cast((int32_b >> b_shift) & $(15), tl.int32) - tl.cast((zp_b >> bzp_shift) & $(15), tl.int32)) * scale_b
  tl.store(fpb_ptr + fpb_offs, fp_weight, mask=mask)
}
```
</details>

<details><summary><code>dequantSpec</code></summary>

```
/-- **The stored value** at absolute cell `(k, n)`: the **signed** nibble
difference (ℤ-subtraction — this is what the `.int` hop of disclosure (1)
buys; ℕ subtraction would truncate), embedded into ℝ and scaled. -/
```
```lean
noncomputable def dequantSpec (s : BlockState) (b_scale_ptr : RegionName)
    (b_ptr b_zp_ptr : Region .nat)
    (group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn : Nat) (k n : Nat) : ℝ :=
  (((dqBNibble s b_ptr stride_bk stride_bn k n : ℤ)
      - (dqZpNibble s b_zp_ptr group_size stride_bzpk stride_bzpn k n : ℤ) : ℤ) : ℝ)
    * dqScElem s b_scale_ptr group_size stride_bsk stride_bsn k n
```
</details>

<details><summary><code>dqBNibble</code></summary>

```
/-- The unpacked 4-bit weight: `(dqBWord >> (k % 8) * 4) & 0xF`. -/
```
```lean
def dqBNibble (s : BlockState) (b_ptr : Region .nat)
    (stride_bk stride_bn k n : Nat) : Nat :=
  dqBWord s b_ptr stride_bk stride_bn k n >>> (k % 8 * 4) &&& 15
```
</details>

<details><summary><code>dqZpNibble</code></summary>

```
/-- The unpacked 4-bit zero-point: `(dqZpWord >> (n % 8) * 4) & 0xF`. -/
```
```lean
def dqZpNibble (s : BlockState) (b_zp_ptr : Region .nat)
    (group_size stride_bzpk stride_bzpn k n : Nat) : Nat :=
  dqZpWord s b_zp_ptr group_size stride_bzpk stride_bzpn k n >>> (n % 8 * 4) &&& 15
```
</details>

<details><summary><code>dqScElem</code></summary>

```
/-- The scale at absolute cell `(k, n)`: `bs[k // group_size, n]`. -/
```
```lean
noncomputable def dqScElem (s : BlockState) (b_scale_ptr : RegionName)
    (group_size stride_bsk stride_bsn k n : Nat) : ℝ :=
  s.readMem b_scale_ptr (k / group_size * stride_bsk + n * stride_bsn)
```
</details>

<details><summary><code>dqBWord</code></summary>

```
/-- The packed 32-bit word holding the weight nibble for absolute cell
`(k, n)`: `b[k // 8, n]`. Eight weights share a word along K, hence `k / 8`. -/
```
```lean
def dqBWord (s : BlockState) (b_ptr : Region .nat)
    (stride_bk stride_bn k n : Nat) : Nat :=
  s.readMemValue .nat b_ptr.cast (k / 8 * stride_bk + n * stride_bn)
```
</details>

<details><summary><code>dqZpWord</code></summary>

```
/-- The packed word holding the zero-point nibble for absolute cell `(k, n)`:
`zp[k // group_size, n // 8]`. Eight zero-points share a word along N, hence
`n / 8`; one group row per `group_size` K rows. -/
```
```lean
def dqZpWord (s : BlockState) (b_zp_ptr : Region .nat)
    (group_size stride_bzpk stride_bzpn k n : Nat) : Nat :=
  s.readMemValue .nat b_zp_ptr.cast
    (k / group_size * stride_bzpk + n / 8 * stride_bzpn)
```
</details>
