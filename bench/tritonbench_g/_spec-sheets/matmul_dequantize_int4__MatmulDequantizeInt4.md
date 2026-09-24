# Spec sheet — `bench/tritonbench_g/matmul_dequantize_int4/MatmulDequantizeInt4.lean`

**Python source:** `bench/tritonbench_g/matmul_dequantize_int4/matmul_dequantize_int4.py`

## Public theorem: `matmul_dequantize_int4_exec_genuine`

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
specification matmul_dequantize_int4_exec_genuine
    (a_ptr c_ptr scales_ptr : RegionName) (b_ptr zeros_ptr : Region .nat)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      groupsize : Nat) (NO_GROUPS : Bool) (BM BN BK GM : Nat) (s : BlockState)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN
        (pidM s M N BM BN GM) (pidN s M N BM BN GM) i)) :
    ∃ sF, exec (matmul_dequantize_int4_surface a_ptr c_ptr scales_ptr b_ptr zeros_ptr
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

**Closed-form spec defs (transitive):** `cAddr`, `pidM`, `pidN`, `matmul_dequantize_int4_surface`, `accSpec`, `firstPidM`, `groupSizeM`, `numPidInGroup`, `numPidK`, `accStep`, `groupId`, `numPidM`, `numPidN`, `aElem`, `bDequant`, `bNibble`, `scalesElem`, `groupRow`, `zeroScaled`, `bWord`, `zerosNibble`, `zerosWord`

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

<details><summary><code>matmul_dequantize_int4_surface</code></summary>

```
/-! ## Kernel surface (faithful transcription) -/
```
```lean
def matmul_dequantize_int4_surface
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
    accumulator += tl.dot(a, (b).to(a.dtype))
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += ($(BLOCK_SIZE_K) // infearure_per_bits) * $(stride_bk)
  }
  c = (accumulator).to(c_ptr.dtype.element_ty)
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
/-- `num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)` — the K-loop trip count. -/
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
