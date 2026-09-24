# Spec sheet — `bench/tritonbench_g/int4_matmul/Int4Matmul.lean`

**Python source:** `bench/tritonbench_g/int4_matmul/int4_matmul.py`

## Public theorem: `int4_matmul_exec_genuine`

<details><summary>docstring</summary>

```
/-- **Genuine, dimension-general correctness.** For every launch state, the
`SPLIT_K = 1` arm of the kernel runs to completion, and every in-range output
lane of `C` holds `accSpec`: the sum over all `numKBlocks` K steps of `tl.dot`
between the loaded `A` tile and the signed-dequantized weight tile
`(nib(B) − nib(BZP)) · BS`, with per-lane group rows `(e + k·BK) / group_size`.

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
specification int4_matmul_exec_genuine
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
    ∃ sF, exec (int4_matmul_surface a_ptr c_ptr bs_ptr b_ptr bzp_ptr
        M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        stride_bsk stride_bsn stride_bzpk stride_bzpn group_size
        BM BN BK GM numKBlocks).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (pidM s M N BM BN GM * BM + idx.1.val < M
            ∧ pidN s M N BM BN GM * BN + idx.2.1.val < N) →
          sF.readMem c_ptr (cAddr stride_cm stride_cn BM BN (pidM s M N BM BN GM)
              (pidN s M N BM BN GM) idx)
            = accSpec s a_ptr bs_ptr b_ptr bzp_ptr group_size stride_am
                stride_ak stride_bk stride_bn stride_bsk stride_bsn
                stride_bzpk stride_bzpn BK numKBlocks
                (pidM s M N BM BN GM * BM + idx.1.val)
                (pidN s M N BM BN GM * BN + idx.2.1.val)
```

**Assumptions / layout contracts:**
- `hK : K = BK * numKBlocks`
- `hBK8 : BK % 8 = 0`
- `hpid1 : s.pids 1 = 0`

**Closed-form spec defs (transitive):** `cAddr`, `pidM`, `pidN`, `int4_matmul_surface`, `accSpec`, `firstPidM`, `groupSizeM`, `numPidInGroup`, `accStep`, `groupId`, `numPidM`, `numPidN`, `aElem`, `bDequant`, `bNibble`, `bzpNibble`, `bsElem`, `bWord`, `bzpWord`

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

<details><summary><code>int4_matmul_surface</code></summary>

```
/-! ## Kernel surface (faithful transcription, `SPLIT_K = 1` arm) -/
```
```lean
def int4_matmul_surface
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
    b = (tl.cast(int_b, tl.int32) - tl.cast(int_bzp, tl.int32)) * bs
    accumulator += tl.dot(a, (b).to(a.dtype))
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += ($(BLOCK_SIZE_K) * $(stride_bk) // $(8))
  }
  c = (accumulator).to(c_ptr.dtype.element_ty)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = c_ptr + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}
```
</details>

<details><summary><code>accSpec</code></summary>

```
/-- **The stored value.** `accumulator` after all `numKBlocks` K steps (the
kernel stores `c = accumulator.to(...)`, which erases to `accumulator`). -/
```
```lean
noncomputable def accSpec (s : BlockState) (a bs : RegionName)
    (b bzp : Region .nat)
    (group_size stride_am stride_ak stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn BK numKBlocks : Nat) (row col : Nat) : ℝ :=
  ∑ k : Fin numKBlocks,
    accStep s a bs b bzp group_size stride_am stride_ak stride_bk stride_bn
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

<details><summary><code>accStep</code></summary>

```
/-- One K step's contribution to output cell `(row, col)`. -/
```
```lean
noncomputable def accStep (s : BlockState) (a bs : RegionName)
    (b bzp : Region .nat)
    (group_size stride_am stride_ak stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn BK : Nat) (row col k : Nat) : ℝ :=
  ∑ e : Fin BK,
    aElem s a stride_am stride_ak BK row k e.val
      * bDequant s bs b bzp group_size stride_bk stride_bn stride_bsk stride_bsn
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

<details><summary><code>aElem</code></summary>

```
/-- `a[row, k*BK + e]` at K step `k`. Unmasked, matching the source's bare
`tl.load(a_ptrs)`. -/
```
```lean
noncomputable def aElem (s : BlockState) (a : RegionName)
    (stride_am stride_ak BK : Nat) (row k e : Nat) : ℝ :=
  s.readMem a (row * stride_am + (e + k * BK) * stride_ak)
```
</details>

<details><summary><code>bDequant</code></summary>

```
/-- The dequantized weight at `(k, e, col)`: the **signed** nibble difference
(`ℤ`-subtraction — this is what the `.int` hop buys; ℕ subtraction would
truncate), embedded into ℝ and scaled. -/
```
```lean
noncomputable def bDequant (s : BlockState) (bs : RegionName)
    (b bzp : Region .nat)
    (group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn BK : Nat) (k e col : Nat) : ℝ :=
  (((bNibble s b stride_bk stride_bn BK k e col : ℤ)
      - (bzpNibble s bzp group_size stride_bzpk stride_bzpn BK k e col : ℤ) : ℤ) : ℝ)
    * bsElem s bs group_size stride_bsk stride_bsn BK k e col
```
</details>

<details><summary><code>bNibble</code></summary>

```
/-- The unpacked 4-bit weight: `(word >> (e % 8) * 4) & 0xF`. -/
```
```lean
def bNibble (s : BlockState) (b : Region .nat)
    (stride_bk stride_bn BK : Nat) (k e col : Nat) : Nat :=
  bWord s b stride_bk stride_bn BK k e col >>> (e % 8 * 4) &&& 15
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

<details><summary><code>bWord</code></summary>

```
/-- The packed 32-bit word holding the weight nibble for `(k, e, col)`. Eight
weights share a word along K, hence the `e / 8`. -/
```
```lean
def bWord (s : BlockState) (b : Region .nat)
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
