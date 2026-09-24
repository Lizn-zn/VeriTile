# Spec sheet — `bench/tritonbench_g/int8_matmul_kernel/Int8MatmulKernel.lean`

**Python source:** `bench/tritonbench_g/int8_matmul_kernel/int8_matmul_kernel.py`

## Public theorem: `int8_matmul_kernel_exec_genuine`

<details><summary>docstring</summary>

```
/-- **Genuine, dimension-general correctness.** For every launch state
satisfying the source's own `tl.static_assert` (`hK`: `K` is divisible by
`4·BLOCK_SIZE_K`, presented as `K = 4·(BK·numKBlocks)` with `numKBlocks`
the inner trip count `tl.cdiv(K//4, BK)`), the kernel runs to completion
and every in-range output lane of `C` holds the `.int` memory cell carrying
the exact ℤ packed-weight matmul

`imSpec = ∑ i<4, ∑ kk<K/4, A[row, i·(K/4)+kk] · (bits_i(B[kk, col]) − 1)`

— the four 2-bit fields of each packed byte of `B`, shifted into
`{−1,0,1,2}` by the `tensor_full` subtraction, dotted against the
continuously-swept columns of `A` (`Op.dotInt` over all `4·numKBlocks`
inner steps). No rounding model appears anywhere: the kernel is
pure-integer, and the store readback is stated at the `MemCell` level
(`MemCell.of .int`).

`hInj` says distinct output lanes get distinct `C` addresses —
`imCAddr_injective` discharges it for a row-major `C`. The `% M` / `% N`
offset wraps disappear on exactly the lanes the store mask lets through
(`Nat.mod_eq_of_lt`). The grid is 1-D, so there is no auxiliary
program-id hypothesis. -/
```
</details>

**Statement:**
```lean
specification int8_matmul_kernel_exec_genuine
    (A : Region .int) (B : Region .nat) (C : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BM BN BK GM numKBlocks : Nat) (s : BlockState)
    (hK : K = 4 * (BK * numKBlocks))
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => imCAddr stride_cm stride_cn BM BN
        (pidM s M N BM BN GM) (pidN s M N BM BN GM) i)) :
    ∃ sF, exec (int8_matmul_kernel_surface A B C M N K
        stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        BM BN BK GM numKBlocks).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (pidM s M N BM BN GM * BM + idx.1.val < M
            ∧ pidN s M N BM BN GM * BN + idx.2.1.val < N) →
          sF.mem C (imCAddr stride_cm stride_cn BM BN (pidM s M N BM BN GM)
              (pidN s M N BM BN GM) idx)
            = MemCell.of .int (imSpec s A B stride_am stride_ak stride_bk
                stride_bn BK numKBlocks
                (pidM s M N BM BN GM * BM + idx.1.val)
                (pidN s M N BM BN GM * BN + idx.2.1.val))
```

**Assumptions / layout contracts:**
- `hK : K = 4 * (BK * numKBlocks)`

**Closed-form spec defs (transitive):** `imCAddr`, `pidM`, `pidN`, `int8_matmul_kernel_surface`, `imSpec`, `firstPidM`, `numPidInGroup`, `groupSizeM`, `aElem`, `bBits`, `bWord`, `groupId`, `numPidN`, `numPidM`

<details><summary><code>imCAddr</code></summary>

```
/-- The `C` store address for output lane `(r, c)` — the kernel's own
const-first order `stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]`. -/
```
```lean
def imCAddr (stride_cm stride_cn BM BN pm pn : Nat)
    (idx : TileIndex [BM, BN]) : Nat :=
  stride_cm * (pm * BM + idx.1.val) + stride_cn * (pn * BN + idx.2.1.val)
```
</details>

<details><summary><code>pidM</code></summary>

```
/-- `pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)` —
note the extra `% num_pid_in_group` vs the family's usual
`pid % group_size_m`; spelled faithfully. -/
```
```lean
def pidM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  firstPidM s N BN GM
    + s.pids 0 % numPidInGroup N BN GM % groupSizeM s M N BM BN GM
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

<details><summary><code>int8_matmul_kernel_surface</code></summary>

```lean
def int8_matmul_kernel_surface
    (A : Region .int) (B : Region .nat) (C : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat) :
    ComputeKernel := triton {
  tl.static_assert($(K) % ($(4) * $(BLOCK_SIZE_K)) == 0, "K / 4 must be divisible by BLOCK_SIZE_K => K divisible by 4*BLOCK_SIZE_K")
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
  a_ptrs = $((A : Region .int)) + (offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak))
  b_ptrs = $((B : Region .nat)) + (offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn))
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.int32)
  for i in range($(0), $(4), $(1)) {
    b_ptrs = $((B : Region .nat)) + (offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn))
    for j in range($(0), $(numKBlocks), $(1)) {
      k = i * $(numKBlocks) + j
      a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K), other=0).to(tl.int8)
      b_uint8 = tl.load(b_ptrs, mask=offs_k[:, None] < $(K), other=0)
      mask = $(3) << ($(2) * i)
      b = ((b_uint8 & mask) >> ($(2) * i)).to(tl.int8)
      tensor_full = tl.full([1], 1, dtype=tl.int8)
      accumulator += tl.dot(a, (b - tensor_full), out_dtype=tl.int32)
      a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
      b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
    }
  }
  c = accumulator
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}
```
</details>

<details><summary><code>imSpec</code></summary>

```
/-- **The stored value**: the packed-layout double sum
`∑ i<4, ∑ kk<K/4, A[row, i·(K/4)+kk] · (bits_i(B[kk, col]) − 1)` over ℤ,
with `K/4 = numKBlocks · BK` (the `hK` side condition). `A`'s column index
`i·(K/4) + kk` sweeps `0..K-1` in order — the continuously-advancing
`a_ptrs`; `B`'s row index `kk` re-sweeps the packed rows once per field. -/
```
```lean
def imSpec (s : BlockState) (A : Region .int) (B : Region .nat)
    (stride_am stride_ak stride_bk stride_bn BK numKBlocks : Nat)
    (row col : Nat) : ℤ :=
  ∑ i : Fin 4, ∑ kk : Fin (numKBlocks * BK),
    aElem s A stride_am stride_ak row (i.val * (numKBlocks * BK) + kk.val)
      * ((bBits i.val (bWord s B stride_bk stride_bn kk.val col) : ℤ) - 1)
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

<details><summary><code>numPidInGroup</code></summary>

```
/-- `num_pid_in_group = GROUP_SIZE_M * num_pid_n`. -/
```
```lean
def numPidInGroup (N BN GM : Nat) : Nat := GM * numPidN N BN
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

<details><summary><code>aElem</code></summary>

```
/-- `A[row, kg]` — a signed `.int`-channel read (the host's `torch.int32`
input; widths are erased, see disclosure (3)). -/
```
```lean
def aElem (s : BlockState) (A : Region .int)
    (stride_am stride_ak : Nat) (row kg : Nat) : ℤ :=
  s.readMemValue .int (Region.cast A) (row * stride_am + kg * stride_ak)
```
</details>

<details><summary><code>bBits</code></summary>

```
/-- `bits_i(w) = (w >>> (2·i)) &&& 3` — the 2-bit field `i` of a packed
byte, a value in `{0, 1, 2, 3}`. -/
```
```lean
def bBits (i w : Nat) : ℕ := w >>> (2 * i) &&& 3
```
</details>

<details><summary><code>bWord</code></summary>

```
/-- `B[kk, col]` — the packed `torch.uint8` byte holding four 2-bit weights,
read on the `.nat` channel (`kk` ranges over the packed rows `0..K/4-1`). -/
```
```lean
def bWord (s : BlockState) (B : Region .nat)
    (stride_bk stride_bn : Nat) (kk col : Nat) : ℕ :=
  s.readMemValue .nat (Region.cast B) (kk * stride_bk + col * stride_bn)
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

<details><summary><code>numPidN</code></summary>

```
/-- `num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)`. -/
```
```lean
def numPidN (N BN : Nat) : Nat := (N + BN - 1) / BN
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
