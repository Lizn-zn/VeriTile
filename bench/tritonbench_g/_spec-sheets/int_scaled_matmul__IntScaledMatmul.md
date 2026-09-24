# Spec sheet — `bench/tritonbench_g/int_scaled_matmul/IntScaledMatmul.lean`

**Python source:** `bench/tritonbench_g/int_scaled_matmul/int_scaled_matmul.py`

## Public theorem: `int_scaled_matmul_matmul_exec_genuine`

<details><summary>docstring</summary>

```
/-- **Genuine, dimension-general correctness of
`matmul_kernel_with_block_pointers`.** For every launch state, the kernel
runs to completion and every in-range output lane of `C` holds the `.int`
memory cell carrying the exact ℤ GEMM

`mmSpec = ∑ kk < K, A[row, kk] · B[kk, col]`

at the block-pointer address `row·stride_cm + col·stride_cn`. No
divisibility of `K` is assumed: the `boundary_check=(0, 1)` loads zero-fill
every lane past `K` (and past `M`/`K` in the row axes), so the ragged tail
contributes nothing — the only loop hypothesis is `0 < BLOCK_K` (a
`range(0, K, BLOCK_K)` step must be positive). `hInj` says distinct output
lanes get distinct `C` addresses — `mmCAddr_injective_rowMajor` discharges
it for a row-major `C`. The grid is 1-D, so there is no auxiliary
program-id hypothesis. -/
```
</details>

**Statement:**
```lean
specification int_scaled_matmul_matmul_exec_genuine
    (a_ptr b_ptr : Region .int) (c_ptr : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BM BN BK GM : Nat) (s : BlockState)
    (hBK : 0 < BK)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => mmCAddr stride_cm stride_cn BM BN
        (mmPidM s M N BM BN GM) (mmPidN s M N BM BN GM) i)) :
    ∃ sF, exec (int_scaled_matmul_matmul_surface a_ptr b_ptr c_ptr M N K
        stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        BM BN BK GM).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (mmPidM s M N BM BN GM * BM + idx.1.val < M
            ∧ mmPidN s M N BM BN GM * BN + idx.2.1.val < N) →
          sF.mem c_ptr (mmCAddr stride_cm stride_cn BM BN (mmPidM s M N BM BN GM)
              (mmPidN s M N BM BN GM) idx)
            = MemCell.of .int (mmSpec s a_ptr b_ptr K stride_am stride_ak
                stride_bk stride_bn
                (mmPidM s M N BM BN GM * BM + idx.1.val)
                (mmPidN s M N BM BN GM * BN + idx.2.1.val))
```

**Assumptions / layout contracts:**
- `hBK : 0 < BK`

**Closed-form spec defs (transitive):** `mmCAddr`, `mmPidM`, `mmPidN`, `int_scaled_matmul_matmul_surface`, `mmSpec`, `mmFirstPidM`, `mmGroupSize`, `mmNumPidInGroup`, `mmAElem`, `mmBElem`, `mmGroupId`, `mmNumPidM`, `mmNumPidN`

<details><summary><code>mmCAddr</code></summary>

```
/-- The `C` store address for output lane `idx` — the `c_block_ptr` address
(base offset `0`, offsets `(pid_m·BM, pid_n·BN)`). -/
```
```lean
def mmCAddr (scm scn BM BN pm pn : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  (pm * BM + idx.1.val) * scm + (pn * BN + idx.2.1.val) * scn
```
</details>

<details><summary><code>mmPidM</code></summary>

```
/-- `pid_m = first_pid_m + (pid % GROUP_M)` — `GROUP_M` here is the rebound
register (the group size), not the parameter. -/
```
```lean
def mmPidM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  mmFirstPidM s N BN GM + s.pids 0 % mmGroupSize s M N BM BN GM
```
</details>

<details><summary><code>mmPidN</code></summary>

```
/-- `pid_n = (pid % num_pid_in_group) // GROUP_M` (the rebound register). -/
```
```lean
def mmPidN (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  s.pids 0 % mmNumPidInGroup N BN GM / mmGroupSize s M N BM BN GM
```
</details>

<details><summary><code>int_scaled_matmul_matmul_surface</code></summary>

```
/-- Faithful transcription of `matmul_kernel_with_block_pointers` (the
launcher `int_matmul_kernel`'s target). See the module docstring's kernel-1
blocker for the four disclosed respells (`dtype=` pins on the block-ptr
loads, tuple→bracket shape/delta lists, the `GROUP_M` register rebind, and
the erased `order` tuples). -/
```
```lean
def int_scaled_matmul_matmul_surface
    (a_ptr b_ptr : Region .int) (c_ptr : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_N))
  num_pid_in_group = $(GROUP_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_M)
  GROUP_M = min(num_pid_m - first_pid_m, $(GROUP_M))
  pid_m = first_pid_m + (pid % GROUP_M)
  pid_n = (pid % num_pid_in_group) // GROUP_M
  a_block_ptr = tl.make_block_ptr(base=$((a_ptr : Region .int)), shape=($(M), $(K)),
    strides=($(stride_am), $(stride_ak)), offsets=(pid_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_K)), order=(1, 0))
  b_block_ptr = tl.make_block_ptr(base=$((b_ptr : Region .int)), shape=($(K), $(N)),
    strides=($(stride_bk), $(stride_bn)), offsets=(0, pid_n * $(BLOCK_N)),
    block_shape=($(BLOCK_K), $(BLOCK_N)), order=(1, 0))
  accumulator = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.int32)
  for k in range(0, $(K), $(BLOCK_K)) {
    a = tl.load(a_block_ptr, boundary_check=(0, 1))
    b = tl.load(b_block_ptr, boundary_check=(0, 1))
    accumulator += tl.dot(a, b)
    a_block_ptr = tl.advance(a_block_ptr, [$(0), $(BLOCK_K)])
    b_block_ptr = tl.advance(b_block_ptr, [$(BLOCK_K), $(0)])
  }
  c = accumulator
  c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=($(M), $(N)),
    strides=($(stride_cm), $(stride_cn)),
    offsets=(pid_m * $(BLOCK_M), pid_n * $(BLOCK_N)),
    block_shape=($(BLOCK_M), $(BLOCK_N)), order=(1, 0))
  tl.store(c_block_ptr, c, boundary_check=(0, 1))
}
```
</details>

<details><summary><code>mmSpec</code></summary>

```
/-- **The kernel-1 stored value**: the genuine ℤ GEMM
`∑ kk < K, A[row, kk] · B[kk, col]`. -/
```
```lean
def mmSpec (s : BlockState) (a_ptr b_ptr : Region .int)
    (K sam sak sbk sbn : Nat) (row col : Nat) : ℤ :=
  ∑ kk : Fin K, mmAElem s a_ptr sam sak row kk.val
    * mmBElem s b_ptr sbk sbn kk.val col
```
</details>

<details><summary><code>mmFirstPidM</code></summary>

```
/-- `first_pid_m = group_id * GROUP_M`. -/
```
```lean
def mmFirstPidM (s : BlockState) (N BN GM : Nat) : Nat :=
  mmGroupId s N BN GM * GM
```
</details>

<details><summary><code>mmGroupSize</code></summary>

```
/-- The rebound register `GROUP_M = min(num_pid_m - first_pid_m, GROUP_M)`. -/
```
```lean
def mmGroupSize (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  min (mmNumPidM M BM - mmFirstPidM s N BN GM) GM
```
</details>

<details><summary><code>mmNumPidInGroup</code></summary>

```
/-- `num_pid_in_group = GROUP_M * num_pid_n`. -/
```
```lean
def mmNumPidInGroup (N BN GM : Nat) : Nat := GM * mmNumPidN N BN
```
</details>

<details><summary><code>mmAElem</code></summary>

```
/-- `A[row, kk]` — a signed `.int`-channel read of the launch state (the
host's `torch.int8` input; widths erased, kernel-1 disclosure (1)). -/
```
```lean
def mmAElem (s : BlockState) (a_ptr : Region .int) (sam sak : Nat)
    (row kk : Nat) : ℤ :=
  s.readMemValue .int (Region.cast a_ptr) (row * sam + kk * sak)
```
</details>

<details><summary><code>mmBElem</code></summary>

```
/-- `B[kk, col]` — same on `B`. -/
```
```lean
def mmBElem (s : BlockState) (b_ptr : Region .int) (sbk sbn : Nat)
    (kk col : Nat) : ℤ :=
  s.readMemValue .int (Region.cast b_ptr) (kk * sbk + col * sbn)
```
</details>

<details><summary><code>mmGroupId</code></summary>

```
/-- `group_id = pid // num_pid_in_group`. -/
```
```lean
def mmGroupId (s : BlockState) (N BN GM : Nat) : Nat :=
  s.pids 0 / mmNumPidInGroup N BN GM
```
</details>

<details><summary><code>mmNumPidM</code></summary>

```
/-- `num_pid_m = tl.cdiv(M, BLOCK_M)`. -/
```
```lean
def mmNumPidM (M BM : Nat) : Nat := (M + BM - 1) / BM
```
</details>

<details><summary><code>mmNumPidN</code></summary>

```
/-- `num_pid_n = tl.cdiv(N, BLOCK_N)`. -/
```
```lean
def mmNumPidN (N BN : Nat) : Nat := (N + BN - 1) / BN
```
</details>

## Public theorem: `int_scaled_matmul_scaled_exec_genuine`

<details><summary>docstring</summary>

```
/-- **Genuine, dimension-general correctness of
`scaled_matmul_kernel_with_block_pointers`.** For every launch state and
both `EVEN_K` arms, the kernel runs to completion and every in-range output
lane of `C` holds the **`.real`-typed** memory cell carrying the exact real

`(∑ kk < K, A[row, kk] · B[kk, col] : ℤ → ℝ) · s1[row]`

at the flat inductor address `col + N·row` (`xindex = idx_n + N·idx_m`).
The `s1` scale is read at the strideless address `row` — the
`tl.broadcast_to(idx_m, …)` load; `stride_s1m`/`stride_s1n` are dead. The
host allocates `C` as int32: the float→int32 container truncation at the
store boundary is outside the model (the #154 fixed-width family — the cell
carries the exact real; see the module docstring).

Loop hypotheses: `hCeil` pins `numKBlocks` as an upper trip count covering
all of `K` (`K ≤ numKBlocks·BLOCK_K`, the ceil form — extra iterations are
all-masked and contribute nothing in the masked arm), and `hEven` demands
the exact form `K = numKBlocks·BLOCK_K` **only** of the unmasked
`EVEN_K = true` arm, whose loads have no masks to protect a ragged tail.
No `hInj` hypothesis: on the masked lanes `col < N`, so the flat address is
injective outright (`smCAddr_inj_active`). The `% M`/`% N` wraps of
`ram`/`rbn` disappear on exactly the lanes the store mask lets through
(`Nat.mod_eq_of_lt`). -/
```
</details>

**Statement:**
```lean
specification int_scaled_matmul_scaled_exec_genuine
    (a_ptr b_ptr : Region .int) (c_ptr s1_ptr : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_s1m stride_s1n : Nat)
    (BM BN BK GM : Nat) (EVEN_K : Bool) (numKBlocks : Nat) (s : BlockState)
    (hCeil : K ≤ numKBlocks * BK)
    (hEven : EVEN_K = Bool.true → K = numKBlocks * BK) :
    ∃ sF, exec (int_scaled_matmul_scaled_surface a_ptr b_ptr c_ptr s1_ptr M N K
        stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        stride_s1m stride_s1n BM BN BK GM EVEN_K numKBlocks).toAlgKernel s
        = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (smPidM s M N BM BN GM * BM + idx.1.val < M
            ∧ smPidN s M N BM BN GM * BN + idx.2.1.val < N) →
          sF.mem c_ptr (smCAddr N BM BN (smPidM s M N BM BN GM)
              (smPidN s M N BM BN GM) idx)
            = MemCell.of .real (some
                (((smSpec s a_ptr b_ptr K stride_am stride_ak stride_bk
                    stride_bn
                    (smPidM s M N BM BN GM * BM + idx.1.val)
                    (smPidN s M N BM BN GM * BN + idx.2.1.val) : ℤ) : ℝ)
                  * smS1 s s1_ptr (smPidM s M N BM BN GM * BM + idx.1.val)))
```

**Assumptions / layout contracts:**
- `hCeil : K ≤ numKBlocks * BK`
- `hEven : EVEN_K = Bool.true → K = numKBlocks * BK`

**Closed-form spec defs (transitive):** `int_scaled_matmul_scaled_surface`, `smPidM`, `smPidN`, `smCAddr`, `smSpec`, `smS1`, `smGroupId`, `smGroupSize`, `smWidth`, `smAElem`, `smBElem`, `smGridM`, `smGridN`

<details><summary><code>int_scaled_matmul_scaled_surface</code></summary>

```
/-- Faithful transcription of `scaled_matmul_kernel_with_block_pointers`
(the launcher `int_scaled_matmul_kernel`'s target; a classic pointer GEMM —
no block pointers despite the name — with the inductor store suffix). See
the module docstring's kernel-2 blocker for the five disclosed respells
(ascending change of variable with `k = K - j·BLOCK_K`, the fixed
`ACC_TYPE = tl.int32` constexpr, the `mask.shape` → literal-dims respell in
`tl.broadcast_to`, the erased `eviction_policy` hint, and the erased int
widths). `stride_s1m`/`stride_s1n` are passed by the launcher but dead in
the body (kept as unused binders). -/
```
```lean
def int_scaled_matmul_scaled_surface
    (a_ptr b_ptr : Region .int) (c_ptr s1_ptr : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_s1m stride_s1n : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M : Nat) (EVEN_K : Bool)
    (numKBlocks : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  grid_m = ($((M : Nat)) + $(BLOCK_M) - $(1)) // $(BLOCK_M)
  grid_n = ($((N : Nat)) + $(BLOCK_N) - $(1)) // $(BLOCK_N)
  width = $(GROUP_M) * grid_n
  group_id = pid // width
  group_size = min(grid_m - group_id * $(GROUP_M), $(GROUP_M))
  pid_m = group_id * $(GROUP_M) + (pid % group_size)
  pid_n = (pid % width) // (group_size)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  ram = tl.max_contiguous(tl.multiple_of(rm % $(M), $(BLOCK_M)), $(BLOCK_M))
  rbn = tl.max_contiguous(tl.multiple_of(rn % $(N), $(BLOCK_N)), $(BLOCK_N))
  rk = tl.arange(0, $(BLOCK_K))
  A = $((a_ptr : Region .int)) + (ram[:, None] * $(stride_am) + rk[None, :] * $(stride_ak))
  B = $((b_ptr : Region .int)) + (rk[:, None] * $(stride_bk) + rbn[None, :] * $(stride_bn))
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.int32)
  for j in range(0, $(numKBlocks), $(1)) {
    k = $(K) - j * $(BLOCK_K)
    if EVEN_K {
      a = tl.load(A)
      b = tl.load(B)
    } else {
      a = tl.load(A, mask=rk[None, :] < k, other=0.0)
      b = tl.load(B, mask=rk[:, None] < k, other=0.0)
    }
    acc += tl.dot(a, b)
    A += $(BLOCK_K) * $(stride_ak)
    B += $(BLOCK_K) * $(stride_bk)
  }
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  idx_m = rm[:, None]
  idx_n = rn[None, :]
  mask = (idx_m < $(M)) & (idx_n < $(N))
  xindex = idx_n + ($(N) * idx_m)
  tmp0 = tl.load(s1_ptr + (tl.broadcast_to(idx_m, [$(BLOCK_M), $(BLOCK_N)])), mask, eviction_policy="evict_last")
  tl.store(c_ptr + (tl.broadcast_to(xindex, [$(BLOCK_M), $(BLOCK_N)])), acc * tmp0, mask)
}
```
</details>

<details><summary><code>smPidM</code></summary>

```
/-- `pid_m = group_id * GROUP_M + (pid % group_size)`. -/
```
```lean
def smPidM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  smGroupId s N BN GM * GM + s.pids 0 % smGroupSize s M N BM BN GM
```
</details>

<details><summary><code>smPidN</code></summary>

```
/-- `pid_n = (pid % width) // group_size`. -/
```
```lean
def smPidN (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  s.pids 0 % smWidth N BN GM / smGroupSize s M N BM BN GM
```
</details>

<details><summary><code>smCAddr</code></summary>

```
/-- The flat inductor store address `col + N·row` (absolute, unwrapped
coordinates). -/
```
```lean
def smCAddr (N BM BN pm pn : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  pn * BN + idx.2.1.val + N * (pm * BM + idx.1.val)
```
</details>

<details><summary><code>smSpec</code></summary>

```
/-- **The kernel-2 integer accumulator value**: the genuine ℤ GEMM. -/
```
```lean
def smSpec (s : BlockState) (a_ptr b_ptr : Region .int)
    (K sam sak sbk sbn : Nat) (row col : Nat) : ℤ :=
  ∑ kk : Fin K, smAElem s a_ptr sam sak row kk.val
    * smBElem s b_ptr sbk sbn kk.val col
```
</details>

<details><summary><code>smS1</code></summary>

```
/-- `s1[row]` — the per-row fp32 scale, read at the **strideless** address
`row` (the `tl.broadcast_to(idx_m, …)` load; `stride_s1m`/`stride_s1n` are
dead parameters). -/
```
```lean
noncomputable def smS1 (s : BlockState) (s1_ptr : RegionName) (row : Nat) : ℝ :=
  s.readMem s1_ptr row
```
</details>

<details><summary><code>smGroupId</code></summary>

```
/-- `group_id = pid // width`. -/
```
```lean
def smGroupId (s : BlockState) (N BN GM : Nat) : Nat :=
  s.pids 0 / smWidth N BN GM
```
</details>

<details><summary><code>smGroupSize</code></summary>

```
/-- `group_size = min(grid_m - group_id * GROUP_M, GROUP_M)`. -/
```
```lean
def smGroupSize (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  min (smGridM M BM - smGroupId s N BN GM * GM) GM
```
</details>

<details><summary><code>smWidth</code></summary>

```
/-- `width = GROUP_M * grid_n`. -/
```
```lean
def smWidth (N BN GM : Nat) : Nat := GM * smGridN N BN
```
</details>

<details><summary><code>smAElem</code></summary>

```
/-- `A[row, kk]` — the `.int` channel of the launch state (`torch.int8`,
widths erased). -/
```
```lean
def smAElem (s : BlockState) (a_ptr : Region .int) (sam sak : Nat)
    (row kk : Nat) : ℤ :=
  s.readMemValue .int (Region.cast a_ptr) (row * sam + kk * sak)
```
</details>

<details><summary><code>smBElem</code></summary>

```
/-- `B[kk, col]`. -/
```
```lean
def smBElem (s : BlockState) (b_ptr : Region .int) (sbk sbn : Nat)
    (kk col : Nat) : ℤ :=
  s.readMemValue .int (Region.cast b_ptr) (kk * sbk + col * sbn)
```
</details>

<details><summary><code>smGridM</code></summary>

```
/-- `grid_m = (M + BLOCK_M - 1) // BLOCK_M`. -/
```
```lean
def smGridM (M BM : Nat) : Nat := (M + BM - 1) / BM
```
</details>

<details><summary><code>smGridN</code></summary>

```
/-- `grid_n = (N + BLOCK_N - 1) // BLOCK_N`. -/
```
```lean
def smGridN (N BN : Nat) : Nat := (N + BN - 1) / BN
```
</details>
