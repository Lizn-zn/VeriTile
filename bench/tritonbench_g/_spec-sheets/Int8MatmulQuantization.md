# Spec sheet — `bench/tritonbench_g/int8_matmul_quantization/Int8MatmulQuantization.lean`

**Python source:** `bench/tritonbench_g/int8_matmul_quantization/int8_matmul_quantization.py`

## Public theorem: `int8_matmul_quantization_quantize_exec_genuine`

<details><summary>docstring</summary>

```
/-- **Genuine, dimension-general correctness of `quantize_int8_perrow_kernel`
(the audit anchor).** From any launch state satisfying the host's own launch
facts, the kernel runs to completion and

* every `(row, kg)` cell of the int8 output `a` holds the `MemCell.of .int`
  carrying `castRealToInt8`'s truncation-toward-zero of
  `fpa[row, kg] / a_scale[row]`, where `a_scale[row] = a_max[row] / 127` and
  `a_max[row]` is the **kernel-computed streaming abs-max**
  `qMaxPartial` — the `0`-seeded running max over all `numKBlocks` masked
  K blocks (`max(prev, max_e (if kb·BK+e < K then |fpa[row, kb·BK+e]| else 0))`
  — the `other=0.0` fill is absorbed by the `0` seed);
* every `a_scale` cell of the fp16 scale vector holds the fp16 cell of the
  same per-row scale (the store's implicit fp16 cast — the host allocates
  `a_scale` as `torch.float16`).

The hypotheses are the launch facts: `hBK` (the `tl.max` reduce needs a
nonempty K tile — `BLOCK_SIZE_K = next_power_of_2(K)` on the host), `hK`
(`numKBlocks = cdiv(K, BLOCK_SIZE_K)` covers `K` — the ceil half that is
semantically forced; extra all-masked blocks contribute `0` to the max and
store nothing), `hFit` (the grid is `M // BLOCK_SIZE_M` programs of
`BLOCK_SIZE_M` rows — exact tiling, so the row-unmasked stores stay in their
window and the `% M` wrap is the identity), the two region-distinctness facts
(the quantize loop reads `fpa` after writing `a`; the scale store must not
clobber `a`), and the store-map injectivity `hInj` (row-major `a`). -/
```
</details>

**Statement:**
```lean
specification int8_matmul_quantization_quantize_exec_genuine
    (fpa_ptr : RegionName) (a_ptr : Region .int) (as_ptr : RegionName)
    (M K : Nat)
    (stride_fpam stride_fpak stride_am stride_ak stride_asm : Nat)
    (BM BK numKBlocks : Nat) (s : BlockState)
    (hBK : 0 < BK)
    (hK : K ≤ numKBlocks * BK)
    (hFit : s.pids 0 * BM + BM ≤ M)
    (hFpaNe : fpa_ptr ≠ (Region.cast a_ptr : RegionName))
    (hAsNe : as_ptr ≠ (Region.cast a_ptr : RegionName))
    (hInj : Function.Injective (fun p : Fin BM × Fin K =>
      (s.pids 0 * BM + p.1.val) * stride_am + p.2.val * stride_ak)) :
    ∃ sF, exec (int8_matmul_quantization_quantize_surface fpa_ptr a_ptr as_ptr
        M K stride_fpam stride_fpak stride_am stride_ak stride_asm
        BM BK numKBlocks).toAlgKernel s = some sF
      ∧ (∀ (r : Fin BM) (kg : Fin K),
          sF.mem (Region.cast a_ptr)
              ((s.pids 0 * BM + r.val) * stride_am + kg.val * stride_ak)
            = MemCell.of .int (qInt8Spec K BK numKBlocks
                (qFpaElem s fpa_ptr stride_fpam stride_fpak
                  (s.pids 0 * BM + r.val)) kg.val))
      ∧ (∀ r : Fin BM,
          sF.mem as_ptr (s.pids 0 * BM * stride_asm + r.val)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (qScaleSpec K BK numKBlocks
                  (qFpaElem s fpa_ptr stride_fpam stride_fpak
                    (s.pids 0 * BM + r.val))))))
```

**Assumptions / layout contracts:**
- `hBK : 0 < BK`
- `hK : K ≤ numKBlocks * BK`
- `hFit : s.pids 0 * BM + BM ≤ M`
- `hFpaNe : fpa_ptr ≠ (Region.cast a_ptr : RegionName)`
- `hAsNe : as_ptr ≠ (Region.cast a_ptr : RegionName)`
- `fun p : Fin BM × Fin K =>
      (s.pids 0 * BM + p.1.val) * stride_am + p.2.val * stride_ak`

**Closed-form spec defs (transitive):** `int8_matmul_quantization_quantize_surface`, `qInt8Spec`, `qFpaElem`, `qScaleSpec`, `qMaxPartial`, `qBlockMax`

<details><summary><code>int8_matmul_quantization_quantize_surface</code></summary>

```lean
def int8_matmul_quantization_quantize_surface
    (fpa_ptr : RegionName) (a_ptr : Region .int) (as_ptr : RegionName)
    (M K : Nat)
    (stride_fpam stride_fpak stride_am stride_ak stride_asm : Nat)
    (BLOCK_SIZE_M BLOCK_SIZE_K numKBlocks : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  fpa_ptrs = fpa_ptr + offs_am[:, None] * $(stride_fpam) + offs_k[None, :] * $(stride_fpak)
  a_ptrs = $((a_ptr : Region .int)) + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
  a_max = tl.zeros([$(BLOCK_SIZE_M)], dtype=tl.float32)
  for k in range($(0), $(numKBlocks), $(1)) {
    fpa = tl.load(fpa_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    a_max = tl.maximum(a_max, tl.max(tl.abs(fpa), axis=1))
    fpa_ptrs += $(BLOCK_SIZE_K) * $(stride_fpak)
  }
  a_scale = (a_max / 127.0)
  fpa_ptrs = fpa_ptr + offs_am[:, None] * $(stride_fpam) + offs_k[None, :] * $(stride_fpak)
  for k in range($(0), $(numKBlocks), $(1)) {
    fpa = tl.load(fpa_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    inta = (fpa / a_scale[:, None]).to(tl.int8)
    tl.store(a_ptrs, inta, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K))
    fpa_ptrs += $(BLOCK_SIZE_K) * $(stride_fpak)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
  }
  as_offs = pid_m * $(BLOCK_SIZE_M) * $(stride_asm) + tl.arange(0, $(BLOCK_SIZE_M))
  tl.store(as_ptr + as_offs, (a_scale).to(tl.float16))
}
```
</details>

<details><summary><code>qInt8Spec</code></summary>

```
/-- **The stored int8 value**: `castRealToInt8`'s own truncation-toward-zero of
`fpa[row, kg] / a_scale[row]`. -/
```
```lean
noncomputable def qInt8Spec (K BK numKBlocks : Nat) (f : Nat → ℝ) (kg : Nat) : ℤ :=
  tritonTruncTowardZero (f kg / qScaleSpec K BK numKBlocks f)
```
</details>

<details><summary><code>qFpaElem</code></summary>

```
/-- `fpa[row, kg]` — the kernel's own address arithmetic over the launch
state's memory. -/
```
```lean
noncomputable def qFpaElem (s : BlockState) (fpa : RegionName)
    (sfm sfk : Nat) (row kg : Nat) : ℝ :=
  s.readMem fpa (row * sfm + kg * sfk)
```
</details>

<details><summary><code>qScaleSpec</code></summary>

```
/-- `a_scale = a_max / 127.` after the full first loop. -/
```
```lean
noncomputable def qScaleSpec (K BK numKBlocks : Nat) (f : Nat → ℝ) : ℝ :=
  qMaxPartial K BK f numKBlocks / 127
```
</details>

<details><summary><code>qMaxPartial</code></summary>

```
/-- **The streaming abs-max**, `0`-seeded: `a_max` after `i` K blocks. This IS
the honest spec — `a_max = tl.zeros(...)` seeds the running max at `0`. -/
```
```lean
noncomputable def qMaxPartial (K BK : Nat) (f : Nat → ℝ) : Nat → ℝ
  | 0 => 0
  | i + 1 => max (qMaxPartial K BK f i) (qBlockMax K BK f i)
```
</details>

<details><summary><code>qBlockMax</code></summary>

```
/-- One K block's masked row abs-max: `max_e (if kb·BK+e < K then |f (kb·BK+e)| else 0)`
(the `other=0.0` fill of the masked lanes, pushed through `tl.abs`). Total
(`0` at `BK = 0`); the kernel's `tl.max` reduce forces `0 < BK` anyway. -/
```
```lean
noncomputable def qBlockMax (K BK : Nat) (f : Nat → ℝ) (kb : Nat) : ℝ :=
  if h : 0 < BK then
    Finset.univ.sup' ⟨⟨0, h⟩, Finset.mem_univ _⟩
      (fun e : Fin BK => if kb * BK + e.val < K then |f (kb * BK + e.val)| else 0)
  else 0
```
</details>

## Public theorem: `int8_matmul_quantization_matmul_exec_genuine`

<details><summary>docstring</summary>

```
/-- **Genuine, dimension-general correctness** (`SPLIT_K = 1` arm — the
autotune table's `reset_to_zero=['c_ptr']` host convention zeroes `C`, and
the grid's second axis has extent `SPLIT_K`, so `hpid1 : s.pids 1 = 0` is
the launch fact and the `tl.atomic_add` else-arm drops with the constexpr).
For every launch state, the kernel runs to completion and every in-range
output lane of `C` holds the `.fp16` memory cell carrying

`fp16( (Σ_{j<K} A[row,j]·B[j,col] : ℤ) · a_scale[row] · b_scale[col] )`

— the exact ℤ int8 matmul (`Op.dotInt`), rescaled in the kernel's own
left-associated order, and downcast once (the placeholder `FloatDType.cast`
is the identity, so the carried value is the product itself). The source's
`accumulator.to(tl.float32)` is carried by the DSL's implicit `Op.intToReal`
promotion (the explicit spelling is ident-blocked; disclosed in the
preamble).

The K-loop loads are **remainder-masked with `other = 0`**, and the guarded
spec terms are exactly those masks' content: a masked-off K lane contributes
the `0·0` product, so extra all-masked trailing blocks add `0` and the only
launch fact forced is `hK : K ≤ numKBlocks · BLOCK_SIZE_K` (the host's
`numKBlocks = tl.cdiv(K, BLOCK_SIZE_K)` satisfies it with no divisibility
assumption — at larger `K` the loop would undercount the sum and this
statement would be false). `hInj` says distinct output lanes get distinct
`C` addresses — `mmCAddr_injective` discharges it for a row-major `C`. The
`% M` / `% N` offset wraps disappear on exactly the lanes the store mask
lets through (`Nat.mod_eq_of_lt`), which also forces the scale-load masks
true, so the stored value reads the plain `A`/`B` rows and scale lanes. -/
```
</details>

**Statement:**
```lean
specification int8_matmul_quantization_matmul_exec_genuine
    (A : Region .int) (as_ptr : RegionName) (B : Region .int) (bs_ptr C : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn
      stride_cm stride_cn : Nat)
    (BM BN BK GM numKBlocks : Nat) (s : BlockState)
    (hK : K ≤ numKBlocks * BK)
    (hpid1 : s.pids 1 = 0)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => mmCAddr stride_cm stride_cn BM BN
        (mmPidM s M N BM BN GM) (mmPidN s M N BM BN GM) i)) :
    ∃ sF, exec (int8_matmul_quantization_matmul_surface A as_ptr B bs_ptr C
        M N K stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn
        stride_cm stride_cn BM BN BK GM numKBlocks).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (mmPidM s M N BM BN GM * BM + idx.1.val < M
            ∧ mmPidN s M N BM BN GM * BN + idx.2.1.val < N) →
          sF.mem C (mmCAddr stride_cm stride_cn BM BN (mmPidM s M N BM BN GM)
              (mmPidN s M N BM BN GM) idx)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some ((((∑ j : Fin K, mmAElem s A stride_am stride_ak
                          (mmPidM s M N BM BN GM * BM + idx.1.val) j.val
                          * mmBElem s B stride_bk stride_bn j.val
                            (mmPidN s M N BM BN GM * BN + idx.2.1.val)) : ℤ) : ℝ)
                  * mmAScaleElem s as_ptr stride_asm
                      (mmPidM s M N BM BN GM * BM + idx.1.val)
                  * mmBScaleElem s bs_ptr stride_bsn
                      (mmPidN s M N BM BN GM * BN + idx.2.1.val))))
```

**Assumptions / layout contracts:**
- `hK : K ≤ numKBlocks * BK`
- `hpid1 : s.pids 1 = 0`
- `fun i : TileIndex [BM, BN] => mmCAddr stride_cm stride_cn BM BN
        (mmPidM s M N BM BN GM) (mmPidN s M N BM BN GM) i`

**Closed-form spec defs (transitive):** `mmCAddr`, `mmPidM`, `mmPidN`, `int8_matmul_quantization_matmul_surface`, `mmAElem`, `mmBElem`, `mmAScaleElem`, `mmBScaleElem`, `mmGroupId`, `mmGroupSize`, `mmWidth`, `mmGridM`, `mmGridN`

<details><summary><code>mmCAddr</code></summary>

```
/-- The `C` store address for output lane `(r, c)` — the kernel's own
const-first order `stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]`. -/
```
```lean
def mmCAddr (stride_cm stride_cn BM BN pm pn : Nat)
    (idx : TileIndex [BM, BN]) : Nat :=
  stride_cm * (pm * BM + idx.1.val) + stride_cn * (pn * BN + idx.2.1.val)
```
</details>

<details><summary><code>mmPidM</code></summary>

```
/-- `pid_m = first_pid_m + (pid % group_size_m)`. -/
```
```lean
def mmPidM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  mmGroupId s N BN GM * GM + s.pids 0 % mmGroupSize s M N BM BN GM
```
</details>

<details><summary><code>mmPidN</code></summary>

```
/-- `pid_n = (pid % num_pid_in_group) // group_size_m`. -/
```
```lean
def mmPidN (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  s.pids 0 % mmWidth N BN GM / mmGroupSize s M N BM BN GM
```
</details>

<details><summary><code>int8_matmul_quantization_matmul_surface</code></summary>

```
/-- The `matmul_kernel` surface (`SPLIT_K = 1` arm: the `tl.atomic_add` else
branch drops with the constexpr; every `* SPLIT_K` factor is folded to its
`SPLIT_K = 1` value; `pid_sp_k` stays faithful). The loop bound
`tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)` is spelled as the antiquoted binder
`numKBlocks`; the K loads carry the source's genuine remainder masks with
`other=0.0`, and `c = (accumulator.to(tl.float32) * a_scale[:, None] *
b_scale[None, :]).to(tl.float16)` is spelled through the DSL's implicit
`Op.intToReal` promotion (the explicit `.to(tl.float32)` is ident-blocked —
disclosed in the preamble). -/
```
```lean
def int8_matmul_quantization_matmul_surface
    (A : Region .int) (as_ptr : RegionName) (B : Region .int) (bs_ptr C : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_asm stride_bk stride_bn stride_bsn stride_cm stride_cn : Nat)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  pid_sp_k = tl.program_id(1)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = pid_sp_k * $(BLOCK_SIZE_K) + tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = $((A : Region .int)) + (offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak))
  b_ptrs = $((B : Region .int)) + (offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn))
  as_ptrs = as_ptr + offs_am * $(stride_asm)
  bs_ptrs = bs_ptr + offs_bn * $(stride_bsn)
  a_scale = tl.load(as_ptrs, mask=offs_am < $(M), other=0.0)
  b_scale = tl.load(bs_ptrs, mask=offs_bn < $(N), other=0.0)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.int32)
  for k in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K), other=0.0)
    accumulator += tl.dot(a, b)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  c = (accumulator * a_scale[:, None] * b_scale[None, :]).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}
```
</details>

<details><summary><code>mmAElem</code></summary>

```
/-- `A[row, kg]` — a signed `.int`-channel read (int8 values are signed). -/
```
```lean
def mmAElem (s : BlockState) (A : Region .int)
    (stride_am stride_ak : Nat) (row kg : Nat) : ℤ :=
  s.readMemValue .int (Region.cast A) (row * stride_am + kg * stride_ak)
```
</details>

<details><summary><code>mmBElem</code></summary>

```
/-- `B[kg, col]` — the signed `.int` read of `B`. -/
```
```lean
def mmBElem (s : BlockState) (B : Region .int)
    (stride_bk stride_bn : Nat) (kg col : Nat) : ℤ :=
  s.readMemValue .int (Region.cast B) (kg * stride_bk + col * stride_bn)
```
</details>

<details><summary><code>mmAScaleElem</code></summary>

```
/-- `a_scale[row]` — the per-row dequant scale lane. -/
```
```lean
noncomputable def mmAScaleElem (s : BlockState) (as_ptr : RegionName)
    (stride_asm : Nat) (row : Nat) : ℝ :=
  s.readMem as_ptr (row * stride_asm)
```
</details>

<details><summary><code>mmBScaleElem</code></summary>

```
/-- `b_scale[col]` — the per-column dequant scale lane. -/
```
```lean
noncomputable def mmBScaleElem (s : BlockState) (bs_ptr : RegionName)
    (stride_bsn : Nat) (col : Nat) : ℝ :=
  s.readMem bs_ptr (col * stride_bsn)
```
</details>

<details><summary><code>mmGroupId</code></summary>

```
/-- `group_id = pid // num_pid_in_group`. -/
```
```lean
def mmGroupId (s : BlockState) (N BN GM : Nat) : Nat :=
  s.pids 0 / mmWidth N BN GM
```
</details>

<details><summary><code>mmGroupSize</code></summary>

```
/-- `group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)` (with
`first_pid_m = group_id * GROUP_SIZE_M`). -/
```
```lean
def mmGroupSize (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  min (mmGridM M BM - mmGroupId s N BN GM * GM) GM
```
</details>

<details><summary><code>mmWidth</code></summary>

```
/-- `num_pid_in_group = GROUP_SIZE_M * num_pid_n`. -/
```
```lean
def mmWidth (N BN GM : Nat) : Nat := GM * mmGridN N BN
```
</details>

<details><summary><code>mmGridM</code></summary>

```
/-- `num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)`. -/
```
```lean
def mmGridM (M BM : Nat) : Nat := (M + BM - 1) / BM
```
</details>

<details><summary><code>mmGridN</code></summary>

```
/-- `num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)`. -/
```
```lean
def mmGridN (N BN : Nat) : Nat := (N + BN - 1) / BN
```
</details>
