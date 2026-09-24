# Spec sheet — `bench/tritonbench_g/int8_dequant_matmul/Int8DequantMatmul.lean`

**Python source:** `bench/tritonbench_g/int8_dequant_matmul/int8_dequant_matmul.py`

## Public theorem: `int8_dequant_matmul_exec_genuine`

<details><summary>docstring</summary>

```
/-- **Genuine, dimension-general correctness.** For every launch state, the
`SPLIT_K = 1` / `EVEN_K = True` arm of the kernel runs to completion, and
every in-range output lane of `C` holds the `.fp16` memory cell carrying
`i8Spec`: the exact ℤ int8 matmul (`Op.dotInt` over all `numKBlocks` K
steps), rescaled per output cell in the kernel's own parenthesization
`w_factor * (x_factor * (acc * divfactor))`, plus the `has_bias`-guarded
per-column fp16 bias. The terminal store writes
`acc = ….to(tl.float16)` — a genuine fp16 quantization event, so the
conclusion is stated at the `MemCell` level:
`C[row, col] = MemCell.of .fp16 (fp16(i8Spec))` (the `matmul_dequantize`
`matmul_kernel` precedent; the placeholder `FloatDType.cast` is the
identity, so the carried value is `i8Spec` itself — the fp16 cast between
the product and the bias add likewise erases to the ℝ sum).

The hypotheses are the kernel's own launch facts: `hK` is the `EVEN_K`
heuristic `K % (BLOCK_K * SPLIT_K) == 0` (the loop trip count `numKBlocks`
is exact — the K-loop loads are unmasked); `hpid1` is grid axis 1 having
extent `SPLIT_K = 1`; `hInj` says distinct output lanes get distinct `C`
addresses — `cAddr_injective` discharges it for a row-major `C`. The
`% M` / `% N` offset wraps disappear on exactly the lanes the store mask
lets through (`Nat.mod_eq_of_lt`); the bias is read at the **unwrapped**
`rn`, so its lane is the plain `col` from the start. -/
```
</details>

**Statement:**
```lean
specification int8_dequant_matmul_exec_genuine
    (A B : Region .int) (C bias state_x_ptr state_w_ptr : RegionName)
    (M N K : Nat) (divfactor : ℝ) (has_bias : Bool)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BM BN BK GM numKBlocks : Nat) (s : BlockState)
    (hK : K = BK * numKBlocks)
    (hpid1 : s.pids 1 = 0)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN
        (pidM s M N BM BN GM) (pidN s M N BM BN GM) i)) :
    ∃ sF, exec (int8_dequant_matmul_surface A B C bias state_x_ptr state_w_ptr
        M N K divfactor has_bias
        stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        BM BN BK GM numKBlocks).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (pidM s M N BM BN GM * BM + idx.1.val < M
            ∧ pidN s M N BM BN GM * BN + idx.2.1.val < N) →
          sF.mem C (cAddr stride_cm stride_cn BM BN (pidM s M N BM BN GM)
              (pidN s M N BM BN GM) idx)
            = MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
                (some (i8Spec s A B bias state_x_ptr state_w_ptr
                  stride_am stride_ak stride_bk stride_bn BK numKBlocks
                  divfactor has_bias
                  (pidM s M N BM BN GM * BM + idx.1.val)
                  (pidN s M N BM BN GM * BN + idx.2.1.val))))
```

**Assumptions / layout contracts:**
- `has_bias : Bool`
- `hK : K = BK * numKBlocks`
- `hpid1 : s.pids 1 = 0`
- `fun i : TileIndex [BM, BN] => cAddr stride_cm stride_cn BM BN
        (pidM s M N BM BN GM) (pidN s M N BM BN GM) i`

**Closed-form spec defs (transitive):** `cAddr`, `pidM`, `pidN`, `int8_dequant_matmul_surface`, `i8Spec`, `groupId`, `groupSize`, `i8Width`, `i8ProdSpec`, `biasElem`, `gridM`, `gridN`, `wFElem`, `xFElem`, `accSpec`, `accStep`, `aElem`, `bElem`

<details><summary><code>cAddr</code></summary>

```
/-- The `C` store address for output cell `(r, c)` — the kernel's
`rm[:, None] * stride_cm + rn[None, :] * stride_cn`. -/
```
```lean
def cAddr (stride_cm stride_cn BM BN pm pn : Nat)
    (idx : TileIndex [BM, BN]) : Nat :=
  (pm * BM + idx.1.val) * stride_cm + (pn * BN + idx.2.1.val) * stride_cn
```
</details>

<details><summary><code>pidM</code></summary>

```
/-- `pid_m = group_id * GROUP_M + (pid % group_size)`. -/
```
```lean
def pidM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  groupId s N BN GM * GM + s.pids 0 % groupSize s M N BM BN GM
```
</details>

<details><summary><code>pidN</code></summary>

```
/-- `pid_n = (pid % width) // (group_size)`. -/
```
```lean
def pidN (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  s.pids 0 % i8Width N BN GM / groupSize s M N BM BN GM
```
</details>

<details><summary><code>int8_dequant_matmul_surface</code></summary>

```lean
def int8_dequant_matmul_surface
    (A B : Region .int) (C bias state_x_ptr state_w_ptr : RegionName)
    (M N K : Nat) (divfactor : ℝ) (has_bias : Bool)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M numKBlocks : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  pid_z = tl.program_id(1)
  grid_m = tl.cdiv($(M), $(BLOCK_M))
  grid_n = tl.cdiv($(N), $(BLOCK_N))
  width = $(GROUP_M) * grid_n
  group_id = pid // width
  group_size = min(grid_m - group_id * $(GROUP_M), $(GROUP_M))
  pid_m = group_id * $(GROUP_M) + (pid % group_size)
  pid_n = (pid % width) // (group_size)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  ram = tl.max_contiguous(tl.multiple_of(rm % $(M), $(BLOCK_M)), $(BLOCK_M))
  rbn = tl.max_contiguous(tl.multiple_of(rn % $(N), $(BLOCK_N)), $(BLOCK_N))
  rk = pid_z * $(BLOCK_K) + tl.arange(0, $(BLOCK_K))
  A = $((A : Region .int)) + (ram[:, None] * $(stride_am) + rk[None, :] * $(stride_ak))
  B = $((B : Region .int)) + (rk[:, None] * $(stride_bk) + rbn[None, :] * $(stride_bn))
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  w_factor = tl.load(state_w_ptr + rbn)[None, :]
  x_factor = tl.load(state_x_ptr + ram)[:, None]
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.int32)
  for k in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(A)
    b = tl.load(B)
    acc += tl.dot(a, b)
    A += $(BLOCK_K) * $(stride_ak)
    B += $(BLOCK_K) * $(stride_bk)
  }
  acc = w_factor * (x_factor * (acc * $((divfactor : ℝ))))
  acc = (acc).to(tl.float16)
  if has_bias {
    bias = tl.load(bias + rn).to(tl.float16)
    acc = acc + bias[None, :]
  }
  C = C + (rm[:, None] * $(stride_cm) + rn[None, :] * $(stride_cn))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  tl.store(C, acc, mask=mask)
}
```
</details>

<details><summary><code>i8Spec</code></summary>

```
/-- **The stored value** (before the terminal fp16 quantization): the rescaled
product plus the `has_bias`-guarded per-column bias. -/
```
```lean
noncomputable def i8Spec (s : BlockState) (A B : Region .int)
    (bias state_x state_w : RegionName)
    (stride_am stride_ak stride_bk stride_bn BK numKBlocks : Nat)
    (divfactor : ℝ) (has_bias : Bool) (row col : Nat) : ℝ :=
  i8ProdSpec s A B state_x state_w stride_am stride_ak stride_bk stride_bn
      BK numKBlocks divfactor row col
    + (if has_bias then biasElem s bias col else 0)
```
</details>

<details><summary><code>groupId</code></summary>

```
/-- `group_id = pid // width`. -/
```
```lean
def groupId (s : BlockState) (N BN GM : Nat) : Nat :=
  s.pids 0 / i8Width N BN GM
```
</details>

<details><summary><code>groupSize</code></summary>

```
/-- `group_size = min(grid_m - group_id * GROUP_M, GROUP_M)`. -/
```
```lean
def groupSize (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  min (gridM M BM - groupId s N BN GM * GM) GM
```
</details>

<details><summary><code>i8Width</code></summary>

```
/-- `width = GROUP_M * grid_n`. -/
```
```lean
def i8Width (N BN GM : Nat) : Nat := GM * gridN N BN
```
</details>

<details><summary><code>i8ProdSpec</code></summary>

```
/-- The rescaled epilogue value at `(row, col)` — the kernel's exact
parenthesization `w_factor * (x_factor * (acc * divfactor))`, with the ℤ
accumulator promoted to ℝ once (`Op.intToReal`) and then scaled. -/
```
```lean
noncomputable def i8ProdSpec (s : BlockState) (A B : Region .int)
    (state_x state_w : RegionName)
    (stride_am stride_ak stride_bk stride_bn BK numKBlocks : Nat)
    (divfactor : ℝ) (row col : Nat) : ℝ :=
  wFElem s state_w col
    * (xFElem s state_x row
        * (((accSpec s A B stride_am stride_ak stride_bk stride_bn BK
              numKBlocks row col : ℤ) : ℝ)
            * divfactor))
```
</details>

<details><summary><code>biasElem</code></summary>

```
/-- `bias[col]` — the fp16 bias lane, decoded exactly the way the kernel's
`.fp16`-typed load decodes it (`readMemAs .fp16`: an fp16 cell's real payload,
`0` on a dtype-mismatched cell). -/
```
```lean
noncomputable def biasElem (s : BlockState) (bias : RegionName) (addr : Nat) : ℝ :=
  FloatDType.fp16.storeValue (s.readMemValue .fp16 bias addr)
```
</details>

<details><summary><code>gridM</code></summary>

```
/-- `grid_m = tl.cdiv(M, BLOCK_M)`. -/
```
```lean
def gridM (M BM : Nat) : Nat := (M + BM - 1) / BM
```
</details>

<details><summary><code>gridN</code></summary>

```
/-- `grid_n = tl.cdiv(N, BLOCK_N)`. -/
```
```lean
def gridN (N BN : Nat) : Nat := (N + BN - 1) / BN
```
</details>

<details><summary><code>wFElem</code></summary>

```
/-- `state_w[col]` — the per-column dequant scale (`w_factor` lane). -/
```
```lean
noncomputable def wFElem (s : BlockState) (state_w : RegionName) (col : Nat) : ℝ :=
  s.readMem state_w col
```
</details>

<details><summary><code>xFElem</code></summary>

```
/-- `state_x[row]` — the per-row dequant scale (`x_factor` lane). -/
```
```lean
noncomputable def xFElem (s : BlockState) (state_x : RegionName) (row : Nat) : ℝ :=
  s.readMem state_x row
```
</details>

<details><summary><code>accSpec</code></summary>

```
/-- The full ℤ accumulator: `acc` after all `numKBlocks` K steps. -/
```
```lean
def accSpec (s : BlockState) (A B : Region .int)
    (stride_am stride_ak stride_bk stride_bn BK numKBlocks : Nat)
    (row col : Nat) : ℤ :=
  ∑ k : Fin numKBlocks,
    accStep s A B stride_am stride_ak stride_bk stride_bn BK row col k.val
```
</details>

<details><summary><code>accStep</code></summary>

```
/-- One K step's ℤ contribution to output cell `(row, col)` — the exact
integer sum-product `Op.dotInt` computes. -/
```
```lean
def accStep (s : BlockState) (A B : Region .int)
    (stride_am stride_ak stride_bk stride_bn BK : Nat) (row col k : Nat) : ℤ :=
  ∑ e : Fin BK,
    aElem s A stride_am stride_ak BK row k e.val
      * bElem s B stride_bk stride_bn BK k e.val col
```
</details>

<details><summary><code>aElem</code></summary>

```
/-- `A[row, k*BK + e]` at K step `k` — a **signed** `.int`-channel read
(int8 values are signed). Unmasked, matching the source's bare `tl.load(A)`. -/
```
```lean
def aElem (s : BlockState) (A : Region .int)
    (stride_am stride_ak BK : Nat) (row k e : Nat) : ℤ :=
  s.readMemValue .int (Region.cast A) (row * stride_am + (e + k * BK) * stride_ak)
```
</details>

<details><summary><code>bElem</code></summary>

```
/-- `B[k*BK + e, col]` at K step `k` — the signed `.int` read of `B`. -/
```
```lean
def bElem (s : BlockState) (B : Region .int)
    (stride_bk stride_bn BK : Nat) (k e col : Nat) : ℤ :=
  s.readMemValue .int (Region.cast B) ((e + k * BK) * stride_bk + col * stride_bn)
```
</details>
