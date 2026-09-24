# Spec sheet — `bench/tritonbench_g/matmul_dequant_int4/MatmulDequantInt4.lean`

**Python source:** `bench/tritonbench_g/matmul_dequant_int4/matmul_dequant_int4.py`

## Public theorem: `matmul_dequant_int4_exec_genuine`

<details><summary>docstring</summary>

```
/-- **Genuine, dimension-general correctness.** For every launch state, the
kernel runs to completion, and every in-range lane `(i, j)` of the program's
`[BK, BN]` output tile — absolute row `k = pid₀·BK + i < K`, absolute column
`n = pid₁·BN + j < N`, exactly the lanes the store mask lets through — holds
the dequantized weight

`fp_b[k, n] = ((bWord >> (k % 8)·4) & 0xF − (zpWord >> (n % 8)·4) & 0xF) · sc`

as an ℝ equation whose nibble difference is computed in ℤ, where `bWord` is
the packed word at `b[k / 8, n]`, `zpWord` the packed word at
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
specification matmul_dequant_int4_exec_genuine
    (b_ptr : Region .nat) (b_scale_ptr : RegionName) (b_zp_ptr : Region .nat)
    (fpb_ptr : RegionName)
    (K N group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn stride_fpbk stride_fpbn : Nat)
    (BK BN : Nat) (s : BlockState)
    (hInj : Function.Injective
      (fun idx : TileIndex [BK, BN] =>
        fpbAddr stride_fpbk stride_fpbn BK BN (s.pids 0) (s.pids 1) idx)) :
    ∃ sF, exec ((matmul_dequant_int4_surface b_ptr b_scale_ptr b_zp_ptr fpb_ptr
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

**Closed-form spec defs (transitive):** `fpbAddr`, `matmul_dequant_int4_surface`, `dequantSpec`, `bNibble`, `zpNibble`, `scElem`, `bWord`, `zpWord`

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

<details><summary><code>matmul_dequant_int4_surface</code></summary>

```
/-! ## Kernel surface (faithful transcription)

Allowed mechanical Lean-syntax-only changes: Python `BLOCK_SIZE_K:
tl.constexpr` / `BLOCK_SIZE_N: tl.constexpr` and the runtime scalar arguments
(`K`, `N`, `group_size`, the eight strides) become Lean `Nat` binders; the
packed-word regions are typed `Region .nat`. Everything else is the two
disclosed respells in the module preamble. -/
```
```lean
def matmul_dequant_int4_surface
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
  (((bNibble s b_ptr stride_bk stride_bn k n : ℤ)
      - (zpNibble s b_zp_ptr group_size stride_bzpk stride_bzpn k n : ℤ) : ℤ) : ℝ)
    * scElem s b_scale_ptr group_size stride_bsk stride_bsn k n
```
</details>

<details><summary><code>bNibble</code></summary>

```
/-- The unpacked 4-bit weight: `(bWord >> (k % 8) * 4) & 0xF`. -/
```
```lean
def bNibble (s : BlockState) (b_ptr : Region .nat)
    (stride_bk stride_bn k n : Nat) : Nat :=
  bWord s b_ptr stride_bk stride_bn k n >>> (k % 8 * 4) &&& 15
```
</details>

<details><summary><code>zpNibble</code></summary>

```
/-- The unpacked 4-bit zero-point: `(zpWord >> (n % 8) * 4) & 0xF`. -/
```
```lean
def zpNibble (s : BlockState) (b_zp_ptr : Region .nat)
    (group_size stride_bzpk stride_bzpn k n : Nat) : Nat :=
  zpWord s b_zp_ptr group_size stride_bzpk stride_bzpn k n >>> (n % 8 * 4) &&& 15
```
</details>

<details><summary><code>scElem</code></summary>

```
/-- The scale at absolute cell `(k, n)`: `bs[k // group_size, n]`. -/
```
```lean
noncomputable def scElem (s : BlockState) (b_scale_ptr : RegionName)
    (group_size stride_bsk stride_bsn k n : Nat) : ℝ :=
  s.readMem b_scale_ptr (k / group_size * stride_bsk + n * stride_bsn)
```
</details>

<details><summary><code>bWord</code></summary>

```
/-- The packed 32-bit word holding the weight nibble for absolute cell
`(k, n)`: `b[k // 8, n]`. Eight weights share a word along K, hence `k / 8`. -/
```
```lean
def bWord (s : BlockState) (b_ptr : Region .nat)
    (stride_bk stride_bn k n : Nat) : Nat :=
  s.readMemValue .nat b_ptr.cast (k / 8 * stride_bk + n * stride_bn)
```
</details>

<details><summary><code>zpWord</code></summary>

```
/-- The packed word holding the zero-point nibble for absolute cell `(k, n)`:
`zp[k // group_size, n // 8]`. Eight zero-points share a word along N, hence
`n / 8`; one group row per `group_size` K rows. -/
```
```lean
def zpWord (s : BlockState) (b_zp_ptr : Region .nat)
    (group_size stride_bzpk stride_bzpn k n : Nat) : Nat :=
  s.readMemValue .nat b_zp_ptr.cast
    (k / group_size * stride_bzpk + n / 8 * stride_bzpn)
```
</details>
