import VeriTile.Triton

/-!
# `matmul_dequant_int4` — strict per-kernel correctness

This file is the DSL port of `matmul_dequant_int4.py`'s `dequantize_kernel` —
the target JIT, the only kernel launched by any host in the file (the chain
is `matmul_dequantize_int4_s1` → `dequantize_int4` → `dequantize_kernel[grid]`,
followed by a host-side `torch.mm(a, fp_b)`; the test exercises
`matmul_dequantize_int4_s1` only). One program owns the
`[BLOCK_SIZE_K, BLOCK_SIZE_N]` tile at `(k_block_idx, n_block_idx) =
(tl.program_id(0), tl.program_id(1))` of the dequantized weight `fp_b`: it
loads a tile of packed int4 weight words (`b_ptr`, eight 4-bit weights per
32-bit word along the K axis, hence the `offs_k // 8` row), the matching
packed zero-point words (`b_zp_ptr`, eight 4-bit zero-points per word along
the N axis, one group row per `group_size` K rows, hence
`offs_k // group_size` and `offs_n // 8`), and the per-group scales
(`b_scale_ptr`, at `offs_k // group_size`), unpacks the two nibbles with a
shift-and-mask, and stores the signed nibble difference times the scale —
`fp_weight = (nib(b) − nib(zp)) · scale` — every load and the store masked by
the same `(offs_n < N) & (offs_k < K)` gate.

The file's first kernel, `matmul4_kernel`, is dead code here: no host in the
file references it, and its body is byte-identical to the kernel already
ported in the `matmul_dequantize_int4` port — following the `rms_rbe_matmul`
dead-kernel precedent it is not modeled.

Translation-surface blocker: two disclosed surface deviations, neither
semantic. **(1)** the dequant statement is spelled
`fp_weight = (tl.cast((int32_b >> b_shift) & $(15), tl.int32) -
tl.cast((zp_b >> bzp_shift) & $(15), tl.int32)) * scale_b`: the two packed
containers live on the `.nat` channel (the DSL's bitwise `>>` / `&` are
defined there), the nibble difference goes negative, and ℕ subtraction
truncates — so the subtraction must run on the signed `.int` channel and the
product promotes to ℝ via `Op.intToReal` (auto-inserted for the
`.int × .real` operand pair). `tl.cast(x, tl.int32)` is the explicit
`Op.castNatToInt` spelling (the `kcache_copy_triton` `tl.cast` precedent); a
bare `.to(tl.int32)` on a nat value is width-erased to a no-op by the
inference layer. This is exactly the `int4_matmul` signed-dequant respell
(its disclosure (3)), applied inline instead of through the twin's
`int_b` / `int_bzp` register split. **(2)** integer literals inside index
arithmetic are antiquoted `$(n)` (a bare literal is inferred `.real` by the
DSL's expression typing): the source's `8` and `4` are written `$(8)` /
`$(4)`, the nibble mask `0xF` is written `$(15)` — same value, decimal
spelling — and the `group_size` runtime argument is the antiquoted binder
`$(group_size)` (the `int4_matmul` literal-spelling precedent).

The three masked loads keep their `other=0.0` kwargs faithfully; on the two
packed `.nat` channels the expander reads the literal `0.0` as the nat
constant `0` (value-identical: `0.0` cast to int32 is `0`, and masked-off
lanes are never consumed — the store mask equals the load mask).

## The two packed channels

`b_ptr` (packed weights) and `b_zp_ptr` (packed zero-points) are
`torch.int32` tensors and both are modeled as `Region .nat`: the DSL's
bitwise operators are defined on the `.nat` channel only. This is a statement
about the *container*, not the extracted values — `(w >> 4·i) & 0xF` selects
the same nibble whether the shift is logical or arithmetic, and every
extracted nibble is in `[0, 15]`, hence non-negative (the `int4_matmul`
packed-channel note). The signedness enters only *after* extraction, in the
nibble difference — the `.int` hop of disclosure (1).

## Proof map

```
matmul_dequant_int4_exec_genuine                 the headline
├─ mdi4_exec_isSome     termination: one big `simp` runs the whole
│                       18-statement straight line (every load carries
│                       `other=`, the masked store is total)
├─ mdi4_correct         per-lane readback: the same `simp` normalizes `exec`
│                       to a masked scatter `foldl` over the store;
│                       `scatter_readback_prop_masked_nd` + the store gate
│                       collapse it to `dequantSpec`
fpbAddr_injective                discharges the headline's `hInj` (row-major)
```

The stored value `dequantSpec` is built from the kernel's own accessors
(`bWord`/`bNibble`, `zpWord`/`zpNibble`, `scElem`) over the **launch** state's
memory. The output region `fpb_ptr` is never read back into the spec, so no
part of the trust path is self-referential.

## Modeling boundary

Arithmetic is over ℝ (not bit-accurate IEEE float); `@triton.autotune` and
the host launch (the 2-D grid `(cdiv(K, BLOCK_SIZE_K), cdiv(N,
BLOCK_SIZE_N))`, the strides, `group_size`, the downstream `torch.mm`) are
the *trusted boundary*. Every dimension, stride and block size stays a
symbolic parameter, so the headline covers every autotune config at once.
Nat division is total in Lean (`x / 0 = 0`) and `group_size` appears under
`/` identically in the kernel and in the spec, so no `0 < group_size`
hypothesis is needed; there is no divisibility side condition either (all
three loads and the store share one mask). The only hypothesis beyond the
dimension variables is `hInj` — distinct tile lanes must land on distinct
`fp_b` cells, which `fpbAddr_injective` discharges for the host's row-major
`fp_b`.
-/

namespace VeriTile.Bench.TritonBenchG.MatmulDequantInt4

open VeriTile.Triton

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-! ## Kernel surface (faithful transcription)

Allowed mechanical Lean-syntax-only changes: Python `BLOCK_SIZE_K:
tl.constexpr` / `BLOCK_SIZE_N: tl.constexpr` and the runtime scalar arguments
(`K`, `N`, `group_size`, the eight strides) become Lean `Nat` binders; the
packed-word regions are typed `Region .nat`. Everything else is the two
disclosed respells in the module preamble. -/

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

theorem matmul_dequant_int4_surface_toAlgorithm_supported
    (b_ptr b_zp_ptr : Region .nat) (b_scale_ptr fpb_ptr : RegionName)
    (K N group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn stride_fpbk stride_fpbn BK BN : Nat) :
    ∃ alg, (matmul_dequant_int4_surface b_ptr b_scale_ptr b_zp_ptr fpb_ptr
      K N group_size stride_bk stride_bn stride_bsk stride_bsn stride_bzpk
      stride_bzpn stride_fpbk stride_fpbn BK BN).toAlgorithm?
        = Except.ok alg := by
  simp [matmul_dequant_int4_surface, ComputeExpr.toAlgorithm?]

/-! ## Element accessors

Each accessor is the kernel's own address arithmetic over the **launch**
state's memory, parameterized by the *absolute* row `k = k_block_idx ·
BLOCK_SIZE_K + i` and column `n = n_block_idx · BLOCK_SIZE_N + j`. -/

/-- The packed 32-bit word holding the weight nibble for absolute cell
`(k, n)`: `b[k // 8, n]`. Eight weights share a word along K, hence `k / 8`. -/
def bWord (s : BlockState) (b_ptr : Region .nat)
    (stride_bk stride_bn k n : Nat) : Nat :=
  s.readMemValue .nat b_ptr.cast (k / 8 * stride_bk + n * stride_bn)

/-- The unpacked 4-bit weight: `(bWord >> (k % 8) * 4) & 0xF`. -/
def bNibble (s : BlockState) (b_ptr : Region .nat)
    (stride_bk stride_bn k n : Nat) : Nat :=
  bWord s b_ptr stride_bk stride_bn k n >>> (k % 8 * 4) &&& 15

/-- The packed word holding the zero-point nibble for absolute cell `(k, n)`:
`zp[k // group_size, n // 8]`. Eight zero-points share a word along N, hence
`n / 8`; one group row per `group_size` K rows. -/
def zpWord (s : BlockState) (b_zp_ptr : Region .nat)
    (group_size stride_bzpk stride_bzpn k n : Nat) : Nat :=
  s.readMemValue .nat b_zp_ptr.cast
    (k / group_size * stride_bzpk + n / 8 * stride_bzpn)

/-- The unpacked 4-bit zero-point: `(zpWord >> (n % 8) * 4) & 0xF`. -/
def zpNibble (s : BlockState) (b_zp_ptr : Region .nat)
    (group_size stride_bzpk stride_bzpn k n : Nat) : Nat :=
  zpWord s b_zp_ptr group_size stride_bzpk stride_bzpn k n >>> (n % 8 * 4) &&& 15

/-- The scale at absolute cell `(k, n)`: `bs[k // group_size, n]`. -/
noncomputable def scElem (s : BlockState) (b_scale_ptr : RegionName)
    (group_size stride_bsk stride_bsn k n : Nat) : ℝ :=
  s.readMem b_scale_ptr (k / group_size * stride_bsk + n * stride_bsn)

/-- **The stored value** at absolute cell `(k, n)`: the **signed** nibble
difference (ℤ-subtraction — this is what the `.int` hop of disclosure (1)
buys; ℕ subtraction would truncate), embedded into ℝ and scaled. -/
noncomputable def dequantSpec (s : BlockState) (b_scale_ptr : RegionName)
    (b_ptr b_zp_ptr : Region .nat)
    (group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn : Nat) (k n : Nat) : ℝ :=
  (((bNibble s b_ptr stride_bk stride_bn k n : ℤ)
      - (zpNibble s b_zp_ptr group_size stride_bzpk stride_bzpn k n : ℤ) : ℤ) : ℝ)
    * scElem s b_scale_ptr group_size stride_bsk stride_bsn k n

/-- The `fp_b` store address for tile cell `idx` of program `(p₀, p₁)`:
`(p₀·BK + i) · stride_fpbk + (p₁·BN + j) · stride_fpbn`. -/
def fpbAddr (stride_fpbk stride_fpbn BK BN p₀ p₁ : Nat)
    (idx : TileIndex [BK, BN]) : Nat :=
  (p₀ * BK + idx.1.val) * stride_fpbk + (p₁ * BN + idx.2.1.val) * stride_fpbn

/-- The kernel's store gate `mask = n_mask & k_mask`, in the kernel's own
conjunct order (the N test comes first in the source). -/
def mdiActive (K N BK BN p₀ p₁ : Nat) (idx : TileIndex [BK, BN]) : Prop :=
  p₁ * BN + idx.2.1.val < N ∧ p₀ * BK + idx.1.val < K

/-- `fpbAddr` is injective for a row-major `fp_b` (`stride_fpbn > 0` and a
K-row stride wide enough to hold `BLOCK_SIZE_N` columns) — the discharge
lemma for the headline's `hInj`. The host's contiguous `fp_b` of
`torch.ones((K, N))` has `stride_fpbk = N ≥ BN · 1` for every launched tile. -/
theorem fpbAddr_injective (stride_fpbk stride_fpbn BK BN p₀ p₁ : Nat)
    (hn : 0 < stride_fpbn) (hfit : BN * stride_fpbn ≤ stride_fpbk) :
    Function.Injective
      (fun idx : TileIndex [BK, BN] =>
        fpbAddr stride_fpbk stride_fpbn BK BN p₀ p₁ idx) := by
  intro i j hij
  obtain ⟨r₁, c₁, u₁⟩ := i
  obtain ⟨r₂, c₂, u₂⟩ := j
  simp only [fpbAddr] at hij
  have hexp : ∀ r c : Nat,
      (p₀ * BK + r) * stride_fpbk + (p₁ * BN + c) * stride_fpbn
        = p₀ * BK * stride_fpbk + r * stride_fpbk
          + (p₁ * BN * stride_fpbn + c * stride_fpbn) := by
    intro r c
    rw [Nat.add_mul, Nat.add_mul]
  rw [hexp, hexp] at hij
  have key : r₁.val * stride_fpbk + c₁.val * stride_fpbn
      = r₂.val * stride_fpbk + c₂.val * stride_fpbn := by omega
  have hrow : ∀ c : Fin BN, c.val * stride_fpbn < stride_fpbk := fun c =>
    lt_of_lt_of_le (Nat.mul_lt_mul_of_lt_of_le c.isLt (le_refl _) hn) hfit
  have hr : r₁.val = r₂.val := by
    rcases Nat.lt_trichotomy r₁.val r₂.val with h | h | h
    · have : r₁.val * stride_fpbk + stride_fpbk ≤ r₂.val * stride_fpbk := by
        rw [← Nat.succ_mul]
        exact Nat.mul_le_mul_right _ h
      have := hrow c₁
      omega
    · exact h
    · have : r₂.val * stride_fpbk + stride_fpbk ≤ r₁.val * stride_fpbk := by
        rw [← Nat.succ_mul]
        exact Nat.mul_le_mul_right _ h
      have := hrow c₂
      omega
  have hc : c₁.val = c₂.val := by
    have : c₁.val * stride_fpbn = c₂.val * stride_fpbn := by
      rw [hr] at key; omega
    exact Nat.eq_of_mul_eq_mul_right hn this
  simp only [Prod.mk.injEq]
  exact ⟨Fin.ext hr, Fin.ext hc, trivial⟩

/-! ## Termination and per-lane readback

The kernel is an 18-statement straight line: one big `simp` over the
step/eval definitions runs it completely, normalizing `exec` to `some` of a
masked scatter `foldl` over the output store (the masked loads are total —
each carries `other=` — and so is the masked store). The readback lemma then
peels that `foldl` with `scatter_readback_prop_masked_nd`. -/

set_option maxHeartbeats 1600000 in
/-- Termination: the kernel executes to completion from any launch state. -/
theorem mdi4_exec_isSome
    (b_ptr b_zp_ptr : Region .nat) (b_scale_ptr fpb_ptr : RegionName)
    (K N group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn stride_fpbk stride_fpbn BK BN : Nat)
    (s : BlockState) :
    ∃ s1, exec ((matmul_dequant_int4_surface b_ptr b_scale_ptr b_zp_ptr fpb_ptr
      K N group_size stride_bk stride_bn stride_bsk stride_bsn stride_bzpk
      stride_bzpn stride_fpbk stride_fpbn BK BN).toAlgKernel) s = some s1 := by
  simp [exec, matmul_dequant_int4_surface, ComputeKernel.toAlgKernel,
        ComputeExpr.toAlgorithm?, stepStmts, stepStmt, evalOp.eq_def,
        Option.bind, Tile.bop, Tile.cop, Tile.expandDim,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt, IntegralDType.floorDiv, IntegralDType.mod,
        TileShape.dropInsertedIndex]

set_option maxHeartbeats 1600000 in
/-- Per-lane readback: on every lane the store gate lets through, the final
memory holds `dequantSpec` at the lane's `fp_b` address. The proof normalizes
`exec` to the masked scatter `foldl`, peels it with
`scatter_readback_prop_masked_nd` (this is where `hInj` is spent), and
collapses the gated loads with the lane's own bounds facts. -/
theorem mdi4_correct
    (b_ptr b_zp_ptr : Region .nat) (b_scale_ptr fpb_ptr : RegionName)
    (K N group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn stride_fpbk stride_fpbn BK BN : Nat)
    (s s' : BlockState)
    (hInj : Function.Injective
      (fun idx : TileIndex [BK, BN] =>
        (s.pids 0 * BK + idx.1.val) * stride_fpbk
          + (s.pids 1 * BN + idx.2.1.val) * stride_fpbn))
    (hExec : exec ((matmul_dequant_int4_surface b_ptr b_scale_ptr b_zp_ptr
      fpb_ptr K N group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn stride_fpbk stride_fpbn BK BN).toAlgKernel) s
        = some s') :
    ∀ idx : TileIndex [BK, BN],
      mdiActive K N BK BN (s.pids 0) (s.pids 1) idx →
      s'.readMem fpb_ptr
          (fpbAddr stride_fpbk stride_fpbn BK BN (s.pids 0) (s.pids 1) idx)
        = dequantSpec s b_scale_ptr b_ptr b_zp_ptr group_size stride_bk
            stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn
            (s.pids 0 * BK + idx.1.val) (s.pids 1 * BN + idx.2.1.val) := by
  intro idx hact
  obtain ⟨hn, hk⟩ := hact
  simp [exec, matmul_dequant_int4_surface, ComputeKernel.toAlgKernel,
        ComputeExpr.toAlgorithm?, stepStmts, stepStmt, evalOp.eq_def,
        Option.bind, Tile.bop, Tile.cop, Tile.expandDim,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt, IntegralDType.floorDiv, IntegralDType.mod,
        TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  simp only [fpbAddr]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hInj idx]
  rw [if_pos ⟨hn, hk⟩]
  simp [dequantSpec, bNibble, bWord, zpNibble, zpWord, scElem, hn, hk]

/-! ## Main theorem -/

set_option maxHeartbeats 1600000 in
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
                (s.pids 0 * BK + idx.1.val) (s.pids 1 * BN + idx.2.1.val) := by
  obtain ⟨sF, hExec⟩ := mdi4_exec_isSome b_ptr b_zp_ptr b_scale_ptr fpb_ptr
    K N group_size stride_bk stride_bn stride_bsk stride_bsn stride_bzpk
    stride_bzpn stride_fpbk stride_fpbn BK BN s
  refine ⟨sF, hExec, fun idx hidx => ?_⟩
  exact mdi4_correct b_ptr b_zp_ptr b_scale_ptr fpb_ptr K N group_size
    stride_bk stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn
    stride_fpbk stride_fpbn BK BN s sF
    (by simpa [fpbAddr] using hInj) hExec idx ⟨hidx.2, hidx.1⟩

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.MatmulDequantInt4
