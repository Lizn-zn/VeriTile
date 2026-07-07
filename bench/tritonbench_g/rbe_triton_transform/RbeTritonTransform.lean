import VeriTile.Core
import VeriTile.Semantics
import VeriTile.Float
import VeriTile.Frontend.Triton.DSL

/-!
# `rbe_triton_transform` — strict per-kernel correctness

`rbe_triton` applies the rotary position embedding with interleaved real/imag
layout: each program owns one batch (`program_id(0)`) and one `(m, k)` tile
(`program_id(1)`, decomposed in-kernel via `pid // tl.cdiv(K, BLOCK_SIZE_K)` /
`pid % tl.cdiv(K, BLOCK_SIZE_K)`). `offs_n` walks the EVEN columns of the tile
(`tl.arange(0, BLOCK_SIZE_K // 2) * 2`); the real parts are loaded from
`x_ptrs`, the imaginary parts from `x_ptrs + 1`, the per-lane rotation angle is
computed by the inlined `get_freq_multi_tokens` helper
(`freqs = (arange(NB_TOKENS) + starting_idx)[:, None] / theta^((offs_cn % DIM)/DIM)`),
and the rotated pair `(real·cos − imag·sin, real·sin + imag·cos)` is stored to
`out_ptrs` / `out_ptrs + 1` under the same masks.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`rbe_triton_wrapper`, the
`(batch, cdiv(M,BM)·cdiv(K,BK))` grid, the `THETA=10000., BLOCK_SIZE_M=2,
BLOCK_SIZE_K=1024` choices, and how the runtime composes per-program writes
into one buffer) is the *trusted boundary*, not a proof obligation here.
Program ids are universally quantified (`s.pids 0` / `s.pids 1`), so the
per-program statement covers every program of the grid.

## Proof architecture

```
rbe_triton_transform_output_summary_general        ← TOP THEOREM (genuine, dimension-general)
  ├─ rbe_triton_surface_toAlgorithm_supported      full surface → algorithm layer
  ├─ rbe_real_compute_correct                      even-offset (out_real) store
  │    └─ rbe_exec_real                            per-lane exec readback
  └─ rbe_imag_compute_correct                      odd-offset (out_imag) store
       └─ rbe_exec_imag                            per-lane exec readback
        ← ComputeCorrect: each OUT store = the genuine rotary closed form
          (rbeOutRealSpec / rbeOutImagSpec over INPUT memory and exact
          Real.cos / Real.sin / Real.rpow), NOT the kernel's executed value
```

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float: `tl.cos` / `tl.sin` are
the exact `Real.cos` / `Real.sin`, `tl.extra.cuda.libdevice.pow` is the exact
`Real.rpow`, and the int→float register casts are value-preserving `Nat → ℝ`
coercions. Address arithmetic is exact `Nat`. `tl.debug_barrier()` is an
intra-program fence — a semantic no-op under the sequential per-program
semantics — and erases at the algorithm layer. Honest side conditions of the
summary: the even-offset output footprint is injective (`hOutInj`) and the
odd-offset (imag) cells never collide with the even-offset (real) cells
(`hRI`) — both hold for the wrapper's contiguous row-major layout
(`stride_out_n = 1`, `stride_out_m = K`, `stride_out_batch = M·K`).

## Translation-surface blocker

Translation-surface blocker: the helper `@triton.jit`
`get_freq_multi_tokens(offs_cn, starting_idx, theta, NB_TOKENS)` is inlined at
its single call site (`cos, sin = get_freq_multi_tokens(...)` becomes the
inlined statement sequence ending in the two bindings `cos = ...` /
`sin = ...`), with:
- the helper-local `DIM: tl.constexpr = 128` generalized to a `DIM : Nat`
  binder (the benchmark value is 128), and `NB_TOKENS` instantiated to
  `BLOCK_SIZE_M` as at the call site;
- Python's in-helper rebindings of `freqs` (which change dtype and shape:
  int 1-D → float 1-D → float 2-D) given fresh register names
  `freqs` / `freqs_f` / `freqs_p` / `freqs_mn` — the DSL types registers, so a
  dtype/shape-changing rebind cannot reuse a name;
- the helper's `freqs.to(tl.float32)` int→float register cast spelled with the
  DSL's nat→real cast `tl.toReal(freqs)`; likewise the implicit int→float
  promotion of `(tl.arange(0, NB_TOKENS) + starting_idx)` in the broadcast
  division is the explicit pair `tks` / `tks_f = tl.toReal(tks)` (the DSL has
  no implicit nat→real promotion and no subscript on call expressions).
`THETA: tl.constexpr` is a real binder `THETA : ℝ` (10000.0 at launch);
`start_token_position` is a `Nat` binder. Everything else — including the
in-kernel `pid // tl.cdiv(K, BLOCK_SIZE_K)` / `pid % tl.cdiv(K, BLOCK_SIZE_K)`
CTA decomposition and both `tl.debug_barrier()` calls — is transcribed
verbatim.
-/

namespace VeriTile.Bench.TritonBenchG.RbeTritonTransform

open VeriTile

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! **★ Main theorem:** `rbe_triton_transform_output_summary_general`
(dimension-general `M`, `K`, all six strides, `start_token_position`, `THETA`,
`DIM`, `BLOCK_SIZE_M`, `BLOCK_SIZE_K`); the Python benchmark shapes are
instantiations. -/

/-! ## Surface -/

/-- Faithful transcription of `rbe_triton_transform.py`'s `rbe_triton`, with
`get_freq_multi_tokens` inlined at its single call site (see the
Translation-surface blocker preamble for the inlining conventions). -/
def rbe_triton_surface
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      start_token_position : Nat)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  pid_batch = tl.program_id(axis=0)
  pid = tl.program_id(axis=1)
  pid_m = pid // tl.cdiv($(K), $(BLOCK_SIZE_K))
  pid_n = pid % tl.cdiv($(K), $(BLOCK_SIZE_K))

  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_K) + tl.arange(0, $(BLOCK_SIZE_K) // $(2)) * $(2)
  x_ptrs = x_ptr + (pid_batch * $(stride_x_batch) + $(stride_x_m) * offs_m[:, None] +
    $(stride_x_n) * offs_n[None, :])
  x_real_mask = (offs_m[:, None] < $(M)) & (offs_n[None, :] < $(K))
  real = tl.load(x_ptrs, mask=x_real_mask, other=0.0)
  x_imag_mask = (offs_m[:, None] < $(M)) & ($((1 : Nat)) + offs_n[None, :] < $(K))
  imag = tl.load(x_ptrs + $(1), mask=x_imag_mask, other=0.0)
  tl.debug_barrier()
  start_block = $((start_token_position : Nat)) + pid_m * $(BLOCK_SIZE_M)
  freqs = offs_n % $(DIM)
  freqs_f = tl.toReal(freqs) / $(DIM)
  freqs_p = tl.extra.cuda.libdevice.pow($((THETA : ℝ)), freqs_f)
  tks = tl.arange(0, $(BLOCK_SIZE_M)) + start_block
  tks_f = tl.toReal(tks)
  freqs_mn = tks_f[:, None] / freqs_p[None, :]
  cos = tl.cos(freqs_mn)
  sin = tl.sin(freqs_mn)

  out_real = real * cos - imag * sin
  out_imag = real * sin + imag * cos
  tl.debug_barrier()
  out_ptrs = out_ptr + (pid_batch * $(stride_out_batch) + $(stride_out_m) * offs_m[:, None] +
    $(stride_out_n) * offs_n[None, :])
  out_real_mask = (offs_m[:, None] < $(M)) & (offs_n[None, :] < $(K))
  tl.store(out_ptrs, out_real, mask=out_real_mask)
  out_imag_mask = (offs_m[:, None] < $(M)) & ($((1 : Nat)) + offs_n[None, :] < $(K))
  tl.store(out_ptrs + $(1), out_imag, mask=out_imag_mask)
}

/-- The full rbe surface (both barriers, both masked loads/stores, the inlined
helper) lowers to the algorithm layer. -/
theorem rbe_triton_surface_toAlgorithm_supported
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      start_token_position : Nat)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) :
    ∃ alg, (rbe_triton_surface x_ptr out_ptr M K stride_x_batch stride_x_m
      stride_x_n stride_out_batch stride_out_m stride_out_n
      start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K).toAlgorithm?
        = Except.ok alg := by
  simp [rbe_triton_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## Index and offset bookkeeping -/

/-- `tl.cdiv(K, BLOCK_SIZE_K)` at the algorithm layer. -/
def kCdiv (K BLOCK_SIZE_K : Nat) : Nat := (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K

/-- `pid_m = pid // tl.cdiv(K, BLOCK_SIZE_K)` of this program. -/
def pidM (s : BlockState) (K BLOCK_SIZE_K : Nat) : Nat :=
  s.pids 1 / kCdiv K BLOCK_SIZE_K

/-- `pid_n = pid % tl.cdiv(K, BLOCK_SIZE_K)` of this program. -/
def pidN (s : BlockState) (K BLOCK_SIZE_K : Nat) : Nat :=
  s.pids 1 % kCdiv K BLOCK_SIZE_K

/-- Global row `offs_m[i] = pid_m·BLOCK_SIZE_M + i` covered by tile lane `i`. -/
def rowIdx (s : BlockState) (K BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (i : Fin BLOCK_SIZE_M) : Nat :=
  pidM s K BLOCK_SIZE_K * BLOCK_SIZE_M + i.val

/-- Global (even) column `offs_n[j] = pid_n·BLOCK_SIZE_K + 2j` covered by tile
lane `j`. -/
def colIdx (s : BlockState) (K BLOCK_SIZE_K : Nat)
    (j : Fin (BLOCK_SIZE_K / 2)) : Nat :=
  pidN s K BLOCK_SIZE_K * BLOCK_SIZE_K + j.val * 2

/-- Input address of the real part read by lane `(i, j)` (the imaginary part
sits at `+ 1`). -/
def xOff (s : BlockState)
    (K stride_x_batch stride_x_m stride_x_n BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]) : Nat :=
  s.pids 0 * stride_x_batch +
    stride_x_m * rowIdx s K BLOCK_SIZE_M BLOCK_SIZE_K idx.1 +
    stride_x_n * colIdx s K BLOCK_SIZE_K idx.2.1

/-- Output address of the real part written by lane `(i, j)` (the imaginary
part is stored at `+ 1`). -/
def outOff (s : BlockState)
    (K stride_out_batch stride_out_m stride_out_n
      BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]) : Nat :=
  s.pids 0 * stride_out_batch +
    stride_out_m * rowIdx s K BLOCK_SIZE_M BLOCK_SIZE_K idx.1 +
    stride_out_n * colIdx s K BLOCK_SIZE_K idx.2.1

/-- `x_real_mask` / `out_real_mask` of lane `(i, j)`:
`offs_m[i] < M ∧ offs_n[j] < K`. -/
def activeReal (s : BlockState) (M K BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]) : Prop :=
  rowIdx s K BLOCK_SIZE_M BLOCK_SIZE_K idx.1 < M ∧
    colIdx s K BLOCK_SIZE_K idx.2.1 < K

instance activeRealDecidable (s : BlockState)
    (M K BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]) :
    Decidable (activeReal s M K BLOCK_SIZE_M BLOCK_SIZE_K idx) := by
  unfold activeReal
  infer_instance

/-- `x_imag_mask` / `out_imag_mask` of lane `(i, j)`:
`offs_m[i] < M ∧ 1 + offs_n[j] < K`. -/
def activeImag (s : BlockState) (M K BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]) : Prop :=
  rowIdx s K BLOCK_SIZE_M BLOCK_SIZE_K idx.1 < M ∧
    1 + colIdx s K BLOCK_SIZE_K idx.2.1 < K

instance activeImagDecidable (s : BlockState)
    (M K BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]) :
    Decidable (activeImag s M K BLOCK_SIZE_M BLOCK_SIZE_K idx) := by
  unfold activeImag
  infer_instance

/-! ## Closed-form spec

The genuine rotary closed form, a pure function of INPUT memory and the exact
`Real.cos` / `Real.sin` / `Real.rpow`: lane `(i, j)` rotates the input pair
`(x[xOff], x[xOff + 1])` by the angle
`freq = (start_token_position + offs_m[i]) / THETA^((offs_n[j] % DIM) / DIM)`. -/

/-- Rotation angle of lane `(i, j)`:
`(i + (start_token_position + pid_m·BLOCK_SIZE_M)) / THETA^((offs_n[j] % DIM)/DIM)`
— the numerator is exactly `start_token_position + offs_m[i]`
(see `freqSpec_eq_token`). -/
noncomputable def freqSpec (s : BlockState)
    (K start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) (THETA : ℝ)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]) : ℝ :=
  ((idx.1.val + (start_token_position + pidM s K BLOCK_SIZE_K * BLOCK_SIZE_M)
      : ℕ) : ℝ) /
    Real.rpow THETA
      (((colIdx s K BLOCK_SIZE_K idx.2.1 % DIM : ℕ) : ℝ) / ((DIM : ℕ) : ℝ))

/-- The angle's numerator is `start_token_position + offs_m[i]` — the token
position this row covers. -/
theorem freqSpec_eq_token (s : BlockState)
    (K start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) (THETA : ℝ)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]) :
    freqSpec s K start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K THETA idx
      = ((start_token_position + rowIdx s K BLOCK_SIZE_M BLOCK_SIZE_K idx.1
            : ℕ) : ℝ) /
        Real.rpow THETA
          (((colIdx s K BLOCK_SIZE_K idx.2.1 % DIM : ℕ) : ℝ) /
            ((DIM : ℕ) : ℝ)) := by
  unfold freqSpec rowIdx
  congr 2
  omega

/-- Genuine spec of the even-offset (`out_real`) store on an active lane:
`x_real·cos(freq) − x_imag·sin(freq)`, where `x_imag` is the masked imaginary
load (`0` on the `1 + offs_n[j] ≥ K` boundary lane, matching `x_imag_mask`'s
`other=0.0`). -/
noncomputable def rbeOutRealSpec (s : BlockState) (x_ptr : RegionName)
    (K stride_x_batch stride_x_m stride_x_n
      start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) (THETA : ℝ)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]) : ℝ :=
  s.readMem x_ptr
      (xOff s K stride_x_batch stride_x_m stride_x_n BLOCK_SIZE_M
        BLOCK_SIZE_K idx) *
    Real.cos (freqSpec s K start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K
      THETA idx) -
  (if 1 + colIdx s K BLOCK_SIZE_K idx.2.1 < K then
      s.readMem x_ptr
        (xOff s K stride_x_batch stride_x_m stride_x_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx + 1)
    else 0) *
    Real.sin (freqSpec s K start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K
      THETA idx)

/-- Genuine spec of the odd-offset (`out_imag`) store on an active lane:
`x_real·sin(freq) + x_imag·cos(freq)` (on an `x_imag_mask`-active lane the
real-part load is also in bounds, so both reads are genuine). -/
noncomputable def rbeOutImagSpec (s : BlockState) (x_ptr : RegionName)
    (K stride_x_batch stride_x_m stride_x_n
      start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) (THETA : ℝ)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]) : ℝ :=
  s.readMem x_ptr
      (xOff s K stride_x_batch stride_x_m stride_x_n BLOCK_SIZE_M
        BLOCK_SIZE_K idx) *
    Real.sin (freqSpec s K start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K
      THETA idx) +
  s.readMem x_ptr
      (xOff s K stride_x_batch stride_x_m stride_x_n BLOCK_SIZE_M
        BLOCK_SIZE_K idx + 1) *
    Real.cos (freqSpec s K start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K
      THETA idx)

/-! ## Per-lane exec readback -/

/-- Algorithm-layer correctness of the even-offset (`out_real`) store: after
executing the full kernel, every `out_real_mask`-active lane holds the genuine
rotation `rbeOutRealSpec`; inactive lanes are preserved. -/
theorem rbe_exec_real
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      start_token_position : Nat)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
        outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx))
    (hRI : ∀ idx k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2],
      outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx
        ≠ outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
            BLOCK_SIZE_K k + 1)
    (hExec : exec (rbe_triton_surface x_ptr out_ptr M K stride_x_batch
        stride_x_m stride_x_n stride_out_batch stride_out_m stride_out_n
        start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K) s
      = some s') :
    ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2],
      s'.readMem out_ptr
          (outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
            BLOCK_SIZE_K idx)
        = if activeReal s M K BLOCK_SIZE_M BLOCK_SIZE_K idx then
            rbeOutRealSpec s x_ptr K stride_x_batch stride_x_m stride_x_n
              start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K THETA idx
          else
            s.readMem out_ptr
              (outOff s K stride_out_batch stride_out_m stride_out_n
                BLOCK_SIZE_M BLOCK_SIZE_K idx) := by
  intro idx
  simp only [outOff, rowIdx, colIdx, pidM, pidN, kCdiv] at hOutInj hRI
  simp [exec, rbe_triton_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.natToReal, Tile.expandDim,
        TileShape.insertAxis, TileShape.dropInsertedIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, IntegralDType.mod, IntegralDType.floorDiv,
        ComparableDType.lt] at hExec
  rw [← hExec]
  simp only [outOff, rowIdx, colIdx, pidM, pidN, kCdiv]
  -- Strip the outer `out_imag` store: it writes only odd-side offsets
  -- (`… + 1`), disjoint from the even-side read offset (`hRI`).
  rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
        (region := out_ptr)
        (P := fun k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
          s.pids 1 / ((K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K) * BLOCK_SIZE_M
              + k.1.val < M ∧
            1 + (s.pids 1 % ((K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K)
              * BLOCK_SIZE_K + k.2.1.val * 2) < K)
        (offsetFn := fun k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
          s.pids 0 * stride_out_batch +
              stride_out_m * (s.pids 1 / ((K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K)
                * BLOCK_SIZE_M + k.1.val) +
            stride_out_n * (s.pids 1 % ((K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K)
              * BLOCK_SIZE_K + k.2.1.val * 2) + 1)
        (hOff := fun k _ _ => hRI idx k)]
  rw [BlockState.setReg_readMem]
  -- Read back the `out_real` scatter at lane `idx`.
  have hRawInj : Function.Injective
      (fun k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
        s.pids 0 * stride_out_batch +
            stride_out_m * (s.pids 1 / ((K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K)
              * BLOCK_SIZE_M + k.1.val) +
          stride_out_n * (s.pids 1 % ((K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K)
            * BLOCK_SIZE_K + k.2.1.val * 2)) := hOutInj
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj idx]
  by_cases hRow : s.pids 1 / ((K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K)
      * BLOCK_SIZE_M + idx.1.val < M
  · by_cases hCol : s.pids 1 % ((K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K)
        * BLOCK_SIZE_K + idx.2.1.val * 2 < K
    · by_cases hImag : 1 + (s.pids 1 % ((K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K)
          * BLOCK_SIZE_K + idx.2.1.val * 2) < K
      · simp [activeReal, rbeOutRealSpec, freqSpec, xOff, outOff, rowIdx,
              colIdx, pidM, pidN, kCdiv, hRow, hCol, hImag,
              Option.map₂, Option.bind, Option.map]
      · simp [activeReal, rbeOutRealSpec, freqSpec, xOff, outOff, rowIdx,
              colIdx, pidM, pidN, kCdiv, hRow, hCol, hImag,
              Option.map₂, Option.bind, Option.map]
        norm_num
    · simp [activeReal, rowIdx, colIdx, pidM, pidN, kCdiv, hRow, hCol]
  · simp [activeReal, rowIdx, colIdx, pidM, pidN, kCdiv, hRow]

/-- Algorithm-layer correctness of the odd-offset (`out_imag`) store: after
executing the full kernel, every `out_imag_mask`-active lane holds the genuine
rotation `rbeOutImagSpec`; inactive lanes are preserved. -/
theorem rbe_exec_imag
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      start_token_position : Nat)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
        outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx))
    (hRI : ∀ idx k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2],
      outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx
        ≠ outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
            BLOCK_SIZE_K k + 1)
    (hExec : exec (rbe_triton_surface x_ptr out_ptr M K stride_x_batch
        stride_x_m stride_x_n stride_out_batch stride_out_m stride_out_n
        start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K) s
      = some s') :
    ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2],
      s'.readMem out_ptr
          (outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
            BLOCK_SIZE_K idx + 1)
        = if activeImag s M K BLOCK_SIZE_M BLOCK_SIZE_K idx then
            rbeOutImagSpec s x_ptr K stride_x_batch stride_x_m stride_x_n
              start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K THETA idx
          else
            s.readMem out_ptr
              (outOff s K stride_out_batch stride_out_m stride_out_n
                BLOCK_SIZE_M BLOCK_SIZE_K idx + 1) := by
  intro idx
  simp only [outOff, rowIdx, colIdx, pidM, pidN, kCdiv] at hOutInj hRI
  simp [exec, rbe_triton_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.natToReal, Tile.expandDim,
        TileShape.insertAxis, TileShape.dropInsertedIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, IntegralDType.mod, IntegralDType.floorDiv,
        ComparableDType.lt] at hExec
  rw [← hExec]
  simp only [outOff, rowIdx, colIdx, pidM, pidN, kCdiv]
  -- Read back the outer `out_imag` scatter at lane `idx` (odd-side offsets
  -- `… + 1` inherit injectivity from the even-side map).
  have hRawInj1 : Function.Injective
      (fun k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
        s.pids 0 * stride_out_batch +
            stride_out_m * (s.pids 1 / ((K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K)
              * BLOCK_SIZE_M + k.1.val) +
          stride_out_n * (s.pids 1 % ((K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K)
            * BLOCK_SIZE_K + k.2.1.val * 2) + 1) := by
    intro a b h
    exact hOutInj (Nat.add_right_cancel h)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj1 idx]
  -- On the preserved branch, strip the inner `out_real` store: it writes only
  -- even-side offsets, disjoint from the odd-side read offset (`hRI`).
  rw [BlockState.setReg_readMem]
  rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
        (region := out_ptr)
        (P := fun k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
          s.pids 1 / ((K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K) * BLOCK_SIZE_M
              + k.1.val < M ∧
            s.pids 1 % ((K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K)
              * BLOCK_SIZE_K + k.2.1.val * 2 < K)
        (offsetFn := fun k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
          s.pids 0 * stride_out_batch +
              stride_out_m * (s.pids 1 / ((K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K)
                * BLOCK_SIZE_M + k.1.val) +
            stride_out_n * (s.pids 1 % ((K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K)
              * BLOCK_SIZE_K + k.2.1.val * 2))
        (hOff := fun k _ _ => (hRI k idx).symm)]
  by_cases hRow : s.pids 1 / ((K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K)
      * BLOCK_SIZE_M + idx.1.val < M
  · by_cases hImag : 1 + (s.pids 1 % ((K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K)
        * BLOCK_SIZE_K + idx.2.1.val * 2) < K
    · have hCol : s.pids 1 % ((K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K)
          * BLOCK_SIZE_K + idx.2.1.val * 2 < K := by omega
      simp [activeImag, rbeOutImagSpec, freqSpec, xOff, outOff, rowIdx,
            colIdx, pidM, pidN, kCdiv, hRow, hCol, hImag,
            Option.map₂, Option.bind, Option.map]
    · simp [activeImag, rowIdx, colIdx, pidM, pidN, kCdiv, hRow, hImag]
  · simp [activeImag, rowIdx, colIdx, pidM, pidN, kCdiv, hRow]

/-! ## Compute-facing correctness -/

/-- Compute-facing correctness of the even-offset (`out_real`) store: every
`out_real_mask`-active lane of the full kernel reads back to the genuine
rotation `rbeOutRealSpec` — a closed form over INPUT memory and exact
`Real.cos`/`Real.sin`/`Real.rpow`, NOT the kernel's executed value. -/
theorem rbe_real_compute_correct
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      start_token_position : Nat)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
        outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx))
    (hRI : ∀ idx k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2],
      outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx
        ≠ outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
            BLOCK_SIZE_K k + 1) :
    ComputeCorrect.Realizes
      (kernel := rbe_triton_surface x_ptr out_ptr M K stride_x_batch
        stride_x_m stride_x_n stride_out_batch stride_out_m stride_out_n
        start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
          activeReal s M K BLOCK_SIZE_M BLOCK_SIZE_K idx)
        (fun idx => (out_ptr,
          outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
            BLOCK_SIZE_K idx)))
      (expected := fun idx =>
        rbeOutRealSpec s x_ptr K stride_x_batch stride_x_m stride_x_n
          start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K THETA idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rbe_triton_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := rbe_exec_real x_ptr out_ptr M K stride_x_batch stride_x_m
    stride_x_n stride_out_batch stride_out_m stride_out_n
    start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K s s' hOutInj hRI
    hExec idx
  simpa [hActive] using h

/-- Compute-facing correctness of the odd-offset (`out_imag`) store: every
`out_imag_mask`-active lane of the full kernel reads back to the genuine
rotation `rbeOutImagSpec`. -/
theorem rbe_imag_compute_correct
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      start_token_position : Nat)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
        outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx))
    (hRI : ∀ idx k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2],
      outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx
        ≠ outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
            BLOCK_SIZE_K k + 1) :
    ComputeCorrect.Realizes
      (kernel := rbe_triton_surface x_ptr out_ptr M K stride_x_batch
        stride_x_m stride_x_n stride_out_batch stride_out_m stride_out_n
        start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
          activeImag s M K BLOCK_SIZE_M BLOCK_SIZE_K idx)
        (fun idx => (out_ptr,
          outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
            BLOCK_SIZE_K idx + 1)))
      (expected := fun idx =>
        rbeOutImagSpec s x_ptr K stride_x_batch stride_x_m stride_x_n
          start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K THETA idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rbe_triton_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := rbe_exec_imag x_ptr out_ptr M K stride_x_batch stride_x_m
    stride_x_n stride_out_batch stride_out_m stride_out_n
    start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K s s' hOutInj hRI
    hExec idx
  simpa [hActive] using h

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **Dimension-general** correctness summary for `rbe_triton_transform.py`'s
`rbe_triton`, against the **genuine rotary closed form** — a pure function of
INPUT memory and the exact `Real.cos` / `Real.sin` / `Real.rpow`, never a
read-back of the kernel's own output — for arbitrary `M`, `K`, all six
strides, `start_token_position`, `THETA`, `DIM`, `BLOCK_SIZE_M`,
`BLOCK_SIZE_K`, and arbitrary program ids (`s.pids 0` = batch,
`s.pids 1` = the fused `(m, k)` CTA id decomposed in-kernel). It packages:

* the full faithful surface (both `tl.debug_barrier()` calls, both masked
  loads/stores, the inlined `get_freq_multi_tokens`) lowers to the algorithm
  layer;
* the even-offset (`out_real`) store: every `out_real_mask`-active lane
  `(i, j)` holds `x_real·cos(freq) − x_imag·sin(freq)` where
  `freq = (start_token_position + offs_m[i]) / THETA^((offs_n[j] % DIM)/DIM)`
  (`freqSpec_eq_token`) and `x_imag` is the masked imaginary load (`0` on the
  `1 + offs_n[j] ≥ K` boundary lane);
* the odd-offset (`out_imag`) store: every `out_imag_mask`-active lane holds
  `x_real·sin(freq) + x_imag·cos(freq)`.

Honest side conditions: the even-offset output footprint is injective
(`hOutInj`) and even-offset cells never collide with odd-offset (`+ 1`) cells
(`hRI`) — both hold for the wrapper's contiguous row-major layout. -/
theorem rbe_triton_transform_output_summary_general
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      start_token_position : Nat)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
        outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx))
    (hRI : ∀ idx k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2],
      outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx
        ≠ outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
            BLOCK_SIZE_K k + 1) :
    -- (1) the full faithful surface lowers to the algorithm layer
    (∃ alg, (rbe_triton_surface x_ptr out_ptr M K stride_x_batch stride_x_m
      stride_x_n stride_out_batch stride_out_m stride_out_n
      start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K).toAlgorithm?
        = Except.ok alg) ∧
    -- (2) even offsets: genuine `x_real·cos − x_imag·sin`
    ComputeCorrect.Realizes
      (kernel := rbe_triton_surface x_ptr out_ptr M K stride_x_batch
        stride_x_m stride_x_n stride_out_batch stride_out_m stride_out_n
        start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
          activeReal s M K BLOCK_SIZE_M BLOCK_SIZE_K idx)
        (fun idx => (out_ptr,
          outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
            BLOCK_SIZE_K idx)))
      (expected := fun idx =>
        rbeOutRealSpec s x_ptr K stride_x_batch stride_x_m stride_x_n
          start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K THETA idx) ∧
    -- (3) odd offsets: genuine `x_real·sin + x_imag·cos`
    ComputeCorrect.Realizes
      (kernel := rbe_triton_surface x_ptr out_ptr M K stride_x_batch
        stride_x_m stride_x_n stride_out_batch stride_out_m stride_out_n
        start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
          activeImag s M K BLOCK_SIZE_M BLOCK_SIZE_K idx)
        (fun idx => (out_ptr,
          outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
            BLOCK_SIZE_K idx + 1)))
      (expected := fun idx =>
        rbeOutImagSpec s x_ptr K stride_x_batch stride_x_m stride_x_n
          start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K THETA idx) :=
  ⟨rbe_triton_surface_toAlgorithm_supported x_ptr out_ptr M K stride_x_batch
      stride_x_m stride_x_n stride_out_batch stride_out_m stride_out_n
      start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K,
    rbe_real_compute_correct x_ptr out_ptr M K stride_x_batch stride_x_m
      stride_x_n stride_out_batch stride_out_m stride_out_n
      start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K s hOutInj hRI,
    rbe_imag_compute_correct x_ptr out_ptr M K stride_x_batch stride_x_m
      stride_x_n stride_out_batch stride_out_m stride_out_n
      start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K s hOutInj hRI⟩

end VeriTile.Bench.TritonBenchG.RbeTritonTransform
