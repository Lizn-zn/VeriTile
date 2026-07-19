import VeriTile.Triton

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

open VeriTile.Triton
open scoped VeriTile.Triton.GroupedMasked2DKernelIO

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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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


/-! ## The `⊨` specification surface (`GroupedMasked2DKernelIO`)

`rbe_triton` reads two interleaved channels (the real part at `x_ptrs` and the
imaginary part at `x_ptrs + 1`) and writes two interleaved channels (`out_ptrs`
and `out_ptrs + 1`), so its IO signature is `nIn = 2`, `nOut = 2`. The grouped
skin indexes lanes by a flat `Fin B`, while the kernel's tile is the 2D
`[BLOCK_SIZE_M, BLOCK_SIZE_K / 2]`; `rbeLane` is the row-major decode of one
flat lane into that tile index, and `rbeLaneInv` its inverse. -/

/-- Row-major decode of a flat lane into the kernel's 2D tile index. -/
def rbeLane (BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (j : Fin (BLOCK_SIZE_M * (BLOCK_SIZE_K / 2))) :
    TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] :=
  have hH : 0 < BLOCK_SIZE_K / 2 := by
    rcases Nat.eq_zero_or_pos (BLOCK_SIZE_K / 2) with h | h
    · exact absurd j.isLt (by simp [h])
    · exact h
  (⟨j.val / (BLOCK_SIZE_K / 2), Nat.div_lt_of_lt_mul
      (Nat.lt_of_lt_of_le j.isLt (Nat.le_of_eq (Nat.mul_comm _ _)))⟩,
   ⟨j.val % (BLOCK_SIZE_K / 2), Nat.mod_lt _ hH⟩,
   PUnit.unit)

/-- Row-major encode of a 2D tile index into a flat lane. -/
def rbeLaneInv (BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]) :
    Fin (BLOCK_SIZE_M * (BLOCK_SIZE_K / 2)) :=
  ⟨idx.2.1.val + idx.1.val * (BLOCK_SIZE_K / 2), by
    calc idx.2.1.val + idx.1.val * (BLOCK_SIZE_K / 2)
        < (BLOCK_SIZE_K / 2) + idx.1.val * (BLOCK_SIZE_K / 2) :=
          Nat.add_lt_add_right idx.2.1.isLt _
      _ = (idx.1.val + 1) * (BLOCK_SIZE_K / 2) := by ring
      _ ≤ BLOCK_SIZE_M * (BLOCK_SIZE_K / 2) :=
          Nat.mul_le_mul_right _ idx.1.isLt⟩

@[simp] theorem rbeLane_rbeLaneInv (BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]) :
    rbeLane BLOCK_SIZE_M BLOCK_SIZE_K
        (rbeLaneInv BLOCK_SIZE_M BLOCK_SIZE_K idx) = idx := by
  have hH : 0 < BLOCK_SIZE_K / 2 := Nat.lt_of_le_of_lt (Nat.zero_le _) idx.2.1.isLt
  obtain ⟨r, c, u⟩ := idx
  refine Prod.ext ?_ (Prod.ext ?_ rfl)
  · refine Fin.ext ?_
    show (c.val + r.val * (BLOCK_SIZE_K / 2)) / (BLOCK_SIZE_K / 2) = r.val
    rw [Nat.add_mul_div_right _ _ hH, Nat.div_eq_of_lt c.isLt, Nat.zero_add]
  · refine Fin.ext ?_
    show (c.val + r.val * (BLOCK_SIZE_K / 2)) % (BLOCK_SIZE_K / 2) = c.val
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt c.isLt]

theorem rbeLane_injective (BLOCK_SIZE_M BLOCK_SIZE_K : Nat) :
    Function.Injective (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K) := by
  intro a b hab
  have hH : 0 < BLOCK_SIZE_K / 2 := by
    rcases Nat.eq_zero_or_pos (BLOCK_SIZE_K / 2) with h | h
    · exact absurd a.isLt (by simp [h])
    · exact h
  have hd : a.val / (BLOCK_SIZE_K / 2) = b.val / (BLOCK_SIZE_K / 2) :=
    congrArg (fun t => (Prod.fst t).val) hab
  have hm : a.val % (BLOCK_SIZE_K / 2) = b.val % (BLOCK_SIZE_K / 2) :=
    congrArg (fun t => (Prod.fst (Prod.snd t)).val) hab
  refine Fin.ext ?_
  calc a.val = (BLOCK_SIZE_K / 2) * (a.val / (BLOCK_SIZE_K / 2))
                + a.val % (BLOCK_SIZE_K / 2) := (Nat.div_add_mod _ _).symm
    _ = (BLOCK_SIZE_K / 2) * (b.val / (BLOCK_SIZE_K / 2))
          + b.val % (BLOCK_SIZE_K / 2) := by rw [hd, hm]
    _ = b.val := Nat.div_add_mod _ _

/-- Termination: the full rbe surface executes to completion from any state. -/
private theorem rbe_triton_surface_exec_isSome
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      start_token_position : Nat)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) (s : BlockState) :
    ∃ s1, exec ((rbe_triton_surface x_ptr out_ptr M K stride_x_batch
        stride_x_m stride_x_n stride_out_batch stride_out_m stride_out_n
        start_token_position THETA DIM BLOCK_SIZE_M
        BLOCK_SIZE_K).toAlgKernel) s = some s1 := by
  simp [exec, rbe_triton_surface, ComputeKernel.toAlgKernel,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.natToReal, Tile.expandDim,
        TileShape.insertAxis, TileShape.dropInsertedIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, IntegralDType.mod, IntegralDType.floorDiv,
        ComparableDType.lt]

/-- The full rbe surface sits inside the flat-memory bridge's covered
fragment. -/
private theorem rbe_triton_surface_flattenOk
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      start_token_position : Nat)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) :
    ((rbe_triton_surface x_ptr out_ptr M K stride_x_batch stride_x_m
        stride_x_n stride_out_batch stride_out_m stride_out_n
        start_token_position THETA DIM BLOCK_SIZE_M
        BLOCK_SIZE_K).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [rbe_triton_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]


/-- A masked scatter-store `foldl` leaves every memory cell it does not
actively hit unchanged (cell-level frame for one masked store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (r : RegionName) (o : Nat) (l : List α) (st : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      st).mem r o = st.mem r o := by
  induction l generalizing st with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP,
          ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMem_mem]
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

/-- Frame half: every memory cell outside the two masked output windows (the
even-offset `out_real` scatter and the odd-offset `out_imag` scatter) is
preserved by the run. -/
private theorem rbe_triton_surface_frame
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      start_token_position : Nat)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) (s s1 : BlockState)
    (hExec : exec ((rbe_triton_surface x_ptr out_ptr M K stride_x_batch
        stride_x_m stride_x_n stride_out_batch stride_out_m stride_out_n
        start_token_position THETA DIM BLOCK_SIZE_M
        BLOCK_SIZE_K).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmissReal : ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2],
      activeReal s M K BLOCK_SIZE_M BLOCK_SIZE_K idx →
      ¬(out_ptr = r ∧ outOff s K stride_out_batch stride_out_m stride_out_n
          BLOCK_SIZE_M BLOCK_SIZE_K idx = o))
    (hmissImag : ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2],
      activeImag s M K BLOCK_SIZE_M BLOCK_SIZE_K idx →
      ¬(out_ptr = r ∧ outOff s K stride_out_batch stride_out_m stride_out_n
          BLOCK_SIZE_M BLOCK_SIZE_K idx + 1 = o)) :
    s1.mem r o = s.mem r o := by
  simp only [activeReal, activeImag, outOff, rowIdx, colIdx, pidM, pidN,
    kCdiv] at hmissReal hmissImag
  simp [exec, rbe_triton_surface, ComputeKernel.toAlgKernel,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.natToReal, Tile.expandDim,
        TileShape.insertAxis, TileShape.dropInsertedIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, IntegralDType.mod, IntegralDType.floorDiv,
        ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) ?_
  · intro k _ hmk hc
    exact hmissImag k hmk hc
  · simp only [BlockState.setReg]
    refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
    intro k _ hmk hc
    exact hmissReal k hmk hc


/-- Per-execution safety walk for the full rbe surface: the two masked loads
(`real` at `x_ptrs`, `imag` at `x_ptrs + 1`) and the two masked stores
(`out_real` at `out_ptrs`, `out_imag` at `out_ptrs + 1`) reduce to the
lane-wise bounds hypotheses; every other statement is memory-silent. -/
private theorem rbe_triton_surface_traceSafe
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      start_token_position : Nat)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (h1 : ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2],
      activeReal s M K BLOCK_SIZE_M BLOCK_SIZE_K idx →
      xOff s K stride_x_batch stride_x_m stride_x_n BLOCK_SIZE_M BLOCK_SIZE_K idx
        < bounds x_ptr)
    (h2 : ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2],
      activeImag s M K BLOCK_SIZE_M BLOCK_SIZE_K idx →
      xOff s K stride_x_batch stride_x_m stride_x_n BLOCK_SIZE_M BLOCK_SIZE_K idx
          + 1 < bounds x_ptr)
    (h3 : ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2],
      activeReal s M K BLOCK_SIZE_M BLOCK_SIZE_K idx →
      outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx < bounds out_ptr)
    (h4 : ∀ idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2],
      activeImag s M K BLOCK_SIZE_M BLOCK_SIZE_K idx →
      outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx + 1 < bounds out_ptr) :
    Kernel.TraceSafe bounds
      ((rbe_triton_surface x_ptr out_ptr M K stride_x_batch stride_x_m
        stride_x_n stride_out_batch stride_out_m stride_out_n
        start_token_position THETA DIM BLOCK_SIZE_M
        BLOCK_SIZE_K).toAlgKernel) s := by
  simp only [activeReal, activeImag, xOff, outOff, rowIdx, colIdx, pidM, pidN,
    kCdiv] at h1 h2 h3 h4
  unfold Kernel.TraceSafe
  simp [rbe_triton_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg, tile_elementwise, Bool.and_eq_true,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, Tile.natToReal, Tile.expandDim,
    TileShape.insertAxis, TileShape.dropInsertedIndex,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    NumericDType.div, IntegralDType.mod, IntegralDType.floorDiv,
    ComparableDType.lt]
  exact ⟨fun a b hr hc => h1 (a, b, PUnit.unit) ⟨hr, hc⟩, fun a b hr hc => h2 (a, b, PUnit.unit) ⟨hr, hc⟩,
    fun a b hr hc => h3 (a, b, PUnit.unit) ⟨hr, hc⟩, fun a b hr hc => h4 (a, b, PUnit.unit) ⟨hr, hc⟩⟩


/-! ### Pid-indexed windows

The grouped skin's windows and masks are functions of the program ids
`(pid₀, pid₁)` and the flat lane, not of a `BlockState`. These are exactly the
`xOff` / `outOff` / `activeReal` / `activeImag` / `freqSpec` of the sections
above, read off `(pid₀, pid₁)` instead of `(s.pids 0, s.pids 1)` and indexed by
the decoded lane; the `*_eq` lemmas below record that they agree by `rfl`. -/

/-- Global row `offs_m[i]` covered by flat lane `j`. -/
def rbeRowP (pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (j : Fin (BLOCK_SIZE_M * (BLOCK_SIZE_K / 2))) : Nat :=
  pid₁ / kCdiv K BLOCK_SIZE_K * BLOCK_SIZE_M +
    (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j).1.val

/-- Global (even) column `offs_n[j]` covered by flat lane `j`. -/
def rbeColP (pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (j : Fin (BLOCK_SIZE_M * (BLOCK_SIZE_K / 2))) : Nat :=
  pid₁ % kCdiv K BLOCK_SIZE_K * BLOCK_SIZE_K +
    (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j).2.1.val * 2

/-- Input address of the real part read by flat lane `j`. -/
def rbeXAddrP (pid₀ pid₁ K stride_x_batch stride_x_m stride_x_n
    BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (j : Fin (BLOCK_SIZE_M * (BLOCK_SIZE_K / 2))) : Nat :=
  pid₀ * stride_x_batch +
    stride_x_m * rbeRowP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j +
    stride_x_n * rbeColP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j

/-- Output address of the real part written by flat lane `j`. -/
def rbeOutAddrP (pid₀ pid₁ K stride_out_batch stride_out_m stride_out_n
    BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (j : Fin (BLOCK_SIZE_M * (BLOCK_SIZE_K / 2))) : Nat :=
  pid₀ * stride_out_batch +
    stride_out_m * rbeRowP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j +
    stride_out_n * rbeColP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j

/-- Rotation angle of flat lane `j`. -/
noncomputable def rbeFreqP (pid₁ K start_token_position DIM
    BLOCK_SIZE_M BLOCK_SIZE_K : Nat) (THETA : ℝ)
    (j : Fin (BLOCK_SIZE_M * (BLOCK_SIZE_K / 2))) : ℝ :=
  (((rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j).1.val +
      (start_token_position + pid₁ / kCdiv K BLOCK_SIZE_K * BLOCK_SIZE_M)
        : ℕ) : ℝ) /
    Real.rpow THETA
      (((rbeColP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j % DIM : ℕ) : ℝ) /
        ((DIM : ℕ) : ℝ))

theorem rbeXAddrP_eq (s : BlockState)
    (K stride_x_batch stride_x_m stride_x_n BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (j : Fin (BLOCK_SIZE_M * (BLOCK_SIZE_K / 2))) :
    rbeXAddrP (s.pids 0) (s.pids 1) K stride_x_batch stride_x_m stride_x_n
        BLOCK_SIZE_M BLOCK_SIZE_K j
      = xOff s K stride_x_batch stride_x_m stride_x_n BLOCK_SIZE_M BLOCK_SIZE_K
          (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j) := rfl

theorem rbeOutAddrP_eq (s : BlockState)
    (K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
      BLOCK_SIZE_K : Nat)
    (j : Fin (BLOCK_SIZE_M * (BLOCK_SIZE_K / 2))) :
    rbeOutAddrP (s.pids 0) (s.pids 1) K stride_out_batch stride_out_m
        stride_out_n BLOCK_SIZE_M BLOCK_SIZE_K j
      = outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j) := rfl

theorem rbeRowP_eq (s : BlockState) (K BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (j : Fin (BLOCK_SIZE_M * (BLOCK_SIZE_K / 2))) :
    rbeRowP (s.pids 1) K BLOCK_SIZE_M BLOCK_SIZE_K j
      = rowIdx s K BLOCK_SIZE_M BLOCK_SIZE_K
          (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j).1 := rfl

theorem rbeColP_eq (s : BlockState) (K BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (j : Fin (BLOCK_SIZE_M * (BLOCK_SIZE_K / 2))) :
    rbeColP (s.pids 1) K BLOCK_SIZE_M BLOCK_SIZE_K j
      = colIdx s K BLOCK_SIZE_K
          (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j).2.1 := rfl

theorem rbeFreqP_eq (s : BlockState)
    (K start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) (THETA : ℝ)
    (j : Fin (BLOCK_SIZE_M * (BLOCK_SIZE_K / 2))) :
    rbeFreqP (s.pids 1) K start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K
        THETA j
      = freqSpec s K start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K THETA
          (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j) := rfl

/-- The **IO signature** of `rbe_triton` — the whole kernel-specific audit
surface of the `⊨` headline.

* `bufs = [x_ptr, out_ptr]`, `nIn = 2`, `nOut = 2`,
  `B = BLOCK_SIZE_M · (BLOCK_SIZE_K / 2)` (the kernel's 2D tile flattened
  row-major by `rbeLane`).
* read channel `0` — `x_ptr` at the even address `x_ptrs`, gated by
  `x_real_mask` (`offs_m < M ∧ offs_n < K`);
  read channel `1` — `x_ptr` at the interleaved odd address `x_ptrs + 1`,
  gated by `x_imag_mask` (`offs_m < M ∧ 1 + offs_n < K`).
* write channel `0` — `out_ptr` at `out_ptrs`, gated by `out_real_mask`;
  write channel `1` — `out_ptr` at `out_ptrs + 1`, gated by `out_imag_mask`.

`cos`/`sin` are *not* read channels: the kernel derives them from index
arithmetic (`get_freq_multi_tokens`, inlined), so they appear in the spec
value as the exact `Real.cos`/`Real.sin` of `rbeFreqP`, not as pinned inputs. -/
def rbeTritonIO (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      start_token_position : Nat)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) :
    GroupedMasked2DKernelIO where
  kernel := rbe_triton_surface x_ptr out_ptr M K stride_x_batch stride_x_m
    stride_x_n stride_out_batch stride_out_m stride_out_n
    start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K
  nIn := 2
  nOut := 2
  bufs := [x_ptr, out_ptr]
  inp := fun _ => x_ptr
  out := fun _ => out_ptr
  B := BLOCK_SIZE_M * (BLOCK_SIZE_K / 2)
  read := fun i pid₀ pid₁ j => match i with
    | ⟨0, _⟩ => rbeXAddrP pid₀ pid₁ K stride_x_batch stride_x_m stride_x_n
        BLOCK_SIZE_M BLOCK_SIZE_K j
    | ⟨_ + 1, _⟩ => rbeXAddrP pid₀ pid₁ K stride_x_batch stride_x_m stride_x_n
        BLOCK_SIZE_M BLOCK_SIZE_K j + 1
  readMask := fun i _pid₀ pid₁ j => match i with
    | ⟨0, _⟩ => rbeRowP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j < M ∧
        rbeColP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j < K
    | ⟨_ + 1, _⟩ => rbeRowP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j < M ∧
        1 + rbeColP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j < K
  write := fun o pid₀ pid₁ j => match o with
    | ⟨0, _⟩ => rbeOutAddrP pid₀ pid₁ K stride_out_batch stride_out_m
        stride_out_n BLOCK_SIZE_M BLOCK_SIZE_K j
    | ⟨_ + 1, _⟩ => rbeOutAddrP pid₀ pid₁ K stride_out_batch stride_out_m
        stride_out_n BLOCK_SIZE_M BLOCK_SIZE_K j + 1
  writeMask := fun o _pid₀ pid₁ j => match o with
    | ⟨0, _⟩ => rbeRowP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j < M ∧
        rbeColP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j < K
    | ⟨_ + 1, _⟩ => rbeRowP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j < M ∧
        1 + rbeColP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j < K


/-- **`rbeTritonIO ⊨ rotary`** — the whole `rbe_triton` kernel as one grouped
masked Hoare triple over **flat** pointer memory: for every disjoint
base-pointer placement of `x_ptr`/`out_ptr`, every program `(pid₀, pid₁)` whose
active windows are in bounds, and every launch state whose two interleaved read
windows hold `xs`, the translated pointer kernel terminates, every
`out_real_mask`-active lane of `out_ptrs` holds
`x_real·cos(freq) − x_imag·sin(freq)` (with `x_imag` replaced by `0` on the
`1 + offs_n ≥ K` boundary lane, matching `x_imag_mask`'s `other=0.0`), every
`out_imag_mask`-active lane of `out_ptrs + 1` holds
`x_real·sin(freq) + x_imag·cos(freq)`, and every other flat cell is untouched.

`freq = rbeFreqP` is the exact `(start_token_position + offs_m) /
THETA^((offs_n % DIM)/DIM)` — the inlined `get_freq_multi_tokens`, computed from
index arithmetic rather than read from memory, hence not a read channel.

Dimension-general: `M`, `K`, all six strides, `start_token_position`, `THETA`,
`DIM`, `BLOCK_SIZE_M`, `BLOCK_SIZE_K` are free parameters. The two side
conditions `hOutInj`/`hRI` are the same honest ones the `Realizes` face carries:
the even-offset output footprint is injective, and even-offset cells never
collide with odd-offset (`+ 1`) cells. Both are required for truth — without
them the two interleaved scatters alias and clobber each other, and no closed
form holds. -/
theorem rbe_triton_surface_implements
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      start_token_position : Nat)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (hOutInj : ∀ s : BlockState, Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
        outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx))
    (hRI : ∀ (s : BlockState) (idx k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]),
      outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx
        ≠ outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
            BLOCK_SIZE_K k + 1) :
    rbeTritonIO x_ptr out_ptr M K stride_x_batch stride_x_m stride_x_n
        stride_out_batch stride_out_m stride_out_n start_token_position THETA
        DIM BLOCK_SIZE_M BLOCK_SIZE_K
      ⊨ fun _pid₀ pid₁ xs o j =>
          let xr := xs (⟨0, by decide⟩ : Fin 2) j
          let xi := xs (⟨1, by decide⟩ : Fin 2) j
          let f := rbeFreqP pid₁ K start_token_position DIM BLOCK_SIZE_M
            BLOCK_SIZE_K THETA j
          match o with
          | ⟨0, _⟩ =>
              xr * Real.cos f -
                (if 1 + rbeColP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j < K then xi
                  else 0) * Real.sin f
          | ⟨_ + 1, _⟩ => xr * Real.sin f + xi * Real.cos f := by
  refine GroupedMasked2DKernelIO.Implements.intro _ ?_ ?_ ?_ ?_
  · -- both output channels live in the declared allocation list
    intro o
    simp [rbeTritonIO]
  · exact rbe_triton_surface_flattenOk x_ptr out_ptr M K stride_x_batch
      stride_x_m stride_x_n stride_out_batch stride_out_m stride_out_n
      start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K
  · intro bounds s hib hob
    simp only [rbeTritonIO] at hib hob
    refine rbe_triton_surface_traceSafe x_ptr out_ptr M K stride_x_batch
      stride_x_m stride_x_n stride_out_batch stride_out_m stride_out_n
      start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K bounds s
      ?_ ?_ ?_ ?_
    · intro idx hact
      simp only [activeReal] at hact
      have h := hib (⟨0, by decide⟩ : Fin 2)
        (rbeLaneInv BLOCK_SIZE_M BLOCK_SIZE_K idx)
        (by simp only [rbeRowP_eq, rbeColP_eq, rbeLane_rbeLaneInv]; exact hact)
      simp only [rbeXAddrP_eq, rbeOutAddrP_eq, rbeLane_rbeLaneInv] at h
      exact h
    · intro idx hact
      simp only [activeImag] at hact
      have h := hib (⟨1, by decide⟩ : Fin 2)
        (rbeLaneInv BLOCK_SIZE_M BLOCK_SIZE_K idx)
        (by simp only [rbeRowP_eq, rbeColP_eq, rbeLane_rbeLaneInv]; exact hact)
      simp only [rbeXAddrP_eq, rbeOutAddrP_eq, rbeLane_rbeLaneInv] at h
      exact h
    · intro idx hact
      simp only [activeReal] at hact
      have h := hob (⟨0, by decide⟩ : Fin 2)
        (rbeLaneInv BLOCK_SIZE_M BLOCK_SIZE_K idx)
        (by simp only [rbeRowP_eq, rbeColP_eq, rbeLane_rbeLaneInv]; exact hact)
      simp only [rbeXAddrP_eq, rbeOutAddrP_eq, rbeLane_rbeLaneInv] at h
      exact h
    · intro idx hact
      simp only [activeImag] at hact
      have h := hob (⟨1, by decide⟩ : Fin 2)
        (rbeLaneInv BLOCK_SIZE_M BLOCK_SIZE_K idx)
        (by simp only [rbeRowP_eq, rbeColP_eq, rbeLane_rbeLaneInv]; exact hact)
      simp only [rbeXAddrP_eq, rbeOutAddrP_eq, rbeLane_rbeLaneInv] at h
      exact h
  · intro s₀ xs hx
    simp only [rbeTritonIO] at hx
    obtain ⟨s1, hs1⟩ := rbe_triton_surface_exec_isSome x_ptr out_ptr M K
      stride_x_batch stride_x_m stride_x_n stride_out_batch stride_out_m
      stride_out_n start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K s₀
    -- restate the two pinned-input hypotheses in the `xOff` spelling the
    -- exec lemmas use (a definitional re-typing, not a rewrite)
    have hxr : ∀ j : Fin (BLOCK_SIZE_M * (BLOCK_SIZE_K / 2)),
        activeReal s₀ M K BLOCK_SIZE_M BLOCK_SIZE_K
          (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j) →
        s₀.readMem x_ptr (xOff s₀ K stride_x_batch stride_x_m stride_x_n
            BLOCK_SIZE_M BLOCK_SIZE_K (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j))
          = xs (⟨0, by decide⟩ : Fin 2) j :=
      fun j hj => hx (⟨0, by decide⟩ : Fin 2) j hj
    have hxi : ∀ j : Fin (BLOCK_SIZE_M * (BLOCK_SIZE_K / 2)),
        activeImag s₀ M K BLOCK_SIZE_M BLOCK_SIZE_K
          (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j) →
        s₀.readMem x_ptr (xOff s₀ K stride_x_batch stride_x_m stride_x_n
            BLOCK_SIZE_M BLOCK_SIZE_K (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j) + 1)
          = xs (⟨1, by decide⟩ : Fin 2) j :=
      fun j hj => hx (⟨1, by decide⟩ : Fin 2) j hj
    refine ⟨s1, hs1, ?_, ?_⟩
    · rintro ⟨o, ho⟩ j hj
      match o, ho with
      | 0, _ =>
          have hact : activeReal s₀ M K BLOCK_SIZE_M BLOCK_SIZE_K
            (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j) := hj
          have h := rbe_exec_real x_ptr out_ptr M K stride_x_batch stride_x_m
            stride_x_n stride_out_batch stride_out_m stride_out_n
            start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K s₀ s1
            (hOutInj s₀) (hRI s₀) hs1 (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j)
          rw [if_pos hact] at h
          have hres : s1.readMem out_ptr
                (outOff s₀ K stride_out_batch stride_out_m stride_out_n
                  BLOCK_SIZE_M BLOCK_SIZE_K
                  (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j))
              = xs (⟨0, by decide⟩ : Fin 2) j *
                  Real.cos (freqSpec s₀ K start_token_position DIM BLOCK_SIZE_M
                    BLOCK_SIZE_K THETA (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j)) -
                (if 1 + colIdx s₀ K BLOCK_SIZE_K
                    (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j).2.1 < K then
                  xs (⟨1, by decide⟩ : Fin 2) j else 0) *
                  Real.sin (freqSpec s₀ K start_token_position DIM BLOCK_SIZE_M
                    BLOCK_SIZE_K THETA (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j)) := by
            rw [h]
            simp only [rbeOutRealSpec]
            rw [hxr j hact]
            by_cases himag : 1 + colIdx s₀ K BLOCK_SIZE_K
                (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j).2.1 < K
            · rw [if_pos himag, if_pos himag, hxi j ⟨hact.1, himag⟩]
            · rw [if_neg himag, if_neg himag]
          exact hres
      | _ + 1, _ =>
          have hact : activeImag s₀ M K BLOCK_SIZE_M BLOCK_SIZE_K
            (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j) := hj
          have h := rbe_exec_imag x_ptr out_ptr M K stride_x_batch stride_x_m
            stride_x_n stride_out_batch stride_out_m stride_out_n
            start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K s₀ s1
            (hOutInj s₀) (hRI s₀) hs1 (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j)
          rw [if_pos hact] at h
          have hact' : activeReal s₀ M K BLOCK_SIZE_M BLOCK_SIZE_K
              (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j) := by
            refine ⟨hact.1, ?_⟩
            have := hact.2
            omega
          have hres : s1.readMem out_ptr
                (outOff s₀ K stride_out_batch stride_out_m stride_out_n
                  BLOCK_SIZE_M BLOCK_SIZE_K
                  (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j) + 1)
              = xs (⟨0, by decide⟩ : Fin 2) j *
                  Real.sin (freqSpec s₀ K start_token_position DIM BLOCK_SIZE_M
                    BLOCK_SIZE_K THETA (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j)) +
                xs (⟨1, by decide⟩ : Fin 2) j *
                  Real.cos (freqSpec s₀ K start_token_position DIM BLOCK_SIZE_M
                    BLOCK_SIZE_K THETA (rbeLane BLOCK_SIZE_M BLOCK_SIZE_K j)) := by
            rw [h]
            simp only [rbeOutImagSpec]
            rw [hxr j hact', hxi j hact]
          exact hres
    · intro r o' hcond
      simp only [rbeTritonIO] at hcond
      refine rbe_triton_surface_frame x_ptr out_ptr M K stride_x_batch
        stride_x_m stride_x_n stride_out_batch stride_out_m stride_out_n
        start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K s₀ s1 hs1 r o'
        ?_ ?_
      · intro idx hact hc
        simp only [activeReal] at hact
        have h := hcond (⟨0, by decide⟩ : Fin 2)
          (rbeLaneInv BLOCK_SIZE_M BLOCK_SIZE_K idx)
          (by simp only [rbeRowP_eq, rbeColP_eq, rbeLane_rbeLaneInv]; exact hact)
        simp only [rbeOutAddrP_eq, rbeLane_rbeLaneInv] at h
        exact h.elim (fun hne => hne hc.1.symm) (fun hne => hne hc.2.symm)
      · intro idx hact hc
        simp only [activeImag] at hact
        have h := hcond (⟨1, by decide⟩ : Fin 2)
          (rbeLaneInv BLOCK_SIZE_M BLOCK_SIZE_K idx)
          (by simp only [rbeRowP_eq, rbeColP_eq, rbeLane_rbeLaneInv]; exact hact)
        simp only [rbeOutAddrP_eq, rbeLane_rbeLaneInv] at h
        exact h.elim (fun hne => hne hc.1.symm) (fun hne => hne hc.2.symm)

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

The final conjunct adds the **`⊨` (`GroupedMasked2DKernelIO.Implements`)** face
of the same two stores: the whole kernel as one grouped masked Hoare triple over
**flat** pointer memory (`nIn = 2` interleaved read channels `x_ptrs` /
`x_ptrs + 1`, `nOut = 2` interleaved write channels `out_ptrs` / `out_ptrs + 1`,
`B = BLOCK_SIZE_M · (BLOCK_SIZE_K / 2)` lanes decoded row-major by `rbeLane`) —
termination, both stored values on their mask-active lanes, and a frame
asserting every cell outside the two output windows is untouched.

Honest side conditions: the even-offset output footprint is injective
(`hOutInj`) and even-offset cells never collide with odd-offset (`+ 1`) cells
(`hRI`) — both hold for the wrapper's contiguous row-major layout. The `⊨`
conjunct quantifies over every launch state, so it needs those same two
conditions at **every** program id (`hOutInjAll` / `hRIAll`, the `∀ s` forms);
`outOff` depends on the state only through `s.pids 0` / `s.pids 1`, so this is
exactly "the layout is collision-free for every CTA", and it is required for
truth — without it the two interleaved scatters alias and no closed form
holds. -/
specification rbe_triton_transform_output_summary_general
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
            BLOCK_SIZE_K k + 1)
    (hOutInjAll : ∀ s : BlockState, Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
        outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx))
    (hRIAll : ∀ (s : BlockState)
        (idx k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]),
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
          start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K THETA idx) ∧
    -- (4) the flat-memory `⊨` face of both interleaved stores
    (rbeTritonIO x_ptr out_ptr M K stride_x_batch stride_x_m stride_x_n
        stride_out_batch stride_out_m stride_out_n start_token_position THETA
        DIM BLOCK_SIZE_M BLOCK_SIZE_K
      ⊨ fun _pid₀ pid₁ xs o j =>
          let xr := xs (⟨0, by decide⟩ : Fin 2) j
          let xi := xs (⟨1, by decide⟩ : Fin 2) j
          let f := rbeFreqP pid₁ K start_token_position DIM BLOCK_SIZE_M
            BLOCK_SIZE_K THETA j
          match o with
          | ⟨0, _⟩ =>
              xr * Real.cos f -
                (if 1 + rbeColP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j < K then xi
                  else 0) * Real.sin f
          | ⟨_ + 1, _⟩ => xr * Real.sin f + xi * Real.cos f) :=
  ⟨rbe_triton_surface_toAlgorithm_supported x_ptr out_ptr M K stride_x_batch
      stride_x_m stride_x_n stride_out_batch stride_out_m stride_out_n
      start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K,
    rbe_real_compute_correct x_ptr out_ptr M K stride_x_batch stride_x_m
      stride_x_n stride_out_batch stride_out_m stride_out_n
      start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K s hOutInj hRI,
    rbe_imag_compute_correct x_ptr out_ptr M K stride_x_batch stride_x_m
      stride_x_n stride_out_batch stride_out_m stride_out_n
      start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K s hOutInj hRI,
    rbe_triton_surface_implements x_ptr out_ptr M K stride_x_batch stride_x_m
      stride_x_n stride_out_batch stride_out_m stride_out_n
      start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K hOutInjAll
      hRIAll⟩

end VeriTile.Bench.TritonBenchG.RbeTritonTransform
