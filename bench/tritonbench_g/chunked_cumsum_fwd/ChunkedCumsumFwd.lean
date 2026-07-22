import VeriTile.Triton

/-!
# `chunked_cumsum_fwd` — strict per-kernel correctness

`_chunk_cumsum_fwd_kernel` is the Mamba-style chunked cumulative-sum forward
pass: program `(pid_b, pid_c, pid_h)` loads a `[BLOCK_SIZE_H, BLOCK_SIZE_CHUNK]`
block of `dt`, optionally adds a per-head bias and applies softplus, clamps to
`[dt_min, dt_max]`, stores the prepared `dt` into `dt_out`, then forms
`dA = dt * A` and stores its per-row cumulative sum `tl.cumsum(dA, axis=1)`
into `dA_cumsum`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`grid = (batch, nchunks, cdiv(nheads, BLOCK_SIZE_H))`,
the autotuned `BLOCK_SIZE_H`, the host-computed `BLOCK_SIZE_CHUNK =
next_power_of_2(chunk_size)`, the strides, and how the runtime composes
per-program writes into one buffer) is the *trusted boundary*, not a proof
obligation here. Because the program ids are universally quantified, the
per-program statements cover every program of the grid.

## Proof architecture

The `dA_cumsum` output is verified against a **genuine standalone closed form**
`dAClosed` — a `Finset.sum` prefix `Σ_{k ≤ idx.chunk, active} dt[head,k]·A[head]`
— not a read-back of the kernel's own output. The within-chunk `tl.cumsum`
(`Tile.scan .sum` along axis 1) is shown equal to that prefix sum by
`scan2d_axis1_sum` / `scan2d_axis1_sum_if` and `dAComputedCumsumSpec_eq_dAClosed`,
stated and proven **general over `nheads`, `chunk_size` and the block sizes**
(hence over the number of chunks, since each chunk `pid_c` is summed
independently along its own axis).

```
chunked_cumsum_fwd_summary_general                            ← TOP THEOREM
  ├─ chunked_cumsum_fwd_surface_toAlgorithm_supported          full surface lowers
  └─ chunked_cumsum_fwd_all_outputs_compute_correct_general
       ├─ chunked_cumsum_dt_out_store_slice_compute_correct
       │     └─ chunked_cumsum_dt_out_store_slice_correct
       ├─ chunked_cumsum_dA_cs_store_slice_compute_correct
       │     └─ chunked_cumsum_dA_cs_store_slice_correct
       └─ chunked_cumsum_dA_cs_compute_slice_closed_form  (= dAClosed)
            ├─ chunked_cumsum_dA_cs_compute_slice_compute_correct
            │    └─ chunked_cumsum_dA_cs_compute_slice_correct
            └─ dAComputedCumsumSpec_eq_dAClosed  (cumsum = prefix Σ)
                 └─ scan2d_axis1_sum / scan2d_axis1_sum_if
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` (the
`BLOCK_SIZE_H ∈ {1..64}` config set) is not modeled — the proofs are
dimension-general (universally quantified over all `Nat` dimensions and strides),
covering the four `HAS_DT_BIAS` / `DT_SOFTPLUS` flag combinations. The
`.to(tl.float32)` casts erase to the identity at the algorithm layer
(post-erasure all dtypes unify to `ℝ`). The per-block prepared-`dt` store, the
elementwise `dA = dt * A`, and the per-row cumsum face (`Tile.scan .sum` along
axis 1, matching `tl.cumsum(dA, axis=1)`) are each modeled exactly; there is no
cross-chunk state carry in this kernel — every chunk `pid_c` is independent, so
no recurrence fold is left implicit, and the genuine closed form `dAClosed` is
the within-chunk prefix sum (not a global cross-chunk fold). The `dA_cumsum`
slice is proven correct against `dAClosed`, a standalone `Finset.sum` — never a
read-back of the kernel's own output. The store slices feed a materialized
prepared-`dt` buffer (`DtPrepared`) so the bias/softplus/clamp preprocessing is
presented rather than re-derived. Output offset injectivity is a side condition
(discharged for the test shape).
-/

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd

open VeriTile.Triton
open scoped VeriTile.Triton.Masked3DKernelIO₂ₓ₂

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! **★ Main theorems:** `chunked_cumsum_fwd_correctness`, `chunked_cumsum_fwd_summary_general` -/

section Correct_without_Rounding

/-! ## Within-chunk cumsum identity (`tl.cumsum` axis=1 = prefix `Finset.sum`)

`tl.cumsum(dA, axis=1)` lowers to `Op.scan .sum` along the chunk axis (axis 1 of
the `[BLOCK_SIZE_H, BLOCK_SIZE_CHUNK]` tile) and hence `Tile.scan .sum`. For a
tile whose lanes hold `some (g idx)`, the scanned value at `idx` is
`some (Σ_{k ≤ idx.chunk} g (idx.head, k))` — the running prefix sum *within the
chunk*, at fixed head. There is no cross-chunk carry in this kernel (each
program `pid_c` handles one chunk independently), so the genuine closed form is
this within-chunk prefix sum. -/

/-- `foldl WithBot.realAdd` over a list of coerced reals collapses to the
coercion of the real-valued `foldl (+)`. -/
private theorem foldl_realAdd_coe (l : List ℝ) (init : ℝ) :
    (l.map (fun r => ((r : ℝ) : WithBot ℝ))).foldl WithBot.realAdd
        ((init : ℝ) : WithBot ℝ)
      = (((l.foldl (· + ·) init : ℝ)) : WithBot ℝ) := by
  induction l generalizing init with
  | nil => simp
  | cons a t ih =>
      simp only [List.map_cons, List.foldl_cons, WithBot.realAdd_coe_coe]
      exact ih (init + a)

/-- A `ScanOp.sum` fold over a list of `some`-valued lanes equals `some` of the
underlying real list-sum. -/
private theorem scan_prefix_eq {α : Type*} (g : α → ℝ) (L : List α) :
    (L.map (fun k => (some (g k) : WithBot ℝ))).foldl WithBot.realAdd
        ((0 : ℝ) : WithBot ℝ)
      = some ((L.map g).sum) := by
  rw [show (L.map (fun k => (some (g k) : WithBot ℝ)))
      = List.map (fun r => ((r : ℝ) : WithBot ℝ)) (L.map g) from by
        rw [List.map_map]; rfl]
  rw [foldl_realAdd_coe]; congr 1; rw [List.sum_eq_foldl]

/-- **2D axis-1 cumsum = within-row prefix sum.** The `Tile.scan .sum` along
axis 1 of a `some`-valued `[H, CH]` tile, evaluated at index `idx`, is
`some (Σ_{k ≤ idx.chunk} g (idx.head, k))` — the prefix sum over the chunk axis
at fixed head. -/
theorem scan2d_axis1_sum (H CH : Nat) (g : Fin H → Fin CH → ℝ)
    (idx : TileIndex [H, CH]) :
    (Tile.scan .sum ⟨1, by simp⟩ .forward
      (⟨fun idx => some (g idx.1 idx.2.1)⟩ : Tile .real [H, CH])).data idx
      = some (∑ k ∈ (Finset.univ.filter
          (fun k : Fin CH => k.val ≤ idx.2.1.val)), g idx.1 k) := by
  rw [Tile.scan_data]
  simp only [ScanOp.eval_sum, TileShape.axisCoord, TileShape.replaceAxisCoord,
    TileShape.axisDim]
  refine Eq.trans (scan_prefix_eq (fun k : Fin CH => g idx.1 k) _) ?_
  rw [Finset.sum_map_toList]; rfl

/-- **Masked 2D axis-1 cumsum = guarded within-row prefix sum.** The
`Tile.scan .sum` (axis 1) of a tile whose lanes hold `some (if P idx then h idx
else 0)`, demoted via `unbotD 0`, is the sum of `h (idx.head, k)` over
`{k : k ≤ idx.chunk ∧ P (idx.head, k)}`. -/
theorem scan2d_axis1_sum_if (H CH : Nat) (h : Fin H → Fin CH → ℝ)
    (P : Fin H → Fin CH → Prop) [∀ a, DecidablePred (P a)]
    (idx : TileIndex [H, CH]) :
    WithBot.unbotD 0
      ((Tile.scan .sum ⟨1, by simp⟩ .forward
        (⟨fun idx => some (if P idx.1 idx.2.1 then h idx.1 idx.2.1 else 0)⟩ :
          Tile .real [H, CH])).data idx)
      = ∑ k ∈ (Finset.univ.filter
          (fun k : Fin CH => k.val ≤ idx.2.1.val ∧ P idx.1 k)), h idx.1 k := by
  rw [scan2d_axis1_sum H CH (fun a k => if P a k then h a k else 0) idx]
  show (∑ k ∈ (Finset.univ.filter (fun k : Fin CH => k.val ≤ idx.2.1.val)),
      if P idx.1 k then h idx.1 k else 0) = _
  rw [Finset.sum_filter, Finset.sum_filter]
  apply Finset.sum_congr rfl
  intro k _
  by_cases hk : k.val ≤ idx.2.1.val <;> by_cases hp : P idx.1 k <;> simp [hk, hp]

/-- Faithful transcription of `chunked_cumsum_fwd.py`'s
`_chunk_cumsum_fwd_kernel`.

This preserves the optional `dt_bias` and softplus paths, the clamp via
`tl.minimum(tl.maximum(...))`, the masked `dt_out` store, and the `dA_cumsum`
store computed with `tl.cumsum` along the chunk axis. -/
def chunked_cumsum_fwd_surface
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr : RegionName)
    (batch seqlen nheads chunk_size : Nat)
    (dt_min dt_max : ℝ)
    (stride_dt_batch stride_dt_seqlen stride_dt_head
      stride_A_head
      stride_dt_bias_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize : Nat)
    (DT_SOFTPLUS HAS_DT_BIAS : Bool)
    (BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=0)
  pid_c = tl.program_id(axis=1)
  pid_h = tl.program_id(axis=2)
  dt_ptr += pid_b * $(stride_dt_batch) +
    pid_c * $(chunk_size) * $(stride_dt_seqlen)
  dt_out_ptr += pid_b * $(stride_dt_out_batch) +
    pid_c * $(stride_dt_out_chunk)
  dA_cumsum_ptr += pid_b * $(stride_dA_cs_batch) +
    pid_c * $(stride_dA_cs_chunk)
  offs_h = pid_h * $(BLOCK_SIZE_H) + tl.arange(0, $(BLOCK_SIZE_H))
  offs_c = tl.arange(0, $(BLOCK_SIZE_CHUNK))
  dt_ptrs = dt_ptr + offs_h[:, None] * $(stride_dt_head) +
    offs_c[None, :] * $(stride_dt_seqlen)
  A_ptrs = A_ptr + offs_h * $(stride_A_head)
  dt_out_ptrs = dt_out_ptr + offs_h[:, None] * $(stride_dt_out_head) +
    offs_c[None, :] * $(stride_dt_out_csize)
  dA_cs_ptrs = dA_cumsum_ptr + offs_h[:, None] * $(stride_dA_cs_head) +
    offs_c[None, :] * $(stride_dA_cs_csize)
  chunk_size_limit = min($(chunk_size), $(seqlen) - pid_c * $(chunk_size))
  dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < $(nheads)) &
    (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
  if HAS_DT_BIAS {
    dt_bias = tl.load(dt_bias_ptr + offs_h * $(stride_dt_bias_head),
      mask=offs_h < $(nheads), other=0.0).to(tl.float32)
    dt += dt_bias[:, None]
  }
  if DT_SOFTPLUS {
    dt = tl.where(dt <= 20.0, tl.log(1.0 + tl.exp(dt)), dt)
  }
  dt = tl.minimum(tl.maximum(dt, $(dt_min)), $(dt_max))
  dt = tl.where((offs_h[:, None] < $(nheads)) & (offs_c[None, :] < chunk_size_limit),
    dt, 0.0)
  tl.store(dt_out_ptrs, dt, mask=(offs_h[:, None] < $(nheads)) &
    (offs_c[None, :] < $(chunk_size)))
  A = tl.load(A_ptrs, mask=offs_h < $(nheads), other=0.0).to(tl.float32)
  dA = dt * A[:, None]
  dA_cs = tl.cumsum(dA, axis=1)
  tl.store(dA_cs_ptrs, dA_cs, mask=(offs_h[:, None] < $(nheads)) &
    (offs_c[None, :] < $(chunk_size)))
}

/-- The full Python-shaped chunked-cumsum forward surface lowers to the
algorithm layer, including optional bias/softplus, clamp, `dt_out`, and
`dA_cumsum` stores. -/
theorem chunked_cumsum_fwd_surface_toAlgorithm_supported
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr : RegionName)
    (batch seqlen nheads chunk_size : Nat)
    (dt_min dt_max : ℝ)
    (stride_dt_batch stride_dt_seqlen stride_dt_head
      stride_A_head
      stride_dt_bias_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize : Nat)
    (DT_SOFTPLUS HAS_DT_BIAS : Bool)
    (BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) :
    ∃ alg, (chunked_cumsum_fwd_surface dt_ptr A_ptr dt_bias_ptr dt_out_ptr
      dA_cumsum_ptr batch seqlen nheads chunk_size dt_min dt_max stride_dt_batch
      stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK).toAlgorithm?
        = Except.ok alg := by
  simp [chunked_cumsum_fwd_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented `dt_out` writeback slice of `chunked_cumsum_fwd.py`'s
`_chunk_cumsum_fwd_kernel`.

The full kernel loads `dt`, optionally adds bias/softplus, clamps it, then
stores `dt_out` and computes/stores `dA_cumsum`. This slice starts from a
preprocessed `DtPrepared` tile and proves the masked `dt_out` writeback using
the original head/chunk indexing and output strides. -/
def chunked_cumsum_dt_out_store_slice
    (DtPrepared DtOut : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=0)
  pid_c = tl.program_id(axis=1)
  pid_h = tl.program_id(axis=2)
  offs_h = pid_h * $(BLOCK_SIZE_H) + tl.arange(0, $(BLOCK_SIZE_H))
  offs_c = tl.arange(0, $(BLOCK_SIZE_CHUNK))
  mask = (offs_h[:, None] < $(nheads)) & (offs_c[None, :] < $(chunk_size))
  dt = tl.load(DtPrepared + pid_b * $(stride_dt_batch) +
      (pid_c * $(chunk_size) + offs_c[None, :]) * $(stride_dt_seqlen) +
      offs_h[:, None] * $(stride_dt_head),
    mask=mask, other=0.0)
  tl.store(DtOut + pid_b * $(stride_dt_out_batch) + pid_c * $(stride_dt_out_chunk) +
      offs_h[:, None] * $(stride_dt_out_head) + offs_c[None, :] * $(stride_dt_out_csize),
    dt, mask=mask)
}

def headIndex (s : BlockState) (BLOCK_SIZE_H : Nat) (i : Fin BLOCK_SIZE_H) : Nat :=
  s.pids 2 * BLOCK_SIZE_H + i.val

def chunkIndex (_s : BlockState) (j : Fin BLOCK_SIZE_CHUNK) : Nat :=
  j.val

def active
    (s : BlockState) (nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK]) : Prop :=
  headIndex s BLOCK_SIZE_H idx.1 < nheads ∧ chunkIndex s idx.2.1 < chunk_size

instance activeDecidable
    (s : BlockState) (nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK]) :
    Decidable (active s nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK idx) := by
  unfold active
  infer_instance

def dtPreparedOffset
    (s : BlockState)
    (stride_dt_batch stride_dt_seqlen stride_dt_head chunk_size BLOCK_SIZE_H : Nat)
    (idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK]) : Nat :=
  s.pids 0 * stride_dt_batch +
    (s.pids 1 * chunk_size + chunkIndex s idx.2.1) * stride_dt_seqlen +
    headIndex s BLOCK_SIZE_H idx.1 * stride_dt_head

def dtOutOffset
    (s : BlockState)
    (stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      BLOCK_SIZE_H : Nat)
    (idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK]) : Nat :=
  s.pids 0 * stride_dt_out_batch + s.pids 1 * stride_dt_out_chunk +
    headIndex s BLOCK_SIZE_H idx.1 * stride_dt_out_head +
    chunkIndex s idx.2.1 * stride_dt_out_csize

/-- Algorithm-layer correctness for the `dt_out` store slice. -/
theorem chunked_cumsum_dt_out_store_slice_correct
    (DtPrepared DtOut : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        dtOutOffset s stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head
          stride_dt_out_csize BLOCK_SIZE_H idx)) :
    ∀ idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK],
      let outAddr := dtOutOffset s stride_dt_out_batch stride_dt_out_chunk
        stride_dt_out_head stride_dt_out_csize BLOCK_SIZE_H idx
      (exec (chunked_cumsum_dt_out_store_slice DtPrepared DtOut
            stride_dt_batch stride_dt_seqlen stride_dt_head stride_dt_out_batch
            stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize nheads
            chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK) s).map (·.readMem DtOut outAddr)
        = some (if active s nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK idx then
            s.readMem DtPrepared
              (dtPreparedOffset s stride_dt_batch stride_dt_seqlen stride_dt_head
                chunk_size BLOCK_SIZE_H idx)
          else s.readMem DtOut outAddr) := by
  intro idx
  simp [exec, chunked_cumsum_dt_out_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        headIndex, chunkIndex, dtPreparedOffset, dtOutOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] → Nat :=
    fun idx =>
      s.pids 0 * stride_dt_out_batch + s.pids 1 * stride_dt_out_chunk +
        (s.pids 2 * BLOCK_SIZE_H + idx.1.val) * stride_dt_out_head +
        idx.2.1.val * stride_dt_out_csize
  let valueFn : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 2 * BLOCK_SIZE_H + idx.1.val < nheads ∧
            idx.2.1.val < chunk_size then
          some (s.readMem DtPrepared
            (s.pids 0 * stride_dt_batch +
              (s.pids 1 * chunk_size + idx.2.1.val) * stride_dt_seqlen +
              (s.pids 2 * BLOCK_SIZE_H + idx.1.val) * stride_dt_head))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] → Prop :=
    fun idx =>
      s.pids 2 * BLOCK_SIZE_H + idx.1.val < nheads ∧
        idx.2.1.val < chunk_size
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, dtOutOffset, headIndex, chunkIndex] using hOutInj
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 2 * BLOCK_SIZE_H + idx.1.val < nheads ∧
        idx.2.1.val < chunk_size
  · simp [offsetFn, valueFn, P, active, headIndex, chunkIndex,
      dtPreparedOffset, dtOutOffset, hActive]
  · simp [offsetFn, valueFn, P, active, headIndex, chunkIndex,
      dtPreparedOffset, dtOutOffset, hActive]

/-- Compute-facing correctness for the `dt_out` store slice. -/
theorem chunked_cumsum_dt_out_store_slice_compute_correct
    (DtPrepared DtOut : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        dtOutOffset s stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head
          stride_dt_out_csize BLOCK_SIZE_H idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunked_cumsum_dt_out_store_slice DtPrepared DtOut
        stride_dt_batch stride_dt_seqlen stride_dt_head stride_dt_out_batch
        stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize nheads
        chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
        (fun idx => (DtOut,
          dtOutOffset s stride_dt_out_batch stride_dt_out_chunk
            stride_dt_out_head stride_dt_out_csize BLOCK_SIZE_H idx)))
      (expected := fun idx =>
        s.readMem DtPrepared
          (dtPreparedOffset s stride_dt_batch stride_dt_seqlen stride_dt_head
            chunk_size BLOCK_SIZE_H idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunked_cumsum_dt_out_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunked_cumsum_dt_out_store_slice_correct DtPrepared DtOut
    stride_dt_batch stride_dt_seqlen stride_dt_head stride_dt_out_batch
    stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize nheads chunk_size
    BLOCK_SIZE_H BLOCK_SIZE_CHUNK s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Proof-oriented `dA_cumsum` writeback slice of `chunked_cumsum_fwd.py`'s
`_chunk_cumsum_fwd_kernel`.

The full kernel produces a per-head running cumsum tile `dA_cs = cumsum(dt * A,
axis=1)`. This slice starts from a precomputed `DAcs` tile and proves the
masked writeback into `dA_cumsum_ptr` using the original head/chunk indexing
and output strides. -/
def chunked_cumsum_dA_cs_store_slice
    (DAcs DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=0)
  pid_c = tl.program_id(axis=1)
  pid_h = tl.program_id(axis=2)
  offs_h = pid_h * $(BLOCK_SIZE_H) + tl.arange(0, $(BLOCK_SIZE_H))
  offs_c = tl.arange(0, $(BLOCK_SIZE_CHUNK))
  mask = (offs_h[:, None] < $(nheads)) & (offs_c[None, :] < $(chunk_size))
  dA_cs = tl.load(DAcs + pid_b * $(stride_dt_batch) +
      (pid_c * $(chunk_size) + offs_c[None, :]) * $(stride_dt_seqlen) +
      offs_h[:, None] * $(stride_dt_head),
    mask=mask, other=0.0)
  tl.store(DACumsum + pid_b * $(stride_dA_cs_batch) + pid_c * $(stride_dA_cs_chunk) +
      offs_h[:, None] * $(stride_dA_cs_head) + offs_c[None, :] * $(stride_dA_cs_csize),
    dA_cs, mask=mask)
}

def dACsOutOffset
    (s : BlockState)
    (stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      BLOCK_SIZE_H : Nat)
    (idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK]) : Nat :=
  s.pids 0 * stride_dA_cs_batch + s.pids 1 * stride_dA_cs_chunk +
    headIndex s BLOCK_SIZE_H idx.1 * stride_dA_cs_head +
    chunkIndex s idx.2.1 * stride_dA_cs_csize

/-- Algorithm-layer correctness for the `dA_cumsum` store slice. -/
theorem chunked_cumsum_dA_cs_store_slice_correct
    (DAcs DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        dACsOutOffset s stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head
          stride_dA_cs_csize BLOCK_SIZE_H idx)) :
    ∀ idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK],
      let outAddr := dACsOutOffset s stride_dA_cs_batch stride_dA_cs_chunk
        stride_dA_cs_head stride_dA_cs_csize BLOCK_SIZE_H idx
      (exec (chunked_cumsum_dA_cs_store_slice DAcs DACumsum
            stride_dt_batch stride_dt_seqlen stride_dt_head stride_dA_cs_batch
            stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize nheads
            chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK) s).map (·.readMem DACumsum outAddr)
        = some (if active s nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK idx then
            s.readMem DAcs
              (dtPreparedOffset s stride_dt_batch stride_dt_seqlen stride_dt_head
                chunk_size BLOCK_SIZE_H idx)
          else s.readMem DACumsum outAddr) := by
  intro idx
  simp [exec, chunked_cumsum_dA_cs_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        headIndex, chunkIndex, dtPreparedOffset, dACsOutOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] → Nat :=
    fun idx =>
      s.pids 0 * stride_dA_cs_batch + s.pids 1 * stride_dA_cs_chunk +
        (s.pids 2 * BLOCK_SIZE_H + idx.1.val) * stride_dA_cs_head +
        idx.2.1.val * stride_dA_cs_csize
  let valueFn : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 2 * BLOCK_SIZE_H + idx.1.val < nheads ∧
            idx.2.1.val < chunk_size then
          some (s.readMem DAcs
            (s.pids 0 * stride_dt_batch +
              (s.pids 1 * chunk_size + idx.2.1.val) * stride_dt_seqlen +
              (s.pids 2 * BLOCK_SIZE_H + idx.1.val) * stride_dt_head))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] → Prop :=
    fun idx =>
      s.pids 2 * BLOCK_SIZE_H + idx.1.val < nheads ∧
        idx.2.1.val < chunk_size
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, dACsOutOffset, headIndex, chunkIndex] using hOutInj
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 2 * BLOCK_SIZE_H + idx.1.val < nheads ∧
        idx.2.1.val < chunk_size
  · simp [offsetFn, valueFn, P, active, headIndex, chunkIndex,
      dtPreparedOffset, dACsOutOffset, hActive]
  · simp [offsetFn, valueFn, P, active, headIndex, chunkIndex,
      dtPreparedOffset, dACsOutOffset, hActive]

/-- Compute-facing correctness for the `dA_cumsum` store slice. -/
theorem chunked_cumsum_dA_cs_store_slice_compute_correct
    (DAcs DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        dACsOutOffset s stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head
          stride_dA_cs_csize BLOCK_SIZE_H idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunked_cumsum_dA_cs_store_slice DAcs DACumsum
        stride_dt_batch stride_dt_seqlen stride_dt_head stride_dA_cs_batch
        stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize nheads
        chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
        (fun idx => (DACumsum,
          dACsOutOffset s stride_dA_cs_batch stride_dA_cs_chunk
            stride_dA_cs_head stride_dA_cs_csize BLOCK_SIZE_H idx)))
      (expected := fun idx =>
        s.readMem DAcs
          (dtPreparedOffset s stride_dt_batch stride_dt_seqlen stride_dt_head
            chunk_size BLOCK_SIZE_H idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunked_cumsum_dA_cs_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunked_cumsum_dA_cs_store_slice_correct DAcs DACumsum
    stride_dt_batch stride_dt_seqlen stride_dt_head stride_dA_cs_batch
    stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize nheads chunk_size
    BLOCK_SIZE_H BLOCK_SIZE_CHUNK s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Computed `dA_cumsum` slice

The store slice above assumes the `dA_cs` tile has already been prepared. The
next slice covers the Python computation immediately before that store:
`A = load(A_ptrs)`, `dA = dt * A[:, None]`, and
`dA_cs = tl.cumsum(dA, axis=1)`. -/

def chunked_cumsum_dA_cs_compute_slice
    (DtPrepared A DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=0)
  pid_c = tl.program_id(axis=1)
  pid_h = tl.program_id(axis=2)
  offs_h = pid_h * $(BLOCK_SIZE_H) + tl.arange(0, $(BLOCK_SIZE_H))
  offs_c = tl.arange(0, $(BLOCK_SIZE_CHUNK))
  mask = (offs_h[:, None] < $(nheads)) & (offs_c[None, :] < $(chunk_size))
  dt = tl.load(DtPrepared + pid_b * $(stride_dt_batch) +
      (pid_c * $(chunk_size) + offs_c[None, :]) * $(stride_dt_seqlen) +
      offs_h[:, None] * $(stride_dt_head),
    mask=mask, other=0.0)
  a = tl.load(A + offs_h * $(stride_A_head),
    mask=offs_h < $(nheads), other=0.0)
  dA = dt * a[:, None]
  dA_cs = tl.cumsum(dA, axis=1)
  tl.store(DACumsum + pid_b * $(stride_dA_cs_batch) + pid_c * $(stride_dA_cs_chunk) +
      offs_h[:, None] * $(stride_dA_cs_head) + offs_c[None, :] * $(stride_dA_cs_csize),
    dA_cs, mask=mask)
}

noncomputable def dATile
    (s : BlockState) (DtPrepared A : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) :
    Tile .real [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] :=
  { data := fun idx =>
      Option.map₂ (fun dt a : ℝ => dt * a)
        (if active s nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK idx then
          some (s.readMem DtPrepared
              (dtPreparedOffset s stride_dt_batch stride_dt_seqlen
                stride_dt_head chunk_size BLOCK_SIZE_H idx))
        else some (0.0 : ℝ))
        (if headIndex s BLOCK_SIZE_H idx.1 < nheads then
          some (s.readMem A (headIndex s BLOCK_SIZE_H idx.1 * stride_A_head))
        else some (0.0 : ℝ)) }

noncomputable def dAComputedCumsumSpec
    (s : BlockState) (DtPrepared A : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK]) : ℝ :=
  WithBot.unbotD 0
    ((Tile.scan .sum ⟨1, by simp⟩ .forward
      (dATile s DtPrepared A stride_dt_batch stride_dt_seqlen
        stride_dt_head stride_A_head nheads chunk_size BLOCK_SIZE_H
        BLOCK_SIZE_CHUNK)).data idx)

/-! ## Genuine within-chunk closed form for `dA_cumsum`

`dAClosed` is the genuine mathematical specification of the `dA_cumsum` output:
a `Finset.sum` over all *active* chunk positions `k ≤ idx.chunk` (with the head
in range and `k < chunk_size`) of the prepared `dt` value at `(head, k)` times
the per-head scaling `A[head]`. It is **not** a read-back of the kernel's own
output, so realizing it is a true correctness statement. There is no cross-chunk
carry: each chunk is summed independently along its own axis. -/

/-- Prepared-`dt` element `dt[batch, seqpos, head]`: batch `pids 0` (stride
`stride_dt_batch`), sequence position `pid_chunk·chunk_size + k` (stride
`stride_dt_seqlen`), head `h` (stride `stride_dt_head`). -/
noncomputable def dtElem
    (s : BlockState) (DtPrepared : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head chunk_size : Nat)
    (k h : Nat) : ℝ :=
  s.readMem DtPrepared
    (s.pids 0 * stride_dt_batch + (s.pids 1 * chunk_size + k) * stride_dt_seqlen +
      h * stride_dt_head)

noncomputable def dAClosed
    (s : BlockState) (DtPrepared A : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK]) : ℝ :=
  ∑ k ∈ (Finset.univ.filter
      (fun k : Fin BLOCK_SIZE_CHUNK =>
        k.val ≤ idx.2.1.val ∧
          (headIndex s BLOCK_SIZE_H idx.1 < nheads ∧ k.val < chunk_size))),
    dtElem s DtPrepared stride_dt_batch stride_dt_seqlen stride_dt_head
        chunk_size k.val (headIndex s BLOCK_SIZE_H idx.1)
      * s.readMem A (headIndex s BLOCK_SIZE_H idx.1 * stride_A_head)

/-- **The within-chunk cumsum equals the genuine closed form.** The kernel's
`tl.cumsum(dt * A, axis=1)` (modeled by `dAComputedCumsumSpec` via `Tile.scan`)
equals the standalone `Finset.sum` prefix `dAClosed`. -/
theorem dAComputedCumsumSpec_eq_dAClosed
    (s : BlockState) (DtPrepared A : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK]) :
    dAComputedCumsumSpec s DtPrepared A stride_dt_batch stride_dt_seqlen
        stride_dt_head stride_A_head nheads chunk_size BLOCK_SIZE_H
        BLOCK_SIZE_CHUNK idx
      = dAClosed s DtPrepared A stride_dt_batch stride_dt_seqlen
        stride_dt_head stride_A_head nheads chunk_size BLOCK_SIZE_H
        BLOCK_SIZE_CHUNK idx := by
  unfold dAComputedCumsumSpec dAClosed dtElem
  -- Rewrite the input tile to the `some (if active then dt*A else 0)` form.
  have hin :
      dATile s DtPrepared A stride_dt_batch stride_dt_seqlen stride_dt_head
          stride_A_head nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK
        = (⟨fun idx => some
            (if ((s.pids 2 * BLOCK_SIZE_H + idx.1.val) < nheads ∧
                  idx.2.1.val < chunk_size) then
              s.readMem DtPrepared
                  (s.pids 0 * stride_dt_batch +
                    (s.pids 1 * chunk_size + idx.2.1.val) * stride_dt_seqlen +
                    (s.pids 2 * BLOCK_SIZE_H + idx.1.val) * stride_dt_head)
                * s.readMem A ((s.pids 2 * BLOCK_SIZE_H + idx.1.val) * stride_A_head)
            else 0)⟩ : Tile .real [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK]) := by
    unfold dATile; congr 1; funext idx
    simp only [active, headIndex, chunkIndex, dtPreparedOffset]
    have hz : (0.0 : ℝ) = 0 := by norm_num
    by_cases hh : s.pids 2 * BLOCK_SIZE_H + idx.1.val < nheads
    · by_cases hc : idx.2.1.val < chunk_size
      · simp [hh, hc]
      · simp only [hh, hc, and_false, if_false, if_pos hh, if_true,
          Option.map₂_some_some, hz, zero_mul]
    · simp only [hh, false_and, if_false, if_neg hh,
        Option.map₂_some_some, hz, zero_mul, mul_zero]
  rw [hin]
  rw [scan2d_axis1_sum_if BLOCK_SIZE_H BLOCK_SIZE_CHUNK
    (fun a k => s.readMem DtPrepared
        (s.pids 0 * stride_dt_batch +
          (s.pids 1 * chunk_size + k.val) * stride_dt_seqlen +
          (s.pids 2 * BLOCK_SIZE_H + a.val) * stride_dt_head)
      * s.readMem A ((s.pids 2 * BLOCK_SIZE_H + a.val) * stride_A_head))
    (fun a k => (s.pids 2 * BLOCK_SIZE_H + a.val) < nheads ∧ k.val < chunk_size)
    idx]
  apply Finset.sum_congr ?_ (fun _ _ => rfl)
  apply Finset.filter_congr
  intro k _
  simp only [headIndex, chunkIndex]

theorem chunked_cumsum_dA_cs_compute_slice_correct
    (DtPrepared A DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        dACsOutOffset s stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head
          stride_dA_cs_csize BLOCK_SIZE_H idx)) :
    ∀ idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK],
      let outAddr := dACsOutOffset s stride_dA_cs_batch stride_dA_cs_chunk
        stride_dA_cs_head stride_dA_cs_csize BLOCK_SIZE_H idx
      (exec (chunked_cumsum_dA_cs_compute_slice DtPrepared A DACumsum
            stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
            stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head
            stride_dA_cs_csize nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
          s).map (·.readMem DACumsum outAddr)
        = some (if active s nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK idx then
            dAComputedCumsumSpec s DtPrepared A stride_dt_batch stride_dt_seqlen
              stride_dt_head stride_A_head nheads chunk_size BLOCK_SIZE_H
              BLOCK_SIZE_CHUNK idx
          else s.readMem DACumsum outAddr) := by
  intro idx
  simp [exec, chunked_cumsum_dA_cs_compute_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, Tile.scan, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, headIndex, chunkIndex, active, dtPreparedOffset,
        dACsOutOffset, dATile, TileShape.dropInsertedIndex,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  let offsetFn : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] → Nat :=
    fun idx =>
      s.pids 0 * stride_dA_cs_batch + s.pids 1 * stride_dA_cs_chunk +
        (s.pids 2 * BLOCK_SIZE_H + idx.1.val) * stride_dA_cs_head +
        idx.2.1.val * stride_dA_cs_csize
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, dACsOutOffset, headIndex, chunkIndex] using hOutInj
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 2 * BLOCK_SIZE_H + idx.1.val < nheads ∧
        idx.2.1.val < chunk_size
  · simp [offsetFn, active, headIndex, chunkIndex, dAComputedCumsumSpec,
      dATile, dtPreparedOffset, dACsOutOffset, hActive]
    rfl
  · simp [offsetFn, active, headIndex, chunkIndex, dACsOutOffset, hActive]

theorem chunked_cumsum_dA_cs_compute_slice_compute_correct
    (DtPrepared A DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        dACsOutOffset s stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head
          stride_dA_cs_csize BLOCK_SIZE_H idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunked_cumsum_dA_cs_compute_slice DtPrepared A DACumsum
        stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
        stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head
        stride_dA_cs_csize nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
        (fun idx => (DACumsum,
          dACsOutOffset s stride_dA_cs_batch stride_dA_cs_chunk
            stride_dA_cs_head stride_dA_cs_csize BLOCK_SIZE_H idx)))
      (expected := fun idx =>
        dAComputedCumsumSpec s DtPrepared A stride_dt_batch stride_dt_seqlen
          stride_dt_head stride_A_head nheads chunk_size BLOCK_SIZE_H
          BLOCK_SIZE_CHUNK idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunked_cumsum_dA_cs_compute_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunked_cumsum_dA_cs_compute_slice_correct DtPrepared A DACumsum
    stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
    stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
    nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- **Genuine within-chunk closed-form correctness for `dA_cumsum`.** The
`dA = dt * A; dA_cs = tl.cumsum(dA, axis=1)` slice realizes the standalone
`Finset.sum` prefix `dAClosed` — `Σ_{k ≤ idx.chunk, active} dt[head,k] · A[head]`
— not a read-back of the kernel's own output. General over `nheads`,
`chunk_size`, and the block sizes (and hence over the number of chunks, since
each chunk is summed independently along its own axis). -/
theorem chunked_cumsum_dA_cs_compute_slice_closed_form
    (DtPrepared A DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        dACsOutOffset s stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head
          stride_dA_cs_csize BLOCK_SIZE_H idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunked_cumsum_dA_cs_compute_slice DtPrepared A DACumsum
        stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
        stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head
        stride_dA_cs_csize nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
        (fun idx => (DACumsum,
          dACsOutOffset s stride_dA_cs_batch stride_dA_cs_chunk
            stride_dA_cs_head stride_dA_cs_csize BLOCK_SIZE_H idx)))
      (expected := fun idx =>
        dAClosed s DtPrepared A stride_dt_batch stride_dt_seqlen
          stride_dt_head stride_A_head nheads chunk_size BLOCK_SIZE_H
          BLOCK_SIZE_CHUNK idx) := by
  have h := chunked_cumsum_dA_cs_compute_slice_compute_correct DtPrepared A
    DACumsum stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
    stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
    nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK s hOutInj
  have hcong :
      (fun idx => dAComputedCumsumSpec s DtPrepared A stride_dt_batch
          stride_dt_seqlen stride_dt_head stride_A_head nheads chunk_size
          BLOCK_SIZE_H BLOCK_SIZE_CHUNK idx)
        = (fun idx => dAClosed s DtPrepared A stride_dt_batch stride_dt_seqlen
            stride_dt_head stride_A_head nheads chunk_size BLOCK_SIZE_H
            BLOCK_SIZE_CHUNK idx) := by
    funext idx
    exact dAComputedCumsumSpec_eq_dAClosed s DtPrepared A stride_dt_batch
      stride_dt_seqlen stride_dt_head stride_A_head nheads chunk_size
      BLOCK_SIZE_H BLOCK_SIZE_CHUNK idx
  rwa [hcong] at h

/-! ## Fused two-output kernel and the `⊨` conversion (three-pid skin)

The three slice theorems above verify the outputs one kernel at a time against a
materialized `DtPrepared` boundary. The `⊨` headline packages the two genuine
outputs (`dt_out` and `dA_cumsum`) of a **single fused** proof-oriented kernel
that loads the prepared `dt` tile and the per-head `A`, stores `dt_out`, forms
`dA = dt · A`, and stores the within-chunk `tl.cumsum` into `dA_cumsum`. Its
three program ids `(pid_b, pid_c, pid_h)` are read as `tl.program_id(0/1/2)` —
the first consumer of the new `Masked3DKernelIO₂ₓ₂` three-program-id skin. The
`dA_cumsum` output is still specified by the standalone within-chunk prefix
`Finset.sum` (`dAClosed`), never a read-back of the kernel's own output. The
bias/softplus preprocessing stays above this verified boundary (materialized in
`DtPrepared`), exactly as in the slice decomposition; the full Python surface
(with `HAS_DT_BIAS` / `DT_SOFTPLUS`) is still faithfully lowered by
`chunked_cumsum_fwd_surface_toAlgorithm_supported`. -/

/-- The fused proof-oriented slice: loads prepared `dt` and per-head `A`, stores
`dt_out`, then stores `dA_cumsum = cumsum(dt · A, axis=1)`. The two loads
precede both stores, so each load reads the launch memory directly. -/
def chunked_cumsum_fused_slice
    (DtPrepared A DtOut DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=0)
  pid_c = tl.program_id(axis=1)
  pid_h = tl.program_id(axis=2)
  offs_h = pid_h * $(BLOCK_SIZE_H) + tl.arange(0, $(BLOCK_SIZE_H))
  offs_c = tl.arange(0, $(BLOCK_SIZE_CHUNK))
  mask = (offs_h[:, None] < $(nheads)) & (offs_c[None, :] < $(chunk_size))
  dt = tl.load(DtPrepared + pid_b * $(stride_dt_batch) +
      (pid_c * $(chunk_size) + offs_c[None, :]) * $(stride_dt_seqlen) +
      offs_h[:, None] * $(stride_dt_head),
    mask=mask, other=0.0)
  a = tl.load(A + offs_h * $(stride_A_head),
    mask=offs_h < $(nheads), other=0.0)
  tl.store(DtOut + pid_b * $(stride_dt_out_batch) + pid_c * $(stride_dt_out_chunk) +
      offs_h[:, None] * $(stride_dt_out_head) + offs_c[None, :] * $(stride_dt_out_csize),
    dt, mask=mask)
  dA = dt * a[:, None]
  dA_cs = tl.cumsum(dA, axis=1)
  tl.store(DACumsum + pid_b * $(stride_dA_cs_batch) + pid_c * $(stride_dA_cs_chunk) +
      offs_h[:, None] * $(stride_dA_cs_head) + offs_c[None, :] * $(stride_dA_cs_csize),
    dA_cs, mask=mask)
}

/-- Termination: the fused kernel executes to completion from any state. -/
private theorem chunked_cumsum_fused_exec_isSome
    (DtPrepared A DtOut DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) (s : BlockState) :
    ∃ s1, exec ((chunked_cumsum_fused_slice DtPrepared A DtOut DACumsum
      stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK).toAlgKernel) s = some s1 := by
  simp [exec, chunked_cumsum_fused_slice, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.scan,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        TileShape.dropInsertedIndex, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]

/-- The fused kernel sits inside the flat-memory bridge's covered fragment. -/
theorem chunked_cumsum_fused_flattenOk
    (DtPrepared A DtOut DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) :
    ((chunked_cumsum_fused_slice DtPrepared A DtOut DACumsum
      stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [chunked_cumsum_fused_slice, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- Per-execution safety walk: the two masked 2-D block accesses (`DtPrepared`
load, `DtOut`/`DACumsum` stores) reduce to lane-wise bounds at the active
`(head, chunk)` lanes; the masked 1-D `A` load reduces to per-head bounds. -/
theorem chunked_cumsum_fused_traceSafe
    (DtPrepared A DtOut DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hin_dt : ∀ (a : Fin BLOCK_SIZE_H) (b : Fin BLOCK_SIZE_CHUNK),
      s.pids 2 * BLOCK_SIZE_H + a.val < nheads → b.val < chunk_size →
      s.pids 0 * stride_dt_batch + (s.pids 1 * chunk_size + b.val) * stride_dt_seqlen +
          (s.pids 2 * BLOCK_SIZE_H + a.val) * stride_dt_head
        < bounds DtPrepared)
    (hin_A : ∀ (a : Fin BLOCK_SIZE_H),
      s.pids 2 * BLOCK_SIZE_H + a.val < nheads →
      (s.pids 2 * BLOCK_SIZE_H + a.val) * stride_A_head < bounds A)
    (hout_dt : ∀ (a : Fin BLOCK_SIZE_H) (b : Fin BLOCK_SIZE_CHUNK),
      s.pids 2 * BLOCK_SIZE_H + a.val < nheads → b.val < chunk_size →
      s.pids 0 * stride_dt_out_batch + s.pids 1 * stride_dt_out_chunk +
          (s.pids 2 * BLOCK_SIZE_H + a.val) * stride_dt_out_head +
          b.val * stride_dt_out_csize
        < bounds DtOut)
    (hout_da : ∀ (a : Fin BLOCK_SIZE_H) (b : Fin BLOCK_SIZE_CHUNK),
      s.pids 2 * BLOCK_SIZE_H + a.val < nheads → b.val < chunk_size →
      s.pids 0 * stride_dA_cs_batch + s.pids 1 * stride_dA_cs_chunk +
          (s.pids 2 * BLOCK_SIZE_H + a.val) * stride_dA_cs_head +
          b.val * stride_dA_cs_csize
        < bounds DACumsum) :
    Kernel.TraceSafe bounds
      ((chunked_cumsum_fused_slice DtPrepared A DtOut DACumsum
        stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
        stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
        stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
        nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [chunked_cumsum_fused_slice, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.scan,
    NumericDType.add, NumericDType.mul, ComparableDType.lt,
    TileShape.dropInsertedIndex]
  exact ⟨fun a b ha hb => hin_dt a b ha hb, fun a ha => hin_A a ha,
    fun a b ha hb => hout_dt a b ha hb, fun a b ha hb => hout_da a b ha hb⟩

/-- Executed-state readback for the fused kernel's `dA_cumsum` store (the last
store): at an active lane the within-chunk cumsum value. -/
theorem chunked_cumsum_fused_dA_cs_correct
    (DtPrepared A DtOut DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (s : BlockState)
    (hDACsInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        dACsOutOffset s stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head
          stride_dA_cs_csize BLOCK_SIZE_H idx)) :
    ∀ idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK],
      active s nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK idx →
      let outAddr := dACsOutOffset s stride_dA_cs_batch stride_dA_cs_chunk
        stride_dA_cs_head stride_dA_cs_csize BLOCK_SIZE_H idx
      (exec ((chunked_cumsum_fused_slice DtPrepared A DtOut DACumsum
            stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
            stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head
            stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk
            stride_dA_cs_head stride_dA_cs_csize nheads chunk_size BLOCK_SIZE_H
            BLOCK_SIZE_CHUNK).toAlgKernel) s).map (·.readMem DACumsum outAddr)
        = some (dAComputedCumsumSpec s DtPrepared A stride_dt_batch stride_dt_seqlen
            stride_dt_head stride_A_head nheads chunk_size BLOCK_SIZE_H
            BLOCK_SIZE_CHUNK idx) := by
  intro idx hActive
  simp [exec, chunked_cumsum_fused_slice, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd, Tile.scan,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        headIndex, chunkIndex, active, dtPreparedOffset, dACsOutOffset, dATile,
        TileShape.dropInsertedIndex, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]
  let offsetFn : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] → Nat :=
    fun idx =>
      s.pids 0 * stride_dA_cs_batch + s.pids 1 * stride_dA_cs_chunk +
        (s.pids 2 * BLOCK_SIZE_H + idx.1.val) * stride_dA_cs_head +
        idx.2.1.val * stride_dA_cs_csize
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, dACsOutOffset, headIndex, chunkIndex] using hDACsInj
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  obtain ⟨hh, hc⟩ := hActive
  simp only [headIndex, chunkIndex] at hh hc
  simp [offsetFn, active, headIndex, chunkIndex, dAComputedCumsumSpec,
    dATile, dtPreparedOffset, dACsOutOffset, hh, hc]
  rfl

/-- Executed-state readback for the fused kernel's `dt_out` store (the first
store): at an active lane the prepared `dt` value. The trailing `dA_cumsum`
scatter (to a distinct region) is stripped first. -/
theorem chunked_cumsum_fused_dt_out_correct
    (DtPrepared A DtOut DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (s : BlockState) (hDtNe : DtOut ≠ DACumsum)
    (hDtOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        dtOutOffset s stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head
          stride_dt_out_csize BLOCK_SIZE_H idx)) :
    ∀ idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK],
      active s nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK idx →
      let outAddr := dtOutOffset s stride_dt_out_batch stride_dt_out_chunk
        stride_dt_out_head stride_dt_out_csize BLOCK_SIZE_H idx
      (exec ((chunked_cumsum_fused_slice DtPrepared A DtOut DACumsum
            stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
            stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head
            stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk
            stride_dA_cs_head stride_dA_cs_csize nheads chunk_size BLOCK_SIZE_H
            BLOCK_SIZE_CHUNK).toAlgKernel) s).map (·.readMem DtOut outAddr)
        = some (s.readMem DtPrepared
            (dtPreparedOffset s stride_dt_batch stride_dt_seqlen stride_dt_head
              chunk_size BLOCK_SIZE_H idx)) := by
  intro idx hActive
  simp [exec, chunked_cumsum_fused_slice, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd, Tile.scan,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        headIndex, chunkIndex, active, dtPreparedOffset, dtOutOffset,
        TileShape.dropInsertedIndex, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]
  rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
    (region := DACumsum) (R := DtOut) (hRR := hDtNe)
    (P := fun i : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
      s.pids 2 * BLOCK_SIZE_H + i.1.val < nheads ∧ i.2.1.val < chunk_size)]
  simp only [BlockState.setReg_readMem]
  let offsetFn : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] → Nat :=
    fun idx =>
      s.pids 0 * stride_dt_out_batch + s.pids 1 * stride_dt_out_chunk +
        (s.pids 2 * BLOCK_SIZE_H + idx.1.val) * stride_dt_out_head +
        idx.2.1.val * stride_dt_out_csize
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, dtOutOffset, headIndex, chunkIndex] using hDtOutInj
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  obtain ⟨hh, hc⟩ := hActive
  simp only [headIndex, chunkIndex] at hh hc
  simp [offsetFn, active, headIndex, chunkIndex, dtPreparedOffset, dtOutOffset,
    hh, hc]

/-- A masked scatter-store `foldl` leaves every memory cell it does not actively
hit unchanged (cell-level frame for one masked block store). -/
private theorem fused_foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (ρ : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = ρ ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s).mem ρ o = s.mem ρ o := by
  induction l generalizing s with
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

/-- Frame half: every memory cell not actively written by the two masked block
stores (`dt_out` and `dA_cumsum`) is preserved by the fused run. -/
theorem chunked_cumsum_fused_frame
    (DtPrepared A DtOut DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (s s1 : BlockState)
    (hExec : exec ((chunked_cumsum_fused_slice DtPrepared A DtOut DACumsum
      stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK).toAlgKernel) s = some s1)
    (ρ : RegionName) (o : Nat)
    (hmissDt : ∀ idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK],
      active s nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK idx →
      ¬(DtOut = ρ ∧
        dtOutOffset s stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head
          stride_dt_out_csize BLOCK_SIZE_H idx = o))
    (hmissDA : ∀ idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK],
      active s nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK idx →
      ¬(DACumsum = ρ ∧
        dACsOutOffset s stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head
          stride_dA_cs_csize BLOCK_SIZE_H idx = o)) :
    s1.mem ρ o = s.mem ρ o := by
  simp [exec, chunked_cumsum_fused_slice, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd, Tile.scan,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        TileShape.dropInsertedIndex, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?] at hExec
  subst hExec
  refine Eq.trans (fused_foldl_store_preserve_cell
    (P := fun i : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
      s.pids 2 * BLOCK_SIZE_H + i.1.val < nheads ∧ i.2.1.val < chunk_size)
    _ _ ρ o _ _ ?_) ?_
  · intro k _ hmk hcc
    exact hmissDA k (by simpa [active, headIndex, chunkIndex] using hmk)
      (by simpa [dACsOutOffset, headIndex, chunkIndex] using hcc)
  · simp only [BlockState.setReg_mem]
    refine Eq.trans (fused_foldl_store_preserve_cell
      (P := fun i : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        s.pids 2 * BLOCK_SIZE_H + i.1.val < nheads ∧ i.2.1.val < chunk_size)
      _ _ ρ o _ _ ?_) ?_
    · intro k _ hmk hcc
      exact hmissDt k (by simpa [active, headIndex, chunkIndex] using hmk)
        (by simpa [dtOutOffset, headIndex, chunkIndex] using hcc)
    · simp only [BlockState.setReg_mem]

/-- The genuine `dA_cumsum` closed form as a **pure function of the pinned
inputs**, indexed by the flat row-major lane `j` (`Fin (BLOCK_SIZE_H ·
BLOCK_SIZE_CHUNK)`). At lane `j` (row `j / C`, column `j % C`) it is the
within-chunk prefix sum `Σ_{k ≤ col, k < chunk_size} xs[(row, k)] · ys[j]` — the
`⊨`-form of `dAClosed`, reading only the pinned `DtPrepared`/`A` tiles. -/
noncomputable def dACumsumLaneClosed (chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (xs ys : Fin (BLOCK_SIZE_H * BLOCK_SIZE_CHUNK) → ℝ)
    (j : Fin (BLOCK_SIZE_H * BLOCK_SIZE_CHUNK)) : ℝ :=
  ∑ k ∈ (Finset.univ.filter
      (fun k : Fin BLOCK_SIZE_CHUNK =>
        k.val ≤ j.val % BLOCK_SIZE_CHUNK ∧ k.val < chunk_size)),
    xs (Lane2D.encode ((Lane2D.decode j).1, k, PUnit.unit)) * ys j

/-- **Lane bridge:** on an active lane the standalone `dAClosed` at the decoded
tile index equals the pure lane closed form, once `xs`/`ys` are pinned to the
kernel's `DtPrepared`/`A` reads. -/
theorem dAClosed_decode_eq_lane
    (DtPrepared A : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (s : BlockState) (xs ys : Fin (BLOCK_SIZE_H * BLOCK_SIZE_CHUNK) → ℝ)
    (hx : ∀ m : Fin (BLOCK_SIZE_H * BLOCK_SIZE_CHUNK),
      s.pids 2 * BLOCK_SIZE_H + m.val / BLOCK_SIZE_CHUNK < nheads ∧
        m.val % BLOCK_SIZE_CHUNK < chunk_size →
      s.readMem DtPrepared
          (s.pids 0 * stride_dt_batch +
            (s.pids 1 * chunk_size + m.val % BLOCK_SIZE_CHUNK) * stride_dt_seqlen +
            (s.pids 2 * BLOCK_SIZE_H + m.val / BLOCK_SIZE_CHUNK) * stride_dt_head)
        = xs m)
    (hy : ∀ m : Fin (BLOCK_SIZE_H * BLOCK_SIZE_CHUNK),
      s.pids 2 * BLOCK_SIZE_H + m.val / BLOCK_SIZE_CHUNK < nheads →
      s.readMem A ((s.pids 2 * BLOCK_SIZE_H + m.val / BLOCK_SIZE_CHUNK) * stride_A_head)
        = ys m)
    (j : Fin (BLOCK_SIZE_H * BLOCK_SIZE_CHUNK))
    (hj : s.pids 2 * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK < nheads ∧
      j.val % BLOCK_SIZE_CHUNK < chunk_size) :
    dAClosed s DtPrepared A stride_dt_batch stride_dt_seqlen stride_dt_head
        stride_A_head nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK
        (Lane2D.decode j)
      = dACumsumLaneClosed chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK xs ys j := by
  unfold dAClosed dACumsumLaneClosed dtElem
  have hhead : headIndex s BLOCK_SIZE_H (Lane2D.decode j).1
      = s.pids 2 * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK := by
    simp [headIndex, Lane2D.decode_row]
  have hcol : (Lane2D.decode j).2.1.val = j.val % BLOCK_SIZE_CHUNK := by
    simp [Lane2D.decode_col]
  apply Finset.sum_congr
  · -- the two index sets coincide (drop the uniformly-true `head < nheads`)
    apply Finset.filter_congr
    intro k _
    rw [hhead, hcol]
    constructor
    · rintro ⟨h1, _, h3⟩; exact ⟨h1, h3⟩
    · rintro ⟨h1, h3⟩; exact ⟨h1, hj.1, h3⟩
  · -- the summands coincide: pin `xs` at the lane `encode (row, k)` and `ys` at `j`
    intro k hk
    rw [Finset.mem_filter] at hk
    obtain ⟨_, hkcol, hkchunk⟩ := hk
    have hlane : s.pids 2 * BLOCK_SIZE_H +
          (Lane2D.encode ((Lane2D.decode j).1, k, PUnit.unit)).val / BLOCK_SIZE_CHUNK < nheads ∧
        (Lane2D.encode ((Lane2D.decode j).1, k, PUnit.unit)).val % BLOCK_SIZE_CHUNK < chunk_size := by
      rw [Lane2D.encode_div, Lane2D.encode_mod, Lane2D.decode_row]
      exact ⟨hj.1, hkchunk⟩
    rw [hhead, ← hy j hj.1,
      ← hx (Lane2D.encode ((Lane2D.decode j).1, k, PUnit.unit)) hlane,
      Lane2D.encode_div, Lane2D.encode_mod, Lane2D.decode_row]

/-- **The region-model masked Hoare triple for the fused kernel** — termination,
active-lane `dt_out` and `dA_cumsum` values, and frame off both written windows,
from any launch state whose active `DtPrepared`/`A` lanes hold `xs`/`ys`. This is
the `hrun` obligation of the `⊨` headline; the two outputs are read back at the
flat row-major lanes `Fin (BLOCK_SIZE_H · BLOCK_SIZE_CHUNK)`. -/
theorem chunked_cumsum_fused_region_run
    (DtPrepared A DtOut DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (hDtNe : DtOut ≠ DACumsum) (s₀' : BlockState)
    (hDtOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        dtOutOffset s₀' stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head
          stride_dt_out_csize BLOCK_SIZE_H idx))
    (hDACsInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        dACsOutOffset s₀' stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head
          stride_dA_cs_csize BLOCK_SIZE_H idx))
    (xs ys : Fin (BLOCK_SIZE_H * BLOCK_SIZE_CHUNK) → ℝ)
    (hx : ∀ j : Fin (BLOCK_SIZE_H * BLOCK_SIZE_CHUNK),
      s₀'.pids 2 * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK < nheads ∧
        j.val % BLOCK_SIZE_CHUNK < chunk_size →
      s₀'.readMem DtPrepared
          (s₀'.pids 0 * stride_dt_batch +
            (s₀'.pids 1 * chunk_size + j.val % BLOCK_SIZE_CHUNK) * stride_dt_seqlen +
            (s₀'.pids 2 * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) * stride_dt_head)
        = xs j)
    (hy : ∀ j : Fin (BLOCK_SIZE_H * BLOCK_SIZE_CHUNK),
      s₀'.pids 2 * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK < nheads →
      s₀'.readMem A
          ((s₀'.pids 2 * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) * stride_A_head)
        = ys j) :
    ∃ s1, exec ((chunked_cumsum_fused_slice DtPrepared A DtOut DACumsum
          stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
          stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head
          stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk
          stride_dA_cs_head stride_dA_cs_csize nheads chunk_size BLOCK_SIZE_H
          BLOCK_SIZE_CHUNK).toAlgKernel) s₀' = some s1
      ∧ (∀ j : Fin (BLOCK_SIZE_H * BLOCK_SIZE_CHUNK),
          s₀'.pids 2 * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK < nheads ∧
            j.val % BLOCK_SIZE_CHUNK < chunk_size →
          s1.readMem DtOut
              (s₀'.pids 0 * stride_dt_out_batch + s₀'.pids 1 * stride_dt_out_chunk +
                (s₀'.pids 2 * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) *
                  stride_dt_out_head +
                j.val % BLOCK_SIZE_CHUNK * stride_dt_out_csize)
            = xs j)
      ∧ (∀ j : Fin (BLOCK_SIZE_H * BLOCK_SIZE_CHUNK),
          s₀'.pids 2 * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK < nheads ∧
            j.val % BLOCK_SIZE_CHUNK < chunk_size →
          s1.readMem DACumsum
              (s₀'.pids 0 * stride_dA_cs_batch + s₀'.pids 1 * stride_dA_cs_chunk +
                (s₀'.pids 2 * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) *
                  stride_dA_cs_head +
                j.val % BLOCK_SIZE_CHUNK * stride_dA_cs_csize)
            = dACumsumLaneClosed chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK xs ys j)
      ∧ (∀ ρ o,
          (ρ ≠ DtOut ∨ ∀ j : Fin (BLOCK_SIZE_H * BLOCK_SIZE_CHUNK),
            (s₀'.pids 2 * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK < nheads ∧
              j.val % BLOCK_SIZE_CHUNK < chunk_size) →
            o ≠ s₀'.pids 0 * stride_dt_out_batch + s₀'.pids 1 * stride_dt_out_chunk +
                (s₀'.pids 2 * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) *
                  stride_dt_out_head +
                j.val % BLOCK_SIZE_CHUNK * stride_dt_out_csize) →
          (ρ ≠ DACumsum ∨ ∀ j : Fin (BLOCK_SIZE_H * BLOCK_SIZE_CHUNK),
            (s₀'.pids 2 * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK < nheads ∧
              j.val % BLOCK_SIZE_CHUNK < chunk_size) →
            o ≠ s₀'.pids 0 * stride_dA_cs_batch + s₀'.pids 1 * stride_dA_cs_chunk +
                (s₀'.pids 2 * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) *
                  stride_dA_cs_head +
                j.val % BLOCK_SIZE_CHUNK * stride_dA_cs_csize) →
          s1.mem ρ o = s₀'.mem ρ o) := by
  obtain ⟨s1, hExec⟩ := chunked_cumsum_fused_exec_isSome DtPrepared A DtOut DACumsum
    stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
    stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
    stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
    nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK s₀'
  -- `active (decode j)` unfolds to the row/chunk gate at lane `j`
  have hactive : ∀ j : Fin (BLOCK_SIZE_H * BLOCK_SIZE_CHUNK),
      (active s₀' nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK (Lane2D.decode j))
        = (s₀'.pids 2 * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK < nheads ∧
            j.val % BLOCK_SIZE_CHUNK < chunk_size) := by
    intro j
    simp [active, headIndex, chunkIndex, Lane2D.decode_row, Lane2D.decode_col]
  refine ⟨s1, hExec, ?_, ?_, ?_⟩
  · -- dt_out readback
    intro j hj
    have h := chunked_cumsum_fused_dt_out_correct DtPrepared A DtOut DACumsum
      stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK s₀' hDtNe hDtOutInj
      (Lane2D.decode j) (by rw [hactive j]; exact hj)
    simp only [hExec, Option.map_some] at h
    have h2 := Option.some.inj h
    rw [show s₀'.pids 0 * stride_dt_out_batch + s₀'.pids 1 * stride_dt_out_chunk +
          (s₀'.pids 2 * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) * stride_dt_out_head +
          j.val % BLOCK_SIZE_CHUNK * stride_dt_out_csize
        = dtOutOffset s₀' stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head
            stride_dt_out_csize BLOCK_SIZE_H (Lane2D.decode j) from by
        simp [dtOutOffset, headIndex, chunkIndex, Lane2D.decode_row, Lane2D.decode_col]]
    rw [h2, show dtPreparedOffset s₀' stride_dt_batch stride_dt_seqlen stride_dt_head
          chunk_size BLOCK_SIZE_H (Lane2D.decode j)
        = s₀'.pids 0 * stride_dt_batch +
            (s₀'.pids 1 * chunk_size + j.val % BLOCK_SIZE_CHUNK) * stride_dt_seqlen +
            (s₀'.pids 2 * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) * stride_dt_head from by
        simp [dtPreparedOffset, headIndex, chunkIndex, Lane2D.decode_row, Lane2D.decode_col]]
    exact hx j hj
  · -- dA_cumsum readback
    intro j hj
    have h := chunked_cumsum_fused_dA_cs_correct DtPrepared A DtOut DACumsum
      stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK s₀' hDACsInj
      (Lane2D.decode j) (by rw [hactive j]; exact hj)
    simp only [hExec, Option.map_some] at h
    have h2 := Option.some.inj h
    rw [dAComputedCumsumSpec_eq_dAClosed] at h2
    rw [dAClosed_decode_eq_lane DtPrepared A stride_dt_batch stride_dt_seqlen
      stride_dt_head stride_A_head nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK
      s₀' xs ys hx hy j hj] at h2
    rw [show s₀'.pids 0 * stride_dA_cs_batch + s₀'.pids 1 * stride_dA_cs_chunk +
          (s₀'.pids 2 * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) * stride_dA_cs_head +
          j.val % BLOCK_SIZE_CHUNK * stride_dA_cs_csize
        = dACsOutOffset s₀' stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head
            stride_dA_cs_csize BLOCK_SIZE_H (Lane2D.decode j) from by
        simp [dACsOutOffset, headIndex, chunkIndex, Lane2D.decode_row, Lane2D.decode_col]]
    exact h2
  · -- frame
    intro ρ o hDt hDA
    refine chunked_cumsum_fused_frame DtPrepared A DtOut DACumsum
      stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK s₀' s1 hExec ρ o
      (fun idx hidx => ?_) (fun idx hidx => ?_)
    · rintro ⟨hr, ho⟩
      rcases hDt with hne | hno
      · exact hne hr.symm
      · refine hno (Lane2D.encode idx) ?_ ?_
        · rw [Lane2D.encode_div, Lane2D.encode_mod]
          simpa [active, headIndex, chunkIndex] using hidx
        · rw [← ho]
          simp [dtOutOffset, headIndex, chunkIndex, Lane2D.encode_div,
            Lane2D.encode_mod]
    · rintro ⟨hr, ho⟩
      rcases hDA with hne | hno
      · exact hne hr.symm
      · refine hno (Lane2D.encode idx) ?_ ?_
        · rw [Lane2D.encode_div, Lane2D.encode_mod]
          simpa [active, headIndex, chunkIndex] using hidx
        · rw [← ho]
          simp [dACsOutOffset, headIndex, chunkIndex, Lane2D.encode_div,
            Lane2D.encode_mod]

/-! ## Dimension-general genuine correctness

The intermediate slice theorems above (`chunked_cumsum_dt_out_store_slice_compute_correct`,
`chunked_cumsum_dA_cs_store_slice_compute_correct`,
`chunked_cumsum_dA_cs_compute_slice_closed_form`) are already stated and proven
**general over all dimensions and strides** (`nheads`, `chunk_size`, the block
sizes, the number of chunks, and every stride). The theorems below package them
into general all-outputs / summary statements universally quantified over those
symbolic `Nat` parameters, mirroring the dimension-general top theorem of the
sibling `chunk_cumsum_kernel`. The genuine `dA_cumsum` specification is
`dAClosed` — the standalone within-chunk prefix `Finset.sum`
`Σ_{k ≤ idx.chunk, active} dt[head,k] · A[head]` — never a read-back of the
kernel's own output. Output-offset injectivity (`dtOutOffset` / `dACsOutOffset`
injective) is the trusted host-side address-layout side condition, carried as an
open hypothesis of the dimension-general top theorem. -/


/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **Dimension-general output coverage for chunked cumsum forward.** Over fully
symbolic dimensions and strides, the `dt_out` store slice, the `dA_cumsum` store
slice, and the `dA_cumsum` compute slice all realize their genuine
specifications; in particular the compute slice realizes the standalone closed
form `dAClosed` (the within-chunk prefix `Finset.sum`), not a read-back of the
kernel's own output. -/
theorem chunked_cumsum_fwd_all_outputs_compute_correct_general
    (DtPrepared DtOut DAcs A DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (s : BlockState)
    (hDtOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        dtOutOffset s stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head
          stride_dt_out_csize BLOCK_SIZE_H idx))
    (hDACsInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        dACsOutOffset s stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head
          stride_dA_cs_csize BLOCK_SIZE_H idx)) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunked_cumsum_dt_out_store_slice DtPrepared DtOut
        stride_dt_batch stride_dt_seqlen stride_dt_head stride_dt_out_batch
        stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize nheads
        chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
        (fun idx => (DtOut,
          dtOutOffset s stride_dt_out_batch stride_dt_out_chunk
            stride_dt_out_head stride_dt_out_csize BLOCK_SIZE_H idx)))
      (expected := fun idx =>
        s.readMem DtPrepared
          (dtPreparedOffset s stride_dt_batch stride_dt_seqlen stride_dt_head
            chunk_size BLOCK_SIZE_H idx))) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunked_cumsum_dA_cs_store_slice DAcs DACumsum
        stride_dt_batch stride_dt_seqlen stride_dt_head stride_dA_cs_batch
        stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize nheads
        chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
        (fun idx => (DACumsum,
          dACsOutOffset s stride_dA_cs_batch stride_dA_cs_chunk
            stride_dA_cs_head stride_dA_cs_csize BLOCK_SIZE_H idx)))
      (expected := fun idx =>
        s.readMem DAcs
          (dtPreparedOffset s stride_dt_batch stride_dt_seqlen stride_dt_head
            chunk_size BLOCK_SIZE_H idx))) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := chunked_cumsum_dA_cs_compute_slice DtPrepared A DACumsum
        stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
        stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head
        stride_dA_cs_csize nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK)
        (fun idx => (DACumsum,
          dACsOutOffset s stride_dA_cs_batch stride_dA_cs_chunk
            stride_dA_cs_head stride_dA_cs_csize BLOCK_SIZE_H idx)))
      (expected := fun idx =>
        dAClosed s DtPrepared A stride_dt_batch stride_dt_seqlen
          stride_dt_head stride_A_head nheads chunk_size BLOCK_SIZE_H
          BLOCK_SIZE_CHUNK idx)) := by
  refine ⟨?_, ?_, ?_⟩
  · exact chunked_cumsum_dt_out_store_slice_compute_correct DtPrepared DtOut
      stride_dt_batch stride_dt_seqlen stride_dt_head stride_dt_out_batch
      stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize nheads chunk_size
      BLOCK_SIZE_H BLOCK_SIZE_CHUNK s hDtOutInj
  · exact chunked_cumsum_dA_cs_store_slice_compute_correct DAcs DACumsum
      stride_dt_batch stride_dt_seqlen stride_dt_head stride_dA_cs_batch
      stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize nheads chunk_size
      BLOCK_SIZE_H BLOCK_SIZE_CHUNK s hDACsInj
  · exact chunked_cumsum_dA_cs_compute_slice_closed_form DtPrepared A DACumsum
      stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK s hDACsInj


/-! ## The three-pid `⊨` conversion (first consumer of `Masked3DKernelIO₂ₓ₂`) -/

/-- The fused chunked-cumsum kernel's masked two-input / two-output **IO
signature** over the **three-program-id** grid `(pid_b, pid_c, pid_h)` — the
first consumer of `Masked3DKernelIO₂ₓ₂`:

* `in1`/`in2`/`out1`/`out2` — the prepared `dt` tile `DtPrepared`, the per-head
  scale `A`, the `dt_out` buffer, and the `dA_cumsum` buffer;
* `B = BLOCK_SIZE_H · BLOCK_SIZE_CHUNK` — the `[head, chunk]` tile each program
  owns, indexed by the flat **row-major lane** `j ↦ (j / BLOCK_SIZE_CHUNK,
  j % BLOCK_SIZE_CHUNK)`;
* `read1`/`write1`/`write2` — the general-stride `(pid_b, pid_c, pid_h)`
  addresses: `pid_h` enters the head coordinate `pid_h · BLOCK_SIZE_H + row`,
  `pid_b`/`pid_c` the batch/chunk offsets — genuinely using the third program
  id;
* `read2` — the per-head scale address `(pid_h · BLOCK_SIZE_H + row) ·
  stride_A_head`, constant across a head's columns;
* `mask` — the active lanes `pid_h · BLOCK_SIZE_H + row < nheads ∧ col <
  chunk_size`; `read2Mask` is the head-only gate. -/
def chunkedCumsumFwdIO (DtPrepared A DtOut DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat) :
    Masked3DKernelIO₂ₓ₂ where
  kernel := chunked_cumsum_fused_slice DtPrepared A DtOut DACumsum
    stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
    stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
    stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
    nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK
  in1 := DtPrepared
  in2 := A
  out1 := DtOut
  out2 := DACumsum
  B := BLOCK_SIZE_H * BLOCK_SIZE_CHUNK
  read1 := fun p₀ p₁ p₂ j =>
    p₀ * stride_dt_batch +
      (p₁ * chunk_size + j.val % BLOCK_SIZE_CHUNK) * stride_dt_seqlen +
      (p₂ * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) * stride_dt_head
  read2 := fun _ _ p₂ j =>
    (p₂ * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) * stride_A_head
  write1 := fun p₀ p₁ p₂ j =>
    p₀ * stride_dt_out_batch + p₁ * stride_dt_out_chunk +
      (p₂ * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) * stride_dt_out_head +
      j.val % BLOCK_SIZE_CHUNK * stride_dt_out_csize
  write2 := fun p₀ p₁ p₂ j =>
    p₀ * stride_dA_cs_batch + p₁ * stride_dA_cs_chunk +
      (p₂ * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK) * stride_dA_cs_head +
      j.val % BLOCK_SIZE_CHUNK * stride_dA_cs_csize
  mask := fun _ _ p₂ j =>
    p₂ * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK < nheads ∧
      j.val % BLOCK_SIZE_CHUNK < chunk_size
  read2Mask := fun _ _ p₂ j => p₂ * BLOCK_SIZE_H + j.val / BLOCK_SIZE_CHUNK < nheads

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **The three-pid `⊨` headline for chunked cumsum forward.** On its masked
two-input / two-output IO signature over the 3-D launch grid, the fused kernel
implements `dt_out[j] = xs j` (the prepared `dt`) and `dA_cumsum[j] =
dACumsumLaneClosed …` — the standalone within-chunk prefix `Finset.sum`
`Σ_{k ≤ col, k < chunk_size} xs[(row, k)] · ys[j]` — **never** a read-back of the
kernel's own output. Fully general over all dimensions and strides and the two
program ids the spec sees (the third enters through the windows/masks). Side
conditions, all truth-forced: `DtOut ≠ DACumsum` (the two masked block stores
must not alias, so each output's readback sees through the other), and per-pid
output-offset injectivity (arbitrary strides may otherwise collide two lanes on
one cell — the same host-side address-layout condition carried by the slice
decomposition). No `0 < _` positivity is needed: the within-chunk cumsum is a
total `Finset.sum` and the `Fin (BLOCK_SIZE_H · BLOCK_SIZE_CHUNK)` lanes supply
their own `0 < BLOCK_SIZE_CHUNK` for the row-major decode. -/
specification chunked_cumsum_fwd_correctness
    (DtPrepared A DtOut DACumsum : RegionName)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (hDtNe : DtOut ≠ DACumsum) (hCK : 0 < BLOCK_SIZE_CHUNK)
    (hDtOutInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        p₀ * stride_dt_out_batch + p₁ * stride_dt_out_chunk +
          (p₂ * BLOCK_SIZE_H + idx.1.val) * stride_dt_out_head +
          idx.2.1.val * stride_dt_out_csize))
    (hDACsInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        p₀ * stride_dA_cs_batch + p₁ * stride_dA_cs_chunk +
          (p₂ * BLOCK_SIZE_H + idx.1.val) * stride_dA_cs_head +
          idx.2.1.val * stride_dA_cs_csize)) :
    chunkedCumsumFwdIO DtPrepared A DtOut DACumsum
        stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
        stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
        stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
        nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK ⊨
      fun _ _ xs ys =>
        (fun j => xs j,
         fun j => dACumsumLaneClosed chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK xs ys j) := by
  refine Masked3DKernelIO₂ₓ₂.Implements.intro _ ?_ ?_ ?_
  · exact chunked_cumsum_fused_flattenOk DtPrepared A DtOut DACumsum
      stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK
  · -- trace safety: bridge the `Fin B` bounds to the `(head, chunk)` walk
    intro bounds s h1 h2 h3 h4
    refine chunked_cumsum_fused_traceSafe DtPrepared A DtOut DACumsum
      stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK bounds s
      (fun a b ha hb => ?_) (fun a ha => ?_) (fun a b ha hb => ?_)
      (fun a b ha hb => ?_)
    · simpa [chunkedCumsumFwdIO, Lane2D.encode_div, Lane2D.encode_mod]
        using h1 (Lane2D.encode (a, b, PUnit.unit))
          (by simp [chunkedCumsumFwdIO, Lane2D.encode_div, Lane2D.encode_mod];
              exact ⟨ha, hb⟩)
    · have h := h2 (Lane2D.encode (a, ⟨0, hCK⟩, PUnit.unit))
        (by simp [chunkedCumsumFwdIO, Lane2D.encode_div, Lane2D.encode_mod]; exact ha)
      simpa [chunkedCumsumFwdIO, Lane2D.encode_div] using h
    · simpa [chunkedCumsumFwdIO, Lane2D.encode_div, Lane2D.encode_mod]
        using h3 (Lane2D.encode (a, b, PUnit.unit))
          (by simp [chunkedCumsumFwdIO, Lane2D.encode_div, Lane2D.encode_mod];
              exact ⟨ha, hb⟩)
    · simpa [chunkedCumsumFwdIO, Lane2D.encode_div, Lane2D.encode_mod]
        using h4 (Lane2D.encode (a, b, PUnit.unit))
          (by simp [chunkedCumsumFwdIO, Lane2D.encode_div, Lane2D.encode_mod];
              exact ⟨ha, hb⟩)
  · -- the region-model run, assembled from the fused region_run
    intro s₀ xs ys hx hy
    obtain ⟨s1, hexec, hval1, hval2, hframe⟩ :=
      chunked_cumsum_fused_region_run DtPrepared A DtOut DACumsum
        stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
        stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
        stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
        nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK hDtNe s₀
        (by simpa [dtOutOffset, headIndex, chunkIndex]
            using hDtOutInj (s₀.pids 0) (s₀.pids 1) (s₀.pids 2))
        (by simpa [dACsOutOffset, headIndex, chunkIndex]
            using hDACsInj (s₀.pids 0) (s₀.pids 1) (s₀.pids 2))
        xs ys hx hy
    exact ⟨s1, hexec, hval1, hval2, hframe⟩

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **Dimension-general genuine correctness summary for chunked cumsum forward.**
The full Python surface (with the `HAS_DT_BIAS` / `DT_SOFTPLUS` flags) lowers to
the algorithm layer for arbitrary dimensions and strides, and — given the
truth-forced side conditions — the fused verified kernel satisfies the genuine
three-pid `⊨` contract (`dt_out` = prepared `dt`, `dA_cumsum` = the standalone
within-chunk prefix `Finset.sum`). -/
specification chunked_cumsum_fwd_summary_general
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr
      DtPrepared A DtOut DACumsum : RegionName)
    (batch seqlen nheads chunk_size : Nat)
    (dt_min dt_max : ℝ)
    (stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_bias_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize : Nat)
    (DT_SOFTPLUS HAS_DT_BIAS : Bool)
    (BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (hDtNe : DtOut ≠ DACumsum) (hCK : 0 < BLOCK_SIZE_CHUNK)
    (hDtOutInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        p₀ * stride_dt_out_batch + p₁ * stride_dt_out_chunk +
          (p₂ * BLOCK_SIZE_H + idx.1.val) * stride_dt_out_head +
          idx.2.1.val * stride_dt_out_csize))
    (hDACsInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_H, BLOCK_SIZE_CHUNK] =>
        p₀ * stride_dA_cs_batch + p₁ * stride_dA_cs_chunk +
          (p₂ * BLOCK_SIZE_H + idx.1.val) * stride_dA_cs_head +
          idx.2.1.val * stride_dA_cs_csize)) :
    (∃ alg, (chunked_cumsum_fwd_surface dt_ptr A_ptr dt_bias_ptr dt_out_ptr
      dA_cumsum_ptr batch seqlen nheads chunk_size dt_min dt_max stride_dt_batch
      stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK).toAlgorithm?
        = Except.ok alg) ∧
    (chunkedCumsumFwdIO DtPrepared A DtOut DACumsum
        stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
        stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
        stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
        nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK ⊨
      fun _ _ xs ys =>
        (fun j => xs j,
         fun j => dACumsumLaneClosed chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK xs ys j)) := by
  refine ⟨?_, ?_⟩
  · exact chunked_cumsum_fwd_surface_toAlgorithm_supported dt_ptr A_ptr
      dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size dt_min
      dt_max stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk
      stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk
      stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H
      BLOCK_SIZE_CHUNK
  · exact chunked_cumsum_fwd_correctness DtPrepared A DtOut DACumsum
      stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      nheads chunk_size BLOCK_SIZE_H BLOCK_SIZE_CHUNK hDtNe hCK hDtOutInj hDACsInj

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd
