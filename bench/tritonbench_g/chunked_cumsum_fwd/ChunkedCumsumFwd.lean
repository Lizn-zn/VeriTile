import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd

open VeriTile.Triton

set_option linter.unusedSimpArgs false

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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
    ((Tile.scan .sum ⟨1, by simp⟩
      (dATile s DtPrepared A stride_dt_batch stride_dt_seqlen
        stride_dt_head stride_A_head nheads chunk_size BLOCK_SIZE_H
        BLOCK_SIZE_CHUNK)).data idx)

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
    ComputeCorrect.Realizes
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

/-! ## Python test-shape wrappers

`chunked_cumsum_fwd.py`'s checked tests use `dt.shape = (2, 10, 4)` and
`chunk_size = 5`. Contiguous `dt` strides are `(40, 4, 1)`. The output tensors
have shape `(2, 4, 2, 5)` and the kernel receives strides
`(batch=40, chunk=5, head=10, csize=1)`. The autotune set includes
`BLOCK_SIZE_H = 4`; `BLOCK_SIZE_CHUNK = next_power_of_2(5) = 8`. -/

theorem chunked_cumsum_dt_out_python_test_shape_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [4, 8] => dtOutOffset s 40 5 10 1 4 idx) := by
  rintro ⟨⟨ha, hha⟩, ⟨ca, hca⟩, _⟩ ⟨⟨hb, hhb⟩, ⟨cb, hcb⟩, _⟩ h
  simp [dtOutOffset, headIndex, chunkIndex] at h
  have hh : ha = hb := by omega
  have hc : ca = cb := by omega
  subst hb
  subst cb
  rfl

theorem chunked_cumsum_dA_cs_python_test_shape_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [4, 8] => dACsOutOffset s 40 5 10 1 4 idx) := by
  rintro ⟨⟨ha, hha⟩, ⟨ca, hca⟩, _⟩ ⟨⟨hb, hhb⟩, ⟨cb, hcb⟩, _⟩ h
  simp [dACsOutOffset, headIndex, chunkIndex] at h
  have hh : ha = hb := by omega
  have hc : ca = cb := by omega
  subst hb
  subst cb
  rfl

theorem chunked_cumsum_dt_out_python_test_shape_compute_correct
    (DtPrepared DtOut : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dt_out_store_slice DtPrepared DtOut
        40 4 1 40 5 10 1 4 5 4 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 5 4 8)
        (fun idx : TileIndex [4, 8] => (DtOut, dtOutOffset s 40 5 10 1 4 idx)))
      (expected := fun idx : TileIndex [4, 8] =>
        s.readMem DtPrepared (dtPreparedOffset s 40 4 1 5 4 idx)) := by
  exact chunked_cumsum_dt_out_store_slice_compute_correct DtPrepared DtOut
    40 4 1 40 5 10 1 4 5 4 8 s
    (chunked_cumsum_dt_out_python_test_shape_offset_injective s)

theorem chunked_cumsum_dA_cs_store_python_test_shape_compute_correct
    (DAcs DACumsum : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dA_cs_store_slice DAcs DACumsum
        40 4 1 40 5 10 1 4 5 4 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 5 4 8)
        (fun idx : TileIndex [4, 8] => (DACumsum, dACsOutOffset s 40 5 10 1 4 idx)))
      (expected := fun idx : TileIndex [4, 8] =>
        s.readMem DAcs (dtPreparedOffset s 40 4 1 5 4 idx)) := by
  exact chunked_cumsum_dA_cs_store_slice_compute_correct DAcs DACumsum
    40 4 1 40 5 10 1 4 5 4 8 s
    (chunked_cumsum_dA_cs_python_test_shape_offset_injective s)

theorem chunked_cumsum_dA_cs_compute_python_test_shape_compute_correct
    (DtPrepared A DACumsum : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dA_cs_compute_slice DtPrepared A DACumsum
        40 4 1 1 40 5 10 1 4 5 4 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 5 4 8)
        (fun idx : TileIndex [4, 8] => (DACumsum, dACsOutOffset s 40 5 10 1 4 idx)))
      (expected := fun idx : TileIndex [4, 8] =>
        dAComputedCumsumSpec s DtPrepared A 40 4 1 1 4 5 4 8 idx) := by
  exact chunked_cumsum_dA_cs_compute_slice_compute_correct DtPrepared A DACumsum
    40 4 1 1 40 5 10 1 4 5 4 8 s
    (chunked_cumsum_dA_cs_python_test_shape_offset_injective s)

/-- Python test-shape output coverage for chunked cumsum forward: `dt_out` and
both `dA_cumsum` writeback surfaces realize the checked masked output shapes. -/
theorem chunked_cumsum_fwd_python_test_shape_all_outputs_compute_correct
    (DtPrepared DtOut DAcs A DACumsum : RegionName) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dt_out_store_slice DtPrepared DtOut
        40 4 1 40 5 10 1 4 5 4 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 5 4 8)
        (fun idx : TileIndex [4, 8] => (DtOut, dtOutOffset s 40 5 10 1 4 idx)))
      (expected := fun idx : TileIndex [4, 8] =>
        s.readMem DtPrepared (dtPreparedOffset s 40 4 1 5 4 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dA_cs_store_slice DAcs DACumsum
        40 4 1 40 5 10 1 4 5 4 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 5 4 8)
        (fun idx : TileIndex [4, 8] =>
          (DACumsum, dACsOutOffset s 40 5 10 1 4 idx)))
      (expected := fun idx : TileIndex [4, 8] =>
        s.readMem DAcs (dtPreparedOffset s 40 4 1 5 4 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dA_cs_compute_slice DtPrepared A DACumsum
        40 4 1 1 40 5 10 1 4 5 4 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 5 4 8)
        (fun idx : TileIndex [4, 8] =>
          (DACumsum, dACsOutOffset s 40 5 10 1 4 idx)))
      (expected := fun idx : TileIndex [4, 8] =>
        dAComputedCumsumSpec s DtPrepared A 40 4 1 1 4 5 4 8 idx)) := by
  constructor
  · exact chunked_cumsum_dt_out_python_test_shape_compute_correct
      DtPrepared DtOut s
  constructor
  · exact chunked_cumsum_dA_cs_store_python_test_shape_compute_correct
      DAcs DACumsum s
  · exact chunked_cumsum_dA_cs_compute_python_test_shape_compute_correct
      DtPrepared A DACumsum s

/-! ## Python test-case surface wrappers

The Python regression calls `_chunk_cumsum_fwd` four times with the same
contiguous shape and block metadata, varying only `HAS_DT_BIAS` and
`DT_SOFTPLUS`. These wrappers pin those four entry points to the source-level
surface, including the optional bias/softplus branches before the `dt_out` and
`dA_cumsum` stores. -/

theorem chunked_cumsum_fwd_python_test_case1_surface_toAlgorithm_supported
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr : RegionName)
    (dt_min dt_max : ℝ) :
    ∃ alg, (chunked_cumsum_fwd_surface dt_ptr A_ptr dt_bias_ptr dt_out_ptr
      dA_cumsum_ptr
      2 10 4 5 dt_min dt_max
      40 4 1 1 1
      40 5 10 1
      40 5 10 1
      Bool.false Bool.false 4 8).toAlgorithm? = Except.ok alg := by
  exact chunked_cumsum_fwd_surface_toAlgorithm_supported
    dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr
    2 10 4 5 dt_min dt_max
    40 4 1 1 1
    40 5 10 1
    40 5 10 1
    Bool.false Bool.false 4 8

theorem chunked_cumsum_fwd_python_test_case2_surface_toAlgorithm_supported
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr : RegionName)
    (dt_min dt_max : ℝ) :
    ∃ alg, (chunked_cumsum_fwd_surface dt_ptr A_ptr dt_bias_ptr dt_out_ptr
      dA_cumsum_ptr
      2 10 4 5 dt_min dt_max
      40 4 1 1 1
      40 5 10 1
      40 5 10 1
      Bool.false Bool.true 4 8).toAlgorithm? = Except.ok alg := by
  exact chunked_cumsum_fwd_surface_toAlgorithm_supported
    dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr
    2 10 4 5 dt_min dt_max
    40 4 1 1 1
    40 5 10 1
    40 5 10 1
    Bool.false Bool.true 4 8

theorem chunked_cumsum_fwd_python_test_case3_surface_toAlgorithm_supported
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr : RegionName)
    (dt_min dt_max : ℝ) :
    ∃ alg, (chunked_cumsum_fwd_surface dt_ptr A_ptr dt_bias_ptr dt_out_ptr
      dA_cumsum_ptr
      2 10 4 5 dt_min dt_max
      40 4 1 1 1
      40 5 10 1
      40 5 10 1
      Bool.true Bool.false 4 8).toAlgorithm? = Except.ok alg := by
  exact chunked_cumsum_fwd_surface_toAlgorithm_supported
    dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr
    2 10 4 5 dt_min dt_max
    40 4 1 1 1
    40 5 10 1
    40 5 10 1
    Bool.true Bool.false 4 8

theorem chunked_cumsum_fwd_python_test_case4_surface_toAlgorithm_supported
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr : RegionName)
    (dt_min dt_max : ℝ) :
    ∃ alg, (chunked_cumsum_fwd_surface dt_ptr A_ptr dt_bias_ptr dt_out_ptr
      dA_cumsum_ptr
      2 10 4 5 dt_min dt_max
      40 4 1 1 1
      40 5 10 1
      40 5 10 1
      Bool.true Bool.true 4 8).toAlgorithm? = Except.ok alg := by
  exact chunked_cumsum_fwd_surface_toAlgorithm_supported
    dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr
    2 10 4 5 dt_min dt_max
    40 4 1 1 1
    40 5 10 1
    40 5 10 1
    Bool.true Bool.true 4 8

/-- Public Python case 1 summary: no bias and no softplus. The full surface
lowers and the checked `dt_out`/`dA_cumsum` output slices are compute-correct. -/
theorem chunked_cumsum_fwd_python_test_case1_output_summary
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr
      DtPrepared DtOut DAcs A DACumsum : RegionName)
    (dt_min dt_max : ℝ) (s : BlockState) :
    (∃ alg, (chunked_cumsum_fwd_surface dt_ptr A_ptr dt_bias_ptr dt_out_ptr
      dA_cumsum_ptr
      2 10 4 5 dt_min dt_max
      40 4 1 1 1
      40 5 10 1
      40 5 10 1
      Bool.false Bool.false 4 8).toAlgorithm? = Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dt_out_store_slice DtPrepared DtOut
        40 4 1 40 5 10 1 4 5 4 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 5 4 8)
        (fun idx : TileIndex [4, 8] => (DtOut, dtOutOffset s 40 5 10 1 4 idx)))
      (expected := fun idx : TileIndex [4, 8] =>
        s.readMem DtPrepared (dtPreparedOffset s 40 4 1 5 4 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dA_cs_store_slice DAcs DACumsum
        40 4 1 40 5 10 1 4 5 4 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 5 4 8)
        (fun idx : TileIndex [4, 8] =>
          (DACumsum, dACsOutOffset s 40 5 10 1 4 idx)))
      (expected := fun idx : TileIndex [4, 8] =>
        s.readMem DAcs (dtPreparedOffset s 40 4 1 5 4 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dA_cs_compute_slice DtPrepared A DACumsum
        40 4 1 1 40 5 10 1 4 5 4 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 5 4 8)
        (fun idx : TileIndex [4, 8] =>
          (DACumsum, dACsOutOffset s 40 5 10 1 4 idx)))
      (expected := fun idx : TileIndex [4, 8] =>
        dAComputedCumsumSpec s DtPrepared A 40 4 1 1 4 5 4 8 idx))) := by
  constructor
  · exact chunked_cumsum_fwd_python_test_case1_surface_toAlgorithm_supported
      dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr dt_min dt_max
  · exact chunked_cumsum_fwd_python_test_shape_all_outputs_compute_correct
      DtPrepared DtOut DAcs A DACumsum s

/-- Public Python case 2 summary: bias enabled and softplus disabled. -/
theorem chunked_cumsum_fwd_python_test_case2_output_summary
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr
      DtPrepared DtOut DAcs A DACumsum : RegionName)
    (dt_min dt_max : ℝ) (s : BlockState) :
    (∃ alg, (chunked_cumsum_fwd_surface dt_ptr A_ptr dt_bias_ptr dt_out_ptr
      dA_cumsum_ptr
      2 10 4 5 dt_min dt_max
      40 4 1 1 1
      40 5 10 1
      40 5 10 1
      Bool.false Bool.true 4 8).toAlgorithm? = Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dt_out_store_slice DtPrepared DtOut
        40 4 1 40 5 10 1 4 5 4 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 5 4 8)
        (fun idx : TileIndex [4, 8] => (DtOut, dtOutOffset s 40 5 10 1 4 idx)))
      (expected := fun idx : TileIndex [4, 8] =>
        s.readMem DtPrepared (dtPreparedOffset s 40 4 1 5 4 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dA_cs_store_slice DAcs DACumsum
        40 4 1 40 5 10 1 4 5 4 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 5 4 8)
        (fun idx : TileIndex [4, 8] =>
          (DACumsum, dACsOutOffset s 40 5 10 1 4 idx)))
      (expected := fun idx : TileIndex [4, 8] =>
        s.readMem DAcs (dtPreparedOffset s 40 4 1 5 4 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dA_cs_compute_slice DtPrepared A DACumsum
        40 4 1 1 40 5 10 1 4 5 4 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 5 4 8)
        (fun idx : TileIndex [4, 8] =>
          (DACumsum, dACsOutOffset s 40 5 10 1 4 idx)))
      (expected := fun idx : TileIndex [4, 8] =>
        dAComputedCumsumSpec s DtPrepared A 40 4 1 1 4 5 4 8 idx))) := by
  constructor
  · exact chunked_cumsum_fwd_python_test_case2_surface_toAlgorithm_supported
      dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr dt_min dt_max
  · exact chunked_cumsum_fwd_python_test_shape_all_outputs_compute_correct
      DtPrepared DtOut DAcs A DACumsum s

/-- Public Python case 3 summary: no bias and softplus enabled. -/
theorem chunked_cumsum_fwd_python_test_case3_output_summary
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr
      DtPrepared DtOut DAcs A DACumsum : RegionName)
    (dt_min dt_max : ℝ) (s : BlockState) :
    (∃ alg, (chunked_cumsum_fwd_surface dt_ptr A_ptr dt_bias_ptr dt_out_ptr
      dA_cumsum_ptr
      2 10 4 5 dt_min dt_max
      40 4 1 1 1
      40 5 10 1
      40 5 10 1
      Bool.true Bool.false 4 8).toAlgorithm? = Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dt_out_store_slice DtPrepared DtOut
        40 4 1 40 5 10 1 4 5 4 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 5 4 8)
        (fun idx : TileIndex [4, 8] => (DtOut, dtOutOffset s 40 5 10 1 4 idx)))
      (expected := fun idx : TileIndex [4, 8] =>
        s.readMem DtPrepared (dtPreparedOffset s 40 4 1 5 4 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dA_cs_store_slice DAcs DACumsum
        40 4 1 40 5 10 1 4 5 4 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 5 4 8)
        (fun idx : TileIndex [4, 8] =>
          (DACumsum, dACsOutOffset s 40 5 10 1 4 idx)))
      (expected := fun idx : TileIndex [4, 8] =>
        s.readMem DAcs (dtPreparedOffset s 40 4 1 5 4 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dA_cs_compute_slice DtPrepared A DACumsum
        40 4 1 1 40 5 10 1 4 5 4 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 5 4 8)
        (fun idx : TileIndex [4, 8] =>
          (DACumsum, dACsOutOffset s 40 5 10 1 4 idx)))
      (expected := fun idx : TileIndex [4, 8] =>
        dAComputedCumsumSpec s DtPrepared A 40 4 1 1 4 5 4 8 idx))) := by
  constructor
  · exact chunked_cumsum_fwd_python_test_case3_surface_toAlgorithm_supported
      dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr dt_min dt_max
  · exact chunked_cumsum_fwd_python_test_shape_all_outputs_compute_correct
      DtPrepared DtOut DAcs A DACumsum s

/-- Public Python case 4 summary: both bias and softplus enabled. -/
theorem chunked_cumsum_fwd_python_test_case4_output_summary
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr
      DtPrepared DtOut DAcs A DACumsum : RegionName)
    (dt_min dt_max : ℝ) (s : BlockState) :
    (∃ alg, (chunked_cumsum_fwd_surface dt_ptr A_ptr dt_bias_ptr dt_out_ptr
      dA_cumsum_ptr
      2 10 4 5 dt_min dt_max
      40 4 1 1 1
      40 5 10 1
      40 5 10 1
      Bool.true Bool.true 4 8).toAlgorithm? = Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dt_out_store_slice DtPrepared DtOut
        40 4 1 40 5 10 1 4 5 4 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 5 4 8)
        (fun idx : TileIndex [4, 8] => (DtOut, dtOutOffset s 40 5 10 1 4 idx)))
      (expected := fun idx : TileIndex [4, 8] =>
        s.readMem DtPrepared (dtPreparedOffset s 40 4 1 5 4 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dA_cs_store_slice DAcs DACumsum
        40 4 1 40 5 10 1 4 5 4 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 5 4 8)
        (fun idx : TileIndex [4, 8] =>
          (DACumsum, dACsOutOffset s 40 5 10 1 4 idx)))
      (expected := fun idx : TileIndex [4, 8] =>
        s.readMem DAcs (dtPreparedOffset s 40 4 1 5 4 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := chunked_cumsum_dA_cs_compute_slice DtPrepared A DACumsum
        40 4 1 1 40 5 10 1 4 5 4 8)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s 4 5 4 8)
        (fun idx : TileIndex [4, 8] =>
          (DACumsum, dACsOutOffset s 40 5 10 1 4 idx)))
      (expected := fun idx : TileIndex [4, 8] =>
        dAComputedCumsumSpec s DtPrepared A 40 4 1 1 4 5 4 8 idx))) := by
  constructor
  · exact chunked_cumsum_fwd_python_test_case4_surface_toAlgorithm_supported
      dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr dt_min dt_max
  · exact chunked_cumsum_fwd_python_test_shape_all_outputs_compute_correct
      DtPrepared DtOut DAcs A DACumsum s

end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd
