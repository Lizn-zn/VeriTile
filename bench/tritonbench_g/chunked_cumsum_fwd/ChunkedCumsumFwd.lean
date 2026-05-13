import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Surface transcription of `chunked_cumsum_fwd.py`'s
`_chunk_cumsum_fwd_kernel`.

This preserves the optional `dt_bias` and softplus branches, the clamp via
`tl.minimum(tl.maximum(...))`, the masked `dt_out` store, and the `dA_cumsum`
store computed with `tl.cumsum` along the chunk axis. -/
def chunked_cumsum_fwd_surface
    (Dt A DtBias DtOut DACumsum : RegionName)
    (_batch seqlen nheads chunk_size
      stride_dt_batch stride_dt_seqlen stride_dt_head
      stride_A_head stride_dt_bias_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize
      BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    (dt_min dt_max : ℝ)
    (DT_SOFTPLUS HAS_DT_BIAS : Bool) :
    ComputeKernel := triton {
  pid_b = tl.program_id(axis=0)
  pid_c = tl.program_id(axis=1)
  pid_h = tl.program_id(axis=2)
  Dt += pid_b * $(stride_dt_batch) +
    pid_c * $(chunk_size) * $(stride_dt_seqlen)
  DtOut += pid_b * $(stride_dt_out_batch) +
    pid_c * $(stride_dt_out_chunk)
  DACumsum += pid_b * $(stride_dA_cs_batch) +
    pid_c * $(stride_dA_cs_chunk)
  offs_h = pid_h * $(BLOCK_SIZE_H) + tl.arange(0, $(BLOCK_SIZE_H))
  offs_c = tl.arange(0, $(BLOCK_SIZE_CHUNK))
  dt_ptrs = Dt + offs_h[:, None] * $(stride_dt_head) +
    offs_c[None, :] * $(stride_dt_seqlen)
  A_ptrs = A + offs_h * $(stride_A_head)
  dt_out_ptrs = DtOut + offs_h[:, None] * $(stride_dt_out_head) +
    offs_c[None, :] * $(stride_dt_out_csize)
  dA_cs_ptrs = DACumsum + offs_h[:, None] * $(stride_dA_cs_head) +
    offs_c[None, :] * $(stride_dA_cs_csize)
  chunk_size_limit = min($(chunk_size), $(seqlen) - pid_c * $(chunk_size))
  dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < $(nheads)) &
    (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
  if HAS_DT_BIAS {
    dt_bias = tl.load(DtBias + offs_h * $(stride_dt_bias_head),
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
  a = tl.load(A_ptrs, mask=offs_h < $(nheads), other=0.0).to(tl.float32)
  dA = dt * a[:, None]
  dA_cs = tl.cumsum(dA, axis=1)
  tl.store(dA_cs_ptrs, dA_cs, mask=(offs_h[:, None] < $(nheads)) &
    (offs_c[None, :] < $(chunk_size)))
}

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
  simp [exec, chunked_cumsum_dt_out_store_slice, stepStmts, stepStmt, evalOp,
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

end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd
