import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.FlashAttn

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Surface transcription/proof-oriented final output-store slice of `flash_attn.py`'s
`_fwd_kernel`.

The full kernel streams over K/V blocks, computes a numerically stable attention
accumulator, and also writes the log-sum-exp vector `L`. This slice starts after
`out_buffer = out_buffer / denom[:, None]` with a precomputed `OutBuffer` tile
and proves the final unmasked `O_block_ptr` writeback. It preserves the source
base offset, which is derived from `stride_q_head`. The inner `tl.float32`
online-softmax state and K/V dot loop are outside this slice. -/
def flash_attn_output_store_slice
    (OutBuffer O : RegionName)
    (stride_buf_h stride_buf_m stride_buf_d
      stride_q_head stride_o_seqlen stride_o_dim
      BLOCK_M DIM : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(DIM))
  out_buffer = tl.load(OutBuffer + off_bs_head * $(stride_buf_h) +
      offs_m[:, None] * $(stride_buf_m) + offs_d[None, :] * $(stride_buf_d))
  tl.store(O + off_bs_head * $(stride_q_head) +
      offs_m[:, None] * $(stride_o_seqlen) + offs_d[None, :] * $(stride_o_dim),
      (out_buffer).to(tl.float16))
}

/-- Surface transcription of `flash_attn.py`'s final `L` vector store.

The full kernel computes the streaming row max and denominator, then stores
`max + tl.math.log2(denom)` into `L + off_bs_head * SEQLEN + off_m`. This
surface starts from precomputed `Max` and `Denom` row tiles and preserves that
addressing. -/
def flash_attn_l_store_slice
    (Max Denom L : RegionName)
    (stride_max_h stride_max_m stride_den_h stride_den_m
      SEQLEN BLOCK_M : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)
  off_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  max_row = tl.load(Max + off_bs_head * $(stride_max_h) + off_m * $(stride_max_m))
  denom = tl.load(Denom + off_bs_head * $(stride_den_h) + off_m * $(stride_den_m))
  tl.store(L + off_bs_head * $(SEQLEN) + off_m, max_row + tl.log2(denom))
}

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def dIndex (idx : TileIndex [BLOCK_M, DIM]) : Nat :=
  idx.2.1.val

def bufferOffset
    (s : BlockState)
    (stride_buf_h stride_buf_m stride_buf_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, DIM]) : Nat :=
  s.pids 1 * stride_buf_h +
    mIndex s BLOCK_M idx.1 * stride_buf_m + dIndex idx * stride_buf_d

def outOffset
    (s : BlockState)
    (stride_q_head stride_o_seqlen stride_o_dim BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, DIM]) : Nat :=
  s.pids 1 * stride_q_head +
    mIndex s BLOCK_M idx.1 * stride_o_seqlen + dIndex idx * stride_o_dim

private theorem foldl_writeMemTyped_fp16_preserves {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → TileCarrier TileDType.fp16)
    (o : Nat) (l : List α) :
    ∀ s : BlockState,
      (∀ k ∈ l, offsetFn k ≠ o) →
        ((l.foldl
          (fun acc k => acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k))
          s).mem region o) = s.mem region o := by
  induction l with
  | nil =>
      intro s _h
      rfl
  | cons hd tl ih =>
      intro s h
      rw [List.foldl_cons]
      have htl : ∀ k ∈ tl, offsetFn k ≠ o :=
        fun k hk => h k (List.mem_cons_of_mem hd hk)
      have hhd : offsetFn hd ≠ o := h hd (List.mem_cons_self)
      rw [ih _ htl]
      unfold BlockState.writeMemTyped BlockState.writeMemAs
      change
        (if region = region ∧ o = offsetFn hd then
          MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn hd)))
        else
          s.mem region o) = s.mem region o
      rw [if_neg (by
        intro hsame
        exact hhd hsame.2.symm)]

private theorem scatter_memcell_fp16_nd {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → TileCarrier TileDType.fp16)
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k => acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k))
       s).mem region (offsetFn i)
    = MemCell.of .fp16
        (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i))) := by
  let l := TileShape.allIndices shape
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  change ((l.foldl
       (fun acc k => acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k))
       s).mem region (offsetFn i))
    = MemCell.of .fp16
        (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, _⟩ := h_nodup
  have hl' : l = l₁ ++ i :: l₂ := by
    simpa [l] using hl
  rw [hl', List.foldl_append, List.foldl_cons]
  have h_l2_not_in : ∀ k ∈ l₂, offsetFn k ≠ offsetFn i := by
    intro k hk heq
    have hki : k = i := h_inj heq
    subst hki
    exact hi_notin_l2 hk
  rw [foldl_writeMemTyped_fp16_preserves offsetFn valueFn (offsetFn i) l₂ _ h_l2_not_in]
  unfold BlockState.writeMemTyped BlockState.writeMemAs
  change
    (if region = region ∧ offsetFn i = offsetFn i then
      MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
    else
      (List.foldl
        (fun acc k => acc.writeMemTyped .fp16 region (offsetFn k) (valueFn k))
        s l₁).mem region (offsetFn i))
      =
      MemCell.of .fp16 (FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)))
  rw [if_pos ⟨rfl, rfl⟩]

/-- Algorithm-layer correctness for the final FlashAttention output store. -/
theorem flash_attn_output_store_slice_correct
    (OutBuffer O : RegionName)
    (stride_buf_h stride_buf_m stride_buf_d
      stride_q_head stride_o_seqlen stride_o_dim
      BLOCK_M DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, DIM] =>
        outOffset s stride_q_head stride_o_seqlen stride_o_dim BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, DIM],
      let outAddr := outOffset s stride_q_head stride_o_seqlen stride_o_dim
        BLOCK_M idx
      (exec (flash_attn_output_store_slice OutBuffer O stride_buf_h
            stride_buf_m stride_buf_d stride_q_head stride_o_seqlen
            stride_o_dim BLOCK_M DIM) s).map (·.mem O outAddr)
        = some (MemCell.of .fp16
            (FloatDType.real.cast FloatDType.fp16
              (some (s.readMem OutBuffer
                (bufferOffset s stride_buf_h stride_buf_m stride_buf_d BLOCK_M idx))))) := by
  intro idx
  simp [exec, flash_attn_output_store_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, mIndex, dIndex,
        bufferOffset, outOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, DIM] → Nat :=
    fun idx =>
      s.pids 1 * stride_q_head +
        (s.pids 0 * BLOCK_M + idx.1.val) * stride_o_seqlen +
        idx.2.1.val * stride_o_dim
  let valueFn : TileIndex [BLOCK_M, DIM] → TileCarrier TileDType.fp16 :=
    fun idx =>
      FloatDType.real.cast FloatDType.fp16
        (some (s.readMem OutBuffer
          (s.pids 1 * stride_buf_h +
            (s.pids 0 * BLOCK_M + idx.1.val) * stride_buf_m +
            idx.2.1.val * stride_buf_d)))
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, mIndex, dIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMemTyped .fp16 O (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BLOCK_M, DIM])).mem O (offsetFn idx) =
    MemCell.of .fp16
      (FloatDType.real.cast FloatDType.fp16
        (some (s.readMem OutBuffer
          (bufferOffset s stride_buf_h stride_buf_m stride_buf_d BLOCK_M idx))))
  rw [scatter_memcell_fp16_nd _ _ _ hOffsetInj idx]
  simp [valueFn, bufferOffset, mIndex, dIndex, FloatDType.cast,
    FloatDType.ofReal, FloatDType.storeValue, FloatDType.ofWithBot, FloatDType.toWithBot]

/-- Compute-facing correctness for the final FlashAttention output store. -/
theorem flash_attn_output_store_slice_compute_correct
    (OutBuffer O : RegionName)
    (stride_buf_h stride_buf_m stride_buf_d
      stride_q_head stride_o_seqlen stride_o_dim
      BLOCK_M DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, DIM] =>
        outOffset s stride_q_head stride_o_seqlen stride_o_dim BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := flash_attn_output_store_slice OutBuffer O stride_buf_h
        stride_buf_m stride_buf_d stride_q_head stride_o_seqlen stride_o_dim
        BLOCK_M DIM)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, DIM] =>
        some (O, outOffset s stride_q_head stride_o_seqlen stride_o_dim
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, DIM] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem OutBuffer
              (bufferOffset s stride_buf_h stride_buf_m stride_buf_d BLOCK_M idx))))) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [flash_attn_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := flash_attn_output_store_slice_correct OutBuffer O stride_buf_h
    stride_buf_m stride_buf_d stride_q_head stride_o_seqlen stride_o_dim
    BLOCK_M DIM s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

end VeriTile.Bench.TritonBenchG.FlashAttn
