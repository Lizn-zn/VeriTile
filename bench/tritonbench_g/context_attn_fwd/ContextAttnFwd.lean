import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful DSL port of `context_attn_fwd.py`'s `_fwd_kernel_int8kv`. -/
def context_attn_fwd_kernel_int8kv_surface
    (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd
      stride_kb stride_kh stride_ks stride_kd
      stride_vb stride_vh stride_vs stride_vd
      stride_obs stride_oh stride_od
      kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  cur_bh = tl.program_id(1)
  cur_batch = cur_bh // $(H)
  cur_head = cur_bh % $(H)

  cur_kv_head = cur_head // $(kv_group_num)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = block_start_loc + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)
  q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))
  block_end_loc = tl.minimum(block_start_loc + $(BLOCK_M) + prompt_cache_len,
    cur_batch_seq_len + prompt_cache_len)
  for start_n in range($(0), block_mask * block_end_loc, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    off_k = cur_batch * $(stride_kb) + (start_n + offs_n[None, :]) * $(stride_ks) +
      cur_kv_head * $(stride_kh) + offs_d[:, None] * $(stride_kd)
    k = tl.load(K + off_k,
      mask=(start_n + offs_n[None, :]) < block_end_loc,
      other=0.0)

    qk = tl.dot(q, k)
    mask = (offs_m[:, None] + prompt_cache_len) >= (start_n + offs_n[None, :])
    qk = tl.where(mask, qk * $((sm_scale : ℝ)), -1.0e8)
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)

    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    off_v = cur_batch * $(stride_vb) + (start_n + offs_n[:, None]) * $(stride_vs) +
      cur_kv_head * $(stride_vh) + offs_d[None, :] * $(stride_vd)
    v = tl.load(V + off_v,
      mask=(start_n + offs_n[:, None]) < block_end_loc,
      other=0.0)

    p = (p).to(v.dtype)
    acc = tl.dot(p, v, acc)
    m_i = m_ij
  }

  acc = acc / l_i[:, None]
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
}

/-- The full int8-KV context-attention surface lowers to the algorithm layer. -/
theorem context_attn_fwd_kernel_int8kv_surface_toAlgorithm_supported
    (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd
      stride_kb stride_kh stride_ks stride_kd
      stride_vb stride_vh stride_vs stride_vd
      stride_obs stride_oh stride_od
      kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) :
    ∃ alg, (context_attn_fwd_kernel_int8kv_surface Q K V sm_scale Out
      B_Start_Loc B_Seqlen b_prompt_cache_len stride_qbs stride_qh stride_qd
      stride_kb stride_kh stride_ks stride_kd stride_vb stride_vh stride_vs
      stride_vd stride_obs stride_oh stride_od kv_group_num H BLOCK_DMODEL
      BLOCK_M BLOCK_N).toAlgorithm? = Except.ok alg := by
  simp [context_attn_fwd_kernel_int8kv_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented final output-store slice of `context_attn_fwd.py`'s
`_fwd_kernel_int8kv`.

The full kernel computes PPL int8-KV context attention. This slice starts from
a precomputed `Acc` tile and proves the final masked writeback into `Out`,
preserving the fused `cur_bh` program-id decomposition, `B_Start_Loc`, and the
prompt-cache-adjusted sequence length. The inner `tl.float32`
streaming-softmax accumulator and int8-KV dequantization are outside this
slice. -/
def context_attn_fwd_final_store_slice
    (Acc : RegionName) (B_Start_Loc B_Seqlen B_Prompt_Cache_Len : Region .nat)
    (Out : RegionName)
    (H
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  cur_bh = tl.program_id(1)
  cur_batch = cur_bh // $(H)
  cur_head = cur_bh % $(H)
  prompt_cache_len = tl.load(B_Prompt_Cache_Len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(BLOCK_DMODEL))
  acc = tl.load(Acc + cur_batch * $(stride_acc_b) + cur_head * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_d[None, :] * $(stride_acc_d),
      mask=mask, other=0.0)
  tl.store(Out + (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
      cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od), acc, mask=mask)
}

def curBatch (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H

def curHead (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H

def promptLen (s : BlockState) (H : Nat) (B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Prompt_Cache_Len (curBatch s H)

def seqLen
    (s : BlockState) (H : Nat) (B_Seqlen B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (curBatch s H) -
    promptLen s H B_Prompt_Cache_Len

def startLoc (s : BlockState) (H : Nat) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (curBatch s H)

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def active
    (s : BlockState) (H : Nat) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s H B_Seqlen B_Prompt_Cache_Len

instance activeDecidable
    (s : BlockState) (H : Nat) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (BLOCK_M BLOCK_DMODEL : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) :
    Decidable (active s H B_Seqlen B_Prompt_Cache_Len BLOCK_M idx) := by
  unfold active
  infer_instance

def accOffset
    (s : BlockState)
    (H stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  curBatch s H * stride_acc_b + curHead s H * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + dIndex idx * stride_acc_d

def outOffset
    (s : BlockState) (H : Nat) (B_Start_Loc : RegionName)
    (stride_obs stride_oh stride_od BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (startLoc s H B_Start_Loc + mIndex s BLOCK_M idx.1) * stride_obs +
    curHead s H * stride_oh + dIndex idx * stride_od

noncomputable def accStoreValue
    (s : BlockState) (Acc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (H stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s H B_Seqlen B_Prompt_Cache_Len BLOCK_M idx then
      some (s.readMem Acc
        (accOffset s H stride_acc_b stride_acc_h stride_acc_m stride_acc_d
          BLOCK_M idx))
    else some (0.0 : ℝ))

/-- Algorithm-layer correctness for the masked context-attention output store. -/
theorem context_attn_fwd_final_store_slice_correct
    (Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (H
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr := outOffset s H B_Start_Loc stride_obs stride_oh stride_od
        BLOCK_M idx
      (exec (context_attn_fwd_final_store_slice Acc B_Start_Loc B_Seqlen
            B_Prompt_Cache_Len Out H stride_acc_b stride_acc_h stride_acc_m
            stride_acc_d stride_obs stride_oh stride_od BLOCK_M BLOCK_DMODEL)
          s).map (·.readMem Out outAddr)
        = some (if active s H B_Seqlen B_Prompt_Cache_Len BLOCK_M idx then
            accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len H stride_acc_b
              stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, context_attn_fwd_final_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.sub, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
        BlockState.readMemValue, curBatch, curHead, promptLen, seqLen, startLoc,
        mIndex, dIndex, active, accOffset, outOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      (s.readMemValue .nat B_Start_Loc (s.pids 1 / H) +
          (s.pids 0 * BLOCK_M + idx.1.val)) * stride_obs +
        s.pids 1 % H * stride_oh + idx.2.1.val * stride_od
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 0 * BLOCK_M + idx.1.val <
            s.readMemValue .nat B_Seqlen (s.pids 1 / H) -
              s.readMemValue .nat B_Prompt_Cache_Len (s.pids 1 / H) then
          some (s.readMem Acc
            ((s.pids 1 / H) * stride_acc_b + (s.pids 1 % H) * stride_acc_h +
              (s.pids 0 * BLOCK_M + idx.1.val) * stride_acc_m +
              idx.2.1.val * stride_acc_d))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx =>
      s.pids 0 * BLOCK_M + idx.1.val <
        s.readMemValue .nat B_Seqlen (s.pids 1 / H) -
          s.readMemValue .nat B_Prompt_Cache_Len (s.pids 1 / H)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, startLoc, curBatch, curHead, mIndex, dIndex,
      BlockState.readMemValue] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    if P idx then
      accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len H stride_acc_b
        stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 0 * BLOCK_M + idx.1.val <
        s.readMemValue .nat B_Seqlen (s.pids 1 / H) -
          s.readMemValue .nat B_Prompt_Cache_Len (s.pids 1 / H)
  · rfl
  · rfl

/-- Compute-facing correctness for the masked context-attention output store. -/
theorem context_attn_fwd_final_store_slice_compute_correct
    (Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (H
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := context_attn_fwd_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out H stride_acc_b stride_acc_h stride_acc_m
        stride_acc_d stride_obs stride_oh stride_od BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s H B_Seqlen B_Prompt_Cache_Len BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len H stride_acc_b
          stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [context_attn_fwd_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := context_attn_fwd_final_store_slice_correct Acc B_Start_Loc
    B_Seqlen B_Prompt_Cache_Len Out H stride_acc_b stride_acc_h stride_acc_m
    stride_acc_d stride_obs stride_oh stride_od BLOCK_M BLOCK_DMODEL s
    hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Python test-shape wrappers

The checked Python test uses `Z = 16`, `H = 16`, `N_CTX = 2048`, and
`D_HEAD = 128`. The output tensor is contiguous with row/head/dimension strides
`(2048, 128, 1)`. `BLOCK_DMODEL = 128`, and `BLOCK_M` is `128` on the regular
path or `64` on the Tesla branch. -/

theorem context_attn_fwd_python_block128_offset_injective
    (s : BlockState) (B_Start_Loc : RegionName) :
    Function.Injective
      (fun idx : TileIndex [128, 128] =>
        outOffset s 16 B_Start_Loc 2048 128 1 128 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, startLoc, curBatch, curHead, mIndex, dIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem context_attn_fwd_python_block64_offset_injective
    (s : BlockState) (B_Start_Loc : RegionName) :
    Function.Injective
      (fun idx : TileIndex [64, 128] =>
        outOffset s 16 B_Start_Loc 2048 128 1 64 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, startLoc, curBatch, curHead, mIndex, dIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem context_attn_fwd_final_store_python_block128_compute_correct
    (Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_fwd_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out 16 4194304 128 2048 1 2048 128 1 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len 16 4194304 128
          2048 1 128 idx) := by
  exact context_attn_fwd_final_store_slice_compute_correct Acc B_Start_Loc
    B_Seqlen B_Prompt_Cache_Len Out 16 4194304 128 2048 1 2048 128 1
    128 128 s (context_attn_fwd_python_block128_offset_injective s B_Start_Loc)

theorem context_attn_fwd_final_store_python_block64_compute_correct
    (Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_fwd_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out 16 4194304 128 2048 1 2048 128 1 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 64 idx)
        (fun idx : TileIndex [64, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 64 idx)))
      (expected := fun idx : TileIndex [64, 128] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len 16 4194304 128
          2048 1 64 idx) := by
  exact context_attn_fwd_final_store_slice_compute_correct Acc B_Start_Loc
    B_Seqlen B_Prompt_Cache_Len Out 16 4194304 128 2048 1 2048 128 1
    64 128 s (context_attn_fwd_python_block64_offset_injective s B_Start_Loc)

/-- Public Python test-shape summary for `context_attn_fwd.py`.

This records the faithful full int8-KV `_fwd_kernel` surface for the checked
layout and both Python launcher block-size branches, then ties those surfaces
to the observable final `Out` writeback slices. The streaming attention
accumulator is represented by the proof-oriented `Acc` tile in the store
slice. -/
theorem context_attn_fwd_python_test_shape_output_summary
    (Q K V Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (s : BlockState) :
    (∃ alg, (context_attn_fwd_kernel_int8kv_surface Q K V
      (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
      B_Start_Loc B_Seqlen B_Prompt_Cache_Len
      2048 128 1 8388608 262144 128 1 8388608 262144 128 1
      2048 128 1 1 16 128 128 128).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (context_attn_fwd_kernel_int8kv_surface Q K V
      (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
      B_Start_Loc B_Seqlen B_Prompt_Cache_Len
      2048 128 1 8388608 262144 128 1 8388608 262144 128 1
      2048 128 1 1 16 128 64 128).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := context_attn_fwd_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out 16 4194304 128 2048 1 2048 128 1 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len 16 4194304 128
          2048 1 128 idx) ∧
    ComputeCorrect.Realizes
      (kernel := context_attn_fwd_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out 16 4194304 128 2048 1 2048 128 1 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 64 idx)
        (fun idx : TileIndex [64, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 64 idx)))
      (expected := fun idx : TileIndex [64, 128] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len 16 4194304 128
          2048 1 64 idx) := by
  constructor
  · exact context_attn_fwd_kernel_int8kv_surface_toAlgorithm_supported Q K V
      (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
      B_Start_Loc B_Seqlen B_Prompt_Cache_Len
      2048 128 1 8388608 262144 128 1 8388608 262144 128 1
      2048 128 1 1 16 128 128 128
  constructor
  · exact context_attn_fwd_kernel_int8kv_surface_toAlgorithm_supported Q K V
      (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
      B_Start_Loc B_Seqlen B_Prompt_Cache_Len
      2048 128 1 8388608 262144 128 1 8388608 262144 128 1
      2048 128 1 1 16 128 64 128
  constructor
  · exact context_attn_fwd_final_store_python_block128_compute_correct Acc
      B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out s
  · exact context_attn_fwd_final_store_python_block64_compute_correct Acc
      B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out s

end VeriTile.Bench.TritonBenchG.ContextAttnFwd
