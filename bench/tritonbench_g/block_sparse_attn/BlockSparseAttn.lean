import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `block_sparse_attn.py`'s
`block_sparse_attention_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `tl.constexpr` parameters become Lean parameters with `$(...)` at use
  sites.
- The Python `tl.static_print(f"...")` f-string payload is represented by a
  fixed debug string; `tl.static_print` is a compile-time/no-op DSL marker.
- `if NUM_D_BLOCKS >= 2:` is represented as the equivalent Bool antiquote
  `if $((NUM_D_BLOCKS >= 2 : Bool))`. -/
def block_sparse_attention_kernel
    (out Q K V : RegionName)
    (layout_csr_row_indices layout_csr_col_indices : Region .nat)
    (layout_csr_row_stride_h layout_csr_col_stride_h num_layout : Nat)
    (softmax_scale : ℝ)
    (stride_qb stride_qh stride_qm stride_kb stride_kh stride_kn
      stride_vb stride_vh stride_vn stride_ob stride_oh stride_om
      num_heads num_kv_heads total_seq_len BLOCK_M BLOCK_N BLOCK_D
      NUM_D_BLOCKS : Nat)
    (EVEN_M EVEN_N : Bool) :
    ComputeKernel := triton {
  tl.static_print("block_sparse_attention_kernel")
  q_seq_len = $(total_seq_len)
  start_m = tl.program_id(0)
  off_bh = tl.program_id(1)
  off_h = off_bh % $(num_heads)
  off_b = off_bh // $(num_heads)
  head_groups = $(num_heads) // $(num_kv_heads)
  off_h_kv = off_h // head_groups
  Q += off_b * $(stride_qb) + off_h * $(stride_qh)
  K += off_b * $(stride_kb) + off_h_kv * $(stride_kh)
  V += off_b * $(stride_vb) + off_h_kv * $(stride_vh)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_D))
  off_q = offs_m[:, None] * $(stride_qm) + offs_d[None, :]
  off_k = offs_n[None, :] * $(stride_kn) + offs_d[:, None]
  off_v = offs_n[:, None] * $(stride_vn) + offs_d[None, :]
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_D)], dtype=tl.float32)
  if $((NUM_D_BLOCKS >= 2 : Bool)) {
    acc2 = tl.zeros([$(BLOCK_M), $(BLOCK_D)], dtype=tl.float32)
  }
  if EVEN_M {
    q = tl.load(q_ptrs)
    if $((NUM_D_BLOCKS >= 2 : Bool)) {
      q2 = tl.load(q_ptrs + $(BLOCK_D))
    }
  } else {
    q = tl.load(q_ptrs, mask=offs_m[:, None] < q_seq_len)
    if $((NUM_D_BLOCKS >= 2 : Bool)) {
      q2 = tl.load(q_ptrs + $(BLOCK_D), mask=offs_m[:, None] < q_seq_len)
    }
  }
  layout_h = off_h % $(num_layout)
  layout_ptr = layout_csr_row_indices + layout_h * $(layout_csr_row_stride_h) + start_m
  start_l = (tl.load(layout_ptr)).to(tl.int32)
  end_l = (tl.load(layout_ptr + $(1))).to(tl.int32)
  for col_idx_idx in range(start_l, end_l) {
    col_idx = (tl.load(layout_csr_col_indices +
      layout_h * $(layout_csr_col_stride_h) + col_idx_idx)).to(tl.int32)
    start_n = col_idx * $(BLOCK_N)
    if EVEN_N {
      k = tl.load(k_ptrs + start_n * $(stride_kn))
    } else {
      k = tl.load(k_ptrs + start_n * $(stride_kn),
        mask=offs_n[None, :] + start_n < $(total_seq_len))
    }
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    if $((NUM_D_BLOCKS >= 2 : Bool)) {
      if EVEN_N {
        k = tl.load(k_ptrs + start_n * $(stride_kn) + $(BLOCK_D))
      } else {
        k = tl.load(k_ptrs + start_n * $(stride_kn) + $(BLOCK_D),
          mask=offs_n[None, :] + start_n < $(total_seq_len))
      }
      qk += tl.dot(q2, k)
    }
    qk *= $(softmax_scale)
    qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float("-inf"))
    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = l_i / l_i_new * alpha
    acc = acc * acc_scale[:, None]
    if $((NUM_D_BLOCKS >= 2 : Bool)) {
      acc2 = acc2 * acc_scale[:, None]
    }
    p = (p).to(Q.dtype.element_ty)
    if EVEN_N {
      v = tl.load(v_ptrs + start_n * $(stride_vn))
    } else {
      v = tl.load(v_ptrs + start_n * $(stride_vn),
        mask=offs_n[:, None] + start_n < $(total_seq_len))
    }
    acc += tl.dot(p, v)
    if $((NUM_D_BLOCKS >= 2 : Bool)) {
      if EVEN_N {
        v = tl.load(v_ptrs + start_n * $(stride_vn) + $(BLOCK_D))
      } else {
        v = tl.load(v_ptrs + start_n * $(stride_vn) + $(BLOCK_D),
          mask=offs_n[:, None] + start_n < $(total_seq_len))
      }
      acc2 += tl.dot(p, v)
    }
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = off_b * $(stride_ob) + off_h * $(stride_oh) +
    offs_m[:, None] * $(stride_om) + offs_d[None, :]
  out_ptrs = out + off_o
  tl.store(out_ptrs, acc, mask=offs_m[:, None] < q_seq_len)
  if $((NUM_D_BLOCKS >= 2 : Bool)) {
    tl.store(out_ptrs + $(BLOCK_D), acc2, mask=offs_m[:, None] < q_seq_len)
  }
}

/-- Surface transcription/proof-oriented first output-block store slice of
`block_sparse_attn.py`'s `block_sparse_attention_kernel`.

The full kernel walks a CSR sparse layout and accumulates one or two D blocks.
This slice starts from a precomputed first-block `Acc` tile and proves the final
masked writeback into `Out`, preserving the source `off_bh` decomposition and
`offs_m < total_seq_len` row mask. The CSR `tl.int32` row/column index casts,
`tl.float32` online-softmax accumulator, and `p.to(Q.dtype.element_ty)` dot input cast
belong to the omitted sparse-attention loop that produces `Acc`. -/
def block_sparse_attn_output_store_slice
    (Acc Out : RegionName)
    (num_heads total_seq_len
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om
      BLOCK_M BLOCK_D : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bh = tl.program_id(1)
  off_h = off_bh % $(num_heads)
  off_b = off_bh // $(num_heads)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_D))
  mask = (offs_m[:, None] < $(total_seq_len)) & (offs_d[None, :] < $(BLOCK_D))
  acc = tl.load(Acc + off_b * $(stride_acc_b) + off_h * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_d[None, :] * $(stride_acc_d),
      mask=mask, other=0.0)
  tl.store(Out + off_b * $(stride_ob) + off_h * $(stride_oh) +
      offs_m[:, None] * $(stride_om) + offs_d[None, :], acc, mask=mask)
}

/-- Surface transcription of the second output-block store in
`block_sparse_attn.py`'s `block_sparse_attention_kernel`.

The benchmark uses `NUM_D_BLOCKS = 2`, so Python stores `acc2` at
`out_ptrs + BLOCK_D` after the first output block. This slice starts from a
precomputed second-block `Acc2` tile and preserves the same row mask and
batch/head decomposition as the first store. -/
def block_sparse_attn_output_store_second_slice
    (Acc2 Out : RegionName)
    (num_heads total_seq_len
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om
      BLOCK_M BLOCK_D : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bh = tl.program_id(1)
  off_h = off_bh % $(num_heads)
  off_b = off_bh // $(num_heads)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_D))
  mask = (offs_m[:, None] < $(total_seq_len)) & (offs_d[None, :] < $(BLOCK_D))
  acc2 = tl.load(Acc2 + off_b * $(stride_acc_b) + off_h * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_d[None, :] * $(stride_acc_d),
      mask=mask, other=0.0)
  tl.store(Out + off_b * $(stride_ob) + off_h * $(stride_oh) +
      offs_m[:, None] * $(stride_om) + $(BLOCK_D) + offs_d[None, :],
    acc2, mask=mask)
}

def offH (s : BlockState) (num_heads : Nat) : Nat :=
  s.pids 1 % num_heads

def offB (s : BlockState) (num_heads : Nat) : Nat :=
  s.pids 1 / num_heads

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def dIndex (idx : TileIndex [BLOCK_M, BLOCK_D]) : Nat :=
  idx.2.1.val

def active
    (s : BlockState) (total_seq_len BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D]) : Prop :=
  mIndex s BLOCK_M idx.1 < total_seq_len

instance activeDecidable
    (s : BlockState) (total_seq_len BLOCK_M BLOCK_D : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D]) :
    Decidable (active s total_seq_len BLOCK_M idx) := by
  unfold active
  infer_instance

def accOffset
    (s : BlockState)
    (num_heads stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D]) : Nat :=
  offB s num_heads * stride_acc_b + offH s num_heads * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + dIndex idx * stride_acc_d

def outOffset
    (s : BlockState)
    (num_heads stride_ob stride_oh stride_om BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D]) : Nat :=
  offB s num_heads * stride_ob + offH s num_heads * stride_oh +
    mIndex s BLOCK_M idx.1 * stride_om + dIndex idx

noncomputable def accStoreValue
    (s : BlockState) (Acc : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m
      stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D]) : ℝ :=
  WithBot.unbotD 0
    (if active s total_seq_len BLOCK_M idx then
      some (s.readMem Acc
        (accOffset s num_heads stride_acc_b stride_acc_h stride_acc_m
          stride_acc_d BLOCK_M idx))
    else some (0.0 : ℝ))

/-- Algorithm-layer correctness for the first block-sparse output store. -/
theorem block_sparse_attn_output_store_slice_correct
    (Acc Out : RegionName)
    (num_heads total_seq_len
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om
      BLOCK_M BLOCK_D : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        outOffset s num_heads stride_ob stride_oh stride_om BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
      let outAddr := outOffset s num_heads stride_ob stride_oh stride_om
        BLOCK_M idx
      (exec (block_sparse_attn_output_store_slice Acc Out num_heads
            total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
            stride_ob stride_oh stride_om BLOCK_M BLOCK_D) s).map
          (·.readMem Out outAddr)
        = some (if active s total_seq_len BLOCK_M idx then
            accStoreValue s Acc num_heads total_seq_len stride_acc_b
              stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, block_sparse_attn_output_store_slice, stepStmts, stepStmt,
        evalOp, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, IntegralDType.floorDiv,
        IntegralDType.mod, ComparableDType.lt, offH, offB, mIndex, dIndex,
        active, accOffset, outOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_D] → Nat :=
    fun idx =>
      s.pids 1 / num_heads * stride_ob + s.pids 1 % num_heads * stride_oh +
        (s.pids 0 * BLOCK_M + idx.1.val) * stride_om + idx.2.1.val
  let valueFn : TileIndex [BLOCK_M, BLOCK_D] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 0 * BLOCK_M + idx.1.val < total_seq_len then
          some (s.readMem Acc
            (s.pids 1 / num_heads * stride_acc_b +
              s.pids 1 % num_heads * stride_acc_h +
              (s.pids 0 * BLOCK_M + idx.1.val) * stride_acc_m +
              idx.2.1.val * stride_acc_d))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_M, BLOCK_D] → Prop :=
    fun idx => s.pids 0 * BLOCK_M + idx.1.val < total_seq_len
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, offH, offB, mIndex, dIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_D])).readMem Out
        (offsetFn idx) =
    if P idx then
      accStoreValue s Acc num_heads total_seq_len stride_acc_b stride_acc_h
        stride_acc_m stride_acc_d BLOCK_M idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 0 * BLOCK_M + idx.1.val < total_seq_len
  · rfl
  · rfl

/-- Compute-facing correctness for the first block-sparse output store. -/
theorem block_sparse_attn_output_store_slice_compute_correct
    (Acc Out : RegionName)
    (num_heads total_seq_len
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om
      BLOCK_M BLOCK_D : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        outOffset s num_heads stride_ob stride_oh stride_om BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := block_sparse_attn_output_store_slice Acc Out num_heads
        total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
        stride_ob stride_oh stride_om BLOCK_M BLOCK_D)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
          active s total_seq_len BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_D] => (Out,
          outOffset s num_heads stride_ob stride_oh stride_om BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        accStoreValue s Acc num_heads total_seq_len stride_acc_b stride_acc_h
          stride_acc_m stride_acc_d BLOCK_M idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [block_sparse_attn_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := block_sparse_attn_output_store_slice_correct Acc Out num_heads
    total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d stride_ob
    stride_oh stride_om BLOCK_M BLOCK_D s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Output offset for the second block-sparse store. -/
def out2Offset
    (s : BlockState)
    (num_heads stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D]) : Nat :=
  offB s num_heads * stride_ob + offH s num_heads * stride_oh +
    mIndex s BLOCK_M idx.1 * stride_om + BLOCK_D + dIndex idx

/-- Algorithm-layer correctness for the second block-sparse output store. -/
theorem block_sparse_attn_output_store_second_slice_correct
    (Acc2 Out : RegionName)
    (num_heads total_seq_len
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om
      BLOCK_M BLOCK_D : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        out2Offset s num_heads stride_ob stride_oh stride_om BLOCK_M BLOCK_D idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
      let outAddr := out2Offset s num_heads stride_ob stride_oh stride_om
        BLOCK_M BLOCK_D idx
      (exec (block_sparse_attn_output_store_second_slice Acc2 Out num_heads
            total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
            stride_ob stride_oh stride_om BLOCK_M BLOCK_D) s).map
          (·.readMem Out outAddr)
        = some (if active s total_seq_len BLOCK_M idx then
            accStoreValue s Acc2 num_heads total_seq_len stride_acc_b
              stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, block_sparse_attn_output_store_second_slice, stepStmts, stepStmt,
        evalOp, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, IntegralDType.floorDiv,
        IntegralDType.mod, ComparableDType.lt, offH, offB, mIndex, dIndex,
        active, accOffset, out2Offset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_D] → Nat :=
    fun idx =>
      s.pids 1 / num_heads * stride_ob + s.pids 1 % num_heads * stride_oh +
        (s.pids 0 * BLOCK_M + idx.1.val) * stride_om + BLOCK_D + idx.2.1.val
  let valueFn : TileIndex [BLOCK_M, BLOCK_D] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 0 * BLOCK_M + idx.1.val < total_seq_len then
          some (s.readMem Acc2
            (s.pids 1 / num_heads * stride_acc_b +
              s.pids 1 % num_heads * stride_acc_h +
              (s.pids 0 * BLOCK_M + idx.1.val) * stride_acc_m +
              idx.2.1.val * stride_acc_d))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_M, BLOCK_D] → Prop :=
    fun idx => s.pids 0 * BLOCK_M + idx.1.val < total_seq_len
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, out2Offset, offH, offB, mIndex, dIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_D])).readMem Out
        (offsetFn idx) =
    if P idx then
      accStoreValue s Acc2 num_heads total_seq_len stride_acc_b stride_acc_h
        stride_acc_m stride_acc_d BLOCK_M idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 0 * BLOCK_M + idx.1.val < total_seq_len
  · rfl
  · rfl

/-- Compute-facing correctness for the second block-sparse output store. -/
theorem block_sparse_attn_output_store_second_slice_compute_correct
    (Acc2 Out : RegionName)
    (num_heads total_seq_len
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om
      BLOCK_M BLOCK_D : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        out2Offset s num_heads stride_ob stride_oh stride_om BLOCK_M BLOCK_D idx)) :
    ComputeCorrect.Realizes
      (kernel := block_sparse_attn_output_store_second_slice Acc2 Out num_heads
        total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
        stride_ob stride_oh stride_om BLOCK_M BLOCK_D)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
          active s total_seq_len BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_D] => (Out,
          out2Offset s num_heads stride_ob stride_oh stride_om BLOCK_M BLOCK_D idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        accStoreValue s Acc2 num_heads total_seq_len stride_acc_b stride_acc_h
          stride_acc_m stride_acc_d BLOCK_M idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [block_sparse_attn_output_store_second_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := block_sparse_attn_output_store_second_slice_correct Acc2 Out num_heads
    total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d stride_ob
    stride_oh stride_om BLOCK_M BLOCK_D s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

theorem block_sparse_attn_python_first_output_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [16, 16] => outOffset s 4 2048 512 32 16 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, offB, offH, mIndex, dIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem block_sparse_attn_python_second_output_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [16, 16] => out2Offset s 4 2048 512 32 16 16 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [out2Offset, offB, offH, mIndex, dIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem block_sparse_attn_python_first_output_compute_correct
    (Acc Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := block_sparse_attn_output_store_slice Acc Out 4 16
        2048 512 16 1 2048 512 32 16 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 16] => active s 16 16 idx)
        (fun idx : TileIndex [16, 16] => (Out, outOffset s 4 2048 512 32 16 idx)))
      (expected := fun idx : TileIndex [16, 16] =>
        accStoreValue s Acc 4 16 2048 512 16 1 16 idx) := by
  exact block_sparse_attn_output_store_slice_compute_correct Acc Out
    4 16 2048 512 16 1 2048 512 32 16 16 s
    (block_sparse_attn_python_first_output_offset_injective s)

theorem block_sparse_attn_python_second_output_compute_correct
    (Acc2 Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := block_sparse_attn_output_store_second_slice Acc2 Out 4 16
        2048 512 16 1 2048 512 32 16 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 16] => active s 16 16 idx)
        (fun idx : TileIndex [16, 16] =>
          (Out, out2Offset s 4 2048 512 32 16 16 idx)))
      (expected := fun idx : TileIndex [16, 16] =>
        accStoreValue s Acc2 4 16 2048 512 16 1 16 idx) := by
  exact block_sparse_attn_output_store_second_slice_compute_correct Acc2 Out
    4 16 2048 512 16 1 2048 512 32 16 16 s
    (block_sparse_attn_python_second_output_offset_injective s)

end VeriTile.Bench.TritonBenchG.BlockSparseAttn
